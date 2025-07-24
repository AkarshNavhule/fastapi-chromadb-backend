import re
import json
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from config import GENERATION_MODEL
from search_engine import get_embedding, query_chroma
from uuid import uuid4
from firestore11 import store_question_paper


print("[questionpaper] Configuring Gemini client...")
client = genai.Client()

def extract_question_requirements(prompt: str) -> Dict[str, Any]:
    print(f"[questionpaper] Parsing prompt: {prompt}")
    requirements = {
        "total_marks": 20,
        "page_range": None,
        "topic": None,
        "mark_distribution": [],
        "paper_type": "medium"
    }

    marks_match = re.search(r"(\d+)\s*marks?", prompt, re.IGNORECASE)
    if marks_match:
        requirements["total_marks"] = int(marks_match.group(1))

    page_match = re.search(r"page\s*(\d+)\s*(?:to|-)\s*(\d+)", prompt, re.IGNORECASE)
    if page_match:
        requirements["page_range"] = (int(page_match.group(1)), int(page_match.group(2)))

    topic_match = re.search(r"on\s+(.+?)(?:\s+and\s|$)", prompt, re.IGNORECASE)
    if topic_match:
        requirements["topic"] = topic_match.group(1).strip()

    mark_dist_matches = re.findall(r"(\d+)\s*marks?", prompt, re.IGNORECASE)
    if len(mark_dist_matches) > 1:
        requirements["mark_distribution"] = [int(m) for m in mark_dist_matches[1:]]  # skip first

    if not requirements["mark_distribution"]:
        total = requirements["total_marks"]
        if total <= 20:
            requirements["mark_distribution"] = [2, 3, 5]
        elif total <= 40:
            requirements["mark_distribution"] = [2, 3, 5, 10]
        else:
            requirements["mark_distribution"] = [2, 3, 5, 10, 15]

    print(f"[questionpaper] Extracted requirements: {requirements}")
    return requirements

def create_mark_allocation(total_marks: int, mark_types: List[int]) -> List[Dict[str, int]]:
    allocation = []
    remaining_marks = total_marks

    sorted_marks = sorted(mark_types)
    for mark_value in sorted_marks:
        if remaining_marks <= 0:
            break
        max_questions = remaining_marks // mark_value
        if mark_value <= 3:
            num_questions = min(max_questions, 3)
        elif mark_value <= 5:
            num_questions = min(max_questions, 2)
        else:
            num_questions = min(max_questions, 1)
        if num_questions > 0:
            allocation.append({
                "marks": mark_value,
                "count": num_questions
            })
            remaining_marks -= (mark_value * num_questions)
    print(f"[questionpaper] Mark allocation: {allocation}")
    return allocation

def extract_json_from_response(response_text: str) -> List[Dict[str, Any]]:
    """Robustly extract a JSON array from Gemini's LLM response"""
    # Try code block with json
    m = re.search(r"``````", response_text, re.DOTALL)
    if m:
        json_str = m.group(1).strip()
    else:
        # Try code block without json
        m2 = re.search(r"``````", response_text, re.DOTALL)
        if m2:
            json_str = m2.group(1).strip()
        else:
            # Try to find a JSON array in the text directly
            m3 = re.search(r"(\[.*\])", response_text, re.DOTALL)
            if m3:
                json_str = m3.group(1)
            else:
                json_str = response_text  # fallback
    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"[questionpaper] Error decoding JSON: {e}")
        return []

def generate_questions_for_content(content: str, requirements: Dict[str, Any],
                                 mark_allocation: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    print("[questionpaper] Generating questions with Gemini...")

    difficulty_map = {
        "easy": "simple, straightforward questions that test basic understanding",
        "medium": "moderate difficulty questions that require some analysis",
        "hard": "challenging questions that require deep understanding and critical thinking"
    }
    difficulty_desc = difficulty_map.get(requirements.get("paper_type", "medium"), difficulty_map["medium"])

    questions_needed = []
    for alloc in mark_allocation:
        questions_needed.extend([f"{alloc['marks']} marks"] * alloc['count'])

    system_prompt = f"""You are an expert teacher creating exam questions. 
Generate {difficulty_desc}.
Requirements:
- Generate questions for: {', '.join(questions_needed)}
- Each question should be clear and specific
- Questions should test different aspects of the content
- Return ONLY a valid JSON array with this exact structure:
[
    {{
        "question": "Question text here?",
        "marks": 2,
        "difficulty": "easy/medium/hard",
        "topic": "relevant topic"
    }}
]
"""
    user_prompt = f"""Based on the following content, create exam questions:

Content:
{content}

Generate exactly these questions:
{', '.join(questions_needed)}

Return only the JSON array, no other text.
"""

    try:
        resp = client.models.generate_content(
            model=GENERATION_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.7
            ),
            contents=user_prompt
        )

        response_text = resp.text.strip()
        questions = extract_json_from_response(response_text)
        if questions:
            print(f"[questionpaper] Generated {len(questions)} questions")
            return questions

    except Exception as e:
        print(f"[questionpaper] Error generating questions: {e}")

    # Fallback
    print("[questionpaper] Using fallback questions.")
    fallback_questions = []
    question_num = 1
    for alloc in mark_allocation:
        for _ in range(alloc['count']):
            fallback_questions.append({
                "question": f"Q{question_num} ({alloc['marks']} marks): Explain key concepts from the given content.",
                "marks": alloc['marks'],
                "difficulty": requirements.get("paper_type", "medium"),
                "topic": requirements.get("topic", "General")
            })
            question_num += 1
    return fallback_questions

def generate_question_paper(collection_name: str, user_prompt: str,
                            paper_type: str = "medium") -> Dict[str, Any]:
    print(f"[questionpaper] Generating paper for: {user_prompt}")

    requirements = extract_question_requirements(user_prompt)
    requirements["paper_type"] = paper_type

    # RAG retrieval
    query_embedding = get_embedding(user_prompt)
    page_filter = requirements.get("page_range")

    hits = query_chroma(collection_name, query_embedding, page_filter=page_filter)

    if not hits:
        return {
            "error": "No relevant content found for the given requirements",
            "question_paper": {},
            "sources": []
        }

    content = "\n\n".join([
        f"(Page {hit['metadata']['page_no']}): {hit['text']}"
        for hit in hits
    ])

    mark_allocation = create_mark_allocation(
        requirements["total_marks"],
        requirements["mark_distribution"]
    )

    questions = generate_questions_for_content(content, requirements, mark_allocation)

    # Add question numbers
    for i, q in enumerate(questions, 1):
        q["question_no"] = i

    question_paper_id = str(uuid4())

    question_paper = {
        "id": question_paper_id,
        "title": f"Question Paper - {requirements['total_marks']} Marks",
        "collection_name": collection_name,
        "total_marks": requirements["total_marks"],
        "difficulty": paper_type,
        "instructions": "Answer all questions. All questions are compulsory.",
        "questions": questions
    }

    sources = [
        {
            "page_no": hit["metadata"]["page_no"],
            "text": hit["text"][:200] + "..." if len(hit["text"]) > 200 else hit["text"]
        }
        for hit in hits
    ]

    response= {
        "question_paper_id": question_paper_id,
        "collection_name": collection_name,
        "question_paper": question_paper,
        "sources": sources
    }
    store_question_paper(response)

    return response

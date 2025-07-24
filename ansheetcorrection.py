from typing import List, Dict
import re, json
import base64

from google import genai
from google.genai import types
from search_engine import get_embedding, query_chroma
from config import GENERATION_MODEL

client = genai.Client()

async def run_ocr_sequential_internal(base64_images: List[str], process_image_func) -> List[Dict]:
    """Runs OCR sequentially using the internal async OCR function."""
    print(f"[ansheetcorrection] Internal OCR for {len(base64_images)} images...")
    results = []
    for idx, img_b64 in enumerate(base64_images):
        decoded_bytes = base64.b64decode(img_b64)
        ocr_result = await process_image_func(decoded_bytes)  # <- await here
        print(f"[ansheetcorrection] OCR done for image #{idx}. Result keys: {list(ocr_result.keys())}")
        results.append(ocr_result)
    return results

    """Runs OCR synchronously using the internal OCR processing function."""
    print(f"[ansheetcorrection] Internal OCR for {len(base64_images)} images...")
    results = []
    for idx, img_b64 in enumerate(base64_images):
        decoded_bytes = base64.b64decode(img_b64)
        ocr_result = process_image_func(decoded_bytes)
        print(f"[ansheetcorrection] OCR done for image #{idx}. Result keys: {list(ocr_result.keys())}")
        results.append(ocr_result)
    return results

def merge_ocr_results(results: List[Dict[str, str]]) -> Dict[str, str]:
    merged = {}
    for idx, result in enumerate(results, 1):
        merged.update(result)
    print(f"[ansheetcorrection] Final merged answer keys: {list(merged.keys())}")
    return merged

def rag_search_for_merged_answers(merged_answers: Dict[str, str], chroma_collection_name: str, top_k=8):
    merged_text = "\n".join([f"Q{qno}: {ans}" for qno, ans in merged_answers.items()])
    merged_embedding = get_embedding(merged_text)
    rag_hits = query_chroma(chroma_collection_name, merged_embedding, page_filter=None)
    rag_contexts = [
        f"(Page {h['metadata']['page_no']}): {h['text']}" for h in rag_hits[:top_k]
    ]
    return rag_contexts

def correct_answers_single_rag(
    merged_answers: Dict[str, str],
    question_paper_doc: Dict,
    chroma_collection_name: str,
    correctiontype: str = "medium",
    rag_top_k: int = 15,
):
    questions = question_paper_doc["question_paper"]["questions"]
    questions_for_prompt = [
        f"{q.get('question_no')}. {q.get('question', q.get('text',''))} [Max: {q.get('marks',0)} marks]"
        for q in questions
    ]
    merged_text = "\n".join([f"Q{qno}: {ans}" for qno, ans in merged_answers.items()])
    rag_contexts = rag_search_for_merged_answers(merged_answers, chroma_collection_name, top_k=rag_top_k)
    rag_context = "\n".join(rag_contexts)

    difficulty_map = {
        "easy": "Be lenient and focus on basic coverage of points.",
        "medium": "Score with typical marking scheme for Indian board exams.",
        "hard": "Mark strictly, expect in-depth, precise, textbook-aligned answers.",
    }
    system_instruction = (
        f"You are a senior examiner.{difficulty_map.get(correctiontype, difficulty_map['medium'])}\n"
        "Given the question paper, student's recognized answers, and the relevant textbook context, "
        "evaluate each answer, assign marks, give a brief justification, and produce short remarks for each answer.\n"
'''Respond as a JSON array:
[
  {
    "question_no": ...,
    "question": "...",
    "marks": ...,
    "chromadbsource": "...",
    "remarks": "..."
  }, ...
]
'''
    )

    prompt = (
        f"QUESTION PAPER (listing all questions):\n"
        f"{chr(10).join(questions_for_prompt)}\n\n"
        f"STUDENT'S ANSWERS (from OCR):\n{merged_text}\n\n"
        f"RELEVANT TEXTBOOK CONTEXT (RAG):\n{rag_context}\n\n"
        "For each question, evaluate and return marks, context, and remarks as a JSON array as detailed above."
    )

    try:
        resp = client.models.generate_content(
            model=GENERATION_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0
            ),
            contents=prompt
        )
        answer_text = resp.text.strip()
        print("[ansheetcorrection] Gemini raw output (truncated):", answer_text[:300])
        # Try to find JSON array in output:
        m = re.search(r'(\[[\s\S]+\])', answer_text)
        if m:
            question_marks = json.loads(m.group(1))
        else:
            print("[ansheetcorrection] Could not parse array; fallback to plain parsing.")
            question_marks = []
    except Exception as e:
        print("[ansheetcorrection] Gemini correction exception:", str(e))
        question_marks = []

    max_total_marks = sum(int(q.get('marks', 0)) for q in questions)
    obtained = sum(int(round(qm.get("marks", 0))) for qm in question_marks)
    totalmarks_str = f"{int(round(obtained))}/{max_total_marks}"

    # Defensive: Fill in question text/number if Gemini missed them
    qno2qtext = {str(q.get('question_no')): q.get('question', q.get('text', '')) for q in questions}
    for qm in question_marks:
        qno_str = str(qm.get("question_no", qm.get("questionNo", "")))
        qm["question_no"] = qno_str
        if not qm.get("question"):
            qm["question"] = qno2qtext.get(qno_str, "")
        qm["marks"] = int(round(qm.get("marks", 0)))
        # Add studentanswer from merged_answers
        qm["studentanswer"] = merged_answers.get(qno_str, "")
    return totalmarks_str, question_marks

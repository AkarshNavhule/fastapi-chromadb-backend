import re
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GENERATION_MODEL, EMBEDDING_MODEL, TOP_K
from vector_store import get_or_create_collection

print("[search_engine] Configuring Gemini client…")
client = genai.Client()  # reads GEMINI_API_KEY from env

def get_embedding(text: str) -> list[float]:
    print("[search_engine] Generating embedding…")
    resp = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return resp.embeddings[0].values

def extract_page_filter(prompt: str):
    m = re.search(r"page\s*(\d+)\s*(?:to|-)\s*(\d+)", prompt, re.IGNORECASE)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        print(f"[search_engine] Page filter: {start}-{end}")
        return (start, end)
    print("[search_engine] No page filter found")
    return None

def query_chroma(collection_name: str, query_embedding: list[float], page_filter=None, n_results=None):
    col = get_or_create_collection(collection_name)
    num_results = n_results if n_results is not None else 100
    where = {"page_no": {"$gte": page_filter[0]}} if page_filter else None
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=1000,
        where=where
    )
    hits = [
        {"metadata": md, "text": txt}
        for md, txt in zip(results["metadatas"][0], results["documents"][0])
    ]
    if page_filter:
        lower, upper = page_filter
        hits = [hit for hit in hits if lower <= hit["metadata"].get("page_no", 0) <= upper]
    return hits

    print(f"[search_engine] Querying Chroma '{collection_name}'")
    col = get_or_create_collection(collection_name)
    num_results = n_results if n_results is not None else TOP_K

    # Only one operator
    where = {"page_no": {"$gte": page_filter[0]}} if page_filter else None

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=1000,
        where=where
    )
    hits = [
        {"metadata": md, "text": txt}
        for md, txt in zip(results["metadatas"][0], results["documents"][0])
    ]

    # Final filtering
    if page_filter:
        lower, upper = page_filter
        hits = [hit for hit in hits if lower <= hit["metadata"].get("page_no", 0) <= upper]

    print(f"[search_engine] Got {len(hits)} hits after filtering")
    return hits

    print(f"[search_engine] Querying Chroma '{collection_name}'")
    col = get_or_create_collection(collection_name)
    where = {"page_no": {"$gte": page_filter[0], "$lte": page_filter[1]}} if page_filter else None
    
    # Use custom n_results or default TOP_K
    num_results = n_results if n_results is not None else TOP_K
    
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=num_results,
        where=where
    )
    hits = [
        {"metadata": md, "text": txt}
        for md, txt in zip(results["metadatas"][0], results["documents"][0])
    ]
    print(f"[search_engine] Got {len(hits)} hits")
    return hits

def query_gemini(prompt: str, context: str) -> str:
    print("[search_engine] Generating answer with Gemini…")
    cfg = types.GenerateContentConfig(system_instruction="You are a helpful tutor.")
    combined = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        config=cfg,
        contents=combined
    )
    print("[search_engine] Answer received")
    return resp.text

def query_gemini_ppt(prompt: str, context: str) -> str:
    print("[search_engine] Generating answer with Gemini…")
    
    system_prompt = """You are a specialized PowerPoint presentation generator designed to create comprehensive, educational presentations from textbook topics. Your role is to transform textbook content into engaging, visually structured slides that enhance learning and comprehension.

## Core Capabilities

**Content Analysis & Structure**: Analyze textbook topics and create logical presentation outlines with clear learning objectives. Break down complex topics into digestible slide segments.

**Slide Generation**: Generate structured presentations with appropriate titles, bullet points, explanatory text, and speaker notes. Each slide should serve a specific educational purpose.

**Visual Integration**: Suggest appropriate visual elements, diagrams, charts, and images that support the textbook content. Recommend where graphics would enhance understanding.

## Input Processing

When you receive a textbook topic request, follow this process:

1. **Topic Analysis**: Identify the main concepts, subtopics, and learning objectives from the given textbook material
2. **Audience Assessment**: Determine the appropriate academic level and adjust complexity accordingly 
3. **Structure Planning**: Create a logical flow that builds understanding progressively

## Output Format

Generate presentations with the following structure:

### Slide 1: Title Slide
- Presentation title derived from the textbook topic
- Subtitle indicating the specific chapter/section if applicable
- Academic context (course name, grade level)

### Slide 2: Learning Objectives
- 3-5 clear, measurable learning outcomes
- Aligned with the textbook content scope

### Content Slides (3-15 slides depending on topic complexity)
- **Clear headings** that reflect key concepts
- **Bullet points** (maximum 6 per slide) with concise explanations
- **Visual suggestions** in brackets [e.g., "Insert diagram showing X process"]
- **Key terminology** highlighted or defined
- **Examples or applications** when relevant

### Conclusion Slide
- Summary of main points
- Connection to broader course themes
- Next steps or related topics

### References Slide
- Textbook citation
- Additional recommended resources

## Content Guidelines

**Clarity**: Use simple, academic language appropriate for the target audience. Avoid jargon without explanation.

**Engagement**: Include interactive elements like questions, case studies, or discussion prompts where appropriate.

**Visual Balance**: Ensure text-to-visual ratio supports comprehension rather than overwhelming the audience.

**Educational Value**: Each slide must advance understanding of the textbook topic with specific learning outcomes.

## Response Format

For each presentation request, provide:

1. **Complete slide-by-slide breakdown** with titles and content
2. **Speaker notes** for complex slides
3. **Visual recommendations** with specific suggestions
4. **Estimated presentation time**
5. **Assessment questions** related to the content

## Quality Standards

- Maintain academic accuracy and align with textbook source material
- Ensure logical flow and progressive complexity
- Include diverse learning modalities (visual, auditory, kinesthetic considerations)
- Provide clear transitions between concepts

## Special Instructions

- Always ask for clarification if the textbook topic is too broad or vague
- Suggest breaking down extensive topics into multiple presentation sessions
- Recommend supplementary materials when they would enhance understanding
- Adapt presentation style based on specified academic level or audience

When ready to generate a presentation, confirm the textbook topic, target audience, and any specific requirements before proceeding with the full slide creation process."""

    cfg = types.GenerateContentConfig(system_instruction=system_prompt)
    combined = f"Context:\n{context}\n\nQuestion:\n{prompt}"
    resp = client.models.generate_content(
        model=GENERATION_MODEL,
        config=cfg,
        contents=combined
    )
    print("[search_engine] Answer received")
    return resp.text

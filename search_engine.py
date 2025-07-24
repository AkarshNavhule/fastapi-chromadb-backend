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

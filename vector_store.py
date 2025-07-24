import os
import chromadb
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from config import CHROMA_DB_DIR, EMBEDDING_MODEL
from google import genai
from google.genai import types
from itertools import islice

# 1) Ensure the directory exists
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
print(f"[vector_store] Ensured dir exists: {CHROMA_DB_DIR}")

# 2) Initialize PersistentClient
print("[vector_store] Initializing PersistentClient…")
client = PersistentClient(
    CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

print("[vector_store] Initializing Gemini client…")
genai_client = genai.Client()

def get_or_create_collection(name: str, reset: bool = False):
    exists = True
    try:
        client.get_collection(name)
    except NotFoundError:
        exists = False

    if reset and exists:
        print(f"[vector_store] Resetting existing collection '{name}'")
        client.delete_collection(name)
        exists = False

    if not exists:
        print(f"[vector_store] Creating collection '{name}'")
        return client.create_collection(name)
    else:
        print(f"[vector_store] Using existing collection '{name}'")
        return client.get_collection(name)

def batch_embed(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Splits texts into batches of <= batch_size and returns a flattened list of embeddings.
    """
    all_embeddings = []
    iterator = iter(texts)
    batch_num = 0

    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        batch_num += 1
        print(f"[vector_store] Embedding batch #{batch_num} (size={len(batch)})")
        resp = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        all_embeddings.extend(emb.values for emb in resp.embeddings)

    print(f"[vector_store] Total embeddings generated: {len(all_embeddings)}")
    return all_embeddings

def store_documents(collection_name: str, docs: list[dict]):
    """
    Stores (and resets) a collection with the provided docs.
    Each doc: {"id": str, "metadata": {"page_no": int, "text": str}}
    """
    print(f"[vector_store] Storing {len(docs)} docs into '{collection_name}'")

    texts     = [d["metadata"]["text"] for d in docs]
    ids       = [d["id"]               for d in docs]
    metadatas = [d["metadata"]         for d in docs]

    # 1) Reset / get the collection
    col = get_or_create_collection(collection_name, reset=True)

    # 2) Generate embeddings in batches of 100
    embeddings = batch_embed(texts, batch_size=100)

    # 3) Add to ChromaDB
    col.add(
        ids=ids,
        metadatas=metadatas,
        documents=texts,
        embeddings=embeddings
    )
    print(f"[vector_store] Added {len(docs)} docs to '{collection_name}' — check {CHROMA_DB_DIR}")

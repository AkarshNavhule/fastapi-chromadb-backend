import os
from chromadb import PersistentClient
from chromadb.config import Settings
from chromadb.errors import NotFoundError
from config import CHROMA_DB_DIR, EMBEDDING_MODEL
from google import genai
from google.genai import types
from google.cloud import firestore
from itertools import islice
import json

# Set up Firestore credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sahayak-d88d3-2e1f13a7b2bc.json"

# 1) Ensure the directory exists
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
print(f"[vector_store] Ensured dir exists: {CHROMA_DB_DIR}")

# 2) Initialize PersistentClient
print("[vector_store] Initializing PersistentClient…")
client = PersistentClient(
    CHROMA_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# 3) Initialize Firestore client
print("[firestore] Initializing Firestore client…")
db = firestore.Client()

# 4) Initialize Genai client
print("[vector_store] Initializing GenAI client…")
genai_client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for a batch of texts using Google's Generative AI."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        print(f"[vector_store] Embedding batch #{batch_num} (size={len(batch)})")
        
        try:
            resp = genai_client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            all_embeddings.extend(emb.values for emb in resp.embeddings)
        except Exception as e:
            print(f"Error generating embeddings for batch {batch_num}: {e}")
            # Add empty embeddings for failed batch to maintain index alignment
            all_embeddings.extend([[] for _ in batch])
    
    return all_embeddings


def fetch_firestore_collection(collection_name: str):
    """Fetch all documents from a Firestore collection."""
    print(f"[firestore] Fetching documents from '{collection_name}' collection…")
    
    try:
        collection_ref = db.collection(collection_name)
        docs = collection_ref.stream()
        
        documents = []
        for doc in docs:
            doc_data = doc.to_dict()
            doc_data['id'] = doc.id  # Include document ID
            documents.append(doc_data)
        
        print(f"[firestore] Retrieved {len(documents)} documents")
        return documents
    
    except Exception as e:
        print(f"Error fetching Firestore collection: {e}")
        return []

def prepare_document_for_chroma(doc_data: dict):
    """Convert Firestore document to format suitable for ChromaDB."""
    # Create a text representation of the document for embedding
    text_content = ""
    metadata = {}
    
    for key, value in doc_data.items():
        if key == 'id':
            continue
        
        # Add to text content for embedding
        text_content += f"{key}: {value}\n"
        
        # Add to metadata (ChromaDB metadata values should be strings, numbers, or booleans)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    
    return {
        'id': doc_data['id'],
        'text': text_content.strip(),
        'metadata': metadata
    }

def upload_to_chroma(documents: list, collection_name: str = "student_leaderboard"):
    """Upload documents to ChromaDB."""
    print(f"[chroma] Creating/getting collection '{collection_name}'…")
    
    try:
        # Try to get existing collection or create new one
        try:
            collection = client.get_collection(name=collection_name)
            print(f"[chroma] Found existing collection '{collection_name}'")
        except NotFoundError:
            collection = client.create_collection(name=collection_name)
            print(f"[chroma] Created new collection '{collection_name}'")
        
        # Prepare documents for ChromaDB
        ids = []
        texts = []
        metadatas = []
        
        print("[chroma] Preparing documents…")
        
        for doc in documents:
            prepared_doc = prepare_document_for_chroma(doc)
            ids.append(prepared_doc['id'])
            texts.append(prepared_doc['text'])
            metadatas.append(prepared_doc['metadata'])
        
        print(f"[chroma] Prepared {len(texts)} documents")
        
        # Generate embeddings for all texts using batch approach
        print("[chroma] Generating embeddings for all documents…")
        embeddings = get_embeddings_batch(texts, batch_size=100)
        
        # Filter out documents with failed embeddings
        valid_data = []
        for i, embedding in enumerate(embeddings):
            if embedding:  # Only include documents with valid embeddings
                valid_data.append({
                    'id': ids[i],
                    'text': texts[i],
                    'metadata': metadatas[i],
                    'embedding': embedding
                })
            else:
                print(f"Warning: Skipping document {ids[i]} due to failed embedding")
        
        if not valid_data:
            print("No valid embeddings generated. Aborting upload.")
            return False
        
        print(f"[chroma] Generated embeddings for {len(valid_data)} documents")
        
        # Upload to ChromaDB in batches
        upload_batch_size = 100
        total_batches = (len(valid_data) + upload_batch_size - 1) // upload_batch_size
        
        for i in range(0, len(valid_data), upload_batch_size):
            batch_data = valid_data[i:i + upload_batch_size]
            
            batch_ids = [item['id'] for item in batch_data]
            batch_texts = [item['text'] for item in batch_data]
            batch_metadatas = [item['metadata'] for item in batch_data]
            batch_embeddings = [item['embedding'] for item in batch_data]
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=batch_embeddings
            )
            
            batch_num = (i // upload_batch_size) + 1
            print(f"[chroma] Uploaded batch {batch_num}/{total_batches} ({len(batch_data)} documents)")
        
        print(f"[chroma] Successfully uploaded {len(valid_data)} documents to ChromaDB")
        return True
        
    except Exception as e:
        print(f"Error uploading to ChromaDB: {e}")
        return False

def main():
    """Main function to orchestrate the upload process."""
    print("Starting Firestore to ChromaDB migration…")
    
    # Fetch documents from Firestore
    firestore_docs = fetch_firestore_collection("student_leaderboard")
    
    if not firestore_docs:
        print("No documents found in Firestore collection. Exiting.")
        return
    
    # Upload to ChromaDB
    success = upload_to_chroma(firestore_docs)
    
    if success:
        print("✅ Migration completed successfully!")
    else:
        print("❌ Migration failed!")

if __name__ == "__main__":
    main()

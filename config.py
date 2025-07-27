import os
from dotenv import load_dotenv

load_dotenv()

# Force absolute path so it's always the same folder
_raw = os.getenv("CHROMA_DB_DIR", "./chroma_db")
CHROMA_DB_DIR = os.path.abspath(_raw)

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL  = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash-lite"
TOP_K            = 10

print(f"[config] Working dir: {os.getcwd()}")
print(f"[config] ChromaDB_DIR: {CHROMA_DB_DIR}")
print(f"[config] Gemini key loaded: {'YES' if GEMINI_API_KEY else 'NO'}")

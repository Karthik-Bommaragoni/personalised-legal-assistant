import os
from dotenv import load_dotenv

load_dotenv()


os.environ["HF_HOME"] = r"D:\Langchain\legal_assistant\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\Langchain\legal_assistant\hf_cache"

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # no default secret placeholder
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embedding model (HF)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths 
TEXT_DIR = os.getenv("TEXT_DIR", r"D:\Langchain\legal_assistant\text_files")
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", r"D:\Langchain\legal_assistant\data\vector_store")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

RESPONSE_MODES = {
    "concise": "Provide a brief, summarized answer in 2-3 sentences focusing only on key points.",
    "detailed": "Provide a comprehensive, detailed explanation with case references, legal reasoning, and relevant precedents."
}

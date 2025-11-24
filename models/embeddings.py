import sys
sys.path.insert(0, r"D:\Langchain\legal_assistant")

from config.config import EMBEDDING_MODEL


try:
    
    from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
    from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings(device: str = "cpu"):
    """Return a HuggingFaceEmbeddings instance. Device can be 'cpu' or 'cuda'."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

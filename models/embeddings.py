# models/embeddings.py — resilient HuggingFaceEmbeddings import
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try a couple of common ways the HuggingFace embeddings class is packaged
HuggingFaceEmbeddings = None
try:
    # langchain-community packaged embeddings
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HuggingFaceEmbeddings = HuggingFaceEmbeddings
except Exception:
    try:
        # older monolithic langchain
        from langchain.embeddings import HuggingFaceEmbeddings
        HuggingFaceEmbeddings = HuggingFaceEmbeddings
    except Exception:
        try:
            # third-party connector name sometimes used
            from langchain_huggingface import HuggingFaceEmbeddings
            HuggingFaceEmbeddings = HuggingFaceEmbeddings
        except Exception:
            raise ImportError(
                "Could not import HuggingFaceEmbeddings. "
                "Install langchain-community or langchain or langchain-huggingface. "
                "See requirements."
            )

from config.config import EMBEDDING_MODEL

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

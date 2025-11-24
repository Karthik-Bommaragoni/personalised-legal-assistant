import os
import sys
import traceback
from typing import List


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


try:
    from langchain_community.vectorstores import FAISS
except Exception:
    try:
        from langchain.vectorstores import FAISS
    except Exception:
        FAISS = None  


try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None  


try:
    
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.docstore.document import Document
    except Exception:
        Document = None  

from config.config import TEXT_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS
from models.embeddings import get_embeddings  

if FAISS is None:
    raise ImportError(
        "FAISS import failed. Install 'langchain-community' or a compatible 'langchain' package. "
        "On some platforms you may need a different FAISS build or to prebuild the vectorstore locally."
    )

if RecursiveCharacterTextSplitter is None:
    raise ImportError(
        "RecursiveCharacterTextSplitter import failed. Install 'langchain-text-splitters' or a compatible langchain package."
    )

if Document is None:
    raise ImportError(
        "Document class import failed. Install a compatible LangChain package (langchain-core or langchain)."
    )


def load_text_files() -> List[Document]:
    """Load all .txt files from TEXT_DIR and return a list of LangChain Documents."""
    docs: List[Document] = []

    if not os.path.exists(TEXT_DIR):
        raise FileNotFoundError(f"TEXT_DIR does not exist: {TEXT_DIR}")

    text_files = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith(".txt")]
    print(f"Found {len(text_files)} text files in {TEXT_DIR}")

    for text_file in text_files:
        file_path = os.path.join(TEXT_DIR, text_file)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
                if not text:
                    print(f"Skipping empty file: {text_file}")
                    continue
                docs.append(Document(page_content=text, metadata={"source": text_file}))
        except Exception as e:
            print(f"Error reading {text_file}: {e}")
            traceback.print_exc()

    return docs


def create_vector_store() -> FAISS:
    """
    Create a FAISS vector store from all text files in TEXT_DIR.
    Saves the vector store locally at VECTOR_DB_DIR.
    """
    documents = load_text_files()

    if not documents:
        raise ValueError("No documents found to process. Please add .txt files to TEXT_DIR.")

    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from documents")

    print("Creating embeddings...")
    embeddings = get_embeddings()  
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"Vector store saved to: {VECTOR_DB_DIR}")
    return vectorstore


def load_vector_store() -> FAISS:
    """Load an existing FAISS vector store from VECTOR_DB_DIR."""
    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_DB_DIR}. Run create_vector_store() first.")

    embeddings = get_embeddings()
    return FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)


def query_vector_store(query: str, top_k: int = TOP_K_RESULTS) -> List[Document]:
    """Query the vector store and return top_k relevant Documents."""
    vectorstore = load_vector_store()
    results = vectorstore.similarity_search(query, k=top_k)
    return results

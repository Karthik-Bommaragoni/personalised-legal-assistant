import os
import traceback
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import sys
sys.path.insert(0, r"D:\Langchain\legal_assistant")
from config.config import TEXT_DIR, VECTOR_DB_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS
from models.embeddings import get_embeddings

def load_text_files():
    docs = []
    if not os.path.exists(TEXT_DIR):
        raise FileNotFoundError(f"TEXT_DIR does not exist: {TEXT_DIR}")
    
    text_files = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith(".txt")]
    print(f" Found {len(text_files)} text files in {TEXT_DIR}")
    
    for text_file in text_files:
        file_path = os.path.join(TEXT_DIR, text_file)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
                if not text:
                    print(f"Skipping empty file: {text_file}")
                    continue
                docs.append(Document(page_content=text, metadata={"source": text_file}))
        except Exception as e:
            print(f"Error reading {text_file}: {e}")
            traceback.print_exc()
    
    return docs

def create_vector_store(device: str = "cpu"):
    
    documents = load_text_files()
    
    if not documents:
        raise ValueError("No documents found to process. Please add .txt files to TEXT_DIR.")
    
    print(f"Splitting {len(documents)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    
    
    
    embeddings = get_embeddings(device=device)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_DIR)
    
    print(f"Vector store saved to: {VECTOR_DB_DIR}")
    return vectorstore

def load_vector_store(device: str = "cpu"):
    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_DB_DIR}. Run create_vector_store() first.")
    
    embeddings = get_embeddings(device=device)
    return FAISS.load_local(VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)

def query_vector_store(query, top_k=TOP_K_RESULTS, device: str = "cpu"):
    vectorstore = load_vector_store(device=device)
    results = vectorstore.similarity_search(query, k=top_k)
    return results

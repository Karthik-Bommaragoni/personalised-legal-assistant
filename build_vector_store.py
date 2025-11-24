import sys
import os

project_root = r"D:\Langchain\legal_assistant"
sys.path.insert(0, project_root)

from utils.vector_store import create_vector_store

if __name__ == "__main__":
    print("Building vector store from existing text files...")
    try:
        vectorstore = create_vector_store()
        print("Vector store created successfully!")
        print(f"Ready to use in app!")
    except Exception as e:
        print(f"Error: {e}")
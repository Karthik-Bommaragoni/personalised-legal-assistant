Personalised Legal Assistant 
An intelligent legal assistant that answers queries about Indian Supreme Court judgments using:
RAG (Retrieval-Augmented Generation)
FAISS vector search
LLMs (Groq/GPT/Gemini)
Streamlit user interface
This project allows users to ask questions about legal cases and receive accurate, context-aware responses grounded in real court documents.

##Retrieval-Augmented Generation (RAG)
Converts judgment PDFs → text
Splits text into chunks
Creates FAISS vector embeddings (sentence-transformers)
Retrieves the most relevant case law for each query

##Large Language Model Integration
Supports:
Groq Llama-3.3-70B
(Optional) OpenAI / Gemini
Configured through the config/config.py file

##Response Modes
Concise – A brief summary
Detailed – In-depth legal reasoning with context.

##Optional Web Search
Uses DuckDuckGo search to supplement outdated legal information.

Streamlit UI


import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import streamlit as st

# Project modules
from models.llm import get_legal_response

try:
    from utils.web_search import search_web
except Exception:
    search_web = None

# Use uploaded screenshot path
UPLOADED_SCREENSHOT_PATH = "/mnt/data/Screenshot 2025-11-24 031110.png"

st.set_page_config(page_title="Personalized Legal Assistant", page_icon="⚖️", layout="wide")


def render_sidebar():
    st.sidebar.title("Settings")
    response_mode = st.sidebar.radio(
        "Response mode",
        ["detailed", "concise"],
        index=0,
        format_func=lambda x: x.capitalize()
    )
    use_web = st.sidebar.checkbox("Enable live web search", value=False)
    return response_mode, use_web


def main():
    st.title("⚖️ Personalized Legal Assistant")
    st.caption("Ask about Supreme Court cases. Retrieval + LLM powered answers.")
    header_left, header_right = st.columns([0.85, 0.15])
    with header_right:
        if st.button("Clear Chat",key="clear_chat_button"):
            st.session_state.messages = []
            st.rerun()



    response_mode, use_web = render_sidebar()

    # Show uploaded screenshot
    if os.path.exists(UPLOADED_SCREENSHOT_PATH):
        st.image(UPLOADED_SCREENSHOT_PATH, caption="UI reference (optional)", use_column_width=False)

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    query = st.text_input("Ask about Indian Supreme Court cases...", key="query_input")

    # show previous chat messages
    for m in st.session_state.messages:
        role = m.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(m.get("content", ""))

    if query:
        if st.button("Ask"):
            # append user message
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            # generate via get_legal_response (
            with st.chat_message("assistant"):
                with st.spinner("Generating answer..."):
                    try:
                        result = get_legal_response(query, response_mode=response_mode)
                        answer = result.get("result", "No answer returned.")
                        sources = result.get("source_documents", []) or []

                        # attach source list if available
                        if sources:
                            answer += "\n\n**Sources:**\n"
                            seen = set()
                            for doc in sources[:6]:
                                src = getattr(doc, "metadata", {}).get("source", str(getattr(doc, "uri", "Unknown")))
                                src = src.replace(".txt", "")
                                if src not in seen:
                                    seen.add(src)
                                    answer += f"- {src}\n"

                        # live web search augmentation
                        if use_web and search_web is not None:
                            try:
                                web_results = search_web(query, max_results=2)
                                if web_results and "error" not in web_results[0]:
                                    answer += "\n\n**Recent web results:**\n"
                                    for r in web_results[:2]:
                                        title = r.get("title") or "Result"
                                        url = r.get("url") or r.get("href") or ""
                                        answer += f"- {title} — {url}\n"
                            except Exception as e:
                                answer += f"\n\n_Web search failed: {e}_"

                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        err = f"Error generating response: {e}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})



if __name__ == "__main__":
    main()

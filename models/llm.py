import json
import traceback
import os
from typing import Any, Dict, List


import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import GROQ_API_KEY, GROQ_MODEL, RESPONSE_MODES
from utils.vector_store import load_vector_store


try:
    from groq import Groq
except Exception:
    Groq = None

# Project-root debug paths 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBUG_DUMP_JSON = os.path.join(PROJECT_ROOT, "groq_debug.json")
DEBUG_DUMP_TXT = os.path.join(PROJECT_ROOT, "groq_debug.txt")


def _safe_to_dict(obj: Any) -> Any:
    """Attempt to convert common objects to python-native structures for debugging."""
    if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
        return obj
    try:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
    except Exception:
        pass
    try:
        if hasattr(obj, "dict"):
            return obj.dict()
    except Exception:
        pass
    # protobuf Message 
    try:
        from google.protobuf.json_format import MessageToDict
        try:
            return MessageToDict(obj)
        except Exception:
            pass
    except Exception:
        pass
    try:
        return repr(obj)
    except Exception:
        return str(type(obj))


def _extract_text_from_groq_completion(completion: Any) -> str:
    """
    Extract text from Groq ChatCompletion shapes we observed:
    - primary path: completion.choices[0].message.content
    - fallback: choices[0].text, dict-like shapes, JSON preview
    """
    if completion is None:
        return ""

   
    try:
        if hasattr(completion, "choices") and isinstance(completion.choices, list) and len(completion.choices) > 0:
            c0 = completion.choices[0]
           
            try:
                if hasattr(c0, "message") and getattr(c0.message, "content", None) is not None:
                    return c0.message.content
            except Exception:
                pass
            
            try:
                if hasattr(c0, "text") and getattr(c0, "text", None):
                    return c0.text
            except Exception:
                pass
            
            try:
                if isinstance(c0, dict):
                    if "message" in c0 and isinstance(c0["message"], dict) and c0["message"].get("content"):
                        return c0["message"]["content"]
                    if c0.get("text"):
                        return c0.get("text")
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: try converting whole completion to dict and drill
    try:
        comp_dict = _safe_to_dict(completion)
        if isinstance(comp_dict, dict):
            choices = comp_dict.get("choices") or comp_dict.get("outputs") or comp_dict.get("results") or []
            if choices:
                first = choices[0]
                if isinstance(first, dict):
                    msg = first.get("message") or first.get("content") or first.get("text")
                    if isinstance(msg, dict) and msg.get("content"):
                        return msg.get("content")
                    if isinstance(msg, str):
                        return msg
                if isinstance(first, str):
                    return first
    except Exception:
        pass

    # If completion is string-like
    try:
        if isinstance(completion, str):
            return completion
    except Exception:
        pass

    
    try:
        safe = _safe_to_dict(completion)
        if isinstance(safe, (dict, list)):
            return json.dumps(safe, indent=2, ensure_ascii=False)[:20000]
        return str(safe)[:20000]
    except Exception:
        return repr(completion)


def _write_debug_dump(completion: Any = None, exc: Exception = None) -> str:
    """Write debug dump to project root; return path written (JSON preferred)."""
    dump = {
        "error": repr(exc) if exc else None,
        "traceback": traceback.format_exc() if exc else None,
        "completion_preview": None,
        "completion_repr": None,
    }
    try:
        dump["completion_preview"] = _safe_to_dict(completion)
    except Exception:
        dump["completion_preview"] = None
    try:
        dump["completion_repr"] = repr(completion)
    except Exception:
        dump["completion_repr"] = str(type(completion))

    try:
        with open(DEBUG_DUMP_JSON, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2, ensure_ascii=False)
        return DEBUG_DUMP_JSON
    except Exception:
        try:
            with open(DEBUG_DUMP_TXT, "w", encoding="utf-8") as fh:
                fh.write("Error: " + (repr(exc) if exc else "None") + "\n\n")
                if exc:
                    fh.write(traceback.format_exc() + "\n\n")
                try:
                    fh.write("completion repr:\n" + repr(completion) + "\n\n")
                except Exception:
                    fh.write("could not repr completion\n")
                try:
                    fh.write("safe dict preview:\n" + json.dumps(_safe_to_dict(completion), indent=2, ensure_ascii=False))
                except Exception:
                    fh.write("could not serialize completion preview\n")
            return DEBUG_DUMP_TXT
        except Exception:
            return ""


def _maybe_remove_debug_files():
    """Remove prior debug files if present (best-effort)."""
    try:
        if os.path.exists(DEBUG_DUMP_JSON):
            os.remove(DEBUG_DUMP_JSON)
    except Exception:
        pass
    try:
        if os.path.exists(DEBUG_DUMP_TXT):
            os.remove(DEBUG_DUMP_TXT)
    except Exception:
        pass


def _protobuf_hint() -> str:
    """Return helpful hint about protobuf version if it appears relevant."""
    try:
        import google.protobuf as pb
        ver = getattr(pb, "__version__", None)
        if ver:
            major = ver.split(".")[0]
            if major.isdigit() and int(major) >= 4:
                return "Detected protobuf >=4.x. If you encounter descriptor errors, run: pip install protobuf==3.20.3"
    except Exception:
        pass
    return ""


def get_legal_response(query: str, response_mode: str = "detailed") -> Dict[str, Any]:
    """
    Retrieve context from vector store, call Groq chat completions, and return:
        {"result": <str>, "source_documents": <list>}
    On failure, a safe fallback is returned and a debug dump is written.
    """
    # Pre-checks
    if Groq is None:
        raise ImportError("groq package not installed. Install with `pip install groq`.")
    if GROQ_API_KEY is None or GROQ_API_KEY.strip() == "":
        raise EnvironmentError("GROQ_API_KEY is not set. Please set it in your .env or environment.")

    # Load vectorstore
    try:
        vectorstore = load_vector_store()
    except Exception as e:
        raise RuntimeError(f"Failed to load vector store: {e}")

    # Retrieve context documents 
    try:
        docs = vectorstore.similarity_search(query, k=5)
    except Exception:
        docs = []
        # Continue without docs if retrieval failed

    # Build prompt context
    context = "\n\n".join([getattr(d, "page_content", str(d)) for d in docs])
    mode_instruction = RESPONSE_MODES.get(response_mode, RESPONSE_MODES.get("detailed", "Provide a detailed answer."))
    prompt = (
        "You are an expert Indian legal assistant specializing in Supreme Court judgments.\n"
        f"{mode_instruction}\n\n"
        "Use the following Supreme Court case documents to answer the question.\n"
        "If you don't know the answer, say so clearly. Always cite case names when relevant.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    client = Groq(api_key=GROQ_API_KEY)

   
    _maybe_remove_debug_files()

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048, 
        )

        text = _extract_text_from_groq_completion(completion)

        
        _maybe_remove_debug_files()

        return {"result": text, "source_documents": docs}

    except Exception as exc:
        
        dump_path = _write_debug_dump(locals().get("completion", None), exc)
        proto_hint = _protobuf_hint()
        proto_hint_text = f"\n\n{proto_hint}" if proto_hint else ""

        
        if docs:
            try:
                if response_mode == "concise":
                    fallback = " ".join([getattr(d, "page_content", "")[:800].split("\n", 1)[0] for d in docs[:2]])
                else:
                    fallback = "\n\n---\n\n".join([getattr(d, "page_content", "")[:2000] for d in docs[:4]])
                fallback_answer = (
                    "LLM call failed. Returning retrieved document excerpts as fallback.\n\n"
                    + fallback
                    + f"\n\n(Debug dump written to: {dump_path})"
                    + proto_hint_text
                )
            except Exception:
                fallback_answer = (
                    f"LLM call failed and assembling fallback also failed. Debug: {dump_path}" + proto_hint_text
                )
        else:
            fallback_answer = f"LLM call failed and no documents available. Debug: {dump_path}" + proto_hint_text

        return {"result": fallback_answer, "source_documents": docs}

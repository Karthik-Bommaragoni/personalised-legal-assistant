from duckduckgo_search import DDGS

def search_web(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} Indian Supreme Court recent", max_results=max_results))
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title") or "",
                "snippet": r.get("body") or r.get("snippet") or "",
                "url": r.get("href") or r.get("url") or ""
            })
        return formatted
    except Exception as e:
        print(f"Web search error: {e}")
        return [{"error": str(e)}]

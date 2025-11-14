import requests
from typing import List, Dict
from config import (
    WEB_SEARCH_MAX_RESULTS,
)

# Change this to your own instance URL:
SEARXNG_URL = "http://localhost:8090"  # or "http://10.40.96.173:8080"

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/127.0.0.0 Safari/537.36"
)


def _searxng(query: str) -> List[Dict]:
    """
    Query your local or LAN SearXNG instance.
    Returns structured list of dicts with title, url, snippet.
    """
    try:
        resp = requests.get(
            f"{SEARXNG_URL.rstrip('/')}/search",
            params={"q": query, "format": "json"},
            headers={"User-Agent": _BROWSER_UA},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results: List[Dict] = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")[:400],
            })
        return results
    except Exception as e:
        return [{
            "title": "Web search failed",
            "url": "",
            "snippet": f"Error contacting SearXNG at {SEARXNG_URL}: {e}",
        }]


def search_web(query: str) -> List[Dict]:
    """
    Unified search entrypoint for the General Agent.
    Always uses local SearXNG instance for live web data.
    """
    try:
        results = _searxng(query)
        if results:
            return results[: (WEB_SEARCH_MAX_RESULTS or 5)]
    except Exception as e:
        return [{
            "title": "Web search error",
            "url": "",
            "snippet": str(e),
        }]

    return [{
        "title": "No results",
        "url": "",
        "snippet": f"No web results found for: {query}",
    }]

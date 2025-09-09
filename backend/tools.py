"""Dynamic documentation URL tools for model-driven grounding.

This module exposes two tools:
  - list_doc_urls(filter: str = "", limit: int = 50, offset: int = 0)
  - answer_with_urls(question: str, urls: List[str])

The model decides which URLs to ground on. No heuristic ranking or static boosts.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, Any, List

from dotenv import load_dotenv
import requests
import google.generativeai as genai

try:  # URL grounding helpers (optional)
    from google.generativeai.types import grounding  # type: ignore
except Exception:  # pragma: no cover
    grounding = None  # type: ignore

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:  # pragma: no cover
        pass

MODEL_NAME = os.getenv("GEMINI_RAG_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

SITEMAPS = [
    "https://docs.atlan.com/sitemap.xml",
    "https://developer.atlan.com/sitemap.xml",
]

ALLOW_DOMAINS = {"docs.atlan.com", "developer.atlan.com"}
# Single-load strategy: fetch sitemaps once and reuse for lifetime of process.
MAX_URLS_PER_ANSWER = 10
LIST_URLS_HARD_LIMIT = 100

_URL_CACHE: Dict[str, Any] = {"loaded_at": 0.0, "urls": []}


def _fetch_sitemap(url: str, timeout: int = 15) -> List[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        out: List[str] = []
        for elem in root.iter():
            if elem.tag.endswith("loc") and elem.text:
                loc = elem.text.strip()
                # Basic allowlist check
                if any(d in loc for d in ALLOW_DOMAINS):
                    out.append(loc)
        return out
    except Exception:
        return []


def _ensure_loaded() -> None:
    # Load only once; no periodic refresh.
    if _URL_CACHE["urls"]:
        return
    combined: List[str] = []
    for sm in SITEMAPS:
        combined.extend(_fetch_sitemap(sm))
    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for u in combined:
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    _URL_CACHE.update({"loaded_at": time.time(), "urls": ordered})


def list_doc_urls(filter: str = "", limit: int = 50, offset: int = 0) -> Dict[str, Any]:  # noqa: A002 (filter name ok here)
    """Return a (possibly filtered) slice of documentation URLs.

    Args:
        filter: substring (case-insensitive) to match inside URL.
        limit: max URLs to return (capped by LIST_URLS_HARD_LIMIT).
        offset: pagination offset.
    """
    _ensure_loaded()
    urls: List[str] = _URL_CACHE.get("urls", [])
    if filter:
        f = filter.lower()
        urls = [u for u in urls if f in u.lower()]
    total = len(urls)
    limit = max(1, min(limit, LIST_URLS_HARD_LIMIT))
    slice_ = urls[offset : offset + limit]
    return {
        "count_total": total,
        "offset": offset,
        "returned": len(slice_),
        "urls": slice_,
        "refreshed_at": _URL_CACHE.get("loaded_at"),
    }


def answer_with_urls(question: str, urls: List[str]) -> Dict[str, Any]:
    """Generate an answer grounded ONLY on the provided URLs.

    Model responsibility: caller chooses relevant URLs first via list_doc_urls.
    We enforce domain allowlist, deduplicate, and cap count.
    Returns: { answer, sources, used_urls, error? }
    """
    if not GEMINI_API_KEY:
        return {"answer": "Grounded answering disabled: missing GEMINI_API_KEY.", "sources": [], "used_urls": []}
    if not urls:
        return {"error": "No URLs provided", "answer": "", "sources": [], "used_urls": []}
    # Normalize & filter
    cleaned: List[str] = []
    seen = set()
    for u in urls:
        if not isinstance(u, str):
            continue
        u = u.strip()
        if not u or not u.startswith("http"):
            continue
        # Domain allowlist
        try:
            host = u.split("//", 1)[1].split("/", 1)[0]
        except Exception:
            continue
        if host not in ALLOW_DOMAINS:
            continue
        if u not in seen:
            cleaned.append(u)
            seen.add(u)
        if len(cleaned) >= MAX_URLS_PER_ANSWER:
            break
    if not cleaned:
        return {"error": "No allowed URLs after filtering", "answer": "", "sources": [], "used_urls": []}

    prompt = (
        "You are Atlan's support assistant. Use ONLY the fetched URL contexts."
        " Provide a concise, factual answer. If a procedural setup question, give numbered steps."
        " After the answer add a blank line then 'Sources:' and list exactly the URLs provided (or only those actually used if you can distinguish)."
        f"\nOriginal Question: {question}\n"
    )
    answer_text = ""
    meta: Dict[str, Any] = {"requested_urls": cleaned}
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        if grounding and hasattr(grounding, "UrlContext"):
            url_ctx = grounding.UrlContext(urls=cleaned)  # type: ignore
            resp = model.generate_content([prompt, url_ctx])
        else:  # Fallback embed URLs in prompt
            embedded = prompt + "\nURLs:\n" + "\n".join(cleaned)
            resp = model.generate_content(embedded)
        answer_text = (getattr(resp, "text", "") or "").strip()
        meta["candidates"] = len(getattr(resp, "candidates", []) or [])
    except Exception as e:  # pragma: no cover
        return {"error": f"generation_error: {e}", "answer": "", "sources": cleaned, "used_urls": cleaned, "meta": meta}
    return {"answer": answer_text, "sources": cleaned, "used_urls": cleaned, "meta": meta}


__all__ = ["list_doc_urls", "answer_with_urls"]

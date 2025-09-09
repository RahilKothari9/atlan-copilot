"""Agent tool implementations (Feature 4: RAG URL-grounded answers).

Currently implements:
  answer_with_rag(question: str) -> Dict[str, Any]

Uses Gemini URL context grounding (best-effort). Falls back gracefully if
grounding types unavailable or request fails.
"""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv

# Ensure .env is loaded even when this module is imported directly (e.g., test.py)
load_dotenv()

import google.generativeai as genai
import requests

try:
    # Newer library versions expose grounding helpers
    from google.generativeai.types import grounding  # type: ignore
except Exception:  # pragma: no cover
    grounding = None  # type: ignore

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

MODEL_NAME = os.getenv("GEMINI_RAG_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

# Base sitemap endpoints
PRODUCT_SITEMAP = "https://docs.atlan.com/sitemap.xml"
DEVELOPER_SITEMAP = "https://developer.atlan.com/sitemap.xml"

# Caches
_SITEMAP_CACHE: Dict[str, Any] = {
    "loaded_at": 0.0,
    "product_urls": [],
    "developer_urls": [],
}

# Static fallback (minimal seed) so we still produce value offline
DOC_FALLBACK: List[str] = [
    "https://docs.atlan.com/docs/connecting-to-snowflake",
    "https://docs.atlan.com/docs/connecting-to-bigquery",
    "https://docs.atlan.com/docs/ingestion-overview",
    "https://docs.atlan.com/docs/lineage",
    "https://docs.atlan.com",
    "https://developer.atlan.com/"
]


def _fetch_sitemap(url: str, timeout: int = 10) -> List[str]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        txt = r.text
        # Parse XML and collect <loc> values
        root = ET.fromstring(txt)
        locs: List[str] = []
        for elem in root.iter():
            if elem.tag.endswith('loc') and elem.text:
                locs.append(elem.text.strip())
        return locs
    except Exception:
        return []


def _ensure_sitemaps(refresh: bool = False) -> None:
    # Refresh every 6 hours or on demand
    if not refresh and time.time() - _SITEMAP_CACHE["loaded_at"] < 6 * 3600 and _SITEMAP_CACHE["product_urls"]:
        return
    product_urls = _fetch_sitemap(PRODUCT_SITEMAP)
    developer_urls = _fetch_sitemap(DEVELOPER_SITEMAP)
    # Basic cleanup: only keep docs pages (exclude images, assets)
    product_urls = [u for u in product_urls if "/docs/" in u][:2000]
    developer_urls = [u for u in developer_urls if u.startswith("https://developer.atlan.com")]  # keep all for now
    if product_urls or developer_urls:
        _SITEMAP_CACHE.update({
            "loaded_at": time.time(),
            "product_urls": product_urls,
            "developer_urls": developer_urls,
        })


def _score_url(url: str, tokens: List[str]) -> int:
    score = 0
    lower = url.lower()
    for t in tokens:
        if t and t in lower:
            score += 1
    return score


def _select_candidate_urls(question: str, max_total: int = 20) -> Tuple[List[str], List[str]]:
    _ensure_sitemaps()
    tokens = [t for t in question.lower().replace('?', ' ').split() if len(t) > 2]
    dev_signals = {"api", "sdk", "endpoint", "rest", "graphql", "token", "header", "bearer", "auth"}
    is_dev_question = any(t in dev_signals for t in tokens)
    product_urls = _SITEMAP_CACHE.get("product_urls") or []
    developer_urls = _SITEMAP_CACHE.get("developer_urls") or []
    if not product_urls and not developer_urls:
        # Fallback
        return DOC_FALLBACK, DOC_FALLBACK
    # Score
    prod_scored = sorted(product_urls, key=lambda u: _score_url(u, tokens), reverse=True)[: max_total]
    dev_scored = sorted(developer_urls, key=lambda u: _score_url(u, tokens), reverse=True)[: max_total]
    if is_dev_question:
        primary = dev_scored[: max_total - 5]
        secondary = prod_scored[:5]
    else:
        primary = prod_scored[: max_total - 5]
        secondary = dev_scored[:5]
    # Deduplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for u in primary + secondary:
        if u not in seen:
            ordered.append(u)
            seen.add(u)
    return ordered, dev_scored if is_dev_question else prod_scored


def _build_prompt(question: str, candidate_urls: List[str]) -> str:
    url_list = "\n".join(candidate_urls)
    return (
        "You are Atlan's helpdesk AI. Use ONLY the provided candidate URLs as authoritative sources.\n"
        "If information is missing, reply that you are unsure.\n"
        "List cited sources at the end under 'Sources:'.\n"
        f"Question: {question}\n\nCandidate URLs (model may choose subset):\n{url_list}\n"
    )


def answer_with_rag(question: str) -> Dict[str, Any]:
    """Answer questions grounded on Atlan documentation URLs.

    Returns dict: { answer: str, sources: [urls], raw: optional }.
    """
    if not GEMINI_API_KEY:
        return {
            "answer": "RAG disabled: missing GEMINI_API_KEY.",
            "sources": DOC_FALLBACK,
        }
    candidate_urls, primary_ranked = _select_candidate_urls(question)
    prompt = _build_prompt(question, candidate_urls)
    answer_text = "(No answer)"
    raw_meta: Dict[str, Any] = {}
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        # Use only top 12 to avoid excessive grounding payload
        grounding_subset = candidate_urls[:12]
        if grounding and hasattr(grounding, "UrlContext"):
            url_ctx = grounding.UrlContext(urls=grounding_subset)  # type: ignore
            response = model.generate_content([prompt, url_ctx])
        else:  # Fallback: embed URLs into prompt
            augmented = prompt + "\n(Embedded URLs)\n" + "\n".join(grounding_subset)
            response = model.generate_content(augmented)
        answer_text = (getattr(response, "text", None) or "").strip() or answer_text
        raw_meta = {
            "candidates": len(getattr(response, "candidates", []) or []),
            "grounding_urls": grounding_subset,
        }
    except Exception as e:  # pragma: no cover
        answer_text = f"RAG error: {e}"[:400]
    return {"answer": answer_text, "sources": candidate_urls, "raw": raw_meta}


__all__ = ["answer_with_rag", "_ensure_sitemaps"]

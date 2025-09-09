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
import re

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


KEYWORD_BOOSTS = {
    'snowflake': 8,
    'connecting-to-snowflake': 10,
    'connection': 3,
    'ingest': 4,
    'ingestion': 4,
    'api': 4,
    'sdk': 4,
    'token': 4,
    'auth': 3,
    'lineage': 3,
    'glossary': 2,
}


def _score_url(url: str, tokens: List[str]) -> int:
    lower = url.lower()
    score = 0
    for t in tokens:
        if t and t in lower:
            score += 1
    for k, b in KEYWORD_BOOSTS.items():
        if k in lower:
            score += b
    # slight preference for docs over developer unless dev-specific
    if 'developer.atlan.com' in lower:
        score -= 1
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
    # Curated key product pages (added if relevant)
    curated_hits: List[str] = []
    curated_map = {
        'snowflake': [
            'https://docs.atlan.com/docs/connecting-to-snowflake',
            'https://docs.atlan.com/docs/ingesting-from-snowflake',
        ],
        'lineage': [
            'https://docs.atlan.com/docs/lineage'
        ],
        'glossary': [
            'https://docs.atlan.com/docs/glossary-overview'
        ],
        'bigquery': [
            'https://docs.atlan.com/docs/connecting-to-bigquery'
        ],
    }
    qlow = question.lower()
    for k, pages in curated_map.items():
        if k in qlow:
            for p in pages:
                curated_hits.append(p)
    if is_dev_question:
        primary = dev_scored[: max_total - 6]
        secondary = prod_scored[:6]
    else:
        primary = prod_scored[: max_total - 4]
        secondary = dev_scored[:4]
    # Prepend curated hits (dedup later)
    primary = curated_hits + primary
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
        "You are Atlan's helpdesk AI specialized in factual, source-grounded answers.\n"
        "Follow these rules strictly:\n"
        "1. Use ONLY content that plausibly exists in the provided URLs (no external knowledge).\n"
        "2. Prefer docs.atlan.com pages for step-by-step product setup (Snowflake, BigQuery, ingestion).\n"
        "3. For connection or ingestion questions, produce clear, concise numbered steps (1., 2., 3., ...).\n"
        "4. If at least one highly relevant page (like connecting-to-snowflake) is present, you MUST answer using it—do not say you're unsure.\n"
        "5. Only if none of the URLs obviously cover the topic may you say you're not fully sure.\n"
        "6. End with a blank line then 'Sources:' and only the URLs you actually relied on, one per line.\n"
        f"Original Question: {question}\n\nCandidate Documentation URLs (ranked):\n{url_list}\n"
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
    # For product-style question (no dev signals) heavily favor product docs in grounding subset
    dev_keywords = {"api","sdk","endpoint","graphql","token","rest"}
    tokens = set(t for t in re.split(r"\W+", question.lower()) if t)
    is_dev = any(t in dev_keywords for t in tokens)
    if not is_dev:
        # Filter out most developer urls keeping at most 2
        prod_only = [u for u in candidate_urls if 'docs.atlan.com' in u]
        dev_tail = [u for u in candidate_urls if 'developer.atlan.com' in u][:2]
        if prod_only:
            candidate_urls = prod_only + dev_tail
    prompt = _build_prompt(question, candidate_urls)
    answer_text = "(No answer)"
    raw_meta: Dict[str, Any] = {}
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        grounding_subset = candidate_urls[:10]
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
    # Second-pass retry if unsure but we have a strong curated page
    lower_ans = answer_text.lower()
    unsure_phrase = "not fully sure" in lower_ans or "i am unsure" in lower_ans
    key_pages = [u for u in candidate_urls if any(k in u for k in ["connecting-to-snowflake","connecting-to-bigquery"]) ]
    if unsure_phrase and key_pages:
        try:
            retry_subset = key_pages[:1]
            if grounding and hasattr(grounding, 'UrlContext'):
                url_ctx = grounding.UrlContext(urls=retry_subset)  # type: ignore
                retry = genai.GenerativeModel(MODEL_NAME).generate_content([
                    _build_prompt(question, retry_subset), url_ctx
                ])
            else:
                aug = _build_prompt(question, retry_subset) + "\n" + "\n".join(retry_subset)
                retry = genai.GenerativeModel(MODEL_NAME).generate_content(aug)
            new_text = (getattr(retry, 'text', '') or '').strip()
            if new_text and 'not fully sure' not in new_text.lower():
                answer_text = new_text
                raw_meta['retry_used'] = True
        except Exception:
            pass
    # Derive sources from answer by checking which candidate URLs appear; fallback to grounding subset
    cited: List[str] = []
    for u in candidate_urls:
        if u in answer_text:
            cited.append(u)
    if not cited:
        cited = list(candidate_urls[:5])
    return {"answer": answer_text, "sources": cited, "raw": raw_meta}


__all__ = ["answer_with_rag", "_ensure_sitemaps"]

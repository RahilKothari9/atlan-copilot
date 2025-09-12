"""RAG pipeline (index + retrieval + answer) for Atlan docs.

Design:
  - On first import, builds/loads a local lightweight index of Atlan documentation.
  - Fetches sitemaps (docs + developer), collects up to RAG_MAX_DOCS pages.
  - Downloads HTML, strips tags, truncates to RAG_DOC_CHAR_LIMIT characters per doc.
  - Embeds each doc with Gemini embedding model (text-embedding-004) and stores vectors in memory.
  - Persists index to JSON (rag_index.json) to avoid full rebuild on each restart (24h TTL).
  - Retrieval: cosine similarity over embeddings; select top-k (RAG_TOP_K, default 8).
  - Answer generation: Provide concatenated context snippets (trim per doc to keep within prompt budget).
  - If answer fails or no docs, returns empty answer triggering routing fallback upstream.

Environment Variables:
  RAG_MAX_DOCS (default 300)
  RAG_DOC_CHAR_LIMIT (default 6000)
  RAG_TOP_K (default 8)
  RAG_INDEX_PATH (default ./rag_index.json)
  RAG_INDEX_TTL_SECONDS (default 86400)
  GEMINI_EMBED_MODEL (default text-embedding-004)
  GEMINI_MODEL (reuse for answering if set)

Note: This is a simplified embedding store; for production consider a vector DB (Faiss, PGVector, etc.)
"""
from __future__ import annotations

import os
import json
import time
import math
import re
import threading
from dataclasses import dataclass
from typing import List, Dict, Any

import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

SITEMAPS = [
    "https://docs.atlan.com/sitemap.xml",
    "https://developer.atlan.com/sitemap.xml",
]
ALLOW_HOSTS = {"docs.atlan.com", "developer.atlan.com"}
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")
GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
RAG_MAX_DOCS = int(os.getenv("RAG_MAX_DOCS", "30000"))
RAG_DOC_CHAR_LIMIT = int(os.getenv("RAG_DOC_CHAR_LIMIT", "6000"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "./rag_index.json")
RAG_INDEX_TTL = int(os.getenv("RAG_INDEX_TTL_SECONDS", "864000000"))

# Thread lock for index operations
_INDEX_LOCK = threading.Lock()

@dataclass
class DocEntry:
    url: str
    text: str
    embedding: List[float]

class RAGIndex:
    def __init__(self):
        self.docs: List[DocEntry] = []
        self.built_at: float = 0.0

    def is_stale(self) -> bool:
        return (not self.docs) or (time.time() - self.built_at > RAG_INDEX_TTL)

    def to_disk(self):
        try:
            data = {
                "built_at": self.built_at,
                "docs": [
                    {"url": d.url, "text": d.text, "embedding": d.embedding}
                    for d in self.docs
                ],
            }
            with open(RAG_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception:
            pass

    def load_disk(self):
        try:
            if not os.path.exists(RAG_INDEX_PATH):
                return False
            with open(RAG_INDEX_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.built_at = data.get("built_at", 0)
            if time.time() - self.built_at > RAG_INDEX_TTL:
                return False
            docs = []
            for d in data.get("docs", [])[:RAG_MAX_DOCS]:
                docs.append(DocEntry(url=d["url"], text=d["text"], embedding=d["embedding"]))
            self.docs = docs
            return True
        except Exception:
            return False

RAG_INDEX = RAGIndex()

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(html: str) -> str:
    # Remove script/style blocks
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    text = _HTML_TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _fetch_sitemap(url: str) -> List[str]:
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        urls = re.findall(r"<loc>(.*?)</loc>", resp.text)
        out: List[str] = []
        for u in urls:
            u = u.strip()
            if not u:
                continue
            try:
                host = u.split("//", 1)[1].split("/", 1)[0]
            except Exception:
                continue
            if host in ALLOW_HOSTS:
                out.append(u)
        return out
    except Exception:
        return []


def _fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, timeout=25)
        if resp.status_code != 200:
            return ""
        txt = _strip_html(resp.text)
        if len(txt) > RAG_DOC_CHAR_LIMIT:
            txt = txt[:RAG_DOC_CHAR_LIMIT]
        return txt
    except Exception:
        return ""


def _embed(text: str) -> List[float]:
    try:
        result = genai.embed_content(model=EMBED_MODEL, content=text)
        emb = result.get("embedding", {}) if isinstance(result, dict) else getattr(result, "embedding", None)
        if isinstance(emb, dict):
            emb = emb.get("values") or emb.get("value")
        if not isinstance(emb, list):
            return []
        return [float(x) for x in emb]
    except Exception:
        return []


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = a[i]
        y = b[i]
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return -1.0
    return dot / math.sqrt(na * nb)


def build_index(force: bool = False):
    with _INDEX_LOCK:
        if not force and RAG_INDEX.load_disk():
            return
        # Fresh build
        urls: List[str] = []
        for sm in SITEMAPS:
            urls.extend(_fetch_sitemap(sm))
        # Deduplicate
        seen = set()
        ordered: List[str] = []
        for u in urls:
            if u not in seen:
                ordered.append(u)
                seen.add(u)
        ordered = ordered[: RAG_MAX_DOCS]
        docs: List[DocEntry] = []
        for u in ordered:
            page_text = _fetch_page(u)
            if not page_text:
                continue
            emb = _embed(page_text[:2000])  # shorter for embedding speed
            if not emb:
                continue
            docs.append(DocEntry(url=u, text=page_text, embedding=emb))
        RAG_INDEX.docs = docs
        RAG_INDEX.built_at = time.time()
        RAG_INDEX.to_disk()


def answer_question(question: str) -> Dict[str, Any]:
    """Return grounded answer and sources or empty answer on failure."""
    if not GEMINI_API_KEY:
        return {"answer": "", "sources": [], "error": "missing_api_key"}
    if RAG_INDEX.is_stale():
        # Trigger (re)build synchronously; could offload to thread if needed.
        build_index()
    if not RAG_INDEX.docs:
        return {"answer": "", "sources": [], "error": "empty_index"}
    q_emb = _embed(question[:2000])
    if not q_emb:
        return {"answer": "", "sources": [], "error": "embed_failed"}
    scored = []
    for d in RAG_INDEX.docs:
        sim = _cosine(q_emb, d.embedding)
        scored.append((sim, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [d for _, d in scored[:RAG_TOP_K] if _ > 0]
    if not top:
        return {"answer": "", "sources": [], "error": "no_similar_docs"}
    # Build context
    per_doc_limit = 1200  # chars per doc in prompt
    contexts = []
    src_urls = []
    for d in top:
        snippet = d.text[:per_doc_limit]
        contexts.append(f"URL: {d.url}\nCONTENT:\n{snippet}\n---")
        src_urls.append(d.url)
    prompt = (
        "You are Atlan's support assistant. Use ONLY the provided documentation excerpts. "
        "If the question is procedural, give concise numbered steps. Provide factual, current info. "
        "After answering add a blank line then 'Sources:' each on its own line. If insufficient context, say you cannot fully answer and still cite sources.\n" 
        f"Question: {question}\n\nContext Documents:\n{chr(10).join(contexts)}\nAnswer:\n"
    )
    try:
        model = genai.GenerativeModel(GEN_MODEL)
        resp = model.generate_content(prompt)
        ans = (getattr(resp, 'text', '') or '').strip()
    except Exception as e:
        return {"answer": "", "sources": src_urls, "error": f"gen_error:{e}"}
    if ans and "Sources:" not in ans:
        ans += "\n\nSources:\n" + "\n".join(src_urls)
    return {"answer": ans, "sources": src_urls, "error": None}

# Build index asynchronously at import to minimize cold-start latency for first ticket.
_thread = threading.Thread(target=build_index, daemon=True)
_thread.start()

__all__ = ["answer_question", "build_index"]
def index_status() -> Dict[str, Any]:
    return {
        "doc_count": len(RAG_INDEX.docs),
        "built_at": RAG_INDEX.built_at,
        "age_seconds": time.time() - RAG_INDEX.built_at if RAG_INDEX.built_at else None,
        "stale": RAG_INDEX.is_stale(),
        "max_docs": RAG_MAX_DOCS,
        "index_path": os.path.abspath(RAG_INDEX_PATH),
        "ttl_seconds": RAG_INDEX_TTL,
    }

__all__.append("index_status")

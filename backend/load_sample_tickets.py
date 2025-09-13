"""Seed sample tickets by calling the backend /create_ticket endpoint (preferred) or directly writing to Firestore.

Why API mode?
    - Ensures the normal creation flow runs (background classification + one‑shot RAG at creation time).
    - Keeps all server logic / future validation in one place.

Default Behaviour (no flags):
    Uses HTTP API mode against http://localhost:8000 (override with --api-base or API_BASE env).

Firestore Direct Mode (legacy):
    Use --direct to write documents straight to Firestore (bypasses classification + RAG unless you also pass --classify).

Usage (PowerShell examples):
    cd backend
    python load_sample_tickets.py                               # API mode -> POST /create_ticket for each sample
    python load_sample_tickets.py --api-base http://127.0.0.1:8000
    python load_sample_tickets.py --skip-duplicates             # query by subject first (API mode)
    python load_sample_tickets.py --direct --replace            # legacy direct load (clears collection first)
    python load_sample_tickets.py --direct --classify           # classify locally (needs GEMINI_API_KEY)

Environment:
    API_BASE (optional) overrides default http://localhost:8000
    FIREBASE_CREDENTIALS (only needed for --direct path)
    GEMINI_API_KEY (only needed when --classify in direct mode)

Sample File Schema (array): objects may include subject, body, conversation[ {sender, message} ], classification, status.
Missing fields are defaulted.

Duplicate Handling:
    API mode: optional --skip-duplicates calls /tickets/query to avoid inserting tickets whose subject already exists.
    Direct mode: existing doc id is skipped unless --overwrite given.
"""
from __future__ import annotations

import os
import time
import json
import argparse
from typing import Any, Dict, List, Optional

import requests

import firebase_admin
from firebase_admin import credentials, firestore

# Optional classification import (lazy)
CLASSIFIER_AVAILABLE = False
try:
    from main import classify_ticket  # reuse existing logic
    CLASSIFIER_AVAILABLE = True
except Exception:
    pass

DEFAULT_CLASSIFICATION = {"topic_tags": [], "sentiment": "Unknown", "priority": "P2 (Low)"}

def load_service_account(path_env: str | None = None) -> None:
    path = path_env or os.getenv("FIREBASE_CREDENTIALS") or "./serviceAccountKey.json"
    if not os.path.exists(path):
        # fallback repo file name used previously
        alt = "./atlan-cop-firebase-adminsdk-fbsvc-36156f99d8.json"
        if os.path.exists(alt):
            path = alt
        else:
            raise SystemExit(f"Service account file not found at {path} (set FIREBASE_CREDENTIALS env)")
    if not firebase_admin._apps:  # type: ignore
        cred = credentials.Certificate(path)
        firebase_admin.initialize_app(cred)


def generate_id(prefix: str, index: int) -> str:
    return f"{prefix}TICKET-{int(time.time()*1000)}-{index}" if prefix else f"TICKET-{int(time.time()*1000)}-{index}"


def normalize_ticket(raw: Dict[str, Any], idx: int, prefix: str) -> Dict[str, Any]:
    subject = raw.get("subject") or raw.get("title") or f"Sample Ticket {idx+1}"
    body = raw.get("body") or raw.get("content") or "(no body provided)"
    now = time.time()
    tid = raw.get("id") or generate_id(prefix, idx)
    classification = raw.get("classification") or DEFAULT_CLASSIFICATION.copy()
    # Ensure required keys in classification
    classification.setdefault("topic_tags", [])
    classification.setdefault("sentiment", "Unknown")
    classification.setdefault("priority", "P2 (Low)")
    conversation = raw.get("conversation") or []
    return {
        "id": tid,
        "subject": subject,
        "body": body,
        "status": raw.get("status", "Open"),
        "createdAt": raw.get("createdAt", now),
        "updatedAt": raw.get("updatedAt", now),
        "classification": classification,
        "conversation": conversation,
    }


def maybe_classify(doc: Dict[str, Any], do_classify: bool) -> None:
    if not do_classify:
        return
    if not CLASSIFIER_AVAILABLE:
        print("[warn] classify_ticket not available; skipping classification.")
        return
    try:
        result = classify_ticket(doc["subject"][:80], doc["body"])
        # Merge preserving any existing keys
        doc["classification"] = {
            "topic_tags": result.get("topic_tags", [])[:5],
            "sentiment": result.get("sentiment", "Neutral"),
            "priority": result.get("priority", "P2 (Low)"),
        }
    except Exception as e:
        print(f"[warn] classification failed for {doc['id']}: {e}")


def api_create_ticket(base: str, subject: str, body: str, initial_message: Optional[str]) -> Optional[Dict[str, Any]]:
    url = base.rstrip("/") + "/create_ticket"
    payload = {"subject": subject, "body": body, "initial_message": initial_message}
    try:
        r = requests.post(url, json=payload, timeout=30)
        if r.status_code >= 400:
            print(f"[error] {r.status_code} creating ticket: {subject[:60]}")
            return None
        return r.json()
    except Exception as e:
        print(f"[error] request failed: {e}")
        return None


def api_ticket_exists(base: str, subject: str) -> bool:
    """Check for existing ticket with same subject using /tickets/query (simple filter)."""
    url = base.rstrip("/") + "/tickets/query"
    q = {"query": subject, "filters": {}}  # backend performs text match over subject/body
    try:
        r = requests.post(url, json=q, timeout=20)
        if r.status_code >= 400:
            return False
        data = r.json() or []
        # naive exact subject match to decide duplicate
        return any((t.get("subject") == subject) for t in data)
    except Exception:
        return False


def direct_mode_load(data: List[Dict[str, Any]], args) -> None:
    load_service_account(os.getenv("FIREBASE_CREDENTIALS"))
    db = firestore.client()
    col_ref = db.collection(args.collection)
    if args.replace:
        print(f"[info] Deleting existing documents in collection {args.collection} ...")
        snaps = list(col_ref.stream())
        batch = db.batch()
        count = 0
        for s in snaps:
            batch.delete(s.reference)
            count += 1
            if count % 400 == 0:
                batch.commit(); batch = db.batch()
        batch.commit()
        print(f"[info] Deleted {count} existing docs.")
    written = 0; skipped = 0
    for idx, raw in enumerate(data):
        doc = normalize_ticket(raw, idx, args.prefix)
        maybe_classify(doc, args.classify)
        doc_ref = col_ref.document(doc["id"])
        existing = doc_ref.get()
        if existing.exists and not args.overwrite:
            skipped += 1; continue
        doc_ref.set(doc)
        written += 1
        if written % 25 == 0:
            print(f"[progress] {written} tickets written (direct mode)...")
    print(f"[done] Direct mode complete. Written: {written} | Skipped (existing): {skipped}")
    if args.classify and not CLASSIFIER_AVAILABLE:
        print("[note] Classification flag was set but classify_ticket function not imported.")


def api_mode_load(data: List[Dict[str, Any]], args) -> None:
    base = args.api_base or os.getenv("API_BASE", "http://localhost:8000")
    print(f"[info] API base: {base}")
    created = 0; skipped = 0; errors = 0
    for idx, raw in enumerate(data):
        doc = normalize_ticket(raw, idx, args.prefix)
        subject, body = doc["subject"], doc["body"]
        initial_message = None
        if doc.get("conversation"):
            # use first user message as initial message if present
            for m in doc["conversation"]:
                if m.get("sender") == "user" and m.get("message"):
                    initial_message = m["message"]
                    break
        if args.skip_duplicates and api_ticket_exists(base, subject):
            skipped += 1; continue
        resp = api_create_ticket(base, subject, body, initial_message)
        if resp is None:
            errors += 1
        else:
            created += 1
        if (idx + 1) % 25 == 0:
            print(f"[progress] processed {idx+1} items | created={created} skipped={skipped} errors={errors}")
        # For long delays give per-ticket feedback so the user sees activity
        if args.delay >= 1:
            status = "created" if resp else "error"
            print(f"[ticket] {idx+1}/{len(data)} {status} | totals: created={created} skipped={skipped} errors={errors} :: {subject[:60]}")
        # light throttle to avoid hammering background classifier if large dataset
        time.sleep(args.delay)
    print(f"[done] API mode complete. Created={created} Skipped={skipped} Errors={errors}")
    if args.classify:
        print("[note] --classify ignored in API mode (classification handled by backend background task).")


def main():
    parser = argparse.ArgumentParser(description="Seed sample tickets via API or direct Firestore")
    parser.add_argument("--file", default="sampleTickets.json", help="Path to sample tickets JSON array")
    # API mode flags
    parser.add_argument("--api-base", dest="api_base", default=None, help="Base URL of backend (default: http://localhost:8000)")
    parser.add_argument("--skip-duplicates", action="store_true", help="Skip if a ticket with identical subject already exists (API mode)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay seconds between API calls to reduce burst load")
    # Direct mode flags
    parser.add_argument("--direct", action="store_true", help="Use direct Firestore writes instead of API mode")
    parser.add_argument("--collection", default="tickets", help="Firestore collection (direct mode)")
    parser.add_argument("--classify", action="store_true", help="Classify tickets locally (direct mode only)")
    parser.add_argument("--replace", action="store_true", help="Delete existing docs first (direct mode)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite documents with same id (direct mode)")
    parser.add_argument("--prefix", default="", help="Optional id prefix (direct mode; API mode ignores custom ids)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise SystemExit(f"File not found: {args.file}")
    with open(args.file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Expected JSON array of ticket objects")

    if args.direct:
        direct_mode_load(data, args)
    else:
        api_mode_load(data, args)

if __name__ == "__main__":
    main()

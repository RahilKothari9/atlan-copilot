"""FastAPI backend for Atlan AI Helpdesk.

Implements:
  - POST /create_ticket : Create a new support ticket (with async classification)
  - POST /agent_query   : Stateful AI agent with Gemini tool-calling

Firestore Schema (collection: tickets):
  id, subject, body, createdAt, updatedAt, status,
  classification { topic_tags, sentiment, priority },
  conversation [ { sender, message, timestamp } ]

Environment Variables (use .env):
  GEMINI_API_KEY=...
  FIREBASE_CREDENTIALS=./atlan-cop-firebase-adminsdk-fbsvc-36156f99d8.json (or serviceAccountKey.json)

Notes:
  - answer_with_rag is a placeholder; integrate a vector store + grounded generation later.
  - Ticket ID generation is time-based for uniqueness (TICKET-<epoch_ms>). Adjust if strict sequencing needed.
"""

from __future__ import annotations

import os
import time
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Firestore / Firebase Admin
import firebase_admin
from firebase_admin import credentials, firestore

# Gemini
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError
try:
	from google.generativeai.types import Tool, FunctionDeclaration  # type: ignore
except Exception:  # Fallback if types module path changes
	Tool = None  # type: ignore
	FunctionDeclaration = None  # type: ignore


load_dotenv()  # Load .env if present

# ----------------------------------------------------------------------------
# Configuration & Initialization
# ----------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_ENABLED = True
if not GEMINI_API_KEY:
	# Defer hard failure; allow server to run so ticket CRUD works.
	AI_ENABLED = False
else:
	genai.configure(api_key=GEMINI_API_KEY)

FIREBASE_CREDENTIALS = os.getenv(
	"FIREBASE_CREDENTIALS", "./serviceAccountKey.json"
)
if not os.path.exists(FIREBASE_CREDENTIALS):
	# Fallback to the provided filename in repo if default missing
	alt = "./atlan-cop-firebase-adminsdk-fbsvc-36156f99d8.json"
	if os.path.exists(alt):
		FIREBASE_CREDENTIALS = alt
	else:
		raise RuntimeError(
			"Firebase service account key not found. Set FIREBASE_CREDENTIALS env or add serviceAccountKey.json"
		)

if not firebase_admin._apps:  # type: ignore
	cred = credentials.Certificate(FIREBASE_CREDENTIALS)
	firebase_admin.initialize_app(cred)

db = firestore.client()  # Firestore client


# ----------------------------------------------------------------------------
# Pydantic Models
# ----------------------------------------------------------------------------

class ConversationMessage(BaseModel):
	sender: str  # 'user' | 'agent'
	message: str
	timestamp: Optional[float] = Field(
		default=None, description="Unix epoch seconds; server fills if missing"
	)


class TicketCreateRequest(BaseModel):
	subject: str
	body: str
	initial_message: Optional[str] = Field(
		default=None, description="Optional first user message for conversation"
	)


class TicketResponse(BaseModel):
	id: str
	subject: str
	body: str
	status: str
	createdAt: float
	updatedAt: float
	classification: Dict[str, Any]
	conversation: List[ConversationMessage]


class AgentQueryRequest(BaseModel):
	user_message: str
	ticket_id: Optional[str] = None
	conversation: Optional[List[ConversationMessage]] = Field(
		default=None, description="Full conversation history if not using ticket"
	)


class AgentQueryResponse(BaseModel):
	reply: str
	tool_calls: Optional[List[Dict[str, Any]]] = None
	ticket_id: Optional[str] = None
	classification: Optional[Dict[str, Any]] = None
	used_urls: Optional[List[str]] = None
	routed: Optional[bool] = None


class TicketFilter(BaseModel):
	status: Optional[str] = None
	priority: Optional[str] = None
	sentiment: Optional[str] = None


# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------

TICKETS_COLLECTION = "tickets"


def generate_ticket_id() -> str:
	# Time-based uniqueness (ms). Replace with Firestore counter if strict ordering needed.
	return f"TICKET-{int(time.time() * 1000)}"


def now_ts() -> float:
	return time.time()


def firestore_ticket_to_model(doc: firestore.DocumentSnapshot) -> TicketResponse:
	data = doc.to_dict() or {}
	return TicketResponse(
		id=data.get("id", doc.id),
		subject=data.get("subject", ""),
		body=data.get("body", ""),
		status=data.get("status", "Open"),
		createdAt=data.get("createdAt", now_ts()),
		updatedAt=data.get("updatedAt", data.get("createdAt", now_ts())),
		classification=data.get(
			"classification",
			{"topic_tags": [], "sentiment": "Unknown", "priority": "P2 (Low)"},
		),
		conversation=[
			ConversationMessage(**m) for m in data.get("conversation", [])
		],
	)


# ----------------------------------------------------------------------------
from tools import list_doc_urls, answer_with_urls  # New dynamic doc tools
try:  # Firestore FieldFilter (new style). Fallback to None if unavailable.
	from google.cloud.firestore_v1 import FieldFilter  # type: ignore
except Exception:  # pragma: no cover
	FieldFilter = None  # type: ignore


def fetch_tickets(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
	"""Fetch tickets with optional filters using explicit FieldFilter objects.

	This removes the positional argument warning emitted by the Firestore SDK.
	Supported filters: status, priority, sentiment.
	"""
	query = db.collection(TICKETS_COLLECTION)
	if filters:
		status = filters.get("status")
		priority = filters.get("priority")
		sentiment = filters.get("sentiment")
		# Use new FieldFilter if available; otherwise fall back to legacy positional form.
		if FieldFilter:
			if status:
				query = query.where(filter=FieldFilter("status", "==", status))
			if priority:
				query = query.where(filter=FieldFilter("classification.priority", "==", priority))
			if sentiment:
				query = query.where(filter=FieldFilter("classification.sentiment", "==", sentiment))
		else:  # Fallback (may emit warning)
			if status:
				query = query.where("status", "==", status)
			if priority:
				query = query.where("classification.priority", "==", priority)
			if sentiment:
				query = query.where("classification.sentiment", "==", sentiment)
	try:
		# Order newest first
		query = query.order_by("createdAt", direction=firestore.Query.DESCENDING)  # type: ignore
	except Exception:
		pass  # Fallback silently if ordering not supported
	docs = query.limit(50).stream()
	return [firestore_ticket_to_model(d).dict() for d in docs]


def update_ticket(ticket_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
	ref = db.collection(TICKETS_COLLECTION).document(ticket_id)
	snap = ref.get()
	if not snap.exists:
		raise ValueError("Ticket not found")
	updates["updatedAt"] = now_ts()
	ref.update(updates)
	return firestore_ticket_to_model(ref.get()).dict()


def delete_ticket(ticket_id: str) -> Dict[str, Any]:
	ref = db.collection(TICKETS_COLLECTION).document(ticket_id)
	snap = ref.get()
	if not snap.exists:
		raise ValueError("Ticket not found")
	ref.delete()
	return {"deleted": True, "ticket_id": ticket_id}


def add_message_to_ticket(ticket_id: str, sender: str, message: str) -> Dict[str, Any]:
	if sender not in {"user", "agent"}:
		raise ValueError("sender must be 'user' or 'agent'")
	ref = db.collection(TICKETS_COLLECTION).document(ticket_id)
	snap = ref.get()
	if not snap.exists:
		raise ValueError("Ticket not found")
	msg = {"sender": sender, "message": message, "timestamp": now_ts()}
	ref.update({"conversation": firestore.ArrayUnion([msg]), "updatedAt": now_ts()})
	return msg


TOOL_REGISTRY = {
	"list_doc_urls": list_doc_urls,
	"answer_with_urls": answer_with_urls,
	"fetch_tickets": fetch_tickets,
	"update_ticket": update_ticket,
	"delete_ticket": delete_ticket,
	"add_message_to_ticket": add_message_to_ticket,
}

# Build Gemini tool/function declarations (schema minimal but explicit)
def _build_tool_declarations():
	if not FunctionDeclaration or not Tool:
		return None
	fds = [
		FunctionDeclaration(
			name="list_doc_urls",
			description=(
				"List slices of Atlan documentation URLs (product + developer). Use filter substring to narrow. "
				"Call this BEFORE answer_with_urls to discover relevant URLs. Supports simple pagination via offset+limit."
			),
			parameters={
				"type": "object",
				"properties": {
					"filter": {"type": "string", "description": "Optional substring filter (case-insensitive)"},
					"limit": {"type": "integer", "description": "Max URLs to return (<=100)"},
					"offset": {"type": "integer", "description": "Pagination offset"},
				},
			},
		),
		FunctionDeclaration(
			name="answer_with_urls",
			description=(
				"Generate grounded answer using ONLY the provided URLs (docs.atlan.com / developer.atlan.com). "
				"You MUST first narrow candidates with list_doc_urls, then pass at most 10 high-signal URLs here."
			),
			parameters={
				"type": "object",
				"properties": {
					"question": {"type": "string"},
					"urls": {"type": "array", "items": {"type": "string"}, "description": "Up to 10 selected URLs"},
				},
				"required": ["question", "urls"],
			},
		),
		FunctionDeclaration(
			name="fetch_tickets",
			description="Fetch up to 50 tickets optionally filtered by status, priority, sentiment.",
			parameters={
				"type": "object",
				"properties": {
					"filters": {
						"type": "object",
						"properties": {
							"status": {"type": "string"},
							"priority": {"type": "string"},
							"sentiment": {"type": "string"},
						},
					}
				},
			},
		),
		FunctionDeclaration(
			name="update_ticket",
			description="Update ticket fields including classification or status.",
			parameters={
				"type": "object",
				"properties": {
					"ticket_id": {"type": "string"},
					"updates": {"type": "object", "description": "Partial update fields"},
				},
				"required": ["ticket_id", "updates"],
			},
		),
		FunctionDeclaration(
			name="delete_ticket",
			description="Delete a ticket permanently.",
			parameters={
				"type": "object",
				"properties": {"ticket_id": {"type": "string"}},
				"required": ["ticket_id"],
			},
		),
		FunctionDeclaration(
			name="add_message_to_ticket",
			description="Append a message to a ticket conversation.",
			parameters={
				"type": "object",
				"properties": {
					"ticket_id": {"type": "string"},
					"sender": {"type": "string", "enum": ["user", "agent"]},
					"message": {"type": "string"},
				},
				"required": ["ticket_id", "sender", "message"],
			},
		),
	]
	return Tool(function_declarations=fds)

GEMINI_TOOL_OBJECT = _build_tool_declarations()


# ----------------------------------------------------------------------------
# Gemini Agent Helper
# ----------------------------------------------------------------------------

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def classify_ticket(subject: str, body: str) -> Dict[str, Any]:
	"""Call Gemini to classify ticket. Returns classification dict."""
	if not AI_ENABLED:
		return {"topic_tags": [], "sentiment": "Neutral", "priority": "P2 (Low)"}
	# Liberal RAG-oriented classification guidance: ALWAYS include a RAG-eligible topic tag
	# (how-to, product, best practices, api/sdk, sso) FIRST when the query seeks steps, setup,
	# configuration, integration, authentication, API usage, optimization, troubleshooting, or workflow guidance.
	# If connector/vendor specific (e.g. snowflake, salesforce) AND the user asks how/steps/configure/connect/setup,
	# include both 'how-to' (first) and the vendor or 'connector' tag.
	# Use at most 3 tags. Approved special / RAG tags: how-to, product, best practices, api/sdk, sso.
	# Other allowed contextual tags (use only if clearly central): connector, lineage, glossary, governance, security,
	# sensitive data, cost, performance, catalog, data quality, access control.
	# Examples:
	#  - "How do I connect to Snowflake" -> ["how-to", "snowflake", "connector"]
	#  - "Set up SSO with Okta" -> ["sso", "how-to"]
	#  - "API token rotation best practice" -> ["api/sdk", "best practices"]
	# Always ensure at least ONE of the RAG tags is present when instructional or informational.
	prompt = f"""
You classify support inputs. Return STRICT JSON only.
Keys:
  topic_tags: array (1-3) lowercase tags (prioritize RAG tags: how-to, product, best practices, api/sdk, sso when instructional / configuration / API / auth / optimization / setup / troubleshooting).
  sentiment: one of Positive, Neutral, Negative.
  priority: one of 'P0 (High)', 'P1 (Medium)', 'P2 (Low)'.
Rules:
  - If user asks how/steps/configure/setup/connect/integrate/authenticate/login/enable/install/add/create/use/rotate/renew => include 'how-to' first (or 'api/sdk' if purely about API syntax; still include 'how-to' if procedural).
  - If mentions API, SDK, endpoint, token, key => include 'api/sdk'.
  - If mentions SSO, SAML, Okta, Azure AD, OneLogin => include 'sso'.
  - If "best practice" or optimization/tuning => include 'best practices'.
  - Add vendor/connector tag (e.g. snowflake, salesforce) or 'connector' second if relevant.
  - Never output duplicates; max 3 tags.
INPUT SUBJECT:\n{subject}\nINPUT BODY:\n{body}\nJSON:
"""
	try:
		model = genai.GenerativeModel(MODEL_NAME)
		resp = model.generate_content(prompt)
		text = resp.text or "{}"
		# Extract first JSON object
		start = text.find("{")
		end = text.rfind("}")
		if start != -1 and end != -1:
			json_str = text[start : end + 1]
			data = json.loads(json_str)
		else:
			data = {}
	except Exception:
		data = {}
	# Defaults if missing
	return {
		"topic_tags": data.get("topic_tags", []),
		"sentiment": data.get("sentiment", "Neutral"),
		"priority": data.get("priority", "P2 (Low)"),
	}


CANON_TOPICS = ["how-to","product","connector","lineage","api/sdk","sso","glossary","best practices","sensitive data","governance","catalog","security","performance","cost","data quality"]
RAG_TOPICS = {"how-to","product","best practices","api/sdk","sso"}

def normalize_topic_tags(raw: List[str]) -> List[str]:
	seen = set()
	out: List[str] = []
	for tag in raw:
		low = (tag or "").strip().lower()
		low = low.replace("best-practice","best practices")
		if low in {"api","sdk"}:
			low = "api/sdk"
		if low and low not in seen:
			seen.add(low)
			out.append(low)
	return out[:3]


def run_agent(user_message: str, conversation: List[ConversationMessage], system_instruction: str | None = None) -> Dict[str, Any]:
	"""Run Gemini agent with possible tool-calling loop.

	For now, implements a simple single-turn with optional tool calls if library returns them.
	"""
	if not AI_ENABLED:
		return {"reply": "AI disabled (missing GEMINI_API_KEY).", "tool_calls": []}
	# Use structured tool declarations instead of raw function objects
	tool_param = [GEMINI_TOOL_OBJECT] if GEMINI_TOOL_OBJECT else None
	model_kwargs = {"tools": tool_param} if tool_param else {}
	if system_instruction:
		model_kwargs["system_instruction"] = system_instruction
	model = genai.GenerativeModel(MODEL_NAME, **model_kwargs)

	# Convert conversation to Gemini format (list of dicts with role/content)
	history_parts = []
	for msg in conversation:
		role = "user" if msg.sender == "user" else "model"
		history_parts.append({"role": role, "parts": [msg.message]})

	chat = model.start_chat(history=history_parts)
	try:
		response = chat.send_message(user_message)
	except GoogleAPIError as e:
		return {"reply": f"Agent error: {e.message}", "tool_calls": []}
	except Exception as e:
		return {"reply": f"Agent unexpected error: {e}", "tool_calls": []}

	tool_calls_accum: List[Dict[str, Any]] = []

	def _sanitize(obj):
		"""Recursively convert object to JSON-serializable primitives."""
		from collections.abc import Mapping, Iterable
		primitive = (str, int, float, bool, type(None))
		if isinstance(obj, primitive):
			return obj
		# Google proto MapComposite -> treat like Mapping
		if hasattr(obj, "items") and not isinstance(obj, (list, tuple, set)):
			try:
				return {str(k): _sanitize(v) for k, v in obj.items()}
			except Exception:
				return str(obj)
		if isinstance(obj, Mapping):  # fallback mapping
			return {str(k): _sanitize(v) for k, v in obj.items()}
		if isinstance(obj, (list, tuple, set)):
			return [_sanitize(x) for x in obj]
		if hasattr(obj, "to_dict"):
			try:
				return _sanitize(obj.to_dict())
			except Exception:
				return str(obj)
		return str(obj)

	# Iteratively resolve tool calls (basic implementation)
	safety_counter = 0
	final_text: str = ""  # ensure defined for all exit paths
	while True:
		safety_counter += 1
		if safety_counter > 5:  # Prevent infinite loops
			break
		candidate = response.candidates[0] if response.candidates else None
		if not candidate:
			break
		# parts may include function_calls attribute
		function_calls = []
		for part in getattr(candidate.content, "parts", []):
			fc = getattr(part, "function_call", None)
			if fc:
				function_calls.append(fc)
		if not function_calls:
			# no more tool calls; attempt to read text then break
			try:
				final_text = response.text  # may fail if residual structured parts
			except Exception:
				parts_out: List[str] = []
				cand = response.candidates[0] if getattr(response, "candidates", None) else None
				if cand:
					for part in getattr(cand.content, "parts", []) or []:
						text_val = getattr(part, "text", None)
						if text_val:
							parts_out.append(text_val)
				final_text = "\n".join(parts_out) if parts_out else "(No response generated)"
			break
		# Execute each function call
		for fc in function_calls:
			name = getattr(fc, "name", None)
			args = {}
			try:
				raw_args = getattr(fc, "args", {}) or {}
				if hasattr(raw_args, "items"):
					args = {k: _sanitize(v) for k, v in raw_args.items()}
				else:
					args = dict(raw_args)
			except Exception:
				args = {}
			func = TOOL_REGISTRY.get(name)
			if not func:
				result = {"error": f"Unknown tool {name}"}
			else:
				try:
					result = func(**args)
				except Exception as e:  # Tool-level error
					result = {"error": str(e)}
			tool_calls_accum.append({"tool": name, "args": _sanitize(args), "result": _sanitize(result)})
			# Provide structured function_response back to model so it can continue reasoning
			tool_payload = {"function_response": {"name": name, "response": _sanitize(result)}}
			try:
				response = chat.send_message({"role": "tool", "parts": [tool_payload]})
			except Exception:
				break
	# If final_text still empty attempt one last extraction
	if not final_text:
		try:
			final_text = response.text  # type: ignore
		except Exception:
			parts_out: List[str] = []
			cand = response.candidates[0] if getattr(response, "candidates", None) else None
			if cand:
				for part in getattr(cand.content, "parts", []) or []:
					text_val = getattr(part, "text", None)
					if text_val:
						parts_out.append(text_val)
			final_text = "\n".join(parts_out) if parts_out else "(No response generated)"
	# Final sanitize just in case
	return {"reply": final_text, "tool_calls": _sanitize(tool_calls_accum)}


def build_system_instruction(ticket: Optional[TicketResponse]) -> str:
	"""Return a robust system prompt guiding the agentic loop.

	Emphasizes:
	- When to classify vs when to RAG.
	- Tool usage policy.
	- Output and logging requirements.
	"""
	core = (
		"You are Atlan's Customer Support AI Copilot. Workflow per user query:\n"
		"1. Ensure ticket classification is current (topic_tags 1-3, sentiment, priority). If missing/stale, call update_ticket. Be LIBERAL assigning 'how-to', 'product', 'api/sdk', 'sso', or 'best practices' when the user is asking for steps, configuration, integration, authentication, setup, usage guidance, API details, tokens, SSO/identity, or optimization. Prefer one of these RAG-eligible topics if doubt exists.\n"
		"2. For informational/how-to/product/API questions: discover documentation URLs via list_doc_urls (use filter terms: e.g. 'snowflake', 'lineage', 'api token'). You may paginate/refine.\n"
		"3. Select at most 10 highly relevant URLs and call answer_with_urls(question, urls=[...]) to generate a grounded answer.\n"
		"4. Cite ONLY the URLs you passed to answer_with_urls under a 'Sources:' section (plain list).\n"
		"5. If the query is clearly a routing topic (e.g., internal connector escalation) and docs won't help, respond with routing message instead of doc tools.\n"
		"6. Log conversation messages with add_message_to_ticket when a ticket is active (user + your reply).\n"
		"7. Do not fabricate URLs. Never cite URLs you did not pass to answer_with_urls.\n"
		"8. Keep answers concise, stepwise for setup questions, and professional.\n"
	)
	if ticket:
		core += (
			f"\nActive Ticket Context:\nID: {ticket.id}\nSubject: {ticket.subject}\nBody: {ticket.body[:800]}\n"
			f"Current Classification: {ticket.classification}\n"
			"If classification fields look empty, generic, or inconsistent with the body, recompute and persist.\n"
		)
	return core


# ----------------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------------

app = FastAPI(title="Atlan AI Helpdesk Backend", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],  # Adjust in production
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/health")
def health():
	return {"status": "ok", "model": MODEL_NAME, "ai_enabled": AI_ENABLED}


# --------------------------- Ticket Creation -------------------------------

@app.post("/create_ticket", response_model=TicketResponse)
def create_ticket(req: TicketCreateRequest, background: BackgroundTasks):
	ticket_id = save_ticket(req.subject, req.body, req.initial_message)
	background.add_task(
		classify_and_update, ticket_id, f"Subject: {req.subject}\nBody: {req.body}"
	)
	return firestore_ticket_to_model(
		db.collection(TICKETS_COLLECTION).document(ticket_id).get()
	)

def classify_and_update(ticket_doc_id: str, ticket_text: str):
	"""Background task: classify the ticket and persist results (Feature 3).

	ticket_text: Combined textual context (subject + body).
	Produces JSON classification object with keys: topic_tags, sentiment, priority.
	"""
	if not AI_ENABLED:
		return
	# System-style instruction
	prompt = f"""
You classify support tickets. Return STRICT JSON only.
Keys:
	topic_tags: array 1-5 lowercase tags. ALWAYS include a RAG tag (how-to, product, best practices, api/sdk, sso) FIRST when user asks for steps, setup, configuration, integration, authentication, API usage, optimization, troubleshooting, tokens, SSO, security setup.
	sentiment: Positive | Neutral | Negative.
	priority: 'P0 (High)' | 'P1 (Medium)' | 'P2 (Low)'.
Vendor how-to example: "How do I connect to Snowflake" => ["how-to","snowflake","connector"].
Add vendor or 'connector' tag second when specific data store/service is referenced.
Avoid duplicates; keep <=5.
TICKET TEXT:\n{ticket_text}\nJSON:
"""
	classification: Dict[str, Any] = {
		"topic_tags": [],
		"sentiment": "Neutral",
		"priority": "P2 (Low)",
	}
	try:
		model = genai.GenerativeModel(MODEL_NAME)
		resp = model.generate_content(prompt)
		raw = resp.text or "{}"
		start = raw.find("{")
		end = raw.rfind("}")
		if start != -1 and end != -1:
			payload = raw[start : end + 1]
			parsed = json.loads(payload)
			# Merge validated fields
			if isinstance(parsed.get("topic_tags"), list):
				classification["topic_tags"] = parsed["topic_tags"][:5]
			if parsed.get("sentiment") in {"Positive", "Neutral", "Negative"}:
				classification["sentiment"] = parsed["sentiment"]
			if parsed.get("priority") in {"P0 (High)", "P1 (Medium)", "P2 (Low)"}:
				classification["priority"] = parsed["priority"]
	except Exception:
		# Keep defaults on failure
		pass
	# Persist using update_ticket helper for consistency
	try:
		updated = update_ticket(ticket_doc_id, {"classification": classification})
	except Exception:
		updated = {"classification": classification}

	# After classification, trigger dynamic agent-driven RAG if eligible
	try:
		cls_tags = normalize_topic_tags((updated.get("classification") or {}).get("topic_tags", []))  # type: ignore
		ref_check = db.collection(TICKETS_COLLECTION).document(ticket_doc_id)
		snap_check = ref_check.get()
		conv_existing = []
		if snap_check.exists:
			conv_existing = (snap_check.to_dict() or {}).get("conversation", [])
		if any(t in RAG_TOPICS for t in cls_tags):
			# RAG flow
			_generate_rag_answer(ticket_doc_id)
		else:
			# Routing flow: add a single routing agent message if none exists yet
			if not any(m.get("sender") == "agent" for m in conv_existing):
				primary = cls_tags[0] if cls_tags else "connector"
				msg = f"This ticket has been classified as a '{primary}' issue and routed to the appropriate team."
				add_message_to_ticket(ticket_doc_id, "agent", msg)
	except Exception:
		pass


def _generate_rag_answer(ticket_doc_id: str):
	"""Invoke the agent tool loop so the MODEL chooses URLs (no manual static list) and generate answer.
	Skips if an agent message already exists.
	"""
	if not AI_ENABLED:
		return
	ref = db.collection(TICKETS_COLLECTION).document(ticket_doc_id)
	snap = ref.get()
	if not snap.exists:
		return
	data = snap.to_dict() or {}
	conversation = data.get("conversation", [])
	if any(m.get("sender") == "agent" for m in conversation):
		return
	question_text = data.get("body") or data.get("subject") or ""
	if not question_text.strip():
		return
	# Build synthetic conversation and system instruction
	ticket_model = firestore_ticket_to_model(snap)
	sys = build_system_instruction(ticket_model) + "\nAUTO-ANSWER MODE: You MUST call list_doc_urls (one or more times) then answer_with_urls for this ticket question. Do not answer without a tool-grounded citation list."
	conv_objs = [ConversationMessage(sender="user", message=question_text, timestamp=now_ts())]
	result = run_agent(question_text, conv_objs, system_instruction=sys)
	reply = result.get("reply", "")
	tool_calls = result.get("tool_calls", []) or []
	# If reply lacks Sources but tool_calls include answer_with_urls, patch in sources
	if "Sources:" not in reply:
		for call in tool_calls:
			if call.get("tool") == "answer_with_urls":
				res = call.get("result") or {}
				urls = res.get("response", {}).get("used_urls") if isinstance(res.get("response"), dict) else res.get("used_urls")
				if not urls:
					urls = res.get("sources")
				if urls:
					reply += "\n\nSources:\n" + "\n".join(urls[:10])
				break
	if reply:
		add_message_to_ticket(ticket_doc_id, "agent", reply)


def save_ticket(subject: str, body: str, initial_message: Optional[str]) -> str:
	"""Persist a new ticket to Firestore and return its ID."""
	ticket_id = generate_ticket_id()
	ts = now_ts()
	conversation = []
	if initial_message:
		conversation.append(
			{
				"sender": "user",
				"message": initial_message,
				"timestamp": ts,
			}
		)
	data = {
		"id": ticket_id,
		"subject": subject,
		"body": body,
		"status": "Open",
		"createdAt": ts,
		"updatedAt": ts,
		"classification": {
			"topic_tags": [],
			"sentiment": "Unknown",
			"priority": "P2 (Low)",
		},
		"conversation": conversation,
	}
	db.collection(TICKETS_COLLECTION).document(ticket_id).set(data)
	return ticket_id


# --------------------------- Agent Query -----------------------------------

@app.post("/agent_query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest):
	# -----------------------------
	# Automatic gating + RAG logic
	# -----------------------------
	ticket_id = req.ticket_id
	classification: Dict[str, Any] = {}
	primary_topic = ""
	# 1. Load ticket & ensure classification if ticket id provided
	if ticket_id:
		snap = db.collection(TICKETS_COLLECTION).document(ticket_id).get()
		if not snap.exists:
			raise HTTPException(status_code=404, detail="Ticket not found")
		t = firestore_ticket_to_model(snap)
		classification = t.classification or {}
		if AI_ENABLED and not classification.get("topic_tags"):
			try:
				classification = classify_ticket(t.subject, t.body)
				update_ticket(ticket_id, {"classification": classification})
			except Exception:
				pass
	else:
		# Ad-hoc question without ticket: classify the question text itself (subject surrogate).
		if AI_ENABLED:
			classification = classify_ticket(req.user_message[:60], req.user_message)
		else:
			classification = {"topic_tags": [], "sentiment": "Neutral", "priority": "P2 (Low)"}

	# 2. Normalize & pick primary topic
	raw_tags = classification.get("topic_tags", []) or []
	normalized_tags = normalize_topic_tags(raw_tags)
	# Apply prioritization heuristic to bias toward RAG-eligible topics
	# No heuristic reordering now; rely solely on model liberal tagging per prompt.
	primary_topic = normalized_tags[0] if normalized_tags else ""

	# 3. Decide routing vs RAG purely from model-provided tags
	use_rag = any(t in RAG_TOPICS for t in normalized_tags)
	tool_calls: List[Dict[str, Any]] = []
	used_urls: List[str] = []
	reply = ""
	routed = False

	if use_rag and AI_ENABLED:
		# Lightweight automatic doc selection using list_doc_urls heuristics over user query keywords.
		from tools import list_doc_urls as _list, answer_with_urls as _anw
		# Extract up to 5 meaningful keywords >3 chars (very naive)
		import re
		words = [w.lower() for w in re.findall(r"[A-Za-z0-9_-]+", req.user_message) if len(w) > 3]
		seen_kw = []
		for w in words:
			if w in seen_kw:
				continue
			seen_kw.append(w)
			res = _list(filter=w, limit=30, offset=0)
			tool_calls.append({"tool": "list_doc_urls", "args": {"filter": w, "limit": 30}, "result": {"returned": res.get("returned"), "sample": res.get("urls", [])[:5]}})
			for u in res.get("urls", []):
				if u not in used_urls:
					used_urls.append(u)
					if len(used_urls) >= 10:
						break
			if len(used_urls) < 5:
				# Fallback broad search if not enough
				fallback = _list(filter="atlan", limit=20, offset=0)
				tool_calls.append({"tool": "list_doc_urls", "args": {"filter": "atlan", "limit": 20}, "result": {"returned": fallback.get("returned")}})
				for u in fallback.get("urls", []):
					if u not in used_urls:
						used_urls.append(u)
						if len(used_urls) >= 10:
							break
		answer_payload = _anw(question=req.user_message, urls=used_urls)
		tool_calls.append({"tool": "answer_with_urls", "args": {"question": req.user_message, "urls": used_urls}, "result": {"meta": answer_payload.get("meta", {}), "error": answer_payload.get("error")}})
		ans_text = answer_payload.get("answer", "")
		sources = answer_payload.get("sources", used_urls)
		if sources and "Sources:" not in ans_text:
			ans_text += "\n\nSources:\n" + "\n".join(sources)
		reply = ans_text or "(No answer generated)"
	else:
		routed = True
		pretty_topic = primary_topic.title() if primary_topic else "general"
		reply = f"This ticket has been classified as a '{pretty_topic}' issue and routed to the appropriate team."

	# 4. Persist conversation if we have a ticket id
	if ticket_id:
		try:
			add_message_to_ticket(ticket_id, "user", req.user_message)
			add_message_to_ticket(ticket_id, "agent", reply)
		except Exception:
			pass

	return AgentQueryResponse(
		reply=reply,
		tool_calls=tool_calls,
		ticket_id=ticket_id,
		classification=classification,
		used_urls=used_urls,
		routed=routed,
	)


# --------------------------- Tool Endpoints (Optional) ---------------------

@app.post("/tickets/query")
def tickets_query(filters: TicketFilter):
	try:
		return {"tickets": fetch_tickets(filters.dict(exclude_none=True))}
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.get("/tickets/{ticket_id}", response_model=TicketResponse)
def ticket_get(ticket_id: str):
	ref = db.collection(TICKETS_COLLECTION).document(ticket_id)
	snap = ref.get()
	if not snap.exists:
		raise HTTPException(status_code=404, detail="Ticket not found")
	return firestore_ticket_to_model(snap)


@app.post("/tickets/{ticket_id}/update")
def ticket_update(ticket_id: str, updates: Dict[str, Any]):
	try:
		return update_ticket(ticket_id, updates)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.delete("/tickets/{ticket_id}")
def ticket_delete(ticket_id: str):
	try:
		return delete_ticket(ticket_id)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


@app.post("/tickets/{ticket_id}/message")
def ticket_add_message(ticket_id: str, msg: ConversationMessage):
	try:
		return add_message_to_ticket(ticket_id, msg.sender, msg.message)
	except Exception as e:
		raise HTTPException(status_code=400, detail=str(e))


# ----------------------------------------------------------------------------
# Run (for local dev): uvicorn main:app --reload
# ----------------------------------------------------------------------------

if __name__ == "__main__":
	import uvicorn

	uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


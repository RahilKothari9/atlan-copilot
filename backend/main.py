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
from tools import answer_with_rag  # Feature 4 RAG implementation


def fetch_tickets(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
	query = db.collection(TICKETS_COLLECTION)
	if filters:
		if status := filters.get("status"):
			query = query.where("status", "==", status)
		if priority := filters.get("priority"):
			query = query.where("classification.priority", "==", priority)
		if sentiment := filters.get("sentiment"):
			query = query.where("classification.sentiment", "==", sentiment)
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
	"answer_with_rag": answer_with_rag,
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
			name="answer_with_rag",
			description=(
				"Use Retrieval-Augmented Generation over Atlan documentation (product) and developer hub (API/SDK). "
				"Call this ONLY after you have a classification and only for topics: How-to, Product, Best practices, API/SDK, SSO. "
				"Provide the original user question as 'question'. Returns an object containing an answer and a list of source URLs which MUST be cited in the final response."
			),
			parameters={
				"type": "object",
				"properties": {"question": {"type": "string", "description": "User question"}},
				"required": ["question"],
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
	prompt = f"""
You are a ticket classification assistant. Analyze the ticket and output STRICT JSON with keys: topic_tags (array of 1-3 short tags), sentiment (one of: Positive, Neutral, Negative), priority (one of: 'P0 (High)', 'P1 (Medium)', 'P2 (Low)').
Ticket Subject: {subject}\nTicket Body: {body}
JSON:
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


CANON_TOPICS = [
	"how-to","product","connector","lineage","api/sdk","sso","glossary","best practices","sensitive data"
]

RAG_TOPICS = {"how-to","product","best practices","api/sdk","sso"}

def normalize_topic_tags(raw: List[str]) -> List[str]:
	seen = set()
	normalized = []
	for tag in raw:
		low = (tag or "").strip().lower()
		low = low.replace("best-practice","best practices")
		if low in {"api","sdk"}:
			low = "api/sdk"
		if low and low not in seen:
			seen.add(low)
			normalized.append(low)
	return normalized[:3]


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
		# Continue loop to detect any further tool calls
		try:
			final_text = response.text  # may raise if function_call parts remain
		except Exception:
			# Manual assembly from textual parts only
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
		"You are Atlan's Customer Support AI Copilot. Your goals per user query:\n"
		"1. Ensure the ticket has an up-to-date classification: topic_tags (1-3 from: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data), sentiment (Frustrated, Curious, Angry, Neutral, Positive, Negative), priority (P0 (High), P1 (Medium), P2 (Low)).\n"
		"2. If classification missing or clearly stale, derive it and persist using update_ticket (with classification.* fields).\n"
		"3. If the (primary) topic is one of: How-to, Product, Best practices, API/SDK, SSO -> call answer_with_rag(question=original user question). Use its sources to craft the final answer and cite them under 'Sources:' as plain URLs.\n"
		"4. For other topics (Connector, Lineage, Glossary, Sensitive data, etc.) do NOT call answer_with_rag; instead respond with a routing message: \"This ticket has been classified as '<Topic>' and routed to the appropriate team.\"\n"
		"5. Always append your natural language answer only after tool usage is done.\n"
		"6. When a specific ticket is in scope you MUST log both the user's latest message and your final answer using add_message_to_ticket (unless already logged by the backend).\n"
		"7. Use fetch_tickets ONLY if you need broader context across tickets (rare).\n"
		"8. Never hallucinate sources; only cite URLs returned by answer_with_rag.\n"
		"9. Keep responses concise, actionable, and professional.\n"
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
You are an assistant that classifies customer support tickets.
Return ONLY valid JSON with keys: topic_tags (array of 1-5 concise lowercase tags), sentiment (one of: Positive, Neutral, Negative), priority (one of: 'P0 (High)','P1 (Medium)','P2 (Low)').
If unsure, make a best-effort guess. No extra text.
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
		update_ticket(ticket_doc_id, {"classification": classification})
	except Exception:
		pass


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
	conversation: List[ConversationMessage] = []
	ticket_id = req.ticket_id
	system_instruction = None
	if ticket_id:
		snap = db.collection(TICKETS_COLLECTION).document(ticket_id).get()
		if not snap.exists:
			raise HTTPException(status_code=404, detail="Ticket not found")
		ticket_data = firestore_ticket_to_model(snap)
		conversation = ticket_data.conversation
		# If classification is still default/empty and AI enabled, attempt on-demand classification
		try:
			if AI_ENABLED and (not ticket_data.classification.get("topic_tags")):
				new_cls = classify_ticket(ticket_data.subject, ticket_data.body)
				update_ticket(ticket_data.id, {"classification": new_cls})
				# refresh ticket_data
				refreshed = db.collection(TICKETS_COLLECTION).document(ticket_id).get()
				ticket_data = firestore_ticket_to_model(refreshed)
		except Exception:
			pass
		# Build dynamic system instruction with ticket metadata
		system_instruction = build_system_instruction(ticket_data)
		# Inject a synthetic context message so model sees full ticket without needing tool call
		cls = ticket_data.classification or {}
		context_message = ConversationMessage(
			sender="user",
			message=(
				f"[TICKET CONTEXT]\nID: {ticket_data.id}\nSubject: {ticket_data.subject}\n"
				f"Body: {ticket_data.body}\nClassification: {cls}"
			),
			timestamp=now_ts(),
		)
		conversation = [context_message] + conversation
	elif req.conversation:
		conversation = req.conversation
	else:
		conversation = []

	# Append user message to conversation context (not yet stored if ticket)
	conversation.append(
		ConversationMessage(sender="user", message=req.user_message, timestamp=now_ts())
	)

	# If no ticket, still provide a generic system instruction
	if not system_instruction:
		system_instruction = build_system_instruction(None)

	result = run_agent(req.user_message, conversation, system_instruction=system_instruction)
	reply = result.get("reply", "")
	tool_calls = result.get("tool_calls", [])

	# Fallback path if agent loop failed or produced empty reply
	if not reply or reply.startswith("Agent error") or reply.startswith("Agent unexpected error"):
		try:
			# Attempt direct classification + optional RAG
			if ticket_id:
				# Refresh latest ticket state
				fresh = db.collection(TICKETS_COLLECTION).document(ticket_id).get()
				if fresh.exists:
					fresh_ticket = firestore_ticket_to_model(fresh)
					classification = fresh_ticket.classification or {}
					if not classification.get("topic_tags"):
						classification = classify_ticket(fresh_ticket.subject, fresh_ticket.body)
						update_ticket(ticket_id, {"classification": classification})
					tags = normalize_topic_tags(classification.get("topic_tags", []))
					rag_topics = {"how-to","product","best","api/sdk","sso","api","sdk","best practices"}
					use_rag = any(tag in RAG_TOPICS for tag in tags)
					if use_rag:
						from tools import answer_with_rag as _rag
						rag_res = _rag(req.user_message)
						sources = rag_res.get("sources", [])
						reply = f"{rag_res.get('answer','(no answer)')}\n\nSources:\n" + "\n".join(sources)
					else:
						primary = tags[0] if tags else "connector"
						reply = f"This ticket has been classified as '{primary}' and routed to the appropriate team."
			else:
				# No ticket context: answer generically via RAG as best effort
				from tools import answer_with_rag as _rag2
				rag_res2 = _rag2(req.user_message)
				reply = rag_res2.get("answer","(no answer)") + "\n\nSources:\n" + "\n".join(rag_res2.get("sources", []))
		except Exception as _fb_err:  # pragma: no cover
			reply = reply or f"(Fallback agent failure: {_fb_err})"

	# Persist user + agent messages if tied to ticket
	if ticket_id:
		try:
			# Avoid duplicate user logging if already present as last entry
			current = db.collection(TICKETS_COLLECTION).document(ticket_id).get()
			if current.exists:
				cur_data = current.to_dict() or {}
				conv = cur_data.get("conversation", [])
				if not conv or conv[-1].get("message") != req.user_message:
					add_message_to_ticket(ticket_id, "user", req.user_message)
			# Always log agent reply if different from last
			current2 = db.collection(TICKETS_COLLECTION).document(ticket_id).get()
			if current2.exists:
				conv2 = (current2.to_dict() or {}).get("conversation", [])
				if not conv2 or conv2[-1].get("message") != reply:
					add_message_to_ticket(ticket_id, "agent", reply)
		except Exception:
			pass  # Don't fail request on DB persistence error

	return AgentQueryResponse(reply=reply, tool_calls=tool_calls, ticket_id=ticket_id)


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


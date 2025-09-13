// Lightweight API client for backend integration
// Uses native fetch; can be swapped for axios if needed.

export interface BackendTicket {
  id: string;
  subject: string;
  body: string;
  status: string;
  createdAt: number;
  updatedAt: number;
  classification: {
    topic_tags?: string[];
    sentiment?: string;
    priority?: string;
  };
  conversation?: { sender: string; message: string; timestamp: number }[];
}

export interface AgentQueryResponse {
  reply: string;
  tool_calls?: any[];
  ticket_id?: string;
  classification?: { topic_tags?: string[]; sentiment?: string; priority?: string };
  used_urls?: string[];
  routed?: boolean;
}

// API base URL: configurable via VITE_API_BASE (see .env.local). Default falls back to localhost.
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function fetchTickets(): Promise<BackendTicket[]> {
  const data = await api<{ tickets: BackendTicket[] }>("/tickets/query", {
    method: 'POST',
    body: JSON.stringify({}),
  });
  return data.tickets || [];
}

export async function createTicket(subject: string, body: string): Promise<BackendTicket> {
  return api<BackendTicket>("/create_ticket", {
    method: 'POST',
    body: JSON.stringify({ subject, body }),
  });
}

export async function getTicket(id: string): Promise<BackendTicket> {
  return api<BackendTicket>(`/tickets/${id}`);
}

export async function deleteTicket(id: string): Promise<{deleted: boolean; ticket_id: string}> {
  return api<{deleted: boolean; ticket_id: string}>(`/tickets/${id}`, { method: 'DELETE' });
}

export async function updateTicketFields(id: string, updates: Record<string, any>): Promise<any> {
  return api<any>(`/tickets/${id}/update`, { method: 'POST', body: JSON.stringify(updates) });
}

export async function agentQuery(user_message: string, ticket_id?: string, conversation?: { sender: string; message: string; timestamp: number }[]): Promise<AgentQueryResponse> {
  return api<AgentQueryResponse>("/agent_query", {
    method: 'POST',
    body: JSON.stringify({ user_message, ticket_id, conversation }),
  });
}

export function mapBackendTicket(t: BackendTicket) {
  // Map backend classification to UI Ticket shape
  const cls = t.classification || {};
  const rawPri = cls.priority || 'P2 (Low)';
  const priority = rawPri.startsWith('P0') ? 'P0' : rawPri.startsWith('P1') ? 'P1' : 'P2';
  const sentimentMap: Record<string, string> = {
    Positive: 'Curious',
    Neutral: 'Neutral',
    Negative: 'Frustrated',
  };
  const rawSentiment = cls.sentiment || 'Neutral';
  const sentiment = sentimentMap[rawSentiment] || (rawSentiment === 'Unknown' ? 'Neutral' : 'Neutral');
  // Normalize topic tags (may be list or stringified list)
  let topicTags: any = cls.topic_tags || [];
  if (typeof topicTags === 'string') {
    const trimmed = topicTags.trim();
    if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
      let attempt = trimmed;
      if (attempt.includes("'") && !attempt.includes('"')) attempt = attempt.replace(/'/g, '"');
      try {
        const parsed = JSON.parse(attempt);
        if (Array.isArray(parsed)) topicTags = parsed;
      } catch {
        const inner = trimmed.slice(1,-1);
        topicTags = inner.split(',').map(s => s.replace(/['"`]/g,'').trim()).filter(Boolean);
      }
    } else if (trimmed) {
      topicTags = [trimmed];
    } else {
      topicTags = [];
    }
  }
  if (!Array.isArray(topicTags)) topicTags = [];
  const topicsRaw: string[] = topicTags.slice(0,3).map((x:any)=> String(x).toLowerCase());
  const topicMap: Record<string,string> = {
    'how-to': 'How-to',
    'api/sdk': 'API/SDK',
    'connector': 'Connector',
    'product': 'Product',
    'best practices': 'How-to',
    'snowflake': 'Connector',
  };
  const topics = topicsRaw.map(x => topicMap[x] || (x.charAt(0).toUpperCase()+x.slice(1)));
  const isClassifying = (!(Array.isArray(topicTags) && topicTags.length)) || (cls.sentiment === 'Unknown');
  return {
    id: t.id,
    title: t.subject,
    content: t.body,
    priority: priority as 'P0'|'P1'|'P2',
    sentiment: sentiment as any,
    topics,
    timestamp: new Date(t.createdAt*1000).toISOString(),
    timeAgo: '—',
  isClassifying,
  // Ensure conversation always present for downstream type consistency
  conversation: t.conversation || [],
  };
}

export function parseSourcesFromReply(reply: string): { title: string; url: string }[] {
  const lines = reply.split(/\n+/);
  const idx = lines.findIndex(l => /^Sources:?/i.test(l.trim()));
  if (idx === -1) return [];
  const sources: { title: string; url: string }[] = [];
  for (let i = idx + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const urlMatch = line.match(/https?:\/\/\S+/);
    if (urlMatch) {
      const url = urlMatch[0].replace(/[).,]$/,'');
      sources.push({ title: url.replace(/^https?:\/\//,''), url });
    }
  }
  return sources;
}

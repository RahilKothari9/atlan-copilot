import { useState } from 'react';
import { agentQuery, parseSourcesFromReply } from '@/lib/api';
import type { ChatMessage } from '@/components/ChatInterface';

export function useAgentChat(activeTicketId?: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);

  const send = async (content: string) => {
    const userMsg: ChatMessage = { id: Date.now().toString(), type: 'user', content, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setLoading(true);
    try {
      const conv = messages.map(m => ({ sender: m.type === 'user' ? 'user' : 'agent', message: m.content, timestamp: m.timestamp.getTime()/1000 }));
      const res = await agentQuery(content, activeTicketId, conv);
      const sources = parseSourcesFromReply(res.reply);
      const assistant: ChatMessage = { id: (Date.now()+1).toString(), type: 'assistant', content: res.reply, timestamp: new Date(), sources: sources.length ? sources : undefined };
      setMessages(prev => [...prev, assistant]);
    } catch (e: any) {
      const errMsg: ChatMessage = { id: (Date.now()+2).toString(), type: 'assistant', content: `Error: ${e.message || e}`, timestamp: new Date() };
      setMessages(prev => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  return { messages, send, loading, setMessages };
}

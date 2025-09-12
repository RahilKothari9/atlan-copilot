import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Edit, Trash2, ArrowLeft, Send } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { Ticket } from "./TicketCard";
import { TicketAnalysisPanel } from "./TicketAnalysisPanel";

interface TicketDetailViewProps {
  ticket: Ticket;
  onBack: () => void;
  onUpdate: (ticketId: string) => void;
  onDelete: (ticketId: string) => void;
  onAskAI: (question: string, ticketId: string) => void;
}

const priorityStyles = {
  P0: "bg-priority-p0 text-white",
  P1: "bg-priority-p1 text-white",  
  P2: "bg-priority-p2 text-gray-200"
};

const sentimentStyles = {
  Frustrated: "text-sentiment-frustrated",
  Angry: "text-sentiment-angry", 
  Curious: "text-sentiment-curious",
  Neutral: "text-sentiment-neutral"
};

const topicStyles: Record<string, string> = {
  "Connector": "bg-topic-connector/20 text-topic-connector border-topic-connector/30",
  "API/SDK": "bg-topic-api/20 text-topic-api border-topic-api/30",
  "How-to": "bg-topic-default/20 text-topic-default border-topic-default/30"
};

export function TicketDetailView({ 
  ticket, 
  onBack, 
  onUpdate, 
  onDelete, 
  onAskAI 
}: TicketDetailViewProps) {
  const [aiQuestion, setAiQuestion] = useState("");

  const handleAISubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (aiQuestion.trim()) {
      onAskAI(aiQuestion.trim(), ticket.id);
      setAiQuestion("");
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-3 mb-3">
          <Button
            variant="ghost"
            size="sm"
            onClick={onBack}
            className="p-1"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h2 className="font-semibold text-foreground">Details for {ticket.id}</h2>
            <p className="text-sm text-muted-foreground">{ticket.timeAgo}</p>
          </div>
        </div>

        {/* Classification badges (hide until classification complete) */}
        {!ticket.isClassifying && (
          <div className="flex flex-wrap gap-2">
            <Badge 
              variant="secondary" 
              className={cn("text-xs font-medium", priorityStyles[ticket.priority])}
            >
              {ticket.priority}
            </Badge>
            <span className={cn("text-xs font-medium", sentimentStyles[ticket.sentiment])}>
              {ticket.sentiment}
            </span>
            {ticket.topics.map((topic, index) => (
              <Badge 
                key={index}
                variant="outline"
                className={cn(
                  "text-xs border",
                  topicStyles[topic] || "bg-muted/20 text-muted-foreground border-muted"
                )}
              >
                {topic}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Ticket content */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-6">
          <TicketAnalysisPanel 
            classification={ticket.isClassifying ? undefined : { topic_tags: ticket.topics.map(t=>t.toLowerCase()), sentiment: ticket.sentiment === 'Curious' ? 'Neutral' : ticket.sentiment, priority: ticket.priority === 'P0' ? 'P0 (High)' : ticket.priority === 'P1' ? 'P1 (Medium)' : 'P2 (Low)' }}
            agentMessage={ticket.conversation?.find(m=>m.sender==='agent')?.message || null}
            loading={ticket.isClassifying || (!ticket.conversation || ticket.conversation.length===0)}
          />
          <div>
            <h3 className="text-sm font-medium text-foreground mb-2">Original Ticket</h3>
            <div className="bg-muted/30 rounded-md p-4 text-sm text-foreground leading-relaxed whitespace-pre-wrap">
              {ticket.content}
            </div>
          </div>
          <div>
            <h3 className="text-sm font-medium text-foreground mb-2">Conversation</h3>
            {(!ticket.conversation || ticket.conversation.length === 0) && (
              <p className="text-xs text-muted-foreground">
                {(() => {
                  const ragTopics = new Set(["How-to","Product","API/SDK","SSO","Best Practices"]);
                  const hasRag = ticket.topics.some(t => ragTopics.has(t));
                  return hasRag ? 'Generating AI answer from docs…' : 'No messages yet.';
                })()}
              </p>
            )}
            <div className="space-y-3">
              {ticket.conversation?.map((m, idx) => {
                const isUser = m.sender === 'user';
                return (
                  <div key={idx} className="flex flex-col gap-1">
                    <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{isUser ? 'User' : 'Agent'} • {new Date(m.timestamp * 1000).toLocaleTimeString()}</div>
                    <div className={cn(
                      'rounded-md px-3 py-2 text-xs border border-border/50 whitespace-pre-wrap',
                      isUser ? 'bg-chat-user text-white border-transparent' : 'bg-chat-assistant text-foreground'
                    )}>{m.message}</div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </ScrollArea>

      {/* Action buttons and AI prompt */}
      <div className="border-t border-border p-4 space-y-4">
        {/* Action buttons */}
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => onUpdate(ticket.id)}
            className="flex items-center gap-2"
          >
            <Edit className="h-3 w-3" />
            Update Ticket
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => onDelete(ticket.id)}
            className="flex items-center gap-2 text-destructive hover:text-destructive"
          >
            <Trash2 className="h-3 w-3" />
            Delete Ticket
          </Button>
        </div>

        {/* AI prompt */}
        <form onSubmit={handleAISubmit} className="flex gap-2">
          <Input
            value={aiQuestion}
            onChange={(e) => setAiQuestion(e.target.value)}
            placeholder="Ask AI about this ticket..."
            className="flex-1"
          />
          <Button 
            type="submit" 
            size="sm" 
            disabled={!aiQuestion.trim()}
            className="px-3"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
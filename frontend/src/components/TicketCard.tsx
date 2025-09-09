import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export interface Ticket {
  id: string;
  title: string;
  content: string;
  priority: "P0" | "P1" | "P2";
  sentiment: "Frustrated" | "Angry" | "Curious" | "Neutral";
  topics: string[];
  timestamp: string;
  timeAgo: string;
  isClassifying?: boolean;
}

interface TicketCardProps {
  ticket: Ticket;
  isSelected?: boolean;
  onClick?: () => void;
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

export function TicketCard({ ticket, isSelected, onClick }: TicketCardProps) {
  return (
    <Card 
      className={cn(
        "p-4 cursor-pointer transition-colors duration-200 border-border",
        "hover:bg-hover-card",
        isSelected && "ring-2 ring-primary bg-accent/50"
      )}
      onClick={onClick}
    >
      <div className="space-y-3">
        {/* Header with ID and Title */}
        <div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-primary">{ticket.id}</span>
            <span className="text-xs text-muted-foreground">{ticket.timeAgo}</span>
          </div>
          <h3 className="text-sm font-medium text-foreground mt-1 line-clamp-2">
            {ticket.title}
          </h3>
        </div>

        {/* Classification badges */}
        <div className="flex flex-wrap gap-2">
          {ticket.isClassifying ? (
            <div className="flex gap-2">
              <Skeleton className="h-5 w-8 rounded-full" />
              <Skeleton className="h-5 w-16 rounded-full" />
              <Skeleton className="h-5 w-12 rounded-full" />
            </div>
          ) : (
            <>
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
            </>
          )}
        </div>
      </div>
    </Card>
  );
}
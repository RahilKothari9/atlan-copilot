import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Send, Sparkles, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ChatMessage {
  id: string;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: { title: string; url: string }[];
}

interface ChatInterfaceProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  placeholder?: string;
  isLoading?: boolean;
}

export function ChatInterface({ 
  messages, 
  onSendMessage, 
  placeholder = "Create a ticket, ask a question, or search for tickets...",
  isLoading = false 
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSendMessage(input.trim());
      setInput("");
    }
  };

  useEffect(() => {
    // Scroll to bottom when new messages are added
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Chat messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
        <div className="space-y-4">
          {messages.length === 0 && (
            <div className="flex items-center gap-2 text-muted-foreground text-sm">
              <Sparkles className="h-4 w-4" />
              <p>Welcome to the Atlan Copilot. How can I assist you today?</p>
            </div>
          )}
          
          {messages.map((message) => (
            <div key={message.id} className={cn(
              "flex",
              message.type === "user" ? "justify-end" : "justify-start"
            )}>
              <div className={cn(
                "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                message.type === "user" 
                  ? "bg-chat-user text-white" 
                  : "bg-chat-assistant border border-border text-foreground"
              )}>
                <div className="whitespace-pre-wrap">{message.content}</div>
                
                {/* Sources for AI responses */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-border/50">
                    <p className="text-xs font-medium text-muted-foreground mb-2">Sources:</p>
                    <div className="space-y-1">
                      {message.sources.map((source, index) => (
                        <a
                          key={index}
                          href={source.url}
                          className="flex items-center gap-1 text-xs text-primary hover:underline"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <ExternalLink className="h-3 w-3" />
                          {source.title}
                        </a>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-chat-assistant border border-border rounded-lg px-3 py-2 text-sm text-foreground">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-100"></div>
                    <div className="w-2 h-2 bg-primary rounded-full animate-pulse delay-200"></div>
                  </div>
                  <span className="text-muted-foreground">AI is thinking...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input form */}
      <div className="border-t border-border p-4">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={placeholder}
            disabled={isLoading}
            className="flex-1"
          />
          <Button 
            type="submit" 
            size="sm" 
            disabled={!input.trim() || isLoading}
            className="px-3"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
    </div>
  );
}
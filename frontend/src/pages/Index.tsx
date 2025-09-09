import { useState } from "react";
import { TicketCard, Ticket } from "@/components/TicketCard";
import { TicketFilters, FilterState } from "@/components/TicketFilters";
import { ChatInterface, ChatMessage } from "@/components/ChatInterface";
import { TicketDetailView } from "@/components/TicketDetailView";
import { NewTicketModal } from "@/components/NewTicketModal";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sparkles } from "lucide-react";
import { mockTickets } from "@/data/mockTickets";
import { useToast } from "@/hooks/use-toast";

type ViewState = 
  | { type: "chat" }
  | { type: "ticket-detail"; ticket: Ticket };

export default function Index() {
  const [tickets, setTickets] = useState<Ticket[]>(mockTickets);
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  const [viewState, setViewState] = useState<ViewState>({ type: "chat" });
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    search: "",
    priority: "All",
    sentiment: "All",
    topic: "All"
  });

  const { toast } = useToast();

  // Filter tickets based on current filters
  const filteredTickets = tickets.filter((ticket) => {
    const matchesSearch = filters.search === "" || 
      ticket.title.toLowerCase().includes(filters.search.toLowerCase()) ||
      ticket.content.toLowerCase().includes(filters.search.toLowerCase()) ||
      ticket.id.toLowerCase().includes(filters.search.toLowerCase());
    
    const matchesPriority = filters.priority === "All" || ticket.priority === filters.priority;
    const matchesSentiment = filters.sentiment === "All" || ticket.sentiment === filters.sentiment;  
    const matchesTopic = filters.topic === "All" || ticket.topics.includes(filters.topic);

    return matchesSearch && matchesPriority && matchesSentiment && matchesTopic;
  });

  const handleTicketSelect = (ticket: Ticket) => {
    setSelectedTicket(ticket);
    setViewState({ type: "ticket-detail", ticket });
  };

  const handleBackToChat = () => {
    setSelectedTicket(null);
    setViewState({ type: "chat" });
  };

  const handleSendMessage = async (message: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: "user",
      content: message,
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      let aiResponse = "";
      let sources: { title: string; url: string }[] = [];

      if (message.toLowerCase().includes("ticket") || message.toLowerCase().includes("search")) {
        aiResponse = `I found ${filteredTickets.length} tickets matching your query. Here are the most relevant ones:\n\n${filteredTickets.slice(0, 3).map(t => `• ${t.id}: ${t.title}`).join('\n')}`;
        sources = [
          { title: "Ticket Search Documentation", url: "#" },
          { title: "Support Knowledge Base", url: "#" }
        ];
      } else {
        aiResponse = "I'm here to help you with ticket management, searches, and support queries. You can ask me to find tickets, classify issues, or get information about customer problems.";
      }

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "assistant", 
        content: aiResponse,
        timestamp: new Date(),
        sources: sources.length > 0 ? sources : undefined
      };

      setChatMessages(prev => [...prev, assistantMessage]);
      setIsLoading(false);
    }, 1500);
  };

  const handleUpdateTicket = (ticketId: string) => {
    toast({
      title: "Ticket Updated",
      description: `Ticket ${ticketId} has been updated successfully.`
    });
  };

  const handleDeleteTicket = (ticketId: string) => {
    toast({
      title: "Ticket Deleted", 
      description: `Ticket ${ticketId} has been deleted.`,
      variant: "destructive"
    });
  };

  const handleAskAI = (question: string, ticketId: string) => {
    const contextMessage = `Question about ${ticketId}: ${question}`;
    handleSendMessage(contextMessage);
    setViewState({ type: "chat" });
  };

  const handleCreateTicket = (subject: string, body: string) => {
    const newTicketId = `ATL-${Math.floor(Math.random() * 100000)}`;
    const newTicket: Ticket = {
      id: newTicketId,
      title: subject,
      content: body,
      priority: "P2",
      sentiment: "Curious",
      topics: ["How-to"],
      timestamp: new Date().toISOString(),
      timeAgo: "Just now",
      isClassifying: true
    };

    // Add ticket to the top of the list
    setTickets(prev => [newTicket, ...prev]);

    // After 2 seconds, remove the loading state
    setTimeout(() => {
      setTickets(prev => prev.map(ticket => 
        ticket.id === newTicketId 
          ? { ...ticket, isClassifying: false }
          : ticket
      ));
    }, 2000);

    toast({
      title: "Ticket Created",
      description: `Ticket ${newTicketId} has been created successfully.`
    });
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card">
        <div className="flex items-center gap-2 p-4">
          <Sparkles className="h-6 w-6 text-primary" />
          <h1 className="text-xl font-semibold text-foreground">
            Atlan Support Copilot ✨
          </h1>
        </div>
      </header>

      {/* Main content */}
      <div className="flex h-[calc(100vh-65px)]">
        {/* Left column - Ticket Dashboard */}
        <div className="w-[65%] border-r border-border flex flex-col">
          <div className="flex items-center justify-between p-4 border-b border-border">
            <div className="flex-1">
              <TicketFilters filters={filters} onFiltersChange={setFilters} />
            </div>
            <div className="ml-4">
              <NewTicketModal onCreateTicket={handleCreateTicket} />
            </div>
          </div>
          
          <ScrollArea className="flex-1">
            <div className="p-4 pt-0 space-y-3">
              {filteredTickets.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <p>No tickets found matching your filters.</p>
                </div>
              ) : (
                filteredTickets.map((ticket) => (
                  <TicketCard
                    key={ticket.id}
                    ticket={ticket}
                    isSelected={selectedTicket?.id === ticket.id}
                    onClick={() => handleTicketSelect(ticket)}
                  />
                ))
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Right column - AI Agent / Ticket Details */}
        <div className="w-[35%] bg-card">
          {viewState.type === "chat" ? (
            <ChatInterface
              messages={chatMessages}
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
            />
          ) : (
            <TicketDetailView
              ticket={viewState.ticket}
              onBack={handleBackToChat}
              onUpdate={handleUpdateTicket}
              onDelete={handleDeleteTicket}
              onAskAI={handleAskAI}
            />
          )}
        </div>
      </div>
    </div>
  );
}
import { useState, useEffect } from "react";
import { TicketCard, Ticket } from "@/components/TicketCard";
import { TicketFilters, FilterState } from "@/components/TicketFilters";
import { ChatInterface } from "@/components/ChatInterface";
import { TicketDetailView } from "@/components/TicketDetailView";
import { NewTicketModal } from "@/components/NewTicketModal";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Sparkles } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useTickets } from "@/hooks/useTickets";
import { useAgentChat } from "@/hooks/useAgentChat";
import { useTicketDetail } from "@/hooks/useTicketDetail";

type ViewState = 
  | { type: "chat" }
  | { type: "ticket-detail"; ticket: Ticket };

export default function Index() {
  const { ticketsQuery, createTicket, creating, deleteTicket, updateTicket } = useTickets();
  const [selectedTicket, setSelectedTicket] = useState<Ticket | null>(null);
  const ticketDetail = useTicketDetail(selectedTicket?.id);
  const [viewState, setViewState] = useState<ViewState>({ type: "chat" });
  const { messages, send, loading, setMessages } = useAgentChat(selectedTicket?.id);
  const [filters, setFilters] = useState<FilterState>({
    search: "",
    priority: "All",
    sentiment: "All",
    topic: "All"
  });

  const { toast } = useToast();

  // Filter tickets based on current filters
  const tickets = ticketsQuery.data || [];
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
    setMessages([]); // reset chat context when switching tickets
  };

  const handleBackToChat = () => {
    setSelectedTicket(null);
    setViewState({ type: "chat" });
  };

  const handleSendMessage = async (message: string) => {
    send(message);
  };

  const handleUpdateTicket = (ticketId: string) => {
    // Simple example: toggle status between Open & In-Progress
    const t = tickets.find(t => t.id === ticketId);
    const nextStatus = 'In-Progress';
    updateTicket(ticketId, { status: nextStatus });
    toast({ title: 'Ticket Updated', description: `Status set to ${nextStatus}.` });
  };

  const handleDeleteTicket = (ticketId: string) => {
    deleteTicket(ticketId);
    if (selectedTicket?.id === ticketId) {
      setSelectedTicket(null);
      setViewState({ type: 'chat' });
    }
    toast({ title: 'Ticket Deleted', description: `Ticket ${ticketId} removed.`, variant: 'destructive' });
  };

  const handleAskAI = (question: string, ticketId: string) => {
    const contextMessage = `Question about ${ticketId}: ${question}`;
    handleSendMessage(contextMessage);
    setViewState({ type: "chat" });
  };

  const handleCreateTicket = (subject: string, body: string) => {
    createTicket(subject, body);
    toast({ title: "Ticket Created", description: `Ticket created. Classification pending...` });
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
              messages={messages}
              onSendMessage={handleSendMessage}
              isLoading={loading}
            />
          ) : (
            <TicketDetailView
              ticket={{ ...viewState.ticket, conversation: ticketDetail.data?.conversation || viewState.ticket.conversation }}
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
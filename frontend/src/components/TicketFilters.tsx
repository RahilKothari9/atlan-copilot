import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Search } from "lucide-react";

export interface FilterState {
  search: string;
  priority: "All" | "P0" | "P1" | "P2";
  sentiment: "All" | "Frustrated" | "Angry" | "Curious" | "Neutral";
  topic: "All" | "How-to" | "Connector" | "API/SDK";
}

interface TicketFiltersProps {
  filters: FilterState;
  onFiltersChange: (filters: FilterState) => void;
}

export function TicketFilters({ filters, onFiltersChange }: TicketFiltersProps) {
  return (
    <div className="p-4 border-b border-border space-y-4">
      {/* Search bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search tickets by content or ID..."
          value={filters.search}
          onChange={(e) => onFiltersChange({ ...filters, search: e.target.value })}
          className="pl-9"
        />
      </div>

      {/* Filter dropdowns */}
      <div className="flex gap-3">
        <Select 
          value={filters.priority} 
          onValueChange={(value) => onFiltersChange({ ...filters, priority: value as FilterState['priority'] })}
        >
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Priority" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="All">All Priority</SelectItem>
            <SelectItem value="P0">P0 (High)</SelectItem>
            <SelectItem value="P1">P1 (Medium)</SelectItem>
            <SelectItem value="P2">P2 (Low)</SelectItem>
          </SelectContent>
        </Select>

        <Select 
          value={filters.sentiment} 
          onValueChange={(value) => onFiltersChange({ ...filters, sentiment: value as FilterState['sentiment'] })}
        >
          <SelectTrigger className="w-[130px]">
            <SelectValue placeholder="Sentiment" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="All">All Sentiment</SelectItem>
            <SelectItem value="Frustrated">Frustrated</SelectItem>
            <SelectItem value="Angry">Angry</SelectItem>
            <SelectItem value="Curious">Curious</SelectItem>
            <SelectItem value="Neutral">Neutral</SelectItem>
          </SelectContent>
        </Select>

        <Select 
          value={filters.topic} 
          onValueChange={(value) => onFiltersChange({ ...filters, topic: value as FilterState['topic'] })}
        >
          <SelectTrigger className="w-[120px]">
            <SelectValue placeholder="Topic" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="All">All Topics</SelectItem>
            <SelectItem value="How-to">How-to</SelectItem>
            <SelectItem value="Connector">Connector</SelectItem>
            <SelectItem value="API/SDK">API/SDK</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
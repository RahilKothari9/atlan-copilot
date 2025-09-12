import React from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

interface Props {
  classification: { topic_tags?: string[]; sentiment?: string; priority?: string } | undefined;
  agentMessage?: string | null;
  loading?: boolean;
}

const priorityColor: Record<string,string> = { 'P0 (High)': 'bg-priority-p0 text-white', 'P1 (Medium)': 'bg-priority-p1 text-white', 'P2 (Low)': 'bg-priority-p2 text-gray-200'};

export const TicketAnalysisPanel: React.FC<Props> = ({ classification, agentMessage, loading }) => {
  const tags = classification?.topic_tags || [];
  return (
    <div className="space-y-4">
      <div className="border border-border rounded-md p-3">
        <h4 className="text-xs font-semibold mb-2 tracking-wide text-muted-foreground">Internal Analysis</h4>
  {!classification && <p className="text-xs text-muted-foreground">Generating analysis…</p>}
        {classification && (
          <div className="text-xs space-y-2">
            <div className="flex gap-2 flex-wrap">
              {tags.slice(0,3).map((t,i)=>(<Badge key={i} variant="outline" className="text-[10px]">{t}</Badge>))}
            </div>
            <div>Sentiment: <span className="font-medium">{classification.sentiment}</span></div>
            <div>Priority: <span className={cn('font-medium px-1 rounded', priorityColor[classification.priority||'P2 (Low)'])}>{classification.priority}</span></div>
          </div>
        )}
      </div>
      <div className="border border-border rounded-md p-3">
        <h4 className="text-xs font-semibold mb-2 tracking-wide text-muted-foreground">Final Response</h4>
        {loading && !agentMessage && <p className="text-xs text-muted-foreground">Generating response…</p>}
        {agentMessage && <div className="text-xs whitespace-pre-wrap leading-relaxed">{agentMessage}</div>}
        {!loading && !agentMessage && <p className="text-xs text-muted-foreground">—</p>}
      </div>
    </div>
  );
};

import { useQuery } from '@tanstack/react-query';
import { getTicket, mapBackendTicket, BackendTicket } from '@/lib/api';

export function useTicketDetail(ticketId: string | undefined) {
  return useQuery({
    queryKey: ['ticket', ticketId],
    queryFn: async () => {
      if (!ticketId) return null;
      const t: BackendTicket = await getTicket(ticketId);
      const mapped = mapBackendTicket(t);
      return { ...mapped, conversation: t.conversation || [] };
    },
    enabled: !!ticketId,
    refetchInterval: 3000,
  });
}

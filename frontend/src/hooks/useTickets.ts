import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { fetchTickets, createTicket, mapBackendTicket, BackendTicket, getTicket, deleteTicket, updateTicketFields } from '@/lib/api';

export function useTickets() {
  const qc = useQueryClient();
  const ticketsQuery = useQuery({
    queryKey: ['tickets'],
    queryFn: async () => {
      const data = await fetchTickets();
      return data.map(mapBackendTicket);
    },
    refetchInterval: 5000, // poll for classification updates
  });

  const createMutation = useMutation({
    mutationFn: async (vars: { subject: string; body: string }) => createTicket(vars.subject, vars.body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tickets'] });
    }
  });

  const refetchTicket = async (id: string) => {
    const t: BackendTicket = await getTicket(id);
    // update cache manually
    qc.setQueryData<any>(['tickets'], (old: any) => {
      if (!Array.isArray(old)) return old;
      const mapped = mapBackendTicket(t);
      const exists = old.findIndex((o: any) => o.id === t.id);
      if (exists === -1) return [mapped, ...old];
      const copy = [...old];
      copy[exists] = mapped;
      return copy;
    });
  };

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => deleteTicket(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tickets'] });
    }
  });

  const updateMutation = useMutation({
    mutationFn: async (vars: { id: string; updates: Record<string, any> }) => updateTicketFields(vars.id, vars.updates),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['tickets'] });
    }
  });

  return {
    ticketsQuery,
    createTicket: (subject: string, body: string) => createMutation.mutate({ subject, body }),
    creating: createMutation.isPending,
    refetchTicket,
  deleteTicket: (id: string) => deleteMutation.mutate(id),
  updateTicket: (id: string, updates: Record<string, any>) => updateMutation.mutate({ id, updates }),
  };
}

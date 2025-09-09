import { Ticket } from "@/components/TicketCard";

export const mockTickets: Ticket[] = [
  {
    id: "ATL-84321",
    title: "Unable to connect BigQuery connector after recent update",
    content: "Hi team, I'm having trouble with our BigQuery connector after the recent Atlan update. The connection keeps timing out and I'm getting authentication errors. I've tried refreshing the credentials but it's still not working. This is affecting our daily data sync jobs and my team is blocked. Can you please help urgently?",
    priority: "P0",
    sentiment: "Frustrated",
    topics: ["Connector", "API/SDK"],
    timestamp: "2024-01-08T14:30:00Z",
    timeAgo: "5m ago"
  },
  {
    id: "ATL-84320",
    title: "How to set up custom lineage tracking for dbt models?",
    content: "I'm looking for guidance on setting up custom lineage tracking for our dbt models in Atlan. We have complex transformations that aren't being captured automatically. Is there documentation on how to configure this manually?",
    priority: "P2",
    sentiment: "Curious", 
    topics: ["How-to", "Connector"],
    timestamp: "2024-01-08T14:15:00Z",
    timeAgo: "20m ago"
  },
  {
    id: "ATL-84319",
    title: "API rate limits exceeded - need enterprise support",
    content: "Our API calls are constantly hitting rate limits and it's severely impacting our operations. We need to understand what enterprise options are available to increase these limits. This is becoming a major blocker for our team.",
    priority: "P1",
    sentiment: "Angry",
    topics: ["API/SDK"],
    timestamp: "2024-01-08T13:45:00Z", 
    timeAgo: "50m ago"
  },
  {
    id: "ATL-84318",
    title: "Snowflake connector failing with SSL certificate error",
    content: "Getting SSL certificate verification errors when trying to connect to Snowflake. The error message says 'certificate verify failed: unable to get local issuer certificate'. This started happening after our infrastructure team updated certificates.",
    priority: "P1",
    sentiment: "Neutral",
    topics: ["Connector"],
    timestamp: "2024-01-08T13:30:00Z",
    timeAgo: "1h ago"
  },
  {
    id: "ATL-84317", 
    title: "Request for data governance training materials",
    content: "Hi! We're onboarding a new team member and would love access to any training materials or best practices documentation for data governance workflows in Atlan. Any resources would be greatly appreciated!",
    priority: "P2",
    sentiment: "Curious",
    topics: ["How-to"],
    timestamp: "2024-01-08T12:15:00Z",
    timeAgo: "2h ago"
  },
  {
    id: "ATL-84316",
    title: "Performance degradation in catalog search functionality", 
    content: "The catalog search has become significantly slower over the past week. Simple searches that used to return results in seconds are now taking 30+ seconds. This is affecting our entire team's productivity. Please investigate.",
    priority: "P0",
    sentiment: "Frustrated",
    topics: ["API/SDK"],
    timestamp: "2024-01-08T11:45:00Z",
    timeAgo: "3h ago"
  },
  {
    id: "ATL-84315",
    title: "Integration with Apache Kafka - metadata extraction", 
    content: "We're exploring integrating Atlan with our Apache Kafka setup to extract metadata from our streaming data topics. Is there existing connector support for this, or would we need to build something custom?",
    priority: "P2", 
    sentiment: "Curious",
    topics: ["Connector", "How-to"],
    timestamp: "2024-01-08T10:30:00Z",
    timeAgo: "4h ago"
  },
  {
    id: "ATL-84314",
    title: "GraphQL API documentation missing examples",
    content: "The GraphQL API documentation is missing concrete examples for complex queries. Specifically, I need help with querying nested asset relationships and applying filters. Could you provide more comprehensive examples?",
    priority: "P2",
    sentiment: "Neutral", 
    topics: ["API/SDK", "How-to"],
    timestamp: "2024-01-08T09:15:00Z",
    timeAgo: "5h ago"
  }
];
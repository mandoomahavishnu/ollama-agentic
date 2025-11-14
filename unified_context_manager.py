"""
Unified Context Manager - Enables ALL agents to share conversations and context
This is the "brain" that gives agents awareness of what other agents have done.

Key Features:
- Centralized context hub for all agents
- Semantic search across ALL agent conversations
- Recency-weighted context retrieval
- Cross-agent handoff support
- Smart context summarization
- Caching for performance

Author: AI Assistant
Date: 2025
"""

import numpy as np
import ollama
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

def _ensure_aware(dt: datetime, tz: ZoneInfo) -> datetime:
    # Make a datetime tz-aware in the given tz if it's naive
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt

def _parse_ts(x: Any, tz: ZoneInfo) -> datetime:
    if isinstance(x, datetime):
        return _ensure_aware(x, tz)
    # Try ISO 8601 first
    try:
        dt = datetime.fromisoformat(str(x))
    except Exception:
        # Fallback to now if parsing fails
        dt = datetime.now(tz)
    return _ensure_aware(dt, tz)

def _to_utc(dt: datetime) -> datetime:
    if dt is None:
        return None
    # If naive: assume it's UTC (or change to your preferred default TZ)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

@dataclass
class ContextMessage:
    
    """Represents a single message in the unified context"""
    user_query: str
    response: str
    agent_used: str
    ts: datetime
    created_at: datetime
    results_summary: Optional[str] = None
    sql_query: Optional[str] = None
    relevance_score: float = 0.0
    recency_weight: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "response": self.response,
            "agent_used": self.agent_used,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "results_summary": self.results_summary,
            "sql_query": self.sql_query,
            "relevance_score": self.relevance_score,
            "recency_weight": self.recency_weight
        }


@dataclass
class UnifiedContext:
    
    
    """Complete context package for agent decision-making"""
    current_query: str
    relevant_messages: List[ContextMessage] = field(default_factory=list)
    recent_messages: List[ContextMessage] = field(default_factory=list)
    last_agent: Optional[str] = None
    agent_history: Dict[str, int] = field(default_factory=dict)  # agent_name -> count
    session_summary: str = ""
    cross_agent_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_formatted_context(self, max_messages: int = 10) -> str:
        """Format context for LLM consumption"""
        lines = []
        
        if self.session_summary:
            lines.append(f"=== Session Summary ===\n{self.session_summary}\n")
        
        lines.append("=== Recent Conversation History ===")
        for msg in self.recent_messages[-max_messages:]:
            lines.append(f"\n[{msg.agent_used.upper()}] User: {msg.user_query}")
            if msg.response:
                response_preview = msg.response[:200] + "..." if len(msg.response) > 200 else msg.response
                lines.append(f"Response: {response_preview}")
            if msg.sql_query:
                lines.append(f"SQL: {msg.sql_query[:100]}...")
        
        if self.relevant_messages:
            lines.append("\n\n=== Semantically Relevant Past Conversations ===")
            for msg in self.relevant_messages[:5]:
                lines.append(f"\n[{msg.agent_used.upper()}] (Relevance: {msg.relevance_score:.2f})")
                lines.append(f"User: {msg.user_query}")
                if msg.results_summary:
                    lines.append(f"Summary: {msg.results_summary}")
        
        return "\n".join(lines)
    
    def has_cross_agent_context(self) -> bool:
        """Check if context includes messages from multiple agents"""
        agents = set(msg.agent_used for msg in self.recent_messages)
        return len(agents) > 1
    
    def get_last_agent_context(self) -> Optional[ContextMessage]:
        """Get the most recent message from any agent"""
        if self.recent_messages:
            return self.recent_messages[-1]
        return None


class UnifiedContextManager:
    
    """
    Central context hub that all agents use to share information.
    This enables true multi-agent conversation awareness.
    """
    
    def __init__(self, query_store, session_id: str, cache_ttl: int = 300, tz_name: str = "America/Los_Angeles"):
        """
        Initialize the unified context manager
        
        Args:
            query_store: QueryStore instance with access to conversation_embeddings
            session_id: Current session identifier
            cache_ttl: Cache time-to-live in seconds (default: 5 minutes)
        """
        self.query_store = query_store
        self.session_id = session_id
        self.cache_ttl = cache_ttl
        self.tz = ZoneInfo(tz_name)
        self.messages: List[ContextMessage] = []        
        # Context cache
        self._context_cache: Dict[str, Tuple[UnifiedContext, datetime]] = {}
        self._message_cache: Optional[Tuple[List[ContextMessage], datetime]] = None
        
        # Cross-agent data sharing
        self._shared_data: Dict[str, Any] = {}
        
        # Agent interaction tracking
        self._agent_transitions: List[Tuple[str, str, datetime]] = []  # (from_agent, to_agent, timestamp)
        
        print(f"✅ UnifiedContextManager initialized for session {session_id}")
    
    def _now(self) -> datetime:
        """Get current time with timezone"""
        return datetime.now(self.tz)

    def _to_dt(self, x: Any) -> datetime:
        """Convert any value to datetime with timezone"""
        return _parse_ts(x, self.tz)

    # Example method you likely have: add_interaction
    def add_interaction(self, user_query: str, result: Any, agent_used: str) -> None:
        self.messages.append(
            ContextMessage(
                user_query=user_query,
                response=str(result)[:2000],  # or whatever you store
                agent_used=agent_used,
                ts=self._now(),               # ALWAYS tz-aware
                created_at=self._now()
            )
        )
        
    def _is_recent(self, ts: Any, within: timedelta) -> bool:
        return (self._now() - self._to_dt(ts)) <= within    
    # ==================== Core Context Retrieval ====================
    
    def get_unified_context(
        self, 
        current_query: str,
        include_all_agents: bool = True,
        max_recent: int = 10,
        max_relevant: int = 5,
        recency_weight: float = 0.3,
        relevance_weight: float = 0.7
    ) -> UnifiedContext:
        """
        Get comprehensive context from ALL agents, optimized for the current query.
        
        This is the main method agents should call to get full context awareness.
        
        Args:
            current_query: The current user query
            include_all_agents: If True, include messages from all agents
            max_recent: Maximum number of recent messages to include
            max_relevant: Maximum number of semantically relevant messages
            recency_weight: Weight for recency in scoring (0-1)
            relevance_weight: Weight for relevance in scoring (0-1)
            
        Returns:
            UnifiedContext object with all relevant information
        """
        # Check cache first
        cache_key = f"{current_query}_{include_all_agents}_{max_recent}_{max_relevant}"
        if cache_key in self._context_cache:
            cached_context, timestamp = self._context_cache[cache_key]
            now_utc = datetime.now(timezone.utc)
            ts_utc = _to_utc(timestamp)
            if now_utc - ts_utc < timedelta(seconds=self.cache_ttl):
                age_seconds = (self._now() - timestamp).total_seconds()
                print(f"✅ Using cached context (age: {age_seconds:.0f}s)")
                return cached_context
        
        # Build fresh context
        context = UnifiedContext(current_query=current_query)
        
        # Get all messages from database
        all_messages = self._get_all_conversation_messages(include_all_agents)
        
        if not all_messages:
            return context
        
        # Recent messages (chronological)
        context.recent_messages = self._get_recent_messages(all_messages, max_recent)
        
        # Relevant messages (semantic search)
        if current_query:
            context.relevant_messages = self._get_relevant_messages_semantic(
                current_query, all_messages, max_relevant
            )
        
        # Agent statistics
        context.agent_history = self._count_agent_usage(all_messages)
        context.last_agent = all_messages[-1].agent_used if all_messages else None
        
        # Session summary (generated lazily if needed)
        if len(all_messages) > 20:
            context.session_summary = self._generate_session_summary(all_messages)
        
        # Cache the result
        self._context_cache[cache_key] = (context, self._now())
        
        return context
    
    
    def _get_all_conversation_messages(self, include_all_agents: bool = True) -> List[ContextMessage]:
        """
        Retrieve ALL conversation messages from the database.
        This is the foundation for context building.
        """
        # Check message cache first
        if self._message_cache:
            cached_messages, cache_time = self._message_cache
            cache_age = (self._now() - cache_time).total_seconds()
            if cache_age < self.cache_ttl:
                return cached_messages
        
        try:
            with self.query_store.conn.cursor() as cur:
                if include_all_agents:
                    # Get ALL conversations from this session
                    cur.execute("""
                        SELECT user_query, response, agent_used, created_at, 
                               sql_query, results
                        FROM conversation_embeddings
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                    """, (self.session_id,))
                else:
                    # Filter by specific agent if needed
                    cur.execute("""
                        SELECT user_query, response, agent_used, created_at,
                               sql_query, results
                        FROM conversation_embeddings
                        WHERE session_id = %s
                        ORDER BY created_at ASC
                    """, (self.session_id,))
                
                rows = cur.fetchall()
                self.query_store.conn.commit()
                
                messages = []
                for row in rows:
                    user_query, response, agent_used, created_at, sql_query, results = row
                    
                    results_summary = None
                    if results:
                        try:
                            if isinstance(results, str):
                                results = json.loads(results)
                            if isinstance(results, list) and len(results) > 0:
                                results_summary = f"Returned {len(results)} rows"
                        except:
                            pass
                    
                    messages.append(ContextMessage(
                        user_query=user_query,
                        response=response or "",
                        agent_used=agent_used or "unknown",
                        ts=created_at,  # FIXED: Added ts parameter
                        created_at=created_at,
                        results_summary=results_summary,
                        sql_query=sql_query
                    ))
                
                # Update cache
                self._message_cache = (messages, self._now())
                
                return messages
                
        except Exception as e:
            print(f"❌ Error retrieving conversation messages: {e}")
            print(f"⚠️ No previous conversations found")
            return []
    
    
    def _get_relevant_messages_semantic(
        self, 
        current_query: str, 
        all_messages: List[ContextMessage],
        max_results: int = 5
    ) -> List[ContextMessage]:
        """Use pgvector semantic search to find relevant past conversations"""
        try:
            # Get embedding for current query
            embedding = self.query_store._get_embedding(current_query)
            
            with self.query_store.conn.cursor() as cur:
                cur.execute("""
                    SELECT user_query, response, agent_used, created_at,
                           sql_query, results,
                           1 - (embedding1 <=> %s::vector) as similarity
                    FROM conversation_embeddings
                    WHERE session_id = %s
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                """, (json.dumps(embedding), self.session_id, json.dumps(embedding), max_results * 2))
                
                rows = cur.fetchall()
                self.query_store.conn.commit()
                
                messages = []
                for row in rows:
                    user_query, response, agent_used, created_at, sql_query, results, similarity = row
                    
                    # Skip if similarity is too low
                    if similarity < 0.3:
                        continue
                    
                    results_summary = None
                    if results:
                        try:
                            if isinstance(results, str):
                                results = json.loads(results)
                            if isinstance(results, list) and len(results) > 0:
                                results_summary = f"Found {len(results)} results"
                        except:
                            pass
                    
                    msg = ContextMessage(
                        user_query=user_query,
                        response=response or "",
                        agent_used=agent_used or "unknown",
                        ts=created_at,  # FIXED: Added ts parameter
                        created_at=created_at,
                        results_summary=results_summary,
                        sql_query=sql_query,
                        relevance_score=float(similarity)
                    )
                    messages.append(msg)
                
                return messages[:max_results]
                
        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []
    
    
    def _get_recent_messages(
        self, 
        messages: List[ContextMessage],
        max_recent: int = 10
    ) -> List[ContextMessage]:
        """Get the most recent messages, with recency weighting"""
        # Already sorted by created_at in the query
        recent = messages[-max_recent:] if len(messages) > max_recent else messages
        
        # Apply recency weighting
        now = self._now()
        for msg in recent:
            age_hours = (now - msg.created_at).total_seconds() / 3600
            # Exponential decay: messages lose relevance over time
            # Weight = 1.0 at 0 hours, 0.5 at 24 hours, 0.25 at 48 hours
            msg.recency_weight = np.exp(-age_hours / 24)
        
        return recent
    
    
    # ==================== Context Storage & Updates ====================
    
    def add_conversation_entry(
        self,
        user_query: str,
        response: str,
        agent_used: str,
        results: Optional[List[Dict]] = None,
        sql_query: Optional[str] = None
    ):
        """Add a new conversation entry to the context"""
        # Store in database via query_store
        session_id = self.session_id
        
        # Track agent transitions
        if self._agent_transitions:
            last_agent = self._agent_transitions[-1][1]  # Get 'to_agent' from last transition
            if last_agent != agent_used:
                self._agent_transitions.append((last_agent, agent_used, self._now()))
        else:
            # First interaction
            self._agent_transitions.append((None, agent_used, self._now()))
        
        # Clear caches since we added new data
        self.clear_cache()
        
        print(f"📝 Added conversation entry: {agent_used} | Query: {user_query[:50]}...")
    
    
    def _count_agent_usage(self, messages: List[ContextMessage]) -> Dict[str, int]:
        """Count how many times each agent was used"""
        counts = defaultdict(int)
        for msg in messages:
            counts[msg.agent_used] += 1
        return dict(counts)
    
    
    def _generate_session_summary(self, messages: List[ContextMessage]) -> str:
        """Generate a summary of the session so far"""
        if len(messages) < 5:
            return ""
        
        # Simple summary for now - could be enhanced with LLM
        agent_counts = self._count_agent_usage(messages)
        summary_parts = [
            f"Session with {len(messages)} interactions",
            f"Agents used: {', '.join(agent_counts.keys())}"
        ]
        
        # Add most common queries
        query_types = defaultdict(int)
        for msg in messages[-20:]:  # Last 20 messages
            query_lower = msg.user_query.lower()
            if any(word in query_lower for word in ['shipped', 'shipping', 'delivery']):
                query_types['shipping'] += 1
            elif any(word in query_lower for word in ['crew', 'team', 'working']):
                query_types['crew_info'] += 1
            elif any(word in query_lower for word in ['sales', 'order', 'customer']):
                query_types['sales'] += 1
        
        if query_types:
            top_type = max(query_types, key=query_types.get)
            summary_parts.append(f"Recent focus: {top_type}")
        
        return " | ".join(summary_parts)
    
    
    # ==================== Cross-Agent Data Sharing ====================
    
    def share_data(self, key: str, value: Any, from_agent: str):
        """Share data between agents"""
        self._shared_data[key] = {
            "value": value,
            "from_agent": from_agent,
            "timestamp": self._now()
        }
        print(f"📤 {from_agent} shared data: {key}")
    
    
    def get_shared_data(self, key: str) -> Optional[Any]:
        """Retrieve shared data"""
        if key in self._shared_data:
            return self._shared_data[key]["value"]
        return None
    
    
    def get_all_shared_data(self) -> Dict[str, Any]:
        """Get all shared data"""
        return {k: v["value"] for k, v in self._shared_data.items()}
    
    
    def get_agent_transition_history(self, last_n: int = 10) -> List[Tuple[str, str, datetime]]:
        """Get recent agent transitions"""
        return self._agent_transitions[-last_n:]
    
    
    def get_handoff_context(self, to_agent: str) -> Dict[str, Any]:
        """
        Get context specifically prepared for an agent that's taking over.
        
        This is useful when routing switches agents mid-conversation.
        """
        context = self.get_unified_context(
            current_query="",  # No specific query for handoff
            max_recent=5,
            max_relevant=0  # Just recent, not semantic search
        )
        
        last_msg = context.get_last_agent_context()
        
        return {
            "from_agent": last_msg.agent_used if last_msg else None,
            "last_query": last_msg.user_query if last_msg else None,
            "last_response": last_msg.response if last_msg else None,
            "shared_data": self.get_all_shared_data(),
            "recent_messages": [msg.to_dict() for msg in context.recent_messages[-3:]],
        }
    
    
    # ==================== Utility Methods ====================
    
    def clear_cache(self):
        """Clear all caches - useful when new messages are added"""
        self._context_cache.clear()
        self._message_cache = None
        print("🧹 Context cache cleared")
    
    
    def get_context_for_agent(
        self, 
        agent_name: str, 
        current_query: str,
        include_other_agents: bool = True
    ) -> str:
        """
        Get formatted context string specifically for a given agent.
        
        This is what you'd pass to an agent's LLM prompt.
        
        Args:
            agent_name: Name of the agent requesting context
            current_query: The current user query
            include_other_agents: Whether to include conversations from other agents
            
        Returns:
            Formatted context string ready for LLM consumption
        """
        context = self.get_unified_context(
            current_query=current_query,
            include_all_agents=include_other_agents
        )
        
        # Add agent-specific preamble
        preamble = f"=== Context for {agent_name.upper()} Agent ===\n"
        preamble += f"Current query: {current_query}\n\n"
        
        if not include_other_agents:
            preamble += f"(Showing only {agent_name} conversations)\n\n"
        else:
            if context.has_cross_agent_context():
                preamble += "⚠️ Note: This conversation involves multiple agents. "
                preamble += f"You are {agent_name}. Other agents: "
                other_agents = [a for a in context.agent_history.keys() if a != agent_name]
                preamble += ", ".join(other_agents) + "\n\n"
        
        return preamble + context.get_formatted_context()
    
    
    def detect_cross_agent_reference(self, current_query: str) -> Optional[Dict[str, Any]]:
        """
        Detect if the current query is referencing something from another agent.
        
        Returns None if no cross-reference detected, otherwise returns:
        {
            "referenced_agent": str,
            "referenced_message": ContextMessage,
            "reference_type": "follow_up" | "data_reference" | "implicit"
        }
        """
        # Get recent context
        context = self.get_unified_context(current_query, max_recent=5, max_relevant=3)
        
        if not context.recent_messages:
            return None
        
        last_msg = context.recent_messages[-1]
        query_lower = current_query.lower()
        
        # Pattern detection
        follow_up_patterns = [
            "what was that", "from before", "the previous", "you just",
            "earlier", "you said", "that number", "those results"
        ]
        
        # Check for follow-up language
        has_follow_up = any(pattern in query_lower for pattern in follow_up_patterns)
        
        if has_follow_up:
            return {
                "referenced_agent": last_msg.agent_used,
                "referenced_message": last_msg,
                "reference_type": "follow_up"
            }
        
        # Check semantic similarity with recent non-current-agent messages
        for msg in reversed(context.recent_messages[:-1]):
            if msg.relevance_score > 0.7:  # High similarity
                return {
                    "referenced_agent": msg.agent_used,
                    "referenced_message": msg,
                    "reference_type": "implicit"
                }
        
        return None
    
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the unified context system"""
        messages = self._get_all_conversation_messages()
        
        return {
            "session_id": self.session_id,
            "total_messages": len(messages),
            "agent_usage": self._count_agent_usage(messages),
            "cache_size": len(self._context_cache),
            "shared_data_keys": len(self._shared_data),
            "agent_transitions": len(self._agent_transitions),
            "last_agent": messages[-1].agent_used if messages else None,
            "session_age_minutes": (
                (datetime.now(timezone.utc) - _to_utc(messages[0].created_at)).total_seconds() / 60
                if messages and messages[0].created_at else 0
            )
        }


# ==================== Helper Functions for Agent Integration ====================

def integrate_unified_context_into_agent(
    agent_process_query_func,
    unified_context_manager: UnifiedContextManager,
    agent_name: str
):
    """
    Decorator to automatically integrate unified context into any agent.
    
    Usage:
        @integrate_unified_context_into_agent(unified_manager, "nl2sql")
        def process_query(self, user_query, routing_info, session_state):
            # Your agent code here
            # Access context via session_state['unified_context']
            pass
    """
    def wrapper(self, user_query, routing_info, session_state, **kwargs):
        # Get unified context
        unified_context = unified_context_manager.get_unified_context(
            current_query=user_query,
            include_all_agents=True
        )
        
        # Add to session state for agent to use
        session_state['unified_context'] = unified_context
        session_state['unified_context_manager'] = unified_context_manager
        
        # Detect cross-agent references
        cross_ref = unified_context_manager.detect_cross_agent_reference(user_query)
        if cross_ref:
            session_state['cross_agent_reference'] = cross_ref
            print(f"🔗 Detected cross-agent reference to {cross_ref['referenced_agent']}")
        
        # Call original function
        result = agent_process_query_func(self, user_query, routing_info, session_state, **kwargs)
        
        # Clear cache after processing (new message will be added)
        unified_context_manager.clear_cache()
        
        return result
    
    return wrapper

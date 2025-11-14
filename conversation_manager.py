"""
Enhanced Conversation Manager - Integrates with UnifiedContextManager
This is a drop-in replacement for your existing conversation_manager.py

Key improvements:
- Backward compatible with existing code
- Automatically uses UnifiedContextManager if available
- Falls back to original behavior if unified context not available
- Adds cross-agent awareness to all conversations
"""

import numpy as np
import ollama
from nlp_utils import extract_query_entities
from config import TABLE_NAMES, KEYWORD_TABLE_MAP, COLUMN_EQUIVALENCE
from typing import Optional, Dict, Any, List


class ConversationManager:
    """
    Enhanced conversation manager with unified context support.
    
    This maintains backward compatibility with your existing code while adding
    powerful cross-agent context awareness when UnifiedContextManager is available.
    """
    
    def __init__(self, session_id, query_store, unified_context_manager=None):
        """
        Initialize conversation manager
        
        Args:
            session_id: Current session identifier
            query_store: QueryStore instance
            unified_context_manager: Optional UnifiedContextManager for cross-agent context
        """
        self.session_id = session_id
        self.query_store = query_store
        self.history = []  # Local in-memory history (original behavior)
        
        # NEW: Unified context support
        self.unified_context_manager = unified_context_manager
        self.unified_enabled = unified_context_manager is not None
        
        if self.unified_enabled:
            print(f"✅ ConversationManager: Unified context ENABLED")
        else:
            print(f"⚠️ ConversationManager: Using local history only (unified context disabled)")

    def add_message(self, role, content):
        """
        Add a message to conversation history
        
        This works exactly like before, but now also clears unified context cache
        so other agents see the new message immediately.
        """
        # Original behavior: add to local history
        self.history.append({"role": role, "content": content})
        
        # NEW: Clear unified context cache so it picks up this new message
        if self.unified_enabled:
            self.unified_context_manager.clear_cache()

    def get_recent_context(self, n=6, include_all_agents=None):
        """
        Get recent conversation context
        
        Args:
            n: Number of recent messages to retrieve
            include_all_agents: If True and unified context available, get from all agents
                               If False or None, use original local history behavior
        
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        # If unified context is available and include_all_agents is True
        if self.unified_enabled and include_all_agents:
            try:
                unified_context = self.unified_context_manager.get_unified_context(
                    current_query="",  # Not query-specific, just getting recent
                    include_all_agents=True,
                    max_recent=n
                )
                
                # Convert ContextMessage objects to dict format matching original
                messages = []
                for msg in unified_context.recent_messages:
                    messages.append({
                        "role": "user",
                        "content": msg.user_query
                    })
                    messages.append({
                        "role": "assistant", 
                        "content": f"[{msg.agent_used}] {msg.response}"
                    })
                
                return messages[-n*2:]  # Return last n exchanges
                
            except Exception as e:
                print(f"⚠️ Failed to get unified context, falling back to local: {e}")
                return self.history[-n:]
        else:
            # Original behavior: return from local history
            return self.history[-n:]

    def get_context_summary(self, n=6, include_all_agents=False):
        """
        Get a formatted summary of recent context
        
        Args:
            n: Number of recent messages
            include_all_agents: Whether to include all agents' conversations
        
        Returns:
            Formatted string summary of conversation
        """
        recent = self.get_recent_context(n, include_all_agents=include_all_agents)
        
        lines = []
        for msg in recent:
            lines.append(f'{msg["role"].capitalize()}: {msg["content"]}')
        
        return "\n".join(lines)

    def get_unified_context_summary(self, current_query: str, max_messages: int = 10) -> str:
        """
        NEW METHOD: Get comprehensive context from all agents for current query
        
        This is the recommended method when you want full cross-agent awareness.
        
        Args:
            current_query: The current user query
            max_messages: Maximum messages to include in summary
            
        Returns:
            Formatted context string with semantic relevance and recency weighting
        """
        if not self.unified_enabled:
            # Fallback to regular context summary
            return self.get_context_summary(max_messages)
        
        try:
            unified_context = self.unified_context_manager.get_unified_context(
                current_query=current_query,
                include_all_agents=True,
                max_recent=max_messages,
                max_relevant=5
            )
            
            return unified_context.get_formatted_context(max_messages=max_messages)
            
        except Exception as e:
            print(f"⚠️ Failed to get unified context summary: {e}")
            return self.get_context_summary(max_messages)

    def get_cross_agent_data(self, key: str, agent_name: Optional[str] = None) -> Optional[Any]:
        """
        NEW METHOD: Retrieve data shared by other agents
        
        Args:
            key: Data key to retrieve
            agent_name: Optional specific agent to get data from
            
        Returns:
            Shared data value or None
        """
        if not self.unified_enabled:
            return None
        
        return self.unified_context_manager.get_shared_data(key, agent_name)

    def share_data_with_agents(self, agent_name: str, key: str, value: Any):
        """
        NEW METHOD: Share data with other agents
        
        Example:
            conv_manager.share_data_with_agents("nl2sql", "po_numbers", ["PO123", "PO456"])
        
        Args:
            agent_name: Name of agent sharing the data
            key: Data key
            value: Data value
        """
        if not self.unified_enabled:
            print(f"⚠️ Cannot share data - unified context not enabled")
            return
        
        self.unified_context_manager.share_data(agent_name, key, value)

    def detect_cross_agent_reference(self, current_query: str) -> Optional[Dict[str, Any]]:
        """
        NEW METHOD: Detect if current query references another agent's conversation
        
        Args:
            current_query: The current user query
            
        Returns:
            Dict with reference info or None if no cross-reference detected
        """
        if not self.unified_enabled:
            return None
        
        return self.unified_context_manager.detect_cross_agent_reference(current_query)

    def get_agent_context(self, agent_name: str, current_query: str) -> str:
        """
        NEW METHOD: Get context formatted specifically for a given agent
        
        Args:
            agent_name: Name of the agent requesting context
            current_query: The current query
            
        Returns:
            Formatted context string for LLM consumption
        """
        if not self.unified_enabled:
            return self.get_context_summary(6)
        
        return self.unified_context_manager.get_context_for_agent(
            agent_name=agent_name,
            current_query=current_query,
            include_other_agents=True
        )

    def _get_embedding(self, text):
        """Get embedding for text (original method, unchanged)"""
        response = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )
        return np.array(response["embedding"])

    def extract_query_entities(self, query):
        """Extract entities from query using the utility function (original method, unchanged)"""
        return extract_query_entities(
            query, 
            TABLE_NAMES, 
            KEYWORD_TABLE_MAP, 
            COLUMN_EQUIVALENCE
        )

    # ==================== Additional Helper Methods ====================

    def get_last_agent_used(self) -> Optional[str]:
        """NEW METHOD: Get the name of the last agent that processed a query"""
        if not self.unified_enabled:
            return None
        
        try:
            context = self.unified_context_manager.get_unified_context(
                current_query="",
                max_recent=1
            )
            return context.last_agent
        except:
            return None

    def get_agent_statistics(self) -> Dict[str, Any]:
        """NEW METHOD: Get statistics about agent usage in this session"""
        if not self.unified_enabled:
            return {"unified_context": False}
        
        return self.unified_context_manager.get_stats()

    def is_follow_up_from_different_agent(self, current_query: str) -> bool:
        """
        NEW METHOD: Check if current query is a follow-up to a different agent
        
        Returns True if:
        - Query seems like a follow-up (uses "that", "those", etc.)
        - Last agent was different from current routing
        """
        if not self.unified_enabled:
            return False
        
        cross_ref = self.detect_cross_agent_reference(current_query)
        return cross_ref is not None


# ==================== Usage Examples ====================

"""
EXAMPLE 1: Using in existing code (backward compatible)
--------------------------------------------------------

# Your existing code works exactly as before:
conv_manager = ConversationManager(session_id, query_store)
conv_manager.add_message("user", "show me sales")
recent = conv_manager.get_recent_context(6)
summary = conv_manager.get_context_summary(6)

# Everything still works!
"""

"""
EXAMPLE 2: Using with unified context (enhanced)
-------------------------------------------------

# Initialize with unified context manager:
unified_mgr = UnifiedContextManager(query_store, session_id)
conv_manager = ConversationManager(session_id, query_store, unified_mgr)

# Now you can use enhanced features:
conv_manager.add_message("user", "show me sales")

# Get context from ALL agents:
context = conv_manager.get_recent_context(6, include_all_agents=True)

# Get intelligent context summary for current query:
summary = conv_manager.get_unified_context_summary("what was that number?")

# Share data with other agents:
conv_manager.share_data_with_agents("nl2sql", "po_numbers", ["PO123", "PO456"])

# Get data from other agents:
po_numbers = conv_manager.get_cross_agent_data("po_numbers", "nl2sql")

# Detect cross-agent references:
cross_ref = conv_manager.detect_cross_agent_reference("remind me about that")
if cross_ref:
    print(f"User is referencing {cross_ref['referenced_agent']}")
"""

"""
EXAMPLE 3: Integration in NL2SQL agent
---------------------------------------

def generate_sql_chain_of_thought(user_query, conv_manager, query_store, 
                                   selected_tables, selected_columns, session_state):
    
    # Check if unified context is available
    if conv_manager.unified_enabled:
        # Get full context from ALL agents
        context_summary = conv_manager.get_unified_context_summary(
            current_query=user_query,
            max_messages=10
        )
        
        # Check for cross-agent references
        cross_ref = conv_manager.detect_cross_agent_reference(user_query)
        if cross_ref:
            # User is referencing something from another agent
            ref_agent = cross_ref['referenced_agent']
            ref_msg = cross_ref['referenced_message']
            
            # Enhance query with that context
            user_query = f'''
            Previous context from {ref_agent}:
            User asked: {ref_msg.user_query}
            Response: {ref_msg.response}
            
            Current question: {user_query}
            '''
    else:
        # Use original behavior
        context_summary = conv_manager.get_context_summary(6)
    
    # Build prompt with context
    prompt = f'''
    Conversation context:
    {context_summary}
    
    Current query: {user_query}
    Tables: {selected_tables}
    Columns: {selected_columns}
    
    Generate SQL for the current query.
    '''
    
    # ... rest of SQL generation ...
"""

"""
MIGRATION NOTES
===============

1. This is a DROP-IN REPLACEMENT for your existing conversation_manager.py
   
2. Backward compatible: If you don't pass unified_context_manager, 
   it works exactly like before
   
3. To enable unified context, update your initialize_session_state():
   
   from unified_context_manager import UnifiedContextManager
   
   unified_mgr = UnifiedContextManager(query_store, session_id)
   conv_manager = ConversationManager(session_id, query_store, unified_mgr)

4. Gradually adopt new methods:
   - Start using get_unified_context_summary() instead of get_context_summary()
   - Add share_data_with_agents() calls when agents find useful data
   - Use detect_cross_agent_reference() to handle follow-ups intelligently
   
5. Performance: The UnifiedContextManager has smart caching, so don't worry
   about calling these methods frequently.
"""

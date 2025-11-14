"""
Enhanced Query Router with semantic examples passed to the LLM for final decision.
FIXED VERSION: Eliminates LLM call duplication by combining routing + conversational response.

This version:
- Retrieves top-k semantically similar routing examples from pgvector
- Passes those examples + lightweight context to the LLM
- Lets the LLM choose the agent (nl2sql | personal_assistant | general_chat)
- FIXED: Contextual router now generates both routing decision AND conversational response in one call
- Keeps analytics, follow-up handling, and explanation output compatible with existing UI

Dependencies:
- config.ROUTING_MODEL (e.g., "gemma2:9b" or similar available in your Ollama)
- QueryStore with search_routing_examples() and get_routing_scores()
"""
from __future__ import annotations

import re
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import ollama

from config import ROUTING_MODEL
from nlp_utils import detect_universal_follow_up
from query_store import QueryStore
from contextual_engineering import ContextualEngineeringManager, UserProfile, ConversationContext
from unified_context_manager import UnifiedContextManager


class EnhancedQueryRouter:
    """
    Router that uses semantic search over curated routing examples and lets the LLM decide.
    """

    def __init__(self, query_store: Optional[QueryStore] = None, *, 
                 top_k_examples: int = 8, 
                 min_similarity: float = 0.25,
                 unified_context_manager=None):  # ADD THIS PARAMETER
        self.query_store = query_store
        self.vector_routing_enabled = query_store is not None
        self.top_k_examples = top_k_examples
        self.min_similarity = min_similarity
        self.unified_context_manager = unified_context_manager  # ADD THIS LINE

        # History and context
        self.routing_history: List[Dict[str, Any]] = []
        self.follow_up_context: Dict[str, Any] = {
            "is_active": False,
            "last_query": "",
            "last_agent": None,
            "last_results": None,
            "context_data": {},
            "timestamp": None,
            "session_id": None,
        }
        self.context_timeout = 600  # seconds
        
        # ADD THIS BLOCK
        if self.unified_context_manager:
            print("✅ Router: Unified context awareness ENABLED")
        else:
            print("⚠️ Router: Operating without unified context")

        if self.vector_routing_enabled:
            try:
                self.query_store.initialize_routing_examples()
                print("✅ Router: semantic example store ready")
            except Exception as e:
                print(f"⚠️ Router could not initialize routing examples: {e}")
                self.vector_routing_enabled = False
        else:
            print("⚠️ Router initialized without QueryStore; routing will degrade to pattern fallback.")


    # ------------ Public API ------------

    def route_query(self, user_query: str, session_state: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Route a user's query using semantic examples and unified context
        """
        q = (user_query or "").strip()
        ql = q.lower()

        # 1) lightweight context
        context_info = self._analyze_context(q, session_state)
        follow_up_info = self._detect_enhanced_follow_up(q, session_state)

        # === ADD THIS BLOCK: Get unified context ===
        unified_context_summary = ""
        cross_agent_hint = None
        
        if self.unified_context_manager:
            try:
                # Get recent conversation context across all agents
                unified_ctx = self.unified_context_manager.get_unified_context(
                    current_query=q,
                    include_all_agents=True,
                    max_recent=3,
                    max_relevant=2
                )
                
                # Build summary for routing decision
                if unified_ctx.last_agent:
                    unified_context_summary = f"Last agent: {unified_ctx.last_agent}"
                    
                    # Check if query might be following up on another agent
                    cross_ref = self.unified_context_manager.detect_cross_agent_reference(q)
                    if cross_ref:
                        ref_agent = cross_ref['referenced_agent']
                        ref_type = cross_ref['reference_type']
                        
                        cross_agent_hint = {
                            "referenced_agent": ref_agent,
                            "reference_type": ref_type,
                            "message": cross_ref['referenced_message']
                        }
                        
                        unified_context_summary += f" (Follow-up to {ref_agent}: {ref_type})"
                        print(f"🔗 Router detected cross-agent reference to {ref_agent}")
                
                # Get agent usage stats
                if unified_ctx.agent_history:
                    top_agents = sorted(
                        unified_ctx.agent_history.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:2]
                    unified_context_summary += f" | Recent agents: {', '.join([a[0] for a in top_agents])}"
                
                print(f"✅ Router: Using unified context - {unified_context_summary}")
                
            except Exception as e:
                print(f"⚠️ Router: Failed to get unified context: {e}")
        # === END BLOCK ===

        # 2) fetch semantic examples (for prompt)
        similar_examples: List[Tuple[str, str, float, float]] = []
        vector_scores: Dict[str, float] = {
            "nl2sql": 0.0, 
            "api_endpoint": 0.0, 
            "personal_assistant": 0.0, 
            "general_chat": 0.0,
            "web_search": 0.0,  # ADD if not present
            "document_rag": 0.0  # ADD if not present
        }
        if self.vector_routing_enabled:
            try:
                similar_examples = self.query_store.search_routing_examples(
                    q, top_k=self.top_k_examples, min_similarity=self.min_similarity
                ) or []
                vector_scores = self.query_store.get_routing_scores(q, top_k=max(10, self.top_k_examples))
            except Exception as e:
                print(f"Vector search failed: {e}")

        # 3) LLM decision using examples AND unified context
        agent_name, band, conf, reason = self._llm_decide_with_examples(
            user_query=q,
            examples=similar_examples,
            context_info=context_info,
            follow_up_info=follow_up_info,
            vector_scores=vector_scores,
            unified_context=unified_context_summary,  # PASS THIS
            cross_agent_hint=cross_agent_hint,  # PASS THIS
        )
        try:
            # Only bother if the LLM didn't already ask for API
            if agent_name in ("nl2sql", "general_chat", "personal_assistant", None):
                from api_agent_multi_endpoint import MultiEndpointAPIAgent
                api_probe = MultiEndpointAPIAgent(
                    query_store=self.query_store,
                    unified_context_manager=self.unified_context_manager
                )
                _use_api, _api_reason = api_probe._should_use_api(q)
                if _use_api:
                    agent_name = "api_endpoint"  # <- your router's name for the API agent
                    # Bump confidence a bit and record why
                    reason = f"[api_first override] {_api_reason}; {reason}"
                    band = band or "override"
                    conf = max(conf or 0.0, 0.66)
                    # We'll also drop a flag into routing_info a few lines later
                    _api_override_info = {"override": "api_first", "override_reason": _api_reason}
            else:
                _api_override_info = None
        except Exception as _e:
            _api_override_info = None
        routing_info: Dict[str, Any] = {
            "method": "llm_semantic_examples_unified",  # UPDATED METHOD NAME
            "confidence": conf,
            "confidence_band": band,
            "scores": vector_scores,
            "context": context_info,
            "follow_up_context": follow_up_info,
            "unified_context": unified_context_summary,  # ADD THIS
            "cross_agent_hint": cross_agent_hint,  # ADD THIS
            "similar_examples": [
                {"query_text": ex[0], "agent": ex[1], "confidence": float(ex[2]), "similarity": float(ex[3])}
                for ex in (similar_examples or [])
            ],
            "llm_reasoning": reason,
            "selected_agent": agent_name,
        }
        if '_api_override_info' in locals() and _api_override_info:
            routing_info.update(_api_override_info)
            routing_info["api_first"] = True
            routing_info["selected_agent"] = "api_endpoint"
        self._update_learning(user_query=q, selected_agent=agent_name, routing_info=routing_info, session_state=session_state)
        return agent_name, routing_info

    def explain_routing(self, user_query: str, agent_name: str, routing_info: Dict[str, Any]) -> str:
        base = {
            "nl2sql": "📊 Routing to Data Analyst",
            "api_endpoint": "🔌 Routing to API Agent",
            "personal_assistant": "🗓️ Routing to Personal Assistant",
            "general_chat": "💬 Routing to General Chat",
        }.get(agent_name, f"🤔 Routing to {agent_name}")

        method = routing_info.get("method", "unknown")
        conf = routing_info.get("confidence", 0.0)
        band = routing_info.get("confidence_band")

        if method == "llm_semantic_examples":
            return f"{base} — LLM decision with semantic examples ({band or 'n/a'}, {conf:.2f})."
        elif method.startswith("llm"):
            return f"{base} — LLM decision ({conf:.2f})."
        else:
            return f"{base} — Confidence: {conf:.2f}"

    def get_routing_analytics(self) -> Dict[str, Any]:
        analytics: Dict[str, Any] = {
            "total_routes": len(self.routing_history),
            "agent_distribution": {},
            "average_confidence": 0.0,
            "recent_performance": self._analyze_recent_performance(),
        }
        if self.routing_history:
            counts: Dict[str, int] = {}
            confs = 0.0
            for r in self.routing_history:
                counts[r["selected_agent"]] = counts.get(r["selected_agent"], 0) + 1
                confs += r.get("confidence", 0.0)
            analytics["agent_distribution"] = counts
            analytics["average_confidence"] = confs / len(self.routing_history)

        # add vector-store stats if available
        if self.vector_routing_enabled:
            try:
                analytics.update(self.query_store.get_routing_stats())
            except Exception as e:
                analytics["vector_routing_error"] = str(e)
        return analytics

    # ------------ Internals ------------

    def _analyze_context(self, user_query: str, session_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        ctx = {
            "time_sensitive": False,
            "requires_data": False,
            "requires_action": False,
            "user_state": {},
        }
        ql = user_query.lower()

        # Heuristics
        time_indicators = ["yesterday", "today", "now", "current", "latest", "recent", "this week", "last week"]
        business_terms = ["sales", "orders", "shipments", "inventory", "containers", "po", "so", "picktix", "outbound"]
        ctx["time_sensitive"] = any(t in ql for t in time_indicators) and any(b in ql for b in business_terms)

        data_ind = ["show", "display", "how many", "count", "total", "list", "find", "get", "which"]
        data_terms = ["orders", "inventory", "products", "shipments", "sales", "po", "so"]
        ctx["requires_data"] = any(d in ql for d in data_ind) and any(t in ql for t in data_terms)

        act_ind = ["add", "create", "schedule", "remind", "complete", "delete", "update", "notify"]
        act_obj = ["task", "reminder", "appointment", "meeting", "todo"]
        ctx["requires_action"] = any(d in ql for d in act_ind) and any(o in ql for o in act_obj)

        # Optional per-user state
        try:
            pending_tasks = 0
            if session_state and 'pa_tasks' in session_state:
                pending_tasks = len([t for t in session_state['pa_tasks'] if t.get('status') == 'pending'])
            ctx["user_state"] = {"pending_tasks": pending_tasks}
        except Exception:
            ctx["user_state"] = {"pending_tasks": 0}
        return ctx

    def _detect_enhanced_follow_up(self, user_query: str, session_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        self._update_session_context(session_state)
        # universal follow-up cues from your utils (if present)
        try:
            if detect_universal_follow_up(user_query):
                return {"is_follow_up": True, "last_agent": self.follow_up_context.get("last_agent"), "cross_agent_intent": False}
        except Exception:
            pass

        # simple pronoun/ellipsis detection
        ql = user_query.lower()
        pronouns = ["those", "these", "that", "this", "them", "it", "above", "previous"]
        is_follow_up = any(p in ql for p in pronouns)
        return {"is_follow_up": bool(is_follow_up), "last_agent": self.follow_up_context.get("last_agent"), "cross_agent_intent": False}

    def _update_session_context(self, session_state: Optional[Dict[str, Any]]):
        if not session_state:
            return
        sid = session_state.get("session_id")
        if self.follow_up_context.get("session_id") != sid:
            self.follow_up_context["session_id"] = sid

    def update_follow_up_context(self, user_query: str, agent_name: str, result: Dict[str, Any]):
        if result.get("success"):
            self.follow_up_context.update({
                "is_active": True,
                "last_query": user_query,
                "last_agent": agent_name,
                "last_results": result.get("results"),
                "context_data": {"sql": result.get("sql"), "type": result.get("type"), "message": (result.get("message") or "")[:100]},
                "timestamp": datetime.now()
            })

    def _is_context_valid(self) -> bool:
        if not self.follow_up_context["is_active"] or not self.follow_up_context["timestamp"]:
            return False
        age = (datetime.now() - self.follow_up_context["timestamp"]).total_seconds()
        if age > self.context_timeout:
            self._clear_context()
            return False
        return True

    def _clear_context(self):
        self.follow_up_context.update({
            "is_active": False, "last_query": "", "last_agent": None,
            "last_results": None, "context_data": {}, "timestamp": None
        })

    # ----- LLM decision using semantic examples -----

    def _llm_decide_with_examples(
        self,
        user_query: str,
        examples: List,
        context_info: Dict,
        follow_up_info: Dict,
        vector_scores: Dict,
        unified_context: str = "",  # ADD THIS
        cross_agent_hint: Optional[Dict] = None,  # ADD THIS
    ) -> Tuple[str, str, float, str]:
        """
        Use LLM to decide routing with semantic examples and unified context
        """
        
        # Build examples section
        examples_section = ""
        if examples:
            examples_section = "SIMILAR PAST QUERIES (for reference):\n"
            for i, (query_text, agent, conf, sim) in enumerate(examples[:5], 1):
                examples_section += f"{i}. Query: '{query_text}' → Agent: {agent} (similarity: {sim:.2f})\n"
            examples_section += "\n"
        
        # Build context section
        context_section = ""
        if follow_up_info.get("is_follow_up"):
            last_agent = follow_up_info.get("last_agent", "unknown")
            context_section += f"NOTE: This appears to be a follow-up query. Previous agent: {last_agent}\n\n"
        
        # === ADD THIS BLOCK: Unified context section ===
        unified_context_section = ""
        if unified_context:
            unified_context_section = f"""
    CONVERSATION CONTEXT (from all agents):
    {unified_context}

    """
        
        cross_agent_section = ""
        if cross_agent_hint:
            ref_agent = cross_agent_hint['referenced_agent']
            ref_type = cross_agent_hint['reference_type']
            ref_msg = cross_agent_hint['message']
            
            cross_agent_section = f"""
    ⚠️ CROSS-AGENT REFERENCE DETECTED:
    User is {ref_type} to {ref_agent} agent's previous response.
    Previous query: "{ref_msg.user_query}"
    Previous response preview: "{ref_msg.response[:150]}..."

    ROUTING RECOMMENDATION: Consider routing to general_chat to synthesize/explain the {ref_agent} results,
    unless the user explicitly wants new data (which would need the appropriate specialist agent).

    """
        # === END BLOCK ===
        
        # Build the routing prompt
        routing_prompt = f"""You are an intelligent query router for a multi-agent system.

    {unified_context_section}{cross_agent_section}{context_section}{examples_section}

    CURRENT USER QUERY: "{user_query}"

    AVAILABLE AGENTS:
    1. nl2sql - For database queries (sales data, inventory, customers, orders, etc.)
    2. web_search - For searching the internet for current information
    3. document_rag - For searching uploaded documents and PDFs
    4. personal_assistant - For tasks, reminders, scheduling, notes
    5. general_chat - For explanations, analysis, synthesis, and general conversation
    6. api_endpoint - For API calls to external services

    ROUTING RULES:
    - If user wants DATA from database → nl2sql
    - If user wants to SEARCH the web → web_search  
    - If user wants to SEARCH documents → document_rag
    - If user wants to CREATE/MANAGE tasks/events → personal_assistant
    - If user wants EXPLANATION/ANALYSIS of previous results → general_chat
    - If user is FOLLOWING UP on previous results → general_chat (unless they explicitly want new data)
    - If user asks "tell me more", "explain that", "what does that mean" → general_chat
    - If user wants API data → api_endpoint
    - For general questions, greetings, casual chat → general_chat

    {f'''
    IMPORTANT: {cross_agent_section}
    ''' if cross_agent_hint else ''}

    Respond with JSON only:
    {{
        "agent": "nl2sql|web_search|document_rag|personal_assistant|general_chat|api_endpoint",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation"
    }}
    """
        
        try:
            response = ollama.chat(
                model=ROUTING_MODEL,
                messages=[
                    {"role": "system", "content": "You are a routing expert. Always respond with valid JSON only."},
                    {"role": "user", "content": routing_prompt}
                ],
                format="json",
                options={"temperature": 0.1}
            )
            
            content = response.get("message", {}).get("content", "{}")
            
            # Parse JSON
            import json
            decision = json.loads(content)
            
            agent = decision.get("agent", "general_chat")
            confidence = float(decision.get("confidence", 0.5))
            reasoning = decision.get("reasoning", "No reasoning provided")
            
            # Validate agent name
            valid_agents = ["nl2sql", "web_search", "document_rag", "personal_assistant", "general_chat", "api_endpoint"]
            if agent not in valid_agents:
                print(f"⚠️ Invalid agent '{agent}', defaulting to general_chat")
                agent = "general_chat"
                confidence = 0.5
            
            # Determine confidence band
            if confidence >= 0.8:
                band = "high"
            elif confidence >= 0.5:
                band = "medium"
            else:
                band = "low"
            
            print(f"🎯 Router decision: {agent} (confidence: {confidence:.2f}, {band})")
            if cross_agent_hint:
                print(f"   └─ Cross-agent hint influenced routing")
            
            return agent, band, confidence, reasoning
            
        except Exception as e:
            print(f"⚠️ LLM routing failed: {e}")
            # Fallback to pattern-based routing
            return self._fallback_pattern_routing(user_query), "low", 0.3, f"Fallback due to error: {e}"

    # -------- Compatibility helper methods (kept to avoid breaking callers) --------

    def _classify_intent_enhanced(self, query_lower: str, context_info: Dict[str, Any],
                                  follow_up_info: Dict[str, Any], user_query: str) -> Dict[str, float]:
        """Return vector-based scores for compatibility; actual routing is LLM-driven."""
        try:
            if self.vector_routing_enabled:
                return self.query_store.get_routing_scores(user_query, top_k=15)
        except Exception:
            pass
        return {"nl2sql": 0.33, "personal_assistant": 0.33, "general_chat": 0.34}

    def _calibrate_confidence(self, intent_scores: Dict[str, float], context_info: Dict[str, Any],
                              follow_up_info: Dict[str, Any]) -> Dict[str, float]:
        return intent_scores

    def _select_agent(self, calibrated_scores: Dict[str, float], user_query: str,
                      context_info: Dict[str, Any], follow_up_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Compatibility shim that simply calls the new LLM-based decision."""
        examples = self.query_store.search_routing_examples(user_query, top_k=self.top_k_examples, min_similarity=self.min_similarity) if self.vector_routing_enabled else []
        vector_scores = self.query_store.get_routing_scores(user_query, top_k=max(10, self.top_k_examples)) if self.vector_routing_enabled else {"nl2sql": 0, "personal_assistant": 0, "general_chat": 0}
        agent, band, conf, reason = self._llm_decide_with_examples(user_query, examples, context_info, follow_up_info, vector_scores)
        info = {
            "method": "llm_semantic_examples",
            "confidence": conf,
            "confidence_band": band,
            "scores": vector_scores,
            "llm_reasoning": reason,
        }
        return agent, info

    def _llm_route_with_context(self, user_query: str, context_info: Dict[str, Any],
                                follow_up_info: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Legacy API: decide using examples + context; returns (agent, info)"""
        examples = self.query_store.search_routing_examples(user_query, top_k=self.top_k_examples, min_similarity=self.min_similarity) if self.vector_routing_enabled else []
        vector_scores = self.query_store.get_routing_scores(user_query, top_k=max(10, self.top_k_examples)) if self.vector_routing_enabled else {"nl2sql": 0, "personal_assistant": 0, "general_chat": 0}
        agent, band, conf, reason = self._llm_decide_with_examples(user_query, examples, context_info, follow_up_info, vector_scores)
        return agent, {"llm_confidence": conf, "llm_reasoning": reason, "band": band, "method": "llm_semantic_examples"}

    def _build_context_description(self, context_info: Dict[str, Any], follow_up_info: Dict[str, Any]) -> str:
        parts = []
        if context_info.get("requires_data"): parts.append("requires data")
        if context_info.get("requires_action"): parts.append("requires action")
        if context_info.get("time_sensitive"): parts.append("time-sensitive")
        if follow_up_info.get("is_follow_up"): parts.append("follow-up")
        return "; ".join(parts) if parts else "none"

    def debug_route_query(self, user_query: str):
        """Print semantic examples and LLM decision for debugging."""
        ctx = self._analyze_context(user_query, {})
        fu = self._detect_enhanced_follow_up(user_query, {})
        examples = self.vector_routing_enabled and self.query_store.search_routing_examples(user_query, top_k=self.top_k_examples, min_similarity=self.min_similarity) or []
        print("---- DEBUG ROUTE ----")
        print("Query:", user_query)
        print("Context:", ctx)
        print("Follow-up:", fu)
        print("Examples:")
        for ex in examples:
            print(f"  - [{ex[1]}] sim={float(ex[3]):.3f} conf={float(ex[2]):.2f} :: {ex[0]}")
        agent, band, conf, reason = self._llm_decide_with_examples(user_query, examples, ctx, fu, {"nl2sql":0,"personal_assistant":0,"general_chat":0})
        print("Decision:", agent, band, conf, reason)

    def _classify_query_type(self, user_query: str) -> str:
        q = user_query.lower()
        if any(k in q for k in ["how many", "list", "show", "count"]): return "data"
        if any(k in q for k in ["remind", "schedule", "task", "todo"]): return "assistant"
        return "chat"

    def _get_user_preference(self, session_state: Optional[Dict[str, Any]]) -> Optional[str]:
        return None

    def _get_recent_agents(self, n: int = 5) -> List[str]:
        return [r["selected_agent"] for r in self.routing_history[-n:]]

    def get_context_status(self) -> Dict[str, Any]:
        """Return status with the 'active' key (backward compatible)."""
        valid = self._is_context_valid()
        if valid:
            age = (datetime.now() - self.follow_up_context["timestamp"]).total_seconds()
            return {
                "active": True,                           # ← keep this name
                "is_active": True,                        # backward compat (optional)
                "last_query": (self.follow_up_context.get("last_query") or "")[:50] + "...",
                "last_agent": self.follow_up_context.get("last_agent"),
                "age_seconds": round(age, 1),
                "expires_in": round(self.context_timeout - age, 1),
            }
        return {"active": False, "is_active": False, "reason": "No valid context"}

    # ------------ Analytics & helpers ------------

    def _analyze_recent_performance(self, window: int = 20) -> Dict[str, Any]:
        recent = self.routing_history[-window:]
        if not recent:
            return {"message": "No recent history"}
        hi = sum(1 for r in recent if r.get("confidence", 0) >= 0.7)
        llm = sum(1 for r in recent if "llm" in (r.get("method") or ""))
        return {
            "high_confidence_rate": round(hi / len(recent), 3),
            "llm_routing_rate": round(llm / len(recent), 3),
        }

    def _update_learning(self, user_query: str, selected_agent: str, 
                        routing_info: Dict[str, Any], session_state: Optional[Dict[str, Any]] = None):
        """Update routing history with unified context info"""
        
        entry = {
            "query": user_query,
            "selected_agent": selected_agent,
            "confidence": routing_info.get("confidence", 0.0),
            "method": routing_info.get("method", "unknown"),
            "timestamp": datetime.now(),
            "had_unified_context": bool(routing_info.get("unified_context")),  # ADD THIS
            "had_cross_agent_hint": bool(routing_info.get("cross_agent_hint")),  # ADD THIS
        }
        
        self.routing_history.append(entry)
        
        # Update follow-up context
        self.follow_up_context.update({
            "is_active": True,
            "last_query": user_query,
            "last_agent": selected_agent,
            "timestamp": datetime.now(),
            "session_id": session_state.get("session_id") if session_state else None,
        })
    def get_unified_context_routing_stats(self) -> Dict[str, Any]:
        """Get statistics about how unified context affects routing"""
        
        if not self.routing_history:
            return {
                "total_routes": 0,
                "with_unified_context": 0,
                "with_cross_agent_hints": 0,
                "percentage_using_context": 0.0
            }
        
        total = len(self.routing_history)
        with_context = sum(1 for r in self.routing_history if r.get("had_unified_context", False))
        with_hints = sum(1 for r in self.routing_history if r.get("had_cross_agent_hint", False))
        
        return {
            "total_routes": total,
            "with_unified_context": with_context,
            "with_cross_agent_hints": with_hints,
            "percentage_using_context": (with_context / total * 100) if total > 0 else 0.0,
            "percentage_with_hints": (with_hints / total * 100) if total > 0 else 0.0,
        }

    # Deprecated but referenced in some debuggers
    def debug_vector_routing(self, query: str):
        if not self.vector_routing_enabled:
            print("Vector routing disabled.")
            return
        try:
            res = self.query_store.search_routing_examples(query, top_k=10, min_similarity=0.0) or []
            print(f"Top vector matches for: {query}")
            for i, (qt, agent, conf, sim) in enumerate(res, 1):
                qs = qt[:60] + ("..." if len(qt) > 60 else "")
                print(f"{i:2d}. [{agent}] sim={float(sim):.3f} conf={float(conf):.2f} :: {qs}")
        except Exception as e:
            print(f"debug_vector_routing error: {e}")

    def check_vector_routing_health(self) -> Dict[str, Any]:
        status = {"vector_enabled": self.vector_routing_enabled}
        return status
    def _detect_api_query_signals(self, user_query: str) -> float:
        """
        Detect if query is better suited for API than SQL
        
        Returns:
            confidence score (0.0 to 1.0) indicating API suitability
            
        Logic:
            - Checks for API-specific keywords and patterns
            - Returns high confidence (0.8+) for strong API queries
            - Returns medium confidence (0.5-0.7) for potential API queries
            - Reduces confidence if SQL indicators are present
        """
        query_lower = user_query.lower()
        
        # Strong API signals (0.8-1.0 confidence)
        strong_api_keywords = [
            'sales order', 'ship', 'shipped', 'shipping', 'track',
            'carrier', 'crew', 'assigned to', 'customer name',
            'po number', 'purchase order', 'order status',
            'shipment', 'delivery', 'tracking number'
        ]
        
        # Medium API signals (0.5-0.7 confidence)
        medium_api_keywords = [
            'today', 'now', 'current', 'active', 'ready',
            'unshipped', 'in production', 'loading',
            'order', 'orders', 'customer', 'customers'
        ]
        
        # Specific customer names (API handles better)
        customer_indicators = [
            'amazon', 'walmart', 'target', 'kcc', 'fns',
            'costco', 'sams club', 'kroger'
        ]
        
        # Crew indicators (API is authoritative)
        crew_indicators = [
            'crew t1', 'crew t2', 'crew t3', 
            't1 ', 't2 ', 't3 ',
            'crew ', 'team '
        ]
        
        # Calculate confidence
        confidence = 0.0
        
        # Check strong indicators
        for keyword in strong_api_keywords:
            if keyword in query_lower:
                confidence = max(confidence, 0.85)
        
        # Check medium indicators
        for keyword in medium_api_keywords:
            if keyword in query_lower:
                confidence = max(confidence, 0.65)
        
        # Check customer indicators
        for customer in customer_indicators:
            if customer in query_lower:
                confidence = max(confidence, 0.75)
        
        # Check crew indicators
        for crew in crew_indicators:
            if crew in query_lower:
                confidence = max(confidence, 0.80)
        
        # Anti-patterns (prefer SQL for these)
        sql_strong_indicators = [
            'how many', 'count', 'sum', 'total', 'average',
            'group by', 'each month', 'breakdown', 'report',
            'container', 'pallet', 'item number', 'upc',
            'expiration', 'received from', 'unload',
            'aggregate', 'analysis', 'trend', 'historical'
        ]
        
        # Reduce confidence if SQL is clearly better
        for indicator in sql_strong_indicators:
            if indicator in query_lower:
                confidence *= 0.4  # Significantly reduce API confidence
                break  # One indicator is enough
        
        return confidence


    def route_query_with_api_priority(
        self, 
        user_query: str, 
        session_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced routing with API-first logic for sales order queries
        
        Flow:
            1. Detect API signals in query
            2. If strong API signal (>0.75) → route to api_endpoint
            3. Otherwise → use normal LLM-based routing
            
        This ensures sales order queries go to API before SQL
        
        Args:
            user_query: The user's query string
            session_state: Optional session state dictionary
            
        Returns:
            Tuple of (agent_name, routing_info)
        """
        
        # Detect API signals
        api_confidence = self._detect_api_query_signals(user_query)
        
        # If strong API signal, route to API first
        if api_confidence > 0.65:
            routing_info = {
                "method": "api_priority",
                "confidence": api_confidence,
                "confidence_band": "high" if api_confidence > 0.80 else "medium",
                "api_first": True,
                "scores": {
                    "api_endpoint": api_confidence,
                    "nl2sql": api_confidence * 0.5,  # Lower SQL score
                    "personal_assistant": 0.1,
                    "general_chat": 0.1
                },
                "reasoning": (
                    f"Query contains strong API indicators (confidence: {api_confidence:.2f}). "
                    "Routing to API endpoint for sales order/shipping data."
                ),
                "similar_examples": [],
                "context": {},
                "follow_up_context": {}
            }
            
            # Update learning
            self._update_learning(
                user_query=user_query,
                selected_agent="api_endpoint",
                routing_info=routing_info,
                session_state=session_state
            )
            
            print(f"🎯 API Priority Routing: {api_confidence:.2f} confidence")
            return "api_endpoint", routing_info
        
        # Otherwise, use normal routing (with vector search + LLM)
        print(f"📊 Normal Routing: API confidence too low ({api_confidence:.2f})")
        return self.route_query(user_query, session_state)


    def route_query_with_smart_fallback(
        self,
        user_query: str,
        session_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Route with smart fallback logic
        
        Returns:
            Tuple of (primary_agent, fallback_agent, routing_info)
            
        This method identifies both the primary agent and a potential fallback,
        allowing the application to try both if the primary fails.
        """
        
        # Get API confidence
        api_confidence = self._detect_api_query_signals(user_query)
        
        # Route normally
        primary_agent, routing_info = self.route_query(user_query, session_state)
        
        # Determine fallback
        fallback_agent = None
        
        if primary_agent == "api_endpoint":
            # If API is primary, SQL is fallback
            if api_confidence < 0.90:  # Not 100% sure
                fallback_agent = "nl2sql"
        
        elif primary_agent == "nl2sql":
            # If SQL is primary but query has API signals, API is fallback
            if api_confidence > 0.50:
                fallback_agent = "api_endpoint"
        
        # Add fallback info to routing_info
        routing_info["fallback_agent"] = fallback_agent
        routing_info["api_confidence"] = api_confidence
        
        return primary_agent, fallback_agent, routing_info


    def explain_api_routing(self, user_query: str, routing_info: Dict[str, Any]) -> str:
        """
        Enhanced routing explanation that includes API priority logic
        
        Returns:
            Human-readable explanation of routing decision
        """
        method = routing_info.get("method", "unknown")
        confidence = routing_info.get("confidence", 0.0)
        api_confidence = routing_info.get("api_confidence", 0.0)
        
        if method == "api_priority":
            return (
                f"🔌 **API Priority Routing** (confidence: {confidence:.2f})\n"
                f"Query contains sales order/shipping indicators. "
                f"Routed to API endpoint for real-time data."
            )
        
        elif api_confidence > 0.5:
            return (
                f"📊 **Normal Routing** (confidence: {confidence:.2f})\n"
                f"API confidence: {api_confidence:.2f} - not strong enough for API priority. "
                f"Using LLM-based routing with vector search."
            )
        
        else:
            return self.explain_routing(user_query, routing_info.get("selected_agent", "unknown"), routing_info)

class ContextualEnhancedQueryRouter(EnhancedQueryRouter):
    """
    FIXED VERSION: Enhanced router with contextual engineering capabilities
    Combines routing decision + conversational response generation in ONE LLM call
    """
    
    def __init__(self, query_store=None, **kwargs):
        super().__init__(query_store, **kwargs)
        self.context_manager = ContextualEngineeringManager(query_store)
        print("✅ Contextual Engineering enabled - Optimized single LLM call mode")
    
    def route_query_contextual(self, user_query: str, session_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        FIXED: Enhanced routing with COMBINED routing + conversational response in ONE LLM call
        This eliminates the duplication issue by generating both outputs together.
        """
        
        # 1. Get or create user profile
        user_id = session_state.get("session_id", "anonymous")
        user_profile = self.context_manager.get_or_create_user_profile(user_id, session_state)
        
        # 2. Update conversation context
        conversation_context = self.context_manager.update_conversation_context(
            session_state.get("session_id", ""), user_query
        )
        
        # 3. Analyze contextual signals
        contextual_signals = self.context_manager.analyze_contextual_signals(user_query, session_state)
        
        # 4. Get semantic examples (your existing logic)
        similar_examples = []
        if self.vector_routing_enabled:
            try:
                similar_examples = self.query_store.search_routing_examples(
                    user_query, top_k=self.top_k_examples, min_similarity=self.min_similarity
                ) or []
            except Exception as e:
                print(f"Vector search failed: {e}")
        
        # 5. Get base context for enhanced decision
        base_context_info = self._analyze_context(user_query, session_state)
        follow_up_info = self._detect_enhanced_follow_up(user_query, session_state)
        
        # 6. FIXED: Single LLM call for BOTH routing AND conversational response
        agent_name, confidence, reasoning, conversational_response = self._combined_contextual_decision(
            user_query, similar_examples, user_profile, conversation_context, 
            contextual_signals, base_context_info, follow_up_info
        )
        
        # 7. Update user profile with routing decision
        self._update_user_profile(user_profile, agent_name, user_query)
        
        # 8. Get adaptive response style for downstream agents
        response_style = self.context_manager.get_adaptive_response_style(
            user_profile, conversation_context
        )
        
        # 9. FIXED: Build comprehensive routing info WITH conversational response
        routing_info = {
            "method": "contextual_engineering_combined",  # Updated method name
            "confidence": confidence,
            "llm_reasoning": reasoning,
            "conversational_response": conversational_response,  # CRITICAL: Add this
            "user_profile": {
                "role": user_profile.role,
                "experience_level": user_profile.experience_level,
                "department": user_profile.department
            },
            "conversation_context": {
                "topic": conversation_context.current_topic,
                "stage": conversation_context.conversation_stage,
                "urgency": conversation_context.urgency_level,
                "complexity": conversation_context.complexity_score
            },
            "contextual_signals": contextual_signals,
            "response_style": response_style,
            "similar_examples": [
                {"query": ex[0], "agent": ex[1], "similarity": float(ex[3])} 
                for ex in similar_examples[:3]
            ]
        }
        
        # 10. Update learning
        self._update_learning(user_query, agent_name, routing_info, session_state)
        
        return agent_name, routing_info
    
    def _combined_contextual_decision(self, user_query: str, similar_examples: List, 
                                    user_profile: UserProfile, conversation_context: ConversationContext,
                                    contextual_signals: Dict, base_context_info: Dict,
                                    follow_up_info: Dict) -> Tuple[str, float, str, str]:
        """
        FIXED: Single LLM call that returns BOTH routing decision AND conversational response
        This is the key fix that eliminates the duplication issue.
        
        Returns: (agent_name, confidence, reasoning, conversational_response)
        """
        
        # Build examples section for prompt
        examples_text = "No similar examples found"
        if similar_examples:
            examples_lines = []
            for i, (q_text, agent, conf, sim) in enumerate(similar_examples[:5], 1):
                q_trim = q_text.strip()[:80] + ("..." if len(q_text) > 80 else "")
                examples_lines.append(
                    f"{i}. [{agent}] \"{q_trim}\" (similarity: {float(sim):.3f})"
                )
            examples_text = "\\n".join(examples_lines)
        
        # Build context summary
        context_summary = f"""
User Profile: {user_profile.role} ({user_profile.experience_level}) in {user_profile.department}
Current Topic: {conversation_context.current_topic or 'general'}
Conversation Stage: {conversation_context.conversation_stage}
Urgency Level: {conversation_context.urgency_level}
Complexity Score: {conversation_context.complexity_score:.2f}
Preferred Detail Level: {user_profile.preferred_detail_level}
"""
        
        # Build contextual signals summary
        signals_summary = []
        if contextual_signals.get("time_sensitivity"):
            signals_summary.append("time-sensitive request")
        if contextual_signals.get("urgency_detected") != "normal":
            signals_summary.append(f"urgency: {contextual_signals.get('urgency_detected')}")
        if contextual_signals.get("technical_level") == "advanced":
            signals_summary.append("advanced technical query")
        if contextual_signals.get("business_domain") != "general":
            signals_summary.append(f"domain: {contextual_signals.get('business_domain')}")
        
        signals_text = "; ".join(signals_summary) if signals_summary else "standard request"
        
        # Build follow-up context
        followup_text = "new conversation"
        if follow_up_info.get("is_follow_up"):
            last_agent = follow_up_info.get("last_agent", "unknown")
            followup_text = f"follow-up to {last_agent} interaction"
        
        # FIXED: Combined prompt for BOTH routing AND conversational response
        combined_prompt = f"""You are an expert AI system that provides BOTH routing decisions AND conversational responses in a single interaction.

CONTEXT ANALYSIS:
{context_summary}

USER QUERY: "{user_query}"

CONTEXTUAL SIGNALS: {signals_text}
CONVERSATION FLOW: {followup_text}

SIMILAR EXAMPLES (semantic matches):
{examples_text}

AGENTS AVAILABLE:
- nl2sql: Database queries, warehouse data analysis, reporting, SQL operations
- personal_assistant: Task management, reminders, scheduling, to-dos, notifications
- general_chat: Greetings, explanations, general conversation, help requests
- document_rag: Document upload, file search, document-based questions, policy/manual queries

CONTEXTUAL ADAPTATION GUIDELINES:
- User Experience Level: {user_profile.experience_level} → Adapt complexity accordingly
- User Role: {user_profile.role} → Tailor response to their responsibilities  
- Urgency: {conversation_context.urgency_level} → Adjust tone and priority
- Detail Preference: {user_profile.preferred_detail_level} → Match information depth
- Conversation Stage: {conversation_context.conversation_stage} → Build on flow

DUAL TASK:
1. ROUTING DECISION: Choose the most appropriate agent based on query intent and context
2. CONVERSATIONAL RESPONSE: Generate a natural, contextually-aware response that:
   - Acknowledges the user's profile and current context
   - References conversation history when relevant
   - Uses appropriate tone for urgency level and experience
   - Builds on previous interactions naturally
   - Provides helpful, role-appropriate information

RESPONSE FORMAT (must follow exactly):
AGENT: agent_name
CONFIDENCE: high/medium/low
REASONING: [max 25 words explaining routing decision based on query intent and context]
RESPONSE: [natural conversational response that feels contextual and builds on our interaction history]

Consider the user's experience level, role, and current context when crafting both the routing decision and conversational response. Make the response feel natural and tailored to this specific user in this specific situation.

Provide your combined routing + conversational response now:"""
        
        try:
            response = ollama.chat(
                model=ROUTING_MODEL,  # Use consistent model from config
                messages=[
                    {"role": "system", "content": "You are an expert contextual router and conversational AI. Always follow the exact response format to provide both routing decisions and natural conversation."},
                    {"role": "user", "content": combined_prompt}
                ],
                stream=False,
                options={"temperature": 0.3, "num_ctx": 4096}  # Balanced temperature for consistency + creativity
            )
            
            response_text = response.get("message", {}).get("content", "").strip()
            
            # Parse the combined response
            agent = "general_chat"
            confidence = 0.5
            reasoning = "parsing_error"
            conversational_response = "I'm here to help! What would you like to work on?"
            
            # More robust parsing to handle multi-line responses
            lines = response_text.split('\n')
            current_section = None
            response_lines = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("AGENT:"):
                    agent = line.replace("AGENT:", "").strip().lower()
                    current_section = "agent"
                elif line.startswith("CONFIDENCE:"):
                    conf_text = line.replace("CONFIDENCE:", "").strip().lower()
                    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.5}
                    confidence = confidence_map.get(conf_text, 0.6)
                    current_section = "confidence"
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
                    current_section = "reasoning"
                elif line.startswith("RESPONSE:"):
                    conversational_response = line.replace("RESPONSE:", "").strip()
                    current_section = "response"
                    response_lines = [conversational_response] if conversational_response else []
                elif current_section == "response" and line:
                    # Handle multi-line responses
                    response_lines.append(line)
            
            # Combine multi-line response if any
            if response_lines:
                conversational_response = " ".join(response_lines).strip()
            
            # Validate agent
            if agent not in ["nl2sql", "personal_assistant", "general_chat", "document_rag"]:
                agent = "general_chat"
            
            # Ensure we have a valid conversational response
            if not conversational_response or len(conversational_response.strip()) < 10:
                # Generate contextual fallback based on user profile
                if user_profile.experience_level == "beginner":
                    conversational_response = "I'm here to help you! Let me know what you'd like to work on and I'll guide you through it."
                elif conversation_context.urgency_level == "high":
                    conversational_response = "I understand this is urgent. I'm ready to help you with whatever you need right away."
                elif follow_up_info.get("is_follow_up"):
                    conversational_response = f"Building on our previous work, I'm ready to help you take the next step."
                else:
                    conversational_response = f"Hello! As a {user_profile.role}, I'm here to help with your warehouse operations. What can I assist you with?"
            
            print(f"✅ Combined contextual decision: {agent} (conf: {confidence:.2f}) + conversational response generated")
            
            return agent, confidence, reasoning, conversational_response
            
        except Exception as e:
            print(f"Combined contextual LLM error: {e}")
            
            # Generate contextual fallback response
            fallback_response = "I encountered a brief issue, but I'm here to help! Please let me know what you'd like to work on."
            if user_profile.experience_level == "beginner":
                fallback_response = "I had a small hiccup, but don't worry - I'm ready to help you with whatever you need. Just let me know!"
            elif conversation_context.urgency_level == "high":
                fallback_response = "I encountered a brief issue, but I'm prioritizing your urgent request. Please try again and I'll help immediately."
            
            return "general_chat", 0.4, f"llm_error: {str(e)[:30]}", fallback_response
    
    def _update_user_profile(self, user_profile: UserProfile, agent_used: str, query: str):
        """Update user profile based on routing decision"""
        
        # Update agent usage frequency
        user_profile.frequently_used_agents[agent_used] = \
            user_profile.frequently_used_agents.get(agent_used, 0) + 1
        
        # Add to common queries (keep only unique, recent ones)
        query_lower = query.lower()
        if query_lower not in user_profile.common_queries:
            user_profile.common_queries.append(query_lower)
            # Keep only last 20 queries
            user_profile.common_queries = user_profile.common_queries[-20:]
        
        # Update last activity
        user_profile.last_activity = datetime.now()
        
        # Adjust experience level based on query complexity
        if len(query.split()) > 10 and any(word in query_lower for word in ["analyze", "compare", "optimize", "complex"]):
            if user_profile.experience_level == "beginner":
                user_profile.experience_level = "intermediate"
                print(f"📈 User profile updated: {user_profile.role} promoted to intermediate level")
    
    def get_contextual_analytics(self) -> Dict[str, Any]:
        """Get analytics about contextual engineering performance"""
        
        analytics = {
            "total_users": len(self.context_manager.user_profiles),
            "total_conversations": len(self.context_manager.conversation_contexts),
            "user_experience_distribution": {},
            "user_role_distribution": {},
            "conversation_stage_distribution": {},
            "urgency_level_distribution": {},
            "combined_llm_calls": 0,  # Track the optimization
            "traditional_llm_calls": 0
        }
        
        # Analyze user profiles
        for profile in self.context_manager.user_profiles.values():
            exp_level = profile.experience_level
            analytics["user_experience_distribution"][exp_level] = \
                analytics["user_experience_distribution"].get(exp_level, 0) + 1
            
            role = profile.role
            analytics["user_role_distribution"][role] = \
                analytics["user_role_distribution"].get(role, 0) + 1
        
        # Analyze conversation contexts
        for context in self.context_manager.conversation_contexts.values():
            stage = context.conversation_stage
            analytics["conversation_stage_distribution"][stage] = \
                analytics["conversation_stage_distribution"].get(stage, 0) + 1
            
            urgency = context.urgency_level
            analytics["urgency_level_distribution"][urgency] = \
                analytics["urgency_level_distribution"].get(urgency, 0) + 1
        
        # Count combined vs traditional LLM calls from routing history
        for route_record in self.routing_history:
            method = route_record.get("method", "")
            if "combined" in method:
                analytics["combined_llm_calls"] += 1
            else:
                analytics["traditional_llm_calls"] += 1
        
        # Calculate optimization metrics
        total_calls = analytics["combined_llm_calls"] + analytics["traditional_llm_calls"]
        if total_calls > 0:
            analytics["optimization_rate"] = analytics["combined_llm_calls"] / total_calls
            analytics["estimated_performance_gain"] = f"{analytics['optimization_rate'] * 50:.1f}% faster responses"
        
        return analytics

    # Override the base route_query method to provide fallback compatibility
    def route_query(self, user_query: str, session_state: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Override base method to use contextual routing when possible, 
        fall back to base implementation otherwise
        """
        if session_state is not None:
            # Use the optimized contextual routing
            try:
                return self.route_query_contextual(user_query, session_state)
            except Exception as e:
                print(f"⚠️ Contextual routing failed, falling back to base: {e}")
        
        # Fallback to base implementation
        return super().route_query(user_query, session_state)
    
    def _detect_api_query_signals(self, user_query: str) -> float:
        """
        Detect if query is better suited for API than SQL
        
        Returns:
            confidence score (0.0 to 1.0) indicating API suitability
            
        Logic:
            - Checks for API-specific keywords and patterns
            - Returns high confidence (0.8+) for strong API queries
            - Returns medium confidence (0.5-0.7) for potential API queries
            - Reduces confidence if SQL indicators are present
        """
        query_lower = user_query.lower()
        
        # Strong API signals (0.8-1.0 confidence)
        strong_api_keywords = [
            'sales order', 'ship', 'shipped', 'shipping', 'track',
            'carrier', 'crew', 'assigned to', 'customer name',
            'po number', 'purchase order', 'order status',
            'shipment', 'delivery', 'tracking number'
        ]
        
        # Medium API signals (0.5-0.7 confidence)
        medium_api_keywords = [
            'today', 'now', 'current', 'active', 'ready',
            'unshipped', 'in production', 'loading',
            'order', 'orders', 'customer', 'customers'
        ]
        
        # Specific customer names (API handles better)
        customer_indicators = [
            'amazon', 'walmart', 'target', 'kcc', 'fns',
            'costco', 'sams club', 'kroger'
        ]
        
        # Crew indicators (API is authoritative)
        crew_indicators = [
            'crew t1', 'crew t2', 'crew t3', 
            't1 ', 't2 ', 't3 ',
            'crew ', 'team '
        ]
        
        # Calculate confidence
        confidence = 0.0
        
        # Check strong indicators
        for keyword in strong_api_keywords:
            if keyword in query_lower:
                confidence = max(confidence, 0.85)
        
        # Check medium indicators
        for keyword in medium_api_keywords:
            if keyword in query_lower:
                confidence = max(confidence, 0.65)
        
        # Check customer indicators
        for customer in customer_indicators:
            if customer in query_lower:
                confidence = max(confidence, 0.75)
        
        # Check crew indicators
        for crew in crew_indicators:
            if crew in query_lower:
                confidence = max(confidence, 0.80)
        
        # Anti-patterns (prefer SQL for these)
        sql_strong_indicators = [
            'how many', 'count', 'sum', 'total', 'average',
            'group by', 'each month', 'breakdown', 'report',
            'container', 'pallet', 'item number', 'upc',
            'expiration', 'received from', 'unload',
            'aggregate', 'analysis', 'trend', 'historical'
        ]
        
        # Reduce confidence if SQL is clearly better
        for indicator in sql_strong_indicators:
            if indicator in query_lower:
                confidence *= 0.4  # Significantly reduce API confidence
                break  # One indicator is enough
        
        return confidence


    def route_query_with_api_priority(
        self, 
        user_query: str, 
        session_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhanced routing with API-first logic for sales order queries
        
        Flow:
            1. Detect API signals in query
            2. If strong API signal (>0.75) → route to api_endpoint
            3. Otherwise → use normal LLM-based routing
            
        This ensures sales order queries go to API before SQL
        
        Args:
            user_query: The user's query string
            session_state: Optional session state dictionary
            
        Returns:
            Tuple of (agent_name, routing_info)
        """
        
        # Detect API signals
        api_confidence = self._detect_api_query_signals(user_query)
        
        # If strong API signal, route to API first
        if api_confidence > 0.75:
            routing_info = {
                "method": "api_priority",
                "confidence": api_confidence,
                "confidence_band": "high" if api_confidence > 0.85 else "medium",
                "api_first": True,
                "scores": {
                    "api_endpoint": api_confidence,
                    "nl2sql": api_confidence * 0.3,  # Lower SQL score
                    "personal_assistant": 0.1,
                    "general_chat": 0.1
                },
                "reasoning": (
                    f"Query contains strong API indicators (confidence: {api_confidence:.2f}). "
                    "Routing to API endpoint for sales order/shipping data."
                ),
                "similar_examples": [],
                "context": {},
                "follow_up_context": {}
            }
            
            # Update learning
            self._update_learning(
                user_query=user_query,
                selected_agent="api_endpoint",
                routing_info=routing_info,
                session_state=session_state
            )
            
            print(f"🎯 API Priority Routing: {api_confidence:.2f} confidence")
            return "api_endpoint", routing_info
        
        # Otherwise, use normal routing (with vector search + LLM)
        print(f"📊 Normal Routing: API confidence too low ({api_confidence:.2f})")
        return self.route_query(user_query, session_state)


    def route_query_with_smart_fallback(
        self,
        user_query: str,
        session_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        Route with smart fallback logic
        
        Returns:
            Tuple of (primary_agent, fallback_agent, routing_info)
            
        This method identifies both the primary agent and a potential fallback,
        allowing the application to try both if the primary fails.
        """
        
        # Get API confidence
        api_confidence = self._detect_api_query_signals(user_query)
        
        # Route normally
        primary_agent, routing_info = self.route_query(user_query, session_state)
        
        # Determine fallback
        fallback_agent = None
        
        if primary_agent == "api_endpoint":
            # If API is primary, SQL is fallback
            if api_confidence < 0.90:  # Not 100% sure
                fallback_agent = "nl2sql"
        
        elif primary_agent == "nl2sql":
            # If SQL is primary but query has API signals, API is fallback
            if api_confidence > 0.50:
                fallback_agent = "api_endpoint"
        
        # Add fallback info to routing_info
        routing_info["fallback_agent"] = fallback_agent
        routing_info["api_confidence"] = api_confidence
        
        return primary_agent, fallback_agent, routing_info


    def explain_api_routing(self, user_query: str, routing_info: Dict[str, Any]) -> str:
        """
        Enhanced routing explanation that includes API priority logic
        
        Returns:
            Human-readable explanation of routing decision
        """
        method = routing_info.get("method", "unknown")
        confidence = routing_info.get("confidence", 0.0)
        api_confidence = routing_info.get("api_confidence", 0.0)
        
        if method == "api_priority":
            return (
                f"🔌 **API Priority Routing** (confidence: {confidence:.2f})\n"
                f"Query contains sales order/shipping indicators. "
                f"Routed to API endpoint for real-time data."
            )
        
        elif api_confidence > 0.5:
            return (
                f"📊 **Normal Routing** (confidence: {confidence:.2f})\n"
                f"API confidence: {api_confidence:.2f} - not strong enough for API priority. "
                f"Using LLM-based routing with vector search."
            )
        
        else:
            return self.explain_routing(user_query, routing_info.get("selected_agent", "unknown"), routing_info)
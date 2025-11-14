"""
Modified Agent implementations - Tools return data silently, GeneralChat formats everything
"""

import json
from datetime import datetime
import streamlit as st
import ollama
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from config import LLM_MODEL, WEB_SEARCH_ENABLED
import time
from enhanced_personal_assistant import EnhancedPersonalAssistantAgent
from api_agent_multi_endpoint import MultiEndpointAPIAgent
from web_search import search_web
from query_store import QueryStore
import pandas as pd
from nlp_utils import detect_follow_up_query, user_references_prev_pos, detect_universal_follow_up
from unified_context_manager import UnifiedContextManager
QUERY_STORE = QueryStore()



class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def process_query(self, user_query: str, routing_info: Dict[str, Any], 
                     session_state: Dict[str, Any], silent_mode: bool = False) -> Dict[str, Any]:
        """Process the user query and return response
        
        Args:
            silent_mode: If True, don't display anything - just return data
        """
        pass


class NL2SQLAgent(BaseAgent):
    """Agent that handles NL2SQL - can run in silent mode for tool use"""
    
    def __init__(self, unified_context_manager=None):
        super().__init__("NL2SQL")
        self.unified_context_manager = unified_context_manager

    
    def process_query(self, user_query: str, routing_info: Dict[str, Any], 
                     session_state: Dict[str, Any], silent_mode: bool = False) -> Dict[str, Any]:
        """Process SQL queries - silent mode returns data without display"""
        
        routing_info = routing_info or {}
        follow_up_context = routing_info.get("follow_up_context", {})
        is_follow_up = follow_up_context.get("is_follow_up", False)
        
        # Import functions
        from schema_analysis import identify_relevant_tables_and_columns
        from sql_operations import (
            validate_sql, refine_sql_with_context_structured, 
            generate_sql_chain_of_thought, maybe_add_po_filter
        )
        from nlp_utils import detect_follow_up_query, user_references_prev_pos, detect_universal_follow_up
        from database import execute_sql_and_return_rows
        
        # Only display if not silent
        if not silent_mode:
            from ui_components import (
                display_query_results, display_feedback_buttons, display_error_handling
            )
        
        try:
            conv_manager = session_state['conv_manager']
            query_store = session_state['query_store']
            session_id = session_state['session_id']
            
            # Add user message
            conv_manager.add_message("user", user_query)
            
            # Build processing query with context
            processing_query = user_query
            if is_follow_up and follow_up_context.get("last_agent") != "nl2sql":
                if not silent_mode:
                    st.info(f"📊 NL2SQL: Building on previous {follow_up_context['last_agent']} context")
                last_query = follow_up_context.get('last_query', '')
                processing_query = f"Previous context: {last_query}. Current request: {user_query}"
            
            # Table/column identification
            selected_tables, selected_columns = identify_relevant_tables_and_columns(
                user_query=processing_query,
                query_store=query_store,
                session_id=session_id
            )
            
            # Check for SQL follow-ups
            sql_follow_up, confidence = detect_follow_up_query(user_query, session_id, query_store)
            last_sql = query_store.fetch_last_query(session_id)
            references_prev_pos = user_references_prev_pos(user_query)
            
            # Generate or refine SQL
            if sql_follow_up and last_sql and follow_up_context.get("last_agent") == "nl2sql":
                if not silent_mode:
                    st.info("Refining previous SQL query")
                refined_sql, explanation = refine_sql_with_context_structured(user_query, last_sql)
                sql_query = refined_sql
                if not silent_mode:
                    st.write(f"**SQL Refinement**: {explanation}")
                if references_prev_pos:
                    sql_query = maybe_add_po_filter(sql_query, session_state.get('last_po_nums', set()))
            else:
                reasoning, sql_query, final_prompt = generate_sql_chain_of_thought(
                    processing_query, conv_manager, query_store,
                    selected_tables, selected_columns, session_state
                )
                conv_manager.add_message("assistant", f"Reasoning: {reasoning}\nSQL: {sql_query}")
                if references_prev_pos:
                    sql_query = maybe_add_po_filter(sql_query, session_state.get('last_po_nums', set()))
            
            # Validate and execute
            validated_sql = validate_sql(sql_query, session_state['full_schema'])
            cols, rows, err = execute_sql_and_return_rows(validated_sql)
            
            if err:
                if not silent_mode:
                    display_error_handling(err, validated_sql, user_query, 
                                          query_store, session_id)
                return {
                    "success": False, 
                    "error": err, 
                    "sql": validated_sql,
                    "agent": self.name
                }
            
            # Store query
            query_store.add_query(
                session_id=session_id,
                user_query=user_query,
                sql_query=validated_sql,
                results=rows if rows else None,
                response="Success" if rows else "No Data",
                sql_error=None,
                agent_used=self.name
            )
            
            # Display only if not silent
            if not silent_mode and rows:
                from ui_components import display_query_results, display_feedback_buttons
                success, message = display_query_results(
                    rows, cols, user_query, session_state['col_comment_map']
                )
                if success:
                    display_feedback_buttons(
                        user_query, validated_sql, rows, 
                        query_store, session_id
                    )
            
            # Return structured data
            return {
                "success": True,
                "results": rows or [],
                "columns": cols or [],
                "sql": validated_sql,
                "row_count": len(rows) if rows else 0,
                "agent": self.name,
                "message": f"Found {len(rows)} results" if rows else "No data found"
            }
                
        except Exception as e:
            if not silent_mode:
                st.error(f"Error in NL2SQL: {str(e)}")
            return {
                "success": False, 
                "error": str(e), 
                "agent": self.name
            }


class WebSearchAgent(BaseAgent):
    """Dedicated agent for web search operations - returns data silently"""
    
    def __init__(self, unified_context_manager=None):
        super().__init__("WebSearch")
        self.unified_context_manager = unified_context_manager

    def process_query(self, user_query: str, routing_info: Dict[str, Any], 
                     session_state: Dict[str, Any], silent_mode: bool = False) -> Dict[str, Any]:
        """Execute web search and return results"""
        
        if not WEB_SEARCH_ENABLED:
            return {
                "success": False,
                "error": "Web search is disabled",
                "agent": self.name
            }
        
        try:
            if not silent_mode:
                with st.status("🔎 Searching the web..."):
                    results = search_web(user_query)
            else:
                results = search_web(user_query)
            
            if not results:
                return {
                    "success": True,
                    "results": [],
                    "message": "No web results found",
                    "agent": self.name
                }
            
            # Format results
            formatted_results = []
            for r in results:
                formatted_results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("snippet", "")
                })
            
            return {
                "success": True,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "agent": self.name,
                "message": f"Found {len(formatted_results)} web results"
            }
            
        except Exception as e:
            if not silent_mode:
                st.error(f"Web search error: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }


class DocumentRAGAgent(BaseAgent):
    """RAG agent that returns results silently"""
    
    def __init__(self, query_store: QueryStore, unified_context_manager=None):
        super().__init__("DocumentRAG")
        self.query_store = query_store
        self.unified_context_manager = unified_context_manager
    
    def process_query(self, user_query: str, routing_info: Dict[str, Any], 
                     session_state: Dict[str, Any], silent_mode: bool = False) -> Dict[str, Any]:
        """Process document queries - returns structured data"""
        
        try:
            # Search documents
            results = self.query_store.search_documents(user_query, k=5)
            
            if not results:
                return {
                    "success": True,
                    "results": [],
                    "message": "No relevant documents found",
                    "agent": self.name
                }
            
            # Format results
            formatted_results = []
            for doc_id, content, similarity, metadata in results:
                formatted_results.append({
                    "document_id": doc_id,
                    "content": content,
                    "similarity": float(similarity),
                    "metadata": metadata or {}
                })
            
            return {
                "success": True,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "agent": self.name,
                "message": f"Found {len(formatted_results)} relevant documents"
            }
            
        except Exception as e:
            if not silent_mode:
                st.error(f"RAG error: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }


class GeneralChatAgent(BaseAgent):
    """Enhanced General Chat - formats ALL tool outputs into conversational responses with full context awareness"""
    
    def __init__(self, unified_context_manager=None):
        super().__init__("GeneralChat")
        self.unified_context_manager = unified_context_manager
    
    def _format_sql_results_for_llm(self, tool_output: Dict[str, Any]) -> str:
        """Format SQL results for LLM consumption"""
        if not tool_output.get("success"):
            return f"Database query failed: {tool_output.get('error', 'Unknown error')}"
        
        results = tool_output.get("results", [])
        if not results:
            return "Database query returned no results."
        
        # Format results as text
        lines = [f"Database query found {len(results)} results:"]
        lines.append(f"SQL executed: {tool_output.get('sql', 'N/A')}")
        lines.append("\nFirst few results:")
        
        for i, row in enumerate(results[:5], 1):
            row_text = ", ".join([f"{k}: {v}" for k, v in row.items()])
            lines.append(f"{i}. {row_text}")
        
        if len(results) > 5:
            lines.append(f"... and {len(results) - 5} more results")
        
        return "\n".join(lines)
    
    def _format_web_results_for_llm(self, tool_output: Dict[str, Any]) -> str:
        """Format web search results for LLM"""
        if not tool_output.get("success"):
            return f"Web search failed: {tool_output.get('error', 'Unknown error')}"
        
        results = tool_output.get("results", [])
        if not results:
            return "Web search returned no results."
        
        lines = [f"Web search found {len(results)} results:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('title', 'No title')}")
            lines.append(f"   {r.get('snippet', 'No snippet')}")
            lines.append(f"   Source: {r.get('url', 'No URL')}")
        
        return "\n".join(lines)
    
    def _format_rag_results_for_llm(self, tool_output: Dict[str, Any]) -> str:
        """Format RAG results for LLM"""
        if not tool_output.get("success"):
            return f"Document search failed: {tool_output.get('error', 'Unknown error')}"
        
        results = tool_output.get("results", [])
        if not results:
            return "No relevant documents found."
        
        lines = [f"Found {len(results)} relevant document sections:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.get('content', '')[:300]}...")
            lines.append(f"   Relevance: {r.get('similarity', 0):.2f}")
        
        return "\n".join(lines)
    
    def _add_download_buttons(self, tool_output: Dict[str, Any], user_query: str):
        """Add download buttons for structured data"""
        
        # Convert to DataFrame
        df = None
        if tool_output.get("agent") == "NL2SQL":
            results = tool_output.get("results", [])
            if results:
                df = pd.DataFrame(results)
        elif tool_output.get("agent") in ("APIEndpoint", "api_endpoint", "API"):
            results = tool_output.get("results", [])
            if results:
                df = pd.DataFrame(results)
        elif tool_output.get("agent") == "WebSearch":
            results = tool_output.get("results", [])
            if results:
                df = pd.DataFrame(results)
        elif tool_output.get("agent") == "DocumentRAG":
            results = tool_output.get("results", [])
            if results:
                # Format RAG results for download
                formatted = []
                for r in results:
                    formatted.append({
                        "content": r.get("content", "")[:500],
                        "similarity": f"{r.get('similarity', 0):.3f}",
                        "document_id": r.get("document_id", "")
                    })
                df = pd.DataFrame(formatted)
        
        if df is None or df.empty:
            return
        
        # Create downloads
        st.markdown("---")
        st.markdown("### 📥 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download (requires openpyxl)
            try:
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                
                st.download_button(
                    label="📊 Download Excel",
                    data=buffer.getvalue(),
                    file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except ImportError:
                st.caption("⚠️ Excel export requires openpyxl")
        
        with col3:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON",
                data=json_data,
                file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        # Optional preview
        with st.expander(f"👁️ Preview Data ({len(df)} rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            if len(df) > 10:
                st.caption(f"Showing first 10 of {len(df)} rows")
    
    def process_query(self, user_query: str, routing_info: Dict[str, Any], 
                     session_state: Dict[str, Any], silent_mode: bool = False,
                     tool_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process query with optional tool output - generates conversational response
        NOW WITH FULL CONVERSATION CONTEXT AWARENESS using embeddings!
        """
        
        # ========================================
        # STEP 1: Get conversation context using existing embedding system
        # ========================================
        conversation_context = ""
        is_follow_up = False
        follow_up_confidence = 0.0
        last_agent_used = None
        
        if self.unified_context_manager:
            try:
                # Get full cross-agent context
                unified_context = self.unified_context_manager.get_unified_context(
                    current_query=user_query,
                    include_all_agents=True,
                    max_recent=5,
                    max_relevant=3
                )
                
                # Format context for LLM
                conversation_context = unified_context.get_formatted_context(max_messages=10)
                
                # Check for cross-agent references
                cross_ref = self.unified_context_manager.detect_cross_agent_reference(user_query)
                if cross_ref:
                    is_follow_up = True
                    last_agent_used = cross_ref['referenced_agent']
                    
                    if not silent_mode:
                        st.info(f"🔗 Following up on {last_agent_used} conversation")
                
                if unified_context.last_agent:
                    last_agent_used = last_agent_used or unified_context.last_agent
                
                print(f"✅ GeneralChat: Using unified context")
                
            except Exception as e:
                print(f"⚠️ Unified context failed: {e}")
                # Fallback to basic detection
                try:
                    query_store = session_state.get('query_store')
                    session_id = session_state.get('session_id')
                    
                    if query_store and session_id:
                        from nlp_utils import detect_universal_follow_up
                        is_follow_up, _, last_agent_used, last_context = detect_universal_follow_up(
                            user_query, session_id, query_store
                        )
                except:
                    pass

        
        # ========================================
        # STEP 2: Get current time
        # ========================================
        now = datetime.now()
        current_time = now.strftime("%A, %B %d, %Y %I:%M %p")
        
        # ========================================
        # STEP 3: Format tool output if available
        # ========================================
        tool_context = ""
        if tool_output:
            tool_type = tool_output.get("agent", "unknown")
            
            if tool_type == "NL2SQL":
                tool_context = self._format_sql_results_for_llm(tool_output)
            elif tool_type == "WebSearch":
                tool_context = self._format_web_results_for_llm(tool_output)
            elif tool_type == "DocumentRAG":
                tool_context = self._format_rag_results_for_llm(tool_output)
            else:
                tool_context = f"Tool output: {json.dumps(tool_output, indent=2)}"
        
        # ========================================
        # STEP 4: Build context-aware prompt
        # ========================================
        if tool_context or conversation_context:
            chat_prompt = f"""You are a friendly, helpful AI assistant with full conversation awareness.
Current time: {current_time}

{conversation_context}

User's current question: "{user_query}"

{f'''Current tool information:
{tool_context}
''' if tool_context else ''}

Instructions:
- You have FULL CONTEXT of the previous conversation (see above)
- If this is a follow-up (context provided), continue the conversation naturally
- If user said "yes", "tell me more", "continue", etc., provide additional details about the previous topic
- If user asks a clarifying question, use the previous context to understand what they're referring to
- Present any tool findings in a natural, conversational way
- For database results, summarize key findings and highlight important patterns
- For web results, synthesize information from multiple sources
- For document results, extract and explain relevant information
- Be concise but thorough
- Reference previous discussion when relevant ("As we discussed earlier...", "Building on those results...")
- Don't mention "the tool" or "previous context" explicitly - just be naturally aware
- If the tool returned an error or no results, explain this helpfully and suggest alternatives
"""
        else:
            # No tool output and no follow-up - regular chat
            chat_prompt = f"""You are a friendly, helpful AI assistant.
Current time: {current_time}

User question: "{user_query}"

Respond naturally and helpfully."""
        
        # ========================================
        # STEP 5: Generate response with LLM
        # ========================================
        try:
            # Stream response
            placeholder = st.empty()
            stream_text = ""
            
            response_generator = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a context-aware AI assistant that maintains conversation continuity and presents information clearly."},
                    {"role": "user", "content": chat_prompt}
                ],
                stream=True,
                options={"temperature": 0.3, "num_ctx": 16384}  # Increased context window
            )
            
            for chunk in response_generator:
                new_text = chunk.get("message", {}).get("content", "")
                if new_text:
                    stream_text += new_text
                    placeholder.markdown(stream_text)
                    time.sleep(0.02)
            
            final_text = stream_text.strip()
            placeholder.markdown(final_text)
            
            # ========================================
            # STEP 6: Add download buttons if we have data
            # ========================================
            if tool_output and tool_output.get("results"):
                self._add_download_buttons(tool_output, user_query)
            
            # ========================================
            # STEP 7: Return result with context metadata
            # ========================================
            return {
                "success": True,
                "message": final_text,
                "agent": self.name,
                "tool_used": tool_output.get("agent") if tool_output else None,
                "is_follow_up": is_follow_up,
                "follow_up_confidence": follow_up_confidence,
                "last_agent": last_agent_used
            }
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.name,
                "is_follow_up": is_follow_up
            }
        
class AgentManager:
    """Manages all agents - supports tool mode and presentational mode"""
    
    def __init__(self, query_store, unified_context_manager=None):
        self.query_store = query_store
        self.unified_context_manager = unified_context_manager
        self.agents = {
            "nl2sql": NL2SQLAgent(unified_context_manager=unified_context_manager),
            "api_endpoint": MultiEndpointAPIAgent(query_store,api_base_url="http://localhost/creme/API_LLM", unified_context_manager=unified_context_manager),
            "web_search": WebSearchAgent(unified_context_manager=unified_context_manager),
            "document_rag": DocumentRAGAgent(QUERY_STORE, unified_context_manager=unified_context_manager),
            "general_chat": GeneralChatAgent(unified_context_manager=unified_context_manager),
            "personal_assistant": EnhancedPersonalAssistantAgent(unified_context_manager=unified_context_manager)
        }
        if unified_context_manager:
            print("✅ All agents initialized with unified context awareness")
        else:
            print("⚠️ Agents initialized WITHOUT unified context (degraded mode)")
   
    def execute_tool(self, tool_name: str, user_query: str, 
                    routing_info: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool in silent mode - returns data only"""
        
        agent = self.agents.get(tool_name)
        if not agent:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "agent": tool_name
            }
        silent_mode = (tool_name != "api_endpoint")
        # Execute in silent mode
        return agent.process_query(
            user_query=user_query,
            routing_info=routing_info,
            session_state=session_state,
            silent_mode=silent_mode
        )
    
    def process_with_general_chat(self, user_query: str, tool_output: Dict[str, Any],
                                 routing_info: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process tool output through GeneralChat for presentation"""
        
        general_chat = self.agents.get("general_chat")
        return general_chat.process_query(
            user_query=user_query,
            routing_info=routing_info,
            session_state=session_state,
            silent_mode=False,
            tool_output=tool_output
        )
    
    def process_with_agent(self, agent_type: str, user_query: str,
                          routing_info: Dict[str, Any], session_state: Dict[str, Any],
                          use_general_chat: bool = True) -> Dict[str, Any]:
        """
        New unified processing:
        1. Execute the tool (nl2sql/rag/web) in silent mode
        2. Pass output to general_chat for presentation
        
        Args:
            use_general_chat: If True, always format output through general_chat
        """
        
        # Map agent types to tools
        tool_map = {
            "nl2sql": "nl2sql",
            "document_rag": "document_rag",
            "web_search": "web_search",
            "general_chat": "general_chat",
            "personal_assistant": "personal_assistant"
        }
        
        tool_name = tool_map.get(agent_type, agent_type)
        
        # Special case: general_chat and personal_assistant don't need tool execution
        if tool_name in ["general_chat", "personal_assistant"]:
            agent = self.agents.get(tool_name)
            return agent.process_query(
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state,
                silent_mode=False
            )
        
        # For other agents: execute tool -> format with general_chat
        if use_general_chat:
            # Step 1: Execute tool silently
            st.info(f"🔧 Executing {tool_name} tool...")
            tool_output = self.execute_tool(
                tool_name=tool_name,
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state
            )
            
            # Step 2: Format with general_chat
            st.info("💬 Formatting response...")
            return self.process_with_general_chat(
                user_query=user_query,
                tool_output=tool_output,
                routing_info=routing_info,
                session_state=session_state
            )
        else:
            # Legacy mode: execute normally with display
            agent = self.agents.get(tool_name)
            return agent.process_query(
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state,
                silent_mode=False
            )

    def execute_tool_with_fallback(
        self, 
        tool_name: str, 
        user_query: str, 
        routing_info: Dict[str, Any], 
        session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tool with smart fallback logic
        
        Flow:
            1. Execute primary tool (api_endpoint or nl2sql)
            2. Check if result is successful and meaningful
            3. If not, try fallback tool automatically
            4. Return best result
        
        Fallback Rules:
            - API → SQL: If API returns no data or fails
            - SQL → API: If SQL fails and query has API indicators
        
        Args:
            tool_name: Primary tool to execute (api_endpoint, nl2sql, etc.)
            user_query: The user's query
            routing_info: Routing decision info
            session_state: Session state dictionary
            
        Returns:
            Dict with tool execution results (includes fallback info if used)
        """
        import streamlit as st
        
        # Execute primary tool
        print(f"🔧 Executing primary tool: {tool_name}")
        result = self.execute_tool(tool_name, user_query, routing_info, session_state)
        
        # Determine if we should try fallback
        should_fallback = False
        fallback_tool = None
        fallback_reason = None
        
        # API → SQL Fallback
        if tool_name == "api_endpoint":
            # Check if API failed or returned no data
            if not result.get("success"):
                should_fallback = True
                fallback_tool = "nl2sql"
                fallback_reason = "API request failed"
                
            elif result.get("row_count", 0) == 0:
                should_fallback = True
                fallback_tool = "nl2sql"
                fallback_reason = "API returned no results"
                
            elif result.get("error"):
                should_fallback = True
                fallback_tool = "nl2sql"
                fallback_reason = f"API error: {result.get('error')}"
        
        # SQL → API Fallback
        elif tool_name == "nl2sql":
            # Check if SQL failed
            if not result.get("success"):
                # Check if query has API indicators
                query_lower = user_query.lower()
                api_keywords = [
                    'sales order', 'shipped', 'shipping', 'crew',
                    'customer', 'po number', 'order status', 'tracking'
                ]
                
                has_api_signal = any(keyword in query_lower for keyword in api_keywords)
                
                if has_api_signal:
                    should_fallback = True
                    fallback_tool = "api_endpoint"
                    fallback_reason = "SQL failed, query has API indicators"
        
        # Execute fallback if needed
        if should_fallback and fallback_tool:
            if 'st' in dir():  # Check if Streamlit is available
                st.info(f"🔄 {fallback_reason}, trying {fallback_tool}...")
            
            print(f"🔄 Attempting fallback: {tool_name} → {fallback_tool}")
            print(f"   Reason: {fallback_reason}")
            
            # Execute fallback tool
            fallback_result = self.execute_tool(
                fallback_tool, 
                user_query, 
                routing_info, 
                session_state
            )
            
            # Check if fallback succeeded
            if fallback_result.get("success"):
                # Annotate that this was a fallback
                fallback_result["primary_tool"] = tool_name
                fallback_result["fallback_from"] = tool_name
                fallback_result["fallback_reason"] = fallback_reason
                fallback_result["fallback_used"] = True
                
                if 'st' in dir():
                    st.success(f"✅ Fallback successful! Using {fallback_tool} results")
                
                print(f"✅ Fallback succeeded: {fallback_tool} returned results")
                return fallback_result
            else:
                # Fallback also failed
                if 'st' in dir():
                    st.warning(f"⚠️ Both {tool_name} and {fallback_tool} failed")
                
                print(f"❌ Fallback also failed: {fallback_tool}")
                
                # Return original result with fallback attempt info
                result["fallback_attempted"] = True
                result["fallback_tool"] = fallback_tool
                result["fallback_succeeded"] = False
                return result
        
        # No fallback needed or available
        result["fallback_attempted"] = False
        return result


    def execute_with_cascade(
        self,
        user_query: str,
        routing_info: Dict[str, Any],
        session_state: Dict[str, Any],
        tool_priority: List[str] = None
    ) -> Dict[str, Any]:
        """
        Execute tools in priority order until one succeeds
        
        This is more aggressive than fallback - it tries multiple tools in order.
        
        Args:
            user_query: The user's query
            routing_info: Routing decision info
            session_state: Session state
            tool_priority: List of tools to try in order (default: ["api_endpoint", "nl2sql"])
            
        Returns:
            First successful result, or last result if all fail
        """
        import streamlit as st
        
        if tool_priority is None:
            tool_priority = ["api_endpoint", "nl2sql"]
        
        results = []
        
        for i, tool_name in enumerate(tool_priority):
            print(f"🔧 Cascade attempt {i+1}/{len(tool_priority)}: {tool_name}")
            
            if 'st' in dir() and i > 0:
                st.info(f"🔄 Trying {tool_name}...")
            
            result = self.execute_tool(tool_name, user_query, routing_info, session_state)
            results.append((tool_name, result))
            
            # Check if successful and has data
            if result.get("success") and (
                result.get("total_count", 0) > 0 or 
                result.get("data") or 
                result.get("rows")
            ):
                # Success! Annotate and return
                result["cascade_tool"] = tool_name
                result["cascade_position"] = i + 1
                result["tools_attempted"] = [t for t, _ in results]
                
                if 'st' in dir() and i > 0:
                    st.success(f"✅ {tool_name} succeeded!")
                
                print(f"✅ Cascade succeeded at position {i+1}: {tool_name}")
                return result
        
        # All failed - return last result with cascade info
        last_tool, last_result = results[-1]
        last_result["cascade_tool"] = last_tool
        last_result["cascade_position"] = len(tool_priority)
        last_result["tools_attempted"] = [t for t, _ in results]
        last_result["all_failed"] = True
        
        if 'st' in dir():
            st.error(f"❌ All tools failed: {', '.join(tool_priority)}")
        
        print(f"❌ Cascade failed - all {len(tool_priority)} tools failed")
        return last_result


    def smart_route_and_execute(
        self,
        user_query: str,
        session_state: Dict[str, Any],
        router = None
    ) -> Dict[str, Any]:
        """
        All-in-one: route, execute, and fallback automatically
        
        This is the highest-level method that does everything:
        1. Route query (using router if provided)
        2. Execute primary tool
        3. Try fallback if needed
        4. Return best result
        
        Args:
            user_query: The user's query
            session_state: Session state
            router: Optional router instance (uses default routing if None)
            
        Returns:
            Dict with execution results
        """
        import streamlit as st
        
        # Step 1: Route query
        if router and hasattr(router, 'route_query_with_api_priority'):
            agent_type, routing_info = router.route_query_with_api_priority(
                user_query, 
                session_state
            )
        elif router:
            agent_type, routing_info = router.route_query(user_query, session_state)
        else:
            # Simple fallback routing
            query_lower = user_query.lower()
            if any(word in query_lower for word in ['sales order', 'shipped', 'crew', 'customer']):
                agent_type = "api_endpoint"
            else:
                agent_type = "nl2sql"
            
            routing_info = {
                "method": "simple_fallback",
                "confidence": 0.6
            }
        
        if 'st' in dir():
            st.info(f"🧭 Routed to: {agent_type}")
        
        print(f"🧭 Routed to: {agent_type}")
        
        # Step 2: Execute with fallback
        result = self.execute_tool_with_fallback(
            agent_type,
            user_query,
            routing_info,
            session_state
        )
        
        # Step 3: Add routing info to result
        result["routed_to"] = agent_type
        result["routing_info"] = routing_info
        
        return result
"""
Enhanced Personal Assistant Agent with comprehensive productivity features
"""
import json
import datetime
import ollama
import streamlit as st
from typing import Dict, Any, List, Optional
from config import LLM_MODEL
import re
class EnhancedPersonalAssistantAgent:
    """
    Enhanced Personal Assistant Agent with advanced productivity features
    """
    
    def __init__(self, unified_context_manager=None):
        self.name = "personal_assistant"  # FIXED: Consistent agent name
        self.display_name = "Enhanced Personal Assistant"
        self.unified_context_manager = unified_context_manager
        self.capabilities = {
            "task_management": True,
            "scheduling": True,
            "email_assistance": True,
            "note_taking": True,
            "goal_tracking": True,
            "habit_tracking": True,
            "productivity_tips": True,
            "natural_conversation": True,  # NEW: Added this capability
            "cross_agent_awareness": unified_context_manager is not None
        }
    
    def process_query(self, user_query: str, routing_info: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process personal assistant queries with enhanced conversation embedding"""
        routing_info = routing_info or {}
        
        # Initialize session state for personal assistant features
        self._initialize_session_state(session_state)
        
        # FIXED: Always log the incoming query first for proper conversation tracking
        self._log_user_query(user_query, session_state)
        
        # Get conversation context BEFORE processing (this is crucial for natural responses)
        conversation_context = self._get_conversation_context(user_query, session_state)
        
        # Handle cross-agent follow-ups
        follow_up_context = routing_info.get("follow_up_context", {})
        is_follow_up = follow_up_context.get("is_follow_up", False)
        
        if is_follow_up and follow_up_context.get("last_agent") != self.name:
            return self._handle_cross_agent_follow_up(user_query, follow_up_context, conversation_context, session_state)
        
        # ENHANCED: Use AI to understand user intent from natural language with conversation context
        intent_analysis = self._analyze_user_intent_with_ai(user_query, conversation_context, session_state)
        
        # Route based on AI-determined intent with conversation awareness
        if intent_analysis.get("intent") == "task_creation":
            return self._handle_intelligent_task_creation(user_query, intent_analysis, conversation_context, session_state)
        elif intent_analysis.get("intent") == "scheduling":
            return self._handle_intelligent_scheduling(user_query, intent_analysis, conversation_context, session_state)
        elif intent_analysis.get("intent") == "task_query":
            return self._handle_task_listing(user_query, conversation_context, session_state)
        elif intent_analysis.get("intent") == "task_management":
            return self._handle_task_management_operations(user_query, intent_analysis, conversation_context, session_state)
        elif intent_analysis.get("intent") == "conversation_query":
            return self._handle_conversation_query(user_query, conversation_context, session_state)
        else:
            return self._handle_general_assistance(user_query, conversation_context, session_state)
    
    
    def _initialize_session_state(self, session_state: Dict[str, Any]):
        """Initialize session state for personal assistant features"""
        if 'pa_tasks' not in session_state:
            session_state['pa_tasks'] = []
        
        if 'pa_calendar' not in session_state:
            session_state['pa_calendar'] = []
        
        if 'pa_notes' not in session_state:
            session_state['pa_notes'] = []
        
        if 'pa_goals' not in session_state:
            session_state['pa_goals'] = []
        
        if 'pa_habits' not in session_state:
            session_state['pa_habits'] = {}

    def _log_user_query(self, user_query: str, session_state: Dict[str, Any]):
        """FIXED: Properly log user queries for conversation tracking"""
        try:
            query_store = session_state.get('query_store')
            session_id = session_state.get('session_id')
            
            if query_store and session_id:
                # Store the user query with proper agent identification
                query_store.add_query(
                    session_id=session_id,
                    user_query=user_query,
                    sql_query=None,
                    results=None,
                    response="Processing",  # Will be updated when response is ready
                    sql_error=None,
                    agent_used=self.name  # FIXED: Use consistent agent name
                )
                print(f"✅ Logged user query for personal assistant: {user_query[:50]}...")
                
        except Exception as e:
            print(f"⚠️ Failed to log user query: {e}")
            # Don't let logging failures break the conversation

    def _get_conversation_context(self, user_query: str, session_state: Dict[str, Any]) -> str:
        """Get conversation context using unified context manager if available"""
        
        if self.unified_context_manager:
            try:
                context_str = self.unified_context_manager.get_context_for_agent(
                    agent_name=self.name,
                    current_query=user_query,
                    include_other_agents=True
                )
                
                cross_ref = self.unified_context_manager.detect_cross_agent_reference(user_query)
                if cross_ref:
                    ref_agent = cross_ref['referenced_agent']
                    ref_msg = cross_ref['referenced_message']
                    context_str += f"\n\n🔗 User is referencing {ref_agent}:\n"
                    context_str += f"  Query: '{ref_msg.user_query}'\n"
                
                return context_str
            except Exception as e:
                print(f"⚠️ Unified context failed: {e}")
        
        # Fallback to original method
        try:
            query_store = session_state.get('query_store')
            session_id = session_state.get('session_id')
            
            if not query_store or not session_id:
                return "No conversation history available."
            
            similar_conversations = query_store.search_query(session_id, user_query, k=5)
            
            if similar_conversations:
                context_parts = []
                for query, sql_query, results in similar_conversations:
                    if query:
                        context_parts.append(f"Similar: '{query}'")
                return "\n".join(context_parts[:3])
        except Exception as e:
            print(f"⚠️ Context retrieval failed: {e}")
        
        return "No conversation history available."


    def _analyze_user_intent_with_ai(self, user_query: str, conversation_context: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """ENHANCED: AI intent analysis with conversation context"""
        
        # Get current tasks for context
        current_tasks = session_state.get('pa_tasks', [])
        pending_count = len([t for t in current_tasks if t.get('status') == 'pending'])
        
        # Get today's tasks for schedule queries
        today = datetime.date.today().isoformat()
        today_tasks = [t for t in current_tasks if t.get('deadline') and today in str(t.get('deadline'))]
        
        # Enhanced intent prompt with conversation context
        intent_prompt = f"""
Analyze this user query for personal assistant intent: "{user_query}"

CONVERSATION CONTEXT:
{conversation_context}

CURRENT STATE:
- User has {pending_count} pending tasks
- {len(today_tasks)} tasks for today
- This is a continuing conversation (reference the context above)

Determine the intent (choose ONE):
1. task_creation - wants to create/add a new task, meeting, reminder
2. scheduling - asks about schedule, agenda, calendar, what's today/tomorrow
3. task_query - wants to see/list existing tasks  
4. task_management - wants to complete/delete/modify existing tasks
5. conversation_query - asking about previous conversations, "what did we discuss", "remember when"
6. general_assistance - other help, general chat

If task_creation, extract:
- Task description
- Any deadline mentioned (today, tomorrow, specific time)
- Priority level (high/medium/low)
- Meeting participants mentioned
- Location mentioned

If scheduling or conversation_query, extract:
- Time period (today/tomorrow/this_week)
- Reference to specific previous conversations

Respond ONLY with this format:
INTENT: [intent_name]
TASK_DESC: [if task_creation]
DEADLINE: [if mentioned]
PRIORITY: [if determinable]
PARTICIPANTS: [if mentioned]
TIME_PERIOD: [if scheduling]
CONVERSATION_REF: [if asking about previous conversations]
"""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You analyze user intent for a personal assistant with conversation awareness. Respond in the exact format requested."},
                    {"role": "user", "content": intent_prompt}
                ],
                stream=False,
                options={"temperature": 0.1, "num_ctx": 3072}
            )
            
            # Parse the structured response
            response_text = response["message"]["content"].strip()
            
            intent_data = {"intent": "general_assistance", "confidence": 0.5, "extracted_info": {}}
            
            # Parse structured response
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('INTENT:'):
                    intent_data["intent"] = line.replace('INTENT:', '').strip().lower()
                    intent_data["confidence"] = 0.9
                elif line.startswith('TASK_DESC:'):
                    task_desc = line.replace('TASK_DESC:', '').strip()
                    if task_desc and task_desc.lower() != 'none':
                        intent_data["extracted_info"]["task_description"] = task_desc
                elif line.startswith('DEADLINE:'):
                    deadline = line.replace('DEADLINE:', '').strip()
                    if deadline and deadline.lower() != 'none':
                        intent_data["extracted_info"]["deadline"] = deadline
                elif line.startswith('PRIORITY:'):
                    priority = line.replace('PRIORITY:', '').strip().lower()
                    if priority in ['high', 'medium', 'low']:
                        intent_data["extracted_info"]["priority"] = priority
                elif line.startswith('PARTICIPANTS:'):
                    participants = line.replace('PARTICIPANTS:', '').strip()
                    if participants and participants.lower() != 'none':
                        intent_data["extracted_info"]["participants"] = [p.strip() for p in participants.split(',')]
                elif line.startswith('TIME_PERIOD:'):
                    time_period = line.replace('TIME_PERIOD:', '').strip().lower()
                    if time_period and time_period.lower() != 'none':
                        intent_data["extracted_info"]["time_period"] = time_period
                elif line.startswith('CONVERSATION_REF:'):
                    conv_ref = line.replace('CONVERSATION_REF:', '').strip()
                    if conv_ref and conv_ref.lower() != 'none':
                        intent_data["extracted_info"]["conversation_reference"] = conv_ref
            
            return intent_data
            
        except Exception as e:
            print(f"⚠️ AI intent analysis failed: {e}")
            return self._fallback_intent_analysis(user_query)



    def _fallback_intent_analysis(self, user_query: str) -> Dict[str, Any]:
        """Fallback intent analysis using simple patterns"""
        query_lower = user_query.lower()
        
        # Conversation query patterns - NEW
        if any(pattern in query_lower for pattern in [
            "what did we discuss", "remember when", "what did i tell you", "previous conversation",
            "earlier we talked", "you mentioned", "we were talking about"
        ]):
            return {"intent": "conversation_query", "confidence": 0.8}
        
        # Scheduling patterns
        elif any(pattern in query_lower for pattern in [
            "schedule", "agenda", "calendar", "today", "tomorrow", "this week",
            "what do i have", "what's on my", "my day", "my week"
        ]):
            return {
                "intent": "scheduling",
                "confidence": 0.8,
                "extracted_info": {
                    "time_period": "today" if "today" in query_lower else "general"
                }
            }
        
        # Task creation patterns
        elif any(pattern in query_lower for pattern in [
            "meeting", "remind me", "need to", "have to", "should", 
            "appointment", "call", "email", "deadline", "due", "add task"
        ]):
            return {
                "intent": "task_creation",
                "confidence": 0.7,
                "extracted_info": {
                    "task_description": user_query,
                    "priority": "medium"
                }
            }
        
        # Task query patterns
        elif any(pattern in query_lower for pattern in ["show", "list", "what tasks", "my tasks"]):
            return {"intent": "task_query", "confidence": 0.9}
        
        # Task management patterns
        elif any(pattern in query_lower for pattern in ["complete", "done", "finished", "delete", "remove"]):
            return {"intent": "task_management", "confidence": 0.8}
        
        else:
            return {"intent": "general_assistance", "confidence": 0.5}

    def _handle_intelligent_task_creation(self, user_query: str, intent_analysis: Dict[str, Any], conversation_context: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """ENHANCED: Task creation with conversation context"""
        
        extracted_info = intent_analysis.get("extracted_info", {})
        
        # Generate natural response using conversation context
        task_response = self._generate_natural_task_response(user_query, extracted_info, conversation_context, session_state)
        
        # Display the natural response
        st.write(task_response)
        
        # Store task in session state
        self._store_task_in_session_optionally(user_query, extracted_info, session_state)
        
        # FIXED: Properly log the complete conversation exchange
        self._log_assistant_response(user_query, task_response, "task_creation", session_state)
        
        return {
            "success": True,
            "message": task_response,
            "agent": self.name,
            "type": "natural_language_task_creation"
        }

    def _handle_conversation_query(self, user_query: str, conversation_context: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """NEW: Handle queries about previous conversations"""
        
        st.info("🧠 Looking through our conversation history...")
        
        # Generate response based on conversation context
        conversation_response = self._generate_conversation_memory_response(user_query, conversation_context, session_state)
        
        # Display the response
        st.write(conversation_response)
        
        # Log this conversation query
        self._log_assistant_response(user_query, conversation_response, "conversation_query", session_state)
        
        return {
            "success": True,
            "message": conversation_response,
            "agent": self.name,
            "type": "conversation_memory"
        }

    def _generate_conversation_memory_response(self, user_query: str, conversation_context: str, session_state: Dict[str, Any]) -> str:
        """Generate response about previous conversations"""
        
        memory_prompt = f"""
You are a personal assistant recalling previous conversations with the user.

USER QUERY: "{user_query}"

CONVERSATION HISTORY:
{conversation_context}

TASK: Respond naturally about what you remember from your conversations with this user.

INSTRUCTIONS:
1. Reference specific things from the conversation history
2. Be conversational and natural
3. If you can't find what they're asking about, say so honestly
4. Mention specific details when you have them
5. Ask if they'd like you to look for something more specific

Generate a natural response about your conversation memory:
"""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a personal assistant with good memory of previous conversations. Be natural and helpful."},
                    {"role": "user", "content": memory_prompt}
                ],
                stream=False,
                options={"temperature": 0.7, "num_ctx": 3072}
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            if conversation_context and len(conversation_context) > 50:
                return "I can see we've had several conversations, but I'm having trouble accessing the specific details right now. Could you be more specific about what you're looking for?"
            else:
                return "I don't have much conversation history to reference yet. As we chat more, I'll be better able to remember and reference our previous discussions!"
    def _log_assistant_response(self, user_query: str, response: str, response_type: str, session_state: Dict[str, Any]):
        """FIXED: Properly log assistant responses for conversation tracking"""
        try:
            query_store = session_state.get('query_store')
            session_id = session_state.get('session_id')
            
            if query_store and session_id:
                # Update the most recent query with the response
                with query_store.conn.cursor() as cur:
                    # Update the most recent entry for this session and agent
                    cur.execute("""
                        UPDATE conversation_embeddings 
                        SET response = %s, sql_error = %s
                        WHERE session_id = %s 
                            AND agent_used = %s 
                            AND user_query = %s
                            AND response = 'Processing'
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """, (f"Success: {response_type}", None, session_id, self.name, user_query))
                    
                    if cur.rowcount == 0:
                        # If no processing entry found, create a new one
                        query_store.add_query(
                            session_id=session_id,
                            user_query=user_query,
                            sql_query=None,
                            results=None,
                            response=f"Success: {response_type} - {response[:200]}",
                            sql_error=None,
                            agent_used=self.name
                        )
                    
                    query_store.conn.commit()
                    print(f"✅ Logged assistant response: {response_type}")
                
                # Also log to conversation manager if available
                conv_manager = session_state.get('conv_manager')
                if conv_manager:
                    conv_manager.add_message("user", user_query)
                    conv_manager.add_message("assistant", response)
                
        except Exception as e:
            print(f"⚠️ Failed to log assistant response: {e}")
            # Don't let logging failures break the conversation

    def _generate_natural_task_response(self, user_query: str, extracted_info: Dict[str, Any], conversation_context: str, session_state: Dict[str, Any]) -> str:
        """ENHANCED: Generate natural response with conversation context"""
        
        response_prompt = f"""
You are a helpful personal assistant having a continuing conversation with someone.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT REQUEST: "{user_query}"

EXTRACTED INFORMATION:
- Task: {extracted_info.get('task_description', user_query)}
- Deadline: {extracted_info.get('deadline', 'Not specified')}
- Priority: {extracted_info.get('priority', 'Medium')}
- Participants: {extracted_info.get('participants', 'None')}

This appears to be a request to create a task, reminder, or commitment. 

INSTRUCTIONS:
1. Reference our conversation history naturally when relevant
2. Acknowledge what they've asked for in a conversational way
3. Confirm the key details (time, people involved, etc.)
4. Show that you understand and will help them remember it
5. Ask clarifying questions if needed, but don't be overly inquisitive
6. Be warm, conversational, and helpful - like talking to a colleague
7. If this relates to something we discussed before, mention that connection

Generate a natural, conversational response (2-3 sentences max):
"""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful personal assistant who has natural conversations about tasks and commitments. Reference conversation history when relevant."},
                    {"role": "user", "content": response_prompt}
                ],
                stream=False,
                options={"temperature": 0.7, "num_ctx": 3072}
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            # Fallback response with conversation awareness
            if conversation_context and "Previous:" in conversation_context:
                return f"Got it! I'll help you remember: {user_query}. Building on what we discussed earlier, is there anything specific about timing or other details you'd like me to note?"
            else:
                return f"Perfect! I'll help you remember: {user_query}. Is there anything specific about timing or other details you'd like me to note?"
            

    def _generate_natural_task_response(self, user_query: str, extracted_info: Dict[str, Any], session_state: Dict[str, Any]) -> str:
        """Generate a natural language response for task creation"""
        
        # Get conversation context
        query_store = session_state.get('query_store')
        session_id = session_state.get('session_id')
        context = ""
        
        if query_store and session_id:
            try:
                recent_conversations = query_store.search_query(session_id, user_query, k=3)
                if recent_conversations:
                    context_parts = []
                    for query, _, _ in recent_conversations:
                        if query:
                            context_parts.append(f"Previously: '{query}'")
                    context = "\n".join(context_parts)
            except:
                context = "No recent context available."
        
        response_prompt = f"""
    The user said: "{user_query}"

    Recent conversation context:
    {context}

    This appears to be a request to create a task, reminder, or commitment. Respond naturally as a helpful assistant who:

    1. Acknowledges what they've asked for
    2. Confirms the key details (time, people involved, etc.)
    3. Shows that you understand and will help them remember it
    4. Asks clarifying questions if needed
    5. References related items from conversation history if relevant

    Be warm, conversational, and helpful. Don't be overly structured or robotic.
    """

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful personal assistant who naturally acknowledges and confirms tasks, meetings, and commitments. Be conversational and warm."},
                    {"role": "user", "content": response_prompt}
                ],
                stream=False,
                options={"temperature": 0.7, "num_ctx": 2048}
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            # Fallback response
            return f"Got it! I'll help you remember: {user_query}. Is there anything specific about timing or other details you'd like me to note?"

    def _store_task_in_session_optionally(self, user_query: str, extracted_info: Dict[str, Any], session_state: Dict[str, Any]):
        """Store task in session state for basic persistence"""
        try:
            if 'pa_tasks' not in session_state:
                session_state['pa_tasks'] = []
            
            # Create a simple task record
            simple_task = {
                "id": len(session_state['pa_tasks']) + 1,
                "task": extracted_info.get("task_description", user_query),
                "deadline": extracted_info.get("deadline"),
                "participants": extracted_info.get("participants", []),
                "status": "pending",
                "created_at": datetime.datetime.now().isoformat(),
                "original_query": user_query
            }
            
            session_state['pa_tasks'].append(simple_task)
            print(f"✅ Stored task in session: {simple_task['task']}")
            
        except Exception as e:
            print(f"⚠️ Failed to store task: {e}")


    def _log_task_creation_safely(self, user_query: str, response: str, session_state: Dict[str, Any]):
        """Safely log task creation"""
        try:
            conv_manager = session_state.get('conv_manager')
            query_store = session_state.get('query_store')
            session_id = session_state.get('session_id')
            
            if conv_manager:
                conv_manager.add_message("user", user_query)
                conv_manager.add_message("assistant", response)
            
            if query_store and session_id:
                query_store.add_query(
                    session_id=session_id,
                    user_query=user_query,
                    sql_query=None,
                    results=None,
                    response="Natural Language Task Response",
                    sql_error=None,
                    agent_used=self.name
                )
        except Exception as e:
            # Silent fail
            pass
    
    def _analyze_query_type(self, user_query: str, context: Dict[str, Any]) -> str:
        """Analyze what type of assistance the user needs (fallback method)"""
        query_lower = user_query.lower()
        
        # Check context first
        if context.get('requires_tasks'):
            return "task_management"
        elif context.get('requires_calendar'):
            return "scheduling"
        elif context.get('requires_email'):
            return "email_assistance"
        
        # Analyze query content
        if any(word in query_lower for word in ['task', 'todo', 'to-do', 'remind', 'deadline']):
            return "task_management"
        elif any(word in query_lower for word in ['schedule', 'calendar', 'meeting', 'appointment']):
            return "scheduling"
        elif any(word in query_lower for word in ['email', 'send', 'draft', 'compose']):
            return "email_assistance"
        elif any(word in query_lower for word in ['note', 'write down', 'remember', 'save']):
            return "note_taking"
        elif any(word in query_lower for word in ['goal', 'objective', 'target', 'achieve']):
            return "goal_tracking"
        elif any(word in query_lower for word in ['habit', 'routine', 'daily', 'track']):
            return "habit_tracking"
        elif any(word in query_lower for word in ['productive', 'efficiency', 'tips', 'organize']):
            return "productivity_tips"
        else:
            return "general_assistance"
    
    def _handle_task_management(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task management operations"""
        st.info("✅ Personal Assistant: Task Management")
        
        query_lower = user_query.lower()
        
        # Add task
        if any(word in query_lower for word in ['add', 'create', 'new']):
            return self._add_task(user_query, session_state)
        
        # List tasks
        elif any(word in query_lower for word in ['list', 'show', 'display', 'view']):
            return self._list_tasks(session_state)
        
        # Complete task
        elif any(word in query_lower for word in ['complete', 'done', 'finish']):
            return self._complete_task(user_query, session_state)
        
        # Delete task
        elif any(word in query_lower for word in ['delete', 'remove', 'cancel']):
            return self._delete_task(user_query, session_state)
        
        # Priority management
        elif any(word in query_lower for word in ['priority', 'urgent', 'important']):
            return self._manage_task_priority(user_query, session_state)
        
        else:
            return self._task_help(session_state)
        
    def _handle_task_management_operations(self, user_query: str, intent_analysis: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task operations (complete, delete, etc.) with AI understanding"""
        
        extracted_info = intent_analysis.get("extracted_info", {})
        operation = extracted_info.get("operation", "unknown")
        target_task = extracted_info.get("target_task", "")
        
        tasks = session_state['pa_tasks']
        
        if operation == "complete":
            # Try to find the task
            target_id = None
            
            # Look for task ID in the query
            import re
            id_match = re.search(r'\b(\d+)\b', user_query)
            if id_match:
                target_id = int(id_match.group(1))
            
            if target_id:
                # Complete by ID
                for task in tasks:
                    if task['id'] == target_id and task['status'] == 'pending':
                        task['status'] = 'completed'
                        task['completed_at'] = datetime.datetime.now().isoformat()
                        st.success(f"✅ Completed: {task['task']}")
                        return {
                            "success": True,
                            "message": f"Task {target_id} completed",
                            "agent": self.name,
                            "type": "intelligent_task_complete"
                        }
                
                st.error(f"Task {target_id} not found or already completed")
                return {"success": False, "error": f"Task {target_id} not found", "agent": self.name}
            
            else:
                # Show pending tasks for selection
                pending_tasks = [t for t in tasks if t['status'] == 'pending']
                if pending_tasks:
                    st.write("Which task would you like to complete?")
                    for task in pending_tasks[:10]:
                        st.write(f"**{task['id']}.** {task['task']}")
                else:
                    st.write("No pending tasks to complete!")
                
                return {
                    "success": True,
                    "message": "Please specify task number to complete",
                    "agent": self.name,
                    "type": "task_complete_help"
                }
        
        else:
            # Fallback to original task management
            return self._handle_task_management(user_query, session_state)
    
    def _add_task(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new task"""
        # Extract task description using LLM
        task_prompt = f"""
        Extract the task description from this user request: "{user_query}"
        
        Also determine:
        - Priority (high, medium, low)
        - Deadline (if mentioned)
        - Category (work, personal, health, etc.)
        
        Respond in JSON format:
        {{
            "task": "task description",
            "priority": "medium",
            "deadline": "YYYY-MM-DD or null",
            "category": "category"
        }}
        """
        
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You extract task information and respond in JSON."},
                    {"role": "user", "content": task_prompt}
                ],
                stream=False,
                options={"temperature": 0.3}
            )
            
            task_data = json.loads(response["message"]["content"])
            
            # Create task object
            new_task = {
                "id": len(session_state['pa_tasks']) + 1,
                "task": task_data.get("task", user_query),
                "priority": task_data.get("priority", "medium"),
                "deadline": task_data.get("deadline"),
                "category": task_data.get("category", "general"),
                "status": "pending",
                "created_at": datetime.datetime.now().isoformat(),
                "completed_at": None
            }
            
            session_state['pa_tasks'].append(new_task)
            
            st.success(f"✅ Added task: {new_task['task']}")
            if new_task['deadline']:
                st.info(f"📅 Deadline: {new_task['deadline']}")
            if new_task['priority'] == 'high':
                st.warning(f"🔥 High priority task!")
            
            return {
                "success": True,
                "message": f"Task added: {new_task['task']}",
                "agent": self.name,
                "type": "task_add",
                "task": new_task
            }
            
        except Exception as e:
            # Fallback to simple task creation
            task_description = user_query.replace('add task', '').replace('create task', '').strip()
            if not task_description:
                task_description = user_query
            
            new_task = {
                "id": len(session_state['pa_tasks']) + 1,
                "task": task_description,
                "priority": "medium",
                "deadline": None,
                "category": "general",
                "status": "pending",
                "created_at": datetime.datetime.now().isoformat(),
                "completed_at": None
            }
            
            session_state['pa_tasks'].append(new_task)
            st.success(f"✅ Added task: {new_task['task']}")
            
            return {
                "success": True,
                "message": f"Task added: {new_task['task']}",
                "agent": self.name,
                "type": "task_add",
                "task": new_task
            }
    
    def _list_tasks(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """List all tasks with smart organization"""
        tasks = session_state['pa_tasks']
        
        if not tasks:
            st.write("📝 No tasks found. Try telling me about something you need to do!")
            return {"success": True, "message": "No tasks", "agent": self.name, "type": "task_list"}
        
        # Organize tasks
        pending_tasks = [t for t in tasks if t['status'] == 'pending']
        completed_tasks = [t for t in tasks if t['status'] == 'completed']
        
        # Sort by priority and deadline
        def sort_key(task):
            priority_order = {"high": 0, "medium": 1, "low": 2}
            return (priority_order.get(task['priority'], 1), task.get('deadline') or 'z')
        
        pending_tasks.sort(key=sort_key)
        
        # Display pending tasks
        if pending_tasks:
            st.write("📋 **Your Tasks:**")
            for task in pending_tasks:
                priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📌"}
                emoji = priority_emoji.get(task['priority'], "📌")
                
                deadline_text = ""
                if task['deadline']:
                    deadline_text = f" (Due: {task['deadline']})"
                
                participants_text = ""
                if task.get('participants'):
                    participants_text = f" [with {', '.join(task['participants'])}]"
                
                st.write(f"{emoji} **{task['id']}.** {task['task']}{deadline_text}{participants_text}")
                
                if task['category'] != 'general':
                    st.caption(f"Category: {task['category']}")
                
                # Show original query if different
                if task.get('original_query') and task['original_query'] != task['task']:
                    st.caption(f"📝 Original: {task['original_query']}")
        
        # Display completed tasks (last 5)
        if completed_tasks:
            recent_completed = completed_tasks[-5:]
            st.write("✅ **Recently Completed:**")
            for task in recent_completed:
                st.write(f"~~{task['task']}~~ ✓")
        
        return {
            "success": True,
            "message": f"Showing {len(pending_tasks)} pending and {len(completed_tasks)} completed tasks",
            "agent": self.name,
            "type": "task_list",
            "pending_count": len(pending_tasks),
            "completed_count": len(completed_tasks)
        }

    # Include all other existing methods from the original implementation
    # (I'm keeping the comment here to indicate that all other methods should remain the same)
    
    def _complete_task(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Mark a task as completed"""
        tasks = session_state['pa_tasks']
        
        # Try to extract task ID or description
        task_id = None
        for word in user_query.split():
            if word.isdigit():
                task_id = int(word)
                break
        
        if task_id:
            # Find task by ID
            for task in tasks:
                if task['id'] == task_id and task['status'] == 'pending':
                    task['status'] = 'completed'
                    task['completed_at'] = datetime.datetime.now().isoformat()
                    st.success(f"✅ Completed: {task['task']}")
                    return {
                        "success": True,
                        "message": f"Task {task_id} completed",
                        "agent": self.name,
                        "type": "task_complete"
                    }
            
            st.error(f"Task {task_id} not found or already completed")
            return {"success": False, "error": f"Task {task_id} not found", "agent": self.name}
        
        else:
            # Show pending tasks for selection
            pending_tasks = [t for t in tasks if t['status'] == 'pending']
            if pending_tasks:
                st.write("Which task would you like to complete? Use the task number:")
                for task in pending_tasks[:10]:  # Show max 10
                    st.write(f"**{task['id']}.** {task['task']}")
            else:
                st.write("No pending tasks to complete!")
            
            return {
                "success": True,
                "message": "Please specify task number to complete",
                "agent": self.name,
                "type": "task_complete_help"
            }
    
    def _delete_task(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a task"""
        tasks = session_state['pa_tasks']
        
        # Extract task ID
        task_id = None
        for word in user_query.split():
            if word.isdigit():
                task_id = int(word)
                break
        
        if task_id:
            # Find and remove task
            for i, task in enumerate(tasks):
                if task['id'] == task_id:
                    removed_task = tasks.pop(i)
                    st.success(f"🗑️ Deleted: {removed_task['task']}")
                    return {
                        "success": True,
                        "message": f"Task {task_id} deleted",
                        "agent": self.name,
                        "type": "task_delete"
                    }
            
            st.error(f"Task {task_id} not found")
            return {"success": False, "error": f"Task {task_id} not found", "agent": self.name}
        
        else:
            st.write("Please specify a task number to delete (e.g., 'delete task 3')")
            return {
                "success": True,
                "message": "Please specify task number to delete",
                "agent": self.name,
                "type": "task_delete_help"
            }
    
    def _manage_task_priority(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Manage task priorities"""
        st.info("🔥 Task Priority Management")
        
        # Show tasks by priority
        tasks = session_state['pa_tasks']
        pending_tasks = [t for t in tasks if t['status'] == 'pending']
        
        high_priority = [t for t in pending_tasks if t['priority'] == 'high']
        medium_priority = [t for t in pending_tasks if t['priority'] == 'medium']
        low_priority = [t for t in pending_tasks if t['priority'] == 'low']
        
        if high_priority:
            st.write("🔥 **High Priority:**")
            for task in high_priority:
                st.write(f"  {task['id']}. {task['task']}")
        
        if medium_priority:
            st.write("⚡ **Medium Priority:**")
            for task in medium_priority:
                st.write(f"  {task['id']}. {task['task']}")
        
        if low_priority:
            st.write("📌 **Low Priority:**")
            for task in low_priority:
                st.write(f"  {task['id']}. {task['task']}")
        
        st.info("To change priority, say 'set task 3 to high priority'")
        
        return {
            "success": True,
            "message": "Priority overview displayed",
            "agent": self.name,
            "type": "task_priority"
        }
    
    def _task_help(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Provide task management help"""
        help_text = """
        📝 **Task Management Commands:**
        
        • **Add tasks:** "Add task: Review quarterly reports"
        • **List tasks:** "Show my tasks" or "List all tasks"
        • **Complete tasks:** "Complete task 3" or "Mark task 1 as done"
        • **Delete tasks:** "Delete task 2" or "Remove task 5"
        • **Priority:** "Show high priority tasks" or "Set task 3 to high priority"
        
        **Tips:**
        - Tasks automatically get priorities and categories
        - Mention deadlines for better organization
        - Use task numbers for quick actions
        """
        
        st.write(help_text)
        
        # Show quick stats
        tasks = session_state['pa_tasks']
        pending = len([t for t in tasks if t['status'] == 'pending'])
        completed = len([t for t in tasks if t['status'] == 'completed'])
        
        st.info(f"📊 You have {pending} pending and {completed} completed tasks")
        
        return {
            "success": True,
            "message": "Task management help displayed",
            "agent": self.name,
            "type": "task_help"
        }
    
    def _handle_scheduling(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scheduling and calendar operations"""
        st.info("📅 Personal Assistant: Scheduling")
        
        response = f"Calendar feature: '{user_query}'. " + \
                  "Full calendar integration coming soon! I can help you plan and organize your schedule."
        
        st.write(response)
        return {"success": True, "message": response, "agent": self.name, "type": "scheduling"}
    
    def _handle_email_assistance(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email assistance"""
        st.info("📧 Personal Assistant: Email Assistance")
        
        response = f"Email assistance: '{user_query}'. " + \
                  "I can help you draft emails and organize your communication."
        
        st.write(response)
        return {"success": True, "message": response, "agent": self.name, "type": "email"}
    
    def _handle_note_taking(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle note taking and information storage"""
        st.info("📝 Personal Assistant: Note Taking")
        
        # Simple note storage (can be enhanced)
        if 'save' in user_query.lower() or 'note' in user_query.lower():
            note_content = user_query.replace('save note', '').replace('take note', '').strip()
            note = {
                "id": len(session_state['pa_notes']) + 1,
                "content": note_content,
                "timestamp": datetime.datetime.now().isoformat()
            }
            session_state['pa_notes'].append(note)
            st.success(f"📝 Note saved: {note_content[:50]}...")
            
        return {"success": True, "message": "Note feature", "agent": self.name, "type": "notes"}
    
    def _handle_goal_tracking(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle goal tracking and progress monitoring"""
        st.info("🎯 Personal Assistant: Goal Tracking")
        
        response = "Goal tracking feature. I can help you set and track your objectives!"
        st.write(response)
        return {"success": True, "message": response, "agent": self.name, "type": "goals"}
    
    def _handle_habit_tracking(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle habit tracking"""
        st.info("🔄 Personal Assistant: Habit Tracking")
        
        response = "Habit tracking feature. I can help you build and maintain good habits!"
        st.write(response)
        return {"success": True, "message": response, "agent": self.name, "type": "habits"}
    
    def _handle_productivity_tips(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Provide productivity tips and advice"""
        st.info("💡 Personal Assistant: Productivity Tips")
        
        tips_prompt = f"""
        The user asked: "{user_query}"
        
        Provide 3-5 practical productivity tips related to their question.
        Be specific and actionable.
        """
        
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a productivity expert providing actionable advice."},
                    {"role": "user", "content": tips_prompt}
                ],
                stream=True,
                options={"temperature": 0.7}
            )
            
            tips = response["message"]["content"]
            st.write(tips)
            
            return {
                "success": True,
                "message": tips,
                "agent": self.name,
                "type": "productivity_tips"
            }
            
        except Exception as e:
            fallback_tips = """
            💡 **Quick Productivity Tips:**
            • Use the Pomodoro Technique (25 min work, 5 min break)
            • Prioritize tasks using the Eisenhower Matrix
            • Batch similar tasks together
            • Set specific, measurable goals
            • Take regular breaks to maintain focus
            """
            st.write(fallback_tips)
            return {"success": True, "message": "Productivity tips provided", "agent": self.name}
    
    def _handle_general_assistance(self, user_query: str, conversation_context: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """ENHANCED: General assistance with conversation context"""
        
        st.info("🤖 Personal Assistant: General Help")
        
        assistance_prompt = f"""
You are a helpful personal assistant having a continuing conversation.

CONVERSATION CONTEXT:
{conversation_context}

CURRENT REQUEST: "{user_query}"

Provide helpful, actionable advice or assistance. Be conversational and reference our previous discussions when relevant.
Focus on productivity, organization, and personal effectiveness.
"""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful personal assistant who remembers previous conversations and provides contextual help."},
                    {"role": "user", "content": assistance_prompt}
                ],
                stream=False,
                options={"temperature": 0.7, "num_ctx": 3072}
            )
            
            assistant_response = response["message"]["content"]
            st.write(assistant_response)
            
            # Log this exchange
            self._log_assistant_response(user_query, assistant_response, "general_assistance", session_state)
            
            return {"success": True, "message": assistant_response, "agent": self.name, "type": "general"}
            
        except Exception as e:
            error_msg = "I'm here to help with productivity and organization. What would you like assistance with?"
            st.write(error_msg)
            self._log_assistant_response(user_query, error_msg, "general_assistance", session_state)
            return {"success": True, "message": error_msg, "agent": self.name}

    # Include other methods from original implementation...
    # (I'm including the key methods that need to be updated, others can remain the same)
    
    def _handle_task_listing(self, user_query: str, conversation_context: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """ENHANCED: List tasks with conversation context"""
        tasks = session_state['pa_tasks']
        
        if not tasks:
            response = "I don't see any tasks yet. Based on our conversations, would you like me to help you organize some priorities?"
            st.write(response)
            self._log_assistant_response(user_query, response, "task_listing", session_state)
            return {"success": True, "message": response, "agent": self.name, "type": "task_list"}
        
        # Display tasks naturally
        pending_tasks = [t for t in tasks if t['status'] == 'pending']
        completed_tasks = [t for t in tasks if t['status'] == 'completed']
        
        if conversation_context and "Similar:" in conversation_context:
            response_start = "Looking at your tasks, continuing from what we discussed:"
        else:
            response_start = "Here's what you have on your plate:"
        
        st.write(response_start)
        
        # Display pending tasks
        if pending_tasks:
            st.write("📋 **Your Tasks:**")
            for task in pending_tasks:
                priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📌"}
                emoji = priority_emoji.get(task.get('priority', 'medium'), "📌")
                
                deadline_text = ""
                if task.get('deadline'):
                    deadline_text = f" (Due: {task['deadline']})"
                
                st.write(f"{emoji} **{task['id']}.** {task['task']}{deadline_text}")
        
        # Display recent completed tasks
        if completed_tasks:
            recent_completed = completed_tasks[-3:]
            st.write("✅ **Recently Completed:**")
            for task in recent_completed:
                st.write(f"~~{task['task']}~~ ✓")
        
        response = f"Showing {len(pending_tasks)} pending tasks"
        self._log_assistant_response(user_query, response, "task_listing", session_state)
        
        return {
            "success": True,
            "message": response,
            "agent": self.name,
            "type": "task_list",
            "pending_count": len(pending_tasks)
        }

    def _handle_intelligent_scheduling(self, user_query: str, intent_analysis: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle scheduling with natural language response based on conversation context"""
        
        extracted_info = intent_analysis.get("extracted_info", {})
        time_period = extracted_info.get("time_period", "today")
        
        st.info("📅 **Your Schedule & Agenda**")
        
        # Get conversation context for natural language response
        schedule_context = self._get_schedule_context_from_conversations(user_query, time_period, session_state)
        
        # Generate natural language response
        schedule_response = self._generate_natural_schedule_response(user_query, time_period, schedule_context, session_state)
        
        # Display the natural response
        st.write(schedule_response)
        
        # Log this query
        self._log_schedule_query_safely(user_query, [], time_period, session_state)
        
        return {
            "success": True,
            "message": schedule_response,
            "agent": self.name,
            "type": "natural_language_scheduling",
            "time_period": time_period
        }
    # ALSO ADD this missing import at the top of enhanced_personal_assistant.py

    def _log_schedule_query_safely(self, user_query: str, results: list, time_period: str, session_state: Dict[str, Any]):
        """Safely log the schedule query without causing errors"""
        try:
            conv_manager = session_state.get('conv_manager')
            query_store = session_state.get('query_store')
            session_id = session_state.get('session_id')
            
            if conv_manager and query_store and session_id:
                conv_manager.add_message("user", user_query)
                response_msg = f"Provided natural language schedule overview for {time_period}"
                conv_manager.add_message("assistant", response_msg)
                
                query_store.add_query(
                    session_id=session_id,
                    user_query=user_query,
                    sql_query=None,
                    results=None,  # No structured results, just natural language
                    response="Natural Language Schedule Response",
                    sql_error=None,
                    agent_used=self.name
                )
        except Exception as e:
            # Silently fail - don't disrupt the user experience
            pass
        
    def _get_schedule_context_from_conversations(self, user_query: str, time_period: str, session_state: Dict[str, Any]) -> str:
        """Get relevant conversation context for schedule queries"""
        
        query_store = session_state.get('query_store')
        session_id = session_state.get('session_id')
        
        if not query_store or not session_id:
            return "No conversation history available."
        
        try:
            # Use the existing vector search to find relevant conversations
            # This searches semantically for related conversations
            relevant_conversations = query_store.search_query(session_id, user_query, k=5)
            
            if not relevant_conversations:
                # Fallback: get recent conversations from this agent
                with query_store.conn.cursor() as cur:
                    cur.execute("""
                        SELECT user_query, response, created_at
                        FROM conversation_embeddings
                        WHERE session_id = %s 
                            AND (agent_used LIKE %s OR user_query LIKE '%meeting%' OR user_query LIKE '%schedule%' OR user_query LIKE '%today%')
                        ORDER BY created_at DESC
                        LIMIT 10
                    """, (session_id, f'%{self.name}%'))
                    
                    recent_conversations = cur.fetchall()
                    
                    # Format as context
                    context_parts = []
                    for query, response, created_at in recent_conversations:
                        if query and len(query.strip()) > 3:  # Skip very short queries
                            context_parts.append(f"User said: '{query}' (Response: {response})")
                    
                    return "\n".join(context_parts[-5:])  # Last 5 relevant conversations
            else:
                # Format vector search results
                context_parts = []
                for query, sql_query, results in relevant_conversations:
                    if query and len(query.strip()) > 3:
                        context_parts.append(f"User previously asked: '{query}'")
                
                return "\n".join(context_parts)
        
        except Exception as e:
            st.warning(f"Could not retrieve conversation context: {e}")
            return "Unable to access conversation history."    

    def _generate_natural_schedule_response(self, user_query: str, time_period: str, context: str, session_state: Dict[str, Any]) -> str:
        """Generate a natural language response about the user's schedule"""
        
        # Also include session state tasks if available
        session_tasks = session_state.get('pa_tasks', [])
        session_context = ""
        
        if session_tasks:
            pending_tasks = [t for t in session_tasks if t.get('status') == 'pending']
            if pending_tasks:
                task_summaries = []
                for task in pending_tasks[:5]:  # Limit to 5 most recent
                    task_summary = task.get('task', 'Unknown task')
                    if task.get('deadline'):
                        task_summary += f" (due: {task['deadline']})"
                    task_summaries.append(task_summary)
                session_context = f"\nCurrent session tasks: {'; '.join(task_summaries)}"
        
        # Create a comprehensive prompt for natural language generation
        response_prompt = f"""
    You are a helpful personal assistant responding to a schedule query.

    User Query: "{user_query}"
    Time Period: {time_period}

    Conversation Context:
    {context}

    Session Context:
    {session_context}

    Based on the conversation history and current context, provide a natural, conversational response about the user's schedule for {time_period}. 

    Guidelines:
    1. Be conversational and personal
    2. Reference specific items from the conversation history when relevant
    3. If you see meetings, tasks, or commitments mentioned in the context, include them naturally
    4. If no specific items are found, acknowledge this but be encouraging
    5. Don't make up specific details that aren't in the context
    6. Use a warm, helpful tone
    7. Suggest next steps or ask follow-up questions when appropriate

    Respond as if you're having a natural conversation about their schedule.
    """

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a warm, helpful personal assistant discussing someone's schedule. Be conversational and reference specific details from their conversation history when available."},
                    {"role": "user", "content": response_prompt}
                ],
                stream=False,
                options={"temperature": 0.7, "num_ctx": 3072}
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            # Fallback response if LLM fails
            if "today" in time_period.lower():
                if context and len(context) > 20:
                    return f"Looking at our conversation history, I can see you've mentioned some tasks and commitments. For today's schedule, let me know if you'd like me to help you organize anything specific!"
                else:
                    return f"I don't see any specific items scheduled for today in our conversation history. What would you like to work on or schedule for today?"
            else:
                return f"I'd be happy to help you check your schedule for {time_period}. Based on our conversations, let me know what specific commitments or tasks you're thinking about!"

    # AND add this enhanced debugging method:
    def debug_database_structure(self, session_state: Dict[str, Any]):
        """Debug the database structure to understand the tuple issue"""
        
        query_store = session_state.get('query_store')
        session_id = session_state.get('session_id')
        
        if not query_store or not session_id:
            st.error("❌ No database connection or session ID")
            return
        
        try:
            with query_store.conn.cursor() as cur:
                # Check table schema
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'conversation_embeddings'
                    ORDER BY ordinal_position
                """)
                schema = cur.fetchall()
                
                st.write("🔍 **Database Schema:**")
                for col_name, data_type, is_nullable in schema:
                    st.write(f"  - {col_name}: {data_type} (nullable: {is_nullable})")
                
                # Check actual data
                cur.execute("""
                    SELECT user_query, response, agent_used, created_at,
                        CASE WHEN results IS NULL THEN 'NULL' ELSE 'HAS_DATA' END as results_status
                    FROM conversation_embeddings
                    WHERE session_id = %s
                    ORDER BY created_at DESC LIMIT 5
                """, (session_id,))
                
                rows = cur.fetchall()
                st.write(f"\n🔍 **Recent Data ({len(rows)} rows):**")
                for i, row in enumerate(rows):
                    st.write(f"  Row {i+1}: {len(row)} columns")
                    if len(row) >= 5:
                        st.write(f"    Query: {row[0][:40]}...")
                        st.write(f"    Response: {row[1]}")
                        st.write(f"    Agent: {row[2]}")
                        st.write(f"    Time: {row[3]}")
                        st.write(f"    Results: {row[4]}")
                    else:
                        st.write(f"    Row data: {row}")
                    st.write("")
                    
        except Exception as e:
            st.error(f"Database debug failed: {e}")

    def _get_tasks_for_date_enhanced(self, tasks: List[Dict], target_date: datetime.date) -> List[Dict]:
        """Enhanced task matching for specific dates"""
        target_str = target_date.isoformat()
        relevant_tasks = []
        
        for task in tasks:
            if task.get('status') != 'pending':
                continue
            
            # Check deadline field
            deadline = task.get('deadline', '')
            if deadline and target_str in str(deadline):
                relevant_tasks.append(task)
                continue
            
            # Check if created today and no specific deadline
            if not deadline:
                created_at = task.get('created_at', '')
                if target_str in created_at:
                    relevant_tasks.append(task)
                    continue
            
            # Check task text for time indicators
            task_text = task.get('task', '').lower()
            original_query = task.get('original_query', '').lower()
            
            # If today and task mentions current time
            if target_str == datetime.date.today().isoformat():
                if any(keyword in task_text or keyword in original_query 
                    for keyword in ['today', 'this morning', 'this afternoon', 'tonight', '4 pm', '4pm']):
                    relevant_tasks.append(task)
        
        return sorted(relevant_tasks, key=lambda x: x.get('deadline', x.get('created_at', '')))

    def debug_task_storage(self, session_state: Dict[str, Any]):
        """Debug method to check task storage"""
        
        st.write("🔍 **Task Storage Debug:**")
        
        # Check session state
        tasks = session_state.get('pa_tasks', [])
        st.write(f"📊 Session state: {len(tasks)} tasks")
        for i, task in enumerate(tasks[:5]):  # Show first 5
            st.write(f"  {i+1}. {task.get('task', 'Unknown')} (deadline: {task.get('deadline', 'None')})")
        
        # Check database
        query_store = session_state.get('query_store')
        session_id = session_state.get('session_id')
        
        if query_store and session_id:
            try:
                with query_store.conn.cursor() as cur:
                    cur.execute("""
                        SELECT user_query, response, agent_used, created_at
                        FROM conversation_embeddings
                        WHERE session_id = %s AND agent_used LIKE %s
                        ORDER BY created_at DESC LIMIT 5
                    """, (session_id, f'%Personal%'))
                    
                    db_entries = cur.fetchall()
                    st.write(f"📊 Database: {len(db_entries)} PA entries")
                    for i, entry in enumerate(db_entries):
                        st.write(f"  {i+1}. {entry[0][:40]}... | {entry[1]} | {entry[3]}")
                        
            except Exception as e:
                st.error(f"Database check failed: {e}")


    def _get_tasks_for_week(self, tasks: List[Dict]) -> List[Dict]:
        """Get tasks for the current week"""
        today = datetime.date.today()
        week_start = today - datetime.timedelta(days=today.weekday())
        week_end = week_start + datetime.timedelta(days=6)
        
        relevant_tasks = []
        for task in tasks:
            if task.get('status') != 'pending':
                continue
                
            deadline = task.get('deadline')
            if deadline:
                try:
                    # Extract date from deadline string
                    if week_start.isoformat() <= deadline <= week_end.isoformat():
                        relevant_tasks.append(task)
                except:
                    # If deadline parsing fails, include task anyway
                    relevant_tasks.append(task)
        
        return sorted(relevant_tasks, key=lambda x: x.get('deadline', ''))

    def _display_schedule_view(self, tasks: List[Dict], time_period: str):
        """Display tasks in a schedule-like format"""
        
        if not tasks:
            return
        
        # Group by time if deadlines have time info
        timed_tasks = []
        untimed_tasks = []
        
        for task in tasks:
            deadline = task.get('deadline', '')
            if ':' in str(deadline):  # Has time component
                timed_tasks.append(task)
            else:
                untimed_tasks.append(task)
        
        # Display timed tasks first (sorted by time)
        if timed_tasks:
            st.write("⏰ **Scheduled Times:**")
            for task in sorted(timed_tasks, key=lambda x: x.get('deadline', '')):
                self._display_schedule_item(task)
        
        # Display untimed tasks
        if untimed_tasks:
            st.write("📋 **Tasks to Complete:**")
            for task in untimed_tasks:
                self._display_schedule_item(task)

    def _display_schedule_item(self, task: Dict):
        """Display a single schedule item"""
        priority_emoji = {"high": "🔥", "medium": "⚡", "low": "📌"}
        emoji = priority_emoji.get(task.get('priority'), "📌")
        
        # Format deadline
        deadline = task.get('deadline', '')
        if deadline:
            if ':' in deadline:
                # Has time - extract and format
                if 'today' in deadline.lower():
                    time_str = deadline.split()[-1] if ':' in deadline else deadline
                    deadline_display = f"at {time_str}"
                else:
                    deadline_display = f"by {deadline}"
            else:
                deadline_display = f"by {deadline}"
        else:
            deadline_display = ""
        
        # Display item
        task_text = task.get('task', 'Untitled task')
        participants = task.get('participants', [])
        location = task.get('location')
        
        # Build display string
        display_parts = [f"{emoji} **{task_text}**"]
        
        if deadline_display:
            display_parts.append(f"({deadline_display})")
        
        if participants:
            display_parts.append(f"with {', '.join(participants)}")
        
        if location:
            display_parts.append(f"at {location}")
        
        st.write(" ".join(display_parts))
        
        # Show category if not general
        if task.get('category') and task['category'] != 'general':
            st.caption(f"Category: {task['category']}")

    def _provide_schedule_suggestions(self, tasks: List[Dict], time_period: str):
        """Provide contextual suggestions based on schedule"""
        
        high_priority_count = len([t for t in tasks if t.get('priority') == 'high'])
        meeting_count = len([t for t in tasks if t.get('category') == 'meeting'])
        
        suggestions = []
        
        if high_priority_count > 0:
            suggestions.append(f"🔥 You have {high_priority_count} high-priority items")
        
        if meeting_count > 0:
            suggestions.append(f"👥 {meeting_count} meetings scheduled")
        
        if len(tasks) > 5:
            suggestions.append("📊 Busy day ahead - consider prioritizing")
        elif len(tasks) == 0:
            suggestions.append("🎯 Free time - perfect for planning ahead")
        
        if suggestions:
            st.info("💡 **Schedule Insights:** " + " • ".join(suggestions))

    def _handle_task_management_operations(self, user_query: str, intent_analysis: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task operations (complete, delete, etc.) with AI understanding"""
        
        extracted_info = intent_analysis.get("extracted_info", {})
        operation = extracted_info.get("operation", "unknown")
        target_task = extracted_info.get("target_task", "")
        
        tasks = session_state['pa_tasks']
        
        if operation == "complete":
            # Try to find the task
            target_id = None
            
            # Look for task ID in the query
            import re
            id_match = re.search(r'\b(\d+)\b', user_query)
            if id_match:
                target_id = int(id_match.group(1))
            
            if target_id:
                # Complete by ID
                for task in tasks:
                    if task['id'] == target_id and task['status'] == 'pending':
                        task['status'] = 'completed'
                        task['completed_at'] = datetime.datetime.now().isoformat()
                        st.success(f"✅ Completed: {task['task']}")
                        return {
                            "success": True,
                            "message": f"Task {target_id} completed",
                            "agent": self.name,
                            "type": "intelligent_task_complete"
                        }
                
                st.error(f"Task {target_id} not found or already completed")
                return {"success": False, "error": f"Task {target_id} not found", "agent": self.name}
            
            else:
                # Show pending tasks for selection
                pending_tasks = [t for t in tasks if t['status'] == 'pending']
                if pending_tasks:
                    st.write("Which task would you like to complete?")
                    for task in pending_tasks[:10]:
                        st.write(f"**{task['id']}.** {task['task']}")
                else:
                    st.write("No pending tasks to complete!")
                
                return {
                    "success": True,
                    "message": "Please specify task number to complete",
                    "agent": self.name,
                    "type": "task_complete_help"
                }
        
        else:
            # Fallback to original task management
            return self._handle_task_management(user_query, session_state)

    def _handle_cross_agent_follow_up(self, user_query: str, follow_up_context: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle follow-ups that come from other agents with intelligent task creation"""
        st.info(f"🤖 Personal Assistant: Following up on {follow_up_context['last_agent']} query")
        
        last_query = follow_up_context.get("last_query", "")
        last_results = follow_up_context.get("last_results", [])
        
        # Create context-aware query for AI analysis
        enhanced_query = f"Based on previous query '{last_query}', user now wants: {user_query}"
        
        # Use AI to understand the cross-agent intent
        intent_analysis = self._analyze_user_intent_with_ai(enhanced_query, session_state)
        
        if intent_analysis.get("intent") == "task_creation":
            # Add context to the task
            extracted_info = intent_analysis.get("extracted_info", {})
            extracted_info["context"] = f"Related to: {last_query}"
            intent_analysis["extracted_info"] = extracted_info
            
            return self._handle_intelligent_task_creation(enhanced_query, intent_analysis, session_state)
        else:
            return self._handle_general_assistance(enhanced_query, session_state)

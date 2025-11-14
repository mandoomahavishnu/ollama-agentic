"""
Enhanced main.py with comprehensive workflow debugging - COMPLETE FIXED VERSION
"""
import streamlit as st
def _render_tabular_results_if_any(tool_output, user_query_label="query"):
    try:
        rows = tool_output.get("results") or []
        method = (tool_output.get("method") or "").lower()
        if not rows or method not in ("api", "nl2sql"):
            return

        import pandas as pd, io, datetime as _dt
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # Download buttons
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        left, right = st.columns(2)
        with left:
            st.download_button(
                "📄 Download CSV",
                csv_buf.getvalue(),
                file_name=f"{method}_results_{stamp}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with right:
            xls_buf = io.BytesIO()
            with pd.ExcelWriter(xls_buf, engine="openpyxl") as w:
                df.to_excel(w, sheet_name="Results", index=False)
            st.download_button(
                "📊 Download Excel",
                xls_buf.getvalue(),
                file_name=f"{method}_results_{stamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except Exception as _e:
        st.caption(f"Results preview unavailable: {_e}")
import datetime
import uuid
from typing import Dict, Any, Optional
# Import all necessary modules
from config import TABLE_NAMES
from database import get_database_schema, get_first_three_rows, execute_sql_and_return_rows
from query_store import QueryStore
from conversation_manager import ConversationManager
from schema_analysis import identify_relevant_tables_and_columns, debug_table_selection
from sql_operations import (
    validate_sql, refine_sql_with_context_structured, generate_sql_chain_of_thought,
    maybe_add_po_filter, build_column_comment_map
)
from nlp_utils import detect_follow_up_query, user_references_prev_pos
from ui_components import (
    display_query_results, display_feedback_buttons, display_conversation_history,
    setup_sidebar_controls, display_error_handling
)
from few_shot_examples import few_shot_examples
from query_router import EnhancedQueryRouter
from agent_implementations import AgentManager, EnhancedPersonalAssistantAgent
from comprehensive_llm_debugger import initialize_comprehensive_llm_debugger
from document_rag_agent import DocumentRAGAgent, display_document_management_panel
from unified_context_manager import UnifiedContextManager
# FIXED: Import the contextual router with proper fallback
try:
    from query_router import ContextualEnhancedQueryRouter
    CONTEXTUAL_ROUTER_AVAILABLE = True
except ImportError:
    # Fallback to regular router if contextual router not available
    ContextualEnhancedQueryRouter = EnhancedQueryRouter
    CONTEXTUAL_ROUTER_AVAILABLE = False
    print("⚠️ Contextual router not available, using standard router")

# Import contextual engineering if available
try:
    from contextual_engineering import ContextualEngineeringManager
    CONTEXTUAL_ENGINEERING_AVAILABLE = True
except ImportError:
    CONTEXTUAL_ENGINEERING_AVAILABLE = False
    print("⚠️ Contextual engineering not available")

# DEBUGGER IMPORTS: Choose ONE of these options:
# OPTION 1: Use the new comprehensive workflow debugger (RECOMMENDED)
try:
    from comprehensive_debugger import initialize_workflow_debugger, export_workflow_data
    DEBUGGER_AVAILABLE = True
except ImportError:
    DEBUGGER_AVAILABLE = False
    print("⚠️ Comprehensive debugger not available")

def initialize_session_state():
    """UNIFIED initialization function - handles both standard and contextual modes"""
    
    # Basic session initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "query_store" not in st.session_state or not isinstance(st.session_state.query_store, QueryStore):
        st.session_state.query_store = QueryStore()
        # Fix any transaction issues immediately
        try:
            st.session_state.query_store.fix_transaction_state()
        except Exception as e:
            print(f"Initial transaction fix failed: {e}")

    if "unified_context_manager" not in st.session_state:
        from unified_context_manager import UnifiedContextManager
        st.session_state.unified_context_manager = UnifiedContextManager(
            query_store=st.session_state.query_store,
            session_id=st.session_state.session_id,
            cache_ttl=300
        )
        print("✅ Unified Context Manager initialized")



    if "conv_manager" not in st.session_state:
        st.session_state.conv_manager = ConversationManager(
            session_id=st.session_state.session_id,
            query_store=st.session_state.query_store,
            unified_context_manager=st.session_state.unified_context_manager
        )
        print("✅ ConversationManager initialized with unified context")
    # Router initialization - try contextual first, fallback to standard
    if "router" not in st.session_state:
        try:
            if CONTEXTUAL_ROUTER_AVAILABLE:
                st.session_state.router = ContextualEnhancedQueryRouter(
                    query_store=st.session_state.query_store,
                    top_k_examples=5,
                    min_similarity=0.25,
                    unified_context_manager=st.session_state.unified_context_manager
                )
                st.session_state.router_type = "contextual"
            else:
                st.session_state.router = EnhancedQueryRouter(
                    query_store=st.session_state.query_store,
                    unified_context_manager=st.session_state.unified_context_manager

                )
                st.session_state.router_type = "standard"
        except Exception as e:
            # Ultimate fallback
            st.session_state.router = EnhancedQueryRouter(
                query_store=st.session_state.query_store,
                unified_context_manager=st.session_state.unified_context_manager
            )
            st.session_state.router_type = "fallback"
            print(f"⚠️ Router initialization fallback: {e}")
    
    # Agent manager initialization
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = AgentManager(st.session_state.query_store, unified_context_manager=st.session_state.unified_context_manager)
        print("✅ AgentManager initialized with unified context")
    # Initialize contextual engineering if available
    if "context_manager" not in st.session_state and CONTEXTUAL_ENGINEERING_AVAILABLE:
        try:
            st.session_state.context_manager = ContextualEngineeringManager(
                query_store=st.session_state.query_store
            )
            st.session_state.contextual_engineering_enabled = True
        except Exception as e:
            st.session_state.contextual_engineering_enabled = False
            print(f"⚠️ Contextual engineering initialization failed: {e}")

         
    # Standard session state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

@st.cache_data(ttl=600)
def init_schema_and_examples():
    """Initialize schema data"""
    schema_data = get_database_schema(TABLE_NAMES)
    return schema_data

def init_table_examples():
    """Initialize table examples"""
    table_examples_map = {}
    for t in TABLE_NAMES:
        ex_rows = get_first_three_rows(t)
        table_examples_map[t] = ex_rows
        st.session_state.query_store.store_table_example_rows(t, ex_rows)
    return table_examples_map

def initialize_data():
    """Initialize all data structures - ENHANCED with Excel table embeddings"""
    if "full_schema" not in st.session_state:
        st.session_state.full_schema = init_schema_and_examples()
        st.session_state.query_store.store_schema_in_pgvector(st.session_state.full_schema)

    if "col_comment_map" not in st.session_state:
        st.session_state.col_comment_map = build_column_comment_map(st.session_state.full_schema)

    if "table_examples" not in st.session_state:
        st.session_state.table_examples = init_table_examples()

    # Initialize few-shot examples
    st.session_state.query_store.initialize_few_shot_examples(few_shot_examples)
    
    # NEW: Initialize enhanced table embeddings using Excel descriptions
    if "enhanced_table_embeddings_initialized" not in st.session_state:
        try:
            # Use the Excel file in your project directory
            excel_path = "table_names.xlsx"  # Make sure this file is in your project root
            
            st.session_state.query_store.store_table_embeddings(
                TABLE_NAMES, 
                excel_file_path=excel_path
            )
            st.session_state.enhanced_table_embeddings_initialized = True
            st.success("✅ Enhanced table embeddings initialized using Excel descriptions!")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize table embeddings from Excel: {e}")
            # Fallback to basic method (your original method)
            try:
                st.session_state.query_store.store_table_embeddings(TABLE_NAMES)
                st.session_state.enhanced_table_embeddings_initialized = True
                st.warning("⚠️ Using fallback descriptions instead of Excel file")
            except Exception as e2:
                st.error(f"❌ Complete failure: {e2}")
                st.session_state.enhanced_table_embeddings_initialized = False

    if "rag_initialized" not in st.session_state:
        try:
            # Initialize document storage tables
            st.session_state.query_store.initialize_document_tables()
            
            # Initialize routing examples FIRST (includes RAG examples)
            routing_count = st.session_state.query_store.initialize_routing_examples()
            
            # Then initialize RAG routing examples
            st.session_state.query_store.initialize_rag_routing_examples()
            
            st.session_state.rag_initialized = True
            st.success(f"✅ RAG system initialized! Routing examples: {routing_count}")
            
        except Exception as e:
            st.error(f"❌ RAG initialization failed: {e}")
            st.session_state.rag_initialized = False

    if "api_vectors_initialized" not in st.session_state:
        try:
            from initialize_api_agent import initialize_api_vectors
            
            with st.spinner("Initializing API Agent..."):
                success = initialize_api_vectors(st.session_state.query_store)
            
            if success:
                st.session_state.api_vectors_initialized = True
                st.success("✅ API Agent initialized with vector embeddings!")
            else:
                st.error("❌ Failed to initialize API Agent")
                st.session_state.api_vectors_initialized = False
                
        except Exception as e:
            st.error(f"API Agent initialization error: {e}")
            st.session_state.api_vectors_initialized = False

def contextual_send_message():
    """
    ENHANCED message processing - NEW FLOW:
    User → Router → Tool (silent) → General Chat (presentation) → User
    """
    st.write("🔍 Processing your request...")
    user_input = st.session_state.user_input.strip()
    
    if not user_input:
        return

    # Get components
    router = st.session_state.router
    agent_manager = st.session_state.agent_manager
    router_type = st.session_state.get("router_type", "standard")
    
    # STEP 1: ROUTE THE QUERY
    try:
        st.write("🧭 Step 1: Routing your query...")
        if hasattr(router, 'route_query_with_api_priority'):
            agent_type, routing_info = router.route_query_with_api_priority(
                user_input, 
                st.session_state
            )
            
            # Display API confidence if available
            if 'api_confidence' in routing_info or 'api_first' in routing_info:
                api_conf = routing_info.get('api_confidence', routing_info.get('confidence', 0))
                st.caption(f"🔌 API Signal Strength: {api_conf:.2f}")        
        # Route query
        elif router_type == "contextual" and hasattr(router, 'route_query_contextual'):
            agent_type, routing_info = router.route_query_contextual(user_input, st.session_state)
        else:
            agent_type, routing_info = router.route_query(user_input, st.session_state)
        
        # Display routing decision
        confidence = routing_info.get("confidence", 0.0)
        method = routing_info.get("method", "unknown")
        
        if confidence > 0.8:
            st.success(f"🎯 **Routing Decision**: {agent_type} (confidence: {confidence:.3f})")
        elif confidence > 0.6:
            st.info(f"🤔 **Routing Decision**: {agent_type} (confidence: {confidence:.3f})")
        else:
            st.warning(f"❓ **Routing Decision**: {agent_type} (confidence: {confidence:.3f})")
        
        st.caption(f"🔧 Method: {method}")
        
    except Exception as e:
        st.error(f"❌ Routing error: {e}")
        # Fallback
        agent_type = "general_chat"
        routing_info = {
            "method": "fallback",
            "confidence": 0.5,
            "error": str(e)
        }

    # STEP 2: EXECUTE TOOL OR AGENT
    try:
        # Determine if we need tool execution followed by general_chat presentation
        tools_requiring_formatting = ["nl2sql", "document_rag", "web_search", "api_endpoint"]
        use_new_flow = agent_type in tools_requiring_formatting
        
        if use_new_flow:
            # NEW FLOW: Tool → General Chat
            st.info(f"📊 Step 2: Executing {agent_type} tool...")
            
            # Execute tool silently (no display)
            tool_output = agent_manager.execute_tool_with_fallback(
                tool_name=agent_type,
                user_query=user_input,
                routing_info=routing_info,
                session_state=st.session_state
            )
            _render_tabular_results_if_any(tool_output, user_query_label=user_input)
            if tool_output.get("fallback_used"):
                st.info(f"🔄 Used fallback: {tool_output['primary_tool']} → {tool_output['fallback_from']}")
                st.caption(f"Reason: {tool_output['fallback_reason']}")            
            # Show what tool found
            if tool_output.get("success"):
                result_summary = tool_output.get("message", "Tool executed successfully")
                st.success(f"✅ {result_summary}")
            else:
                st.warning(f"⚠️ Tool issue: {tool_output.get('error', 'Unknown')}")
            
            # Step 3: Format with general_chat
            st.info("💬 Step 3: Formatting conversational response...")
            
            result = agent_manager.process_with_general_chat(
                user_query=user_input,
                tool_output=tool_output,
                routing_info=routing_info,
                session_state=st.session_state
            )
            
            st.success("✅ Response complete!")
            
        else:
            # DIRECT EXECUTION: general_chat or personal_assistant
            st.info(f"💬 Processing with {agent_type}...")
            
            agent = agent_manager.agents.get(agent_type)
            result = agent.process_query(
                user_query=user_input,
                routing_info=routing_info,
                session_state=st.session_state,
                silent_mode=False
            )
        
        # Update learning
        if result.get("success"):
            if hasattr(router, 'context_manager'):
                router.context_manager.update_conversation_context(
                    st.session_state.session_id, user_input, agent_type, result
                )
            
            if hasattr(router, 'update_follow_up_context'):
                router.update_follow_up_context(user_input, agent_type, result)
        
    except Exception as e:
        st.error(f"❌ Processing error: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    # Clear input
    st.session_state.user_input = ""


# Alternative: Simple toggle for users who want old behavior
def contextual_send_message_with_toggle():
    """Version with user toggle between new and old flow"""
    
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return
    
    # Check user preference
    use_new_flow = st.session_state.get("use_unified_presentation", True)
    
    router = st.session_state.router
    agent_manager = st.session_state.agent_manager
    router_type = st.session_state.get("router_type", "standard")
    
    # Route
    try:
        if router_type == "contextual" and hasattr(router, 'route_query_contextual'):
            agent_type, routing_info = router.route_query_contextual(user_input, st.session_state)
        else:
            agent_type, routing_info = router.route_query(user_input, st.session_state)
    except Exception as e:
        st.error(f"Routing error: {e}")
        agent_type = "general_chat"
        routing_info = {"method": "fallback", "confidence": 0.5}
    
    # Process
    try:
        if use_new_flow and agent_type in ["nl2sql", "document_rag", "web_search"]:
            # New flow
            st.info(f"🔧 {agent_type} → 💬 general_chat")
            
            tool_output = agent_manager.execute_tool(
                tool_name=agent_type,
                user_query=user_input,
                routing_info=routing_info,
                session_state=st.session_state
            )
            _render_tabular_results_if_any(tool_output, user_query_label=user_input)
            result = agent_manager.process_with_general_chat(
                user_query=user_input,
                tool_output=tool_output,
                routing_info=routing_info,
                session_state=st.session_state
            )
        else:
            # Old flow
            agent = agent_manager.agents.get(agent_type)
            result = agent.process_query(
                user_query=user_input,
                routing_info=routing_info,
                session_state=st.session_state,
                silent_mode=False
            )
    except Exception as e:
        st.error(f"Error: {e}")
    
    st.session_state.user_input = ""


# For adding to sidebar - allows users to toggle presentation mode
def add_presentation_mode_toggle():
    """Add to your display_enhanced_contextual_sidebar() function"""
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Presentation Mode")
    
    use_unified = st.sidebar.checkbox(
        "Unified Chat Presentation",
        value=st.session_state.get("use_unified_presentation", True),
        help="When enabled, all tool outputs are formatted as conversational responses. "
             "When disabled, tools display their results directly (classic mode)."
    )
    st.session_state.use_unified_presentation = use_unified
    
    if use_unified:
        st.sidebar.success("✅ All responses unified through general_chat")
    else:
        st.sidebar.info("📊 Tools display results directly")

def display_enhanced_contextual_sidebar():
    """Enhanced sidebar with contextual engineering analytics"""
    
    with st.sidebar:
        st.title("🧠 AI Assistant Pro")
        # Quick access to document uploader
        if st.button("📄 Upload Documents"):
            st.session_state.show_rag_uploader = True
            st.rerun()
        
        router_type = st.session_state.get("router_type", "standard")
        if router_type == "contextual":
            st.caption("*Powered by Contextual Engineering*")
        else:
            st.caption(f"*{router_type.title()} Routing Mode*")
        
        # Contextual Intelligence Section
        st.subheader("🎯 Contextual Intelligence")
        
        router = st.session_state.router
        
        try:
            # Current user profile quick view (if contextual routing available)
            if router_type == "contextual" and hasattr(router, 'context_manager'):
                session_id = st.session_state.session_id
                user_profile = router.context_manager.get_or_create_user_profile(session_id, st.session_state)
                
                # Profile summary
                st.markdown("**Your Profile:**")
                role_icons = {"warehouse_user": "👷", "manager": "👔", "analyst": "📊", "admin": "⚙️"}
                exp_icons = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
                
                st.write(f"{role_icons.get(user_profile.role, '👤')} {user_profile.role}")
                st.write(f"{exp_icons.get(user_profile.experience_level, '⚪')} {user_profile.experience_level}")
                
                # Contextual analytics
                if hasattr(router, 'get_contextual_analytics'):
                    analytics = router.get_contextual_analytics()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Active Users", analytics.get("total_users", 0))
                    with col2:
                        st.metric("Conversations", analytics.get("total_conversations", 0))
                
                # Current context status
                context_status = router.context_manager.get_context_summary(session_id)
                if context_status.get("status") != "no_context":
                    st.markdown("**Current Context:**")
                    st.write(f"🎯 Topic: {context_status.get('current_topic', 'general')}")
                    st.write(f"📍 Stage: {context_status.get('conversation_stage', 'unknown')}")
                    
                    urgency = context_status.get('urgency_level', 'normal')
                    urgency_icon = {"normal": "🟢", "high": "🟡", "urgent": "🔴"}.get(urgency, "⚪")
                    st.write(f"{urgency_icon} Urgency: {urgency}")
            else:
                # Standard analytics
                analytics = router.get_routing_analytics()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Routes", analytics.get("total_routes", 0))
                with col2:
                    st.metric("Avg Confidence", f"{analytics.get('average_confidence', 0):.2f}")
                
        except Exception as e:
            st.error(f"Context error: {e}")
        
        # Contextual Controls
        st.subheader("⚙️ Contextual Controls")
        
        # Profile adjustment (if contextual routing available)
        if router_type == "contextual" and hasattr(router, 'context_manager'):
            if st.button("👤 View/Edit Profile"):
                st.session_state.show_profile_editor = not st.session_state.get("show_profile_editor", False)
            
            if st.session_state.get("show_profile_editor", False):
                st.markdown("**Adjust Your Profile:**")
                
                try:
                    session_id = st.session_state.session_id
                    user_profile = router.context_manager.get_or_create_user_profile(session_id, st.session_state)
                    
                    current_role = user_profile.role
                    new_role = st.selectbox(
                        "Role", 
                        ["warehouse_user", "manager", "analyst", "admin"],
                        index=["warehouse_user", "manager", "analyst", "admin"].index(current_role),
                        key="profile_role"
                    )
                    
                    current_exp = user_profile.experience_level
                    new_exp = st.selectbox(
                        "Experience", 
                        ["beginner", "intermediate", "advanced"],
                        index=["beginner", "intermediate", "advanced"].index(current_exp),
                        key="profile_exp"
                    )
                    
                    current_detail = user_profile.preferred_detail_level
                    new_detail = st.selectbox(
                        "Detail Level", 
                        ["brief", "standard", "detailed"],
                        index=["brief", "standard", "detailed"].index(current_detail),
                        key="profile_detail"
                    )
                    
                    if st.button("💾 Save Profile"):
                        user_profile.role = new_role
                        user_profile.experience_level = new_exp
                        user_profile.preferred_detail_level = new_detail
                        st.success("Profile updated!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Profile editor error: {e}")
        
        # Context debugging
        st.subheader("🔍 Context Debugging")
        
        debug_query = st.text_input("Test contextual routing:", placeholder="show inventory levels")
        if st.button("🧪 Test Contextual Route") and debug_query:
            try:
                if router_type == "contextual" and hasattr(router, 'route_query_contextual'):
                    agent, routing_info = router.route_query_contextual(debug_query, st.session_state)
                else:
                    agent, routing_info = router.route_query(debug_query, st.session_state)
                
                st.write(f"**Route**: {agent}")
                st.write(f"**Confidence**: {routing_info.get('confidence', 0):.3f}")
                st.write(f"**Reasoning**: {routing_info.get('llm_reasoning', 'N/A')}")
                
                # Show contextual factors if available
                if "user_profile" in routing_info:
                    user_prof = routing_info.get("user_profile", {})
                    conv_ctx = routing_info.get("conversation_context", {})
                    
                    st.write("**Context Factors**:")
                    st.write(f"- Role: {user_prof.get('role')}")
                    st.write(f"- Experience: {user_prof.get('experience_level')}")
                    st.write(f"- Topic: {conv_ctx.get('topic')}")
                    st.write(f"- Urgency: {conv_ctx.get('urgency')}")
                
            except Exception as e:
                st.error(f"Test failed: {e}")

        st.subheader("📄 Document RAG")
        display_document_management_panel()        
        # Performance insights
        st.subheader("📊 Performance Insights")
        
        try:
            # Agent usage for this user (if contextual routing available)
            if router_type == "contextual" and hasattr(router, 'context_manager'):
                session_id = st.session_state.session_id
                user_profile = router.context_manager.get_or_create_user_profile(session_id, st.session_state)
                agent_usage = user_profile.frequently_used_agents
                if agent_usage:
                    st.write("**Your Agent Usage:**")
                    total_usage = sum(agent_usage.values())
                    for agent, count in agent_usage.items():
                        percentage = (count / total_usage) * 100
                        st.write(f"  • {agent}: {count} ({percentage:.1f}%)")
                
                # Recent conversation trends
                if len(router.context_manager.conversation_contexts) > 0:
                    urgency_counts = {}
                    for ctx in router.context_manager.conversation_contexts.values():
                        urgency = ctx.urgency_level
                        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
                    
                    if urgency_counts:
                        st.write("**Recent Urgency Trends:**")
                        for urgency, count in urgency_counts.items():
                            urgency_icon = {"normal": "🟢", "high": "🟡", "urgent": "🔴"}.get(urgency, "⚪")
                            st.write(f"  {urgency_icon} {urgency}: {count}")
            else:
                # Standard analytics
                analytics = router.get_routing_analytics()
                agent_dist = analytics.get("agent_distribution", {})
                if agent_dist:
                    st.write("**Agent Distribution:**")
                    for agent, count in agent_dist.items():
                        st.write(f"  • {agent}: {count}")
                        
        except Exception as e:
            st.error(f"Performance insights error: {e}")

def display_contextual_examples():
    """Display examples of contextual engineering in action"""
    
    with st.expander("🧠 **Contextual Engineering Examples**", expanded=False):
        st.markdown("""
        **See how contextual intelligence adapts to different users and situations:**
        
        ### 👷 **Warehouse User (Beginner)**
        - **Query**: "show inventory"
        - **Context**: New employee, morning shift, first week
        - **Adaptation**: Simple language, detailed explanations, step-by-step guidance
        - **Response Style**: Patient, educational, includes definitions
        
        ### 👔 **Manager (Advanced)**  
        - **Query**: "show inventory"
        - **Context**: Management role, end of quarter, high urgency
        - **Adaptation**: Business insights, trend analysis, executive summary
        - **Response Style**: Concise, strategic, action-oriented
        
        ### 📊 **Analyst (Expert)**
        - **Query**: "show inventory" 
        - **Context**: Data analysis role, complex project, technical focus
        - **Adaptation**: Advanced SQL, performance optimization, statistical context
        - **Response Style**: Technical depth, optimization suggestions, data quality notes
        
        ### 🚨 **Urgent Context**
        - **Query**: "urgent: show critical inventory levels"
        - **Context**: Emergency situation, immediate action needed
        - **Adaptation**: Priority processing, immediate results, escalation suggestions
        - **Response Style**: Fast response, clear priorities, emergency protocols
        
        ### 🔄 **Follow-up Context**
        - **Previous**: User just viewed sales data
        - **Query**: "add task about those numbers"
        - **Context**: Cross-agent intent, task creation with data context
        - **Adaptation**: Auto-populate task with previous results, smart defaults
        - **Response Style**: Contextual understanding, reduced input required
        """)

def main():
    """ENHANCED main application with contextual engineering"""
    
    # Page config
    st.set_page_config(
        page_title="🧠 Contextual AI Assistant",
        page_icon="🧠",
        layout="wide"
    )
    initialize_comprehensive_llm_debugger(
        auto_patch=True,           # Automatically patch all functions
        show_full_prompts=True,    # Show complete LLM prompts
        show_responses=True        # Show complete LLM responses
    )    
    # Initialize session state (unified function)
    initialize_session_state()
    initialize_data()

    # Init persistent RAG uploader flag
    if "show_rag_uploader" not in st.session_state:
        st.session_state.show_rag_uploader = False

    # Enhanced sidebar with contextual intelligence
    display_enhanced_contextual_sidebar()

    # Main interface
    st.title("🧠 Contextual AI Assistant")
    
    router_type = st.session_state.get("router_type", "standard")
    if router_type == "contextual":
        st.markdown("*Intelligent responses adapted to your role, experience, and context*")
    else:
        st.markdown(f"*AI Assistant with {router_type} routing and conversation management*")
    
    # Contextual examples
    display_contextual_examples()
    st.markdown("---")
    st.subheader("🔧 Debug Tools")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧪 Test Direct Upload"):
            st.write("🔍 Testing DocumentRAG agent directly...")
            
            # Get agent manager
            agent_manager = st.session_state.get('agent_manager')
            if not agent_manager:
                st.error("❌ Agent manager not found!")
            else:
                st.success("✅ Agent manager found")
                
                # Get document RAG agent
                doc_agent = agent_manager.agents.get('document_rag')
                if not doc_agent:
                    st.error("❌ DocumentRAG agent not found!")
                    st.write(f"Available agents: {list(agent_manager.agents.keys())}")
                else:
                    st.success("✅ DocumentRAG agent found")
                    
                    # Test upload detection
                    test_query = "upload document"
                    is_upload = doc_agent._is_upload_request(test_query)
                    st.write(f"Upload detection test: '{test_query}' → {is_upload}")
                    
                    # Try calling the agent directly
                    try:
                        st.write("🔍 Calling agent directly...")
                        result = doc_agent.process_query(
                            user_query="upload document",
                            routing_info={},
                            session_state=st.session_state
                        )
                        st.write(f"Direct agent call result: {result}")
                    except Exception as e:
                        st.error(f"Direct agent call failed: {e}")

    with col2:
        if st.button("🔍 Test Routing"):
            st.write("🔍 Testing router...")
            
            router = st.session_state.get('router')
            if not router:
                st.error("❌ Router not found!")
            else:
                st.success("✅ Router found")
                
                # Test routing
                test_query = "upload document"
                try:
                    if hasattr(router, 'route_query_contextual'):
                        agent_type, routing_info = router.route_query_contextual(test_query, st.session_state)
                    else:
                        agent_type, routing_info = router.route_query(test_query, st.session_state)
                    
                    st.write(f"Routing test: '{test_query}' → {agent_type}")
                    st.write(f"Confidence: {routing_info.get('confidence', 'N/A')}")
                    st.write(f"Method: {routing_info.get('method', 'N/A')}")
                    
                except Exception as e:
                    st.error(f"Routing test failed: {e}")

    with col3:
        if st.button("📊 System Status"):
            st.write("🔍 System status check...")
            
            # Check all components
            checks = {
                "Session ID": st.session_state.get('session_id') is not None,
                "Query Store": st.session_state.get('query_store') is not None,
                "Router": st.session_state.get('router') is not None,
                "Agent Manager": st.session_state.get('agent_manager') is not None,
                "RAG Initialized": st.session_state.get('rag_initialized', False),
            }
            
            for check_name, status in checks.items():
                if status:
                    st.success(f"✅ {check_name}")
                else:
                    st.error(f"❌ {check_name}")
            
            # Check agent manager details
            agent_manager = st.session_state.get('agent_manager')
            if agent_manager:
                st.write("Available agents:")
                for agent_name, agent_obj in agent_manager.agents.items():
                    st.write(f"  • {agent_name}: {type(agent_obj).__name__}")

    st.markdown("---")
    # Input interface with contextual hints
    input_container = st.container()
    conversation_container = st.container()

    with input_container:
        st.subheader("💬 Contextual Chat Interface")
        
        # Show contextual hints based on user profile
        try:
            session_id = st.session_state.session_id
            router = st.session_state.router
            router_type = st.session_state.get("router_type", "standard")
            
            if router_type == "contextual" and hasattr(router, 'context_manager'):
                user_profile = router.context_manager.get_or_create_user_profile(session_id, st.session_state)
                
                # Contextual suggestions
                role_suggestions = {
                    "warehouse_user": "Try: 'show today's shipments' or 'add task to check inventory'",
                    "manager": "Try: 'analyze monthly performance' or 'show urgent tasks'", 
                    "analyst": "Try: 'generate sales trend report' or 'optimize warehouse queries'"
                }
                
                suggestion = role_suggestions.get(user_profile.role, "Try any warehouse or task-related query")
                st.caption(f"💡 **Contextual suggestion for {user_profile.role}**: {suggestion}")
            else:
                st.caption("💡 **Try**: 'show sales data' or 'add task to review reports' or 'hello!'")
            
        except Exception:
            st.caption("💡 **Try**: 'show sales data' or 'add task to review reports' or 'hello!'")
        
        # Enhanced input with contextual processing
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.text_input(
                "Your contextually-aware assistant is ready:",
                key="user_input",
                on_change=contextual_send_message,  # Use contextual version
                placeholder="Ask anything - I'll adapt to your role and context..."
            )
        
        with col2:
            if st.button("🧠 Send", use_container_width=True):
                contextual_send_message()

    # Display conversation history with contextual enhancements
    with conversation_container:
        display_conversation_history(st.session_state.query_store, st.session_state.session_id)

if __name__ == "__main__":
    st.set_page_config(
        page_title="🧠 Contextual AI Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"  # ADD THIS LINE
    )
    
    # ADD THIS CSS BLOCK
    st.markdown("""
    <style>
    .stExpander > div {
        max-height: 400px;
        overflow-y: auto;
    }
    div[data-testid="metric-container"] {
        margin: 2px;
        padding: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    main()
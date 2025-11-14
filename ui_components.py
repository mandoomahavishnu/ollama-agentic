"""
UI components and helper functions for Streamlit interface
"""
import pandas as pd
import streamlit as st
from database import handle_duplicate_columns
from sql_operations import (
    generate_nl_summary_of_results, 
    handle_no_results_response,
    store_key_fields_in_memory
)

def display_query_results(rows, cols, user_input, col_comment_map):
    """Display query results in the UI"""
    if rows:
        df = pd.DataFrame(rows, columns=cols)
        df = handle_duplicate_columns(df)
        st.subheader("Query Results")
        st.dataframe(df)

        # Store key fields (e.g. po_num)
        store_key_fields_in_memory(rows, cols, st.session_state)

        summary_text = generate_nl_summary_of_results(user_input, rows, cols, col_comment_map)
        return True, summary_text
    else:
        no_results_message = handle_no_results_response(user_input)
        return False, no_results_message

def display_feedback_buttons(user_input, validated_sql, rows, query_store, session_id):
    """Display confirm/reject buttons for query results"""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirm Results"):
            query_store.add_query(
                session_id=session_id,
                user_query=user_input,
                sql_query=validated_sql,
                results=rows,
                response="Confirmed",
                sql_error=None,
                user_feedback="Confirmed"
            )
            st.success("Results confirmed! Future queries can use this as a good example.")
    with col2:
        if st.button("Reject Results"):
            query_store.add_query(
                session_id=session_id,
                user_query=user_input,
                sql_query=validated_sql,
                results=rows,
                response="Rejected",
                sql_error=None,
                user_feedback="Rejected"
            )
            st.warning("Results rejected. Not used for future suggestions.")

def display_conversation_history(query_store, session_id):
    """Display conversation history with agent information"""
    st.subheader("Conversation History (All Queries)")
    all_queries = query_store.fetch_session_queries(session_id)
    if not all_queries:
        st.write("No conversation yet. Ask something below!")
    else:
        for i, (user_q, sql_q, results_q, response_status, created_at, agent_used) in enumerate(all_queries):  # ADD agent_used
            # Determine agent icon
            agent_icon = "💬"  # default
            if agent_used == "nl2sql":
                agent_icon = "📊"
            elif agent_used == "personal_assistant":
                agent_icon = "🤖"
            elif agent_used == "general_chat":
                agent_icon = "💬"
            elif sql_q and ("SELECT" in str(sql_q).upper() or "FROM" in str(sql_q).upper()):
                agent_icon = "📊"  # fallback for old entries without agent_used
            
            st.markdown(f"**{agent_icon} Message {i+1}** | *{created_at}* | **Status**: {response_status}")
            st.write(f"**User Query:** {user_q}")
            
            # Only show SQL if it exists
            if sql_q:
                st.write(f"**Generated SQL:** `{sql_q}`")

            if results_q is not None:
                if isinstance(results_q, list) and results_q:
                    df = pd.DataFrame(results_q)
                    st.dataframe(df)
                else:
                    st.write("*(No rows or empty result set)*")
            
            st.write("---")

def setup_sidebar_controls(conv_manager, query_store, session_id):
    """Setup sidebar controls"""
    st.sidebar.title("Session Controls")
    if st.sidebar.button("Refresh Memory"):
        conv_manager.history.clear()
        query_store.clear_session_conversation(session_id)
        st.session_state.user_input = ""
        # Clear short-term memory for POs
        if "last_po_nums" in st.session_state:
            st.session_state.last_po_nums.clear()
        if "last_so_nums" in st.session_state:
            st.session_state.last_so_nums.clear()

def display_error_handling(err, validated_sql, user_input, query_store, session_id):
    """Handle and display SQL errors with auto-fix attempt"""
    from sql_operations import attempt_sql_error_fix
    from database import execute_sql_and_return_rows
    
    st.error(f"SQL Execution Error: {err}")
    
    # Attempt an auto-fix
    with st.spinner("Attempting to fix the SQL error..."):
        fixed_sql = attempt_sql_error_fix(
            original_sql=validated_sql,
            error_message=err,
            user_prompt=user_input,
            schema_info="(Optional) Additional schema or context can go here"
        )

    st.warning("Proposed Fix for the SQL:")
    st.code(fixed_sql, language='sql')

    if fixed_sql and fixed_sql.lower() != validated_sql.lower():
        st.info("Re-running with the fixed SQL...")
        cols2, rows2, err2 = execute_sql_and_return_rows(fixed_sql)
        if err2:
            st.error(f"Fixed SQL also failed: {err2}")
            query_store.add_query(
                session_id=session_id,
                user_query=user_input,
                sql_query=fixed_sql,
                results=None,
                response="Error",
                sql_error=err2
            )
        else:
            resp_type = "Success" if rows2 else "No Data"
            query_store.add_query(
                session_id=session_id,
                user_query=user_input,
                sql_query=fixed_sql,
                results=rows2,
                response=resp_type,
                sql_error=None
            )
            if rows2:
                success, message = display_query_results(
                    rows2, cols2, user_input, st.session_state.col_comment_map
                )
                if success:
                    st.success(message)
    else:
        st.warning("No improved SQL or fix identical. Not retrying.")

    # Store original attempt as well
    query_store.add_query(
        session_id=session_id,
        user_query=user_input,
        sql_query=validated_sql,
        results=None,
        response="Error",
        sql_error=err
    )
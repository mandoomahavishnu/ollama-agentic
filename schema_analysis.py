"""
Schema analysis for table and column identification
"""
import re
import streamlit as st
from nlp_utils import detect_follow_up_query
from config import TABLE_NAMES, KEYWORD_TABLE_MAP

def default_table_selection(
    user_query,
    query_store,
    table_names=TABLE_NAMES,
    keyword_table_map=KEYWORD_TABLE_MAP,
    max_tables=3,
    max_columns=5
):
    """
    ENHANCED table & column selection with semantic table search
    """
    st.write(f"🔎 Processing User Query: {user_query}")
    lowered_query = user_query.lower()

    # 1. KEYWORD SCORING (existing logic - highest weight)
    keyword_scores = {}
    for keyword, tables in keyword_table_map.items():
        if keyword in lowered_query:
            for table in tables:
                keyword_scores[table] = keyword_scores.get(table, 0) + 3

    # 2. FEW-SHOT SCORING (existing logic)
    few_shot_scores = {}
    few_shot_examples = query_store.search_few_shot_examples(user_query, k=5)
    for example in few_shot_examples:
        sql_query = example[1].lower()
        for t in table_names:
            if t in sql_query:
                few_shot_scores[t] = few_shot_scores.get(t, 0) + 3

    # 3. CONVERSATION SCORING (existing logic)
    conv_scores = {}
    past_queries = query_store.search_query(
        session_id=st.session_state.session_id, query=user_query, k=5
    )
    for pq in past_queries:
        past_sql = pq[1].lower()
        for t in table_names:
            if t in past_sql:
                conv_scores[t] = conv_scores.get(t, 0) + 1

    # 4. SEMANTIC COLUMN SCORING (existing logic)
    semantic_column_scores = {}
    schema_search_results = query_store.search_relevant_schema(user_query, k=10)
    for table_name, _, _ in schema_search_results:
        semantic_column_scores[table_name] = semantic_column_scores.get(table_name, 0) + 1

    # 5. NEW: SEMANTIC TABLE SCORING using your Excel descriptions
    semantic_table_scores = {}
    try:
        # Use the enhanced semantic table search
        table_semantic_results = query_store.search_tables_semantically_enhanced(
            user_query, k=len(table_names), include_scores=True
        )
        
        for table_name, description, similarity_score in table_semantic_results:
            # Convert similarity to points (0.0-1.0 -> 0-3 points)
            points = similarity_score * 3
            semantic_table_scores[table_name] = points
            
        st.write(f"🧠 Semantic table matches: {[(t, round(s, 2)) for t, s in list(semantic_table_scores.items())[:3]]}")
        
    except Exception as e:
        st.warning(f"Semantic table search unavailable: {e}")
        semantic_table_scores = {}

    # 6. COMBINED SCORING with enhanced weights
    table_scores = {}
    for t in table_names:
        table_scores[t] = (
            keyword_scores.get(t, 0) * 4        # Increased weight for keywords
            + few_shot_scores.get(t, 0) * 3     # Increased weight for few-shot
            + conv_scores.get(t, 0) * 2         # Conversation history
            + semantic_column_scores.get(t, 0) * 1   # Column-level semantic
            + semantic_table_scores.get(t, 0) * 2    # NEW: Table-level semantic
        )

    # Debug output
    top_tables_debug = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    st.write(f"📊 Table scores: {[(t, round(s, 1)) for t, s in top_tables_debug]}")

    selected_tables = sorted(table_scores, key=table_scores.get, reverse=True)[:max_tables]
    
    # ENHANCED COLUMN SELECTION
    selected_columns = {}
    for table in selected_tables:
        # First try semantic column search for this specific table
        try:
            table_specific_columns = query_store.search_columns_semantically(user_query, table, k=max_columns)
            
            if table_specific_columns:
                selected_columns[table] = [col[0] for col in table_specific_columns]
            else:
                raise Exception("No semantic columns found")
                
        except:
            # Fallback to general schema search
            raw_schema_matches = query_store.search_relevant_schema(user_query, k=30)
            table_only = [r for r in raw_schema_matches if r[0] == table]
            top_cols_for_table = [r[1] for r in table_only][:max_columns]
            
            if not top_cols_for_table:
                # Final fallback to all columns
                fallback_cols = [
                    row[1] for row in st.session_state.full_schema if row[0] == table
                ][:max_columns]
                selected_columns[table] = fallback_cols
            else:
                selected_columns[table] = top_cols_for_table

    return selected_tables, selected_columns

def identify_relevant_tables_and_columns(
    user_query,
    query_store,
    session_id,
    max_tables=3,
    max_columns=5
):
    """
    ENHANCED logic with better follow-up detection and semantic search
    """
    is_follow_up, confidence = detect_follow_up_query(user_query, session_id, query_store)
    last_sql = query_store.fetch_last_query(session_id)

    if is_follow_up and last_sql:
        if 0 < confidence < 0.8:
            st.warning(
                "It looks like you're referring to the last query, but I'm not entirely sure. "
                "I'll attempt to treat it as a follow-up."
            )
        
        # Extract tables from last SQL more robustly
        last_tables = []
        
        # Look for FROM clauses
        from_matches = re.findall(r"FROM\s+(\w+)", last_sql, re.IGNORECASE)
        last_tables.extend(from_matches)
        
        # Look for JOIN clauses  
        join_matches = re.findall(r"JOIN\s+(\w+)", last_sql, re.IGNORECASE)
        last_tables.extend(join_matches)
        
        # Remove duplicates and limit
        last_tables = list(set(last_tables))[:max_tables]
        
        # Get columns for these tables
        last_columns = {}
        for t in last_tables:
            # Try to get columns from the last SQL first
            sql_columns = re.findall(rf"\b{t}\.(\w+)", last_sql, re.IGNORECASE)
            if sql_columns:
                last_columns[t] = list(set(sql_columns))[:max_columns]
            else:
                # Fallback to all columns
                fallback_cols = [
                    row[1] for row in st.session_state.full_schema if row[0] == t
                ][:max_columns]
                last_columns[t] = fallback_cols
                
        st.info(f"🔄 Follow-up detected (confidence: {confidence:.2f}). Reusing tables: {last_tables}")
        return last_tables, last_columns

    # Use enhanced table selection for new queries
    return default_table_selection(
        user_query, query_store, TABLE_NAMES, KEYWORD_TABLE_MAP, max_tables, max_columns
    )

# Additional helper function for debugging
def debug_table_selection(user_query, query_store, session_id):
    """Debug function to see all scoring components"""
    
    st.subheader("🔍 Table Selection Debug")
    
    # Run all scoring mechanisms
    lowered_query = user_query.lower()
    
    # Keyword scoring
    keyword_matches = []
    for keyword, tables in KEYWORD_TABLE_MAP.items():
        if keyword in lowered_query:
            keyword_matches.append((keyword, tables))
    
    if keyword_matches:
        st.write("**Keyword Matches:**")
        for keyword, tables in keyword_matches:
            st.write(f"  • '{keyword}' → {tables}")
    
    # Few-shot examples
    few_shot_examples = query_store.search_few_shot_examples(user_query, k=3)
    if few_shot_examples:
        st.write("**Similar Examples:**")
        for i, (nl, sql) in enumerate(few_shot_examples, 1):
            st.write(f"  {i}. {nl[:50]}...")
    
    # Semantic table search
    try:
        semantic_results = query_store.search_tables_semantically_enhanced(
            user_query, k=5, include_scores=True
        )
        if semantic_results:
            st.write("**Semantic Table Matches:**")
            for table, desc, score in semantic_results:
                st.write(f"  • {table}: {score:.3f}")
    except:
        st.write("**Semantic Table Search:** Not available")
    
    # Run the actual selection
    selected_tables, selected_columns = identify_relevant_tables_and_columns(
        user_query, query_store, session_id
    )
    
    st.write("**Final Selection:**")
    st.write(f"  Tables: {selected_tables}")
    st.write(f"  Columns: {selected_columns}")
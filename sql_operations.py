"""
SQL operations including generation, validation, and execution
"""
import re
import time
import datetime
import ollama
import sqlparse
import streamlit as st
from config import LLM_MODEL

def attempt_sql_error_fix(original_sql, error_message, user_prompt, schema_info=""):
    """
    Calls the LLM to fix or regenerate SQL based on the error message and user's request.
    Returns the 'fixed' SQL if the model can propose one.
    """
    prompt_for_fix = f'''
The following SQL query caused an error in MySQL:

[Original SQL]
{original_sql}

[Error Message]
{error_message}

[User Prompt]
{user_prompt}

SCHEMA (use ONLY these tables and columns):
{schema_info}

STRICT RULES:
1. Please provide a corrected SQL query that resolves the error while answering the user's prompt.
2. Use only tables/columns from the schema.
3. Only output the SQL, do not wrap in code fences.
'''

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a MySQL SQL-fixer assistant."},
            {"role": "user", "content": prompt_for_fix}
        ],
        stream=False,
        options={"temperature": 0.1, "num_ctx": 16384}
    )

    corrected_sql = response["message"]["content"].strip()
    corrected_sql = re.sub(r'```[a-zA-Z]*', '', corrected_sql).strip()
    return corrected_sql

def refine_sql_with_context_structured(user_query, prev_sql):
    """
    Uses sqlparse to partially modify the existing SQL (e.g., adding conditions to WHERE)
    while preserving existing logic. Returns (refined_sql, explanation).
    """
    parsed = sqlparse.parse(prev_sql)
    if not parsed:
        return prev_sql, "No valid SQL to refine."

    statement = parsed[0]
    tokens = [t for t in statement.tokens if not t.is_whitespace]
    where_found = False
    explanation = ""

    for idx, token in enumerate(tokens):
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "WHERE":
            where_found = True
            tokens.insert(idx + 1, sqlparse.sql.Token(sqlparse.tokens.Keyword, f" AND ({user_query})"))
            explanation = "Appended new condition to the WHERE clause."
            break

    if not where_found:
        tokens.append(sqlparse.sql.Token(sqlparse.tokens.Keyword, f" WHERE ({user_query})"))
        explanation = "Added new WHERE clause based on user input."

    refined_sql = " ".join(t.value for t in tokens)
    refined_sql = re.sub(r'\s+', ' ', refined_sql).strip()
    return refined_sql, explanation

def validate_sql(sql_query, schema):
    """Validate SQL query (basic implementation)"""
    parsed = sqlparse.parse(sql_query)
    if not parsed:
        return sql_query
    return sql_query

def sanitize_sql(sql_query):
    """Clean and sanitize SQL query"""
    sql_query = sql_query.strip()
    sql_query = re.sub(r'```[a-zA-Z]*', '', sql_query).strip()
    sql_query = sql_query.replace("**", "").strip()
    sql_query = re.sub(r'\s+', ' ', sql_query)
    return sql_query

def maybe_add_po_filter(sql_query, last_po_nums):
    """
    If user references the previous PO's, we add a filter 'WHERE po_num IN (...)' if not present.
    Naive approach. A real system might parse the existing SQL more deeply.
    """
    if not last_po_nums:
        return sql_query  # no stored POs

    # If there's no existing WHERE, we add one; otherwise, append an AND condition
    join_str = " WHERE "
    if re.search(r"\bWHERE\b", sql_query, re.IGNORECASE):
        join_str = " AND "

    placeholders = ",".join(f"'{po}'" for po in last_po_nums)
    filter_condition = f"po_num IN ({placeholders})"
    enhanced_sql = sql_query + f" {join_str} {filter_condition}"
    return enhanced_sql

def store_key_fields_in_memory(rows, columns, session_state):
    """
    Parse out 'po_num', 'so_num', etc. from the query results and store them in session state.
    We'll only store up to 50 for brevity.
    """
    if 'last_po_nums' not in session_state:
        session_state.last_po_nums = set()
    if 'last_so_nums' not in session_state:
        session_state.last_so_nums = set()

    for row in rows[:50]:
        if 'po_num' in columns and row.get('po_num'):
            session_state.last_po_nums.add(row['po_num'])
        if 'so_num' in columns and row.get('so_num'):
            session_state.last_so_nums.add(row['so_num'])

def build_relevant_schema_summary_enhanced(tables, full_schema, selected_columns, query_store):
    """Enhanced version that includes table descriptions from Excel"""
    schema_info = []
    
    # First, add table-level descriptions
    for tbl in tables:
        try:
            # Get the table description from embeddings (from Excel)
            table_results = query_store.search_tables_semantically_enhanced(
                tbl, k=1, include_scores=False
            )
            if table_results:
                # Get the actual description from the database
                with query_store.conn.cursor() as cur:
                    cur.execute(
                        "SELECT description FROM table_embeddings WHERE table_name = %s",
                        (tbl,)
                    )
                    result = cur.fetchone()
                    if result:
                        table_desc = result[0]
                        schema_info.append(f"TABLE {tbl.upper()}: {table_desc}")
        except Exception as e:
            print(f"Could not get description for {tbl}: {e}")
            schema_info.append(f"TABLE {tbl.upper()}: Standard database table")
    
    # Then add column-level info
    for tbl in tables:
        table_cols = [
            (row[1], row[7] if row[7] else "No description")
            for row in full_schema
            if row[0] == tbl and (tbl in selected_columns and row[1] in selected_columns[tbl])
        ]
        if not table_cols:
            table_cols = [
                (row[1], row[7] if row[7] else "No description")
                for row in full_schema
                if row[0] == tbl
            ]
        
        for col_name, comment in table_cols:
            schema_info.append(f"  - {tbl}.{col_name}: {comment}")
    
    return schema_info

def build_table_examples_summary(tables, table_examples, max_rows=1):
    """Build a summary of table examples"""
    summary_map = {}
    for tbl in tables:
        if tbl in table_examples:
            summary_map[tbl] = table_examples[tbl][:max_rows]
        else:
            summary_map[tbl] = []
    return summary_map

def limit_few_shot_examples(user_query, query_store, session_id, k=3):
    """Get limited few-shot examples for the query"""
    db_few_shots = query_store.search_few_shot_examples(user_query, k)
    sess_shots = query_store.search_query(session_id, user_query, k)
    combined = []
    for r in db_few_shots:
        combined.append({"natural_language": r[0], "sql_query": r[1]})
    for r in sess_shots:
        combined.append({"natural_language": r[0], "sql_query": r[1]})
    return combined[:k]

def generate_prompt_with_cot(
    user_query,
    conversation_summary,
    few_shot_examples,
    relevant_schema,
    table_example_rows,
    previous_sql
):
    """Generate chain-of-thought prompt for SQL generation"""
    if isinstance(relevant_schema, list) and relevant_schema and isinstance(relevant_schema[0], str):
        schema_info = "\n".join(relevant_schema)
    else:
        # Fallback for old format
        schema_lines = []
        for (tbl, col, comm) in relevant_schema:
            schema_lines.append(f"Table: {tbl}, Column: {col}, Comment: {comm}")
        schema_info = "\n".join(schema_lines)

    ex_lines = []
    for i, ex in enumerate(few_shot_examples, start=1):
        ex_lines.append(f"Example {i}:\nNL: {ex['natural_language']}\nSQL: {ex['sql_query']}")
    examples_text = "\n".join(ex_lines)

    table_rows_text = []
    for tbl, rowlist in table_example_rows.items():
        table_rows_text.append(f"Table: {tbl}, Example Rows: {rowlist}")
    table_rows_text = "\n".join(table_rows_text)

    current_time = datetime.datetime.now()
    guidelines = f'''
Guidelines:
1. You are an NL→SQL assistant specialized in warehouse/WMS databases.
2. You MUST use ONLY the tables and columns provided in the SCHEMA below.
3. If a requested table or column does NOT appear in the SCHEMA, do NOT guess or invent.
   Instead, output exactly: /* NOT IN SCHEMA */
4. Do NOT hallucinate new tables, columns, or joins.
5. Use the provided conversation history to be contextually aware.
6. If the query builds on past discussion, modify the last SQL query accordingly.
7. If it's a new request, generate a fresh SQL query.
8. Maintain consistency in tables and column names based on the schema provided.
9. Use time-awareness (current time: {current_time}).
10. Pay close attention to TABLE DESCRIPTIONS - they explain what each table contains and how they relate.

🔥 CRITICAL GROUP BY RULES (READ CAREFULLY):

11. When using aggregate functions (COUNT, SUM, AVG, MAX, MIN), you MUST include GROUP BY 
    for ALL non-aggregated columns in the SELECT clause.
    ✅ CORRECT: SELECT customer, SUM(amount) FROM orders GROUP BY customer
    ❌ WRONG:   SELECT customer, SUM(amount) FROM orders

12. When joining tables, ALWAYS consider if you need GROUP BY to eliminate duplicates.
    Common pattern: JOIN creates duplicate rows → Use GROUP BY on key columns
    Example: "show products with their sales" often needs GROUP BY sap_code, item_num

13. Key warehouse patterns requiring GROUP BY:
    - "each product" → GROUP BY sap_code, item_num
    - "per vendor" → GROUP BY vendor
    - "by month" → GROUP BY YEAR(date), MONTH(date)
    - "total for each X" → GROUP BY X
    - Any query with COUNT/SUM + non-aggregated columns → GROUP BY those columns

14. Examples when you MUST use GROUP BY:
    - "show total units for each product" → GROUP BY item_code, item_num
    - "count orders per customer" → GROUP BY customer
    - "list products with multiple locations" → GROUP BY item_code HAVING COUNT(DISTINCT bin) > 1
    - ANY join with picktix/salesorder/warehouse → Usually needs GROUP BY to deduplicate

15. DISTINCT vs GROUP BY:
    - Use DISTINCT only for simple single-column uniqueness
    - Use GROUP BY when you have aggregates or multiple columns
    - When in doubt, prefer GROUP BY over DISTINCT

16. Output format:
    Reasoning:
    [Briefly explain your SQL logic. If using GROUP BY, mention why]
    
    SQL:
    [final sql only, no code fences]
'''

    followup_txt = f"Previous SQL:\n{previous_sql}\n" if previous_sql else ""
    prompt = f'''You are a helpful DATA analyst.
Current Time:
{current_time}
Schema:
{schema_info}

Examples:
{examples_text}

Table Example Rows:
{table_rows_text}

Conversation Context:
{conversation_summary}

{followup_txt}
{guidelines}

User Query: {user_query}

Please respond in this format:

Reasoning:
[brief reasoning steps]

SQL:
[final sql only]
'''
    return prompt

def generate_sql_chain_of_thought(user_query, conv_manager, query_store, selected_tables, selected_columns, session_state):
    """Generate SQL using chain-of-thought approach"""
    entities = conv_manager.extract_query_entities(user_query)
    prev_sql = None
    minimal_schema = build_relevant_schema_summary_enhanced(
        tables=selected_tables,
        full_schema=session_state.full_schema,
        selected_columns=selected_columns,
        query_store=query_store
    )
    minimal_table_examples = build_table_examples_summary(
        tables=selected_tables,
        table_examples=session_state.table_examples,
        max_rows=3
    )
    limited_examples = limit_few_shot_examples(
        user_query=user_query,
        query_store=query_store,
        session_id=conv_manager.session_id,
        k=8
    )
    conversation_summary = conv_manager.get_context_summary(6)

    prompt = generate_prompt_with_cot(
        user_query=user_query,
        conversation_summary=f"Entities Identified: {entities}\n\n{conversation_summary}",
        few_shot_examples=limited_examples,
        relevant_schema=minimal_schema,
        table_example_rows=minimal_table_examples,
        previous_sql=prev_sql
    )

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are an SQL assistant that reveals a short chain-of-thought."},
            {"role": "user", "content": prompt}
        ],
        stream=False,
        options={"temperature": 0.1, "num_ctx": 16384}
    )

    full_text = response["message"]["content"]
    reasoning_part = ""
    sql_part = ""
    if "Reasoning:" in full_text and "SQL:" in full_text:
        try:
            _, after_r = full_text.split("Reasoning:", 1)
            reasoning_part, sql_part = after_r.split("SQL:", 1)
            reasoning_part = reasoning_part.strip()
            sql_part = sql_part.strip()
        except:
            sql_part = full_text.strip()
    else:
        sql_part = full_text.strip()

    sql_part = re.sub(r'```[a-z]*', '', sql_part).strip()
    sql_part = re.sub(r'```', '', sql_part).strip()
    sql_part = re.sub(r'\s+', ' ', sql_part).replace("**", "").strip()

    return reasoning_part, sql_part, prompt

def build_column_comment_map(full_schema):
    """
    Returns a dict like {"column_name": "comment", ...}
    for quick lookups in the summary prompt.
    """
    col_map = {}
    for row in full_schema:
        table_name    = row[0]
        column_name   = row[1]
        column_comment= row[7] if len(row) > 7 else ""
        
        # If collisions are possible, store by (table_name, column_name).
        # For simplicity, storing by column name alone:
        col_map[column_name] = column_comment if column_comment else "No comment provided."
    return col_map

def generate_nl_summary_of_results(user_query, rows, columns, col_comment_map):
    """Generate natural language summary of query results with smart truncation handling"""
    if not rows:
        return "No data returned. Nothing to summarize."

    max_rows_to_show = 50
    total_rows = len(rows)
    limited_rows = rows[:max_rows_to_show]
    rows_text = []

    for r in limited_rows:
        row_str = ", ".join(f"{col}={r[col]}" for col in columns)
        rows_text.append(row_str)
    rows_str = "\n".join(rows_text)
    current_time = datetime.datetime.now()

    # Build column comment text
    comments_text_list = []
    for c in columns:
        comment = col_comment_map.get(c, "No comment available.")
        comments_text_list.append(f"• {c}: {comment}")
    column_comments_str = "\n".join(comments_text_list)

    # Determine truncation status and instructions
    if total_rows <= max_rows_to_show:
        truncation_note = f"ALL {total_rows} rows are shown below. Provide detailed information about ALL records."
        response_instruction = "Since all results fit within the limit, provide DETAILED information about each relevant record."
    else:
        truncation_note = f"⚠️ IMPORTANT: Only {max_rows_to_show} of {total_rows} total rows are shown below. There are {total_rows - max_rows_to_show} MORE rows not displayed."
        response_instruction = f"""CRITICAL: You MUST inform the user that:
1. Only {max_rows_to_show} out of {total_rows} results are shown
2. There are {total_rows - max_rows_to_show} additional rows not displayed
3. The user should refine their query if they want to see specific data
4. Provide a summary/overview of the patterns in the shown data, but DO NOT claim this is comprehensive"""

    summary_prompt = f"""
    You are a bright data analyst.
Current Time:
{current_time}
The user asked: "{user_query}"

Actual query results:
Column Names: {columns}
Total Rows Returned by Query: {total_rows}
Rows Shown for Analysis: {len(limited_rows)}

{truncation_note}

Column Comments:
{column_comments_str}

Rows (showing {len(limited_rows)} of {total_rows}):
{rows_str}

{response_instruction}

Use time-awareness and provide a concise English explanation and analysis based on these rows.
"""

    placeholder = st.empty()
    stream_text = ""

    response_generator = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You summarize and analyze SQL query results in plain English."},
            {"role": "user", "content": summary_prompt}
        ],
        stream=True,
        options={"temperature": 0.1,"num_ctx":16384}
    )

    for chunk in response_generator:
        new_text = chunk.get("message", {}).get("content", "")
        stream_text += new_text
        placeholder.text(stream_text)
        time.sleep(0.05)

    final_text = re.sub(r'```[a-z]*', '', stream_text).strip()
    final_text = re.sub(r'```', '', final_text).strip()
    placeholder.text(final_text)
    return final_text

def handle_no_results_response(user_query):
    """Handle case when query returns no results"""
    prompt = f'''
    The SQL query based on the user's request "{user_query}" returned no results.
    Provide a helpful explanation or possible reasons why briefly.
    '''
    placeholder = st.empty()
    stream_text = ""

    response_generator = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Provide explanations when queries return no results."},
            {"role": "user", "content": prompt}
        ],
        stream=True,
        options={"temperature":0.1,"num_ctx":16384}
    )

    for chunk in response_generator:
        new_text = chunk.get("message", {}).get("content", "").strip()
        if new_text and new_text not in stream_text:
            stream_text += new_text + " "
            placeholder.text(stream_text)
            time.sleep(0.05)

    return stream_text.strip()
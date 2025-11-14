"""
NLP utilities for text processing and analysis
"""
import re
import nltk
import spacy
import numpy as np
from nltk.stem import WordNetLemmatizer
from config import FOLLOW_UP_KEYWORDS, COREFERENCE_PHRASES

# Initialize NLP components
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

def detect_verb_tense(user_query: str) -> str:
    """
    Uses spaCy POS tags to identify whether the query is predominantly
    past, present, or present progressive tense.
    
    - Past: VBD or VBN (e.g. "shipped", "was delivered")
    - Present: VBZ or VBP (e.g. "ships", "ship", "deliver")
    - Present Progressive: VBG (e.g. "is shipping", "delivering")
    
    Returns: "past", "present_progressive", "present", or "unknown"
    """
    doc = nlp(user_query.lower())
    
    has_past = False
    has_pres = False
    has_prog = False
    
    for token in doc:
        if token.tag_ in ["VBD", "VBN"]:
            has_past = True
        elif token.tag_ in ["VBZ", "VBP"]:
            has_pres = True
        elif token.tag_ == "VBG":
            has_prog = True
    
    # Decide final outcome
    if has_past:
        return "past"
    elif has_prog:
        return "present_progressive"
    elif has_pres:
        return "present"
    else:
        return "unknown"

def detect_follow_up_query(new_query, session_id, query_store, threshold=0.75):
    """
    Determines whether new_query is a follow-up by:
    1) Checking for lemmatized follow-up keywords
    2) Checking coreference phrases (like 'those PO's')
    3) Checking named entities from spaCy between last query & new query
    4) Checking embedding similarity with the last query
    Returns (bool, confidence_score).
    """
    last_query = query_store.fetch_last_query(session_id)
    if not last_query:
        # No previous queries found
    
        return False, 0.0

    # Step 1: Lemmatized keyword check
    new_tokens = [lemmatizer.lemmatize(tok.lower()) for tok in new_query.split()]
    if any(k in new_tokens for k in FOLLOW_UP_KEYWORDS):
        return True, 1.0

    # 1a: Check coreference phrases
    lowered_query = new_query.lower()
    for phrase in COREFERENCE_PHRASES:
        if phrase in lowered_query:
            return True, 0.95

    # Step 2: Named Entity Overlap
    new_ents = {ent.text.lower() for ent in nlp(new_query).ents}
    last_ents = {ent.text.lower() for ent in nlp(last_query).ents}
    overlap = new_ents.intersection(last_ents)
    if overlap:
        return True, 0.85

    # Step 3: Embedding similarity
    new_emb = np.array(query_store._get_embedding(new_query))
    last_emb = np.array(query_store._get_embedding(last_query))
    similarity = np.dot(new_emb, last_emb) / (np.linalg.norm(new_emb)*np.linalg.norm(last_emb))

    if similarity > threshold:
        return True, similarity

    return False, similarity

def detect_universal_follow_up(new_query, session_id, query_store, threshold=0.75):
    """Enhanced universal follow-up detection with comprehensive error handling"""
    try:
        # Get the last query with agent info
        last_query = query_store.fetch_last_query_with_agent(session_id)
        
        
        if not last_query:
            # No previous queries found
            safe_context = {
                "last_query": "",
                "last_agent": None,
                "last_sql": "",
                "last_results": [],
                "is_follow_up": False
            }
            return False, 0.0, None, safe_context
        
        # Safely unpack the tuple
        if isinstance(last_query, (list, tuple)) and len(last_query) >= 4:
            last_query, last_sql, last_results, last_agent = last_query[:4]
        else:
            print(f"Unexpected format from fetch_last_query_with_agent: {last_query}")

            safe_context = {
                "last_query": "",
                "last_agent": None,
                "last_sql": "",
                "last_results": [],
                "is_follow_up": False
            }
            return False, 0.0, None, safe_context
        
        # Ensure we have at least a query to work with
        if not last_query:

            safe_context = {
                "last_query": "",
                "last_agent": None,
                "last_sql": "",
                "last_results": [],
                "is_follow_up": False
            }
            return False, 0.0, None, safe_context
                
                # Use existing follow-up detection logic
        try:
            is_follow_up, confidence = detect_follow_up_query(new_query, session_id, query_store)
        except Exception as e:
            print(f"Error in detect_follow_up_query: {e}")
            # Fallback: simple keyword-based detection
            follow_up_keywords = ["this", "that", "these", "those", "previous", "last", "above"]
            is_follow_up = any(keyword in new_query.lower() for keyword in follow_up_keywords)
            confidence = 0.8 if is_follow_up else 0.0
        
        # Build enhanced context
        last_context = {
            "last_query": str(last_query) if last_query else "",
            "last_agent": str(last_agent) if last_agent else "nl2sql",
            "last_sql": str(last_sql) if last_sql else "",
            "last_results": last_results if last_results else [],
            "is_follow_up": is_follow_up
        }
        
        return is_follow_up, confidence, last_agent or "nl2sql", last_context
        
    except Exception as e:
        print(f"Critical error in detect_universal_follow_up: {e}")
        safe_context = {
            "last_query": "",
            "last_agent": None,
            "last_sql": "",
            "last_results": [],
            "is_follow_up": False
        }
        return False, 0.0, None, safe_context

def user_references_prev_pos(user_query):
    """
    Check if the user references 'those pos', 'these orders', etc.
    If yes, return True.
    """
    lower_q = user_query.lower()
    for phrase in COREFERENCE_PHRASES:
        if phrase in lower_q:
            return True
    return False



def extract_query_entities(query, table_names, keyword_table_map, column_equivalence):
    """
    Basic entity extraction: references to table names, synonyms, columns, date patterns, etc.
    """
    entities = set()
    lower_q = query.lower()
    
    for tbl in table_names:
        if tbl.lower() in lower_q:
            entities.add(f"table:{tbl}")

    for synonym, actual_tables in keyword_table_map.items():
        if synonym in lower_q:
            for t in actual_tables:
                entities.add(f"table:{t}")

    for col_synonym, actual_cols in column_equivalence.items():
        if col_synonym in lower_q:
            for c in actual_cols:
                entities.add(f"column:{c}")

    date_patterns = [
        r"\d{4}-\d{2}-\d{2}",
        r"yesterday|today|tomorrow",
        r"last\s+(?:week|month|year)",
        r"next\s+(?:week|month|year)"
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, lower_q, re.IGNORECASE)
        for match in matches:
            entities.add(f"date:{match}")
    return entities
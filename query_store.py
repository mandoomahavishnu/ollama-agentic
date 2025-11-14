"""
Query storage and embedding management using PostgreSQL with pgvector
COMPLETE UPDATED VERSION with robust transaction management
"""
import datetime
import json
import psycopg2
import ollama
from psycopg2.extras import Json
from decimal import Decimal
from config import DB_URL
from typing import Dict, Any, Optional, List, Tuple
import traceback
import importlib

class QueryStore:
    def __init__(self, db_url=DB_URL):
        self.db_url = db_url
        self.conn = None
        self._initialize_connection()
        print("DEBUG: Initialized QueryStore with advanced follow-up detection.")

    def _initialize_connection(self):
        """Initialize database connection with proper error handling"""
        try:
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = False
            self._ensure_connection_health()
        except Exception as e:
            print(f"Failed to initialize database connection: {e}")
            raise

    def _ensure_connection_health(self):
        """Ensure database connection is healthy"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                self.conn.commit()
        except Exception as e:
            print(f"Connection issue detected: {e}")
            self._handle_connection_error()

    def _handle_connection_error(self):
        """Handle connection errors by rolling back and reconnecting if needed"""
        try:
            if self.conn and not self.conn.closed:
                self.conn.rollback()
                print("Rolled back failed transaction")
        except Exception as rollback_error:
            print(f"Rollback failed: {rollback_error}")
            self._reconnect()

    def _reconnect(self):
        """Reconnect to the database"""
        try:
            if self.conn and not self.conn.closed:
                self.conn.close()
            self.conn = psycopg2.connect(self.db_url)
            self.conn.autocommit = False
            print("Successfully reconnected to database")
        except Exception as e:
            print(f"Reconnection failed: {e}")
            raise

    def _execute_with_retry(self, operation_func, *args, **kwargs):
        """Execute database operation with retry on connection failure"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return operation_func(*args, **kwargs)
            except (psycopg2.InterfaceError, psycopg2.OperationalError, psycopg2.InternalError) as e:
                print(f"Database error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    self._handle_connection_error()
                    continue
                else:
                    raise
            except Exception as e:
                try:
                    self.conn.rollback()
                except:
                    pass
                raise

    def _float(self, value):
        """Convert any numeric value to float"""
        return float(value) if value is not None else 0.0
    
    def _get_embedding(self, text):
        """Get embedding for text using Ollama"""
        response = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )
        return response["embedding"]

    def _serialize_results(self, results):
        """Serialize results for JSON storage"""
        if results is None:
            return None
        output = []
        for row in results:
            row_dict = {}
            for k, v in row.items():
                if isinstance(v, (datetime.date, datetime.datetime)):
                    row_dict[k] = v.isoformat()
                elif isinstance(v, Decimal):
                    row_dict[k] = float(v)
                else:
                    row_dict[k] = v
            output.append(row_dict)
        return output

    def fix_transaction_state(self):
        """Fix any existing transaction issues by forcing a rollback and reconnection if needed"""
        try:
            self.conn.rollback()
            print("Rolled back any existing failed transaction")
            
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                self.conn.commit()
                print("Database connection is healthy")
                return True
                
        except Exception as e:
            print(f"Transaction state fix failed: {e}")
            try:
                self._reconnect()
                print("Reconnected to database")
                return True
            except Exception as reconnect_error:
                print(f"Reconnection failed: {reconnect_error}")
                return False

    # Few-Shot Examples Management
    def initialize_few_shot_examples(self, examples):
        """Initialize few-shot examples in the database with proper transaction management"""
        def _do_initialize(examples):
            inserted_count = 0
            failed_count = 0
            
            for ex in examples:
                try:
                    nl = ex["natural_language"]
                    sql_q = ex["sql_query"]
                    
                    with self.conn.cursor() as cur:
                        cur.execute(
                            "SELECT COUNT(*) FROM few_shot_examples WHERE natural_language=%s",
                            (nl,)
                        )
                        count = cur.fetchone()[0]
                        
                        if count == 0:
                            emb = self._get_embedding(nl)
                            cur.execute(
                                """
                                INSERT INTO few_shot_examples (natural_language, sql_query, embedding1)
                                VALUES (%s, %s, %s::vector)
                                """,
                                (nl, sql_q, Json(emb))
                            )
                            inserted_count += 1
                            
                    self.conn.commit()
                    
                except Exception as e:
                    print(f"Failed to insert few-shot example '{nl[:50]}...': {e}")
                    failed_count += 1
                    try:
                        self.conn.rollback()
                    except Exception as rollback_error:
                        print(f"Rollback failed: {rollback_error}")
                        self._handle_connection_error()
                    continue
            
            print(f"Few-shot examples: {inserted_count} inserted, {failed_count} failed")
            return inserted_count

        return self._execute_with_retry(_do_initialize, examples)

    def search_few_shot_examples(self, query, k=3):
        """Search for similar few-shot examples with proper error handling"""
        def _do_search(query, k):
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT natural_language, sql_query
                    FROM few_shot_examples
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), k)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search, query, k)
        except Exception as e:
            print(f"Error searching few-shot examples: {e}")
            return []

    # Debugging and Utility Methods
    def debug_personal_assistant_conversations(self, session_id):
        """Debug method to check personal assistant conversation storage with proper error handling"""
        def _do_debug_pa_conversations(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, response, agent_used, created_at
                    FROM conversation_embeddings
                    WHERE session_id = %s 
                        AND (agent_used = 'personal_assistant' 
                             OR agent_used LIKE '%personal%' 
                             OR agent_used LIKE '%assistant%')
                    ORDER BY created_at DESC
                    LIMIT 10
                    """,
                    (session_id,)
                )
                results = cur.fetchall()
                self.conn.commit()
                
                print(f"Found {len(results)} personal assistant conversations:")
                for i, (query, response, agent, created_at) in enumerate(results, 1):
                    print(f"{i}. Query: {query[:50]}...")
                    print(f"   Response: {response}")
                    print(f"   Agent: {agent}")
                    print(f"   Time: {created_at}")
                    print()
                
                return results

        try:
            return self._execute_with_retry(_do_debug_pa_conversations, session_id)
        except Exception as e:
            print(f"Error debugging personal assistant conversations: {e}")
            return []

    def debug_last_query(self, session_id):
        """Debug method to see what's in the database"""
        def _do_debug_last_query(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, response, agent_used, created_at
                    FROM conversation_embeddings
                    WHERE session_id=%s
                    ORDER BY created_at DESC LIMIT 5
                    """,
                    (session_id,)
                )
                rows = cur.fetchall()
                self.conn.commit()
                
                print(f"DEBUG: Found {len(rows)} queries for session {session_id}")
                for i, row in enumerate(rows):
                    print(f"  {i+1}. Query: {row[0][:50]}... | Response: {row[1]} | Agent: {row[2]} | Time: {row[3]}")
                
                return rows

        try:
            return self._execute_with_retry(_do_debug_last_query, session_id)
        except Exception as e:
            print(f"Debug query failed: {e}")
            return []

    def debug_conversation_schema(self):
        """Debug method to check the conversation_embeddings table schema"""
        def _do_debug_conversation_schema():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'conversation_embeddings'
                    ORDER BY ordinal_position
                """)
                schema = cur.fetchall()
                
                print("DEBUG: conversation_embeddings table schema:")
                for col_name, data_type, is_nullable in schema:
                    print(f"  {col_name}: {data_type} (nullable: {is_nullable})")
                
                cur.execute("""
                    SELECT user_query, sql_query, results, agent_used, response, created_at
                    FROM conversation_embeddings
                    ORDER BY created_at DESC LIMIT 3
                """)
                sample_data = cur.fetchall()
                
                print(f"\nDEBUG: Sample data (last 3 rows):")
                for i, row in enumerate(sample_data):
                    print(f"  Row {i+1}: {len(row)} columns")
                    for j, col in enumerate(row):
                        col_str = str(col)[:50] + "..." if col and len(str(col)) > 50 else str(col)
                        print(f"    Column {j+1}: {col_str}")
                
                self.conn.commit()
                return schema, sample_data

        try:
            return self._execute_with_retry(_do_debug_conversation_schema)
        except Exception as e:
            print(f"Debug schema check failed: {e}")
            return None, None

    def debug_routing_examples(self, query, top_k=5):
        """Debug method to see what routing examples are being matched"""
        search_results = self.search_routing_examples(query, top_k=top_k)
        
        print(f"\nROUTING DEBUG for: '{query}'")
        print(f"Top {top_k} matching routing examples:")
        
        if not search_results:
            print("  No matching examples found")
            return
        
        for i, (query_text, agent_name, confidence, similarity_score) in enumerate(search_results, 1):
            print(f"{i:2d}. {agent_name:15s} | sim: {similarity_score:.3f} | conf: {confidence} | '{query_text}'")
        
        scores = self.get_routing_scores(query, top_k=10)
        print(f"\nFinal Agent Scores: {scores}")
        
        return search_results

    def get_routing_examples_info(self):
        """Get detailed information about loaded routing examples"""
        def _do_get_routing_examples_info():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT agent_name, example_type, COUNT(*) as count
                    FROM routing_examples 
                    GROUP BY agent_name, example_type
                    ORDER BY agent_name, example_type
                """)
                detailed_counts = cur.fetchall()
                
                cur.execute("""
                    SELECT agent_name, 
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence
                    FROM routing_examples 
                    GROUP BY agent_name
                    ORDER BY agent_name
                """)
                confidence_stats = cur.fetchall()
                
                cur.execute("""
                    SELECT query_text, agent_name, confidence, example_type, created_at
                    FROM routing_examples 
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                latest_examples = cur.fetchall()
                
                self.conn.commit()
                
                return {
                    "detailed_counts": detailed_counts,
                    "confidence_stats": confidence_stats,
                    "latest_examples": latest_examples
                }

        try:
            return self._execute_with_retry(_do_get_routing_examples_info)
        except Exception as e:
            print(f"Error getting routing examples info: {e}")
            return None

    def validate_routing_examples_in_db(self):
        """Validate routing examples currently in the database"""
        def _do_validate_routing_examples():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT query_text, COUNT(*) as count
                    FROM routing_examples 
                    GROUP BY query_text 
                    HAVING COUNT(*) > 1
                """)
                duplicates = cur.fetchall()
                
                cur.execute("""
                    SELECT query_text, agent_name, confidence
                    FROM routing_examples 
                    WHERE confidence < 0.7
                    ORDER BY confidence
                """)
                low_confidence = cur.fetchall()
                
                cur.execute("""
                    SELECT query_text, agent_name
                    FROM routing_examples 
                    WHERE array_length(string_to_array(query_text, ' '), 1) <= 2
                """)
                short_queries = cur.fetchall()
                
                cur.execute("""
                    SELECT agent_name, COUNT(*) as count
                    FROM routing_examples 
                    GROUP BY agent_name
                """)
                agent_counts = dict(cur.fetchall())
                
                self.conn.commit()
                
                print(f"Database Validation Results:")
                print(f"  Duplicates: {len(duplicates)}")
                print(f"  Low confidence (<0.7): {len(low_confidence)}")
                print(f"  Short queries (≤2 words): {len(short_queries)}")
                print(f"  Agent distribution: {agent_counts}")
                
                if agent_counts:
                    counts = list(agent_counts.values())
                    balance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
                    print(f"  Balance ratio: {balance_ratio:.2f}")
                
                return {
                    "duplicates": duplicates,
                    "low_confidence": low_confidence,
                    "short_queries": short_queries,
                    "agent_counts": agent_counts
                }

        try:
            return self._execute_with_retry(_do_validate_routing_examples)
        except Exception as e:
            print(f"Error validating routing examples: {e}")
            return None

    def add_routing_example_from_feedback(self, query_text, correct_agent, user_confidence=0.9):
        """Add a new routing example based on user feedback with validation"""
        def _do_add_routing_example_from_feedback(query_text, correct_agent, user_confidence):
            valid_agents = ["nl2sql", "personal_assistant", "general_chat", "document_rag"]
            if correct_agent not in valid_agents:
                print(f"Invalid agent: {correct_agent}. Must be one of {valid_agents}")
                return False
            
            if not 0.0 <= user_confidence <= 1.0:
                print(f"Invalid confidence: {user_confidence}. Must be between 0.0 and 1.0")
                return False
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT agent_name, confidence FROM routing_examples 
                    WHERE query_text = %s
                """, (query_text,))
                existing = cur.fetchone()
                
                if existing:
                    existing_agent, existing_confidence = existing
                    if existing_agent == correct_agent:
                        print(f"Example already exists with correct agent ({correct_agent})")
                        return True
                    else:
                        print(f"Example exists but with different agent: {existing_agent} → {correct_agent}")
                        embedding = self._get_embedding(query_text)
                        cur.execute("""
                            UPDATE routing_examples 
                            SET agent_name = %s, confidence = %s, embedding1 = %s::vector, 
                                example_type = 'learned', created_at = CURRENT_TIMESTAMP
                            WHERE query_text = %s
                        """, (correct_agent, user_confidence, Json(embedding), query_text))
                        self.conn.commit()
                        print(f"Updated routing example: '{query_text}' → {correct_agent}")
                        return True
                else:
                    embedding = self._get_embedding(query_text)
                    cur.execute("""
                        INSERT INTO routing_examples (query_text, agent_name, confidence, example_type, embedding1)
                        VALUES (%s, %s, %s, %s, %s::vector)
                    """, (query_text, correct_agent, user_confidence, 'learned', Json(embedding)))
                    self.conn.commit()
                    print(f"Added new routing example: '{query_text}' → {correct_agent}")
                    return True

        try:
            return self._execute_with_retry(_do_add_routing_example_from_feedback, query_text, correct_agent, user_confidence)
        except Exception as e:
            print(f"Error adding routing example: {e}")
            return False

    def refresh_routing_examples(self):
        """Refresh routing examples from the external file (useful for updates)"""
        print("Refreshing routing examples from routing_examples.py...")
        return self.initialize_routing_examples(force_refresh=True)

    def add_routing_example(self, query_text, agent_name, confidence=0.95):
        """Add a new routing example (for learning from user feedback)"""
        def _do_add_routing_example(query_text, agent_name, confidence):
            embedding = self._get_embedding(query_text)
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO routing_examples (query_text, agent_name, confidence, example_type, embedding1)
                    VALUES (%s, %s, %s, %s, %s::vector)
                    ON CONFLICT (query_text, agent_name) DO UPDATE SET 
                        confidence = EXCLUDED.confidence,
                        embedding1 = EXCLUDED.embedding1
                """, (query_text, agent_name, confidence, 'learned', Json(embedding)))
                
            self.conn.commit()
            print(f"Added routing example: '{query_text}' → {agent_name}")

        try:
            self._execute_with_retry(_do_add_routing_example, query_text, agent_name, confidence)
        except Exception as e:
            print(f"Error adding routing example: {e}")

    def get_routing_stats(self):
        """Get statistics about routing examples"""
        def _do_get_routing_stats():
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT agent_name, COUNT(*) as count
                    FROM routing_examples 
                    GROUP BY agent_name 
                    ORDER BY agent_name
                """)
                agent_counts = dict(cur.fetchall())
                
                cur.execute("""
                    SELECT example_type, COUNT(*) as count
                    FROM routing_examples 
                    GROUP BY example_type
                """)
                type_counts = dict(cur.fetchall())
                
                self.conn.commit()
                
                return {
                    "total_examples": sum(agent_counts.values()),
                    "agent_distribution": agent_counts,
                    "example_types": type_counts
                }

        try:
            return self._execute_with_retry(_do_get_routing_stats)
        except Exception as e:
            print(f"Error getting routing stats: {e}")
            return {"error": str(e)}

    def clear_routing_examples(self, example_type=None):
        """Clear routing examples (for maintenance)"""
        def _do_clear_routing_examples(example_type):
            with self.conn.cursor() as cur:
                if example_type:
                    cur.execute("DELETE FROM routing_examples WHERE example_type = %s", (example_type,))
                    print(f"Cleared {example_type} routing examples")
                else:
                    cur.execute("DELETE FROM routing_examples")
                    print("Cleared all routing examples")
                
            self.conn.commit()

        try:
            self._execute_with_retry(_do_clear_routing_examples, example_type)
        except Exception as e:
            print(f"Error clearing routing examples: {e}")

    def search_few_shot_examples_with_routing_context(self, query, last_agent=None, k=3):
        """Enhanced few-shot search that considers routing context"""
        def _do_search_few_shot_with_context(query, last_agent, k):
            embedding = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                if last_agent:
                    cur.execute("""
                        SELECT natural_language, sql_query,
                            1 - (embedding1 <=> %s::vector) as similarity_score
                        FROM few_shot_examples
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                    """, (Json(embedding), Json(embedding), k * 2))
                    
                    results = cur.fetchall()
                    self.conn.commit()
                    return results[:k]
                else:
                    cur.execute("""
                        SELECT natural_language, sql_query
                        FROM few_shot_examples
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                    """, (Json(embedding), k))
                    
                    results = cur.fetchall()
                    self.conn.commit()
                    return results

        try:
            return self._execute_with_retry(_do_search_few_shot_with_context, query, last_agent, k)
        except Exception as e:
            print(f"Error in enhanced few-shot search: {e}")
            return []

    # Conversation Storage
    def add_query(self, session_id, user_query, sql_query=None, results=None,
                  response=None, sql_error=None, user_feedback=None, agent_used=None):
        """Add a query to the conversation history with proper error handling"""
        def _do_add_query(session_id, user_query, sql_query, results, response, sql_error, user_feedback, agent_used):
            emb = self._get_embedding(user_query)
            serialized = Json(self._serialize_results(results))
            now_time = datetime.datetime.now()

            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_embeddings
                    (session_id, embedding1, user_query, sql_query, results, response,
                     sql_error, user_feedback, agent_used, created_at)
                    VALUES (%s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session_id, Json(emb), user_query, sql_query,
                        serialized, response, sql_error, user_feedback, agent_used, now_time
                    )
                )
            self.conn.commit()

        try:
            self._execute_with_retry(_do_add_query, session_id, user_query, sql_query, results, response, sql_error, user_feedback, agent_used)
        except Exception as e:
            print(f"Error adding query to conversation history: {e}")
            raise

    def search_query(self, session_id, query, k=3):
        """Search for similar queries in conversation history with proper error handling"""
        def _do_search_query(session_id, query, k):
            query = query.strip()
            if not query:
                return []
            
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, sql_query, results
                    FROM conversation_embeddings
                    WHERE session_id=%s
                          AND user_feedback!='Rejected' AND response = 'Success'
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (session_id, Json(emb), k)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search_query, session_id, query, k)
        except Exception as e:
            print(f"Error searching conversation history: {e}")
            return []

    def fetch_last_query(self, session_id):
        """Fetch the last successful query"""
        def _do_fetch_last_query(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT sql_query
                    FROM conversation_embeddings
                    WHERE session_id=%s AND response='Success' AND user_feedback!='Rejected'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id,)
                )
                row = cur.fetchone()
                self.conn.commit()
                return row[0] if row else None

        try:
            return self._execute_with_retry(_do_fetch_last_query, session_id)
        except Exception as e:
            print(f"Error fetching last query: {e}")
            return None

    def fetch_last_results(self, session_id):
        """Fetch the last successful results"""
        def _do_fetch_last_results(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT results
                    FROM conversation_embeddings
                    WHERE session_id=%s AND response='Success' AND user_feedback!='Rejected'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id,)
                )
                row = cur.fetchone()
                self.conn.commit()
                return row[0] if row else None

        try:
            return self._execute_with_retry(_do_fetch_last_results, session_id)
        except Exception as e:
            print(f"Error fetching last results: {e}")
            return None

    def fetch_last_agent(self, session_id):
        """Fetch the agent that handled the last successful query"""
        def _do_fetch_last_agent(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT agent_used
                    FROM conversation_embeddings
                    WHERE session_id=%s AND response = 'Success'
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (session_id,)
                )
                row = cur.fetchone()
                self.conn.commit()
                return row[0] if row else None

        try:
            return self._execute_with_retry(_do_fetch_last_agent, session_id)
        except Exception as e:
            print(f"Error fetching last agent: {e}")
            return None

    def fetch_last_query_with_agent(self, session_id):
        """Fetch the last query with agent info with proper error handling"""
        def _do_fetch_last_query_with_agent(session_id):
            with self.conn.cursor() as cur:
                for response_pattern in ['Success', '%Success%', 'Confirmed']:
                    cur.execute("""
                        SELECT user_query, sql_query, results, agent_used
                        FROM conversation_embeddings
                        WHERE session_id = %s AND response LIKE %s
                        ORDER BY created_at DESC LIMIT 1
                    """, (session_id, f'%{response_pattern}%'))
                    
                    row = cur.fetchone()
                    if row:
                        row_list = list(row)
                        while len(row_list) < 4:
                            row_list.append(None)
                        print(f"Found {response_pattern} query: {row_list[0][:50] if row_list[0] else 'None'}...")
                        self.conn.commit()
                        return tuple(row_list[:4])
                
                cur.execute("""
                    SELECT user_query, sql_query, results, agent_used
                    FROM conversation_embeddings
                    WHERE session_id = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (session_id,))
                
                row = cur.fetchone()
                if row:
                    row_list = list(row)
                    while len(row_list) < 4:
                        row_list.append(None)
                    print(f"Found any query: {row_list[0][:50] if row_list[0] else 'None'}...")
                    self.conn.commit()
                    return tuple(row_list[:4])
                
                print(f"No queries found for session {session_id}")
                self.conn.commit()
                return (None, None, None, None)

        try:
            return self._execute_with_retry(_do_fetch_last_query_with_agent, session_id)
        except Exception as e:
            print(f"Error in fetch_last_query_with_agent: {e}")
            return (None, None, None, None)

    # Schema & Table Examples
    def store_schema_in_pgvector(self, schema):
        """Store database schema with embeddings and proper error handling"""
        def _do_store_schema(schema):
            successful_inserts = 0
            failed_inserts = 0
            
            for row in schema:
                try:
                    table_name, column_name, data_type, is_nullable, _, _, _, column_comment = row
                    desc = f"{table_name} - {column_name} ({data_type}, nullable={is_nullable}): {column_comment}"
                    
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT COUNT(*) FROM schema_embeddings
                            WHERE table_name=%s AND column_name=%s
                            """,
                            (table_name, column_name)
                        )
                        count = cur.fetchone()[0]
                        
                        if count == 0:
                            emb = self._get_embedding(desc)
                            cur.execute(
                                """
                                INSERT INTO schema_embeddings
                                (table_name, column_name, column_description, data_type,
                                 is_nullable, embedding1)
                                VALUES (%s, %s, %s, %s, %s, %s::vector)
                                """,
                                (table_name, column_name, column_comment, data_type, is_nullable, Json(emb))
                            )
                            
                    self.conn.commit()
                    successful_inserts += 1
                    
                except Exception as e:
                    print(f"Error storing schema for {table_name}.{column_name}: {e}")
                    failed_inserts += 1
                    try:
                        self.conn.rollback()
                    except Exception as rollback_error:
                        print(f"Rollback failed: {rollback_error}")
                        self._handle_connection_error()
                    continue
            
            print(f"Schema storage: {successful_inserts} successful, {failed_inserts} failed")
            return successful_inserts

        return self._execute_with_retry(_do_store_schema, schema)

    def search_relevant_schema(self, query, k=5):
        """Search for relevant schema elements"""
        def _do_search_schema(query, k):
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name, column_name, column_description
                    FROM schema_embeddings
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), k)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search_schema, query, k)
        except Exception as e:
            print(f"Error searching schema: {e}")
            return []

    def search_tables_semantically(self, query, k=3):
        """Search for tables semantically"""
        def _do_search_tables(query, k):
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name
                    FROM table_embeddings
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), k)
                )
                rows = cur.fetchall()
                self.conn.commit()
                return [r[0] for r in rows]

        try:
            return self._execute_with_retry(_do_search_tables, query, k)
        except Exception as e:
            print(f"Error searching tables: {e}")
            return []

    def search_columns_semantically(self, query, table, k=5):
        """Search for columns semantically within a table"""
        def _do_search_columns(query, table, k):
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT column_name, column_description,
                        1 - (embedding1 <=> %s::vector) as similarity_score
                    FROM schema_embeddings
                    WHERE table_name = %s
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), table, Json(emb), k)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search_columns, query, table, k)
        except Exception as e:
            print(f"Error searching columns: {e}")
            return []

    def search_tables_semantically_enhanced(self, query, k=3, include_scores=False):
        """Enhanced semantic table search with similarity scores"""
        def _do_search_tables_enhanced(query, k, include_scores):
            emb = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                if include_scores:
                    cur.execute(
                        """
                        SELECT table_name, description, 
                            1 - (embedding1 <=> %s::vector) as similarity_score
                        FROM table_embeddings
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                        """,
                        (Json(emb), Json(emb), k)
                    )
                    results = cur.fetchall()
                    self.conn.commit()
                    return results
                else:
                    cur.execute(
                        """
                        SELECT table_name
                        FROM table_embeddings
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                        """,
                        (Json(emb), k)
                    )
                    rows = cur.fetchall()
                    self.conn.commit()
                    return [r[0] for r in rows]

        try:
            return self._execute_with_retry(_do_search_tables_enhanced, query, k, include_scores)
        except Exception as e:
            print(f"Error in enhanced table search: {e}")
            return [] if not include_scores else []

    def store_table_embeddings(self, table_names, excel_file_path="table_names.xlsx"):
        """Store table embeddings using your Excel descriptions with proper transaction handling"""
        def _do_store_table_embeddings(table_names, excel_file_path):
            try:
                excel_descriptions = self.load_table_descriptions_from_excel(excel_file_path)
                
                fallback_descriptions = {
                    'errand': 'Table: ERRAND. Contains errand tasks and miscellaneous warehouse activities including special assignments, maintenance tasks, and other non-standard operations. Related search terms: errand',
                    'sto_inbound': 'Table: STO_INBOUND. Contains inbound stock transfer information including transfers received from other locations, warehouses, or departments. Includes transfer details, quantities, and receiving status. Related search terms: from KCC, from FNS, receive, received'
                }
                
                stored_count = 0
                
                for table_name in table_names:
                    try:
                        table_name_lower = table_name.lower()
                        
                        if table_name_lower in excel_descriptions:
                            description = excel_descriptions[table_name_lower]
                        elif table_name_lower in fallback_descriptions:
                            description = fallback_descriptions[table_name_lower]
                        else:
                            description = f"Table: {table_name}. This table is about {table_name} information."
                        
                        emb = self._get_embedding(description)
                        
                        with self.conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO table_embeddings (table_name, description, embedding1)
                                VALUES (%s, %s, %s::vector)
                                ON CONFLICT (table_name) DO UPDATE SET 
                                    description = EXCLUDED.description,
                                    embedding1 = EXCLUDED.embedding1
                                """,
                                (table_name, description, Json(emb))
                            )
                        self.conn.commit()
                        stored_count += 1
                        
                    except Exception as e:
                        print(f"Error storing embedding for {table_name}: {e}")
                        self.conn.rollback()
                        continue
                
                print(f"Stored enhanced embeddings for {stored_count}/{len(table_names)} tables using Excel descriptions")
                return stored_count
                
            except Exception as e:
                print(f"Critical error in store_table_embeddings: {e}")
                self.conn.rollback()
                return 0

        return self._execute_with_retry(_do_store_table_embeddings, table_names, excel_file_path)

    def load_table_descriptions_from_excel(self, excel_file_path="table_names.xlsx"):
        """Load table descriptions from your Excel file"""
        try:
            import pandas as pd
            
            df = pd.read_excel(excel_file_path)
            table_descriptions = {}
            
            for _, row in df.iterrows():
                table_name = str(row['TABLE NAME']).lower()
                description = str(row['DESCRIPTION'])
                
                if table_name == 'pictix':
                    table_name = 'picktix'
                
                enhanced_description = self._enhance_description_with_keywords(table_name, description)
                table_descriptions[table_name] = enhanced_description
                
            return table_descriptions
            
        except Exception as e:
            print(f"Error loading Excel descriptions: {e}")
            return {}

    def _enhance_description_with_keywords(self, table_name, base_description):
        """Enhance your Excel descriptions with keyword mappings"""
        try:
            from config import KEYWORD_TABLE_MAP
            
            related_keywords = []
            for keyword, tables in KEYWORD_TABLE_MAP.items():
                if table_name in tables:
                    related_keywords.append(keyword)
            
            enhanced_parts = [f"Table: {table_name.upper()}."]
            enhanced_parts.append(base_description)
            
            if related_keywords:
                enhanced_parts.append(f"Related search terms: {', '.join(related_keywords)}")
            
            return " ".join(enhanced_parts)
        except Exception as e:
            print(f"Error enhancing description: {e}")
            return f"Table: {table_name.upper()}. {base_description}"

    def search_personal_assistant_conversations(self, session_id, query, k=5):
        """Search specifically for personal assistant conversations with proper error handling"""
        def _do_search_pa_conversations(session_id, query, k):
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, response, created_at, 
                        1 - (embedding1 <=> %s::vector) as similarity_score
                    FROM conversation_embeddings
                    WHERE session_id=%s 
                        AND agent_used = 'personal_assistant'
                        AND response != 'Processing'
                        AND response LIKE 'Success:%'
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), session_id, Json(emb), k)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search_pa_conversations, session_id, query, k)
        except Exception as e:
            print(f"Error searching personal assistant conversations: {e}")
            return []

    def get_personal_assistant_conversation_summary(self, session_id, days_back=7):
        """Get summary of recent personal assistant conversations"""
        def _do_get_pa_summary(session_id, days_back):
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
            
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, response, created_at
                    FROM conversation_embeddings
                    WHERE session_id = %s 
                        AND agent_used = 'personal_assistant'
                        AND created_at > %s
                        AND response LIKE 'Success:%'
                    ORDER BY created_at DESC
                    LIMIT 20
                    """,
                    (session_id, cutoff_date)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_get_pa_summary, session_id, days_back)
        except Exception as e:
            print(f"Error getting PA conversation summary: {e}")
            return []

    def store_table_example_rows(self, table_name, rows):
        """Store example rows for tables"""
        def _do_store_table_examples(table_name, rows):
            stored_count = 0
            for row_data in rows:
                try:
                    row_text = f"TABLE {table_name}, ROW: {row_data}"
                    emb = self._get_embedding(row_text)
                    
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT COUNT(*) FROM table_examples_embeddings
                            WHERE table_name=%s AND row_text=%s
                            """,
                            (table_name, str(row_data))
                        )
                        existing_count = cur.fetchone()[0]
                        
                        if existing_count == 0:
                            cur.execute(
                                """
                                INSERT INTO table_examples_embeddings
                                (table_name, row_text, embedding1)
                                VALUES (%s, %s, %s::vector)
                                """,
                                (table_name, str(row_data), Json(emb))
                            )
                            stored_count += 1
                    
                    self.conn.commit()
                    
                except Exception as e:
                    print(f"Error storing table example for {table_name}: {e}")
                    self.conn.rollback()
                    continue
                    
            return stored_count

        try:
            return self._execute_with_retry(_do_store_table_examples, table_name, rows)
        except Exception as e:
            print(f"Error in store_table_example_rows: {e}")
            return 0

    def search_table_example_rows(self, query, k=5):
        """Search for similar table example rows"""
        def _do_search_table_examples(query, k):
            emb = self._get_embedding(query)
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT table_name, row_text
                    FROM table_examples_embeddings
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), k)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search_table_examples, query, k)
        except Exception as e:
            print(f"Error searching table examples: {e}")
            return []

    def clear_session_conversation(self, session_id):
        """Clear conversation history for a session"""
        def _do_clear_session(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_embeddings WHERE session_id = %s",
                    (session_id,)
                )
            self.conn.commit()

        try:
            self._execute_with_retry(_do_clear_session, session_id)
        except Exception as e:
            print(f"Error clearing session conversation: {e}")

    def fetch_session_queries(self, session_id):
        """Fetch all queries for a session"""
        def _do_fetch_session_queries(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_query, sql_query, results, response, created_at, agent_used
                    FROM conversation_embeddings
                    WHERE session_id=%s
                    ORDER BY created_at ASC
                    """,
                    (session_id,)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_fetch_session_queries, session_id)
        except Exception as e:
            print(f"Error fetching session queries: {e}")
            return []

    def fetch_session_results(self, session_id):
        """Fetch all results for a session"""
        def _do_fetch_session_results(session_id):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT results
                    FROM conversation_embeddings
                    WHERE session_id=%s
                    ORDER BY created_at ASC
                    """,
                    (session_id,)
                )
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_fetch_session_results, session_id)
        except Exception as e:
            print(f"Error fetching session results: {e}")
            return []

    # Routing Examples Management
    def initialize_routing_examples(self, force_refresh: bool = False):
        """Initialize routing examples with proper transaction handling"""
        def _do_initialize_routing_examples(force_refresh):
            try:
                rex = importlib.import_module("routing_examples")

                with self.conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS routing_examples (
                            id SERIAL PRIMARY KEY,
                            query_text TEXT NOT NULL,
                            agent_name VARCHAR(50) NOT NULL,
                            confidence DECIMAL(3,2) DEFAULT 0.95,
                            example_type VARCHAR(20) DEFAULT 'curated',
                            embedding1 vector(1024),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(query_text, agent_name)
                        )
                    """)
                    self.conn.commit()

                    if force_refresh:
                        cur.execute("DELETE FROM routing_examples WHERE example_type='curated'")
                        self.conn.commit()

                    all_examples = getattr(rex, "get_all_examples")()
                    validate = getattr(rex, "validate_examples")()
                    stats = getattr(rex, "get_example_stats")()

                    if any(validate.values()):
                        print("Validation issues:", validate)

                    inserted, failed = 0, 0
                    
                    for (query_text, agent_name, confidence) in all_examples:
                        try:
                            emb = self._get_embedding(query_text)
                            
                            with self.conn.cursor() as cur_inner:
                                cur_inner.execute("""
                                    INSERT INTO routing_examples
                                    (query_text, agent_name, confidence, example_type, embedding1)
                                    VALUES (%s, %s, %s, 'curated', %s::vector)
                                    ON CONFLICT (query_text, agent_name) DO UPDATE SET
                                    confidence = EXCLUDED.confidence,
                                    embedding1 = EXCLUDED.embedding1,
                                    example_type = 'curated'
                                """, (query_text, agent_name, confidence, Json(emb)))
                                
                            self.conn.commit()
                            inserted += 1
                            
                        except Exception as e:
                            failed += 1
                            print(f"Insert error for '{query_text[:60]}': {e}")
                            try:
                                self.conn.rollback()
                            except Exception as rollback_error:
                                print(f"Rollback failed: {rollback_error}")
                                self._handle_connection_error()
                            continue

                    print(f"Routing examples upserted: {inserted} (failed: {failed}) – stats={stats}")
                    return inserted

            except Exception as e:
                print("Error initializing routing examples:", e)
                traceback.print_exc()
                try:
                    self.conn.rollback()
                except Exception as rollback_error:
                    print(f"Rollback failed: {rollback_error}")
                    self._handle_connection_error()
                return 0

        return self._execute_with_retry(_do_initialize_routing_examples, force_refresh)

    def search_routing_examples(self, query, top_k=10, min_similarity=0.3):
        """Search for similar routing examples using vector similarity"""
        def _do_search_routing_examples(query, top_k, min_similarity):
            embedding = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        query_text, 
                        agent_name, 
                        confidence,
                        1 - (embedding1 <=> %s::vector) as similarity_score
                    FROM routing_examples
                    WHERE 1 - (embedding1 <=> %s::vector) > %s
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                """, (Json(embedding), Json(embedding), min_similarity, Json(embedding), top_k))
                
                results = cur.fetchall()
                self.conn.commit()
                return results

        try:
            return self._execute_with_retry(_do_search_routing_examples, query, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching routing examples: {e}")
            return []

    def get_routing_scores(self, query, top_k=10):
        """Get agent scores based on semantic similarity to routing examples"""
        search_results = self.search_routing_examples(query, top_k=top_k)
        
        if not search_results:
            return {"nl2sql": 0.0, "personal_assistant": 0.0, "general_chat": 0.0, "document_rag": 0.0}
        
        agent_scores = {"nl2sql": 0.0, "personal_assistant": 0.0, "general_chat": 0.0, "document_rag": 0.0}
        agent_top_scores = {"nl2sql": [], "personal_assistant": [], "general_chat": [], "document_rag": []}
        
        for query_text, agent_name, confidence, similarity_score in search_results:
            if agent_name in agent_top_scores:
                confidence_float = float(confidence) if confidence is not None else 0.95
                similarity_float = float(similarity_score) if similarity_score is not None else 0.0
                
                score = similarity_float * confidence_float
                agent_top_scores[agent_name].append(score)
        
        for agent in agent_scores.keys():
            if agent_top_scores[agent]:
                top_scores = sorted(agent_top_scores[agent], reverse=True)[:5]
                agent_scores[agent] = sum(top_scores) / len(top_scores)
        
        return agent_scores

    def initialize_rag_routing_examples(self):
        """Initialize RAG routing examples in the database"""
        def _do_initialize_rag_routing():
            try:
                from document_rag_agent import get_rag_routing_examples
                rag_examples = get_rag_routing_examples()
                
                inserted_count = 0
                for query_text, agent_name, confidence in rag_examples:
                    try:
                        emb = self._get_embedding(query_text)
                        with self.conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO routing_examples (query_text, agent_name, confidence, example_type, embedding1)
                                VALUES (%s, %s, %s, 'curated_rag', %s::vector)
                                ON CONFLICT (query_text, agent_name) DO UPDATE SET
                                confidence = EXCLUDED.confidence,
                                embedding1 = EXCLUDED.embedding1
                            """, (query_text, agent_name, confidence, Json(emb)))
                        
                        self.conn.commit()
                        inserted_count += 1
                        
                    except Exception as e:
                        print(f"Error inserting RAG routing example '{query_text}': {e}")
                        self.conn.rollback()
                        continue
                
                print(f"Initialized {inserted_count} RAG routing examples")
                return inserted_count
                
            except ImportError:
                print("document_rag_agent module not found, skipping RAG routing examples")
                return 0
            except Exception as e:
                print(f"Error initializing RAG routing examples: {e}")
                return 0

        return self._execute_with_retry(_do_initialize_rag_routing)

    # Document Management
    def initialize_document_tables(self):
        """Initialize document storage tables with proper error handling"""
        def _do_initialize_document_tables():
            with self.conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        filename VARCHAR(255) NOT NULL,
                        file_type VARCHAR(10) NOT NULL,
                        file_size INTEGER,
                        file_hash VARCHAR(64) UNIQUE,
                        session_id VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                        chunk_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        word_count INTEGER,
                        char_count INTEGER,
                        embedding1 vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                    ON document_chunks USING ivfflat (embedding1 vector_cosine_ops)
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_documents_session 
                    ON documents(session_id)
                """)
                
            self.conn.commit()
            print("Document tables initialized successfully")

        try:
            self._execute_with_retry(_do_initialize_document_tables)
        except Exception as e:
            print(f"Error initializing document tables: {e}")
            raise

    def store_document_with_chunks(self, document_metadata: Dict[str, Any], chunks: List[Dict[str, Any]]) -> int:
        """Store document and its chunks with embeddings"""
        def _do_store_document(document_metadata, chunks):
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM documents WHERE file_hash = %s",
                    (document_metadata['file_hash'],)
                )
                existing = cur.fetchone()
                
                if existing:
                    print(f"Document {document_metadata['filename']} already exists")
                    return existing[0]
                
                cur.execute("""
                    INSERT INTO documents (filename, file_type, file_size, file_hash, session_id, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    document_metadata['filename'],
                    document_metadata['file_type'], 
                    document_metadata['file_size'],
                    document_metadata['file_hash'],
                    document_metadata['session_id'],
                    Json(document_metadata)
                ))
                
                document_id = cur.fetchone()[0]
                
                for chunk in chunks:
                    try:
                        embedding = self._get_embedding(chunk['text'])
                        
                        cur.execute("""
                            INSERT INTO document_chunks 
                            (document_id, chunk_index, text, word_count, char_count, embedding1)
                            VALUES (%s, %s, %s, %s, %s, %s::vector)
                        """, (
                            document_id,
                            chunk['chunk_index'],
                            chunk['text'],
                            chunk['word_count'],
                            chunk['char_count'],
                            Json(embedding)
                        ))
                        
                    except Exception as e:
                        print(f"Error storing chunk {chunk['chunk_index']}: {e}")
                        continue
                
                self.conn.commit()
                print(f"Stored document {document_metadata['filename']} with {len(chunks)} chunks")
                return document_id

        try:
            return self._execute_with_retry(_do_store_document, document_metadata, chunks)
        except Exception as e:
            print(f"Error storing document: {e}")
            raise

    def search_document_chunks(self, query: str, session_id: str, top_k: int = 5, min_similarity: float = 0.1) -> List[Dict[str, Any]]:
        """Search for relevant document chunks using vector similarity"""
        def _do_search_document_chunks(query, session_id, top_k, min_similarity):
            query_embedding = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        dc.text,
                        dc.chunk_index,
                        dc.word_count,
                        d.filename,
                        d.file_type,
                        1 - (dc.embedding1 <=> %s::vector) as similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE d.session_id = %s 
                        AND 1 - (dc.embedding1 <=> %s::vector) > %s
                    ORDER BY dc.embedding1 <=> %s::vector
                    LIMIT %s
                """, (Json(query_embedding), session_id, Json(query_embedding), min_similarity, Json(query_embedding), top_k))
                
                results = cur.fetchall()
                self.conn.commit()
                
                chunks = []
                for row in results:
                    chunks.append({
                        'text': row[0],
                        'chunk_index': row[1],
                        'word_count': row[2],
                        'filename': row[3],
                        'file_type': row[4],
                        'similarity': float(row[5])
                    })
                
                return chunks

        try:
            return self._execute_with_retry(_do_search_document_chunks, query, session_id, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching document chunks: {e}")
            return []

    def list_user_documents(self, session_id: str) -> List[Dict[str, Any]]:
        """List all documents for a user session"""
        def _do_list_user_documents(session_id):
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        d.id,
                        d.filename,
                        d.file_type,
                        d.file_size,
                        d.created_at,
                        COUNT(dc.id) as chunk_count
                    FROM documents d
                    LEFT JOIN document_chunks dc ON d.id = dc.document_id
                    WHERE d.session_id = %s
                    GROUP BY d.id, d.filename, d.file_type, d.file_size, d.created_at
                    ORDER BY d.created_at DESC
                """, (session_id,))
                
                results = cur.fetchall()
                self.conn.commit()
                
                documents = []
                for row in results:
                    documents.append({
                        'id': row[0],
                        'filename': row[1],
                        'file_type': row[2],
                        'file_size': row[3],
                        'created_at': row[4].isoformat() if row[4] else None,
                        'chunk_count': row[5]
                    })
                
                return documents

        try:
            return self._execute_with_retry(_do_list_user_documents, session_id)
        except Exception as e:
            print(f"Error listing user documents: {e}")
            return []

    def initialize_api_tables(self):
        """Create tables for API parameter and example embeddings"""
        def _do_initialize_api_tables():
            with self.conn.cursor() as cur:
                # Table for API parameter descriptions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_parameters (
                        id SERIAL PRIMARY KEY,
                        param_name VARCHAR(255) UNIQUE NOT NULL,
                        param_type VARCHAR(50),
                        operator VARCHAR(50),
                        description TEXT,
                        examples TEXT,
                        api_endpoint VARCHAR(255) DEFAULT 'warehouse_detail.php',
                        embedding1 vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Index for vector similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_parameters_embedding 
                    ON api_parameters USING ivfflat (embedding1 vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Table for API few-shot examples
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_examples (
                        id SERIAL PRIMARY KEY,
                        user_query TEXT NOT NULL,
                        api_params JSONB,
                        explanation TEXT,
                        api_endpoint VARCHAR(255) DEFAULT 'warehouse_detail.php',
                        example_type VARCHAR(50) DEFAULT 'parameter_mapping',
                        confidence FLOAT DEFAULT 1.0,
                        embedding1 vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Index for vector similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_examples_embedding 
                    ON api_examples USING ivfflat (embedding1 vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                self.conn.commit()
                print("✅ API parameter and example tables initialized")
                
            return True
        
        try:
            return self._execute_with_retry(_do_initialize_api_tables)
        except Exception as e:
            print(f"❌ Error initializing API tables: {e}")
            return False


    def store_api_parameter_embeddings(self, parameters_dict: Dict[str, Dict[str, Any]]):
        """
        Store API parameter descriptions as embeddings
        
        Args:
            parameters_dict: Dict mapping param_name to param info
                {
                    "so_num": {
                        "type": "string",
                        "operator": "LIKE",
                        "description": "Sales order number...",
                        "examples": ["SO12345", "12345"]
                    },
                    ...
                }
        """
        def _do_store_api_parameters(parameters_dict):
            stored_count = 0
            failed_count = 0
            
            for param_name, param_info in parameters_dict.items():
                try:
                    # Create rich text for embedding that includes all context
                    description_text = f"""
    Parameter: {param_name}
    Type: {param_info.get('type', 'string')}
    Operator: {param_info.get('operator', 'exact')}
    Description: {param_info.get('description', '')}
    Examples: {', '.join(param_info.get('examples', []))}
    """
                    
                    # Generate embedding
                    emb = self._get_embedding(description_text)
                    
                    # Store in database
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO api_parameters 
                            (param_name, param_type, operator, description, examples, embedding1)
                            VALUES (%s, %s, %s, %s, %s, %s::vector)
                            ON CONFLICT (param_name) DO UPDATE SET
                                param_type = EXCLUDED.param_type,
                                operator = EXCLUDED.operator,
                                description = EXCLUDED.description,
                                examples = EXCLUDED.examples,
                                embedding1 = EXCLUDED.embedding1
                            """,
                            (
                                param_name,
                                param_info.get('type', 'string'),
                                param_info.get('operator', 'exact'),
                                param_info.get('description', ''),
                                json.dumps(param_info.get('examples', [])),
                                Json(emb)
                            )
                        )
                    
                    self.conn.commit()
                    stored_count += 1
                    
                except Exception as e:
                    print(f"Error storing parameter {param_name}: {e}")
                    self.conn.rollback()
                    failed_count += 1
                    continue
            
            print(f"API Parameters: {stored_count} stored, {failed_count} failed")
            return stored_count
        
        try:
            return self._execute_with_retry(_do_store_api_parameters, parameters_dict)
        except Exception as e:
            print(f"Error in store_api_parameter_embeddings: {e}")
            return 0


    def search_api_parameters(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Tuple[str, str, str, List[str], float]]:
        """
        Search for relevant API parameters based on user query
        
        Returns:
            List of (param_name, param_type, description, examples, similarity_score)
        """
        def _do_search_api_parameters(query, top_k, min_similarity):
            emb = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        param_name,
                        param_type,
                        description,
                        examples,
                        1 - (embedding1 <=> %s::vector) as similarity_score
                    FROM api_parameters
                    WHERE 1 - (embedding1 <=> %s::vector) >= %s
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), Json(emb), min_similarity, Json(emb), top_k)
                )
                
                results = cur.fetchall()
                self.conn.commit()
                
                # Parse examples from JSON string
                parsed_results = []
                for param_name, param_type, description, examples_json, similarity in results:
                    try:
                        examples = json.loads(examples_json) if examples_json else []
                    except:
                        examples = []
                    parsed_results.append((param_name, param_type, description, examples, similarity))
                
                return parsed_results
        
        try:
            return self._execute_with_retry(_do_search_api_parameters, query, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching API parameters: {e}")
            return []


    def store_api_example_embeddings(self, examples: List[Dict[str, Any]]):
        """
        Store API few-shot examples as embeddings
        
        Args:
            examples: List of example dicts
                [
                    {
                        "user_query": "Show me all sales orders shipped today",
                        "api_params": {"shipped": "not_empty", "shipped_date_from": "2024-11-09"},
                        "explanation": "User wants orders that have been shipped today...",
                        "confidence": 1.0
                    },
                    ...
                ]
        """
        def _do_store_api_examples(examples):
            stored_count = 0
            failed_count = 0
            
            for example in examples:
                try:
                    user_query = example.get("user_query", "")
                    api_params = example.get("api_params", {})
                    explanation = example.get("explanation", "")
                    confidence = example.get("confidence", 1.0)
                    
                    # Create rich text for embedding
                    example_text = f"""
    User Query: {user_query}
    API Parameters: {json.dumps(api_params, indent=2)}
    Explanation: {explanation}
    """
                    
                    # Generate embedding
                    emb = self._get_embedding(example_text)
                    
                    # Store in database
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO api_examples 
                            (user_query, api_params, explanation, confidence, embedding1)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            """,
                            (
                                user_query,
                                Json(api_params),
                                explanation,
                                confidence,
                                Json(emb)
                            )
                        )
                    
                    self.conn.commit()
                    stored_count += 1
                    
                except Exception as e:
                    print(f"Error storing API example: {e}")
                    self.conn.rollback()
                    failed_count += 1
                    continue
            
            print(f"API Examples: {stored_count} stored, {failed_count} failed")
            return stored_count
        
        try:
            return self._execute_with_retry(_do_store_api_examples, examples)
        except Exception as e:
            print(f"Error in store_api_example_embeddings: {e}")
            return 0


    def search_api_examples(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Tuple[str, Dict[str, Any], str, float, float]]:
        """
        Search for relevant API examples based on user query
        
        Returns:
            List of (user_query, api_params, explanation, confidence, similarity_score)
        """
        def _do_search_api_examples(query, top_k, min_similarity):
            emb = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        user_query,
                        api_params,
                        explanation,
                        confidence,
                        1 - (embedding1 <=> %s::vector) as similarity_score
                    FROM api_examples
                    WHERE 1 - (embedding1 <=> %s::vector) >= %s
                    ORDER BY embedding1 <=> %s::vector
                    LIMIT %s
                    """,
                    (Json(emb), Json(emb), min_similarity, Json(emb), top_k)
                )
                
                results = cur.fetchall()
                self.conn.commit()
                return results
        
        try:
            return self._execute_with_retry(_do_search_api_examples, query, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching API examples: {e}")
            return []


    def get_api_parameter_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific API parameter"""
        def _do_get_api_parameter_info(param_name):
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT param_name, param_type, operator, description, examples
                    FROM api_parameters
                    WHERE param_name = %s
                    """,
                    (param_name,)
                )
                
                result = cur.fetchone()
                self.conn.commit()
                
                if result:
                    param_name, param_type, operator, description, examples_json = result
                    try:
                        examples = json.loads(examples_json) if examples_json else []
                    except:
                        examples = []
                    
                    return {
                        "param_name": param_name,
                        "type": param_type,
                        "operator": operator,
                        "description": description,
                        "examples": examples
                    }
                return None
        
        try:
            return self._execute_with_retry(_do_get_api_parameter_info, param_name)
        except Exception as e:
            print(f"Error getting API parameter info: {e}")
            return None


    def debug_api_embeddings(self, query: str, top_k: int = 5):
        """Debug method to see what API parameters and examples match a query"""
        print(f"\n{'='*60}")
        print(f"API EMBEDDING DEBUG for: '{query}'")
        print(f"{'='*60}")
        
        # Search parameters
        print(f"\n📋 Top {top_k} Matching Parameters:")
        param_results = self.search_api_parameters(query, top_k=top_k, min_similarity=0.0)
        
        if not param_results:
            print("  No matching parameters found")
        else:
            for i, (param_name, param_type, description, examples, similarity) in enumerate(param_results, 1):
                print(f"{i:2d}. {param_name:20s} | sim: {similarity:.3f} | type: {param_type}")
                print(f"    Description: {description[:80]}...")
                print(f"    Examples: {', '.join(examples[:3])}")
        
        # Search examples
        print(f"\n📝 Top {top_k} Matching Examples:")
        example_results = self.search_api_examples(query, top_k=top_k, min_similarity=0.0)
        
        if not example_results:
            print("  No matching examples found")
        else:
            for i, (user_query, api_params, explanation, confidence, similarity) in enumerate(example_results, 1):
                print(f"{i:2d}. sim: {similarity:.3f} | conf: {confidence:.2f}")
                print(f"    Query: {user_query}")
                print(f"    Params: {json.dumps(api_params, indent=8)}")
                print(f"    Explanation: {explanation[:100]}...")
        
        print(f"\n{'='*60}\n")
        
        return param_results, example_results


    def get_api_stats(self) -> Dict[str, Any]:
        """Get statistics about stored API embeddings"""
        def _do_get_api_stats():
            stats = {}
            
            with self.conn.cursor() as cur:
                # Parameter count
                cur.execute("SELECT COUNT(*) FROM api_parameters")
                stats['parameter_count'] = cur.fetchone()[0]
                
                # Example count
                cur.execute("SELECT COUNT(*) FROM api_examples")
                stats['example_count'] = cur.fetchone()[0]
                
                # Parameter types
                cur.execute("""
                    SELECT param_type, COUNT(*) as count
                    FROM api_parameters
                    GROUP BY param_type
                    ORDER BY count DESC
                """)
                stats['parameters_by_type'] = dict(cur.fetchall())
                
                # Example confidence stats
                cur.execute("""
                    SELECT 
                        AVG(confidence) as avg_confidence,
                        MIN(confidence) as min_confidence,
                        MAX(confidence) as max_confidence
                    FROM api_examples
                """)
                conf_stats = cur.fetchone()
                stats['confidence_stats'] = {
                    'avg': float(conf_stats[0]) if conf_stats[0] else 0,
                    'min': float(conf_stats[1]) if conf_stats[1] else 0,
                    'max': float(conf_stats[2]) if conf_stats[2] else 0
                }
                
                self.conn.commit()
            
            return stats
        
        try:
            return self._execute_with_retry(_do_get_api_stats)
        except Exception as e:
            print(f"Error getting API stats: {e}")
            return {}
    def initialize_api_tables_v2(self):
        """
        Initialize enhanced API tables with proper endpoint support
        """
        def _do_initialize_api_tables_v2():
            with self.conn.cursor() as cur:
                # Table for API endpoint descriptions
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_endpoints (
                        id SERIAL PRIMARY KEY,
                        endpoint_name VARCHAR(255) UNIQUE NOT NULL,
                        endpoint_url VARCHAR(255) NOT NULL,
                        description TEXT NOT NULL,
                        keywords TEXT[],
                        embedding1 vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Index for vector similarity search on endpoints
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_endpoints_embedding 
                    ON api_endpoints USING ivfflat (embedding1 vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Enhanced API parameters table with endpoint reference
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_parameters_v2 (
                        id SERIAL PRIMARY KEY,
                        endpoint_name VARCHAR(255) NOT NULL,
                        param_name VARCHAR(255) NOT NULL,
                        param_type VARCHAR(50),
                        operator VARCHAR(50),
                        description TEXT,
                        examples TEXT,
                        embedding1 vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(endpoint_name, param_name)
                    )
                """)
                
                # Index for vector similarity search on parameters
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_parameters_v2_embedding 
                    ON api_parameters_v2 USING ivfflat (embedding1 vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Index for endpoint filtering
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_parameters_v2_endpoint 
                    ON api_parameters_v2(endpoint_name)
                """)
                
                # Enhanced API examples table - no explanation field
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_examples_v2 (
                        id SERIAL PRIMARY KEY,
                        natural_language TEXT NOT NULL,
                        endpoint_name VARCHAR(255) NOT NULL,
                        api_params JSONB,
                        confidence FLOAT DEFAULT 1.0,
                        embedding1 vector(1024),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Index for vector similarity search on examples
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_examples_v2_embedding 
                    ON api_examples_v2 USING ivfflat (embedding1 vector_cosine_ops)
                    WITH (lists = 100)
                """)
                
                # Index for endpoint filtering
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_examples_v2_endpoint 
                    ON api_examples_v2(endpoint_name)
                """)
                
                self.conn.commit()
                print("✅ Enhanced multi-endpoint API tables initialized")
                
            return True
        
        try:
            return self._execute_with_retry(_do_initialize_api_tables_v2)
        except Exception as e:
            print(f"❌ Error initializing enhanced API tables: {e}")
            return False


    def store_api_endpoints(self, endpoints_config: dict):
        """
        Store API endpoint configurations with embeddings
        
        Args:
            endpoints_config: Dict from api_endpoints_config.py API_ENDPOINTS
        """
        def _do_store_api_endpoints(endpoints_config):
            stored_count = 0
            failed_count = 0
            
            for endpoint_name, config in endpoints_config.items():
                try:
                    # Create rich text for embedding
                    
                    endpoint_text = f"""
    Endpoint: {endpoint_name}
    URL: {config.get('url', '')}
    Description: {config.get('description', '')}
    Keywords: {', '.join(config.get('keywords', []))}
    """
                    
                    # Generate embedding
                    emb = self._get_embedding(endpoint_text)
                    
                    # Store in database
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO api_endpoints 
                            (endpoint_name, endpoint_url, description, keywords, embedding1)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            ON CONFLICT (endpoint_name) DO UPDATE SET
                                endpoint_url = EXCLUDED.endpoint_url,
                                description = EXCLUDED.description,
                                keywords = EXCLUDED.keywords,
                                embedding1 = EXCLUDED.embedding1
                            """,
                            (
                                endpoint_name,
                                config.get('url', ''),
                                config.get('description', ''),
                                config.get('keywords', []),
                                Json(emb)
                            )
                        )
                    
                    self.conn.commit()
                    stored_count += 1
                    
                except Exception as e:
                    print(f"Error storing endpoint {endpoint_name}: {e}")
                    self.conn.rollback()
                    failed_count += 1
                    continue
            
            print(f"API Endpoints: {stored_count} stored, {failed_count} failed")
            return stored_count
        
        try:
            return self._execute_with_retry(_do_store_api_endpoints, endpoints_config)
        except Exception as e:
            print(f"Error in store_api_endpoints: {e}")
            return 0


    def search_api_endpoints(self, query: str, top_k: int = 3, min_similarity: float = 0.15):
        """
        Search for relevant API endpoints based on user query
        
        Returns:
            List of (endpoint_name, endpoint_url, description, keywords, similarity_score)
        """
        def _do_search_api_endpoints(query, top_k, min_similarity):
            emb = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        endpoint_name,
                        endpoint_url,
                        description,
                        keywords,
                        1 - (embedding1 <=> %s::vector) as similarity_score
                    FROM api_endpoints
                    
                    ORDER BY embedding1 <=> %s::vector NULLS LAST
                    LIMIT %s
                    """,
                    (Json(emb), Json(emb), top_k)
                )
                
                results = cur.fetchall()
                self.conn.commit()
                return results
        
        try:
            return self._execute_with_retry(_do_search_api_endpoints, query, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching API endpoints: {e}")
            return []


    def store_api_parameters_v2(self, endpoint_name: str, parameters_dict: dict):
        """
        Store API parameters for a specific endpoint
        
        Args:
            endpoint_name: Name of the endpoint
            parameters_dict: Dict of parameter configurations
        """
        def _do_store_api_parameters_v2(endpoint_name, parameters_dict):
            stored_count = 0
            failed_count = 0
            
            for param_name, param_info in parameters_dict.items():
                try:
                    # Create rich text for embedding
                    param_text = f"""
    Endpoint: {endpoint_name}
    Parameter: {param_name}
    Type: {param_info.get('type', 'string')}
    Operator: {param_info.get('operator', 'exact')}
    Description: {param_info.get('description', '')}
    Examples: {', '.join(param_info.get('examples', []))}
    """
                    
                    # Generate embedding
                    emb = self._get_embedding(param_text)
                    
                    # Store in database
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO api_parameters_v2 
                            (endpoint_name, param_name, param_type, operator, description, examples, embedding1)
                            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                            ON CONFLICT (endpoint_name, param_name) DO UPDATE SET
                                param_type = EXCLUDED.param_type,
                                operator = EXCLUDED.operator,
                                description = EXCLUDED.description,
                                examples = EXCLUDED.examples,
                                embedding1 = EXCLUDED.embedding1
                            """,
                            (
                                endpoint_name,
                                param_name,
                                param_info.get('type', 'string'),
                                param_info.get('operator', 'exact'),
                                param_info.get('description', ''),
                                json.dumps(param_info.get('examples', [])),
                                Json(emb)
                            )
                        )
                    
                    self.conn.commit()
                    stored_count += 1
                    
                except Exception as e:
                    print(f"Error storing parameter {endpoint_name}.{param_name}: {e}")
                    self.conn.rollback()
                    failed_count += 1
                    continue
            
            print(f"Parameters for {endpoint_name}: {stored_count} stored, {failed_count} failed")
            return stored_count
        
        try:
            return self._execute_with_retry(_do_store_api_parameters_v2, endpoint_name, parameters_dict)
        except Exception as e:
            print(f"Error in store_api_parameters_v2: {e}")
            return 0


    def search_api_parameters_v2(self, query: str, endpoint_name: str = None, 
                                top_k: int = 10, min_similarity: float = 0.15):
        """
        Search for relevant API parameters, optionally filtered by endpoint
        
        Args:
            query: User query
            endpoint_name: Optional endpoint to filter by
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (endpoint_name, param_name, param_type, description, examples, similarity_score)
        """
        def _do_search_api_parameters_v2(query, endpoint_name, top_k, min_similarity):
            emb = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                if endpoint_name:
                    # Filter by specific endpoint
                    cur.execute(
                        """
                        SELECT 
                            endpoint_name,
                            param_name,
                            param_type,
                            description,
                            examples,
                            1 - (embedding1 <=> %s::vector) as similarity_score
                        FROM api_parameters_v2
                        WHERE endpoint_name = %s 
                        AND 1 - (embedding1 <=> %s::vector) >= %s
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                        """,
                        (Json(emb), endpoint_name, Json(emb), min_similarity, Json(emb), top_k)
                    )
                else:
                    # Search across all endpoints
                    cur.execute(
                        """
                        SELECT 
                            endpoint_name,
                            param_name,
                            param_type,
                            description,
                            examples,
                            1 - (embedding1 <=> %s::vector) as similarity_score
                        FROM api_parameters_v2
                        WHERE 1 - (embedding1 <=> %s::vector) >= %s
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                        """,
                        (Json(emb), Json(emb), min_similarity, Json(emb), top_k)
                    )
                
                results = cur.fetchall()
                self.conn.commit()
                
                # Parse examples from JSON
                parsed_results = []
                for endpoint, param_name, param_type, description, examples_json, similarity in results:
                    try:
                        examples = json.loads(examples_json) if examples_json else []
                    except:
                        examples = []
                    parsed_results.append((endpoint, param_name, param_type, description, examples, similarity))
                
                return parsed_results
        
        try:
            return self._execute_with_retry(_do_search_api_parameters_v2, query, endpoint_name, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching API parameters v2: {e}")
            return []


    def store_api_examples_v2(self, examples: list):
        """
        Store API examples with endpoint information (no explanation field)
        
        Args:
            examples: List of example dicts from api_examples_generator.py
        """
        def _do_store_api_examples_v2(examples):
            stored_count = 0
            failed_count = 0
            
            for example in examples:
                try:
                    natural_language = example.get("natural_language", "")
                    endpoint_name = example.get("endpoint", "")
                    api_params = example.get("api_params", {})
                    confidence = example.get("confidence", 1.0)
                    
                    # Create text for embedding - just natural language + endpoint
                    # The parameter descriptions provide all context needed
                    example_text = f"""
    Query: {natural_language}
    Endpoint: {endpoint_name}
    Parameters: {', '.join(api_params.keys())}
    """
                    
                    # Generate embedding
                    emb = self._get_embedding(example_text)
                    
                    # Store in database
                    with self.conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO api_examples_v2 
                            (natural_language, endpoint_name, api_params, confidence, embedding1)
                            VALUES (%s, %s, %s, %s, %s::vector)
                            """,
                            (
                                natural_language,
                                endpoint_name,
                                Json(api_params),
                                confidence,
                                Json(emb)
                            )
                        )
                    
                    self.conn.commit()
                    stored_count += 1
                    
                except Exception as e:
                    print(f"Error storing API example v2: {e}")
                    self.conn.rollback()
                    failed_count += 1
                    continue
            
            print(f"API Examples v2: {stored_count} stored, {failed_count} failed")
            return stored_count
        
        try:
            return self._execute_with_retry(_do_store_api_examples_v2, examples)
        except Exception as e:
            print(f"Error in store_api_examples_v2: {e}")
            return 0


    def search_api_examples_v2(self, query: str, endpoint_name: str = None,
                            top_k: int = 10, min_similarity: float = 0.15):
        """
        Search for relevant API examples, optionally filtered by endpoint
        
        Returns:
            List of (natural_language, endpoint_name, api_params, confidence, similarity_score)
        """
        def _do_search_api_examples_v2(query, endpoint_name, top_k, min_similarity):
            emb = self._get_embedding(query)
            
            with self.conn.cursor() as cur:
                if endpoint_name:
                    # Filter by specific endpoint
                    cur.execute(
                        """
                        SELECT 
                            natural_language,
                            endpoint_name,
                            api_params,
                            confidence,
                            1 - (embedding1 <=> %s::vector) as similarity_score
                        FROM api_examples_v2
                        WHERE endpoint_name = %s
                        AND 1 - (embedding1 <=> %s::vector) >= %s
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                        """,
                        (Json(emb), endpoint_name, Json(emb), min_similarity, Json(emb), top_k)
                    )
                else:
                    # Search across all endpoints
                    cur.execute(
                        """
                        SELECT 
                            natural_language,
                            endpoint_name,
                            api_params,
                            confidence,
                            1 - (embedding1 <=> %s::vector) as similarity_score
                        FROM api_examples_v2
                        WHERE 1 - (embedding1 <=> %s::vector) >= %s
                        ORDER BY embedding1 <=> %s::vector
                        LIMIT %s
                        """,
                        (Json(emb), Json(emb), min_similarity, Json(emb), top_k)
                    )
                
                results = cur.fetchall()
                self.conn.commit()
                return results
        
        try:
            return self._execute_with_retry(_do_search_api_examples_v2, query, endpoint_name, top_k, min_similarity)
        except Exception as e:
            print(f"Error searching API examples v2: {e}")
            return []


    def initialize_all_api_data(self):
        """
        Convenience method to initialize all API endpoints, parameters, and examples
        """
        try:
            # Import configuration
            from api_endpoints_config import API_ENDPOINTS
            from api_examples_generator import generate_api_examples
            
            # Initialize tables
            self.initialize_api_tables_v2()
            
            # Store endpoints
            print("\n📍 Storing API endpoints...")
            self.store_api_endpoints(API_ENDPOINTS)
            
            # Store parameters for each endpoint
            print("\n⚙️ Storing API parameters...")
            for endpoint_name, config in API_ENDPOINTS.items():
                parameters = config.get('parameters', {})
                if parameters:
                    self.store_api_parameters_v2(endpoint_name, parameters)
            
            # Store examples
            print("\n📚 Storing API examples...")
            examples = generate_api_examples()
            self.store_api_examples_v2(examples)
            
            print("\n✅ All API data initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing API data: {e}")
            import traceback
            traceback.print_exc()
            return False
        

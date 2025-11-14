"""
Enhanced Multi-Endpoint API Agent
==================================

This agent supports multiple API endpoints with intelligent routing:
1. Detects which endpoint to use based on query
2. Maps parameters for the selected endpoint
3. Calls the appropriate API
4. Falls back to NL2SQL if needed

Key improvements:
- Multi-endpoint support
- No explanation field (cleaner structure)
- Dynamic endpoint and parameter loading
- Better semantic matching
"""

import json
import requests
import io
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import ollama
from config import LLM_MODEL
import streamlit as st
from query_store import QueryStore
import time


class MultiEndpointAPIAgent:
    """Enhanced agent that supports multiple API endpoints"""
    
    def __init__(self, query_store: QueryStore, api_base_url: str = "http://localhost/creme/API_LLM", unified_context_manager=None, **kwargs):
        self.name = "MultiEndpointAPI"
        self.query_store = query_store
        self.api_base_url = api_base_url.rstrip('/')
        self.unified_context_manager = unified_context_manager
        
        # Initialize enhanced API tables if not exists
        self.query_store.initialize_all_api_data()  # Populate endpoints
        self.query_store.initialize_api_tables_v2()
        try:
            with self.query_store.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM api_endpoints")
                n = cur.fetchone()[0]
            if n == 0:
                from api_endpoints_config import API_ENDPOINTS
                self.query_store.store_api_endpoints(API_ENDPOINTS)
                for name, cfg in API_ENDPOINTS.items():
                    self.query_store.store_api_parameters_v2(name, cfg.get("parameters", {}))
                    self.query_store.store_api_examples_v2(name, cfg.get("examples", []))
        except Exception as e:
            print("Bootstrap endpoints failed:", e)
    
    def _detect_endpoint(self, user_query: str) -> Tuple[Optional[str], str, float]:
        """
        Detect which API endpoint to use based on query.

        Returns:
            (endpoint_name, reason, confidence_score)
        """
        query_lower = (user_query or "").lower()

        # ------------------------------------------------------------------
        # 1) Synonym boosts: hard-coded phrases that strongly indicate an endpoint
        # ------------------------------------------------------------------
        synonym_boosts = {
            # Sales orders / shipping / outbound style queries
            "salesorder": [
                "sales order", "salesorder", "so ", "so#", "so number",
                "customer po", "po number", "products"
                
            ],
            "salesorder detail": [
                "sales order detail", "so detail", "missing", "eta", "eta item","repacking item"
                "missing product","missing units"
                
            ],
            "outbound": [
                "outbound", "ship", "ship out", "pick up", "shipping schedule", "pick up schedule",
                "etd", "carrier" 
                
            ],            
            # Inventory / stock / bin queries
            "inventory": [
                "inventory", "stock", "on hand", "in stock", "available units",
                "how many boxes", "bin", "location", "where is", "stock level",
                "qty on hand"
            ],
            # Inbound / receiving / STO queries
            "inbound": [
                "inbound", "receive", "receiving", "received", "incoming",
                "coming in", "container", "unload", "FNS", "return", "air shipment", "air", "ltl"
                
            ],
            "warehouse": [
                "crew", "t1", "t3", "t2","t4","t5","t6","song", "local","working", "routing","finish","fulfillment","work", "process","summary"
                               
            ],
            "picktix": [
                "picked location", "picked qty", "missing"
                               
            ],


            # Add more endpoints here as you define them...
        }

        # Helper to parse the keywords text column from api_endpoints
        def _parse_keywords(kw_raw: Any) -> List[str]:
            if kw_raw is None:
                return []
            if isinstance(kw_raw, list):
                return [str(k).strip().lower() for k in kw_raw if str(k).strip()]
            s = str(kw_raw).strip()
            # Strip braces if it's like "{a,b,c}"
            if s.startswith("{") and s.endswith("}"):
                s = s[1:-1]
            parts = [p.strip().lower() for p in s.split(",")]
            return [p for p in parts if p]

        # ------------------------------------------------------------------
        # 2) Retrieve candidates from Postgres via embeddings
        # ------------------------------------------------------------------
        try:
            candidates = self.query_store.search_api_endpoints(
                user_query,      # query: str
                top_k=5,
                min_similarity=0.0  # let everything in, we’ll filter by combined score
            )
        except Exception as e:
            print(f"[API_AGENT] Error searching API endpoints: {e}")
            return None, f"Endpoint search failed: {e}", 0.0

        if not candidates:
            return None, "No relevant endpoints found", 0.0

        # candidates: List[(endpoint_name, endpoint_url, description, keywords, emb_sim)]
        scored_candidates = []
        debug_lines = []

        for endpoint_name, endpoint_url, description, keywords, emb_sim in candidates:
            ep_name = str(endpoint_name)
            emb_sim = float(emb_sim or 0.0)

            # ---- Keyword score from DB 'keywords' column ----
            kw_terms = _parse_keywords(keywords)
            kw_hits = 0
            for kw in kw_terms:
                if kw and kw in query_lower:
                    kw_hits += 1
            # each hit gives 0.2, capped at 1.0
            kw_score = min(kw_hits * 0.2, 1.0)

            # ---- Synonym score from hard-coded synonym_boosts ----
            syn_hits = 0
            for syn in synonym_boosts.get(ep_name, []):
                if syn and syn in query_lower:
                    syn_hits += 1
            # each synonym hit gives 0.35, capped at 1.0
            syn_score = min(syn_hits * 0.35, 1.0)

            # combine keyword + synonym score, capped at 1.0
            kw_total = min(kw_score + syn_score, 1.0)

            # ---- Final combined score: blend embeddings + kw/syn ----
            emb_weight = 0.6
            kw_weight = 0.4
            final_score = emb_weight * emb_sim + kw_weight * kw_total

            scored_candidates.append((ep_name, endpoint_url, emb_sim, kw_total, final_score))

            debug_lines.append(
                f"{ep_name}: emb={emb_sim:.2f}, kw={kw_score:.2f}, syn={syn_score:.2f}, "
                f"kw_total={kw_total:.2f}, final={final_score:.2f}, "
                f"kw_terms={kw_terms}"
            )

        # Pick best by final_score
        scored_candidates.sort(key=lambda x: x[4], reverse=True)
        best_ep, best_url, best_emb, best_kwtotal, best_final = scored_candidates[0]

        # ------------------------------------------------------------------
        # 3) Use examples as an additional signal (can override endpoint)
        # ------------------------------------------------------------------
        endpoint_votes: Dict[str, float] = {}
        try:
            examples_all = self.query_store.search_api_examples_v2(
                user_query,
                endpoint_name=None,   # search all endpoints
                top_k=8,
                min_similarity=0.15
            )
        except Exception as e:
            print(f"[API_AGENT] Error searching API examples: {e}")
            examples_all = []

        for natural_language, ep_name, api_params, conf, sim in examples_all:
            ep_name = str(ep_name)
            # base vote from similarity
            vote = float(sim or 0.0)

            # extra synonym vote if this endpoint's synonyms appear
            for syn in synonym_boosts.get(ep_name, []):
                if syn and syn in query_lower:
                    vote += 0.25

            endpoint_votes[ep_name] = endpoint_votes.get(ep_name, 0.0) + vote

        examples_override_reason = ""
        final_endpoint = best_ep
        final_confidence = best_final

        if endpoint_votes:
            best_example_ep = max(endpoint_votes, key=endpoint_votes.get)
            best_example_vote = endpoint_votes[best_example_ep]

            # If examples strongly agree on a different endpoint, override
            if best_example_ep != best_ep and best_example_vote >= 0.7:
                examples_override_reason = (
                    f"; examples override to {best_example_ep} (vote={best_example_vote:.2f})"
                )
                final_endpoint = best_example_ep
                # use a blend of the two confidences
                final_confidence = max(final_confidence, min(1.0, best_example_vote))

        # ------------------------------------------------------------------
        # 4) Build human-readable reason string for debugging / UI
        # ------------------------------------------------------------------
        # compact summary for the best candidate
        base_reason = (
            f"kw/emb blend → {best_ep} "
            f"(emb={best_emb:.2f}, kw_total={best_kwtotal:.2f}, final={best_final:.2f})"
        )

        # If we overrode with examples, mention that
        if examples_override_reason:
            reason = base_reason + examples_override_reason
        else:
            reason = base_reason

        # Also log detailed candidate breakdown to console for debugging
        print("[API_AGENT] _detect_endpoint candidates:")
        for line in debug_lines:
            print("   " + line)
        if endpoint_votes:
            print(f"[API_AGENT] example votes: {endpoint_votes}")

        return final_endpoint, reason, float(final_confidence)




    
    def _map_parameters_for_endpoint(
        self,
        user_query: str,
        endpoint_name: str,
        session_state: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str]:
        """
        Map query to API parameters for a specific endpoint, mixing:
        - api_parameters_v2 (schema/definitions)  ~50%
        - api_examples_v2 (real calls)           ~50%

        Behavior:
        1) If the query is almost identical to a stored example for this endpoint,
           we copy that example's params exactly (no LLM).
        2) Otherwise, we call the LLM but strictly constrain:
           - Only allowed param names (from parameters + examples)
           - Prefer params that are relevant to the query
           - It’s OK to omit params that cannot be inferred.
        """

        # -----------------------------
        # 1) Fetch parameter schema and examples for this endpoint
        # -----------------------------
        try:
            # All-ish params for whitelist (low similarity threshold)
            all_params_raw = self.query_store.search_api_parameters_v2(
                user_query,                      # query
                endpoint_name=endpoint_name,
                top_k=100,
                min_similarity=0.0,             # include everything for this endpoint
            )
        except TypeError:
            # Fallback for older signatures (without min_similarity)
            all_params_raw = self.query_store.search_api_parameters_v2(
                user_query,
                endpoint_name=endpoint_name,
                top_k=100,
            )

        # Parameters most semantically relevant to this particular query
        relevant_params_raw = self.query_store.search_api_parameters_v2(
            user_query,
            endpoint_name=endpoint_name,
            top_k=15,
            min_similarity=0.15,
        )

        # Examples for this endpoint (we’ll use these both for:
        # 1) exact/near-exact match shortcut, and
        # 2) LLM conditioning + default fill-in)
        examples_raw = self.query_store.search_api_examples_v2(
            user_query,
            endpoint_name=endpoint_name,
            top_k=10,
            min_similarity=0.0,   # get a decent spread, we’ll look at similarity ourselves
        )

        # -----------------------------
        # 2) Build whitelist of allowed parameter names
        # -----------------------------
        allowed_param_names: set[str] = set()

        # From parameter definitions
        for ep, param_name, param_type, description, examples, sim in all_params_raw:
            if ep == endpoint_name:
                allowed_param_names.add(param_name)

        # From examples (just in case some params aren't covered by the most relevant params)
        for nl, ep_name, api_params, conf, sim in examples_raw:
            if ep_name == endpoint_name and isinstance(api_params, (dict, list, str)):
                if isinstance(api_params, str):
                    try:
                        api_params = json.loads(api_params)
                    except Exception:
                        api_params = {}
                if isinstance(api_params, dict):
                    for k in api_params.keys():
                        allowed_param_names.add(k)

        # Nothing to map? Bail early.
        if not allowed_param_names:
            msg = f"No parameter definitions found for endpoint '{endpoint_name}'"
            st.warning(msg)
            return {}, msg

        # -----------------------------
        # 3) Exact / near-exact example shortcut
        # -----------------------------
        best_example = None
        best_sim = -1.0

        for nl, ep_name, api_params, conf, sim in examples_raw:
            if ep_name != endpoint_name:
                continue
            if sim > best_sim:
                best_sim = sim
                best_example = (nl, ep_name, api_params, conf, sim)

        # If the top example is extremely close to the user query, just reuse its params.
        # This fixes the “I type exactly like the example but mapping is different” problem.
        if best_example is not None and best_sim >= 0.95:
            nl, ep_name, api_params, conf, sim = best_example
            if isinstance(api_params, str):
                try:
                    api_params = json.loads(api_params)
                except Exception:
                    api_params = {}

            if isinstance(api_params, dict):
                # Strip unknown keys just in case, then replace date placeholders
                cleaned = {
                    k: v for k, v in api_params.items()
                    if k in allowed_param_names
                }
                cleaned = self._replace_date_placeholders(cleaned)
                explanation = (
                    f"Parameters copied from near-identical example "
                    f"(sim={best_sim:.2f}) for {endpoint_name}"
                )
                return cleaned, explanation

        # -----------------------------
        # 4) Build LLM context (params + examples + date placeholders)
        # -----------------------------
        # "Relevant" params give more focused descriptions for this query
        param_context = self._build_parameter_context(relevant_params_raw)
        example_context = self._build_example_context(examples_raw[:3])
        date_context = self._build_date_context()

        allowed_list_text = "\n".join(
            f"- {name}" for name in sorted(allowed_param_names)
        )

        prompt = f"""You are an API parameter mapper for the '{endpoint_name}' endpoint.

Your job is to produce a JSON object of API parameters that best answer the user's query.

CURRENT DATE INFORMATION:
{date_context}

USER QUERY:
"{user_query}"

ALLOWED PARAMETER NAMES (you MUST NOT invent new keys):
{allowed_list_text}

AVAILABLE PARAMETERS FOR {endpoint_name.upper()} ENDPOINT (definitions & examples):
{param_context}

SIMILAR QUERY EXAMPLES FOR {endpoint_name.upper()}:
{example_context}

TASK:
1. Decide which parameters are relevant to this user query.
2. Fill only those parameters using:
   - The parameter definitions above.
   - The patterns and values from similar examples.
3. If you are NOT sure about a parameter, leave it out (do NOT guess).
4. For date-like parameters, you may use placeholders such as {{today}}, {{yesterday}}, {{week_start}}, etc.
5. You must ONLY use parameter names from the allowed list above.
6. The JSON must be valid and must NOT contain comments, explanations, or extra text.

OUTPUT:
Return ONLY a JSON object. No markdown. No backticks. No comments. Example:

{{
  "some_param": "value",
  "another_param": "value"
}}
"""

        # -----------------------------
        # 5) Call LLM to propose parameter values
        # -----------------------------
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert API parameter mapper. "
                            "Output only valid JSON using the allowed parameter names."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                options={"temperature": 0.1, "num_ctx": 8192},
            )

            raw_text = response["message"]["content"].strip()
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            llm_params = json.loads(raw_text)
            if not isinstance(llm_params, dict):
                raise ValueError("LLM did not return a JSON object")

            # -----------------------------
            # 6) Whitelist keys and replace date placeholders
            # -----------------------------
            cleaned_params: Dict[str, Any] = {}
            for key, value in llm_params.items():
                if key in allowed_param_names:
                    cleaned_params[key] = value

            # Optionally blend in defaults from best example for missing keys
            if best_example is not None and best_sim >= 0.6:
                _, _, ex_params, _, sim_val = best_example
                if isinstance(ex_params, str):
                    try:
                        ex_params = json.loads(ex_params)
                    except Exception:
                        ex_params = {}
                if isinstance(ex_params, dict):
                    for k, v in ex_params.items():
                        # Only fill if:
                        # - key is allowed
                        # - LLM didn't already set it
                        # This is the “50% examples, 50% LLM” flavor.
                        if k in allowed_param_names and k not in cleaned_params:
                            cleaned_params[k] = v

            cleaned_params = self._replace_date_placeholders(cleaned_params)

            explanation = (
                f"Mapped using {len(relevant_params_raw)} parameter definitions and "
                f"{len(examples_raw)} examples for endpoint '{endpoint_name}'. "
                f"Best example similarity={best_sim:.2f}."
            )
            return cleaned_params, explanation

        except json.JSONDecodeError as e:
            st.warning(f"Failed to parse LLM parameter JSON: {e}")
            # Fallback: use best example if it was at least moderately similar
            if best_example is not None and best_sim >= 0.6:
                _, _, api_params, _, _ = best_example
                if isinstance(api_params, str):
                    try:
                        api_params = json.loads(api_params)
                    except Exception:
                        api_params = {}
                if isinstance(api_params, dict):
                    cleaned = {
                        k: v for k, v in api_params.items()
                        if k in allowed_param_names
                    }
                    cleaned = self._replace_date_placeholders(cleaned)
                    explanation = (
                        f"LLM JSON error; falling back to best example parameters "
                        f"(sim={best_sim:.2f}) for {endpoint_name}"
                    )
                    return cleaned, explanation
            return {}, f"JSON parsing error during parameter mapping: {e}"

        except Exception as e:
            st.warning(f"Error in parameter mapping: {e}")
            # Conservative fallback
            return {}, f"Mapping error: {e}"




    
    def _build_parameter_context(self, relevant_params: List[Tuple]) -> str:
        """Build parameter context for LLM prompt"""
        if not relevant_params:
            return "No parameters found."
        
        context_lines = []
        for endpoint, param_name, param_type, description, examples, similarity in relevant_params:
            examples_str = ", ".join(f'"{ex}"' for ex in examples[:3])
            context_lines.append(
                f"• {param_name} ({param_type})\n"
                f"  {description}\n"
                f"  Examples: {examples_str}"
            )
        
        return "\n\n".join(context_lines)
    
    def _build_example_context(self, relevant_examples: List[Tuple]) -> str:
        """Build example context for LLM prompt"""
        if not relevant_examples:
            return "No examples found."
        
        context_lines = []
        for natural_language, endpoint, api_params, confidence, similarity in relevant_examples:
            params_str = json.dumps(api_params, indent=2)
            context_lines.append(
                f"Query: \"{natural_language}\"\n"
                f"Parameters:\n{params_str}"
            )
        
        return "\n\n".join(context_lines[:3])  # Limit to top 3 examples
    
    def _build_date_context(self) -> str:
        """Build current date context"""
        today = datetime.now()
        
        # Calculate various date points
        dates = {
            "today": today.strftime("%Y-%m-%d"),
            "yesterday": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "tomorrow": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        
        # Week calculations
        week_start = today - timedelta(days=today.weekday())
        dates["week_start"] = week_start.strftime("%Y-%m-%d")
        dates["week_end"] = (week_start + timedelta(days=6)).strftime("%Y-%m-%d")
        
        last_week_start = week_start - timedelta(days=7)
        dates["last_week_start"] = last_week_start.strftime("%Y-%m-%d")
        dates["last_week_end"] = (last_week_start + timedelta(days=6)).strftime("%Y-%m-%d")
        
        # Month calculations
        dates["month_start"] = today.replace(day=1).strftime("%Y-%m-%d")
        
        context = f"Today is {dates['today']}\n"
        context += f"Available date placeholders:\n"
        for key, value in dates.items():
            context += f"  {{{key}}} = {value}\n"
        
        return context
    
    def _replace_date_placeholders(self, api_params: Dict[str, Any]) -> Dict[str, Any]:
        """Replace date placeholders with actual dates"""
        today = datetime.now()
        
        replacements = {
            "{today}": today.strftime("%Y-%m-%d"),
            "{yesterday}": (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            "{tomorrow}": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
        }
        
        # Week
        week_start = today - timedelta(days=today.weekday())
        replacements["{week_start}"] = week_start.strftime("%Y-%m-%d")
        replacements["{week_end}"] = (week_start + timedelta(days=6)).strftime("%Y-%m-%d")
        
        last_week_start = week_start - timedelta(days=7)
        replacements["{last_week_start}"] = last_week_start.strftime("%Y-%m-%d")
        replacements["{last_week_end}"] = (last_week_start + timedelta(days=6)).strftime("%Y-%m-%d")
        
        # Month
        replacements["{month_start}"] = today.replace(day=1).strftime("%Y-%m-%d")
        # Month end
        if today.month == 12:
            next_month = today.replace(year=today.year + 1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month + 1, day=1)
        month_end = next_month - timedelta(days=1)
        replacements["{month_end}"] = month_end.strftime("%Y-%m-%d")
        
        # Replace in params
        result = {}
        for key, value in api_params.items():
            if isinstance(value, str):
                for placeholder, actual_date in replacements.items():
                    value = value.replace(placeholder, actual_date)
            result[key] = value
        
        return result
    
    def _add_download_buttons(self, df, user_query: str):
        """Add CSV and Excel download buttons for results"""
        import pandas as pd
        import io
        
        if df is None or len(df) == 0:
            return
        
        st.write("### Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_csv = f"api_results_{timestamp}.csv"
            
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=filename_csv,
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Results', index=False)
                
                # Add a metadata sheet
                metadata_df = pd.DataFrame({
                    'Property': ['Query', 'Timestamp', 'Row Count', 'Agent', 'Endpoint'],
                    'Value': [
                        user_query,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        len(df),
                        self.name,
                        'Multi-Endpoint API'
                    ]
                })
                metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            excel_data = excel_buffer.getvalue()
            filename_excel = f"api_results_{timestamp}.xlsx"
            
            st.download_button(
                label="📊 Download Excel",
                data=excel_data,
                file_name=filename_excel,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    def _call_api(self, endpoint_name: str, api_params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
        try:
            from api_endpoints_config import get_endpoint_by_name
            cfg = get_endpoint_by_name(endpoint_name)
            if not cfg:
                return False, {}, f"Unknown endpoint: {endpoint_name}"
            endpoint_url = f"{self.api_base_url}{cfg['url']}"

            resp = requests.get(endpoint_url, params=api_params, timeout=30)
            if resp.status_code != 200:
                return False, {}, f"HTTP error {resp.status_code}"

            data = resp.json()
            # tolerate list-root responses
            if isinstance(data, list):
                data = {"success": True, "data": data}

            if data.get("success"):
                rows = data.get("data", []) or []
                return True, data, f"Found {len(rows)} results from {endpoint_name}"
            return False, {}, f"API returned error: {data.get('error', 'Unknown API error')}"
        except requests.exceptions.Timeout:
            return False, {}, "API request timed out"
        except requests.exceptions.RequestException as e:
            return False, {}, f"API request failed: {str(e)}"
        except Exception as e:
            return False, {}, f"Unexpected error: {str(e)}"

    
    def _should_use_api(self, user_query: str) -> Tuple[bool, str]:
        """
        Determine if query should use API or fall back to NL2SQL
        
        Returns:
            (should_use_api, reason)
        """
        query_lower = user_query.lower()
        
        # Keywords suggesting complex queries better for SQL
        complex_keywords = [
            'join', 'aggregate', 'sum', 'average', 'count', 'total',
            'group by', 'calculation', 'trend', 'compare', 'comparison',
            'analysis', 'report', 'breakdown', 'distribution'
        ]
        
        complex_score = sum(1 for keyword in complex_keywords if keyword in query_lower)
        
        if complex_score >= 2:
            return False, f"Query appears complex (score: {complex_score}), better for SQL"
        
        # Try to detect endpoint
        endpoint_name, reason, confidence = self._detect_endpoint(user_query)
        
        if endpoint_name and confidence >= 0.4:
            return True, f"Endpoint '{endpoint_name}' detected with confidence {confidence:.2f}"
        
        if endpoint_name and confidence >= 0.3:
            return True, f"Endpoint '{endpoint_name}' detected with low confidence {confidence:.2f}, attempting API"
        
        return False, "No suitable endpoint detected"
    
    def process_query(self, user_query: str, routing_info: Dict[str, Any],
                    session_state: Dict[str, Any], silent_mode: bool = False) -> Dict[str, Any]:
        """
        Process query using multi-endpoint API with NL2SQL fallback.
        Router signals are trusted first; local heuristics are only used when the router
        did not explicitly select the API agent.
        """
        if not silent_mode:
            st.info("🔌 Multi-Endpoint API Agent activated")

        routing_info = routing_info or {}

        # --- Trust router selection if present ---
        selected_agent_key = (routing_info.get("selected_agent") or routing_info.get("agent") or "").lower()
        tool_name_key     = (routing_info.get("tool_name") or "").lower()
        method_key        = (routing_info.get("method") or "").lower()
        api_first_flag    = bool(routing_info.get("api_first")) or routing_info.get("override") == "api_first"

        router_wants_api = any([
            selected_agent_key in ("api_endpoint", "multi_endpoint_api", "api"),
            tool_name_key     in ("api_endpoint", "multi_endpoint_api", "api"),
            api_first_flag,
            method_key == "api_priority",
        ])

        # Decide whether to try API before NL2SQL
        if router_wants_api:
            should_use_api = True
            if not silent_mode:
                st.success(
                    f"✅ Router selected API "
                    f"(method={method_key or 'n/a'}, selected_agent={selected_agent_key or 'n/a'})"
                )
        else:
            # Only consult local heuristic if router didn't explicitly choose API
            should_use_api, api_decision = self._should_use_api(user_query)
            if not silent_mode:
                st.write(f"**API Decision**: {api_decision}")

        # Early fallback ONLY if the router did NOT want API and local heuristic says no
        if (not should_use_api) and (not router_wants_api):
            if not silent_mode:
                st.info("↩️ Falling back to NL2SQL agent")
            from agent_implementations import NL2SQLAgent
            nl2sql_agent = NL2SQLAgent()
            return nl2sql_agent.process_query(
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state,
                silent_mode=True  # Force silent for fallback
            )

        # ---- Step 1: Detect endpoint ----
        if not silent_mode:
            with st.spinner("🎯 Detecting API endpoint..."):
                endpoint_name, detection_reason, confidence = self._detect_endpoint(user_query)
        else:
            endpoint_name, detection_reason, confidence = self._detect_endpoint(user_query)

        if not endpoint_name:
            if not silent_mode:
                st.warning("No suitable endpoint detected, falling back to NL2SQL")
            from agent_implementations import NL2SQLAgent
            nl2sql_agent = NL2SQLAgent()
            return nl2sql_agent.process_query(
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state,
                silent_mode=True  # Force silent for fallback
            )

        if not silent_mode:
            st.success(f"**Endpoint Detected**: `{endpoint_name}` (confidence: {confidence:.2f})")
            st.write(f"**Reason**: {detection_reason}")

        # ---- Step 2: Map parameters ----
        if not silent_mode:
            with st.spinner(f"🔍 Mapping parameters for {endpoint_name} endpoint..."):
                api_params, mapping_explanation = self._map_parameters_for_endpoint(
                    user_query, endpoint_name, session_state
                )
        else:
            api_params, mapping_explanation = self._map_parameters_for_endpoint(
                user_query, endpoint_name, session_state
            )

        if not api_params:
            if not silent_mode:
                st.warning("Parameter mapping failed, falling back to NL2SQL")
            from agent_implementations import NL2SQLAgent
            nl2sql_agent = NL2SQLAgent()
            return nl2sql_agent.process_query(
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state,
                silent_mode=True  # Force silent for fallback
            )

        if not silent_mode:
            st.write("**Mapped Parameters**:")
            st.json(api_params)
            st.caption(f"_{mapping_explanation}_")

        # ---- Step 3: Call API ----
        if not silent_mode:
            with st.spinner(f"📡 Calling {endpoint_name} API..."):
                api_success, api_data, api_message = self._call_api(endpoint_name, api_params)
        else:
            api_success, api_data, api_message = self._call_api(endpoint_name, api_params)

        if not api_success:
            if not silent_mode:
                st.warning(f"API call failed: {api_message}. Falling back to NL2SQL")
            from agent_implementations import NL2SQLAgent
            nl2sql_agent = NL2SQLAgent()
            return nl2sql_agent.process_query(
                user_query=user_query,
                routing_info=routing_info,
                session_state=session_state,
                silent_mode=True  # Force silent for fallback
            )

        # ---- Step 4: Present results ----
        results = api_data.get("data", []) or []
        columns = list(results[0].keys()) if results else []

        if not silent_mode:
            st.success(f"✅ {api_message}")
            if results:
                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                # CSV/XLSX downloads
                self._add_download_buttons(df, user_query)
            else:
                st.info("No results found")

        # ---- Step 5: Store in history (best-effort) ----
        try:
            session_id = session_state.get("session_id")
            query_store = session_state.get("query_store")
            if query_store and session_id:
                query_store.add_query(
                    session_id=session_id,
                    user_query=user_query,
                    sql_query=f"API[{endpoint_name}]: {json.dumps(api_params)}",
                    results=results,
                    response=api_message,
                    sql_error=None,
                    agent_used=self.name
                )
        except Exception as e:
            if not silent_mode:
                st.warning(f"Failed to store query: {e}")

        # ---- Final return payload ----
        return {
            "success": True,
            "results": results,
            "columns": columns,
            "endpoint": endpoint_name,
            "api_params": api_params,
            "row_count": len(results),
            "agent": self.name,      # "MultiEndpointAPI"
            "message": api_message,
            "method": "api",         # lowercase for compatibility
            "confidence": confidence
        }
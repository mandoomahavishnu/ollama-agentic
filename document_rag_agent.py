"""
Document RAG Agent - ENHANCED with Better Drag & Drop UI
"""
import streamlit as st
import ollama
import hashlib
import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import PyPDF2
import docx
import pandas as pd
from io import BytesIO

from agent_implementations import BaseAgent
from config import LLM_MODEL


class DocumentRAGAgent(BaseAgent):
    """
    RAG Agent with ENHANCED drag-and-drop interface
    """
    
    def __init__(self):
        super().__init__("Document RAG")
        self.supported_formats = {'.pdf', '.docx', '.txt', '.csv', '.xlsx'}
        self.chunk_size = 1000
        self.chunk_overlap = 200
    
    def process_query(self, user_query: str, routing_info: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process document RAG queries with follow-up context support"""
        
        routing_info = routing_info or {}
        
        # Handle cross-agent follow-ups
        follow_up_context = routing_info.get("follow_up_context", {})
        is_follow_up = follow_up_context.get("is_follow_up", False)
        
        if is_follow_up and follow_up_context.get("last_agent") != "document_rag":
            return self._handle_cross_agent_follow_up(user_query, follow_up_context, session_state)
        
        try:
            query_store = session_state['query_store']
            session_id = session_state['session_id']
            
            # Single upload detection check
            is_upload = self._is_upload_request(user_query)
            
            # Check if user wants to upload documents
            if is_upload:
                return self._handle_upload_request(user_query, session_state)
            
            # Check if user wants to list documents
            if self._is_list_request(user_query):
                return self._handle_list_request(session_state)
            
            # Handle document-based queries
            return self._handle_document_query(user_query, query_store, session_id, routing_info)
            
        except Exception as e:
            st.error(f"Document RAG error: {str(e)}")
            return {"success": False, "error": str(e), "agent": self.name}
    
    def _handle_cross_agent_follow_up(self, user_query: str, follow_up_context: Dict[str, Any], session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle follow-ups from other agents"""
        st.info(f"📄 Document RAG: Following up on {follow_up_context['last_agent']} query")
        
        last_query = follow_up_context.get("last_query", "")
        enhanced_query = f"Based on previous query '{last_query}', user asks: {user_query}"
        
        return self._handle_document_query(enhanced_query, session_state['query_store'], 
                                          session_state['session_id'], {})
    
    def _is_upload_request(self, query: str) -> bool:
        """Check if user wants to upload documents"""
        upload_keywords = [
            'upload document', 'upload', 'add document', 'load file', 
            'import document', 'new document', 'file upload',
            'upload file', 'add file'
        ]
        query_lower = query.lower().strip()
        return any(keyword in query_lower for keyword in upload_keywords)
    
    def _is_list_request(self, query: str) -> bool:
        """Check if user wants to list documents"""
        list_keywords = ['list documents', 'show documents', 'what documents', 'my documents', 'available documents']
        return any(keyword in query.lower() for keyword in list_keywords)
    
    def _handle_upload_request(self, user_query: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ ENHANCED: Better drag-and-drop interface with visual feedback
        """
        # Add custom CSS for better drag-and-drop visibility
        st.markdown("""
        <style>
        /* Make file uploader more visible and drag-drop friendly */
        div[data-testid="stFileUploader"] {
            border: 3px dashed #4CAF50 !important;
            border-radius: 10px !important;
            padding: 30px !important;
            background-color: #f0f8f0 !important;
            transition: all 0.3s ease !important;
        }
        
        div[data-testid="stFileUploader"]:hover {
            border-color: #45a049 !important;
            background-color: #e8f5e9 !important;
            transform: scale(1.01) !important;
        }
        
        /* Style the drag-drop area text */
        div[data-testid="stFileUploader"] > label {
            font-size: 18px !important;
            font-weight: bold !important;
            color: #2e7d32 !important;
        }
        
        /* Make the browse button more visible */
        div[data-testid="stFileUploader"] button {
            background-color: #4CAF50 !important;
            color: white !important;
            border: none !important;
            padding: 10px 20px !important;
            font-size: 16px !important;
            border-radius: 5px !important;
            cursor: pointer !important;
        }
        
        div[data-testid="stFileUploader"] button:hover {
            background-color: #45a049 !important;
        }
        
        /* Style for uploaded files */
        div[data-testid="stFileUploader"] section {
            background-color: white !important;
            border-radius: 5px !important;
            padding: 10px !important;
            margin-top: 10px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("## 📤 Document Upload Center")
        st.markdown("---")
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 Drag & Drop Your Documents Here!
            
            **Supported formats:** PDF, DOCX, TXT, CSV, XLSX
            
            **How to upload:**
            1. **Drag files** directly into the area below
            2. Or **click Browse** to select files
            3. Upload multiple files at once!
            """)
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            try:
                docs = session_state['query_store'].list_user_documents(session_state['session_id'])
                st.metric("Current Docs", len(docs) if docs else 0)
            except:
                st.metric("Current Docs", 0)
        
        st.markdown("---")
        
        # ✅ ENHANCED: More prominent file uploader with better label
        uploaded_files = st.file_uploader(
            "👇 Drag files here or click 'Browse files' below 👇",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx'],
            accept_multiple_files=True,
            key="persistent_rag_uploader",
            help="Drag and drop files or click to browse. You can select multiple files at once!",
            label_visibility="visible"
        )
        
        # Show helpful tips while waiting
        if not uploaded_files:
            st.info("💡 **Tip:** You can drag multiple files at once into the box above!")
            
            # Show example queries they can ask after upload
            with st.expander("💬 What you can ask after uploading"):
                st.markdown("""
                - "What does the document say about X?"
                - "Summarize the main points"
                - "Find information about Y"
                - "According to my documents, what is Z?"
                - "Compare the documents on topic A"
                """)
            
            return {
                "success": True,
                "message": "Document upload interface ready - waiting for files",
                "agent": self.name,
                "type": "upload_ready"
            }
        
        # ✅ Process files immediately when uploaded
        st.markdown("---")
        st.markdown("### 🔄 Processing Your Files...")
        
        query_store = session_state['query_store']
        session_id = session_state['session_id']
        processed_count = 0
        failed_files = []
        
        # Create progress tracking
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Process each file with detailed feedback
        for idx, uploaded_file in enumerate(uploaded_files):
            progress_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Show file info
            file_size_mb = uploaded_file.size / (1024 * 1024)
            with st.expander(f"📄 {uploaded_file.name} ({file_size_mb:.2f} MB)", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {uploaded_file.type}")
                with col2:
                    st.write(f"**Size:** {file_size_mb:.2f} MB")
                
                # Process the file
                try:
                    status_placeholder = st.empty()
                    status_placeholder.info("⏳ Processing...")
                    
                    success = self._process_and_store_document(uploaded_file, query_store, session_id)
                    
                    if success:
                        processed_count += 1
                        status_placeholder.success("✅ Successfully processed and stored!")
                    else:
                        failed_files.append(uploaded_file.name)
                        status_placeholder.error("❌ Failed to process")
                        
                except Exception as e:
                    failed_files.append(uploaded_file.name)
                    status_placeholder.error(f"❌ Error: {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Clear progress indicators
        progress_text.empty()
        progress_bar.empty()
        
        # Show final results
        st.markdown("---")
        st.markdown("### 📊 Upload Results")
        
        if processed_count > 0:
            st.balloons()
            st.success(f"🎉 **Successfully processed {processed_count} out of {len(uploaded_files)} document(s)!**")
            
            if failed_files:
                st.warning(f"⚠️ **Failed files:** {', '.join(failed_files)}")
            
            # Show what they can do now
            st.markdown("---")
            st.markdown("### 🎯 What's Next?")
            st.info("""
            **Your documents are ready!** You can now:
            - Ask questions about your documents
            - Request summaries
            - Search for specific information
            - Compare information across documents
            
            **Try asking:** "What are the main points in my documents?"
            """)
            
            return {
                "success": True,
                "message": f"Successfully processed {processed_count} documents",
                "agent": self.name,
                "type": "document_upload",
                "count": processed_count,
                "failed": len(failed_files)
            }
        else:
            st.error("❌ **No documents were successfully processed**")
            if failed_files:
                st.write("**Failed files:**")
                for file in failed_files:
                    st.write(f"  • {file}")
            
            return {
                "success": False,
                "message": "Failed to process any documents",
                "agent": self.name,
                "failed_files": failed_files
            }
    
    def _handle_list_request(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document listing requests"""
        try:
            query_store = session_state['query_store']
            session_id = session_state['session_id']
            
            documents = query_store.list_user_documents(session_id)
            
            if documents:
                st.markdown("## 📚 Your Document Library")
                st.markdown("---")
                
                for idx, doc in enumerate(documents, 1):
                    with st.expander(f"📄 {idx}. {doc['filename']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Type", doc['file_type'])
                        with col2:
                            st.metric("Chunks", doc['chunk_count'])
                        with col3:
                            st.write(f"**Uploaded:**")
                            st.write(doc['created_at'][:16])
                
                return {
                    "success": True,
                    "message": f"Found {len(documents)} documents",
                    "agent": self.name,
                    "type": "document_list",
                    "documents": documents
                }
            else:
                st.info("📄 **No documents uploaded yet.**")
                
                if st.button("📤 Upload Documents Now", key="list_upload_btn"):
                    st.session_state.user_input = "upload document"
                    st.rerun()
                
                return {
                    "success": True,
                    "message": "No documents found",
                    "agent": self.name,
                    "type": "no_documents"
                }
                
        except Exception as e:
            st.error(f"Error listing documents: {str(e)}")
            return {"success": False, "error": str(e), "agent": self.name}
    
    def _handle_document_query(self, user_query: str, query_store, session_id: str, routing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries against uploaded documents"""
        
        # Check if user has documents
        try:
            user_docs = query_store.list_user_documents(session_id)
            if not user_docs:
                st.warning("📄 **No documents found**. Please upload some documents first!")
                
                # Show upload button
                if st.button("📤 Upload Documents Now", key="inline_upload_btn"):
                    st.session_state.user_input = "upload document"
                    st.rerun()
                
                return {
                    "success": False,
                    "error": "No documents available",
                    "agent": self.name,
                    "suggestion": "upload_documents"
                }
        except Exception as e:
            pass
        
        st.info(f"🔍 **Searching your documents for:** {user_query}")
        
        try:
            # Search for relevant chunks
            relevant_chunks = query_store.search_document_chunks(
                user_query=user_query,
                session_id=session_id,
                top_k=5,
                min_similarity=0.3
            )
            
            if not relevant_chunks:
                st.warning("📄 **No relevant information found** in your documents for this query.")
                st.write("Try rephrasing your question or upload more documents.")
                
                return {
                    "success": True,
                    "message": "No relevant information found",
                    "agent": self.name,
                    "type": "no_results"
                }
            
            # Display source references
            st.write(f"📚 Found **{len(relevant_chunks)}** relevant sections")
            
            # Generate RAG response
            response = self._generate_rag_response(user_query, relevant_chunks, routing_info)
            
            # Display response
            st.markdown("### 💡 Answer:")
            st.write(response)
            
            # Display sources
            self._display_sources(relevant_chunks)
            
            # Store in query history
            query_store.add_query(
                session_id=session_id,
                user_query=user_query,
                sql_query=None,
                results=None,
                response=response,
                sql_error=None,
                agent_used=self.name
            )
            
            return {
                "success": True,
                "message": response,
                "agent": self.name,
                "type": "rag_response",
                "chunks_used": len(relevant_chunks)
            }
            
        except Exception as e:
            st.error(f"❌ Error during document search: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "agent": self.name
            }
    
    def _process_and_store_document(self, uploaded_file, query_store, session_id: str) -> bool:
        """Process uploaded file and store in database"""
        try:
            # Extract text based on file type
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.pdf':
                text = self._extract_pdf_text(uploaded_file)
            elif file_extension == '.docx':
                text = self._extract_docx_text(uploaded_file)
            elif file_extension == '.txt':
                text = uploaded_file.read().decode('utf-8')
            elif file_extension in ['.csv', '.xlsx']:
                text = self._extract_spreadsheet_text(uploaded_file, file_extension)
            else:
                return False
            
            if not text or len(text.strip()) < 10:
                st.warning(f"⚠️ {uploaded_file.name}: No text content found")
                return False
            
            # Generate document hash
            doc_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Store document and chunks
            query_store.store_document(
                session_id=session_id,
                filename=uploaded_file.name,
                file_type=file_extension,
                text_content=text,
                doc_hash=doc_hash,
                chunks=chunks
            )
            
            return True
            
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return False
    
    def _extract_pdf_text(self, uploaded_file) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"PDF extraction error: {str(e)}")
            return ""
    
    def _extract_docx_text(self, uploaded_file) -> str:
        """Extract text from DOCX"""
        try:
            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            st.error(f"DOCX extraction error: {str(e)}")
            return ""
    
    def _extract_spreadsheet_text(self, uploaded_file, file_extension: str) -> str:
        """Extract text from CSV/XLSX"""
        try:
            if file_extension == '.csv':
                df = pd.read_csv(BytesIO(uploaded_file.read()))
            else:  # xlsx
                df = pd.read_excel(BytesIO(uploaded_file.read()))
            
            # Convert dataframe to readable text
            text = f"Spreadsheet: {uploaded_file.name}\n\n"
            text += f"Columns: {', '.join(df.columns)}\n\n"
            text += df.to_string(index=False)
            return text
        except Exception as e:
            st.error(f"Spreadsheet extraction error: {str(e)}")
            return ""
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments"""
        
        # Simple sentence-based chunking
        sentences = text.split('. ')
        chunks = []
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            # Add sentence to current chunk
            potential_chunk = current_chunk + sentence + ". "
            
            if len(potential_chunk) > self.chunk_size and current_chunk:
                # Save current chunk and start new one
                chunks.append({
                    'chunk_index': chunk_index,
                    'text': current_chunk.strip(),
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk)
                })
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.chunk_overlap//10:]
                current_chunk = ' '.join(overlap_words) + ' ' + sentence + '. '
                chunk_index += 1
            else:
                current_chunk = potential_chunk
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'chunk_index': chunk_index,
                'text': current_chunk.strip(),
                'word_count': len(current_chunk.split()),
                'char_count': len(current_chunk)
            })
        
        return chunks
    
    def _generate_rag_response(self, user_query: str, relevant_chunks: List[Dict], routing_info: Dict[str, Any]) -> str:
        """Generate RAG response using retrieved chunks"""
        
        # Prepare context from chunks
        context_text = "\n\n".join([f"Source {i+1}: {chunk['text']}" for i, chunk in enumerate(relevant_chunks)])
        
        # Get user context for personalization
        user_profile = routing_info.get("user_profile", {})
        experience_level = user_profile.get("experience_level", "intermediate")
        role = user_profile.get("role", "user")
        
        # Build RAG prompt
        rag_prompt = f"""You are answering a question based on retrieved document content. Use the provided sources to give accurate, helpful answers.

USER PROFILE:
- Role: {role}
- Experience Level: {experience_level}

USER QUESTION: {user_query}

RETRIEVED CONTEXT:
{context_text}

INSTRUCTIONS:
1. Answer the question based primarily on the retrieved content
2. If the context doesn't fully answer the question, say so
3. Adapt your language to the user's experience level
4. Reference specific sources when appropriate
5. Be conversational and helpful
6. If you need to make inferences, make that clear

RESPONSE:"""
        
        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful document analysis assistant that answers questions based on retrieved content."},
                    {"role": "user", "content": rag_prompt}
                ],
                stream=False,
                options={"temperature": 0.3, "num_ctx": 4096}
            )
            
            return response["message"]["content"].strip()
            
        except Exception as e:
            return f"I found relevant information in your documents, but encountered an error generating the response: {str(e)}"
    
    def _display_sources(self, relevant_chunks: List[Dict]):
        """Display source information for retrieved chunks"""
        
        with st.expander("📚 **Sources Used**", expanded=False):
            for i, chunk in enumerate(relevant_chunks, 1):
                st.write(f"**Source {i}:** {chunk.get('filename', 'Unknown Document')}")
                st.write(f"*Similarity: {chunk.get('similarity', 0):.3f}*")
                
                # Show chunk preview
                chunk_text = chunk.get('text', '')
                preview = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
                st.write(f"*Preview:* {preview}")
                st.write("---")


def get_rag_routing_examples():
    """Get RAG routing examples for integration"""
    return [
        # Document upload queries
        ("upload a document", "document_rag", 0.95),
        ("add new document", "document_rag", 0.95),
        ("load file for analysis", "document_rag", 0.90),
        ("import document", "document_rag", 0.90),
        ("upload file", "document_rag", 0.95),
        ("upload documents", "document_rag", 0.95),
        
        # Document listing queries
        ("show my documents", "document_rag", 0.95),
        ("list uploaded files", "document_rag", 0.95),
        ("what documents do I have", "document_rag", 0.90),
        ("available documents", "document_rag", 0.85),
        ("my documents", "document_rag", 0.90),
        
        # Document-based queries
        ("what does the document say about", "document_rag", 0.95),
        ("search my documents for", "document_rag", 0.95),
        ("find information about", "document_rag", 0.85),
        ("according to my documents", "document_rag", 0.90),
        ("in the uploaded files", "document_rag", 0.90),
        ("based on the document", "document_rag", 0.85),
        ("what do my files say", "document_rag", 0.85),
        ("document analysis of", "document_rag", 0.80),
        ("summarize the document", "document_rag", 0.85),
        ("key points from document", "document_rag", 0.80),
        
        # Cross-domain queries
        ("policies about warehouse", "document_rag", 0.70),
        ("procedures for shipping", "document_rag", 0.70),
        ("guidelines for inventory", "document_rag", 0.70),
        ("documentation on process", "document_rag", 0.75),
    ]


def display_document_management_panel():
    """Display document management panel in sidebar with proper button handling"""
    
    st.subheader("📄 Document RAG")
    
    # Quick document stats
    try:
        query_store = st.session_state.get('query_store')
        session_id = st.session_state.get('session_id')
        
        if query_store and session_id:
            docs = query_store.list_user_documents(session_id)
            if docs:
                st.metric("Documents", len(docs))
                total_chunks = sum(doc.get('chunk_count', 0) for doc in docs)
                st.metric("Chunks", total_chunks)
            else:
                st.write("No documents uploaded")
        else:
            st.write("System not ready")
            
    except Exception as e:
        st.write("Document stats unavailable")
    
    # Quick actions
    st.write("**Quick Actions:**")
    
    # Upload button
    if st.button("📤 Upload Documents", use_container_width=True, key="sidebar_upload_btn"):
        st.session_state.user_input = "upload document"
        st.rerun()
    
    # List button
    if st.button("📋 List Documents", use_container_width=True, key="sidebar_list_btn"):
        st.session_state.user_input = "show my documents"
        st.rerun()
    
    # RAG tips
    with st.expander("💡 RAG Tips"):
        st.write("""
        **Supported formats:** PDF, DOCX, TXT, CSV, XLSX
        
        **Upload methods:**
        - **Drag & drop** files into the upload area
        - Click **Browse files** button
        - Upload **multiple files** at once
        
        **Example queries:**
        - "What does the manual say about safety?"
        - "Summarize the policy document"
        - "Find procedures for returns"
        - "According to my documents, what is..."
        """)
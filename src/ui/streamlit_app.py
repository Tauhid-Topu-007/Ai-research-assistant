# src/ui/streamlit_app.py
import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DocumentLoader
from src.data.processor import TextProcessor, Chunk
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.llm import LLM
from src.generation.context_builder import ContextBuilder

import yaml

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .main-header {
        color: #1f1f1f;
        font-family: 'Segoe UI', sans-serif;
    }
    .citation-box {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #2c8c8c;
        margin: 10px 0;
    }
    .source-box {
        background-color: #f0f0f0;
        padding: 8px;
        border-radius: 4px;
        font-size: 0.9em;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

class ResearchAssistant:
    def __init__(self):
        self.config = config
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all RAG components."""
        with st.spinner("Loading models..."):
            # Initialize embedding model
            self.embedding_model = EmbeddingModel(config)
            
            # Initialize retrievers
            self.dense_retriever = DenseRetriever(self.embedding_model, config)
            self.sparse_retriever = SparseRetriever(config)
            self.reranker = CrossEncoderReranker(config)
            
            # Try to load existing indices
            if self.load_indices():
                st.success("✅ Loaded existing indices")
            else:
                st.warning("⚠️ No indices found. Please upload documents first.")
            
            # Initialize LLM
            self.llm = LLM(config)
            self.context_builder = ContextBuilder(config)
            
            # Initialize hybrid retriever
            self.hybrid_retriever = HybridRetriever(
                self.dense_retriever,
                self.sparse_retriever,
                self.reranker,
                config
            )
            
    def load_indices(self) -> bool:
        """Load existing indices."""
        try:
            if Path('outputs/indices/dense').exists():
                self.dense_retriever.load('outputs/indices/dense')
            if Path('outputs/indices/sparse').exists():
                self.sparse_retriever.load('outputs/indices/sparse')
            return True
        except Exception as e:
            st.error(f"Error loading indices: {e}")
            return False
            
    def process_documents(self, uploaded_files):
        """Process uploaded documents."""
        loader = DocumentLoader(config)
        processor = TextProcessor(config)
        
        all_chunks = []
        
        for uploaded_file in uploaded_files:
            # Save temporarily
            temp_path = Path('data/raw') / uploaded_file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
                
            # Load and process
            doc = loader.load_document(str(temp_path))
            chunks = processor.process_document(doc)
            all_chunks.extend(chunks)
            
        return all_chunks
    
    def add_documents(self, chunks):
        """Add chunks to retrievers."""
        # Add to dense retriever
        self.dense_retriever.add_chunks(chunks)
        
        # Add to sparse retriever
        self.sparse_retriever.add_chunks(chunks)
        
        # Save indices
        self.dense_retriever.save('outputs/indices/dense')
        self.sparse_retriever.save('outputs/indices/sparse')
        
        st.success(f"✅ Added {len(chunks)} chunks to the index")

def main():
    st.title("📚 AI Research Assistant")
    st.markdown("### Hybrid + Agentic RAG System for Research Papers")
    
    # Initialize assistant
    if 'assistant' not in st.session_state:
        st.session_state.assistant = ResearchAssistant()
        st.session_state.messages = []
        
    assistant = st.session_state.assistant
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Research Papers",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                chunks = assistant.process_documents(uploaded_files)
                assistant.add_documents(chunks)
                
        # System stats
        st.divider()
        st.header("📊 System Stats")
        
        stats = assistant.hybrid_retriever.get_stats()
        st.metric("Total Chunks", stats['dense']['total_vectors'])
        st.metric("Embedding Model", stats['dense']['embedding_model'])
        
        st.divider()
        st.header("⚙️ Settings")
        
        use_reranker = st.checkbox("Use Reranker", True)
        retrieval_k = st.slider("Retrieval K", 3, 20, 5)
        
        st.divider()
        st.info("Built with LangChain, FAISS, BM25, and Cross-Encoder")
    
    # Main chat interface
    st.header("💬 Research Assistant")
    
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message:
                with st.expander("📚 Sources"):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>{source.get('document_title', 'Unknown')}</strong><br>
                            Page: {source.get('page_start', 'N/A')} | 
                            Score: {source.get('rerank_score', 0.0):.3f}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Input
    if prompt := st.chat_input("Ask a question about your research papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    # Retrieve
                    results = assistant.hybrid_retriever.search(
                        prompt,
                        k=retrieval_k,
                        use_reranker=use_reranker
                    )
                    
                    if not results:
                        st.warning("No relevant documents found. Please upload some research papers first.")
                        return
                        
                    # Build context
                    context = assistant.context_builder.build_context(results)
                    citations = assistant.context_builder.build_citations(results)
                    
                    # Generate answer
                    rag_prompt = assistant.llm.build_rag_prompt(prompt, context, citations)
                    response = assistant.llm.generate(
                        rag_prompt,
                        system_prompt="You are a research assistant. Answer questions based only on the provided context."
                    )
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display sources
                    with st.expander("📚 Sources"):
                        for source in results:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>{source.get('document_title', 'Unknown')}</strong><br>
                                Page: {source.get('page_start', 'N/A')} | 
                                Score: {source.get('rerank_score', source.get('fused_score', 0.0)):.3f}
                                <br><small>{source.get('text', '')[:200]}...</small>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Add to messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": results
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
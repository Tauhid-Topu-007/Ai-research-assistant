# src/generation/context_builder.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ContextBuilder:
    """Build context from retrieved chunks."""
    
    def __init__(self, config: dict):
        self.config = config
        self.max_tokens = config['llm'].get('max_context_tokens', 4000)
        
    def build_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source_info = []
            if chunk.get('document_title'):
                source_info.append(f"Document: {chunk['document_title']}")
            if chunk.get('page_start'):
                source_info.append(f"Page: {chunk['page_start']}")
            if chunk.get('section'):
                source_info.append(f"Section: {chunk['section']}")
                
            source_str = f"[{i}] " + " | ".join(source_info) if source_info else f"[{i}]"
            
            context_parts.append(f"{source_str}\n{chunk['text']}\n")
            
        return "\n".join(context_parts)
    
    def build_citations(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build citations from retrieved chunks."""
        citations = []
        
        for chunk in retrieved_chunks:
            citation = {
                'document_id': chunk.get('document_id', 'unknown'),
                'document_title': chunk.get('document_title', 'unknown'),
                'page_start': chunk.get('page_start'),
                'page_end': chunk.get('page_end'),
                'section': chunk.get('section'),
                'score': chunk.get('rerank_score', chunk.get('fused_score', chunk.get('score', 0)))
            }
            citations.append(citation)
            
        return citations
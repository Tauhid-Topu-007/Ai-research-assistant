# src/retrieval/hybrid_retriever.py
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrieval."""
    
    def __init__(self,
                 dense_retriever: DenseRetriever,
                 sparse_retriever: SparseRetriever,
                 reranker: Optional[CrossEncoderReranker] = None,
                 config: Optional[dict] = None):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.reranker = reranker
        self.config = config or {}
        
        self.k = self.config.get('hybrid_k', 20)
        self.rrf_k = self.config.get('rrf_k', 60)
        
    def reciprocal_rank_fusion(self, 
                               dense_results: List[Dict[str, Any]],
                               sparse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine results using Reciprocal Rank Fusion."""
        scores = defaultdict(float)
        result_maps = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            chunk_id = result['chunk_id']
            scores[chunk_id] += 1 / (self.rrf_k + rank)
            result_maps[chunk_id] = result
            
        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            chunk_id = result['chunk_id']
            scores[chunk_id] += 1 / (self.rrf_k + rank)
            if chunk_id not in result_maps:
                result_maps[chunk_id] = result
                
        # Sort by fused score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for chunk_id, score in sorted_chunks:
            result = result_maps[chunk_id].copy()
            result['fused_score'] = score
            fused_results.append(result)
            
        return fused_results
    
    def search(self, 
               query: str, 
               k: int = 10,
               use_reranker: bool = True) -> List[Dict[str, Any]]:
        """Perform hybrid search."""
        # Get results from both retrievers
        dense_results = self.dense_retriever.search(query, k=self.k)
        sparse_results = self.sparse_retriever.search(query, k=self.k)
        
        # Fuse results
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Apply reranker if available
        if self.reranker and use_reranker:
            fused_results = self.reranker.rerank(query, fused_results, top_k=k)
        else:
            fused_results = fused_results[:k]
            
        # Add rank
        for rank, result in enumerate(fused_results, start=1):
            result['rank'] = rank
            
        return fused_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        return {
            'dense': self.dense_retriever.get_stats(),
            'sparse': self.sparse_retriever.get_stats(),
            'rrf_k': self.rrf_k,
            'hybrid_k': self.k
        }
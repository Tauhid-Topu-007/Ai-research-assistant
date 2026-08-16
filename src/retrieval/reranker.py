# src/retrieval/reranker.py
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Cross-encoder reranker for improving retrieval results."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['reranker']['model_name']
        self.top_k = config['reranker']['top_k']
        self.device = self._get_device()
        
        self.model = CrossEncoder(
            self.model_name,
            device=self.device
        )
        logger.info(f"Reranker loaded: {self.model_name}")
        
    def _get_device(self) -> str:
        """Determine device for inference."""
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def rerank(self, 
               query: str, 
               candidates: List[Dict[str, Any]],
               top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []
            
        if top_k is None:
            top_k = self.top_k
            
        # Prepare query-candidate pairs
        pairs = [[query, candidate['text']] for candidate in candidates]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Sort by score
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        reranked = []
        for i, (result, score) in enumerate(scored_results[:top_k], start=1):
            result = result.copy()
            result['rerank_score'] = float(score)
            result['rank'] = i
            reranked.append(result)
            
        return reranked
    
    def __call__(self, query: str, candidates: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Callable interface."""
        return self.rerank(query, candidates, **kwargs)
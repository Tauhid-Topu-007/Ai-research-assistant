# src/evaluation/metrics.py
import logging
import numpy as np
from typing import List, Set, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RetrievalMetrics:
    """Evaluation metrics for retrieval systems."""
    
    @staticmethod
    def recall_at_k(retrieved_ids: List[str], 
                    relevant_ids: List[str], 
                    k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_ids:
            return 0.0
            
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        if not relevant_set:
            return 0.0
            
        return len(retrieved_set.intersection(relevant_set)) / len(relevant_set)
    
    @staticmethod
    def precision_at_k(retrieved_ids: List[str],
                       relevant_ids: List[str],
                       k: int) -> float:
        """Calculate Precision@K."""
        if k == 0 or not retrieved_ids:
            return 0.0
            
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        if not retrieved_set:
            return 0.0
            
        return len(retrieved_set.intersection(relevant_set)) / k
    
    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str],
                        relevant_ids: List[str]) -> float:
        """Calculate Reciprocal Rank."""
        relevant_set = set(relevant_ids)
        
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / i
                
        return 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_lists: List[List[str]],
                             relevant_lists: List[List[str]]) -> float:
        """Calculate Mean Reciprocal Rank."""
        if not retrieved_lists:
            return 0.0
            
        rr_sum = 0.0
        for retrieved, relevant in zip(retrieved_lists, relevant_lists):
            rr_sum += RetrievalMetrics.reciprocal_rank(retrieved, relevant)
            
        return rr_sum / len(retrieved_lists)
    
    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str],
                  relevant_ids: List[str],
                  k: int) -> float:
        """Calculate NDCG@K."""
        if not relevant_ids:
            return 0.0
            
        # Convert to relevance scores (1 if relevant, 0 otherwise)
        relevant_set = set(relevant_ids)
        relevance = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids[:k]]
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance, start=1):
            dcg += rel / np.log2(i + 1)
            
        # Calculate IDCG (ideal DCG)
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance, start=1):
            idcg += rel / np.log2(i + 1)
            
        if idcg == 0:
            return 0.0
            
        return dcg / idcg
    
    @staticmethod
    def average_precision(retrieved_ids: List[str],
                          relevant_ids: List[str]) -> float:
        """Calculate Average Precision."""
        if not relevant_ids:
            return 0.0
            
        relevant_set = set(relevant_ids)
        score = 0.0
        num_hits = 0.0
        
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                num_hits += 1
                score += num_hits / i
                
        if num_hits == 0:
            return 0.0
            
        return score / len(relevant_ids)
    
    @staticmethod
    def compute_all_metrics(retrieved_ids: List[str],
                           relevant_ids: List[str],
                           k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Compute all metrics for a single query."""
        metrics = {}
        
        # Recall@K
        for k in k_values:
            metrics[f'recall@{k}'] = RetrievalMetrics.recall_at_k(
                retrieved_ids, relevant_ids, k
            )
            
        # Precision@K
        for k in k_values:
            metrics[f'precision@{k}'] = RetrievalMetrics.precision_at_k(
                retrieved_ids, relevant_ids, k
            )
            
        # MRR
        metrics['mrr'] = RetrievalMetrics.reciprocal_rank(retrieved_ids, relevant_ids)
        
        # NDCG
        for k in k_values:
            metrics[f'ndcg@{k}'] = RetrievalMetrics.ndcg_at_k(
                retrieved_ids, relevant_ids, k
            )
            
        # MAP
        metrics['map'] = RetrievalMetrics.average_precision(retrieved_ids, relevant_ids)
        
        return metrics
    
    @staticmethod
    def evaluate_batch(predictions: List[List[str]],
                       ground_truths: List[List[str]],
                       k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Evaluate a batch of queries."""
        metrics = {}
        
        # Initialize accumulators
        for k in k_values:
            metrics[f'recall@{k}'] = 0.0
            metrics[f'precision@{k}'] = 0.0
            metrics[f'ndcg@{k}'] = 0.0
            
        metrics['mrr'] = 0.0
        metrics['map'] = 0.0
        
        n = len(predictions)
        
        for pred, gt in zip(predictions, ground_truths):
            single_metrics = RetrievalMetrics.compute_all_metrics(pred, gt, k_values)
            
            for key in single_metrics:
                if key in metrics:
                    metrics[key] += single_metrics[key]
                    
        # Average
        for key in metrics:
            metrics[key] /= n
            
        return metrics
# src/evaluation/evaluator.py
import logging
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd

from src.evaluation.metrics import RetrievalMetrics

logger = logging.getLogger(__name__)

class Evaluator:
    """Main evaluation class for retrieval systems."""
    
    def __init__(self, config: dict):
        self.config = config
        self.k_values = config['evaluation']['k_values']
        self.dev_queries_path = Path(config['data']['evaluation_path']) / 'dev' / 'retrieval_dev.json'
        self.test_queries_path = Path(config['data']['evaluation_path']) / 'test' / 'retrieval_test.json'
        self.results_path = Path('outputs/reports/')
        self.results_path.mkdir(parents=True, exist_ok=True)
        
    def load_queries(self, path: Path) -> List[Dict[str, Any]]:
        """Load evaluation queries."""
        with open(path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        return queries
    
    def evaluate_retriever(self,
                           retriever,
                           queries: List[Dict[str, Any]],
                           name: str = "retriever") -> Dict[str, float]:
        """Evaluate a retriever on a set of queries."""
        predictions = []
        ground_truths = []
        latencies = []
        
        for query_data in queries:
            question = query_data['question']
            relevant_ids = query_data['relevant_chunk_ids']
            
            # Measure latency
            start_time = time.perf_counter()
            results = retriever.search(question, k=max(self.k_values))
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            retrieved_ids = [r['chunk_id'] for r in results]
            
            predictions.append(retrieved_ids)
            ground_truths.append(relevant_ids)
            
        # Compute metrics
        metrics = RetrievalMetrics.evaluate_batch(predictions, ground_truths, self.k_values)
        metrics['avg_latency_ms'] = sum(latencies) / len(latencies)
        metrics['p95_latency_ms'] = sorted(latencies)[int(len(latencies) * 0.95)]
        
        logger.info(f"Evaluation results for {name}:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
            
        return metrics
    
    def evaluate_multiple_retrievers(self,
                                     retrievers: Dict[str, Any],
                                     split: str = "dev") -> pd.DataFrame:
        """Evaluate multiple retrievers and compare results."""
        query_path = self.dev_queries_path if split == "dev" else self.test_queries_path
        queries = self.load_queries(query_path)
        
        results = []
        
        for name, retriever in retrievers.items():
            metrics = self.evaluate_retriever(retriever, queries, name)
            metrics['method'] = name
            results.append(metrics)
            
        df = pd.DataFrame(results)
        
        # Save results
        output_path = self.results_path / f"evaluation_{split}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        return df
    
    def create_evaluation_dataset(self,
                                  chunks: List,
                                  output_path: Path,
                                  num_queries: int = 30) -> None:
        """Create an evaluation dataset from chunks."""
        # This is a placeholder - you'll need to manually create or
        # automatically generate queries and relevant chunks
        pass
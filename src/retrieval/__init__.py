# src/retrieval/__init__.py
from .embedding import EmbeddingModel
from .dense_retriever import DenseRetriever
from .sparse_retriever import SparseRetriever  # Now works without rank_bm25
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker

__all__ = [
    'EmbeddingModel',
    'DenseRetriever',
    'SparseRetriever',
    'HybridRetriever',
    'CrossEncoderReranker'
]
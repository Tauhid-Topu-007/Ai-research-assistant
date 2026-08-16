# src/retrieval/sparse_retriever.py
import logging
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import numpy as np
from collections import Counter
import math

logger = logging.getLogger(__name__)

class BM25:
    """
    Simple BM25 implementation from scratch.
    No external dependencies needed.
    """
    
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        """
        Initialize BM25.
        
        Args:
            corpus: List of tokenized documents
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
            epsilon: Smoothing parameter for IDF
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        
        # Calculate document frequency
        self.df = Counter()
        for doc in corpus:
            # Use set to count each term once per document
            self.df.update(set(doc))
        
        # Calculate IDF
        self.idf = {}
        N = len(corpus)
        for term, freq in self.df.items():
            # BM25 IDF formula
            idf = math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            self.idf[term] = idf
    
    def get_scores(self, query):
        """
        Get BM25 scores for a query.
        
        Args:
            query: Tokenized query list
            
        Returns:
            List of scores for each document
        """
        scores = []
        for doc in self.corpus:
            score = 0.0
            doc_len = len(doc)
            
            for term in query:
                if term not in self.idf:
                    continue
                    
                term_freq = doc.count(term)
                if term_freq == 0:
                    continue
                    
                idf = self.idf[term]
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * (numerator / denominator)
            
            scores.append(score)
        return scores

class SparseRetriever:
    """BM25-based sparse retriever."""
    
    def __init__(self, config: dict):
        self.config = config
        self.bm25 = None
        self.chunks = []
        self.tokenized_chunks = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Split by whitespace
        tokens = text.split()
        
        # Remove short tokens (1 character)
        tokens = [t for t in tokens if len(t) > 1]
        
        # Remove stopwords (optional - basic list)
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'am', 'do', 'does', 'did', 'done', 'has', 'have', 'had', 'having',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'this', 'that', 'these', 'those', 'which', 'what', 'who', 'whom',
            'with', 'without', 'by', 'for', 'from', 'into', 'of', 'on', 'to',
            'than', 'so', 'too', 'very', 'just', 'but', 'yet', 'nor', 'or',
            'at', 'in', 'via', 'per'
        }
        tokens = [t for t in tokens if t not in stopwords]
        
        return tokens
    
    def add_chunks(self, chunks: List['Chunk']):
        """Add chunks to BM25 index."""
        if not chunks:
            logger.warning("No chunks to add")
            return
            
        logger.info(f"Adding {len(chunks)} chunks to BM25...")
        
        # Tokenize all chunks
        self.tokenized_chunks = [
            self._tokenize(chunk.text) for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25(self.tokenized_chunks)
        
        # Store chunks
        self.chunks = chunks
        
        logger.info(f"Added {len(chunks)} chunks to BM25 index")
        
    def search(self, 
               query: str, 
               k: int = 10,
               return_scores: bool = True) -> List[Dict[str, Any]]:
        """Search using BM25."""
        if not self.bm25:
            logger.warning("BM25 index is empty")
            return []
            
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            logger.warning("Empty query tokens")
            return []
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Prepare results
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == 0:
                continue
                
            chunk = self.chunks[idx]
            
            result = {
                'rank': rank + 1,
                'score': float(scores[idx]),
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'document_title': chunk.document_title,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'section': chunk.section,
                'text': chunk.text
            }
            
            if not return_scores:
                del result['score']
                
            results.append(result)
            
        return results
    
    def save(self, path: str):
        """Save BM25 index."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 data
        if self.bm25:
            data_path = path / "bm25_data.pkl"
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'tokenized_chunks': self.tokenized_chunks,
                    'chunks': self.chunks
                }, f)
                
            logger.info(f"Saved BM25 index to {path}")
            
    def load(self, path: str):
        """Load BM25 index."""
        path = Path(path)
        
        data_path = path / "bm25_data.pkl"
        if not data_path.exists():
            logger.warning(f"BM25 data not found at {data_path}")
            return
            
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.tokenized_chunks = data['tokenized_chunks']
            self.chunks = data['chunks']
            
        logger.info(f"Loaded BM25 index from {path} with {len(self.chunks)} documents")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_documents': len(self.chunks) if self.chunks else 0,
            'total_terms': len(self.bm25.idf) if self.bm25 else 0,
            'avg_doc_length': self.bm25.avgdl if self.bm25 else 0
        }
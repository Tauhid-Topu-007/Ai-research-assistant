# src/retrieval/dense_retriever.py
import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from tqdm import tqdm

from src.retrieval.embedding import EmbeddingModel
from src.data.processor import Chunk

logger = logging.getLogger(__name__)

class DenseRetriever:
    """FAISS-based dense vector retriever."""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 config: dict):
        self.embedding_model = embedding_model
        self.config = config
        self.dimension = embedding_model.get_dimension()
        self.index = None
        self.chunks = []
        self.metadata_mapping = []
        
        # Build index
        self._create_index()
        
    def _create_index(self):
        """Create FAISS index."""
        # Use inner product for cosine similarity with normalized vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        
    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks to the index."""
        if not chunks:
            logger.warning("No chunks to add")
            return
            
        logger.info(f"Adding {len(chunks)} chunks to FAISS index...")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress=True
        )
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.metadata_mapping.append({
                'chunk_id': chunk.chunk_id,
                'document_id': chunk.document_id,
                'document_title': chunk.document_title,
                'page_start': chunk.page_start,
                'page_end': chunk.page_end,
                'section': chunk.section,
                'text': chunk.text
            })
            
        logger.info(f"Added {len(chunks)} chunks to index. Total: {self.index.ntotal}")
        
    def search(self, 
               query: str, 
               k: int = 10,
               return_scores: bool = True) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
            
        # Encode query
        query_embedding = self.embedding_model.encode_single(query)
        
        # Reshape for FAISS
        query_vector = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_vector, k)
        
        # Prepare results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                break
                
            metadata = self.metadata_mapping[idx]
            
            result = {
                'rank': i + 1,
                'score': float(score),
                'chunk_id': metadata['chunk_id'],
                'document_id': metadata['document_id'],
                'document_title': metadata['document_title'],
                'page_start': metadata['page_start'],
                'page_end': metadata['page_end'],
                'section': metadata['section'],
                'text': metadata['text']
            }
            
            if not return_scores:
                del result['score']
                
            results.append(result)
            
        return results
    
    def save(self, path: str):
        """Save index and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss_path = path / "index.faiss"
        faiss.write_index(self.index, str(faiss_path))
        
        # Save metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata_mapping': self.metadata_mapping
            }, f)
            
        # Save config
        config_path = path / "config.pkl"
        with open(config_path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'embedding_model': self.embedding_model.model_name
            }, f)
            
        logger.info(f"Saved index to {path}")
        
    def load(self, path: str):
        """Load index and metadata."""
        path = Path(path)
        
        # Load FAISS index
        faiss_path = path / "index.faiss"
        self.index = faiss.read_index(str(faiss_path))
        
        # Load metadata
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata_mapping = data['metadata_mapping']
            
        logger.info(f"Loaded index from {path} with {self.index.ntotal} vectors")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'embedding_model': self.embedding_model.model_name
        }
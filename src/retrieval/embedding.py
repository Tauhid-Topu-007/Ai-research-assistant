# src/retrieval/embedding.py
import logging
import numpy as np
from typing import List, Optional, Union
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """Handles text embeddings using sentence-transformers."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['embedding']['model_name']
        self.normalize = config['embedding']['normalize']
        self.dimension = config['embedding']['dimension']
        self.batch_size = config['embedding']['batch_size']
        self.device = self._get_device()
        
        self.model = self._load_model()
        logger.info(f"Embedding model loaded: {self.model_name}")
        logger.info(f"Dimension: {self.dimension}, Device: {self.device}")
        
    def _get_device(self) -> str:
        """Determine device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def _load_model(self) -> SentenceTransformer:
        """Load the embedding model."""
        model = SentenceTransformer(self.model_name, device=self.device)
        return model
    
    def encode(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text."""
        return self.encode([text])[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Callable interface."""
        return self.encode(texts)
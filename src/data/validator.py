# src/data/validator.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate data quality."""
    
    @staticmethod
    def validate_chunk(chunk: Dict[str, Any]) -> bool:
        """Validate a single chunk."""
        required_fields = ['chunk_id', 'text', 'document_id']
        
        for field in required_fields:
            if field not in chunk:
                logger.warning(f"Missing field: {field}")
                return False
                
        if not chunk['text'].strip():
            logger.warning("Empty text in chunk")
            return False
            
        return True
    
    @staticmethod
    def validate_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate all chunks."""
        valid_chunks = []
        
        for chunk in chunks:
            if DataValidator.validate_chunk(chunk):
                valid_chunks.append(chunk)
                
        return valid_chunks
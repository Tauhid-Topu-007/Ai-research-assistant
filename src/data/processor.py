# src/data/processor.py
import re
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    text: str
    document_id: str
    document_title: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class TextProcessor:
    """Process and chunk text documents."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config['chunking']['chunk_size']
        self.overlap = config['chunking']['overlap']
        self.min_chunk_size = config['chunking']['min_chunk_size']
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
        
        # Normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple rules."""
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_by_sentences(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text by sentences with sliding window."""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_idx = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_len
            else:
                # Save current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunk_id = f"{metadata.get('file_name', 'doc')}_{chunk_idx:04d}"
                        chunks.append(Chunk(
                            chunk_id=chunk_id,
                            text=chunk_text,
                            document_id=metadata.get('file_name', 'unknown'),
                            document_title=metadata.get('title', metadata.get('file_name', 'unknown')),
                            chunk_index=chunk_idx
                        ))
                        chunk_idx += 1
                
                # Start new chunk with overlap
                overlap_count = 0
                overlap_chunk = []
                overlap_len = 0
                
                # Add overlapping sentences
                for prev_sentence in reversed(current_chunk):
                    if overlap_len + len(prev_sentence) <= self.overlap:
                        overlap_chunk.insert(0, prev_sentence)
                        overlap_len += len(prev_sentence)
                        overlap_count += 1
                    else:
                        break
                
                current_chunk = overlap_chunk + [sentence]
                current_length = overlap_len + sentence_len
        
        # Final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = f"{metadata.get('file_name', 'doc')}_{chunk_idx:04d}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    document_id=metadata.get('file_name', 'unknown'),
                    document_title=metadata.get('title', metadata.get('file_name', 'unknown')),
                    chunk_index=chunk_idx
                ))
                
        return chunks
    
    def chunk_by_paragraph(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Chunk text by paragraphs."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        chunk_idx = 0
        
        for paragraph in paragraphs:
            if len(paragraph) >= self.min_chunk_size:
                # If paragraph is too long, split further
                if len(paragraph) > self.chunk_size:
                    # Split by sentences within paragraph
                    sub_chunks = self.chunk_by_sentences(paragraph, metadata)
                    for sub_chunk in sub_chunks:
                        sub_chunk.chunk_index = chunk_idx
                        sub_chunk.chunk_id = f"{metadata.get('file_name', 'doc')}_{chunk_idx:04d}"
                        chunks.append(sub_chunk)
                        chunk_idx += 1
                else:
                    chunk_id = f"{metadata.get('file_name', 'doc')}_{chunk_idx:04d}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=paragraph,
                        document_id=metadata.get('file_name', 'unknown'),
                        document_title=metadata.get('title', metadata.get('file_name', 'unknown')),
                        chunk_index=chunk_idx
                    ))
                    chunk_idx += 1
                    
        return chunks
    
    def process_document(self, document: Dict[str, Any]) -> List[Chunk]:
        """Process a single document into chunks."""
        text = document.get('text', '')
        
        if not text:
            return []
        
        text = self.clean_text(text)
        metadata = {
            'file_name': document.get('file_name', 'unknown'),
            'title': document.get('title', document.get('file_name', 'unknown'))
        }
        
        # Try paragraph-based chunking first
        chunks = self.chunk_by_paragraph(text, metadata)
        
        # If no chunks, try sentence-based
        if not chunks:
            chunks = self.chunk_by_sentences(text, metadata)
            
        # Add additional metadata
        for chunk in chunks:
            chunk.document_id = document.get('file_name', 'unknown')
            chunk.document_title = document.get('title', chunk.document_title)
            
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Chunk]:
        """Process multiple documents into chunks."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.process_document(doc)
            all_chunks.extend(chunks)
            logger.info(f"Processed {doc.get('file_name', 'unknown')}: {len(chunks)} chunks")
            
        return all_chunks
    
    def save_chunks(self, chunks: List[Chunk], path: str = "data/processed/chunks.json"):
        """Save chunks to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        chunk_dicts = [chunk.to_dict() for chunk in chunks]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(chunk_dicts, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved {len(chunks)} chunks to {path}")
        return path
    
    def load_chunks(self, path: str) -> List[Chunk]:
        """Load chunks from JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            chunk_dicts = json.load(f)
            
        chunks = [Chunk(**data) for data in chunk_dicts]
        logger.info(f"Loaded {len(chunks)} chunks from {path}")
        return chunks
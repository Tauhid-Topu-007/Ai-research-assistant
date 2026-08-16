# src/data/loader.py
import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

import pdfplumber
import pymupdf  # Use pymupdf instead of fitz
from bs4 import BeautifulSoup

# Try to import chardet, fallback to built-in if not available
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logging.warning("chardet not installed. Using fallback encoding detection.")

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Load and extract text from various document formats."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.raw_path = Path(config['data']['raw_path'])
        self.processed_path = Path(config['data']['processed_path'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def load_pdf_pymupdf(self, file_path: str) -> str:
        """Load PDF using PyMuPDF (fastest)."""
        text = ""
        try:
            doc = pymupdf.open(file_path)
            for page in doc:
                page_text = page.get_text()
                if page_text:
                    text += page_text
            doc.close()
        except Exception as e:
            logger.error(f"PyMuPDF error for {file_path}: {e}")
        return text
    
    def load_pdf_pdfplumber(self, file_path: str) -> str:
        """Load PDF using pdfplumber (better for tables)."""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
        except Exception as e:
            logger.error(f"pdfplumber error for {file_path}: {e}")
        return text
    
    def detect_encoding(self, raw_data: bytes) -> str:
        """Detect encoding of text data."""
        if HAS_CHARDET:
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8')
        else:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    raw_data.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue
            return 'utf-8'
    
    def load_text(self, file_path: str) -> str:
        """Load text file with encoding detection."""
        try:
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = self.detect_encoding(raw_data)
                return raw_data.decode(encoding, errors='ignore')
        except Exception as e:
            logger.error(f"Text loading error for {file_path}: {e}")
            return ""
    
    def load_html(self, file_path: str) -> str:
        """Load and parse HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                soup = BeautifulSoup(file, 'lxml')
                return soup.get_text()
        except Exception as e:
            logger.error(f"HTML loading error for {file_path}: {e}")
            return ""
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load document and extract text with metadata."""
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_type': file_ext,
            'file_size': file_path.stat().st_size if file_path.exists() else 0
        }
        
        text = ""
        
        if file_ext == '.pdf':
            # Try PyMuPDF first (fastest)
            text = self.load_pdf_pymupdf(str(file_path))
            # Fallback to pdfplumber if PyMuPDF fails
            if not text.strip():
                text = self.load_pdf_pdfplumber(str(file_path))
                
        elif file_ext in ['.txt', '.md']:
            text = self.load_text(str(file_path))
            
        elif file_ext in ['.html', '.htm']:
            text = self.load_html(str(file_path))
            
        else:
            logger.warning(f"Unsupported file type: {file_ext}")
            
        metadata['text'] = text
        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        
        return metadata
    
    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """Load all documents from a directory."""
        directory_path = Path(directory_path)
        documents = []
        
        supported_extensions = {'.pdf', '.txt', '.md', '.html', '.htm'}
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    doc = self.load_document(str(file_path))
                    if doc.get('text', '').strip():
                        documents.append(doc)
                        logger.info(f"Loaded: {file_path.name} ({len(doc['text'])} chars)")
                    else:
                        logger.warning(f"Empty text from: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
                    
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def save_processed(self, documents: List[Dict[str, Any]], name: str = "documents"):
        """Save processed documents to JSON."""
        output_path = self.processed_path / f"{name}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(documents)} documents to {output_path}")
        return output_path
    
    def load_processed(self, name: str = "documents") -> List[Dict[str, Any]]:
        """Load processed documents from JSON."""
        input_path = self.processed_path / f"{name}.json"
        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return []
            
        with open(input_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        logger.info(f"Loaded {len(documents)} documents from {input_path}")
        return documents
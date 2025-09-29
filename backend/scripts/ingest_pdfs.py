#!/usr/bin/env python3
"""
Script to ingest dermatology PDFs into the vector knowledge base
"""

import os
import glob
from pypdf import PdfReader
from app.services.rag_service import RAGService
from app.core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error reading {pdf_path}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_pdfs():
    """Ingest PDFs into vector database"""
    
    rag_service = RAGService()
    
    pdf_paths = glob.glob(os.path.join(settings.KNOWLEDGE_BASE_PATH, "*.pdf"))
    
    if not pdf_paths:
        logger.warning(f"No PDFs found in {settings.KNOWLEDGE_BASE_PATH}")
        return
    
    documents = []
    
    for pdf_path in pdf_paths:
        try:
            logger.info(f"Processing {os.path.basename(pdf_path)}")
            text = extract_text_from_pdf(pdf_path)
            
            if not text:
                continue
                
            chunks = chunk_text(text)
            
            for i, chunk in enumerate(chunks):
                documents.append({
                    'id': f"{os.path.basename(pdf_path)}_chunk_{i}",
                    'text': chunk,
                    'metadata': {
                        'source': os.path.basename(pdf_path),
                        'chunk': i,
                        'total_chunks': len(chunks)
                    }
                })
                
            logger.info(f"Processed {len(chunks)} chunks from {os.path.basename(pdf_path)}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
    
    if documents:
        logger.info(f"Successfully processed {len(documents)} documents")
    else:
        logger.warning("No documents to ingest")

if __name__ == "__main__":
    ingest_pdfs()
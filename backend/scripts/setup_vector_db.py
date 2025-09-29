#!/usr/bin/env python3
"""
Script to setup and initialize vector database
"""

from app.services.rag_service import RAGService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_vector_db():
    """Initialize vector database"""
    try:
        rag_service = RAGService()
        if rag_service.collection:
            logger.info("Vector database initialized successfully")
        else:
            logger.warning("Vector database initialization failed")
    except Exception as e:
        logger.error(f"Error setting up vector database: {str(e)}")

if __name__ == "__main__":
    setup_vector_db()
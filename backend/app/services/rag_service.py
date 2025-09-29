import os
from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.initialize_services()
    
    def initialize_services(self):
        """Initialize embedding model and vector database"""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu'
            )
            
            # Initialize ChromaDB
            self.chroma_client = chromadb.PersistentClient(
                path=settings.VECTOR_STORE_PATH
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="medical_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("RAG service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}")
            self.embedding_model = None
            self.collection = None
    
    def query_knowledge_base(self, query: str, max_results: int = 3) -> List[Dict]:
        """Query the medical knowledge base"""
        if not self.collection or not self.embedding_model:
            return [{
                "content": "Knowledge base service not available",
                "source": "system",
                "confidence": 0.0
            }]
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    "content": results['documents'][0][i],
                    "source": results['metadatas'][0][i].get('source', 'unknown'),
                    "confidence": float(1.0 - results['distances'][0][i]),
                    "id": results['ids'][0][i]
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return [{
                "content": f"Query error: {str(e)}",
                "source": "error",
                "confidence": 0.0
            }]

# Global RAG service instance
rag_service = RAGService()
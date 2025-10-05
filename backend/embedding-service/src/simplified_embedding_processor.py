"""
Simplified Embedding Processor for Embedding Service

This processor only generates embeddings without storing them in Qdrant.
The storage responsibility is moved to the ingestion service.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class SimplifiedEmbeddingProcessor:
    """
    Simplified embedding processor that only generates embeddings.
    No database connections or storage operations.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_size: int = 384,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize Simplified Embedding Processor.
        
        Args:
            model_name: Name of the sentence-transformer model
            vector_size: Dimension of embedding vectors
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        self.model_name = model_name
        self.vector_size = vector_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            actual_vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Vector dimension: {actual_vector_size}")
            
            # Update vector size if different from expected
            if actual_vector_size != vector_size:
                logger.warning(f"Model vector size ({actual_vector_size}) differs from expected ({vector_size})")
                self.vector_size = actual_vector_size
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model {model_name}: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            if not text.strip():
                logger.warning("Empty text provided for embedding")
                return [0.0] * self.vector_size
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text.strip()]
            if not valid_texts:
                logger.warning("No valid texts provided for batch embedding")
                return [[0.0] * self.vector_size] * len(texts)
            
            # Generate embeddings in batch
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False)
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Pad with zeros for empty texts
            result = []
            valid_idx = 0
            for text in texts:
                if text.strip():
                    result.append(embeddings[valid_idx])
                    valid_idx += 1
                else:
                    result.append([0.0] * self.vector_size)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        try:
            if not text.strip():
                return []
            
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + self.chunk_size
                
                # Find a good break point (sentence end, word boundary)
                if end < len(text):
                    # Look for sentence endings
                    for i in range(end, max(start + self.chunk_size // 2, end - 100), -1):
                        if text[i] in '.!?':
                            end = i + 1
                            break
                    else:
                        # Look for word boundaries
                        for i in range(end, max(start + self.chunk_size // 2, end - 50), -1):
                            if text[i] == ' ':
                                end = i
                                break
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "start": start,
                        "end": end,
                        "chunk_index": len(chunks)
                    })
                
                # Move start position with overlap
                start = end - self.chunk_overlap
                if start >= len(text):
                    break
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk text: {e}")
            raise
    
    def process_document_with_chunking(
        self,
        text: str,
        document_id: str,
        filename: str = "document",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a document by chunking and generating embeddings for each chunk.
        
        Args:
            text: Document text
            document_id: Document identifier
            filename: Document filename
            metadata: Additional metadata
            
        Returns:
            Processing result with chunk information and embeddings
        """
        try:
            start_time = time.time()
            
            # Chunk the text
            chunks = self.chunk_text(text)
            
            if not chunks:
                logger.warning(f"No chunks created for document {document_id}")
                return {
                    "document_id": document_id,
                    "filename": filename,
                    "total_chunks": 0,
                    "chunk_results": [],
                    "processing_time_ms": 0,
                    "method": "chunked_embedding"
                }
            
            # Generate embeddings for each chunk
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = self.generate_batch_embeddings(chunk_texts)
            
            # Prepare chunk results
            chunk_results = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_result = {
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "text": chunk["text"],
                    "embedding": embedding,
                    "chunk_index": chunk["chunk_index"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "metadata": metadata or {}
                }
                chunk_results.append(chunk_result)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Processed document {document_id}: {len(chunks)} chunks in {processing_time:.2f}ms")
            
            return {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "chunk_results": chunk_results,
                "processing_time_ms": processing_time,
                "method": "chunked_embedding"
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            raise
    
    def health_check(self) -> Dict[str, bool]:
        """
        Check processor health status.
        
        Returns:
            Health status dictionary
        """
        try:
            # Test embedding generation
            test_embedding = self.generate_embedding("test")
            model_loaded = len(test_embedding) == self.vector_size
            
            return {
                "embedding_model_loaded": model_loaded,
                "model_name": self.model_name,
                "vector_size": self.vector_size
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "embedding_model_loaded": False,
                "model_name": self.model_name,
                "vector_size": self.vector_size
            }

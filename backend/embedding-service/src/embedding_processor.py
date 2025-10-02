"""
Embedding Processor Module

Handles text embedding generation and vector storage using:
- sentence-transformers for embedding generation
- Qdrant for vector storage and similarity search
- Support for various text formats and batch processing
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import csv
import io

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Processes text content to generate embeddings and manage vector storage.
    
    Supports:
    - Text embedding generation using sentence-transformers
    - Vector storage and retrieval with Qdrant
    - Batch processing for efficiency
    - Similarity search and filtering
    """
    
    # Supported file extensions
    TEXT_EXTENSIONS = {'.txt', '.md', '.json', '.csv'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc'}
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "embeddings",
        vector_size: int = 384
    ):
        """
        Initialize Embedding Processor.
        
        Args:
            model_name: Name of the sentence-transformer model
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            qdrant_api_key: Qdrant API key (if authentication required)
            collection_name: Name of the Qdrant collection
            vector_size: Dimension of embedding vectors
        """
        self.model_name = model_name
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Initialize embedding model
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Model loaded successfully. Vector dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Failed to load embedding model {model_name}: {str(e)}")
        
        # Initialize Qdrant client
        try:
            self.qdrant_client = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                api_key=qdrant_api_key
            )
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise RuntimeError(f"Failed to connect to Qdrant: {str(e)}")
        
        # Ensure collection exists
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists with correct configuration."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise RuntimeError(f"Failed to setup Qdrant collection: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {str(e)}")
    
    def store_embedding(
        self,
        text: str,
        embedding: List[float],
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store an embedding in Qdrant.
        
        Args:
            text: Original text content
            embedding: Embedding vector
            document_id: Optional document identifier
            metadata: Optional metadata dictionary
            
        Returns:
            Point ID in Qdrant
        """
        try:
            import uuid
            
            # Generate point ID if not provided - Qdrant requires UUID or unsigned int
            if document_id:
                # Try to convert string to UUID if it looks like a UUID
                try:
                    if len(document_id) == 36 and document_id.count('-') == 4:
                        point_id = uuid.UUID(document_id)
                    else:
                        # Convert string to UUID by hashing
                        point_id = uuid.uuid5(uuid.NAMESPACE_DNS, document_id)
                except ValueError:
                    # If conversion fails, generate UUID from string
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, document_id)
            else:
                # Generate a new UUID
                point_id = uuid.uuid4()
            
            # Prepare metadata
            point_metadata = {
                "text": text,
                "text_length": len(text),
                "created_at": time.time(),
                "original_document_id": document_id  # Store original ID in metadata
            }
            if metadata:
                point_metadata.update(metadata)
            
            # Create point
            point = PointStruct(
                id=str(point_id),  # Convert UUID to string for Qdrant
                vector=embedding,
                payload=point_metadata
            )
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Stored embedding for document: {point_id}")
            return str(point_id)
            
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            raise RuntimeError(f"Failed to store embedding: {str(e)}")
    
    def search_similar(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_embedding(query)
            
            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": limit,
                "with_payload": True,
                "with_vectors": False
            }
            
            # Add score threshold if specified
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold
            
            # Add filters if specified
            if filter_conditions:
                search_params["query_filter"] = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            # Perform search
            search_results = self.qdrant_client.search(**search_params)
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "document_id": result.payload.get("original_document_id"),  # Include original document ID
                    "metadata": {k: v for k, v in result.payload.items() if k not in ["text", "original_document_id"]}
                })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {e}")
            raise RuntimeError(f"Similarity search failed: {str(e)}")
    
    def process_text_file(
        self,
        file_content: bytes,
        filename: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a text file and generate embeddings.
        
        Args:
            file_content: Raw bytes of the file
            filename: Original filename
            document_id: Optional document identifier
            metadata: Optional metadata dictionary
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            # Extract text based on file type
            file_ext = Path(filename).suffix.lower()
            
            if file_ext == '.txt':
                text = file_content.decode('utf-8')
            elif file_ext == '.md':
                text = file_content.decode('utf-8')
            elif file_ext == '.json':
                json_data = json.loads(file_content.decode('utf-8'))
                text = self._extract_text_from_json(json_data)
            elif file_ext == '.csv':
                text = self._extract_text_from_csv(file_content)
            else:
                raise ValueError(f"Unsupported text file format: {file_ext}")
            
            # Generate embedding
            embedding = self.generate_embedding(text)
            
            # Store in Qdrant
            point_id = self.store_embedding(
                text=text,
                embedding=embedding,
                document_id=document_id,
                metadata=metadata
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "text": text,
                "embedding": embedding,
                "embedding_dimension": len(embedding),
                "model_name": self.model_name,
                "processing_time_ms": processing_time,
                "text_length": len(text),
                "filename": filename,
                "point_id": point_id,
                "method": f"embed_text_{file_ext[1:]}"
            }
            
        except Exception as e:
            logger.error(f"Failed to process text file {filename}: {e}")
            raise RuntimeError(f"Text file processing failed: {str(e)}")
    
    def _extract_text_from_json(self, json_data: Any) -> str:
        """Extract text content from JSON data."""
        if isinstance(json_data, dict):
            # Try common text fields
            text_fields = ['text', 'content', 'body', 'description', 'summary']
            for field in text_fields:
                if field in json_data and isinstance(json_data[field], str):
                    return json_data[field]
            
            # Fallback: convert entire JSON to string
            return json.dumps(json_data, indent=2)
        elif isinstance(json_data, list):
            # Join list items as text
            return ' '.join(str(item) for item in json_data)
        else:
            return str(json_data)
    
    def _extract_text_from_csv(self, csv_content: bytes) -> str:
        """Extract text content from CSV data."""
        try:
            csv_text = csv_content.decode('utf-8')
            csv_reader = csv.reader(io.StringIO(csv_text))
            
            # Extract text from all cells
            text_parts = []
            for row in csv_reader:
                text_parts.extend([cell for cell in row if cell.strip()])
            
            return ' '.join(text_parts)
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            return csv_content.decode('utf-8', errors='ignore')
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Qdrant collection."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vector_size": collection_info.config.params.vectors.size,
                "points_count": collection_info.points_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise RuntimeError(f"Failed to get collection info: {str(e)}")
    
    def delete_embedding(self, point_id: str) -> bool:
        """Delete an embedding from Qdrant."""
        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=[point_id]
            )
            logger.info(f"Deleted embedding: {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete embedding {point_id}: {e}")
            return False
    
    def update_metadata(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing embedding in Qdrant.
        
        Args:
            document_id: Document identifier
            metadata: Updated metadata dictionary
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            import uuid
            
            # Convert document_id to UUID format (same logic as store_embedding)
            if len(document_id) == 36 and document_id.count('-') == 4:
                point_id = uuid.UUID(document_id)
            else:
                point_id = uuid.uuid5(uuid.NAMESPACE_DNS, document_id)
            
            # Update metadata in Qdrant
            self.qdrant_client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[str(point_id)]
            )
            
            logger.info(f"Updated metadata for document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update metadata for document {document_id}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of embedding processor and dependencies."""
        health_status = {
            "embedding_model_loaded": False,
            "qdrant_available": False,
            "collection_exists": False
        }
        
        # Check embedding model
        try:
            test_embedding = self.model.encode("test", convert_to_tensor=False)
            health_status["embedding_model_loaded"] = True
        except Exception as e:
            logger.error(f"Embedding model health check failed: {e}")
        
        # Check Qdrant connection
        try:
            collections = self.qdrant_client.get_collections()
            health_status["qdrant_available"] = True
            
            # Check if our collection exists
            collection_names = [col.name for col in collections.collections]
            health_status["collection_exists"] = self.collection_name in collection_names
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
        
        return health_status

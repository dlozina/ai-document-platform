"""
Enhanced embedding processor with intelligent document chunking
"""

import logging
import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EnhancedEmbeddingProcessor:
    """
    Enhanced embedding processor with intelligent document chunking.

    Features:
    - Intelligent document chunking with overlap
    - Semantic boundary detection
    - Chunk metadata tracking
    - Batch processing for efficiency
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: str | None = None,
        collection_name: str = "embeddings",
        vector_size: int = 384,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.model_name = model_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.qdrant_api_key = qdrant_api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize components
        self._initialize_model()
        self._initialize_qdrant()

    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def _initialize_qdrant(self):
        """Initialize Qdrant client and collection."""
        try:
            logger.info(
                f"Connecting to Qdrant at {self.qdrant_host}:{self.qdrant_port}"
            )
            self.qdrant_client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                api_key=self.qdrant_api_key,
            )

            # Create collection if it doesn't exist
            self._ensure_collection_exists()
            logger.info("Qdrant client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")

        except Exception as e:
            logger.error(f"Failed to ensure collection exists: {e}")
            raise

    def chunk_document_intelligently(
        self, text: str, document_id: str
    ) -> list[dict[str, Any]]:
        """
        Split document into intelligent chunks with metadata.

        Args:
            text: Document text
            document_id: Original document ID

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if len(text) <= self.chunk_size:
            return [
                {
                    "text": text,
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "start_char": 0,
                    "end_char": len(text),
                    "chunk_type": "single",
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_chunk_0",
                }
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence boundaries in the last 200 characters
                search_start = max(start, end - 200)
                last_period = text.rfind(".", search_start, end)
                last_exclamation = text.rfind("!", search_start, end)
                last_question = text.rfind("?", search_start, end)
                last_newline = text.rfind("\n", search_start, end)

                # Find the best break point
                break_points = [
                    last_period,
                    last_exclamation,
                    last_question,
                    last_newline,
                ]
                break_points = [
                    bp for bp in break_points if bp > start + self.chunk_size // 2
                ]

                if break_points:
                    end = max(break_points) + 1

            # Extract chunk
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{document_id}_chunk_{chunk_index}"
                chunks.append(
                    {
                        "text": chunk_text,
                        "chunk_index": chunk_index,
                        "total_chunks": 0,  # Will be updated later
                        "start_char": start,
                        "end_char": end,
                        "chunk_type": "middle" if chunk_index > 0 else "start",
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                    }
                )
                chunk_index += 1

            # Move start position with overlap
            start = end - self.chunk_overlap

        # Update total chunks count and final chunk type
        for chunk in chunks:
            chunk["total_chunks"] = len(chunks)
            if chunk["chunk_index"] == len(chunks) - 1:
                chunk["chunk_type"] = "end"

        return chunks

    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    def store_chunk_embedding(
        self,
        chunk: dict[str, Any],
        embedding: list[float],
        metadata: dict[str, Any | None] | None = None,
    ) -> str:
        """
        Store a chunk embedding in Qdrant.

        Args:
            chunk: Chunk dictionary with text and metadata
            embedding: Embedding vector
            metadata: Additional metadata

        Returns:
            Point ID in Qdrant
        """
        try:
            import uuid

            # Use chunk_id as the point ID
            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                try:
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
                except ValueError:
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)
            else:
                point_id = uuid.uuid4()

            # Prepare metadata
            point_metadata = {
                "text": chunk["text"],
                "text_length": len(chunk["text"]),
                "created_at": time.time(),
                "original_document_id": chunk["document_id"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": chunk["total_chunks"],
                "start_char": chunk["start_char"],
                "end_char": chunk["end_char"],
                "chunk_type": chunk["chunk_type"],
                "chunk_id": chunk["chunk_id"],
            }

            if metadata:
                point_metadata.update(metadata)

            # Create point
            point = PointStruct(
                id=str(point_id), vector=embedding, payload=point_metadata
            )

            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name, points=[point]
            )

            logger.info(f"Stored chunk embedding: {chunk_id}")
            return str(point_id)

        except Exception as e:
            logger.error(f"Failed to store chunk embedding: {e}")
            raise RuntimeError(f"Failed to store chunk embedding: {str(e)}") from e

    def process_document_with_chunking(
        self,
        text: str,
        document_id: str,
        filename: str,
        metadata: dict[str, Any | None] | None = None,
    ) -> dict[str, Any]:
        """
        Process a document by chunking it and storing embeddings for each chunk.

        Args:
            text: Document text
            document_id: Document identifier
            filename: Original filename
            metadata: Additional metadata

        Returns:
            Processing result with chunk information
        """
        start_time = time.time()

        try:
            logger.info(
                f"Processing document {document_id} with chunking (text length: {len(text)})"
            )

            # Chunk the document
            chunks = self.chunk_document_intelligently(text, document_id)
            logger.info(f"Created {len(chunks)} chunks for document {document_id}")

            # Process each chunk
            chunk_results = []
            for chunk in chunks:
                # Generate embedding for chunk
                embedding = self.generate_embedding(chunk["text"])

                # Prepare chunk metadata
                chunk_metadata = {
                    "filename": filename,
                    "original_text_length": len(text),
                }
                if metadata:
                    chunk_metadata.update(metadata)

                # Store chunk embedding
                point_id = self.store_chunk_embedding(chunk, embedding, chunk_metadata)

                chunk_results.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_index": chunk["chunk_index"],
                        "text_length": len(chunk["text"]),
                        "point_id": point_id,
                    }
                )

            processing_time = (time.time() - start_time) * 1000

            return {
                "document_id": document_id,
                "filename": filename,
                "total_chunks": len(chunks),
                "chunk_results": chunk_results,
                "processing_time_ms": processing_time,
                "method": "chunked_embedding",
            }

        except Exception as e:
            logger.error(f"Failed to process document with chunking: {e}")
            raise RuntimeError(f"Document chunking processing failed: {str(e)}") from e

    def search_similar_chunks(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: dict[str, Any | None] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional metadata filters

        Returns:
            List of similar chunks with scores
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
                "with_vectors": False,
            }

            # Add score threshold if specified
            if score_threshold is not None:
                search_params["score_threshold"] = score_threshold

            # Add filters if specified
            if filter_conditions:
                search_params["query_filter"] = Filter(
                    must=[
                        FieldCondition(key=key, match=MatchValue(value=value))
                        for key, value in filter_conditions.items()
                        if value is not None and isinstance(value, bool | int | str)
                    ]
                )

            # Perform search
            search_results = self.qdrant_client.search(**search_params)

            # Format results
            results = []
            for result in search_results:
                results.append(
                    {
                        "id": result.id,
                        "score": result.score,
                        "text": result.payload.get("text", ""),
                        "document_id": result.payload.get("original_document_id"),
                        "chunk_id": result.payload.get("chunk_id"),
                        "chunk_index": result.payload.get("chunk_index"),
                        "total_chunks": result.payload.get("total_chunks"),
                        "chunk_type": result.payload.get("chunk_type"),
                        "filename": result.payload.get("filename", "Unknown"),
                        "metadata": result.payload,
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Chunk search failed: {e}")
            raise

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        """
        Retrieve all chunks for a specific document.

        Args:
            document_id: Document identifier

        Returns:
            List of chunks for the document
        """
        try:
            # Search for all chunks of this document
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0.0] * self.vector_size,  # Dummy vector
                limit=1000,  # Large limit to get all chunks
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="original_document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )

            # Format results
            chunks = []
            for result in search_results:
                chunks.append(
                    {
                        "chunk_id": result.payload.get("chunk_id"),
                        "chunk_index": result.payload.get("chunk_index"),
                        "text": result.payload.get("text", ""),
                        "chunk_type": result.payload.get("chunk_type"),
                        "start_char": result.payload.get("start_char"),
                        "end_char": result.payload.get("end_char"),
                    }
                )

            # Sort by chunk index
            chunks.sort(key=lambda x: x["chunk_index"])

            return chunks

        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            raise

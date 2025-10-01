"""
Query processor for semantic search and question-answering
"""

import logging
import time
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import spacy
from sklearn.metrics.pairwise import cosine_similarity

from .config import get_settings
from .llm_client import LLMManager

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Main query processor for semantic search and QA."""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_model = None
        self.rerank_model = None
        self.qdrant_client = None
        self.nlp = None
        self.db_engine = None
        self.db_session = None
        self.llm_manager = None
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_qdrant()
        self._initialize_database()
        self._initialize_nlp()
        self._initialize_rerank_model()
        self._initialize_llm()
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            self.embedding_model = SentenceTransformer(self.settings.embedding_model)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _initialize_qdrant(self):
        """Initialize Qdrant client."""
        try:
            logger.info(f"Connecting to Qdrant at {self.settings.qdrant_url}")
            self.qdrant_client = QdrantClient(
                host=self.settings.qdrant_host,
                port=self.settings.qdrant_port,
                api_key=self.settings.qdrant_api_key
            )
            logger.info("Qdrant client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize database connection."""
        try:
            logger.info("Connecting to database")
            self.db_engine = create_engine(
                self.settings.database_url,
                pool_size=self.settings.database_pool_size,
                max_overflow=self.settings.database_max_overflow
            )
            self.db_session = sessionmaker(bind=self.db_engine)
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model."""
        try:
            logger.info("Loading spaCy model for NER")
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}. NER features will be limited.")
            self.nlp = None
    
    def _initialize_rerank_model(self):
        """Initialize reranking model."""
        if self.settings.enable_reranking:
            try:
                logger.info(f"Loading rerank model: {self.settings.rerank_model}")
                self.rerank_model = CrossEncoder(self.settings.rerank_model)
                logger.info("Rerank model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load rerank model: {e}. Reranking disabled.")
                self.rerank_model = None
    
    def _initialize_llm(self):
        """Initialize LLM manager."""
        try:
            provider = self.settings.llm_provider.lower()
            api_key = getattr(self.settings, f"{provider}_api_key", None)
            model = getattr(self.settings, f"{provider}_model", None)
            
            if not api_key:
                logger.warning(f"No API key found for {provider}. LLM features will be disabled.")
                return
            
            if not model:
                logger.warning(f"No model specified for {provider}. LLM features will be disabled.")
                return
            
            logger.info(f"Initializing LLM: {provider} with model {model}")
            self.llm_manager = LLMManager(provider, api_key, model)
            logger.info("LLM manager initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize LLM manager: {e}. LLM features will be disabled.")
            self.llm_manager = None
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query text."""
        try:
            embedding = self.embedding_model.encode(query, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
    
    def _build_qdrant_filter(self, filter_params: Optional[Dict[str, Any]], tenant_id: Optional[str] = None) -> Optional[Filter]:
        """Build Qdrant filter from query parameters."""
        if not filter_params and not tenant_id:
            return None
        
        conditions = []
        
        # Add tenant filter
        if tenant_id:
            conditions.append(
                FieldCondition(
                    key="tenant_id",
                    match=MatchValue(value=tenant_id)
                )
            )
        
        # Add other filters
        if filter_params:
            for key, value in filter_params.items():
                if value is None:
                    continue
                
                if key == "content_type" and value:
                    conditions.append(
                        FieldCondition(
                            key="content_type",
                            match=MatchValue(value=value)
                        )
                    )
                elif key == "file_type" and value:
                    conditions.append(
                        FieldCondition(
                            key="file_type",
                            match=MatchValue(value=value)
                        )
                    )
                elif key == "processing_status" and value:
                    conditions.append(
                        FieldCondition(
                            key="processing_status",
                            match=MatchValue(value=value)
                        )
                    )
                elif key == "created_by" and value:
                    conditions.append(
                        FieldCondition(
                            key="created_by",
                            match=MatchValue(value=value)
                        )
                    )
                elif key == "file_size_min" and value:
                    conditions.append(
                        FieldCondition(
                            key="file_size_bytes",
                            range=Range(gte=value)
                        )
                    )
                elif key == "file_size_max" and value:
                    conditions.append(
                        FieldCondition(
                            key="file_size_bytes",
                            range=Range(lte=value)
                        )
                    )
                elif key == "date_range" and value:
                    if hasattr(value, 'start') and value.start:
                        conditions.append(
                            FieldCondition(
                                key="upload_timestamp",
                                range=Range(gte=value.start.isoformat())
                            )
                        )
                    if hasattr(value, 'end') and value.end:
                        conditions.append(
                            FieldCondition(
                                key="upload_timestamp",
                                range=Range(lte=value.end.isoformat())
                            )
                        )
        
        return Filter(must=conditions) if conditions else None
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)
            
            # Build filter
            qdrant_filter = self._build_qdrant_filter(filter_params, tenant_id)
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.settings.qdrant_collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "document_id": result.payload.get("original_document_id"),  # Fixed field name
                    "filename": result.payload.get("filename", ""),
                    "metadata": result.payload
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            raise
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rerank search results using cross-encoder model."""
        if not self.rerank_model or not results:
            return results
        
        try:
            # Prepare pairs for reranking
            pairs = [(query, result["text"]) for result in results]
            
            # Get rerank scores
            rerank_scores = self.rerank_model.predict(pairs)
            
            # Update scores and sort
            for i, result in enumerate(results):
                result["rerank_score"] = float(rerank_scores[i])
                result["original_score"] = result["score"]
                result["score"] = result["rerank_score"]
            
            # Sort by rerank score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original results.")
            return results
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "confidence": 0.8,  # spaCy doesn't provide confidence scores
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    def filter_by_entities(
        self,
        results: List[Dict[str, Any]],
        entity_labels: Optional[List[str]] = None,
        entity_text: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filter results by NER entities."""
        if not entity_labels and not entity_text:
            return results
        
        filtered_results = []
        
        for result in results:
            # Get entities from document metadata
            doc_entities = result.get("metadata", {}).get("ner_entities", [])
            
            if not doc_entities:
                continue
            
            # Check entity label filter
            if entity_labels:
                has_matching_label = any(
                    any(entity.get("label") == label for entity in doc_entities)
                    for label in entity_labels
                )
                if not has_matching_label:
                    continue
            
            # Check entity text filter
            if entity_text:
                has_matching_text = any(
                    any(entity.get("text", "").lower() == text.lower() for entity in doc_entities)
                    for text in entity_text
                )
                if not has_matching_text:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def extract_answer_span(self, question: str, document_text: str, document_metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, float]:
        """Extract answer span from document text using entity-aware extraction."""
        try:
            # Extract entities from question
            question_entities = self.extract_entities(question)
            
            # Get document entities from metadata
            doc_entities = []
            if document_metadata:
                doc_entities = document_metadata.get("ner_entities", [])
            
            # Try entity-aware extraction first
            if question_entities and doc_entities:
                answer, confidence = self._extract_entity_aware_answer(
                    question, question_entities, document_text, doc_entities
                )
                if answer and confidence > 0.5:
                    return answer, confidence
            
            # Fallback to improved keyword-based extraction
            return self._extract_keyword_based_answer(question, document_text)
            
        except Exception as e:
            logger.error(f"Answer span extraction failed: {e}")
            return document_text[:500], 0.1
    
    def _extract_entity_aware_answer(
        self, 
        question: str, 
        question_entities: List[Dict[str, Any]], 
        document_text: str, 
        doc_entities: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Extract answer using entity matching and context."""
        try:
            question_lower = question.lower()
            
            # Find matching entities between question and document
            matching_entities = []
            for q_entity in question_entities:
                q_text = q_entity["text"].lower()
                for d_entity in doc_entities:
                    d_text = d_entity["text"].lower()
                    if q_text in d_text or d_text in q_text:
                        matching_entities.append({
                            "question_entity": q_entity,
                            "document_entity": d_entity
                        })
            
            if not matching_entities:
                return "", 0.0
            
            # Extract contextual answers based on entity types and relationships
            sentences = [s.strip() for s in document_text.split('.') if s.strip()]
            best_answer = ""
            best_confidence = 0.0
            
            for match in matching_entities:
                q_entity = match["question_entity"]
                d_entity = match["document_entity"]
                
                # Find sentences containing the matching entity
                entity_text = d_entity["text"]
                entity_sentences = [s for s in sentences if entity_text.lower() in s.lower()]
                
                if not entity_sentences:
                    continue
                
                # Extract contextual answer based on entity type
                answer, confidence = self._extract_contextual_answer(
                    q_entity, d_entity, entity_sentences, question_lower
                )
                
                if confidence > best_confidence:
                    best_answer = answer
                    best_confidence = confidence
            
            return best_answer, best_confidence
            
        except Exception as e:
            logger.warning(f"Entity-aware extraction failed: {e}")
            return "", 0.0
    
    def _extract_contextual_answer(
        self, 
        q_entity: Dict[str, Any], 
        d_entity: Dict[str, Any], 
        entity_sentences: List[str], 
        question_lower: str
    ) -> Tuple[str, float]:
        """Extract contextual answer based on entity type and question context."""
        try:
            entity_label = d_entity.get("label", "").upper()
            entity_text = d_entity["text"]
            
            # Handle different entity types
            if entity_label in ["PERSON", "PER"]:
                return self._extract_person_context(entity_text, entity_sentences, question_lower)
            elif entity_label in ["GPE", "LOCATION", "LOC"]:
                return self._extract_location_context(entity_text, entity_sentences, question_lower)
            elif entity_label in ["ORG", "ORGANIZATION"]:
                return self._extract_organization_context(entity_text, entity_sentences, question_lower)
            else:
                # Generic entity extraction
                return self._extract_generic_context(entity_text, entity_sentences, question_lower)
                
        except Exception as e:
            logger.warning(f"Contextual extraction failed: {e}")
            return "", 0.0
    
    def _extract_location_context(self, entity_text: str, sentences: List[str], question_lower: str) -> Tuple[str, float]:
        """Extract location-related context."""
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if entity_text.lower() in sentence_lower:
                # Look for location patterns
                if any(word in sentence_lower for word in ["based in", "located in", "from", "in", "at"]):
                    # Extract the location phrase
                    words = sentence.split()
                    entity_words = entity_text.split()
                    
                    # Find entity position and extract surrounding context
                    for i, word in enumerate(words):
                        if entity_words[0].lower() in word.lower():
                            # Extract 3-5 words around the entity
                            start = max(0, i - 2)
                            end = min(len(words), i + len(entity_words) + 3)
                            context = " ".join(words[start:end])
                            
                            # Clean up the context
                            context = context.replace("  ", " ").strip()
                            return context, 0.8
        
        # Fallback to sentence containing entity
        if sentences:
            return sentences[0], 0.6
        
        return "", 0.0
    
    def _extract_person_context(self, entity_text: str, sentences: List[str], question_lower: str) -> Tuple[str, float]:
        """Extract person-related context."""
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if entity_text.lower() in sentence_lower:
                # Look for introduction patterns
                if any(word in sentence_lower for word in ["i'm", "i am", "my name is", "hello"]):
                    return sentence, 0.7
        
        # Fallback to sentence containing entity
        if sentences:
            return sentences[0], 0.5
        
        return "", 0.0
    
    def _extract_organization_context(self, entity_text: str, sentences: List[str], question_lower: str) -> Tuple[str, float]:
        """Extract organization-related context."""
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if entity_text.lower() in sentence_lower:
                # Look for work/company patterns
                if any(word in sentence_lower for word in ["worked at", "company", "startup", "firm"]):
                    return sentence, 0.7
        
        # Fallback to sentence containing entity
        if sentences:
            return sentences[0], 0.5
        
        return "", 0.0
    
    def _extract_generic_context(self, entity_text: str, sentences: List[str], question_lower: str) -> Tuple[str, float]:
        """Extract generic context for any entity type."""
        if sentences:
            return sentences[0], 0.5
        return "", 0.0
    
    def _extract_keyword_based_answer(self, question: str, document_text: str) -> Tuple[str, float]:
        """Improved keyword-based extraction as fallback."""
        try:
            question_lower = question.lower()
            text_lower = document_text.lower()
            
            # Find sentences containing question keywords
            sentences = [s.strip() for s in document_text.split('.') if s.strip()]
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Check if sentence contains question words (improved logic)
                question_words = [word for word in question_lower.split() if len(word) > 2]  # Reduced threshold
                if any(word in sentence_lower for word in question_words):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                # Take the most relevant sentence (first one with highest keyword density)
                best_sentence = relevant_sentences[0]
                answer = best_sentence
                confidence = min(0.7, len(relevant_sentences) / 3.0)  # Improved confidence scoring
            else:
                # Fallback to first few sentences
                answer = '. '.join(sentences[:2])
                confidence = 0.3
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Keyword-based extraction failed: {e}")
            return document_text[:500], 0.1
    
    async def generate_rag_answer(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        max_context_length: Optional[int] = None
    ) -> Tuple[str, float]:
        """Generate answer using RAG (Retrieval-Augmented Generation)."""
        try:
            if not self.llm_manager:
                logger.warning("LLM manager not available, falling back to template response")
                return self._generate_fallback_answer(question, context_documents)
            
            # Use the LLM manager to generate the answer
            context_length = max_context_length or self.settings.max_context_length
            answer, confidence = await self.llm_manager.generate_rag_answer(
                question=question,
                context_documents=context_documents,
                max_context_length=context_length,
                temperature=0.7,
                max_tokens=1000
            )
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"RAG answer generation failed: {e}")
            return self._generate_fallback_answer(question, context_documents)
    
    def _generate_fallback_answer(
        self,
        question: str,
        context_documents: List[Dict[str, Any]]
    ) -> Tuple[str, float]:
        """Generate a fallback answer when LLM is not available."""
        try:
            # Prepare context
            context_texts = []
            
            for doc in context_documents[:3]:  # Limit to top 3 documents
                text = doc.get("text", "")
                if len(text) > 200:  # Truncate very long texts
                    text = text[:200] + "..."
                context_texts.append(text)
            
            context = "\n\n".join(context_texts)
            
            # Generate a simple answer
            answer = f"Based on the available documents, here's what I found regarding your question: '{question}'\n\n"
            answer += f"Relevant information:\n{context[:500]}..."
            
            confidence = 0.3  # Lower confidence for fallback
            
            return answer, confidence
            
        except Exception as e:
            logger.error(f"Fallback answer generation failed: {e}")
            return "I apologize, but I couldn't generate a proper answer at this time.", 0.1
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of all components."""
        health_status = {
            "embedding_model_loaded": self.embedding_model is not None,
            "qdrant_available": False,
            "database_connected": False,
            "nlp_loaded": self.nlp is not None,
            "rerank_model_loaded": self.rerank_model is not None,
            "llm_available": self.llm_manager is not None
        }
        
        # Check Qdrant
        try:
            collections = self.qdrant_client.get_collections()
            health_status["qdrant_available"] = True
        except Exception:
            pass
        
        # Check database
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            health_status["database_connected"] = True
        except Exception:
            pass
        
        return health_status

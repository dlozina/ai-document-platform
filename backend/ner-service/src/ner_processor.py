"""
NER Processor Module

Handles Named Entity Recognition using spaCy models:
- en_core_web_sm (small, fast model)
- en_core_web_lg (large, accurate model)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import spacy
from spacy import displacy
import json

logger = logging.getLogger(__name__)


class NERProcessor:
    """
    Processes text to extract named entities using spaCy models.
    
    Supports:
    - Multiple spaCy models (small and large)
    - Custom entity type filtering
    - Batch processing
    - Confidence scoring
    """
    
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        fallback_model: str = "en_core_web_lg"
    ):
        """
        Initialize NER Processor.
        
        Args:
            model_name: Primary spaCy model to use
            fallback_model: Fallback model if primary fails
        """
        self.model_name = model_name
        self.fallback_model = fallback_model
        self.nlp = None
        self.fallback_nlp = None
        
        # Load primary model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError as e:
            logger.warning(f"Failed to load primary model {model_name}: {e}")
            self._load_fallback_model()
        
        # Load fallback model
        if self.fallback_model != model_name:
            try:
                self.fallback_nlp = spacy.load(fallback_model)
                logger.info(f"Loaded fallback spaCy model: {fallback_model}")
            except OSError as e:
                logger.warning(f"Failed to load fallback model {fallback_model}: {e}")
    
    def _load_fallback_model(self):
        """Load fallback model as primary."""
        try:
            self.nlp = spacy.load(self.fallback_model)
            logger.info(f"Using fallback model as primary: {self.fallback_model}")
        except OSError as e:
            logger.error(f"Failed to load any spaCy model: {e}")
            raise RuntimeError(
                "No spaCy models available. Install with: "
                "python -m spacy download en_core_web_sm"
            )
    
    def process_text(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        include_confidence: bool = True,
        use_fallback: bool = False
    ) -> Dict[str, any]:
        """
        Process text and extract named entities.
        
        Args:
            text: Input text to process
            entity_types: Specific entity types to extract (None = all)
            include_confidence: Whether to include confidence scores
            use_fallback: Whether to use fallback model
        
        Returns:
            {
                'text': str,                    # Original text
                'entities': List[EntityInfo],   # Detected entities
                'entity_count': int,           # Number of entities
                'model_used': str,             # Model used
                'processing_time_ms': float     # Processing time
            }
        
        Raises:
            ValueError: If text is empty or too long
            RuntimeError: If processing fails
        """
        start_time = time.time()
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if len(text) > 1000000:  # 1M characters
            raise ValueError("Text too long. Maximum length: 1,000,000 characters")
        
        try:
            # Choose model
            nlp_model = self.fallback_nlp if use_fallback else self.nlp
            if nlp_model is None:
                raise RuntimeError("No spaCy model available")
            
            # Process text
            doc = nlp_model(text)
            
            # Extract entities
            entities = self._extract_entities(
                doc, entity_types, include_confidence
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'text': text,
                'entities': entities,
                'entity_count': len(entities),
                'model_used': self.fallback_model if use_fallback else self.model_name,
                'processing_time_ms': round(processing_time, 2),
                'text_length': len(text)
            }
            
            logger.info(
                f"Processed text ({len(text)} chars) using {result['model_used']} "
                f"in {processing_time:.2f}ms - found {len(entities)} entities"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process text: {e}", exc_info=True)
            raise RuntimeError(f"NER processing failed: {str(e)}")
    
    def _extract_entities(
        self,
        doc,
        entity_types: Optional[List[str]] = None,
        include_confidence: bool = True
    ) -> List[Dict[str, any]]:
        """Extract entities from spaCy document."""
        entities = []
        
        for ent in doc.ents:
            # Filter by entity type if specified
            if entity_types and ent.label_ not in entity_types:
                continue
            
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            
            # Add confidence if requested and available
            if include_confidence:
                # spaCy doesn't provide confidence scores by default
                # We can estimate based on entity length and context
                entity_info['confidence'] = self._estimate_confidence(ent, doc)
            
            entities.append(entity_info)
        
        return entities
    
    def _estimate_confidence(self, ent, doc) -> float:
        """
        Estimate confidence score for an entity.
        
        This is a heuristic since spaCy doesn't provide confidence scores
        for NER by default. We use factors like:
        - Entity length
        - Context around the entity
        - Entity type frequency
        """
        # Base confidence
        confidence = 0.7
        
        # Adjust based on entity length
        if len(ent.text) > 20:
            confidence += 0.1
        elif len(ent.text) < 3:
            confidence -= 0.2
        
        # Adjust based on entity type
        high_confidence_types = ['PERSON', 'ORG', 'GPE']
        if ent.label_ in high_confidence_types:
            confidence += 0.1
        
        # Adjust based on context (capitalization)
        if ent.text.istitle() or ent.text.isupper():
            confidence += 0.05
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def process_batch(
        self,
        texts: List[str],
        entity_types: Optional[List[str]] = None,
        include_confidence: bool = True,
        batch_size: int = 100
    ) -> Dict[str, any]:
        """
        Process multiple texts in batch.
        
        Args:
            texts: List of texts to process
            entity_types: Specific entity types to extract
            include_confidence: Whether to include confidence scores
            batch_size: Number of texts to process at once
        
        Returns:
            {
                'results': List[NERResponse],  # Results for each text
                'total_processing_time_ms': float,
                'batch_size': int
            }
        """
        start_time = time.time()
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Use spaCy's batch processing if available
            if self.nlp and hasattr(self.nlp, 'pipe'):
                batch_docs = list(self.nlp.pipe(batch_texts))
                
                for j, doc in enumerate(batch_docs):
                    entities = self._extract_entities(doc, entity_types, include_confidence)
                    
                    result = {
                        'text': batch_texts[j],
                        'entities': entities,
                        'entity_count': len(entities),
                        'model_used': self.model_name,
                        'processing_time_ms': 0,  # Will be calculated for entire batch
                        'text_length': len(batch_texts[j])
                    }
                    results.append(result)
            else:
                # Process individually
                for text in batch_texts:
                    result = self.process_text(text, entity_types, include_confidence)
                    results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'total_processing_time_ms': round(total_time, 2),
            'batch_size': len(texts)
        }
    
    def get_entity_statistics(self, entities: List[Dict[str, any]]) -> Dict[str, any]:
        """
        Generate statistics about detected entities.
        
        Args:
            entities: List of entity dictionaries
        
        Returns:
            Statistics dictionary
        """
        if not entities:
            return {
                'total_entities': 0,
                'entity_types': [],
                'most_common_entities': []
            }
        
        # Count by type
        type_counts = {}
        entity_counts = {}
        
        for entity in entities:
            label = entity['label']
            text = entity['text']
            
            # Count by type
            type_counts[label] = type_counts.get(label, 0) + 1
            
            # Count individual entities
            key = f"{text} ({label})"
            entity_counts[key] = entity_counts.get(key, 0) + 1
        
        # Create entity type statistics
        entity_types = []
        for label, count in type_counts.items():
            # Get examples of this type
            examples = [e['text'] for e in entities if e['label'] == label][:5]
            unique_count = len(set(e['text'] for e in entities if e['label'] == label))
            
            entity_types.append({
                'entity_type': label,
                'count': count,
                'unique_count': unique_count,
                'examples': examples
            })
        
        # Get most common entities
        most_common = sorted(
            entity_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_entities': len(entities),
            'entity_types': entity_types,
            'most_common_entities': [
                {'entity': entity, 'count': count}
                for entity, count in most_common
            ]
        }
    
    def visualize_entities(self, text: str, entities: List[Dict[str, any]]) -> str:
        """
        Generate HTML visualization of entities.
        
        Args:
            text: Original text
            entities: Detected entities
        
        Returns:
            HTML string for visualization
        """
        if not self.nlp:
            return "<p>No spaCy model available for visualization</p>"
        
        # Create a spaCy doc
        doc = self.nlp(text)
        
        # Override entities with our detected ones
        doc.ents = []
        for entity in entities:
            span = doc.char_span(
                entity['start'],
                entity['end'],
                label=entity['label']
            )
            if span:
                doc.ents = list(doc.ents) + [span]
        
        # Generate HTML
        html = displacy.render(doc, style="ent", page=True)
        return html
    
    def get_available_models(self) -> Dict[str, bool]:
        """Get information about available spaCy models."""
        models = {}
        
        try:
            spacy.load(self.model_name)
            models[self.model_name] = True
        except OSError:
            models[self.model_name] = False
        
        if self.fallback_model != self.model_name:
            try:
                spacy.load(self.fallback_model)
                models[self.fallback_model] = True
            except OSError:
                models[self.fallback_model] = False
        
        return models

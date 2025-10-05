"""
NER Processor Module

Handles Named Entity Recognition using spaCy models:
- English: en_core_web_sm, en_core_web_lg
- Croatian: hr_core_news_sm, hr_core_news_lg
"""

import logging
import time

import spacy
from spacy import displacy

logger = logging.getLogger(__name__)

# Supported languages and their models
SUPPORTED_LANGUAGES = {
    "en": {"small": "en_core_web_sm", "large": "en_core_web_lg"},
    "hr": {"small": "hr_core_news_sm", "large": "hr_core_news_lg"},
}


class NERProcessor:
    """
    Processes text to extract named entities using spaCy models.

    Supports:
    - Multiple languages (English, Croatian)
    - Multiple spaCy models per language (small and large)
    - Custom entity type filtering
    - Batch processing
    - Confidence scoring
    """

    def __init__(self):
        """
        Initialize NER Processor with support for multiple languages.
        """
        self.models = {}  # Dictionary to store loaded models by language

        # Load models for all supported languages
        for lang_code, models in SUPPORTED_LANGUAGES.items():
            self._load_language_models(lang_code, models)

    def _load_language_models(self, lang_code: str, models: dict[str, str]):
        """Load small and large models for a specific language."""
        lang_models = {}

        # Try to load small model first
        try:
            lang_models["small"] = spacy.load(models["small"])
            logger.info(f"Loaded {lang_code} small model: {models['small']}")
        except OSError as e:
            logger.warning(
                f"Failed to load {lang_code} small model {models['small']}: {e}"
            )
            lang_models["small"] = None

        # Try to load large model
        try:
            lang_models["large"] = spacy.load(models["large"])
            logger.info(f"Loaded {lang_code} large model: {models['large']}")
        except OSError as e:
            logger.warning(
                f"Failed to load {lang_code} large model {models['large']}: {e}"
            )
            lang_models["large"] = None

        # If no models loaded for this language, log error
        if not any(lang_models.values()):
            logger.error(f"No models available for language: {lang_code}")
        else:
            self.models[lang_code] = lang_models

    def _get_model(self, language: str, use_large: bool = False):
        """Get the appropriate model for the given language."""
        if language not in self.models:
            raise ValueError(f"Unsupported language: {language}")

        lang_models = self.models[language]
        model_type = "large" if use_large else "small"

        # Try preferred model type first
        if lang_models[model_type] is not None:
            return lang_models[model_type], SUPPORTED_LANGUAGES[language][model_type]

        # Fallback to other model type
        fallback_type = "small" if use_large else "large"
        if lang_models[fallback_type] is not None:
            logger.warning(f"Using {fallback_type} model as fallback for {language}")
            return (
                lang_models[fallback_type],
                SUPPORTED_LANGUAGES[language][fallback_type],
            )

        raise RuntimeError(f"No models available for language: {language}")

    def process_text(
        self,
        text: str,
        language: str = "en",
        entity_types: list[str | None] = None,
        include_confidence: bool = True,
        use_large_model: bool = False,
    ) -> dict[str, any]:
        """
        Process text and extract named entities.

        Args:
            text: Input text to process
            language: Language code (en, hr)
            entity_types: Specific entity types to extract (None = all)
            include_confidence: Whether to include confidence scores
            use_large_model: Whether to use large model (default: small)

        Returns:
            {
                'text': str,                    # Original text
                'entities': List[EntityInfo],   # Detected entities
                'entity_count': int,           # Number of entities
                'model_used': str,             # Model used
                'language': str,              # Language processed
                'processing_time_ms': float     # Processing time
            }

        Raises:
            ValueError: If text is empty, too long, or unsupported language
            RuntimeError: If processing fails
        """
        start_time = time.time()

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > 1000000:  # 1M characters
            raise ValueError("Text too long. Maximum length: 1,000,000 characters")

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )

        try:
            # Get appropriate model
            nlp_model, model_name = self._get_model(language, use_large_model)

            # Process text
            doc = nlp_model(text)

            # Extract entities
            entities = self._extract_entities(doc, entity_types, include_confidence)

            processing_time = (time.time() - start_time) * 1000

            result = {
                "text": text,
                "entities": entities,
                "entity_count": len(entities),
                "spacy_model_used": model_name,
                "language": language,
                "processing_time_ms": round(processing_time, 2),
                "text_length": len(text),
            }

            logger.info(
                f"Processed {language} text ({len(text)} chars) using {model_name} "
                f"in {processing_time:.2f}ms - found {len(entities)} entities"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to process text: {e}", exc_info=True)
            raise RuntimeError(f"NER processing failed: {str(e)}") from e

    def _extract_entities(
        self,
        doc,
        entity_types: list[str | None] = None,
        include_confidence: bool = True,
    ) -> list[dict[str, any]]:
        """Extract entities from spaCy document."""
        entities = []

        for ent in doc.ents:
            # Filter by entity type if specified
            if entity_types and ent.label_ not in entity_types:
                continue

            entity_info = {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }

            # Add confidence if requested and available
            if include_confidence:
                # spaCy doesn't provide confidence scores by default
                # We can estimate based on entity length and context
                entity_info["confidence"] = self._estimate_confidence(ent, doc)

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
        high_confidence_types = ["PERSON", "ORG", "GPE"]
        if ent.label_ in high_confidence_types:
            confidence += 0.1

        # Adjust based on context (capitalization)
        if ent.text.istitle() or ent.text.isupper():
            confidence += 0.05

        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))

    def process_batch(
        self,
        texts: list[str],
        language: str = "en",
        entity_types: list[str | None] = None,
        include_confidence: bool = True,
        batch_size: int = 100,
        use_large_model: bool = False,
    ) -> dict[str, any]:
        """
        Process multiple texts in batch.

        Args:
            texts: List of texts to process
            language: Language code (en, hr)
            entity_types: Specific entity types to extract
            include_confidence: Whether to include confidence scores
            batch_size: Number of texts to process at once
            use_large_model: Whether to use large model (default: small)

        Returns:
            {
                'results': List[NERResponse],  # Results for each text
                'total_processing_time_ms': float,
                'batch_size': int,
                'language': str
            }
        """
        start_time = time.time()
        results = []

        # Get appropriate model
        nlp_model, model_name = self._get_model(language, use_large_model)

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Use spaCy's batch processing if available
            if nlp_model and hasattr(nlp_model, "pipe"):
                batch_docs = list(nlp_model.pipe(batch_texts))

                for j, doc in enumerate(batch_docs):
                    entities = self._extract_entities(
                        doc, entity_types, include_confidence
                    )

                    result = {
                        "text": batch_texts[j],
                        "entities": entities,
                        "entity_count": len(entities),
                        "spacy_model_used": model_name,
                        "language": language,
                        "processing_time_ms": 0,  # Will be calculated for entire batch
                        "text_length": len(batch_texts[j]),
                    }
                    results.append(result)
            else:
                # Process individually
                for text in batch_texts:
                    result = self.process_text(
                        text,
                        language,
                        entity_types,
                        include_confidence,
                        use_large_model,
                    )
                    results.append(result)

        total_time = (time.time() - start_time) * 1000

        return {
            "results": results,
            "total_processing_time_ms": round(total_time, 2),
            "batch_size": len(texts),
            "language": language,
        }

    def get_entity_statistics(self, entities: list[dict[str, any]]) -> dict[str, any]:
        """
        Generate statistics about detected entities.

        Args:
            entities: List of entity dictionaries

        Returns:
            Statistics dictionary
        """
        if not entities:
            return {"total_entities": 0, "entity_types": [], "most_common_entities": []}

        # Count by type
        type_counts = {}
        entity_counts = {}

        for entity in entities:
            label = entity["label"]
            text = entity["text"]

            # Count by type
            type_counts[label] = type_counts.get(label, 0) + 1

            # Count individual entities
            key = f"{text} ({label})"
            entity_counts[key] = entity_counts.get(key, 0) + 1

        # Create entity type statistics
        entity_types = []
        for label, count in type_counts.items():
            # Get examples of this type
            examples = [e["text"] for e in entities if e["label"] == label][:5]
            unique_count = len({e["text"] for e in entities if e["label"] == label})

            entity_types.append(
                {
                    "entity_type": label,
                    "count": count,
                    "unique_count": unique_count,
                    "examples": examples,
                }
            )

        # Get most common entities
        most_common = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "most_common_entities": [
                {"entity": entity, "count": count} for entity, count in most_common
            ],
        }

    def visualize_entities(
        self, text: str, entities: list[dict[str, any]], language: str = "en"
    ) -> str:
        """
        Generate HTML visualization of entities.

        Args:
            text: Original text
            entities: Detected entities
            language: Language code (en, hr)

        Returns:
            HTML string for visualization
        """
        try:
            nlp_model, _ = self._get_model(
                language, False
            )  # Use small model for visualization
        except (ValueError, RuntimeError):
            return f"<p>No spaCy model available for language: {language}</p>"

        # Create a spaCy doc
        doc = nlp_model(text)

        # Override entities with our detected ones
        doc.ents = []
        for entity in entities:
            span = doc.char_span(entity["start"], entity["end"], label=entity["label"])
            if span:
                doc.ents = list(doc.ents) + [span]

        # Generate HTML
        html = displacy.render(doc, style="ent", page=True)
        return html

    def get_available_models(self) -> dict[str, bool]:
        """Get information about available spaCy models for all languages."""
        models = {}

        for lang_code, lang_models in self.models.items():
            for model_type, model_name in SUPPORTED_LANGUAGES[lang_code].items():
                models[model_name] = lang_models[model_type] is not None

        return models

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes."""
        return list(SUPPORTED_LANGUAGES.keys())

    def is_language_supported(self, language: str) -> bool:
        """Check if a language is supported."""
        return language in SUPPORTED_LANGUAGES and language in self.models

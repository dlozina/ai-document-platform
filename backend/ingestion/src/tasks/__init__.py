"""
Celery Tasks Package for Ingestion Service

Contains all task definitions for the processing pipeline.
"""

from .ocr_tasks import process_document_ocr, trigger_next_processing, move_to_dead_letter
from .ner_tasks import process_document_ner, check_processing_completion as check_ner_completion
from .embedding_tasks import process_document_embedding, check_processing_completion as check_embedding_completion
from .completion_tasks import (
    finalize_document_processing, 
    cleanup_failed_processing, 
    generate_processing_report
)

__all__ = [
    "process_document_ocr",
    "trigger_next_processing", 
    "move_to_dead_letter",
    "process_document_ner",
    "process_document_embedding",
    "check_ner_completion",
    "check_embedding_completion",
    "finalize_document_processing",
    "cleanup_failed_processing",
    "generate_processing_report"
]

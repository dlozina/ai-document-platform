"""
Celery Tasks Package for Ingestion Service

Contains all task definitions for the processing pipeline.
"""

from .completion_tasks import (
    cleanup_failed_processing,
    finalize_document_processing,
    generate_processing_report,
)
from .embedding_tasks import check_processing_completion as check_embedding_completion
from .embedding_tasks import process_document_embedding
from .ner_tasks import check_processing_completion as check_ner_completion
from .ner_tasks import process_document_ner
from .ocr_tasks import (
    move_to_dead_letter,
    process_document_ocr,
    trigger_next_processing,
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
    "generate_processing_report",
]

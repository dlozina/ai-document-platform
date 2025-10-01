"""
Database models and operations for Ingestion Service
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, Text, JSON, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

from .config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class Document(Base):
    """Document metadata table."""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(String(255), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    content_type = Column(String(100), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    storage_path = Column(String(1000), nullable=False)
    upload_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_by = Column(String(255), nullable=True)
    
    # Processing status
    processing_status = Column(String(50), nullable=False, default="pending")
    ocr_status = Column(String(50), nullable=True)
    ner_status = Column(String(50), nullable=True)
    embedding_status = Column(String(50), nullable=True)
    
    # Processing results
    ocr_text = Column(Text, nullable=True)
    ner_entities = Column(JSON, nullable=True)
    embedding_vector = Column(JSON, nullable=True)
    
    # Metadata
    tags = Column(JSON, nullable=True, default=list)
    description = Column(Text, nullable=True)
    retention_date = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class ProcessingJob(Base):
    """Processing job tracking table."""
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    tenant_id = Column(String(255), nullable=False, index=True)
    job_type = Column(String(50), nullable=False)  # ocr, ner, embedding
    status = Column(String(50), nullable=False, default="pending")
    progress_percentage = Column(Integer, nullable=False, default=0)
    message = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class TenantQuota(Base):
    """Tenant quota configuration table."""
    __tablename__ = "tenant_quotas"
    
    tenant_id = Column(String(255), primary_key=True)
    max_storage_bytes = Column(Integer, nullable=False, default=1073741824)  # 1GB default
    max_files = Column(Integer, nullable=False, default=1000)
    max_file_size_bytes = Column(Integer, nullable=False, default=104857600)  # 100MB default
    retention_days = Column(Integer, nullable=False, default=365)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class DatabaseManager:
    """Database connection and operations manager."""
    
    def __init__(self):
        self.settings = get_settings()
        self.engine = create_engine(
            self.settings.database_url,
            pool_size=self.settings.database_pool_size,
            max_overflow=self.settings.database_max_overflow,
            echo=False
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()
    
    def create_document(self, session: Session, document_data: Dict[str, Any]) -> Document:
        """Create a new document record."""
        document = Document(**document_data)
        session.add(document)
        session.commit()
        session.refresh(document)
        return document
    
    def get_document(self, session: Session, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return session.query(Document).filter(
            Document.id == document_id,
            Document.is_deleted == False
        ).first()
    
    def get_documents_by_tenant(
        self, 
        session: Session, 
        tenant_id: str, 
        page: int = 1, 
        page_size: int = 20,
        file_type: Optional[str] = None,
        processing_status: Optional[str] = None
    ) -> List[Document]:
        """Get documents for a tenant with pagination and filtering."""
        query = session.query(Document).filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted == False
        )
        
        if file_type:
            query = query.filter(Document.file_type == file_type)
        
        if processing_status:
            query = query.filter(Document.processing_status == processing_status)
        
        offset = (page - 1) * page_size
        return query.offset(offset).limit(page_size).all()
    
    def count_documents_by_tenant(
        self, 
        session: Session, 
        tenant_id: str,
        file_type: Optional[str] = None,
        processing_status: Optional[str] = None
    ) -> int:
        """Count documents for a tenant."""
        query = session.query(Document).filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted == False
        )
        
        if file_type:
            query = query.filter(Document.file_type == file_type)
        
        if processing_status:
            query = query.filter(Document.processing_status == processing_status)
        
        return query.count()
    
    def update_document(self, session: Session, document_id: str, update_data: Dict[str, Any]) -> Optional[Document]:
        """Update document record."""
        document = self.get_document(session, document_id)
        if document:
            for key, value in update_data.items():
                setattr(document, key, value)
            document.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(document)
        return document
    
    def soft_delete_document(self, session: Session, document_id: str) -> bool:
        """Soft delete a document."""
        document = self.get_document(session, document_id)
        if document:
            document.is_deleted = True
            document.updated_at = datetime.utcnow()
            session.commit()
            return True
        return False
    
    def get_document_by_hash(self, session: Session, file_hash: str) -> Optional[Document]:
        """Get document by file hash (for deduplication)."""
        return session.query(Document).filter(
            Document.file_hash == file_hash,
            Document.is_deleted == False
        ).first()
    
    def create_processing_job(self, session: Session, job_data: Dict[str, Any]) -> ProcessingJob:
        """Create a new processing job or update existing one if it already exists."""
        job_id = job_data.get('id')
        
        # Try to get existing job first
        existing_job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        
        if existing_job:
            # Update existing job
            for key, value in job_data.items():
                if hasattr(existing_job, key):
                    setattr(existing_job, key, value)
            existing_job.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(existing_job)
            return existing_job
        else:
            # Create new job
            job = ProcessingJob(**job_data)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def update_processing_job(self, session: Session, job_id: str, update_data: Dict[str, Any]) -> Optional[ProcessingJob]:
        """Update processing job."""
        job = session.query(ProcessingJob).filter(ProcessingJob.id == job_id).first()
        if job:
            for key, value in update_data.items():
                setattr(job, key, value)
            job.updated_at = datetime.utcnow()
            session.commit()
            session.refresh(job)
        return job
    
    def get_tenant_quota(self, session: Session, tenant_id: str) -> Optional[TenantQuota]:
        """Get tenant quota configuration."""
        return session.query(TenantQuota).filter(TenantQuota.tenant_id == tenant_id).first()
    
    def create_tenant_quota(self, session: Session, tenant_id: str, quota_data: Dict[str, Any]) -> TenantQuota:
        """Create tenant quota configuration."""
        quota = TenantQuota(tenant_id=tenant_id, **quota_data)
        session.add(quota)
        session.commit()
        session.refresh(quota)
        return quota
    
    def get_tenant_storage_usage(self, session: Session, tenant_id: str) -> Dict[str, int]:
        """Get tenant storage usage statistics."""
        result = session.query(
            Document.file_size_bytes,
            Document.id
        ).filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted == False
        ).all()
        
        total_size = sum(row.file_size_bytes for row in result)
        file_count = len(result)
        
        return {
            "used_storage_bytes": total_size,
            "used_files": file_count
        }
    
    def search_documents(
        self, 
        session: Session, 
        tenant_id: str, 
        query: Optional[str] = None,
        file_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 20
    ) -> List[Document]:
        """Search documents with various filters."""
        db_query = session.query(Document).filter(
            Document.tenant_id == tenant_id,
            Document.is_deleted == False
        )
        
        if query:
            db_query = db_query.filter(
                Document.filename.ilike(f"%{query}%") |
                Document.description.ilike(f"%{query}%")
            )
        
        if file_types:
            db_query = db_query.filter(Document.file_type.in_(file_types))
        
        if tags:
            # PostgreSQL JSON array contains operator
            for tag in tags:
                db_query = db_query.filter(Document.tags.contains([tag]))
        
        if date_from:
            db_query = db_query.filter(Document.upload_timestamp >= date_from)
        
        if date_to:
            db_query = db_query.filter(Document.upload_timestamp <= date_to)
        
        offset = (page - 1) * page_size
        return db_query.offset(offset).limit(page_size).all()


# Global database manager instance
db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def get_db_session() -> Session:
    """Get database session."""
    return get_db_manager().get_session()

"""
Pydantic models for API Gateway Service
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_serializer


class UserRole(str, Enum):
    """User role enumeration."""

    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"


class TokenType(str, Enum):
    """Token type enumeration."""

    ACCESS = "access"
    REFRESH = "refresh"


class UserBase(BaseModel):
    """Base user model."""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: str | None = Field(None, max_length=100, description="Full name")
    is_active: bool = Field(True, description="Whether user is active")
    role: UserRole = Field(UserRole.USER, description="User role")


class UserCreate(UserBase):
    """User creation model."""

    password: str = Field(..., min_length=8, description="Password")


class UserUpdate(BaseModel):
    """User update model."""

    username: str | None = Field(None, min_length=3, max_length=50)
    email: EmailStr | None = None
    full_name: str | None = Field(None, max_length=100)
    is_active: bool | None = None
    role: UserRole | None = None


class User(UserBase):
    """User model with ID."""

    id: int = Field(..., description="User ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None

    @field_serializer("created_at", "updated_at", "last_login")
    def serialize_datetime(self, value: datetime) -> str:
        return value.isoformat() if value else None


class UserInDB(User):
    """User model with hashed password."""

    hashed_password: str = Field(..., description="Hashed password")


class Token(BaseModel):
    """Token model."""

    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")


class TokenData(BaseModel):
    """Token data model."""

    username: str | None = None
    user_id: int | None = None
    role: UserRole | None = None


class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str = Field(..., description="Refresh token")


class PasswordChangeRequest(BaseModel):
    """Password change request model."""

    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    redis_connected: bool = Field(..., description="Redis connection status")
    dependencies: dict[str, bool] = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")
    error_code: str | None = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO string."""
        return value.isoformat()


class RateLimitResponse(BaseModel):
    """Rate limit response model."""

    limit: int = Field(..., description="Request limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="Reset time")
    retry_after: int | None = Field(None, description="Retry after seconds")


class UserListResponse(BaseModel):
    """User list response model."""

    users: list[User] = Field(..., description="List of users")
    total_count: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class ProxyRequest(BaseModel):
    """Proxy request model."""

    method: str = Field(..., description="HTTP method")
    url: str = Field(..., description="Target URL")
    headers: dict[str, str] | None = Field(None, description="Request headers")
    data: dict[str, Any] | None = Field(None, description="Request data")
    params: dict[str, Any] | None = Field(None, description="Query parameters")


class ProxyResponse(BaseModel):
    """Proxy response model."""

    status_code: int = Field(..., description="Response status code")
    headers: dict[str, str] = Field(..., description="Response headers")
    data: Any | None = Field(None, description="Response data")
    content_type: str | None = Field(None, description="Response content type")


class QueryRequest(BaseModel):
    """Request model for query operations."""

    question: str = Field(
        ...,
        description="Question to ask about uploaded documents",
        min_length=1,
        max_length=1000,
    )
    top_k: int | None = Field(
        5, description="Number of documents to retrieve", ge=1, le=20
    )
    filter: dict[str, Any] | None = Field(
        None, description="Filter parameters for documents"
    )
    max_context_length: int | None = Field(
        None, description="Maximum context length for LLM"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"question": "What is this document about?"},
                {"question": "Tell me about the main topics discussed", "top_k": 5},
                {
                    "question": "What are the key findings?",
                    "top_k": 10,
                    "filter": {
                        "content_type": "application/pdf",
                        "file_type": "document",
                    },
                },
            ]
        }
    )


class UploadResponse(BaseModel):
    """Response model for file upload."""

    success: bool = Field(..., description="Upload success status")
    message: str = Field(..., description="User-friendly message")
    file_id: str | None = Field(None, description="File identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "message": "File uploaded successfully",
                    "file_id": "doc_12345",
                    "timestamp": "2025-10-04T18:44:20.054330",
                },
                {
                    "success": False,
                    "message": "File upload failed. Please try again.",
                    "file_id": None,
                    "timestamp": "2025-10-04T18:44:20.054330",
                },
            ]
        }
    )


class QueryResponse(BaseModel):
    """Response model for query operations."""

    success: bool = Field(..., description="Query success status")
    message: str = Field(..., description="User-friendly message")
    answer: str | None = Field(None, description="Query answer")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "message": "Query processed successfully",
                    "answer": "This document discusses machine learning algorithms and their applications in data science.",
                    "timestamp": "2025-10-04T18:44:41.761673",
                },
                {
                    "success": True,
                    "message": "Query processed successfully",
                    "answer": "Found 0 relevant documents for your query.",
                    "timestamp": "2025-10-04T18:44:41.761673",
                },
                {
                    "success": False,
                    "message": "Query failed. Please check your question and try again.",
                    "answer": None,
                    "timestamp": "2025-10-04T18:44:41.761673",
                },
            ]
        }
    )


class ServiceHealthResponse(BaseModel):
    """Response model for service health check."""

    status: str = Field(..., description="Overall service status")
    message: str = Field(..., description="Service readiness message")
    services: dict[str, str] = Field(..., description="Individual service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("timestamp")
    def serialize_timestamp(self, value: datetime) -> str:
        return value.isoformat()

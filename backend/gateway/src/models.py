"""
Pydantic models for API Gateway Service
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, EmailStr, ConfigDict, field_serializer


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
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    is_active: bool = Field(True, description="Whether user is active")
    role: UserRole = Field(UserRole.USER, description="User role")


class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, description="Password")


class UserUpdate(BaseModel):
    """User update model."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None


class User(UserBase):
    """User model with ID."""
    id: int = Field(..., description="User ID")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    @field_serializer('created_at', 'updated_at', 'last_login')
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
    username: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[UserRole] = None


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
    dependencies: Dict[str, bool] = Field(..., description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    error_code: Optional[str] = Field(None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        """Serialize datetime to ISO string."""
        return value.isoformat()


class RateLimitResponse(BaseModel):
    """Rate limit response model."""
    limit: int = Field(..., description="Request limit")
    remaining: int = Field(..., description="Remaining requests")
    reset_time: datetime = Field(..., description="Reset time")
    retry_after: Optional[int] = Field(None, description="Retry after seconds")


class UserListResponse(BaseModel):
    """User list response model."""
    users: List[User] = Field(..., description="List of users")
    total_count: int = Field(..., description="Total number of users")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    has_next: bool = Field(..., description="Whether there are more pages")


class ProxyRequest(BaseModel):
    """Proxy request model."""
    method: str = Field(..., description="HTTP method")
    url: str = Field(..., description="Target URL")
    headers: Optional[Dict[str, str]] = Field(None, description="Request headers")
    data: Optional[Dict[str, Any]] = Field(None, description="Request data")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


class ProxyResponse(BaseModel):
    """Proxy response model."""
    status_code: int = Field(..., description="Response status code")
    headers: Dict[str, str] = Field(..., description="Response headers")
    data: Optional[Any] = Field(None, description="Response data")
    content_type: Optional[str] = Field(None, description="Response content type")

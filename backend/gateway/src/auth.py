"""
Authentication and security utilities for API Gateway Service
"""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import redis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings
from .models import TokenData, UserRole

settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

# JWT security
security = HTTPBearer()

# Redis connection
redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


class AuthManager:
    """Authentication manager."""

    def __init__(self):
        self.secret_key = settings.jwt_secret_key
        self.algorithm = settings.jwt_algorithm
        self.access_token_expire_minutes = settings.jwt_access_token_expire_minutes
        self.refresh_token_expire_days = settings.jwt_refresh_token_expire_days

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password."""
        return pwd_context.hash(password)

    def create_access_token(
        self, data: dict[str, Any], expires_delta: timedelta | None = None
    ) -> str:
        """Create an access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(UTC) + expires_delta
        else:
            expire = datetime.now(UTC) + timedelta(
                minutes=self.access_token_expire_minutes
            )

        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_refresh_token(self, data: dict[str, Any]) -> str:
        """Create a refresh token."""
        to_encode = data.copy()
        expire = datetime.now(UTC) + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode a token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            role: str = payload.get("role")

            if username is None or user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return TokenData(
                username=username,
                user_id=user_id,
                role=UserRole(role) if role else None,
            )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from None


class RateLimiter:
    """Rate limiting manager."""

    def __init__(self):
        self.enabled = settings.rate_limit_enabled
        self.requests_per_minute = settings.rate_limit_requests_per_minute
        self.requests_per_hour = settings.rate_limit_requests_per_hour
        self.requests_per_day = settings.rate_limit_requests_per_day
        self.burst_size = settings.rate_limit_burst_size

    def is_rate_limited(
        self, user_id: int, endpoint: str
    ) -> tuple[bool, dict[str, Any]]:
        """Check if user is rate limited."""
        if not self.enabled:
            return False, {}

        now = datetime.now(UTC)
        minute_key = (
            f"rate_limit:{user_id}:{endpoint}:minute:{now.strftime('%Y-%m-%d-%H-%M')}"
        )
        hour_key = f"rate_limit:{user_id}:{endpoint}:hour:{now.strftime('%Y-%m-%d-%H')}"
        day_key = f"rate_limit:{user_id}:{endpoint}:day:{now.strftime('%Y-%m-%d')}"

        # Check minute limit
        minute_count = redis_client.incr(minute_key)
        if minute_count == 1:
            redis_client.expire(minute_key, 60)

        if minute_count > self.requests_per_minute:
            return True, {
                "limit": self.requests_per_minute,
                "remaining": 0,
                "reset_time": now + timedelta(minutes=1),
                "retry_after": 60,
            }

        # Check hour limit
        hour_count = redis_client.incr(hour_key)
        if hour_count == 1:
            redis_client.expire(hour_key, 3600)

        if hour_count > self.requests_per_hour:
            return True, {
                "limit": self.requests_per_hour,
                "remaining": 0,
                "reset_time": now + timedelta(hours=1),
                "retry_after": 3600,
            }

        # Check day limit
        day_count = redis_client.incr(day_key)
        if day_count == 1:
            redis_client.expire(day_key, 86400)

        if day_count > self.requests_per_day:
            return True, {
                "limit": self.requests_per_day,
                "remaining": 0,
                "reset_time": now + timedelta(days=1),
                "retry_after": 86400,
            }

        return False, {
            "limit": self.requests_per_minute,
            "remaining": self.requests_per_minute - minute_count,
            "reset_time": now + timedelta(minutes=1),
        }


class ServiceProxy:
    """Service proxy for routing requests to backend services."""

    def __init__(self):
        self.services = {
            "ingestion": settings.ingestion_service_url,
            "ocr": settings.ocr_service_url,
            "ner": settings.ner_service_url,
            "embedding": settings.embedding_service_url,
            "query": settings.query_service_url,
        }

    async def proxy_request(
        self,
        service: str,
        path: str,
        method: str = "GET",
        headers: dict[str, str | None] = None,
        data: dict[str, Any | None] = None,
        params: dict[str, Any | None] = None,
    ) -> dict[str, Any]:
        """Proxy a request to a backend service."""
        if service not in self.services:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Service '{service}' not found",
            )

        base_url = self.services[service]
        url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"

        async with httpx.AsyncClient() as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=30.0,
                )

                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "data": (
                        response.json()
                        if response.headers.get("content-type", "").startswith(
                            "application/json"
                        )
                        else response.text
                    ),
                    "content_type": response.headers.get("content-type"),
                }
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Service '{service}' unavailable: {str(e)}",
                ) from e


# Global instances
auth_manager = AuthManager()
rate_limiter = RateLimiter()
service_proxy = ServiceProxy()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    return auth_manager.verify_token(token)


def require_role(required_role: str):
    """Dependency to require specific user role."""

    def role_checker(current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if (
            current_user.role.value != required_role
            and current_user.role != UserRole.ADMIN
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return current_user

    return role_checker


def check_rate_limit(user_id: int, endpoint: str):
    """Check rate limit for user and endpoint."""
    is_limited, limit_info = rate_limiter.is_rate_limited(user_id, endpoint)
    if is_limited:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(limit_info["limit"]),
                "X-RateLimit-Remaining": str(limit_info["remaining"]),
                "X-RateLimit-Reset": limit_info["reset_time"].isoformat(),
                "Retry-After": str(limit_info["retry_after"]),
            },
        )
    return limit_info

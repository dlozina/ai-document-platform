"""
Tests for API Gateway Service authentication and security components
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from src.auth import AuthManager, RateLimiter, ServiceProxy
from src.models import TokenData, UserRole


class TestAuthManager:
    """Test authentication manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auth_manager = AuthManager()

    def test_password_hashing(self):
        """Test password hashing and verification."""
        password = "testpassword123"
        hashed = self.auth_manager.get_password_hash(password)

        assert hashed != password
        assert self.auth_manager.verify_password(password, hashed)
        assert not self.auth_manager.verify_password("wrongpassword", hashed)

    def test_create_access_token(self):
        """Test access token creation."""
        data = {"sub": "testuser", "user_id": 1, "role": "user"}
        token = self.auth_manager.create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_refresh_token(self):
        """Test refresh token creation."""
        data = {"sub": "testuser", "user_id": 1, "role": "user"}
        token = self.auth_manager.create_refresh_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_token_success(self):
        """Test successful token verification."""
        data = {"sub": "testuser", "user_id": 1, "role": "user"}
        token = self.auth_manager.create_access_token(data)

        token_data = self.auth_manager.verify_token(token)

        assert token_data.username == "testuser"
        assert token_data.user_id == 1
        assert token_data.role == UserRole.USER

    def test_verify_token_invalid(self):
        """Test token verification with invalid token."""
        with pytest.raises(HTTPException):  # HTTPException in actual usage
            self.auth_manager.verify_token("invalid_token")

    def test_token_expiration(self):
        """Test token expiration."""
        data = {"sub": "testuser", "user_id": 1, "role": "user"}
        # Create token with very short expiration
        token = self.auth_manager.create_access_token(
            data, expires_delta=timedelta(seconds=-1)
        )

        with pytest.raises(HTTPException):  # HTTPException in actual usage
            self.auth_manager.verify_token(token)


class TestRateLimiter:
    """Test rate limiter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rate_limiter = RateLimiter()

    @patch("src.auth.redis_client")
    def test_rate_limit_check_not_limited(self, mock_redis):
        """Test rate limit check when not limited."""
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        is_limited, limit_info = self.rate_limiter.is_rate_limited(1, "test_endpoint")

        assert not is_limited
        assert "limit" in limit_info
        assert "remaining" in limit_info
        assert "reset_time" in limit_info

    @patch("src.auth.redis_client")
    def test_rate_limit_check_limited(self, mock_redis):
        """Test rate limit check when limited."""
        mock_redis.incr.return_value = 61  # Exceeds minute limit

        is_limited, limit_info = self.rate_limiter.is_rate_limited(1, "test_endpoint")

        assert is_limited
        assert limit_info["remaining"] == 0
        assert "retry_after" in limit_info

    @patch("src.auth.redis_client")
    def test_rate_limit_disabled(self, mock_redis):
        """Test rate limit when disabled."""
        self.rate_limiter.enabled = False

        is_limited, limit_info = self.rate_limiter.is_rate_limited(1, "test_endpoint")

        assert not is_limited
        assert limit_info == {}


class TestServiceProxy:
    """Test service proxy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service_proxy = ServiceProxy()

    def test_service_proxy_initialization(self):
        """Test service proxy initialization."""
        assert "ingestion" in self.service_proxy.services
        assert "ocr" in self.service_proxy.services
        assert "ner" in self.service_proxy.services
        assert "embedding" in self.service_proxy.services
        assert "query" in self.service_proxy.services

    @pytest.mark.asyncio
    async def test_proxy_request_success(self):
        """Test successful proxy request."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.json.return_value = {"message": "success"}

            mock_client.return_value.__aenter__.return_value.request.return_value = (
                mock_response
            )

            result = await self.service_proxy.proxy_request(
                service="ingestion", path="health", method="GET"
            )

            assert result["status_code"] == 200
            assert result["data"]["message"] == "success"

    @pytest.mark.asyncio
    async def test_proxy_request_service_not_found(self):
        """Test proxy request to non-existent service."""
        with pytest.raises(HTTPException):  # HTTPException in actual usage
            await self.service_proxy.proxy_request(
                service="nonexistent", path="health", method="GET"
            )

    @pytest.mark.asyncio
    async def test_proxy_request_network_error(self):
        """Test proxy request with network error."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.request.side_effect = (
                Exception("Network error")
            )

            with pytest.raises(HTTPException):  # HTTPException in actual usage
                await self.service_proxy.proxy_request(
                    service="ingestion", path="health", method="GET"
                )


class TestTokenData:
    """Test token data model."""

    def test_token_data_creation(self):
        """Test token data model creation."""
        token_data = TokenData(username="testuser", user_id=1, role=UserRole.USER)

        assert token_data.username == "testuser"
        assert token_data.user_id == 1
        assert token_data.role == UserRole.USER

    def test_token_data_optional_fields(self):
        """Test token data with optional fields."""
        token_data = TokenData(username="testuser")

        assert token_data.username == "testuser"
        assert token_data.user_id is None
        assert token_data.role is None


class TestUserRole:
    """Test user role enumeration."""

    def test_user_role_values(self):
        """Test user role enumeration values."""
        assert UserRole.USER == "user"
        assert UserRole.ADMIN == "admin"
        assert UserRole.MODERATOR == "moderator"

    def test_user_role_creation(self):
        """Test user role creation from string."""
        role = UserRole("user")
        assert role == UserRole.USER

        role = UserRole("admin")
        assert role == UserRole.ADMIN

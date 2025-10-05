"""
Tests for API Gateway Service
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
        "role": "user",
    }


@pytest.fixture
def admin_user_data():
    """Admin user data for testing."""
    return {
        "username": "admin",
        "email": "admin@example.com",
        "password": "adminpassword123",
        "full_name": "Admin User",
        "role": "admin",
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch("src.auth.redis_client") as mock:
        mock.ping.return_value = True
        mock.incr.return_value = 1
        mock.expire.return_value = True
        yield mock


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    with patch("src.database.user_manager.get_db") as mock:
        mock_session = MagicMock()
        mock.return_value.__next__.return_value = mock_session
        yield mock_session


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, mock_redis):
        """Test health endpoint returns correct status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "redis_connected" in data
        assert "dependencies" in data
        assert data["service"] == "api-gateway-service"
        assert data["version"] == "1.0.0"


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""

    def test_login_success(self, mock_db_session, sample_user_data):
        """Test successful login."""
        # Mock user authentication
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = sample_user_data["username"]
        mock_user.email = sample_user_data["email"]
        mock_user.is_active = True
        mock_user.role = "user"
        mock_user.hashed_password = "$2b$12$test_hash"

        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        with patch(
            "src.database.user_manager.authenticate_user", return_value=mock_user
        ):
            response = client.post(
                "/auth/login",
                json={
                    "username": sample_user_data["username"],
                    "password": sample_user_data["password"],
                },
            )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert "expires_in" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_credentials(self, mock_db_session):
        """Test login with invalid credentials."""
        mock_db_session.query.return_value.filter.return_value.first.return_value = None

        with patch("src.database.user_manager.authenticate_user", return_value=None):
            response = client.post(
                "/auth/login", json={"username": "invalid", "password": "invalid"}
            )

        assert response.status_code == 401
        data = response.json()
        assert "error" in data
        assert "Invalid username or password" in data["error"]

    def test_login_inactive_user(self, mock_db_session, sample_user_data):
        """Test login with inactive user."""
        mock_user = MagicMock()
        mock_user.is_active = False

        mock_db_session.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        with patch(
            "src.database.user_manager.authenticate_user", return_value=mock_user
        ):
            response = client.post(
                "/auth/login",
                json={
                    "username": sample_user_data["username"],
                    "password": sample_user_data["password"],
                },
            )

        assert response.status_code == 401
        data = response.json()
        assert "Account is disabled" in data["error"]

    def test_refresh_token_success(self):
        """Test successful token refresh."""
        # Create a valid refresh token
        with patch("src.auth.auth_manager.verify_token") as mock_verify:
            mock_verify.return_value = MagicMock(
                username="testuser", user_id=1, role=MagicMock(value="user")
            )

            response = client.post(
                "/auth/refresh", json={"refresh_token": "valid_refresh_token"}
            )

        assert response.status_code == 200
        data = response.json()

        assert "access_token" in data
        assert "refresh_token" in data
        assert "token_type" in data
        assert "expires_in" in data

    def test_refresh_token_invalid(self):
        """Test refresh with invalid token."""
        with patch(
            "src.auth.auth_manager.verify_token", side_effect=Exception("Invalid token")
        ):
            response = client.post(
                "/auth/refresh", json={"refresh_token": "invalid_token"}
            )

        assert response.status_code == 401
        data = response.json()
        assert "Invalid refresh token" in data["error"]


class TestUserManagementEndpoints:
    """Test user management endpoints."""

    def test_create_user_success(self, mock_db_session, admin_user_data):
        """Test successful user creation."""
        # Mock admin user for authentication
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            # Mock user creation
            mock_user = MagicMock()
            mock_user.id = 2
            mock_user.username = admin_user_data["username"]
            mock_user.email = admin_user_data["email"]
            mock_user.full_name = admin_user_data["full_name"]
            mock_user.is_active = True
            mock_user.role = admin_user_data["role"]
            mock_user.created_at = "2024-01-01T00:00:00"
            mock_user.updated_at = "2024-01-01T00:00:00"
            mock_user.last_login = None

            with patch("src.database.user_manager.create_user", return_value=mock_user):
                response = client.post("/users", json=admin_user_data)

        assert response.status_code == 200
        data = response.json()

        assert data["username"] == admin_user_data["username"]
        assert data["email"] == admin_user_data["email"]
        assert data["role"] == admin_user_data["role"]
        assert "id" in data

    def test_create_user_duplicate(self, mock_db_session, admin_user_data):
        """Test user creation with duplicate username."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            with patch(
                "src.database.user_manager.create_user",
                side_effect=ValueError("Username or email already exists"),
            ):
                response = client.post("/users", json=admin_user_data)

        assert response.status_code == 400
        data = response.json()
        assert "Username or email already exists" in data["detail"]

    def test_list_users(self, mock_db_session):
        """Test listing users."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            # Mock users
            mock_user = MagicMock()
            mock_user.id = 1
            mock_user.username = "testuser"
            mock_user.email = "test@example.com"
            mock_user.full_name = "Test User"
            mock_user.is_active = True
            mock_user.role = "user"
            mock_user.created_at = "2024-01-01T00:00:00"
            mock_user.updated_at = "2024-01-01T00:00:00"
            mock_user.last_login = None

            with patch("src.database.user_manager.get_users", return_value=[mock_user]):
                with patch("src.database.user_manager.count_users", return_value=1):
                    response = client.get("/users")

        assert response.status_code == 200
        data = response.json()

        assert "users" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data
        assert "has_next" in data
        assert len(data["users"]) == 1

    def test_get_user_by_id(self, mock_db_session):
        """Test getting user by ID."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            # Mock user
            mock_user = MagicMock()
            mock_user.id = 1
            mock_user.username = "testuser"
            mock_user.email = "test@example.com"
            mock_user.full_name = "Test User"
            mock_user.is_active = True
            mock_user.role = "user"
            mock_user.created_at = "2024-01-01T00:00:00"
            mock_user.updated_at = "2024-01-01T00:00:00"
            mock_user.last_login = None

            with patch(
                "src.database.user_manager.get_user_by_id", return_value=mock_user
            ):
                response = client.get("/users/1")

        assert response.status_code == 200
        data = response.json()

        assert data["id"] == 1
        assert data["username"] == "testuser"

    def test_get_user_not_found(self, mock_db_session):
        """Test getting non-existent user."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            with patch("src.database.user_manager.get_user_by_id", return_value=None):
                response = client.get("/users/999")

        assert response.status_code == 404
        data = response.json()
        assert "User not found" in data["detail"]

    def test_update_user(self, mock_db_session):
        """Test updating user."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            # Mock updated user
            mock_user = MagicMock()
            mock_user.id = 1
            mock_user.username = "updateduser"
            mock_user.email = "updated@example.com"
            mock_user.full_name = "Updated User"
            mock_user.is_active = True
            mock_user.role = "user"
            mock_user.created_at = "2024-01-01T00:00:00"
            mock_user.updated_at = "2024-01-01T00:00:00"
            mock_user.last_login = None

            with patch("src.database.user_manager.update_user", return_value=mock_user):
                response = client.put(
                    "/users/1",
                    json={"username": "updateduser", "email": "updated@example.com"},
                )

        assert response.status_code == 200
        data = response.json()

        assert data["username"] == "updateduser"
        assert data["email"] == "updated@example.com"

    def test_delete_user(self, mock_db_session):
        """Test deleting user."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            with patch("src.database.user_manager.delete_user", return_value=True):
                response = client.delete("/users/1")

        assert response.status_code == 200
        data = response.json()
        assert "User deleted successfully" in data["message"]

    def test_delete_user_not_found(self, mock_db_session):
        """Test deleting non-existent user."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="admin", role=MagicMock(value="admin")
            )

            with patch("src.database.user_manager.delete_user", return_value=False):
                response = client.delete("/users/999")

        assert response.status_code == 404
        data = response.json()
        assert "User not found" in data["detail"]


class TestAuthorization:
    """Test authorization and role-based access."""

    def test_unauthorized_access(self):
        """Test accessing protected endpoint without token."""
        response = client.get("/auth/me")

        assert response.status_code == 401
        data = response.json()
        assert "Not authenticated" in data["detail"]

    def test_admin_only_endpoint(self):
        """Test admin-only endpoint access."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="user", role=MagicMock(value="user")
            )

            response = client.get("/users")

        assert response.status_code == 403
        data = response.json()
        assert "Role 'admin' required" in data["detail"]


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_check(self, mock_redis):
        """Test rate limit checking."""
        with patch(
            "src.auth.rate_limiter.is_rate_limited",
            return_value=(
                False,
                {"limit": 60, "remaining": 59, "reset_time": "2024-01-01T00:01:00"},
            ),
        ):
            with patch("src.auth.get_current_user") as mock_auth:
                mock_auth.return_value = MagicMock(
                    user_id=1, username="testuser", role=MagicMock(value="user")
                )

                response = client.get("/auth/me")

        assert response.status_code == 200

    def test_rate_limit_exceeded(self, mock_redis):
        """Test rate limit exceeded."""
        with patch(
            "src.auth.rate_limiter.is_rate_limited",
            return_value=(
                True,
                {
                    "limit": 60,
                    "remaining": 0,
                    "reset_time": "2024-01-01T00:01:00",
                    "retry_after": 60,
                },
            ),
        ):
            with patch("src.auth.get_current_user") as mock_auth:
                mock_auth.return_value = MagicMock(
                    user_id=1, username="testuser", role=MagicMock(value="user")
                )

                response = client.get("/auth/me")

        assert response.status_code == 429
        data = response.json()
        assert "Rate limit exceeded" in data["detail"]


class TestServiceProxy:
    """Test service proxy functionality."""

    def test_proxy_request_success(self, mock_redis):
        """Test successful proxy request."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="testuser", role=MagicMock(value="user")
            )

            with patch(
                "src.auth.service_proxy.proxy_request",
                return_value={
                    "status_code": 200,
                    "headers": {"content-type": "application/json"},
                    "data": {"message": "success"},
                    "content_type": "application/json",
                },
            ):
                response = client.post(
                    "/proxy/ingestion/health", json={"method": "GET", "url": "/health"}
                )

        assert response.status_code == 200
        data = response.json()

        assert data["status_code"] == 200
        assert data["data"]["message"] == "success"

    def test_proxy_request_service_not_found(self, mock_redis):
        """Test proxy request to non-existent service."""
        with patch("src.auth.get_current_user") as mock_auth:
            mock_auth.return_value = MagicMock(
                user_id=1, username="testuser", role=MagicMock(value="user")
            )

            with patch(
                "src.auth.service_proxy.proxy_request",
                side_effect=Exception("Service 'unknown' not found"),
            ):
                response = client.post(
                    "/proxy/unknown/health", json={"method": "GET", "url": "/health"}
                )

        assert response.status_code == 500
        data = response.json()
        assert "Proxy request failed" in data["error"]


class TestErrorHandling:
    """Test error handling."""

    def test_http_exception_handler(self):
        """Test HTTP exception handling."""
        response = client.get("/nonexistent")

        assert response.status_code == 404
        data = response.json()

        assert "error" in data
        assert "detail" in data
        assert "timestamp" in data

    def test_validation_error(self):
        """Test validation error handling."""
        response = client.post(
            "/auth/login",
            json={
                "username": "test",
                # Missing password
            },
        )

        assert response.status_code == 422
        data = response.json()

        assert "detail" in data


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema(self):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        data = response.json()

        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "API Gateway Service"

    def test_docs_endpoint(self):
        """Test docs endpoint is accessible."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self):
        """Test ReDoc endpoint is accessible."""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

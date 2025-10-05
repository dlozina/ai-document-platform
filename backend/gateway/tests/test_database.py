"""
Tests for API Gateway Service database operations
"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import IntegrityError
from src.database import UserManager, UserModel
from src.models import UserCreate, UserRole, UserUpdate


class TestUserManager:
    """Test user manager database operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.user_manager = UserManager()
        self.mock_db = MagicMock()

    def test_get_user_by_id(self):
        """Test getting user by ID."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        result = self.user_manager.get_user_by_id(self.mock_db, 1)

        assert result == mock_user
        self.mock_db.query.assert_called_once_with(UserModel)

    def test_get_user_by_username(self):
        """Test getting user by username."""
        mock_user = MagicMock()
        mock_user.username = "testuser"

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        result = self.user_manager.get_user_by_username(self.mock_db, "testuser")

        assert result == mock_user

    def test_get_user_by_email(self):
        """Test getting user by email."""
        mock_user = MagicMock()
        mock_user.email = "test@example.com"

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        result = self.user_manager.get_user_by_email(self.mock_db, "test@example.com")

        assert result == mock_user

    def test_create_user_success(self):
        """Test successful user creation."""
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            full_name="Test User",
            role=UserRole.USER,
        )

        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.role = "user"
        mock_user.hashed_password = "hashed_password"

        self.mock_db.add.return_value = None
        self.mock_db.commit.return_value = None
        self.mock_db.refresh.return_value = None

        with patch("src.database.auth_manager") as mock_auth:
            mock_auth.get_password_hash.return_value = "hashed_password"

            result = self.user_manager.create_user(self.mock_db, user_data)

        assert result.username == "testuser"
        assert result.email == "test@example.com"
        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
        self.mock_db.refresh.assert_called_once()

    def test_create_user_duplicate(self):
        """Test user creation with duplicate username."""
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            role=UserRole.USER,
        )

        self.mock_db.add.side_effect = IntegrityError("statement", "params", "orig")

        with patch("src.database.auth_manager") as mock_auth:
            mock_auth.get_password_hash.return_value = "hashed_password"

            with pytest.raises(ValueError, match="Username or email already exists"):
                self.user_manager.create_user(self.mock_db, user_data)

        self.mock_db.rollback.assert_called_once()

    def test_update_user_success(self):
        """Test successful user update."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.role = "user"
        mock_user.hashed_password = "hashed_password"

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        user_update = UserUpdate(
            username="updateduser",
            email="updated@example.com",
            full_name="Updated User",
        )

        result = self.user_manager.update_user(self.mock_db, 1, user_update)

        assert result == mock_user
        assert mock_user.username == "updateduser"
        assert mock_user.email == "updated@example.com"
        assert mock_user.full_name == "Updated User"
        self.mock_db.commit.assert_called_once()

    def test_update_user_not_found(self):
        """Test updating non-existent user."""
        self.mock_db.query.return_value.filter.return_value.first.return_value = None

        user_update = UserUpdate(username="updateduser")

        result = self.user_manager.update_user(self.mock_db, 999, user_update)

        assert result is None

    def test_update_user_password(self):
        """Test updating user password."""
        mock_user = MagicMock()
        mock_user.id = 1
        mock_user.username = "testuser"
        mock_user.hashed_password = "old_hash"

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        user_update = UserUpdate(password="newpassword123")

        with patch("src.database.auth_manager") as mock_auth:
            mock_auth.get_password_hash.return_value = "new_hash"

            result = self.user_manager.update_user(self.mock_db, 1, user_update)

        assert result == mock_user
        assert mock_user.hashed_password == "new_hash"

    def test_delete_user_success(self):
        """Test successful user deletion."""
        mock_user = MagicMock()
        mock_user.id = 1

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        result = self.user_manager.delete_user(self.mock_db, 1)

        assert result is True
        self.mock_db.delete.assert_called_once_with(mock_user)
        self.mock_db.commit.assert_called_once()

    def test_delete_user_not_found(self):
        """Test deleting non-existent user."""
        self.mock_db.query.return_value.filter.return_value.first.return_value = None

        result = self.user_manager.delete_user(self.mock_db, 999)

        assert result is False

    def test_authenticate_user_success(self):
        """Test successful user authentication."""
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_user.hashed_password = "hashed_password"
        mock_user.is_active = True

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        with patch("src.database.auth_manager") as mock_auth:
            mock_auth.verify_password.return_value = True

            result = self.user_manager.authenticate_user(
                self.mock_db, "testuser", "password"
            )

        assert result == mock_user
        self.mock_db.commit.assert_called_once()  # For last_login update

    def test_authenticate_user_wrong_password(self):
        """Test user authentication with wrong password."""
        mock_user = MagicMock()
        mock_user.username = "testuser"
        mock_user.hashed_password = "hashed_password"

        self.mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user
        )

        with patch("src.database.auth_manager") as mock_auth:
            mock_auth.verify_password.return_value = False

            result = self.user_manager.authenticate_user(
                self.mock_db, "testuser", "wrongpassword"
            )

        assert result is None

    def test_authenticate_user_not_found(self):
        """Test user authentication with non-existent user."""
        self.mock_db.query.return_value.filter.return_value.first.return_value = None

        result = self.user_manager.authenticate_user(
            self.mock_db, "nonexistent", "password"
        )

        assert result is None

    def test_get_users_pagination(self):
        """Test getting users with pagination."""
        mock_users = [MagicMock(), MagicMock()]

        self.mock_db.query.return_value.offset.return_value.limit.return_value.all.return_value = mock_users

        result = self.user_manager.get_users(self.mock_db, skip=10, limit=20)

        assert result == mock_users
        self.mock_db.query.return_value.offset.assert_called_once_with(10)
        self.mock_db.query.return_value.offset.return_value.limit.assert_called_once_with(
            20
        )

    def test_count_users(self):
        """Test counting users."""
        self.mock_db.query.return_value.count.return_value = 5

        result = self.user_manager.count_users(self.mock_db)

        assert result == 5

    def test_convert_to_user(self):
        """Test converting database user to Pydantic user."""
        mock_db_user = MagicMock()
        mock_db_user.id = 1
        mock_db_user.username = "testuser"
        mock_db_user.email = "test@example.com"
        mock_db_user.full_name = "Test User"
        mock_db_user.is_active = True
        mock_db_user.role = "user"
        mock_db_user.created_at = "2024-01-01T00:00:00"
        mock_db_user.updated_at = "2024-01-01T00:00:00"
        mock_db_user.last_login = None

        result = self.user_manager.convert_to_user(mock_db_user)

        assert result.id == 1
        assert result.username == "testuser"
        assert result.email == "test@example.com"
        assert result.full_name == "Test User"
        assert result.is_active is True
        assert result.role == UserRole.USER

    def test_convert_to_user_in_db(self):
        """Test converting database user to Pydantic user with password."""
        mock_db_user = MagicMock()
        mock_db_user.id = 1
        mock_db_user.username = "testuser"
        mock_db_user.email = "test@example.com"
        mock_db_user.full_name = "Test User"
        mock_db_user.is_active = True
        mock_db_user.role = "user"
        mock_db_user.hashed_password = "hashed_password"
        mock_db_user.created_at = "2024-01-01T00:00:00"
        mock_db_user.updated_at = "2024-01-01T00:00:00"
        mock_db_user.last_login = None

        result = self.user_manager.convert_to_user_in_db(mock_db_user)

        assert result.id == 1
        assert result.username == "testuser"
        assert result.email == "test@example.com"
        assert result.hashed_password == "hashed_password"
        assert result.role == UserRole.USER

"""
Database models and operations for API Gateway Service
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import User, UserCreate, UserInDB, UserRole, UserUpdate

settings = get_settings()

# Database setup
SQLALCHEMY_DATABASE_URL = settings.database_url
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserModel(Base):
    """User database model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(100), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    role = Column(String(20), default="user")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


# Create tables
Base.metadata.create_all(bind=engine)


def create_default_admin():
    """Create default admin user if no admin users exist."""
    db = SessionLocal()
    try:
        # Check if any admin users exist
        admin_count = db.query(UserModel).filter(UserModel.role == "admin").count()

        if admin_count == 0:
            # Import here to avoid circular imports
            from .auth import auth_manager

            # Create default admin user
            default_admin = UserModel(
                username="admin",
                email="admin@example.com",
                full_name="Default Admin",
                hashed_password=auth_manager.get_password_hash("admin123"),
                is_active=True,
                role="admin",
            )

            db.add(default_admin)
            db.commit()
            print("Default admin user created: username='admin', password='admin123'")
        else:
            print(f"Admin users already exist ({admin_count} found)")

    except Exception as e:
        print(f"Error creating default admin: {e}")
        db.rollback()
    finally:
        db.close()


# Create default admin on startup
create_default_admin()


class UserManager:
    """User management operations."""

    def __init__(self):
        self.auth_manager = None  # Will be set after auth module is imported

    def get_db(self) -> Session:
        """Get database session."""
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    def get_user_by_id(self, db: Session, user_id: int) -> UserModel | None:
        """Get user by ID."""
        return db.query(UserModel).filter(UserModel.id == user_id).first()

    def get_user_by_username(self, db: Session, username: str) -> UserModel | None:
        """Get user by username."""
        return db.query(UserModel).filter(UserModel.username == username).first()

    def get_user_by_email(self, db: Session, email: str) -> UserModel | None:
        """Get user by email."""
        return db.query(UserModel).filter(UserModel.email == email).first()

    def create_user(self, db: Session, user: UserCreate) -> UserModel:
        """Create a new user."""
        # Import here to avoid circular imports
        from .auth import auth_manager

        hashed_password = auth_manager.get_password_hash(user.password)
        db_user = UserModel(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            hashed_password=hashed_password,
            is_active=user.is_active,
            role=user.role.value,
        )

        try:
            db.add(db_user)
            db.commit()
            db.refresh(db_user)
            return db_user
        except IntegrityError as e:
            db.rollback()
            raise ValueError("Username or email already exists") from e

    def update_user(
        self, db: Session, user_id: int, user_update: UserUpdate
    ) -> UserModel | None:
        """Update user information."""
        db_user = self.get_user_by_id(db, user_id)
        if not db_user:
            return None

        update_data = user_update.model_dump(exclude_unset=True)

        # Handle password update separately
        if "password" in update_data:
            from .auth import auth_manager

            update_data["hashed_password"] = auth_manager.get_password_hash(
                update_data.pop("password")
            )

        for field, value in update_data.items():
            if field == "role" and isinstance(value, UserRole):
                value = value.value
            setattr(db_user, field, value)

        db_user.updated_at = datetime.utcnow()

        try:
            db.commit()
            db.refresh(db_user)
            return db_user
        except IntegrityError as e:
            db.rollback()
            raise ValueError("Username or email already exists") from e

    def delete_user(self, db: Session, user_id: int) -> bool:
        """Delete a user."""
        db_user = self.get_user_by_id(db, user_id)
        if not db_user:
            return False

        db.delete(db_user)
        db.commit()
        return True

    def authenticate_user(
        self, db: Session, username: str, password: str
    ) -> UserModel | None:
        """Authenticate a user."""
        # Import here to avoid circular imports
        from .auth import auth_manager

        user = self.get_user_by_username(db, username)
        if not user:
            return None

        if not auth_manager.verify_password(password, user.hashed_password):
            return None

        # Update last login
        user.last_login = datetime.utcnow()
        db.commit()

        return user

    def get_users(
        self, db: Session, skip: int = 0, limit: int = 100
    ) -> list[UserModel]:
        """Get list of users with pagination."""
        return db.query(UserModel).offset(skip).limit(limit).all()

    def count_users(self, db: Session) -> int:
        """Count total users."""
        return db.query(UserModel).count()

    def convert_to_user(self, db_user: UserModel) -> User:
        """Convert database user to Pydantic user model."""
        return User(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            full_name=db_user.full_name,
            is_active=db_user.is_active,
            role=UserRole(db_user.role),
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
            last_login=db_user.last_login,
        )

    def convert_to_user_in_db(self, db_user: UserModel) -> UserInDB:
        """Convert database user to Pydantic user model with password."""
        return UserInDB(
            id=db_user.id,
            username=db_user.username,
            email=db_user.email,
            full_name=db_user.full_name,
            is_active=db_user.is_active,
            role=UserRole(db_user.role),
            hashed_password=db_user.hashed_password,
            created_at=db_user.created_at,
            updated_at=db_user.updated_at,
            last_login=db_user.last_login,
        )


# Global user manager instance
user_manager = UserManager()


def get_db():
    """Get database session."""
    return next(user_manager.get_db())

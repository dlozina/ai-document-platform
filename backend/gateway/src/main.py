"""
API Gateway Service

FastAPI application that provides authentication, JWT tokens, and per-user rate limiting.
Routes requests to backend services with proper authentication and authorization.
"""

import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from sqlalchemy.orm import Session

from .auth import (
    TokenData,
    auth_manager,
    check_rate_limit,
    get_current_user,
    require_role,
    service_proxy,
)
from .config import get_settings
from .database import user_manager
from .models import (
    ErrorResponse,
    LoginRequest,
    PasswordChangeRequest,
    ProxyRequest,
    ProxyResponse,
    QueryRequest,
    QueryResponse,
    RefreshTokenRequest,
    ServiceHealthResponse,
    Token,
    UploadResponse,
    User,
    UserCreate,
    UserListResponse,
    UserRole,
    UserUpdate,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
settings = get_settings()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "gateway_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code", "user_id"],
)

REQUEST_DURATION = Histogram(
    "gateway_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint", "user_id"],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

AUTH_COUNT = Counter(
    "gateway_auth_attempts_total", "Total authentication attempts", ["result", "method"]
)

RATE_LIMIT_COUNT = Counter(
    "gateway_rate_limit_hits_total", "Total rate limit hits", ["user_id", "endpoint"]
)

SERVICE_HEALTH = Gauge(
    "gateway_service_health_status",
    "Service health status (1=healthy, 0=unhealthy)",
    ["component"],
)

SERVICE_UPTIME = Gauge("gateway_service_uptime_seconds", "Service uptime in seconds")

# Track service start time
SERVICE_START_TIME = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global SERVICE_START_TIME
    SERVICE_START_TIME = time.time()

    logger.info("Starting API Gateway Service...")

    # Test Redis connection
    try:
        import redis

        redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection successful")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    logger.info("API Gateway Service ready")

    yield

    logger.info("Shutting down API Gateway Service...")


# Create FastAPI app
app = FastAPI(
    title="API Gateway Service",
    description="""
    ## API Gateway with Authentication and Rate Limiting

    Production-ready API gateway service providing authentication, authorization, and proxy functionality for the document processing system.

        ### User Access Levels:

        **🔓 Public Endpoints** - No authentication required
        - User registration

        **👤 User Endpoints** - Requires user authentication
        - File upload
        - Question answering (QA)
        - Personal account management

        **👑 Admin Endpoints** - Requires admin authentication
        - User management
        - System administration
        - Advanced operations

    ### Authentication:
    All protected endpoints require a valid JWT token in the Authorization header:
    ```
    Authorization: Bearer <your-jwt-token>
    ```

        ### Rate Limiting:
        - **Upload**: 10 requests/minute
        - **QA**: 60 requests/minute
        - **General**: 100 requests/minute
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "API Gateway Service Team",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {"url": "http://localhost:8005", "description": "Development server"},
        {"url": "https://api.example.com", "description": "Production server"},
    ],
    tags_metadata=[
        {
            "name": "Authentication",
            "description": "User authentication and token management",
            "externalDocs": {
                "description": "JWT Authentication Guide",
                "url": "https://jwt.io/introduction",
            },
        },
        {
            "name": "User Registration",
            "description": "Public user registration endpoints",
        },
        {
            "name": "User Operations",
            "description": "Endpoints available to authenticated users",
        },
        {
            "name": "Admin Operations",
            "description": "Endpoints available only to admin users",
        },
        {
            "name": "System",
            "description": "System health and monitoring endpoints",
        },
    ],
)

# Add CORS middleware
if settings.enable_cors:
    logger.info(f"CORS enabled with origins: {settings.cors_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    logger.info("CORS disabled")


# Metrics middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect HTTP request metrics."""
    start_time = time.time()

    # Extract user ID from JWT token if present
    user_id = "anonymous"
    try:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            token_data = auth_manager.verify_token(token)
            user_id = str(token_data.user_id)
    except Exception:  # nosec B110 - Intentionally ignore auth errors in middleware
        pass  # Ignore auth errors in middleware

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
        user_id=user_id,
    ).inc()

    REQUEST_DURATION.labels(
        method=request.method, endpoint=request.url.path, user_id=user_id
    ).observe(duration)

    return response


# Dependency functions
def get_db_session():
    """Get database session."""
    return next(user_manager.get_db())


# API Endpoints


@app.get("/health", response_model=ServiceHealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the current health status and service readiness.
    """
    try:
        # Check Redis connection
        try:
            import redis

            redis_client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True
            )
            redis_client.ping()
        except Exception:  # nosec B110 - Intentionally ignore Redis connection errors
            pass

        # Check backend services
        services = {}
        upload_ready = False
        query_ready = False

        try:
            async with httpx.AsyncClient() as client:
                # Check ingestion service
                response = await client.get(
                    f"{settings.ingestion_service_url}/health",
                    headers={
                        "X-Tenant-ID": "system"
                    },  # System tenant for health checks
                    timeout=5.0,
                )
                services["ingestion_service"] = (
                    "healthy" if response.status_code == 200 else "unhealthy"
                )
                upload_ready = response.status_code == 200
        except Exception:
            services["ingestion_service"] = "unhealthy"

        try:
            async with httpx.AsyncClient() as client:
                # Check query service
                response = await client.get(
                    f"{settings.query_service_url}/health",
                    headers={
                        "X-Tenant-ID": "system"
                    },  # System tenant for health checks
                    timeout=5.0,
                )
                services["query_service"] = (
                    "healthy" if response.status_code == 200 else "unhealthy"
                )
                query_ready = response.status_code == 200
        except Exception:
            services["query_service"] = "unhealthy"

        # Determine overall status and message
        if upload_ready and query_ready:
            health_status = "healthy"
            message = "Service is ready for uploading and question answering"
        elif upload_ready:
            health_status = "partial"
            message = "Service is ready for uploading (QA service unavailable)"
        elif query_ready:
            health_status = "partial"
            message = (
                "Service is ready for question answering (upload service unavailable)"
            )
        else:
            health_status = "unhealthy"
            message = "Service is not ready - both upload and QA services unavailable"

        return ServiceHealthResponse(
            status=health_status, message=message, services=services
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service unhealthy"
        ) from e


@app.get("/metrics", tags=["System"])
async def get_metrics():
    """
    Prometheus metrics endpoint for service monitoring.

    Returns comprehensive metrics in Prometheus text format.
    """
    try:
        # Update uptime metric
        if SERVICE_START_TIME is not None:
            SERVICE_UPTIME.set(time.time() - SERVICE_START_TIME)
        else:
            SERVICE_UPTIME.set(0)

        # Update health metrics
        try:
            import redis

            redis_client = redis.Redis.from_url(
                settings.redis_url, decode_responses=True
            )
            redis_client.ping()
            SERVICE_HEALTH.labels(component="redis").set(1)
        except Exception:
            SERVICE_HEALTH.labels(component="redis").set(0)

        SERVICE_HEALTH.labels(component="database").set(1)  # SQLite is always available

        # Generate all metrics
        metrics_data = generate_latest()

        return Response(metrics_data, media_type=CONTENT_TYPE_LATEST)

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get metrics: {str(e)}"
        ) from e


# Authentication endpoints


@app.post("/auth/login", response_model=Token, tags=["Authentication"])
async def login(login_request: LoginRequest, db: Session = Depends(get_db_session)):
    """
    Login endpoint.

    Authenticate user and return JWT tokens.
    """
    try:
        # Try username first, then email
        user = user_manager.get_user_by_username(db, login_request.username)
        if not user:
            user = user_manager.get_user_by_email(db, login_request.username)

        if not user or not user_manager.authenticate_user(
            db, user.username, login_request.password
        ):
            AUTH_COUNT.labels(result="failed", method="password").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not user.is_active:
            AUTH_COUNT.labels(result="failed", method="inactive").inc()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is disabled"
            )

        # Create tokens
        token_data = {"sub": user.username, "user_id": user.id, "role": user.role}

        access_token = auth_manager.create_access_token(token_data)
        refresh_token = auth_manager.create_refresh_token(token_data)

        AUTH_COUNT.labels(result="success", method="password").inc()

        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",  # nosec B106 - Standard OAuth2 token type
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        ) from e


@app.post("/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Refresh access token.

    Use refresh token to get new access token.
    """
    try:
        token_data = auth_manager.verify_token(refresh_request.refresh_token)

        # Create new access token
        new_token_data = {
            "sub": token_data.username,
            "user_id": token_data.user_id,
            "role": token_data.role.value if token_data.role else "user",
        }

        access_token = auth_manager.create_access_token(new_token_data)

        return Token(
            access_token=access_token,
            refresh_token=refresh_request.refresh_token,  # Keep same refresh token
            token_type="bearer",  # nosec B106 - Standard OAuth2 token type
            expires_in=settings.jwt_access_token_expire_minutes * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        ) from e


@app.get("/auth/me", response_model=User, tags=["Authentication"])
async def get_current_user_info(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Get current user information.

    Returns information about the authenticated user.
    """
    user = user_manager.get_user_by_id(db, current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return user_manager.convert_to_user(user)


@app.post("/auth/change-password", tags=["Authentication"])
async def change_password(
    password_request: PasswordChangeRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Change user password.

    Change password for the authenticated user.
    """
    user = user_manager.get_user_by_id(db, current_user.user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    # Verify current password
    if not auth_manager.verify_password(
        password_request.current_password, user.hashed_password
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    # Update password
    user_update = UserUpdate(password=password_request.new_password)
    user_manager.update_user(db, current_user.user_id, user_update)

    return {"message": "Password changed successfully"}


# User management endpoints


@app.post("/register", response_model=User, tags=["User Registration"])
async def register_user(user: UserCreate, db: Session = Depends(get_db_session)):
    """
    Register a new regular user.

    Public endpoint - anyone can create a user account with role 'user'.
    """
    # Force role to be 'user' for public registration
    user.role = UserRole.USER

    try:
        db_user = user_manager.create_user(db, user)
        return user_manager.convert_to_user(db_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@app.post("/admin/users", response_model=User, tags=["Admin Operations"])
async def create_admin_user(
    user: UserCreate,
    current_user: TokenData = Depends(require_role("admin")),
    db: Session = Depends(get_db_session),
):
    """
    Create a new admin user.

    Admin-only endpoint to create users with admin privileges.
    """
    # Ensure the user being created is an admin
    if user.role != UserRole.ADMIN:
        user.role = UserRole.ADMIN

    try:
        db_user = user_manager.create_user(db, user)
        return user_manager.convert_to_user(db_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@app.get("/users", response_model=UserListResponse, tags=["Admin Operations"])
async def list_users(
    page: int = 1,
    page_size: int = 20,
    current_user: TokenData = Depends(require_role("admin")),
    db: Session = Depends(get_db_session),
):
    """
    List users with pagination.

    Admin-only endpoint to list all users.
    """
    skip = (page - 1) * page_size
    users = user_manager.get_users(db, skip=skip, limit=page_size)
    total_count = user_manager.count_users(db)

    user_list = [user_manager.convert_to_user(user) for user in users]

    return UserListResponse(
        users=user_list,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next=(page * page_size) < total_count,
    )


@app.get("/users/{user_id}", response_model=User, tags=["Admin Operations"])
async def get_user(
    user_id: int,
    current_user: TokenData = Depends(require_role("admin")),
    db: Session = Depends(get_db_session),
):
    """
    Get user by ID.

    Admin-only endpoint to get user details.
    """
    user = user_manager.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return user_manager.convert_to_user(user)


@app.put("/users/{user_id}", response_model=User, tags=["Admin Operations"])
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: TokenData = Depends(require_role("admin")),
    db: Session = Depends(get_db_session),
):
    """
    Update user information.

    Admin-only endpoint to update user details.
    """
    try:
        updated_user = user_manager.update_user(db, user_id, user_update)
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return user_manager.convert_to_user(updated_user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@app.delete("/users/{user_id}", tags=["Admin Operations"])
async def delete_user(
    user_id: int,
    current_user: TokenData = Depends(require_role("admin")),
    db: Session = Depends(get_db_session),
):
    """
    Delete a user.

    Admin-only endpoint to delete users.
    """
    success = user_manager.delete_user(db, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {"message": "User deleted successfully"}


# Service proxy endpoints


@app.post("/api/upload", response_model=UploadResponse, tags=["User Operations"])
async def upload_file(
    file: UploadFile = File(..., description="Document file to upload (PDF or images)"),
    current_user: TokenData = Depends(get_current_user),
):
    """
    Upload a file for processing.

    Upload documents to be processed and made searchable. The file will be:
    - Stored securely
    - Processed through OCR (for images/scanned PDFs)
    - Extracted entities using NER
    - Generated embeddings for search

    **Supported File Types:**
    - **PDFs**: `.pdf` (both searchable and scanned)
    - **Images**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.gif`

    **File Size Limit**: 10MB per file

    **Rate Limit**: 10 uploads per minute per user

    **Authentication**: Requires valid JWT token
    """
    try:
        # Check upload rate limit
        check_rate_limit(current_user.user_id, "upload")

        # Prepare file data
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}

        # Proxy to ingestion service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.ingestion_service_url}/upload",
                files=files,
                headers={
                    "X-Tenant-ID": str(current_user.user_id),  # Use user ID as tenant
                    "X-User-ID": str(current_user.user_id),
                    "X-User-Name": current_user.username,
                    "X-User-Role": (
                        current_user.role.value if current_user.role else "user"
                    ),
                },
                timeout=30.0,
            )

        if response.status_code == 200:
            response_data = response.json()
            return UploadResponse(
                success=True,
                message="File uploaded successfully",
                file_id=response_data.get("file_id"),
            )
        else:
            return UploadResponse(
                success=False, message="File upload failed. Please try again."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return UploadResponse(
            success=False,
            message="Upload service temporarily unavailable. Please try again later.",
        )


@app.post("/api/qa", response_model=QueryResponse, tags=["User Operations"])
async def ask_question(
    request: QueryRequest, current_user: TokenData = Depends(get_current_user)
):
    """
    Ask questions about your uploaded documents.

    Get answers to questions about your uploaded documents using AI-powered question answering.

    **Parameters:**
    - **question**: Your question about the documents (required)
    - **top_k**: Number of documents to consider (default: 5, max: 20)
    - **filter**: Filter documents by metadata (optional)
    - **max_context_length**: Maximum context length for AI processing (optional)

    **Rate Limit**: 60 queries per minute per user

    **Authentication**: Requires valid JWT token

    **Example Questions:**
    - "What is this document about?"
    - "Summarize the main points"
    - "What are the key findings?"
    - "Tell me about the methodology used"
    """
    try:
        # Check QA rate limit
        check_rate_limit(current_user.user_id, "qa")

        # Proxy to query service QA endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.query_service_url}/qa",
                json={
                    "question": request.question,
                    "mode": "rag",  # Use RAG mode for QA
                    "top_k": request.top_k,
                    "filter": request.filter,
                    "max_context_length": request.max_context_length,
                },
                headers={
                    "X-Tenant-ID": str(current_user.user_id),  # Use user ID as tenant
                    "X-User-ID": str(current_user.user_id),
                    "X-User-Name": current_user.username,
                    "X-User-Role": (
                        current_user.role.value if current_user.role else "user"
                    ),
                },
                timeout=30.0,
            )

        if response.status_code == 200:
            response_data = response.json()
            return QueryResponse(
                success=True,
                message="Query processed successfully",
                answer=response_data.get("answer"),
            )
        else:
            return QueryResponse(
                success=False,
                message="Query failed. Please check your question and try again.",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return QueryResponse(
            success=False,
            message="Query service temporarily unavailable. Please try again later.",
        )


@app.post(
    "/proxy/{service}/{path:path}",
    response_model=ProxyResponse,
    tags=["Admin Operations"],
)
async def proxy_request(
    service: str,
    path: str,
    request: ProxyRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Proxy request to backend service.

    Forwards authenticated requests to the appropriate backend service.
    Admin-only endpoint for advanced operations.
    """
    # Check rate limit
    check_rate_limit(current_user.user_id, f"proxy:{service}")

    # Add user context to headers
    headers = request.headers or {}
    headers.update(
        {
            "X-User-ID": str(current_user.user_id),
            "X-User-Name": current_user.username,
            "X-User-Role": current_user.role.value if current_user.role else "user",
        }
    )

    try:
        result = await service_proxy.proxy_request(
            service=service,
            path=path,
            method=request.method,
            headers=headers,
            data=request.data,
            params=request.params,
        )

        return ProxyResponse(**result)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Proxy error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Proxy request failed",
        ) from e


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, detail=str(exc)).model_dump(
            mode="json"
        ),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error", detail=str(exc)
        ).model_dump(mode="json"),
    )


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower(),
    )

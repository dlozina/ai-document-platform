# API Gateway Service

## Overview

The API Gateway Service provides authentication, JWT token management, and per-user rate limiting for the microservices architecture. It acts as a single entry point for all client requests, handling authentication and routing to backend services.

## Features

- **JWT Authentication**: Access and refresh token management
- **Per-user Rate Limiting**: Redis-backed rate limiting with configurable limits
- **Service Proxying**: Route authenticated requests to backend services
- **User Management**: CRUD operations for users with role-based access control
- **Monitoring**: Health checks and Prometheus metrics
- **Security**: Password hashing, CORS support, and input validation

## Quick Start

### Using Docker Compose

```bash
# Start the service with dependencies
docker-compose up -d

# Check health
curl http://localhost:8005/health

# View API documentation
open http://localhost:8005/docs
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp env.example .env
# Edit .env with your configuration

# Run the service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8005 --reload
```

## API Endpoints

### Authentication
- `POST /auth/login` - Login with username/email and password
- `POST /auth/refresh` - Refresh access token
- `GET /auth/me` - Get current user information
- `POST /auth/change-password` - Change user password

### User Registration & Management
- `POST /register` - **Public registration** - Create regular user account (no auth required)
- `POST /admin/users` - Create admin user (admin only)
- `GET /users` - List users with pagination (admin only)
- `GET /users/{user_id}` - Get user by ID (admin only)
- `PUT /users/{user_id}` - Update user (admin only)
- `DELETE /users/{user_id}` - Delete user (admin only)

### Service Proxy
- `POST /proxy/{service}/{path}` - Proxy request to backend service

### Monitoring
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Quick Start

### Default Admin User
The service automatically creates a default admin user on first startup:
- **Username**: `admin`
- **Password**: `admin123`
- **Email**: `admin@example.com`

### Public User Registration
Anyone can create a regular user account using the `/register` endpoint:
```bash
curl -X POST "http://localhost:8005/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newuser",
    "email": "user@example.com",
    "full_name": "New User",
    "password": "password123"
  }'
```

**Response:**
```json
{
  "username": "newuser",
  "email": "user@example.com",
  "full_name": "New User",
  "is_active": true,
  "role": "user",
  "id": 2,
  "created_at": "2025-10-04T18:16:21.199010",
  "updated_at": "2025-10-04T18:16:21.199015",
  "last_login": null
}
```

### Admin Capabilities
Admin users have access to additional endpoints:

#### 1. Create Admin Users
```bash
# First login as admin
TOKEN=$(curl -s -X POST "http://localhost:8005/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

# Create new admin user
curl -X POST "http://localhost:8005/admin/users" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "newadmin",
    "email": "admin2@example.com",
    "full_name": "New Admin",
    "password": "adminpass123"
  }'
```

#### 2. List All Users
```bash
curl -X GET "http://localhost:8005/users?page=1&page_size=10" \
  -H "Authorization: Bearer $TOKEN"
```

#### 3. Get User Details
```bash
curl -X GET "http://localhost:8005/users/2" \
  -H "Authorization: Bearer $TOKEN"
```

#### 4. Update User Information
```bash
curl -X PUT "http://localhost:8005/users/2" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "full_name": "Updated Name",
    "is_active": false
  }'
```

#### 5. Delete User
```bash
curl -X DELETE "http://localhost:8005/users/2" \
  -H "Authorization: Bearer $TOKEN"
```

### Security Features
- **Role Enforcement**: Public registration always creates "user" role, even if "admin" is specified
- **Admin Protection**: Only existing admins can create new admin users
- **Authentication**: All admin operations require valid JWT tokens
- **Password Security**: All passwords are properly hashed with bcrypt

## Configuration

Key environment variables:

- `JWT_SECRET_KEY` - Secret key for JWT tokens
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` - Access token expiration (default: 30)
- `JWT_REFRESH_TOKEN_EXPIRE_DAYS` - Refresh token expiration (default: 7)
- `RATE_LIMIT_REQUESTS_PER_MINUTE` - Rate limit per minute (default: 60)
- `REDIS_URL` - Redis connection URL for rate limiting
- `INGESTION_SERVICE_URL` - Backend ingestion service URL
- `OCR_SERVICE_URL` - Backend OCR service URL
- `NER_SERVICE_URL` - Backend NER service URL
- `EMBEDDING_SERVICE_URL` - Backend embedding service URL
- `QUERY_SERVICE_URL` - Backend query service URL

## Rate Limiting

The service implements per-user rate limiting with the following default limits:
- 60 requests per minute
- 1000 requests per hour
- 10000 requests per day

Rate limit information is included in response headers:
- `X-RateLimit-Limit` - Request limit
- `X-RateLimit-Remaining` - Remaining requests
- `X-RateLimit-Reset` - Reset time
- `Retry-After` - Seconds to wait when rate limited

## Security

- Passwords are hashed using bcrypt
- JWT tokens are signed with HS256 algorithm
- CORS is configurable
- Input validation with Pydantic models
- Role-based access control (admin, moderator, user)

## Monitoring

The service exposes Prometheus metrics at `/metrics`:
- HTTP request counts and durations
- Authentication attempts
- Rate limit hits
- Service health status
- Service uptime

## Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html
```

## Development

The service follows the same patterns as other services in the project:
- FastAPI with Pydantic models
- SQLAlchemy for database operations
- Redis for rate limiting and caching
- Comprehensive error handling
- Structured logging
- Docker support

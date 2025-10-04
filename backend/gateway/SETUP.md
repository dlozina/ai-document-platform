# API Gateway Service Setup Guide

## Prerequisites

- Python 3.12+
- Redis server
- Docker and Docker Compose (optional)

## Environment Setup

1. **Copy environment file**:
   ```bash
   cp env.example .env
   ```

2. **Configure environment variables**:
   - Set `JWT_SECRET_KEY` to a secure random string
   - Configure Redis connection details
   - Set backend service URLs
   - Adjust rate limiting settings as needed

## Installation

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# Check service health
curl http://localhost:8005/health

# View logs
docker-compose logs -f api-gateway-service
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the service
python -m uvicorn src.main:app --host 0.0.0.0 --port 8005 --reload
```

## Initial Setup

1. **Create admin user** (first time only):
   ```bash
   # Use the API to create an admin user
   curl -X POST "http://localhost:8005/users" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "admin",
       "email": "admin@example.com",
       "password": "admin123",
       "role": "admin"
     }'
   ```

2. **Login to get tokens**:
   ```bash
   curl -X POST "http://localhost:8005/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "admin",
       "password": "admin123"
     }'
   ```

## Testing the Service

### Health Check
```bash
curl http://localhost:8005/health
```

### Authentication Flow
```bash
# 1. Login
TOKEN=$(curl -s -X POST "http://localhost:8005/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

# 2. Use token for authenticated requests
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8005/auth/me
```

### Rate Limiting Test
```bash
# Make multiple requests to test rate limiting
for i in {1..70}; do
  curl -H "Authorization: Bearer $TOKEN" \
    http://localhost:8005/auth/me
done
```

## Configuration Options

### JWT Settings
- `JWT_SECRET_KEY`: Secret key for signing tokens
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`: Access token lifetime
- `JWT_REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token lifetime

### Rate Limiting
- `RATE_LIMIT_ENABLED`: Enable/disable rate limiting
- `RATE_LIMIT_REQUESTS_PER_MINUTE`: Requests per minute limit
- `RATE_LIMIT_REQUESTS_PER_HOUR`: Requests per hour limit
- `RATE_LIMIT_REQUESTS_PER_DAY`: Requests per day limit

### Backend Services
Configure URLs for backend services:
- `INGESTION_SERVICE_URL`
- `OCR_SERVICE_URL`
- `NER_SERVICE_URL`
- `EMBEDDING_SERVICE_URL`
- `QUERY_SERVICE_URL`

## Monitoring

### Metrics
Access Prometheus metrics at:
```
http://localhost:8005/metrics
```

### Health Checks
Monitor service health at:
```
http://localhost:8005/health
```

## Troubleshooting

### Common Issues

1. **Redis Connection Failed**
   - Ensure Redis is running
   - Check Redis URL configuration
   - Verify Redis is accessible from the service

2. **JWT Token Issues**
   - Verify JWT_SECRET_KEY is set
   - Check token expiration settings
   - Ensure tokens are properly formatted

3. **Rate Limiting Not Working**
   - Check Redis connection
   - Verify rate limiting is enabled
   - Check rate limit configuration

### Logs
```bash
# Docker Compose logs
docker-compose logs -f api-gateway-service

# Local development logs
# Logs are printed to stdout with INFO level
```

## Security Considerations

1. **JWT Secret Key**: Use a strong, random secret key in production
2. **Password Policy**: Configure strong password requirements
3. **Rate Limiting**: Adjust limits based on your use case
4. **CORS**: Configure CORS origins appropriately
5. **HTTPS**: Use HTTPS in production environments

## Production Deployment

1. **Environment Variables**: Set all required environment variables
2. **Database**: Consider using PostgreSQL instead of SQLite for production
3. **Redis**: Use a managed Redis service for production
4. **Monitoring**: Set up proper monitoring and alerting
5. **Security**: Review and harden security settings
6. **Scaling**: Consider horizontal scaling with load balancers

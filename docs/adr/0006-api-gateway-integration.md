# ADR-006: API Gateway Integration

## Status
Accepted

## Context
As the system grew with multiple microservices, we needed a central entry point that could handle authentication, rate limiting, and request routing. Direct client access to individual services was becoming unwieldy and insecure.

## Decision
We will implement an **API Gateway** as the central entry point for all client requests.

### Gateway Responsibilities
1. **Authentication & Authorization**
   - JWT token validation
   - User role-based access control
   - Session management

2. **Rate Limiting**
   - Per-user rate limiting
   - Different limits for different endpoints
   - Redis-backed rate limiting

3. **Request Routing**
   - Route requests to appropriate backend services
   - Load balancing across service instances
   - Service discovery

4. **Security**
   - CORS handling
   - Request validation
   - Security headers

### Gateway Endpoints
- `/auth/*` - Authentication endpoints
- `/api/upload` - File upload proxy to ingestion service
- `/api/qa` - Question answering proxy to query service
- `/users/*` - User management (admin only)

## Consequences

### Positive
- **Centralized Security**: Single point for authentication and authorization
- **Rate Limiting**: Prevents abuse and ensures fair usage
- **Simplified Client**: Clients only need to know about the gateway
- **Service Abstraction**: Backend services can change without affecting clients
- **Monitoring**: Centralized request monitoring and logging

### Negative
- **Single Point of Failure**: Gateway becomes critical infrastructure
- **Additional Latency**: Extra hop through gateway
- **Complexity**: Additional service to maintain and monitor
- **Scaling**: Gateway needs to scale with client load

## Implementation Timeline
- **2025-10-04**: Added API Gateway service
- **2025-10-04**: Implemented authentication and routing
- **2025-10-04**: Added proxy for user endpoints

## Related Decisions
- ADR-001: Microservices Architecture
- ADR-002: Processing Pipeline Architecture

#!/bin/bash

# API Gateway Service Test Script
# This script tests the basic functionality of the API Gateway Service

echo "🚀 Testing API Gateway Service..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Function to test endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local expected_status=$3
    local description=$4
    local data=$5
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "%{http_code}" -X $method "http://localhost:8005$endpoint" \
            -H "Content-Type: application/json" \
            -d "$data")
    else
        response=$(curl -s -w "%{http_code}" -X $method "http://localhost:8005$endpoint")
    fi
    
    status_code="${response: -3}"
    body="${response%???}"
    
    if [ "$status_code" = "$expected_status" ]; then
        print_status 0 "$description"
        return 0
    else
        print_status 1 "$description (Expected: $expected_status, Got: $status_code)"
        echo "Response: $body"
        return 1
    fi
}

# Check if service is running
echo -e "${YELLOW}Checking if service is running...${NC}"
if curl -s http://localhost:8005/health > /dev/null; then
    print_status 0 "Service is running"
else
    print_status 1 "Service is not running. Please start the service first."
    echo "Run: docker-compose up -d"
    exit 1
fi

echo ""
echo -e "${YELLOW}Running API Gateway Service Tests...${NC}"
echo ""

# Test 1: Health Check
test_endpoint "GET" "/health" "200" "Health check endpoint"

# Test 2: OpenAPI Schema
test_endpoint "GET" "/openapi.json" "200" "OpenAPI schema endpoint"

# Test 3: API Documentation
test_endpoint "GET" "/docs" "200" "API documentation endpoint"

# Test 4: Metrics Endpoint
test_endpoint "GET" "/metrics" "200" "Prometheus metrics endpoint"

# Test 5: Unauthorized Access
test_endpoint "GET" "/auth/me" "401" "Unauthorized access to protected endpoint"

# Test 6: Invalid Login
test_endpoint "POST" "/auth/login" "422" "Invalid login request (missing fields)" '{"username": "test"}'

# Test 7: Admin Endpoint Access (should fail without auth)
test_endpoint "GET" "/users" "401" "Admin endpoint access without authentication"

# Test 8: Service Proxy (should fail without auth)
test_endpoint "POST" "/proxy/ingestion/health" "401" "Service proxy without authentication" '{"method": "GET", "url": "/health"}'

echo ""
echo -e "${YELLOW}Authentication Flow Test${NC}"

# Test 9: Create Admin User (this will fail in real scenario without proper setup)
echo -e "${YELLOW}Note: User creation requires proper database setup${NC}"
test_endpoint "POST" "/users" "401" "User creation without authentication" '{"username": "admin", "email": "admin@example.com", "password": "admin123", "role": "admin"}'

echo ""
echo -e "${YELLOW}Rate Limiting Test${NC}"

# Test 10: Rate Limiting (make multiple requests)
echo "Making multiple requests to test rate limiting..."
for i in {1..5}; do
    test_endpoint "GET" "/health" "200" "Rate limit test request $i"
done

echo ""
echo -e "${YELLOW}Error Handling Test${NC}"

# Test 11: Non-existent Endpoint
test_endpoint "GET" "/nonexistent" "404" "Non-existent endpoint"

# Test 12: Invalid Method
test_endpoint "DELETE" "/health" "405" "Invalid HTTP method"

echo ""
echo -e "${GREEN}🎉 API Gateway Service tests completed!${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Set up Redis for rate limiting: docker-compose up -d redis"
echo "2. Create an admin user through the API"
echo "3. Test authentication flow with real tokens"
echo "4. Test service proxying to backend services"
echo ""
echo -e "${YELLOW}For full testing with authentication:${NC}"
echo "1. Start Redis: docker-compose up -d redis"
echo "2. Run the service: python -m uvicorn src.main:app --host 0.0.0.0 --port 8005 --reload"
echo "3. Create admin user via API"
echo "4. Test login and authenticated endpoints"

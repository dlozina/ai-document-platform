#!/bin/bash

# Simple Metrics Test Script
# This script generates test traffic to populate the Grafana dashboard

echo "🧪 Simple Metrics Test"
echo "======================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8003"
TENANT_ID="test-tenant-$(date +%s)"

echo -e "${BLUE}Testing with tenant: $TENANT_ID${NC}"
echo ""

# Function to make a request and show result
make_request() {
    local method=$1
    local endpoint=$2
    local description=$3
    local data=$4
    
    echo -n "Testing $description... "
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "%{http_code}" -H "X-Tenant-ID: $TENANT_ID" "$BASE_URL$endpoint")
    else
        response=$(curl -s -w "%{http_code}" -H "X-Tenant-ID: $TENANT_ID" -X "$method" "$BASE_URL$endpoint" -d "$data")
    fi
    
    http_code="${response: -3}"
    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        echo -e "${GREEN}✓ OK${NC}"
    else
        echo -e "✗ FAILED (HTTP $http_code)"
    fi
}

# Test basic endpoints
echo "1. Basic Health Checks"
echo "----------------------"
make_request "GET" "/health" "Health check"
make_request "GET" "/metrics" "Metrics endpoint"
make_request "GET" "/documents" "Document list"

echo ""
echo "2. Upload Simulation"
echo "-------------------"
# Create a small test file
echo "This is a test document for metrics testing." > /tmp/test-doc.txt

make_request "POST" "/upload" "File upload" "@/tmp/test-doc.txt"

# Clean up
rm -f /tmp/test-doc.txt

echo ""
echo "3. Generate Traffic"
echo "------------------"
echo "Generating some traffic to populate metrics..."

for i in {1..5}; do
    make_request "GET" "/health" "Health check $i"
    make_request "GET" "/documents" "Document list $i"
    sleep 1
done

echo ""
echo "4. Check Metrics"
echo "---------------"
echo "Current metrics:"
curl -s "$BASE_URL/metrics" | grep -E "(ingestion_http_requests_total|ingestion_service_uptime_seconds)" | head -5

echo ""
echo "📊 Dashboard URLs"
echo "================"
echo "Grafana Dashboard: http://localhost:3000/d/ingestion-overview/ingestion-service-overview"
echo "Prometheus:        http://localhost:9090"
echo ""
echo "✅ Test completed! Check the Grafana dashboard to see the metrics."

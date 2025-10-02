#!/bin/bash

# Test script for Ingestion Service Metrics
# This script tests the metrics collection by making various API calls

echo "Testing Ingestion Service Metrics Collection"
echo "============================================="

# Base URL for the ingestion service
BASE_URL="http://localhost:8003"

# Test tenant ID
TENANT_ID="test-tenant-001"

echo "1. Testing health endpoint..."
curl -s -H "X-Tenant-ID: $TENANT_ID" "$BASE_URL/health" | jq '.'

echo -e "\n2. Testing metrics endpoint..."
curl -s "$BASE_URL/metrics" | head -20

echo -e "\n3. Testing document list endpoint..."
curl -s -H "X-Tenant-ID: $TENANT_ID" "$BASE_URL/documents" | jq '.'

echo -e "\n4. Testing tenant quota endpoint..."
curl -s -H "X-Tenant-ID: $TENANT_ID" "$BASE_URL/tenants/$TENANT_ID/quota" | jq '.'

echo -e "\n5. Testing events health endpoint..."
curl -s "$BASE_URL/events/health" | jq '.'

echo -e "\n6. Testing metrics endpoint again to see updated metrics..."
curl -s "$BASE_URL/metrics" | grep -E "(ingestion_http_requests_total|ingestion_service_health_status)"

echo -e "\n7. Testing with different tenant..."
curl -s -H "X-Tenant-ID: test-tenant-002" "$BASE_URL/health" > /dev/null

echo -e "\n8. Final metrics check..."
curl -s "$BASE_URL/metrics" | grep -E "ingestion_http_requests_total" | head -10

echo -e "\nTesting completed!"
echo "Check Grafana at http://localhost:3000 (admin/admin)"
echo "Check Prometheus at http://localhost:9090"

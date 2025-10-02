#!/bin/bash

# Monitoring Setup Verification Script
# This script verifies that the monitoring stack is working correctly

echo "🔍 Verifying Monitoring Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if a service is responding
check_service() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $name... "
    
    if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
        if [ "$response" = "$expected_status" ]; then
            echo -e "${GREEN}✓ OK${NC}"
            return 0
        else
            echo -e "${RED}✗ FAILED (HTTP $response)${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED (Connection error)${NC}"
        return 1
    fi
}

# Function to check if a metric exists
check_metric() {
    local metric_name=$1
    local description=$2
    
    echo -n "Checking $description... "
    
    if curl -s "http://localhost:8003/metrics" | grep -q "$metric_name"; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED (Metric not found)${NC}"
        return 1
    fi
}

# Function to check Prometheus target
check_prometheus_target() {
    local job_name=$1
    local description=$2
    
    echo -n "Checking $description... "
    
    if curl -s "http://localhost:9090/api/v1/targets" | jq -e ".data.activeTargets[] | select(.job==\"$job_name\" and .health==\"up\")" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED (Target not healthy)${NC}"
        return 1
    fi
}

# Function to check Grafana datasource
check_grafana_datasource() {
    echo -n "Checking Grafana datasource... "
    
    if curl -s -u admin:admin "http://localhost:3000/api/datasources" | jq -e '.[] | select(.name=="Prometheus" and .uid=="prometheus")' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED (Datasource not found)${NC}"
        return 1
    fi
}

# Function to check Grafana dashboard
check_grafana_dashboard() {
    echo -n "Checking Grafana dashboard... "
    
    if curl -s -u admin:admin "http://localhost:3000/api/dashboards/uid/ingestion-overview" | jq -e '.dashboard.title' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED (Dashboard not found)${NC}"
        return 1
    fi
}

# Function to check if metrics are flowing
check_metrics_flow() {
    echo -n "Checking metrics flow... "
    
    if curl -s "http://localhost:9090/api/v1/query?query=ingestion_http_requests_total" | jq -e '.data.result | length > 0' > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED (No metrics in Prometheus)${NC}"
        return 1
    fi
}

# Main verification
echo "1. Service Health Checks"
echo "-----------------------"
check_service "Ingestion Service" "http://localhost:8003/health"
check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3000/api/health"

echo -e "\n2. Metrics Collection"
echo "---------------------"
check_metric "ingestion_http_requests_total" "HTTP request metrics"
check_metric "ingestion_uploads_total" "Upload metrics"
check_metric "ingestion_service_health_status" "Service health metrics"

echo -e "\n3. Prometheus Targets"
echo "---------------------"
check_prometheus_target "ingestion-service" "Ingestion service target"
check_prometheus_target "prometheus" "Prometheus self-monitoring"

echo -e "\n4. Grafana Configuration"
echo "------------------------"
check_grafana_datasource
check_grafana_dashboard

echo -e "\n5. Data Flow"
echo "-------------"
check_metrics_flow

echo -e "\n6. Quick Test"
echo "--------------"
echo -n "Generating test traffic... "
curl -s -H "X-Tenant-ID: test-tenant" "http://localhost:8003/health" > /dev/null
curl -s "http://localhost:8003/metrics" > /dev/null
echo -e "${GREEN}✓ Done${NC}"

echo -e "\n📊 Monitoring URLs"
echo "=================="
echo "Grafana Dashboard: http://localhost:3000/d/ingestion-overview/ingestion-service-overview"
echo "Prometheus:        http://localhost:9090"
echo "Ingestion Service: http://localhost:8003"
echo "Flower (Celery):   http://localhost:5555"

echo -e "\n🔧 Troubleshooting"
echo "=================="
echo "If any checks failed:"
echo "1. Check service logs: docker compose logs <service-name>"
echo "2. Restart services: docker compose restart"
echo "3. Check network connectivity between containers"
echo "4. Verify configuration files are in correct locations"

echo -e "\n✅ Verification complete!"

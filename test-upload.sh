#!/bin/bash

# Upload Test Script
# This script tests file uploads to generate upload metrics

echo "📁 Upload Test Script"
echo "===================="

BASE_URL="http://localhost:8003"
TENANT_ID="test-tenant-$(date +%s)"

echo "Testing with tenant: $TENANT_ID"
echo ""

# Create a test PDF file (simulate a real PDF)
echo "%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test Document) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000204 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
298
%%EOF" > /tmp/test-document.pdf

# Function to make upload request
upload_file() {
    local description=$1
    local file_path=$2

    echo -n "Testing $description... "

    response=$(curl -s -w "%{http_code}" \
        -H "X-Tenant-ID: $TENANT_ID" \
        -F "file=@$file_path" \
        "$BASE_URL/upload")

    http_code="${response: -3}"
    if [ "$http_code" = "200" ]; then
        echo -e "\033[0;32m✓ SUCCESS\033[0m"
        return 0
    else
        echo -e "\033[0;31m✗ FAILED (HTTP $http_code)\033[0m"
        echo "Response: ${response%???}"
        return 1
    fi
}

# Test uploads
echo "1. Testing File Uploads"
echo "----------------------"
upload_file "PDF upload" "/tmp/test-document.pdf"

# Create a text file to test different file type
echo "This is a test text document for upload testing." > /tmp/test-document.txt
upload_file "Text file upload" "/tmp/test-document.txt"

# Create an image file (fake)
echo "fake image data" > /tmp/test-image.jpg
upload_file "Image upload" "/tmp/test-image.jpg"

echo ""
echo "2. Check Upload Metrics"
echo "----------------------"
echo "Current upload metrics:"
curl -s "$BASE_URL/metrics" | grep "ingestion_uploads_total" | head -5

echo ""
echo "3. Wait for Prometheus Scrape"
echo "----------------------------"
echo "Waiting 30 seconds for Prometheus to scrape metrics..."
sleep 30

echo ""
echo "4. Query Prometheus"
echo "------------------"
echo "Total successful uploads:"
curl -s "http://localhost:9090/api/v1/query?query=sum(ingestion_uploads_total%7Bstatus%3D%22success%22%7D)" | jq -r '.data.result[0].value[1]'

echo "Upload rate (last 5 minutes):"
curl -s "http://localhost:9090/api/v1/query?query=sum(rate(ingestion_uploads_total%5B5m%5D))" | jq -r '.data.result[0].value[1]'

echo ""
echo "📊 Dashboard URLs"
echo "================"
echo "Grafana Dashboard: http://localhost:3000/d/ingestion-overview/ingestion-service-overview"
echo "Prometheus:        http://localhost:9090"

# Clean up
rm -f /tmp/test-document.pdf /tmp/test-document.txt /tmp/test-image.jpg

echo ""
echo "✅ Upload test completed!"

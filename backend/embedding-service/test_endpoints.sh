#!/bin/bash
# Simple curl-based test script for Embedding Service endpoints

BASE_URL="http://localhost:8002"

echo "🚀 Testing Embedding Service Endpoints"
echo "Service URL: $BASE_URL"
echo "=========================================="

# Test 1: Health Check
echo -e "\n🔍 Testing Health Endpoint..."
curl -s -X GET "$BASE_URL/health" | jq '.' 2>/dev/null || curl -s -X GET "$BASE_URL/health"
echo -e "\n"

# Test 2: Generate Single Embedding
echo "🔍 Testing Single Embedding Generation..."
curl -s -X POST "$BASE_URL/embed" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a test sentence for embedding generation.",
    "document_id": "test-doc-001",
    "metadata": {"source": "curl_test", "category": "demo"}
  }' | jq '.' 2>/dev/null || echo "Response received (install jq for pretty printing)"
echo -e "\n"

# Test 3: Batch Embedding Generation
echo "🔍 Testing Batch Embedding Generation..."
curl -s -X POST "$BASE_URL/embed-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "First batch test sentence.",
      "Second batch test sentence.",
      "Third batch test sentence."
    ],
    "document_ids": ["batch-doc-1", "batch-doc-2", "batch-doc-3"]
  }' | jq '.' 2>/dev/null || echo "Response received (install jq for pretty printing)"
echo -e "\n"

# Test 4: Similarity Search
echo "🔍 Testing Similarity Search..."
curl -s -X POST "$BASE_URL/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test sentence",
    "limit": 5,
    "score_threshold": 0.5
  }' | jq '.' 2>/dev/null || echo "Response received (install jq for pretty printing)"
echo -e "\n"

# Test 5: Collection Info
echo "🔍 Testing Collection Info..."
curl -s -X GET "$BASE_URL/collection/info" | jq '.' 2>/dev/null || curl -s -X GET "$BASE_URL/collection/info"
echo -e "\n"

# Test 6: File Upload (if test file exists)
if [ -f "test.txt" ]; then
    echo "🔍 Testing File Upload..."
    curl -s -X POST "$BASE_URL/embed-file" \
      -F "file=@test.txt" \
      -H "X-Document-ID: curl_file_test" | jq '.' 2>/dev/null || echo "Response received (install jq for pretty printing)"
    echo -e "\n"
else
    echo "🔍 Skipping File Upload Test (no test.txt file found)"
    echo "Create a test.txt file to test file upload functionality"
    echo -e "\n"
fi

# Test 7: Error Handling
echo "🔍 Testing Error Handling (Empty Text)..."
curl -s -X POST "$BASE_URL/embed" \
  -H "Content-Type: application/json" \
  -d '{"text": ""}' | jq '.' 2>/dev/null || echo "Response received (install jq for pretty printing)"
echo -e "\n"

echo "=========================================="
echo "🎉 All curl tests completed!"
echo ""
echo "💡 Tips:"
echo "   - Install 'jq' for pretty JSON output: brew install jq"
echo "   - Check service logs for detailed information"
echo "   - Use interactive API docs at http://localhost:8002/docs"
echo "   - Monitor Qdrant at http://localhost:6333/dashboard"

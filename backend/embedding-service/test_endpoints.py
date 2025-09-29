#!/usr/bin/env python3
"""
Comprehensive test script for Embedding Service endpoints
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8002"
TEST_FILES_DIR = Path("test_files")

def create_test_files():
    """Create test files for file upload testing."""
    TEST_FILES_DIR.mkdir(exist_ok=True)
    
    # Create test.txt
    with open(TEST_FILES_DIR / "test.txt", "w") as f:
        f.write("This is a test text file for embedding generation.")
    
    # Create test.json
    with open(TEST_FILES_DIR / "test.json", "w") as f:
        json.dump({
            "title": "Test Document",
            "content": "This is JSON content for testing embeddings.",
            "metadata": {"source": "test", "category": "demo"}
        }, f)
    
    # Create test.csv
    with open(TEST_FILES_DIR / "test.csv", "w") as f:
        f.write("name,description\n")
        f.write("Item1,First test item\n")
        f.write("Item2,Second test item\n")
        f.write("Item3,Third test item\n")
    
    print(f"✅ Created test files in {TEST_FILES_DIR}")

def test_health_endpoint():
    """Test the health check endpoint."""
    print("\n🔍 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                print("✅ Health check passed - Service is healthy")
            else:
                print("⚠️  Health check shows degraded status")
        else:
            print("❌ Health check failed")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service. Make sure it's running on port 8002")
        return False
    except Exception as e:
        print(f"❌ Error testing health endpoint: {e}")
        return False
    
    return True

def test_embed_endpoint():
    """Test the single embedding generation endpoint."""
    print("\n🔍 Testing Embed Endpoint...")
    
    test_cases = [
        {
            "name": "Basic text embedding",
            "data": {"text": "This is a simple test sentence for embedding generation."}
        },
        {
            "name": "Text with document ID",
            "data": {
                "text": "Another test sentence with document identifier.",
                "document_id": "test-doc-001"
            }
        },
        {
            "name": "Text with metadata",
            "data": {
                "text": "Test sentence with metadata information.",
                "document_id": "test-doc-002",
                "metadata": {"source": "test", "category": "demo", "priority": "high"}
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/embed",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Success - Embedding dimension: {data['embedding_dimension']}")
                print(f"  Model: {data['model_name']}")
                print(f"  Processing time: {data['processing_time_ms']:.2f}ms")
                print(f"  Text length: {data['text_length']} characters")
            else:
                print(f"  ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_batch_embed_endpoint():
    """Test the batch embedding generation endpoint."""
    print("\n🔍 Testing Batch Embed Endpoint...")
    
    test_cases = [
        {
            "name": "Small batch",
            "data": {
                "texts": [
                    "First batch test sentence.",
                    "Second batch test sentence.",
                    "Third batch test sentence."
                ]
            }
        },
        {
            "name": "Batch with document IDs",
            "data": {
                "texts": [
                    "Document one content.",
                    "Document two content.",
                    "Document three content."
                ],
                "document_ids": ["batch-doc-1", "batch-doc-2", "batch-doc-3"]
            }
        },
        {
            "name": "Batch with metadata",
            "data": {
                "texts": [
                    "Metadata test one.",
                    "Metadata test two."
                ],
                "document_ids": ["meta-doc-1", "meta-doc-2"],
                "metadata": [
                    {"category": "test", "priority": "high"},
                    {"category": "demo", "priority": "medium"}
                ]
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n  Testing: {test_case['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/embed-batch",
                json=test_case["data"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Success - Batch size: {data['batch_size']}")
                print(f"  Processing time: {data['processing_time_ms']:.2f}ms")
                print(f"  Embeddings generated: {len(data['embeddings'])}")
            else:
                print(f"  ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_search_endpoint():
    """Test the similarity search endpoint."""
    print("\n🔍 Testing Search Endpoint...")
    
    # First, let's add some test data
    print("  Adding test data for search...")
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Computer vision enables machines to interpret visual information.",
        "Data science combines statistics, programming, and domain expertise."
    ]
    
    for i, text in enumerate(test_texts):
        try:
            response = requests.post(
                f"{BASE_URL}/embed",
                json={
                    "text": text,
                    "document_id": f"search-doc-{i+1}",
                    "metadata": {"category": "AI", "topic": "machine learning"}
                }
            )
            if response.status_code == 200:
                print(f"  ✅ Added test document {i+1}")
            else:
                print(f"  ⚠️  Failed to add test document {i+1}")
        except Exception as e:
            print(f"  ❌ Error adding test document {i+1}: {e}")
    
    # Wait a moment for data to be processed
    time.sleep(1)
    
    # Now test search queries
    search_queries = [
        {
            "name": "Basic search",
            "data": {"query": "artificial intelligence", "limit": 3}
        },
        {
            "name": "Search with score threshold",
            "data": {
                "query": "neural networks",
                "limit": 5,
                "score_threshold": 0.5
            }
        },
        {
            "name": "Search with metadata filter",
            "data": {
                "query": "machine learning",
                "limit": 3,
                "filter": {"category": "AI"}
            }
        }
    ]
    
    for query_test in search_queries:
        print(f"\n  Testing: {query_test['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/search",
                json=query_test["data"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Success - Found {data['total_results']} results")
                print(f"  Search time: {data['search_time_ms']:.2f}ms")
                
                for i, result in enumerate(data['results'][:2]):  # Show first 2 results
                    print(f"    Result {i+1}: Score {result['score']:.3f}")
                    print(f"      Document ID: {result.get('document_id', 'N/A')}")
                    print(f"      Text: {result['text'][:60]}...")
            else:
                print(f"  ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_file_upload_endpoint():
    """Test the file upload endpoint."""
    print("\n🔍 Testing File Upload Endpoint...")
    
    # Create test files first
    create_test_files()
    
    test_files = [
        ("test.txt", "text/plain"),
        ("test.json", "application/json"),
        ("test.csv", "text/csv")
    ]
    
    for filename, content_type in test_files:
        file_path = TEST_FILES_DIR / filename
        if not file_path.exists():
            print(f"  ⚠️  Test file {filename} not found, skipping")
            continue
            
        print(f"\n  Testing: Upload {filename}")
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (filename, f, content_type)}
                response = requests.post(
                    f"{BASE_URL}/embed-file",
                    files=files,
                    headers={"X-Document-ID": f"file_test_{filename}"}
                )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Success - File processed")
                print(f"  Method: {data['method']}")
                print(f"  Text length: {data['text_length']} characters")
                print(f"  Processing time: {data['processing_time_ms']:.2f}ms")
            else:
                print(f"  ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_collection_info_endpoint():
    """Test the collection information endpoint."""
    print("\n🔍 Testing Collection Info Endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/collection/info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success - Collection Info:")
            print(f"  Name: {data['name']}")
            print(f"  Vector size: {data['vector_size']}")
            print(f"  Points count: {data['points_count']}")
            print(f"  Status: {data['status']}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_delete_endpoint():
    """Test the delete embedding endpoint."""
    print("\n🔍 Testing Delete Endpoint...")
    
    # First create a test embedding to delete
    print("  Creating test embedding to delete...")
    try:
        response = requests.post(
            f"{BASE_URL}/embed",
            json={
                "text": "This embedding will be deleted for testing.",
                "document_id": "delete-test-doc"
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            point_id = data.get('document_id')  # This will be the UUID
            print("  ✅ Test embedding created")
            
            # Now try to delete it using the UUID
            print("  Attempting to delete test embedding...")
            delete_response = requests.delete(f"{BASE_URL}/embedding/{point_id}")
            
            print(f"  Delete Status Code: {delete_response.status_code}")
            
            if delete_response.status_code == 200:
                print("  ✅ Successfully deleted test embedding")
            else:
                print(f"  ❌ Delete failed: {delete_response.text}")
        else:
            print("  ❌ Failed to create test embedding for deletion")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")

def test_error_handling():
    """Test error handling scenarios."""
    print("\n🔍 Testing Error Handling...")
    
    error_tests = [
        {
            "name": "Empty text",
            "endpoint": "/embed",
            "data": {"text": ""},
            "expected_status": 422
        },
        {
            "name": "Text too long",
            "endpoint": "/embed",
            "data": {"text": "a" * 10001},
            "expected_status": 422
        },
        {
            "name": "Batch too large",
            "endpoint": "/embed-batch",
            "data": {"texts": ["text"] * 101},
            "expected_status": 400
        },
        {
            "name": "Invalid search limit",
            "endpoint": "/search",
            "data": {"query": "test", "limit": 0},
            "expected_status": 422
        }
    ]
    
    for test in error_tests:
        print(f"\n  Testing: {test['name']}")
        try:
            response = requests.post(
                f"{BASE_URL}{test['endpoint']}",
                json=test["data"],
                headers={"Content-Type": "application/json"}
            )
            
            print(f"  Status Code: {response.status_code}")
            
            if response.status_code == test["expected_status"]:
                print(f"  ✅ Correctly handled error (expected {test['expected_status']})")
            else:
                print(f"  ⚠️  Unexpected status code (expected {test['expected_status']})")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")

def cleanup_test_files():
    """Clean up test files."""
    if TEST_FILES_DIR.exists():
        import shutil
        shutil.rmtree(TEST_FILES_DIR)
        print(f"\n🧹 Cleaned up test files")

def main():
    """Run all endpoint tests."""
    print("🚀 Starting Embedding Service Endpoint Tests")
    print(f"Testing service at: {BASE_URL}")
    print("=" * 60)
    
    # Test if service is running
    if not test_health_endpoint():
        print("\n❌ Service is not running. Please start the service first:")
        print("   docker-compose up -d")
        print("   or")
        print("   python -m uvicorn src.main:app --host 0.0.0.0 --port 8002")
        return
    
    # Run all tests
    test_embed_endpoint()
    test_batch_embed_endpoint()
    test_search_endpoint()
    test_file_upload_endpoint()
    test_collection_info_endpoint()
    test_delete_endpoint()
    test_error_handling()
    
    # Cleanup
    cleanup_test_files()
    
    print("\n" + "=" * 60)
    print("🎉 All endpoint tests completed!")
    print("\n💡 Tips:")
    print("   - Check the service logs for detailed information")
    print("   - Use the interactive API docs at http://localhost:8002/docs")
    print("   - Monitor Qdrant at http://localhost:6333/dashboard")

if __name__ == "__main__":
    main()

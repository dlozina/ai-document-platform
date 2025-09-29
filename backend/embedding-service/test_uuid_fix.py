#!/usr/bin/env python3
"""
Quick test to verify the Qdrant UUID fix
"""

import requests
import json

BASE_URL = "http://localhost:8002"

def test_uuid_fix():
    """Test that the UUID fix works for embedding generation."""
    print("🧪 Testing UUID Fix for Qdrant...")
    
    # Test 1: Generate embedding with string document ID
    print("\n1. Testing embedding generation with string document ID...")
    try:
        response = requests.post(
            f"{BASE_URL}/embed",
            json={
                "text": "This is a test sentence for UUID fix verification.",
                "document_id": "test-doc-string-id",
                "metadata": {"test": "uuid_fix", "category": "verification"}
            }
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Embedding generated with UUID")
            print(f"   Document ID: {data.get('document_id')}")
            print(f"   Embedding dimension: {data.get('embedding_dimension')}")
            print(f"   Processing time: {data.get('processing_time_ms'):.2f}ms")
            
            # Test 2: Search for the embedding
            print("\n2. Testing search for the generated embedding...")
            search_response = requests.post(
                f"{BASE_URL}/search",
                json={
                    "query": "test sentence UUID fix",
                    "limit": 3
                }
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                print("✅ Search successful!")
                print(f"   Found {search_data['total_results']} results")
                
                if search_data['results']:
                    result = search_data['results'][0]
                    print(f"   Top result score: {result['score']:.3f}")
                    print(f"   Document ID: {result.get('document_id')}")
                    print(f"   Text: {result['text'][:50]}...")
                
            else:
                print(f"❌ Search failed: {search_response.text}")
                
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def test_batch_uuid_fix():
    """Test batch embedding with UUID fix."""
    print("\n3. Testing batch embedding with UUID fix...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/embed-batch",
            json={
                "texts": [
                    "First batch test for UUID fix.",
                    "Second batch test for UUID fix.",
                    "Third batch test for UUID fix."
                ],
                "document_ids": ["batch-uuid-test-1", "batch-uuid-test-2", "batch-uuid-test-3"]
            }
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Batch embedding successful!")
            print(f"   Batch size: {data.get('batch_size')}")
            print(f"   Processing time: {data.get('processing_time_ms'):.2f}ms")
            print(f"   Document IDs: {data.get('document_ids')}")
        else:
            print(f"❌ Failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Quick UUID Fix Verification Test")
    print("=" * 50)
    
    # Check if service is running
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        if health_response.status_code == 200:
            print("✅ Service is running")
            test_uuid_fix()
            test_batch_uuid_fix()
        else:
            print("❌ Service health check failed")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service. Make sure it's running on port 8002")
        print("   Start with: docker-compose up -d")
        print("   Or: python -m uvicorn src.main:app --host 0.0.0.0 --port 8002")
    
    print("\n" + "=" * 50)
    print("🎉 UUID fix verification completed!")

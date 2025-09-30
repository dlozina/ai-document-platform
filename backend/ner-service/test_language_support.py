#!/usr/bin/env python3
"""
Test script to verify language support functionality in NER service.
"""

import requests
import json
import time

# Service configuration
BASE_URL = "http://localhost:8001"

def test_language_support():
    """Test the language support functionality."""
    print("🧪 Testing NER Service Language Support")
    print("=" * 50)
    
    # Test data
    english_text = "John Smith works at Microsoft in Seattle."
    croatian_text = "Marko Petrović radi u Zagrebu za Hrvatsku poštu."
    
    # Test 1: Check supported languages
    print("\n1. Testing supported languages endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/languages")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Supported languages: {data['supported_languages']}")
            for lang, info in data['language_info'].items():
                status = "✅ Available" if info['available'] else "❌ Not Available"
                print(f"   {lang} ({info['name']}): {status}")
        else:
            print(f"❌ Failed to get languages: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting languages: {e}")
    
    # Test 2: Check available models
    print("\n2. Testing available models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print("✅ Available models:")
            for model, available in data['available_models'].items():
                status = "✅" if available else "❌"
                print(f"   {status} {model}")
        else:
            print(f"❌ Failed to get models: {response.status_code}")
    except Exception as e:
        print(f"❌ Error getting models: {e}")
    
    # Test 3: Test English text processing
    print("\n3. Testing English text processing...")
    try:
        payload = {
            "text": english_text,
            "language": "en",
            "include_confidence": True
        }
        response = requests.post(f"{BASE_URL}/extract", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ English processing successful!")
            print(f"   Model used: {data['spacy_model_used']}")
            print(f"   Language: {data['language']}")
            print(f"   Entities found: {data['entity_count']}")
            for entity in data['entities']:
                print(f"   - {entity['text']} ({entity['label']})")
        else:
            print(f"❌ English processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error processing English text: {e}")
    
    # Test 4: Test Croatian text processing
    print("\n4. Testing Croatian text processing...")
    try:
        payload = {
            "text": croatian_text,
            "language": "hr",
            "include_confidence": True
        }
        response = requests.post(f"{BASE_URL}/extract", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Croatian processing successful!")
            print(f"   Model used: {data['spacy_model_used']}")
            print(f"   Language: {data['language']}")
            print(f"   Entities found: {data['entity_count']}")
            for entity in data['entities']:
                print(f"   - {entity['text']} ({entity['label']})")
        else:
            print(f"❌ Croatian processing failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Error processing Croatian text: {e}")
    
    # Test 5: Test unsupported language
    print("\n5. Testing unsupported language...")
    try:
        payload = {
            "text": "Test text",
            "language": "fr",  # French - not supported
            "include_confidence": True
        }
        response = requests.post(f"{BASE_URL}/extract", json=payload)
        if response.status_code == 400:
            print("✅ Unsupported language correctly rejected!")
            print(f"   Error message: {response.json()['detail']}")
        else:
            print(f"❌ Expected 400 error, got: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing unsupported language: {e}")
    
    # Test 6: Test batch processing with different languages
    print("\n6. Testing batch processing with mixed languages...")
    try:
        payload = {
            "texts": [english_text, croatian_text],
            "language": "en",  # Process both as English (this will work but may not be optimal)
            "include_confidence": True
        }
        response = requests.post(f"{BASE_URL}/extract-batch", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch processing successful!")
            print(f"   Language: {data['language']}")
            print(f"   Texts processed: {data['batch_size']}")
            for i, result in enumerate(data['results']):
                print(f"   Text {i+1}: {result['entity_count']} entities found")
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error in batch processing: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Language support testing completed!")

if __name__ == "__main__":
    print("Starting NER Service Language Support Test")
    print("Make sure the NER service is running on http://localhost:8001")
    print("Press Ctrl+C to cancel, or wait 5 seconds to start...")
    
    try:
        time.sleep(5)
        test_language_support()
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

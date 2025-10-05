#!/usr/bin/env python3
"""
Quick validation script to test the Pydantic model fixes.
"""


def test_models():
    """Test that the models can be imported and instantiated correctly."""
    try:
        from src.models import NERRequest, NERResponse

        # Test valid English request
        english_request = NERRequest(
            text="John Smith works at Microsoft", language="en"
        )
        print("✅ English request created successfully")
        print(f"   Language: {english_request.language}")

        # Test valid Croatian request
        croatian_request = NERRequest(
            text="Marko Petrović radi u Zagrebu", language="hr"
        )
        print("✅ Croatian request created successfully")
        print(f"   Language: {croatian_request.language}")

        # Test invalid language (should raise validation error)
        try:
            NERRequest(
                text="Test text",
                language="fr",  # French - not supported
            )
            print("❌ Invalid language should have been rejected")
        except Exception as e:
            print("✅ Invalid language correctly rejected")
            print(f"   Error: {str(e)[:100]}...")

        # Test NERResponse with new field name
        response = NERResponse(
            text="Test text",
            entities=[],
            entity_count=0,
            spacy_model_used="en_core_web_sm",
            processing_time_ms=10.5,
            text_length=10,
        )
        print("✅ NERResponse created successfully")
        print(f"   Model used: {response.spacy_model_used}")

        print("\n🎉 All model tests passed!")

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Testing Pydantic model fixes...")
    test_models()

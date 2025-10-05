"""
Unit tests for OCR Processor
"""

import io

import pytest
from PIL import Image
from src.ocr_processor import OCRProcessor


@pytest.fixture
def ocr_processor():
    """Create OCR processor instance for testing."""
    return OCRProcessor(dpi=150, language="eng")  # Lower DPI for faster tests


@pytest.fixture
def sample_image():
    """Create a simple test image with text."""
    # Create a white image with black text
    img = Image.new("RGB", (400, 100), color="white")

    # Add some text using PIL's ImageDraw (requires pillow)
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    # Use default font
    text = "Test Document 12345"
    draw.text((10, 40), text, fill="black")

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return img_bytes.getvalue()


@pytest.fixture
def sample_pdf():
    """Create a simple test PDF with text."""
    import fitz  # PyMuPDF

    # Create a new PDF
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    # Add text
    text = "This is a test PDF document.\nIt has multiple lines.\nPage 1 content."
    page.insert_text((50, 50), text, fontsize=12)

    # Convert to bytes
    pdf_bytes = doc.write()
    doc.close()

    return pdf_bytes


class TestOCRProcessor:
    """Test suite for OCR Processor."""

    def test_initialization(self):
        """Test OCR processor initialization."""
        processor = OCRProcessor(dpi=300, language="eng")
        assert processor.dpi == 300
        assert processor.language == "eng"

    def test_process_image(self, ocr_processor, sample_image):
        """Test processing a simple image."""
        result = ocr_processor.process_file(
            file_content=sample_image, filename="test.png"
        )

        # Verify structure
        assert "text" in result
        assert "page_count" in result
        assert "method" in result
        assert "confidence" in result
        assert "processing_time_ms" in result

        # Verify content
        assert result["page_count"] == 1
        assert result["method"] == "ocr_image"
        assert isinstance(result["confidence"], float)
        assert result["confidence"] >= 0

        # Check if text was extracted (OCR might not be perfect)
        assert len(result["text"]) > 0

    def test_process_searchable_pdf(self, ocr_processor, sample_pdf):
        """Test processing a searchable PDF (native text extraction)."""
        result = ocr_processor.process_file(
            file_content=sample_pdf, filename="test.pdf"
        )

        # Should use native extraction for searchable PDF
        assert result["method"] == "native_pdf"
        assert result["confidence"] is None  # Native extraction has no confidence
        assert result["page_count"] == 1

        # Check extracted text
        assert "test PDF document" in result["text"].lower()
        assert "multiple lines" in result["text"].lower()

    def test_force_ocr_on_pdf(self, ocr_processor, sample_pdf):
        """Test forcing OCR on a searchable PDF."""
        result = ocr_processor.process_file(
            file_content=sample_pdf, filename="test.pdf", force_ocr=True
        )

        # Should use OCR even though PDF is searchable
        assert result["method"] == "ocr_pdf"
        assert isinstance(result["confidence"], float)

    def test_unsupported_file_type(self, ocr_processor):
        """Test handling of unsupported file types."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            ocr_processor.process_file(
                file_content=b"fake content", filename="test.txt"
            )

    def test_empty_file(self, ocr_processor):
        """Test handling of empty files."""
        with pytest.raises(RuntimeError):  # Should raise some error
            ocr_processor.process_file(file_content=b"", filename="test.pdf")

    def test_corrupted_file(self, ocr_processor):
        """Test handling of corrupted files."""
        with pytest.raises(RuntimeError, match="OCR processing failed"):
            ocr_processor.process_file(
                file_content=b"not a valid pdf or image", filename="test.pdf"
            )

    def test_extract_layout(self, ocr_processor, sample_image):
        """Test extracting text with layout information."""
        words = ocr_processor.extract_text_with_layout(
            file_content=sample_image, filename="test.png"
        )

        # Should return list of words
        assert isinstance(words, list)

        if len(words) > 0:  # OCR might detect words
            word = words[0]

            # Verify structure
            assert "text" in word
            assert "confidence" in word
            assert "left" in word
            assert "top" in word
            assert "width" in word
            assert "height" in word
            assert "page_num" in word

    def test_preprocess_image(self, ocr_processor):
        """Test image preprocessing."""
        # Create a low-contrast image
        img = Image.new("L", (100, 100), color=128)

        # Preprocess
        processed = ocr_processor._preprocess_image(img)

        # Verify it's still an image
        assert isinstance(processed, Image.Image)
        assert processed.mode == "L"  # Grayscale

    def test_contrast_stretching(self, ocr_processor):
        """Test contrast stretching algorithm."""
        import numpy as np

        # Create low-contrast array
        arr = np.random.randint(100, 150, size=(100, 100), dtype=np.uint8)

        # Stretch contrast
        stretched = ocr_processor._stretch_contrast(arr)

        # Verify contrast was increased
        assert stretched.max() > arr.max() or stretched.min() < arr.min()
        assert stretched.dtype == np.uint8

    def test_multiple_page_pdf(self, ocr_processor):
        """Test processing multi-page PDF."""
        import fitz

        # Create 3-page PDF
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((50, 50), f"Page {i + 1} content", fontsize=12)

        pdf_bytes = doc.write()
        doc.close()

        # Process
        result = ocr_processor.process_file(
            file_content=pdf_bytes, filename="multi_page.pdf"
        )

        assert result["page_count"] == 3
        assert len(result["pages"]) == 3

    def test_large_image(self, ocr_processor):
        """Test processing a larger image."""
        # Create a larger image
        img = Image.new("RGB", (2000, 1500), color="white")
        from PIL import ImageDraw

        draw = ImageDraw.Draw(img)
        draw.text((100, 750), "Large Image Test", fill="black")

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        result = ocr_processor.process_file(
            file_content=img_bytes.getvalue(), filename="large_test.png"
        )

        assert result["method"] == "ocr_image"
        assert result["processing_time_ms"] > 0

    def test_different_image_formats(self, ocr_processor):
        """Test processing different image formats."""
        formats = ["PNG", "JPEG", "BMP"]

        for fmt in formats:
            img = Image.new("RGB", (200, 100), color="white")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)
            draw.text((50, 40), f"{fmt} Test", fill="black")

            img_bytes = io.BytesIO()
            img.save(img_bytes, format=fmt)
            img_bytes.seek(0)

            result = ocr_processor.process_file(
                file_content=img_bytes.getvalue(), filename=f"test.{fmt.lower()}"
            )

            assert result["method"] == "ocr_image"
            assert len(result["text"]) > 0


class TestOCRProcessorEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_image(self, ocr_processor):
        """Test processing a very small image."""
        img = Image.new("RGB", (10, 10), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        result = ocr_processor.process_file(
            file_content=img_bytes.getvalue(), filename="tiny.png"
        )

        # Should process without error, but may extract no text
        assert result["method"] == "ocr_image"

    def test_blank_page(self, ocr_processor):
        """Test processing a blank page."""
        import fitz

        doc = fitz.open()
        doc.new_page()  # Blank page
        pdf_bytes = doc.write()
        doc.close()

        result = ocr_processor.process_file(
            file_content=pdf_bytes, filename="blank.pdf"
        )

        # Should process without error
        assert result["page_count"] == 1
        # Text might be empty or minimal

    def test_special_characters(self, ocr_processor):
        """Test handling special characters."""
        import fitz

        doc = fitz.open()
        page = doc.new_page()

        # Add text with special characters
        text = "Special: @#$%^&*()_+-=[]{}|;:',.<>?/`~"
        page.insert_text((50, 50), text, fontsize=12)

        pdf_bytes = doc.write()
        doc.close()

        result = ocr_processor.process_file(
            file_content=pdf_bytes, filename="special_chars.pdf"
        )

        assert result["method"] == "native_pdf"
        # Some special characters should be present
        assert len(result["text"]) > 0


@pytest.mark.parametrize(
    "dpi,expected_quality",
    [
        (150, "low"),
        (300, "medium"),
        (600, "high"),
    ],
)
def test_dpi_settings(dpi, expected_quality):
    """Test different DPI settings affect quality."""
    processor = OCRProcessor(dpi=dpi, language="eng")
    assert processor.dpi == dpi


@pytest.mark.parametrize(
    "language",
    [
        "eng",
        "hrv",
        "eng+hrv",
        "fra",
        "deu",
        "spa",
    ],
)
def test_language_settings(language):
    """Test different language settings."""
    # Note: This test will only pass if language packs are installed
    try:
        processor = OCRProcessor(language=language)
        assert processor.language == language
    except RuntimeError:
        pytest.skip(f"Language pack {language} not installed")


def test_language_detection():
    """Test automatic language detection functionality."""
    processor = OCRProcessor(language="eng+hrv", enable_language_detection=True)

    # Test Croatian text detection
    croatian_text = "Ovo je hrvatski tekst sa č, ć, đ, š, ž karakterima."
    detected_lang = processor._detect_language(croatian_text)
    assert detected_lang == "hrv"

    # Test English text detection
    english_text = "This is English text with common words like the, and, or."
    detected_lang = processor._detect_language(english_text)
    assert detected_lang == "eng"

    # Test mixed text detection
    mixed_text = (
        "This is mixed text. Ovo je miješani tekst sa English i Croatian words."
    )
    detected_lang = processor._detect_language(mixed_text)
    assert detected_lang == "eng+hrv"

    # Test short text (should return default language)
    short_text = "Hi"
    detected_lang = processor._detect_language(short_text)
    assert detected_lang == "eng+hrv"  # Default language


def test_language_detection_disabled():
    """Test that language detection can be disabled."""
    processor = OCRProcessor(language="eng", enable_language_detection=False)

    # Even with Croatian text, should return configured language
    croatian_text = "Ovo je hrvatski tekst sa č, ć, đ, š, ž karakterima."
    detected_lang = processor._detect_language(croatian_text)
    assert detected_lang == "eng"  # Should return configured language, not detected


def test_process_file_with_language_detection():
    """Test that process_file includes detected language in result."""
    processor = OCRProcessor(language="eng+hrv", enable_language_detection=True)

    # Create a simple test image with text
    import io

    from PIL import Image, ImageDraw, ImageFont

    # Create a test image
    img = Image.new("RGB", (200, 50), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 15), "Test English Text", fill="black", font=font)

    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Process the image
    result = processor.process_file(img_bytes, "test.png")

    # Check that detected_language is included
    assert "detected_language" in result
    assert result["detected_language"] is not None
    assert isinstance(result["detected_language"], str)

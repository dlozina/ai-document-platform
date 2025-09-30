"""
OCR Processor Module

Handles text extraction from PDFs and images using:
- pytesseract for OCR (Optical Character Recognition)
- PyMuPDF (fitz) for PDF text extraction
- pdf2image for PDF to image conversion
- Pillow for image processing
"""

import io
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
import numpy as np

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Processes PDFs and images to extract text content.
    
    Supports:
    - Native PDF text extraction (fast, high quality)
    - OCR for scanned PDFs and images (slower, handles non-searchable content)
    """
    
    # Supported file extensions
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    PDF_EXTENSION = '.pdf'
    
    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        dpi: int = 300,
        language: str = 'eng+hrv',
        enable_language_detection: bool = True
    ):
        """
        Initialize OCR Processor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (None = use system default)
            dpi: DPI for PDF to image conversion (higher = better quality, slower)
            language: Tesseract language code (eng, hrv, eng+hrv, etc.)
            enable_language_detection: Whether to detect language automatically
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.dpi = dpi
        self.language = language
        self.enable_language_detection = enable_language_detection
        
        # Verify tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {pytesseract.get_tesseract_version()}")
            
            # Log available languages
            available_langs = pytesseract.get_languages()
            logger.info(f"Available languages: {available_langs}")
            
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise RuntimeError(
                "Tesseract OCR is not installed. "
                "Install: apt-get install tesseract-ocr (Linux) or brew install tesseract (Mac)"
            )
    
    def process_file(
        self,
        file_content: bytes,
        filename: str,
        force_ocr: bool = False
    ) -> Dict[str, any]:
        """
        Process a file and extract text.
        
        Args:
            file_content: Raw bytes of the file
            filename: Original filename (used to detect file type)
            force_ocr: If True, always use OCR even for searchable PDFs
        
        Returns:
            {
                'text': str,              # Extracted text
                'page_count': int,        # Number of pages
                'method': str,            # 'native_pdf', 'ocr_pdf', or 'ocr_image'
                'confidence': float,      # OCR confidence (0-100, None for native)
                'detected_language': str, # Detected language code
                'processing_time_ms': float
            }
        
        Raises:
            ValueError: If file type is unsupported
            RuntimeError: If processing fails
        """
        import time
        start_time = time.time()
        
        file_ext = Path(filename).suffix.lower()
        
        try:
            if file_ext == self.PDF_EXTENSION:
                result = self._process_pdf(file_content, force_ocr)
            elif file_ext in self.IMAGE_EXTENSIONS:
                result = self._process_image(file_content)
            else:
                raise ValueError(
                    f"Unsupported file type: {file_ext}. "
                    f"Supported: {self.IMAGE_EXTENSIONS | {self.PDF_EXTENSION}}"
                )
            
            # Add processing time
            result['processing_time_ms'] = (time.time() - start_time) * 1000
            
            # Add detected language if enabled
            if self.enable_language_detection and result['method'] != 'native_pdf':
                result['detected_language'] = self._detect_language(result['text'])
            else:
                result['detected_language'] = self.language
            
            logger.info(
                f"Processed {filename} using {result['method']} "
                f"in {result['processing_time_ms']:.2f}ms "
                f"(detected language: {result['detected_language']})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}", exc_info=True)
            raise RuntimeError(f"OCR processing failed: {str(e)}")
    
    def _process_pdf(
        self,
        pdf_content: bytes,
        force_ocr: bool = False
    ) -> Dict[str, any]:
        """
        Extract text from PDF.
        
        Strategy:
        1. Try native text extraction (fast, for searchable PDFs)
        2. If no text found or force_ocr=True, use OCR (slower, for scanned PDFs)
        """
        # Try native extraction first
        if not force_ocr:
            native_result = self._extract_native_pdf_text(pdf_content)
            
            # If we got meaningful text, return it
            if native_result['text'].strip() and len(native_result['text']) > 50:
                native_result['method'] = 'native_pdf'
                native_result['confidence'] = None
                return native_result
            
            logger.info("Native PDF extraction yielded minimal text, falling back to OCR")
        
        # Fall back to OCR
        return self._extract_pdf_text_via_ocr(pdf_content)
    
    def _extract_native_pdf_text(self, pdf_content: bytes) -> Dict[str, any]:
        """Extract text from searchable PDF using PyMuPDF."""
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        
        pages_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            pages_text.append(text)
        
        doc.close()
        
        return {
            'text': '\n\n'.join(pages_text),
            'page_count': len(pages_text),
            'pages': pages_text  # Individual page texts
        }
    
    def _extract_pdf_text_via_ocr(self, pdf_content: bytes) -> Dict[str, any]:
        """Convert PDF to images and run OCR on each page."""
        # Convert PDF pages to images
        images = convert_from_bytes(
            pdf_content,
            dpi=self.dpi,
            fmt='png'
        )
        
        logger.info(f"Converted PDF to {len(images)} images for OCR")
        
        pages_text = []
        confidences = []
        
        for i, image in enumerate(images):
            logger.debug(f"Running OCR on page {i+1}/{len(images)}")
            
            # Run OCR with detailed data
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text = pytesseract.image_to_string(image, lang=self.language)
            pages_text.append(text)
            
            # Calculate average confidence (ignore -1 values which indicate no text)
            page_confidences = [
                float(conf) for conf in ocr_data['conf'] if conf != '-1'
            ]
            if page_confidences:
                confidences.append(np.mean(page_confidences))
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': '\n\n'.join(pages_text),
            'page_count': len(pages_text),
            'pages': pages_text,
            'method': 'ocr_pdf',
            'confidence': round(avg_confidence, 2)
        }
    
    def _process_image(self, image_content: bytes) -> Dict[str, any]:
        """Extract text from image using OCR."""
        # Load image
        image = Image.open(io.BytesIO(image_content))
        
        # Preprocess image for better OCR results
        image = self._preprocess_image(image)
        
        # Run OCR with detailed data
        ocr_data = pytesseract.image_to_data(
            image,
            lang=self.language,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract text
        text = pytesseract.image_to_string(image, lang=self.language)
        
        # Calculate confidence
        confidences = [
            float(conf) for conf in ocr_data['conf'] if conf != '-1'
        ]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'text': text,
            'page_count': 1,
            'pages': [text],
            'method': 'ocr_image',
            'confidence': round(avg_confidence, 2)
        }
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.
        
        Techniques:
        - Convert to grayscale
        - Increase contrast
        - Remove noise (optional)
        """
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Convert to numpy array for processing
        img_array = np.array(image)
        
        # Increase contrast using histogram stretching
        img_array = self._stretch_contrast(img_array)
        
        # Convert back to PIL Image
        return Image.fromarray(img_array)
    
    @staticmethod
    def _stretch_contrast(img_array: np.ndarray) -> np.ndarray:
        """Stretch image histogram to improve contrast."""
        min_val = np.percentile(img_array, 2)
        max_val = np.percentile(img_array, 98)
        
        if max_val > min_val:
            img_array = np.clip(img_array, min_val, max_val)
            img_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return img_array
    
    def extract_text_with_layout(
        self,
        file_content: bytes,
        filename: str
    ) -> List[Dict[str, any]]:
        """
        Extract text with positional information (bounding boxes).
        
        Useful for preserving document layout or extracting tables.
        
        Returns:
            List of word dictionaries with:
            - text: str
            - confidence: float
            - left, top, width, height: int (bounding box)
            - page_num: int
        """
        file_ext = Path(filename).suffix.lower()
        
        if file_ext in self.IMAGE_EXTENSIONS:
            image = Image.open(io.BytesIO(file_content))
            return self._extract_layout_from_image(image, page_num=1)
        
        elif file_ext == self.PDF_EXTENSION:
            images = convert_from_bytes(file_content, dpi=self.dpi)
            
            all_words = []
            for page_num, image in enumerate(images, start=1):
                words = self._extract_layout_from_image(image, page_num)
                all_words.extend(words)
            
            return all_words
        
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def _extract_layout_from_image(
        self,
        image: Image.Image,
        page_num: int
    ) -> List[Dict[str, any]]:
        """Extract words with bounding boxes from image."""
        ocr_data = pytesseract.image_to_data(
            image,
            lang=self.language,
            output_type=pytesseract.Output.DICT
        )
        
        words = []
        n_boxes = len(ocr_data['text'])
        
        for i in range(n_boxes):
            # Skip empty detections
            if int(ocr_data['conf'][i]) == -1:
                continue
            
            word = {
                'text': ocr_data['text'][i],
                'confidence': float(ocr_data['conf'][i]),
                'left': int(ocr_data['left'][i]),
                'top': int(ocr_data['top'][i]),
                'width': int(ocr_data['width'][i]),
                'height': int(ocr_data['height'][i]),
                'page_num': page_num
            }
            
            words.append(word)
        
        return words
    
    def _detect_language(self, text: str) -> str:
        """
        Detect the primary language of the extracted text.
        
        Uses a simple heuristic approach based on character patterns
        and common words to determine if text is Croatian or English.
        
        Args:
            text: Extracted text to analyze
            
        Returns:
            Detected language code ('eng', 'hrv', or 'eng+hrv' for mixed)
        """
        if not text or len(text.strip()) < 10:
            return self.language
        
        text_lower = text.lower()
        
        # Croatian-specific characters and patterns
        croatian_chars = ['č', 'ć', 'đ', 'š', 'ž', 'dž']
        croatian_words = [
            'i', 'je', 'na', 'za', 'od', 'do', 'sa', 'iz', 'po', 'u', 'o',
            'da', 'ne', 'ili', 'ali', 'kada', 'gdje', 'kako', 'zašto',
            'hrvatski', 'hrvatska', 'hrvatsko', 'republika', 'grad',
            'ulica', 'adresa', 'telefon', 'email', 'datum', 'vrijeme'
        ]
        
        # English-specific patterns
        english_words = [
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between',
            'address', 'phone', 'email', 'date', 'time', 'street'
        ]
        
        # Count Croatian-specific characters
        croatian_char_count = sum(1 for char in text_lower if char in croatian_chars)
        
        # Count Croatian words
        croatian_word_count = sum(1 for word in croatian_words if word in text_lower)
        
        # Count English words
        english_word_count = sum(1 for word in english_words if word in text_lower)
        
        # Calculate scores
        total_words = len(text.split())
        croatian_score = (croatian_char_count * 2) + croatian_word_count
        english_score = english_word_count
        
        # Determine language based on scores
        if croatian_score > english_score and croatian_score > 2:
            return 'hrv'
        elif english_score > croatian_score and english_score > 2:
            return 'eng'
        elif croatian_score > 0 and english_score > 0:
            return 'eng+hrv'  # Mixed content
        else:
            # Default to configured language if no clear indicators
            return self.language
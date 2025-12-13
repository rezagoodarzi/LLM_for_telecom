"""
Image Processor
================

Utilities for handling images extracted from documents:
- OCR text extraction
- Image preprocessing
- Caption extraction
- Image metadata handling
- Visual content indexing

Integrates with Docling for enhanced image processing.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional image processing imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImageInfo:
    """Information about an extracted image"""
    index: int
    page: int
    path: Optional[str] = None
    width: int = 0
    height: int = 0
    format: str = ""
    ocr_text: str = ""
    caption: str = ""
    alt_text: str = ""
    hash: str = ""
    metadata: Dict = field(default_factory=dict)
    
    @property
    def has_text(self) -> bool:
        return bool(self.ocr_text or self.caption or self.alt_text)
    
    def get_searchable_text(self) -> str:
        """Get all text associated with image for indexing."""
        parts = []
        if self.caption:
            parts.append(f"Figure: {self.caption}")
        if self.alt_text:
            parts.append(f"Description: {self.alt_text}")
        if self.ocr_text:
            parts.append(f"Content: {self.ocr_text}")
        return '\n'.join(parts)


# =============================================================================
# IMAGE PROCESSOR
# =============================================================================

class ImageProcessor:
    """
    Process images extracted from documents.
    
    Features:
    - OCR text extraction (EasyOCR or Tesseract)
    - Image preprocessing for better OCR
    - Caption and alt text extraction
    - Image deduplication via hashing
    - Thumbnail generation
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_languages: List[str] = None,
        save_images: bool = True,
        output_dir: str = ".docling_images",
        max_image_size: int = 2048,
        min_image_size: int = 50,
        thumbnail_size: Tuple[int, int] = (200, 200)
    ):
        """
        Initialize the image processor.
        
        Args:
            enable_ocr: Enable OCR text extraction
            ocr_languages: Languages for OCR (default: ['en'])
            save_images: Save extracted images to disk
            output_dir: Directory for saving images
            max_image_size: Maximum dimension for processing
            min_image_size: Minimum dimension to process
            thumbnail_size: Size for thumbnail generation
        """
        self.enable_ocr = enable_ocr and EASYOCR_AVAILABLE
        self.ocr_languages = ocr_languages or ['en']
        self.save_images = save_images
        self.output_dir = output_dir
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.thumbnail_size = thumbnail_size
        
        # Initialize OCR reader (lazy loading)
        self._ocr_reader = None
        
        # Image hash registry for deduplication
        self._image_hashes: Dict[str, str] = {}
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'ocr_performed': 0,
            'images_saved': 0,
            'duplicates_found': 0,
            'images_skipped': 0,
        }
    
    @property
    def ocr_reader(self):
        """Lazy-load OCR reader."""
        if self._ocr_reader is None and self.enable_ocr:
            try:
                self._ocr_reader = easyocr.Reader(
                    self.ocr_languages,
                    gpu=True  # Use GPU if available
                )
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.enable_ocr = False
        return self._ocr_reader
    
    def process_image(
        self,
        image: Any,
        page_number: int = 0,
        index: int = 0,
        caption: str = "",
        document_path: str = ""
    ) -> ImageInfo:
        """
        Process a single image.
        
        Args:
            image: PIL Image or path to image file
            page_number: Page number in source document
            index: Image index in document
            caption: Image caption if available
            document_path: Path to source document
            
        Returns:
            ImageInfo with extracted data
        """
        info = ImageInfo(
            index=index,
            page=page_number,
            caption=caption,
        )
        
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not PIL_AVAILABLE:
                    logger.warning("PIL not available for image processing")
                    return info
                image = Image.open(image)
            elif not PIL_AVAILABLE:
                logger.warning("PIL not available for image processing")
                return info
            
            # Get image properties
            info.width, info.height = image.size
            info.format = image.format or "PNG"
            
            # Check size constraints
            if info.width < self.min_image_size or info.height < self.min_image_size:
                self.stats['images_skipped'] += 1
                return info
            
            # Compute hash for deduplication
            info.hash = self._compute_image_hash(image)
            
            # Check for duplicate
            if info.hash in self._image_hashes:
                self.stats['duplicates_found'] += 1
                info.metadata['duplicate_of'] = self._image_hashes[info.hash]
                return info
            
            # Preprocess for OCR
            if self.enable_ocr:
                processed_image = self._preprocess_for_ocr(image)
                ocr_text = self._perform_ocr(processed_image)
                info.ocr_text = ocr_text
                self.stats['ocr_performed'] += 1
            
            # Save image
            if self.save_images:
                saved_path = self._save_image(
                    image, index, page_number, document_path
                )
                info.path = saved_path
                self.stats['images_saved'] += 1
            
            # Register hash
            self._image_hashes[info.hash] = info.path or f"image_{index}"
            
            self.stats['images_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing image {index}: {e}")
            info.metadata['error'] = str(e)
        
        return info
    
    def _compute_image_hash(self, image: 'Image.Image') -> str:
        """Compute perceptual hash for image."""
        # Resize to small size for hashing
        small = image.resize((16, 16), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        if small.mode != 'L':
            small = small.convert('L')
        
        # Get pixel data
        pixels = list(small.getdata())
        
        # Compute average
        avg = sum(pixels) / len(pixels)
        
        # Create hash based on pixel comparison to average
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        
        # Convert to hex
        return format(int(bits, 2), '064x')[:32]
    
    def _preprocess_for_ocr(self, image: 'Image.Image') -> 'Image.Image':
        """Preprocess image for better OCR results."""
        # Convert to RGB if needed
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Resize if too large
        if max(image.size) > self.max_image_size:
            ratio = self.max_image_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    
    def _perform_ocr(self, image: 'Image.Image') -> str:
        """Perform OCR on image."""
        if not self.ocr_reader:
            return ""
        
        try:
            # Convert PIL image to numpy array
            import numpy as np
            image_array = np.array(image)
            
            # Perform OCR
            results = self.ocr_reader.readtext(image_array)
            
            # Extract text
            texts = [result[1] for result in results]
            return ' '.join(texts)
            
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""
    
    def _save_image(
        self,
        image: 'Image.Image',
        index: int,
        page_number: int,
        document_path: str
    ) -> str:
        """Save image to disk."""
        # Create output directory
        if document_path:
            doc_name = Path(document_path).stem
            output_dir = Path(document_path).parent / self.output_dir / doc_name
        else:
            output_dir = Path(self.output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"page{page_number}_img{index}.png"
        filepath = output_dir / filename
        
        # Save
        image.save(str(filepath), "PNG")
        
        return str(filepath)
    
    def process_batch(
        self,
        images: List[Dict],
        document_path: str = ""
    ) -> List[ImageInfo]:
        """
        Process a batch of images from a document.
        
        Args:
            images: List of image dictionaries from Docling
            document_path: Path to source document
            
        Returns:
            List of ImageInfo objects
        """
        results = []
        
        for i, img_data in enumerate(images):
            # Extract image info from Docling format
            image = img_data.get('image')
            page = img_data.get('page', 0)
            caption = img_data.get('caption', '')
            
            if image is None and 'saved_path' in img_data:
                image = img_data['saved_path']
            
            if image is not None:
                info = self.process_image(
                    image=image,
                    page_number=page,
                    index=i,
                    caption=caption,
                    document_path=document_path
                )
                results.append(info)
        
        return results
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        for key in self.stats:
            self.stats[key] = 0


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_figure_references(text: str) -> List[Dict]:
    """
    Extract figure references from text.
    
    Finds patterns like:
    - "Figure 1", "Fig. 2.3"
    - "see Figure 5"
    - "(Figure 3)"
    
    Args:
        text: Document text
        
    Returns:
        List of figure reference dictionaries
    """
    patterns = [
        r'(?:Figure|Fig\.?)\s*(\d+(?:\.\d+)?)',
        r'(?:figure|fig\.?)\s*(\d+(?:\.\d+)?)',
        r'(?:Image|Img\.?)\s*(\d+(?:\.\d+)?)',
    ]
    
    references = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            references.append({
                'reference': match.group(0),
                'number': match.group(1),
                'position': match.start(),
            })
    
    return references


def create_image_index(images: List[ImageInfo]) -> Dict:
    """
    Create searchable index of images.
    
    Args:
        images: List of ImageInfo objects
        
    Returns:
        Index dictionary for search
    """
    index = {
        'by_page': {},
        'by_content': [],
        'with_text': [],
        'captions': [],
    }
    
    for img in images:
        # Index by page
        page = img.page
        if page not in index['by_page']:
            index['by_page'][page] = []
        index['by_page'][page].append(img)
        
        # Index by content
        if img.has_text:
            index['with_text'].append({
                'index': img.index,
                'page': img.page,
                'text': img.get_searchable_text(),
            })
        
        # Index captions
        if img.caption:
            index['captions'].append({
                'index': img.index,
                'caption': img.caption,
            })
    
    return index


def images_to_chunks(
    images: List[ImageInfo],
    min_text_length: int = 50
) -> List[Dict]:
    """
    Convert images with text to searchable chunks.
    
    Args:
        images: List of ImageInfo objects
        min_text_length: Minimum text length to create chunk
        
    Returns:
        List of chunk dictionaries
    """
    chunks = []
    
    for img in images:
        text = img.get_searchable_text()
        
        if len(text) >= min_text_length:
            chunks.append({
                'text': text,
                'pages': [img.page],
                'page_start': img.page,
                'page_end': img.page,
                'section': f"Figure {img.index + 1}",
                'source': img.path or f"image_{img.index}",
                'chunk_type': 'image',
                'char_count': len(text),
                'word_count': len(text.split()),
                'image_index': img.index,
                'has_ocr': bool(img.ocr_text),
            })
    
    return chunks

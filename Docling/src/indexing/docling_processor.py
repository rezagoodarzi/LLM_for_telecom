"""
import gc
try:
    import torch
except ImportError:
    torch = None
Docling Document Processor
===========================

Advanced document processing using Docling for superior:
- Table extraction and preservation
- Section hierarchy detection
- Image/figure handling with OCR
- Multi-format support (PDF, HTML, Markdown)
- Layout-aware text extraction

This processor integrates with the existing RAG system while providing
enhanced document understanding through AI-powered document analysis.

Key Features:
- Hierarchical document structure extraction
- Table-aware chunking (keeps tables intact)
- Image text extraction via OCR
- Duplicate detection using content fingerprinting
- Rich metadata extraction (titles, authors, dates)
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Generator
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

# Docling imports
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        EasyOcrOptions,
        TesseractOcrOptions,
    )
    from docling.datamodel.document import ConversionResult
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logging.warning("Docling not installed. Run: pip install docling")

from tqdm import tqdm

# Local imports
import sys
sys.path.append('..')
from config import settings
from utils.helpers import clean_text, extract_keywords, extract_phrases
from utils.metadata_extractor import MetadataExtractor


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DocumentElement:
    """Represents a document element (paragraph, table, image, etc.)"""
    element_type: str          # 'paragraph', 'table', 'image', 'heading', 'list'
    content: str               # Text content or markdown representation
    page_number: int = 0
    section: str = ""
    level: int = 0             # Heading level (1-6)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def is_heading(self) -> bool:
        return self.element_type == 'heading'
    
    @property
    def is_table(self) -> bool:
        return self.element_type == 'table'
    
    @property
    def is_image(self) -> bool:
        return self.element_type == 'image'


@dataclass
class DocumentStructure:
    """Hierarchical document structure"""
    title: str = ""
    authors: List[str] = field(default_factory=list)
    date: str = ""
    sections: List[Dict] = field(default_factory=list)
    elements: List[DocumentElement] = field(default_factory=list)
    tables: List[Dict] = field(default_factory=list)
    images: List[Dict] = field(default_factory=list)
    page_count: int = 0
    
    def get_section_hierarchy(self) -> Dict:
        """Returns nested section structure"""
        hierarchy = {}
        for section in self.sections:
            hierarchy[section['title']] = {
                'level': section.get('level', 1),
                'page': section.get('page', 0),
                'subsections': section.get('subsections', [])
            }
        return hierarchy


# =============================================================================
# DOCLING PROCESSOR
# =============================================================================

class DoclingProcessor:
    """
    Advanced document processor using Docling.
    
    Provides superior document understanding through:
    - AI-powered layout analysis
    - Table structure extraction
    - OCR for images and scanned content
    - Hierarchical section detection
    """
    
    def __init__(
        self,
        enable_ocr: bool = True,
        enable_table_extraction: bool = True,
        ocr_engine: str = "easyocr",  # 'easyocr' or 'tesseract'
        table_mode: str = "accurate",  # 'accurate' or 'fast'
        extract_images: bool = True,
        max_image_size: int = 1024,
        similarity_threshold: float = 0.92,
    ):
        """
        Initialize the Docling processor.
        
        Args:
            enable_ocr: Enable OCR for scanned PDFs/images
            enable_table_extraction: Extract tables with structure
            ocr_engine: OCR engine to use ('easyocr' or 'tesseract')
            table_mode: Table extraction mode ('accurate' or 'fast')
            extract_images: Extract and process images
            max_image_size: Maximum image dimension for processing
            similarity_threshold: Threshold for duplicate detection
        """
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Please run: pip install docling"
            )
        
        self.enable_ocr = enable_ocr
        self.enable_table_extraction = enable_table_extraction
        self.ocr_engine = ocr_engine
        self.table_mode = table_mode
        self.extract_images = extract_images
        self.max_image_size = max_image_size
        self.similarity_threshold = similarity_threshold
        
        # Initialize converter with options
        self.converter = self._create_converter()
        
        # Metadata extractor (from existing system)
        self.metadata_extractor = MetadataExtractor()
        
        # Statistics tracking
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'tables_extracted': 0,
            'images_processed': 0,
            'sections_detected': 0,
            'ocr_pages': 0,
            'duplicates_found': 0,
        }
        
        # Content fingerprints for duplicate detection
        self._content_fingerprints: Dict[str, str] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _create_converter(self) -> DocumentConverter:
        """Create Docling converter with configured options."""
        # Configure PDF pipeline
        pipeline_options = PdfPipelineOptions()
        
        # Table extraction settings
        if self.enable_table_extraction:
            if self.table_mode == "accurate":
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            else:
                pipeline_options.table_structure_options.mode = TableFormerMode.FAST
            pipeline_options.do_table_structure = True
        
        # OCR settings
        if self.enable_ocr:
            pipeline_options.do_ocr = True
            if self.ocr_engine == "easyocr":
                pipeline_options.ocr_options = EasyOcrOptions()
            else:
                pipeline_options.ocr_options = TesseractOcrOptions()
        
        # Image extraction
        pipeline_options.images_scale = 2.0
        pipeline_options.generate_page_images = self.extract_images
        pipeline_options.generate_picture_images = self.extract_images
        
        # Create converter
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        
        return converter
    
    def process_file(self, filepath: str) -> Optional[Dict]:
        """
        Process a single file and return structured document data.
        
        Args:
            filepath: Path to the document file
            
        Returns:
            Dictionary containing:
            - path: Source file path
            - metadata: Extracted metadata
            - structure: Document structure (sections, hierarchy)
            - chunks: List of text chunks with metadata
            - document_summary: Summary for document-level retrieval
            - tables: Extracted tables
            - images: Extracted image information
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        # Check supported formats
        if suffix not in settings.SUPPORTED_EXTENSIONS:
            self.logger.warning(f"Unsupported format: {suffix}")
            return None
        
        try:
            # Convert document using Docling
            result = self.converter.convert(filepath)
            
            if result is None or result.document is None:
                self.logger.warning(f"Failed to convert: {filepath}")
                return None
            
            # Extract structure
            structure = self._extract_structure(result)
            
            # Extract metadata
            metadata = self._extract_metadata(filepath, result, structure)
            
            # Check for duplicates
            content_hash = self._compute_content_hash(result)
            if self._is_duplicate(filepath, content_hash):
                self.stats['duplicates_found'] += 1
                self.logger.info(f"Duplicate detected: {filepath}")
                return None
            
            # Register content fingerprint
            self._content_fingerprints[filepath] = content_hash
            
            # Create hierarchical chunks
            chunks = self._create_chunks(result, structure, filepath, metadata)
            
            # Extract tables
            tables = self._extract_tables(result)
            
            # Extract images
            images = self._extract_images(result, filepath)
            
            # Create document summary
            doc_summary = self._create_document_summary(
                result, metadata, structure, tables
            )
            
            # Update statistics
            self.stats['documents_processed'] += 1
            self.stats['chunks_created'] += len(chunks)
            self.stats['tables_extracted'] += len(tables)
            self.stats['images_processed'] += len(images)
            self.stats['sections_detected'] += len(structure.sections)
            
            return {
                'path': str(path),
                'metadata': metadata,
                'structure': structure.__dict__,
                'chunks': chunks,
                'document_summary': doc_summary,
                'tables': tables,
                'images': images,
                'content_hash': content_hash,
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {filepath}: {e}")
            return None
    
    def _extract_structure(self, result) -> DocumentStructure:
        """Extract hierarchical document structure."""
        structure = DocumentStructure()
        
        current_section = ""
        current_level = 0
        
        for element in result.document.iterate_items():
            item = element[0] if isinstance(element, tuple) else element
            
            # Get element type and content
            elem_type = self._get_element_type(item)
            content = self._get_element_text(item)
            page_num = self._get_page_number(item)
            
            if elem_type == 'heading':
                level = self._get_heading_level(item)
                
                section_info = {
                    'title': content,
                    'level': level,
                    'page': page_num,
                    'subsections': []
                }
                structure.sections.append(section_info)
                current_section = content
                current_level = level
            
            # Create document element
            doc_element = DocumentElement(
                element_type=elem_type,
                content=content,
                page_number=page_num,
                section=current_section,
                level=current_level if elem_type == 'heading' else 0,
            )
            structure.elements.append(doc_element)
        
        # Extract title (first heading or from metadata)
        structure.title = self._extract_title(result, structure)
        
        # Count pages
        structure.page_count = self._get_page_count(result)
        
        return structure
    
    def _get_element_type(self, item: Any) -> str:
        """Determine element type from Docling item."""
        type_name = type(item).__name__.lower()
        
        type_mapping = {
            'heading': 'heading',
            'sectionheader': 'heading',
            'title': 'heading',
            'paragraph': 'paragraph',
            'text': 'paragraph',
            'table': 'table',
            'picture': 'image',
            'figure': 'image',
            'image': 'image',
            'listitem': 'list_item',
            'list': 'list',
            'caption': 'caption',
            'formula': 'formula',
            'equation': 'formula',
        }
        
        for key, value in type_mapping.items():
            if key in type_name:
                return value
        
        return 'paragraph'
    
    def _get_element_text(self, item: Any) -> str:
        """Extract text content from Docling item."""
        # Try different attributes
        if hasattr(item, 'text'):
            return str(item.text)
        if hasattr(item, 'export_to_markdown'):
            return item.export_to_markdown()
        if hasattr(item, 'content'):
            return str(item.content)
        return str(item)
    
    def _get_page_number(self, item: Any) -> int:
        """Get page number from Docling item."""
        if hasattr(item, 'prov') and item.prov:
            for prov in item.prov:
                if hasattr(prov, 'page_no'):
                    return prov.page_no
        if hasattr(item, 'page'):
            return item.page
        return 0
    
    def _get_heading_level(self, item: Any) -> int:
        """Determine heading level (1-6)."""
        if hasattr(item, 'level'):
            return min(max(item.level, 1), 6)
        
        # Infer from text formatting
        text = self._get_element_text(item)
        
        # Check markdown-style headings
        if text.startswith('#'):
            return min(len(text.split()[0]), 6)
        
        # Check numbered sections
        match = re.match(r'^(\d+(?:\.\d+)*)\s', text)
        if match:
            return min(match.group(1).count('.') + 1, 6)
        
        return 1
    
    def _get_page_count(self, result) -> int:
        """Get total page count from result."""
        if hasattr(result.document, 'pages'):
            return len(result.document.pages)
        
        # Count from elements
        max_page = 0
        for element in result.document.iterate_items():
            item = element[0] if isinstance(element, tuple) else element
            page = self._get_page_number(item)
            max_page = max(max_page, page)
        
        return max_page or 1
    
    def _extract_title(
        self, 
        result, 
        structure: DocumentStructure
    ) -> str:
        """Extract document title."""
        # Try document metadata first
        if hasattr(result.document, 'title') and result.document.title:
            return result.document.title
        
        # Try first heading
        for element in structure.elements:
            if element.is_heading:
                return element.content
        
        # Use filename as fallback
        if hasattr(result, 'input') and result.input:
            return Path(str(result.input.file)).stem
        
        return "Untitled Document"
    
    def _extract_metadata(
        self,
        filepath: str,
        result,
        structure: DocumentStructure
    ) -> Dict:
        """Extract comprehensive metadata."""
        # Start with filename-based metadata
        metadata = self.metadata_extractor.extract_from_filename(filepath)
        
        # Document properties
        metadata['title'] = structure.title
        metadata['page_count'] = structure.page_count
        metadata['section_count'] = len(structure.sections)
        
        # Extract from content
        full_text = self._get_full_text(result)
        metadata = self.metadata_extractor.extract_from_content(full_text, metadata)
        
        # Document metadata from Docling
        if hasattr(result.document, 'metadata'):
            doc_meta = result.document.metadata
            if hasattr(doc_meta, 'author'):
                metadata['authors'] = doc_meta.author
            if hasattr(doc_meta, 'created'):
                metadata['created_date'] = str(doc_meta.created)
            if hasattr(doc_meta, 'modified'):
                metadata['modified_date'] = str(doc_meta.modified)
        
        # Processing timestamp
        metadata['processed_at'] = datetime.now().isoformat()
        
        return metadata
    
    def _get_full_text(self, result) -> str:
        """Get full document text."""
        parts = []
        for element in result.document.iterate_items():
            item = element[0] if isinstance(element, tuple) else element
            text = self._get_element_text(item)
            if text:
                parts.append(text)
        return '\n'.join(parts)
    
    def _compute_content_hash(self, result) -> str:
        """Compute content hash for duplicate detection."""
        text = self._get_full_text(result)
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]
    
    def _is_duplicate(self, filepath: str, content_hash: str) -> bool:
        """Check if document is a duplicate based on content hash."""
        for other_path, other_hash in self._content_fingerprints.items():
            if content_hash == other_hash and other_path != filepath:
                return True
        return False
    
    def _create_chunks(
        self,
        result,
        structure: DocumentStructure,
        filepath: str,
        metadata: Dict
    ) -> List[Dict]:
        """
        Create hierarchical chunks respecting document structure.
        
        Key features:
        - Tables are kept intact as single chunks
        - Sections are not split mid-content
        - Overlap is added for context continuity
        - Rich metadata attached to each chunk
        """
        chunks = []
        current_chunk = ""
        current_pages = []
        current_section = ""
        chunk_idx = 0
        
        for element in structure.elements:
            # Handle tables specially - keep intact
            if element.is_table:
                # Save current chunk first
                if current_chunk and len(current_chunk) >= settings.MIN_CHUNK_SIZE:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, current_pages, current_section,
                        filepath, metadata, chunk_idx, 'text'
                    ))
                    chunk_idx += 1
                    current_chunk = ""
                    current_pages = []
                
                # Add table as separate chunk
                chunks.append(self._create_chunk_dict(
                    element.content, [element.page_number], element.section,
                    filepath, metadata, chunk_idx, 'table'
                ))
                chunk_idx += 1
                continue
            
            # Handle section changes
            if element.is_heading:
                # Save current chunk on section change
                if current_chunk and len(current_chunk) >= settings.MIN_CHUNK_SIZE:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, current_pages, current_section,
                        filepath, metadata, chunk_idx, 'text'
                    ))
                    chunk_idx += 1
                    current_chunk = ""
                    current_pages = []
                
                current_section = element.content
            
            # Accumulate content
            content = element.content
            page = element.page_number
            
            # Check if adding this would exceed max size
            if len(current_chunk) + len(content) + 1 > settings.MAX_CHUNK_SIZE:
                # Save current chunk
                if len(current_chunk) >= settings.MIN_CHUNK_SIZE:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, current_pages, current_section,
                        filepath, metadata, chunk_idx, 'text'
                    ))
                    chunk_idx += 1
                
                # Start new chunk with overlap
                overlap = current_chunk[-settings.CHUNK_OVERLAP:] if current_chunk else ""
                current_chunk = overlap + "\n" + content if overlap else content
                current_pages = [page]
            else:
                current_chunk += "\n" + content if current_chunk else content
                if page not in current_pages:
                    current_pages.append(page)
            
            # Check if we should split (reached target size)
            if len(current_chunk) >= settings.CHUNK_SIZE:
                chunks.append(self._create_chunk_dict(
                    current_chunk, current_pages, current_section,
                    filepath, metadata, chunk_idx, 'text'
                ))
                chunk_idx += 1
                
                # Keep overlap
                overlap = current_chunk[-settings.CHUNK_OVERLAP:]
                current_chunk = overlap
                current_pages = [page] if page else []
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= settings.MIN_CHUNK_SIZE:
            chunks.append(self._create_chunk_dict(
                current_chunk, current_pages, current_section,
                filepath, metadata, chunk_idx, 'text'
            ))
        
        return chunks
    
    def _create_chunk_dict(
        self,
        text: str,
        pages: List[int],
        section: str,
        filepath: str,
        doc_metadata: Dict,
        idx: int,
        chunk_type: str = 'text'
    ) -> Dict:
        """Create a chunk dictionary with comprehensive metadata."""
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Extract keywords
        keywords_upper, keywords_lower = extract_keywords(cleaned_text)
        phrases = extract_phrases(cleaned_text)
        
        return {
            'text': cleaned_text,
            'pages': pages.copy() if pages else [0],
            'page_start': min(pages) if pages else 0,
            'page_end': max(pages) if pages else 0,
            'section': section,
            'source': Path(filepath).name,
            'source_path': filepath,
            'chunk_id': f"{Path(filepath).stem}_chunk_{idx}",
            'chunk_type': chunk_type,  # 'text', 'table', 'image'
            'char_count': len(cleaned_text),
            'word_count': len(cleaned_text.split()),
            'keywords': keywords_upper[:15],
            'keywords_lower': keywords_lower[:15],
            'phrases': phrases[:10],
            # Document-level metadata
            'vendor': doc_metadata.get('vendor'),
            'version': doc_metadata.get('version'),
            'release': doc_metadata.get('release'),
            'doc_type': doc_metadata.get('doc_type'),
            'topics': doc_metadata.get('topics', []),
            'title': doc_metadata.get('title', ''),
        }
    
    def _extract_tables(self, result) -> List[Dict]:
        """Extract tables with structure preservation."""
        tables = []
        table_idx = 0
        
        for element in result.document.iterate_items():
            item = element[0] if isinstance(element, tuple) else element
            
            if self._get_element_type(item) == 'table':
                table_data = {
                    'index': table_idx,
                    'page': self._get_page_number(item),
                    'markdown': self._get_element_text(item),
                }
                
                # Try to extract structured data
                if hasattr(item, 'export_to_dataframe'):
                    try:
                        df = item.export_to_dataframe()
                        table_data['headers'] = list(df.columns)
                        table_data['rows'] = df.values.tolist()
                        table_data['row_count'] = len(df)
                        table_data['col_count'] = len(df.columns)
                    except Exception:
                        pass
                
                # Get caption if available
                if hasattr(item, 'caption'):
                    table_data['caption'] = str(item.caption)
                
                tables.append(table_data)
                table_idx += 1
        
        return tables
    
    def _extract_images(
        self, 
        result, 
        filepath: str
    ) -> List[Dict]:
        """Extract images with OCR text if available."""
        images = []
        image_idx = 0
        output_dir = Path(filepath).parent / '.docling_images'
        
        for element in result.document.iterate_items():
            item = element[0] if isinstance(element, tuple) else element
            
            if self._get_element_type(item) == 'image':
                image_data = {
                    'index': image_idx,
                    'page': self._get_page_number(item),
                }
                
                # Get OCR text if available
                if hasattr(item, 'text') and item.text:
                    image_data['ocr_text'] = str(item.text)
                    self.stats['ocr_pages'] += 1
                
                # Get caption if available
                if hasattr(item, 'caption'):
                    image_data['caption'] = str(item.caption)
                
                # Save image if extraction is enabled
                if self.extract_images and hasattr(item, 'image'):
                    try:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        image_path = output_dir / f"image_{image_idx}.png"
                        item.image.save(str(image_path))
                        image_data['saved_path'] = str(image_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to save image: {e}")
                
                images.append(image_data)
                image_idx += 1
        
        return images
    
    def _create_document_summary(
        self,
        result,
        metadata: Dict,
        structure: DocumentStructure,
        tables: List[Dict]
    ) -> str:
        """Create document-level summary for 2-stage retrieval."""
        parts = []
        
        # Title
        if metadata.get('title'):
            parts.append(f"Title: {metadata['title']}")
        
        # Vendor
        if metadata.get('vendor'):
            parts.append(f"Vendor: {metadata['vendor']}")
        
        # Version/Release
        if metadata.get('version'):
            parts.append(f"Version: {metadata['version']}")
        if metadata.get('release'):
            parts.append(f"Release: {metadata['release']}")
        
        # Spec number
        if metadata.get('spec_number'):
            parts.append(f"Specification: {metadata['spec_number']}")
        
        # Topics
        if metadata.get('topics'):
            parts.append(f"Topics: {', '.join(metadata['topics'][:10])}")
        
        # Section titles (limited)
        section_titles = [s['title'] for s in structure.sections[:15]]
        if section_titles:
            parts.append(f"Sections: {'; '.join(section_titles)}")
        
        # Table summaries
        if tables:
            table_info = [t.get('caption', f"Table {t['index']+1}") for t in tables[:5]]
            parts.append(f"Tables: {'; '.join(table_info)}")
        
        # First content (abstract/overview)
        for element in structure.elements[:10]:
            if element.element_type == 'paragraph' and len(element.content) > 100:
                parts.append(f"Overview: {element.content[:500]}")
                break
        
        return '\n'.join(parts)
    
    def get_stats(self) -> Dict:
        """Return processing statistics."""
        return dict(self.stats)
    
    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0


# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def process_directory_with_docling(
    docs_dir: str,
    extensions: List[str] = None,
    enable_ocr: bool = True,
    enable_table_extraction: bool = True,
    similarity_threshold: float = 0.92,
    progress_callback = None
) -> Tuple[List[Dict], Dict]:
    """
    Process all documents in a directory using Docling.
    
    Args:
        docs_dir: Directory containing documents
        extensions: File extensions to process
        enable_ocr: Enable OCR for scanned content
        enable_table_extraction: Extract tables with structure
        similarity_threshold: Threshold for duplicate detection
        progress_callback: Optional callback for progress updates
        
    Returns:
        (list of processed documents, statistics dict)
    """
    if extensions is None:
        extensions = settings.SUPPORTED_EXTENSIONS
    
    # Initialize processor
    processor = DoclingProcessor(
        enable_ocr=enable_ocr,
        enable_table_extraction=enable_table_extraction,
        similarity_threshold=similarity_threshold,
    )
    
    docs_path = Path(docs_dir)
    
    # Find all files
    all_files = []
    for ext in extensions:
        all_files.extend(list(docs_path.rglob(f"*{ext}")))
    
    print(f"Found {len(all_files):,} files")
    
    # Process files
    results = []
    errors = []
    
    for filepath in tqdm(all_files, desc="Processing with Docling"):
        try:
            result = processor.process_file(str(filepath))
            if result:
                results.append(result)
                
            if progress_callback:
                progress_callback(len(results), len(all_files))
                
        except Exception as e:
            errors.append({'file': str(filepath), 'error': str(e)})
            logging.error(f"Error processing {filepath}: {e}")
    
    # Collect statistics
    stats = processor.get_stats()
    stats['files_found'] = len(all_files)
    stats['files_processed'] = len(results)
    stats['files_failed'] = len(errors)
    stats['errors'] = errors
    
    print(f"\nProcessing Complete")
    print(f"   Documents processed: {len(results):,}")
    print(f"   Total chunks: {stats['chunks_created']:,}")
    print(f"   Tables extracted: {stats['tables_extracted']:,}")
    print(f"   Images processed: {stats['images_processed']:,}")
    print(f"   Sections detected: {stats['sections_detected']:,}")
    print(f"   Duplicates skipped: {stats['duplicates_found']:,}")
    
    return results, stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_single_document(
    filepath: str,
    enable_ocr: bool = True,
    enable_table_extraction: bool = True
) -> Optional[Dict]:
    """
    Convert a single document using Docling.
    
    Convenience function for quick document conversion.
    
    Args:
        filepath: Path to the document
        enable_ocr: Enable OCR
        enable_table_extraction: Extract tables
        
    Returns:
        Processed document dictionary or None
    """
    processor = DoclingProcessor(
        enable_ocr=enable_ocr,
        enable_table_extraction=enable_table_extraction,
    )
    return processor.process_file(filepath)


def check_docling_installation() -> Dict[str, bool]:
    """
    Check Docling installation and available features.
    
    Returns:
        Dictionary with feature availability
    """
    features = {
        'docling_available': DOCLING_AVAILABLE,
        'easyocr_available': False,
        'tesseract_available': False,
        'torch_available': False,
    }
    
    try:
        import easyocr
        features['easyocr_available'] = True
    except ImportError:
        pass
    
    try:
        import pytesseract
        features['tesseract_available'] = True
    except ImportError:
        pass
    
    try:
        import torch
        features['torch_available'] = True
        features['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    return features


if __name__ == "__main__":
    # Quick test
    features = check_docling_installation()
    print("Docling Installation Status:")
    for feature, available in features.items():
        status = "OK" if available else "MISSING"
        print(f"  {status} {feature}")

def clear_gpu_memory():
    " \\Clear GPU and CPU memory.\\\

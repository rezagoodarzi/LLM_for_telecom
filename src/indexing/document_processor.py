"""
Document Processor with Hierarchical Chunking
==============================================

Implements proper chunking strategy for technical documents:
1. Extract document structure (sections)
2. Create hierarchical chunks (document → section → chunk)
3. Preserve context across chunk boundaries
4. Extract rich metadata for filtering

Key improvements over naive chunking:
- Section-aware boundaries (doesn't split mid-section)
- Smaller chunks (2000 chars) for better retrieval precision
- Vendor/version metadata extraction
- Document-level indexing for 2-stage retrieval
"""

import os
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.append('..')
from config import settings
from utils.helpers import detect_section, clean_text, extract_keywords, extract_phrases
from utils.metadata_extractor import MetadataExtractor
from utils.document_cleaner import DocumentCleaner, filter_noise_pages


class DocumentProcessor:
    """Process documents with hierarchical chunking"""
    
    def __init__(self, enable_noise_filtering: bool = True):
        self.metadata_extractor = MetadataExtractor()
        self.enable_noise_filtering = enable_noise_filtering
        self.stats = {
            'documents_processed': 0,
            'chunks_created': 0,
            'sections_detected': 0,
            'pages_removed_noise': 0,
        }
    
    def process_file(self, filepath: str) -> Optional[Dict]:
        """
        Process a single file and return structured document data.
        
        Returns:
            {
                'path': str,
                'metadata': {...},
                'sections': [...],
                'chunks': [...],
                'document_text': str (for document-level retrieval)
            }
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        # Extract pages/content
        if suffix == '.pdf':
            pages = self._extract_pdf(filepath)
        elif suffix in ['.html', '.htm']:
            pages = self._extract_html(filepath)
        elif suffix == '.md':
            pages = self._extract_markdown(filepath)
        elif suffix == '.txt':
            pages = self._extract_text(filepath)
        else:
            return None
        
        if not pages:
            return None
        
        # Filter noise pages (TOC, legal, revision history)
        original_page_count = len(pages)
        if self.enable_noise_filtering:
            pages, noise_stats = filter_noise_pages(pages)
            self.stats['pages_removed_noise'] += noise_stats.get('removed_pages', 0)
        
        if not pages:
            return None
        
        # Extract metadata from filename
        metadata = self.metadata_extractor.extract_from_filename(filepath)
        metadata['original_pages'] = original_page_count
        metadata['filtered_pages'] = len(pages)
        
        # Extract document structure
        structure = self._extract_structure(pages)
        
        # Enhance metadata from content
        full_text = '\n'.join(pages)
        metadata = self.metadata_extractor.extract_from_content(full_text, metadata)
        
        # Create hierarchical chunks
        chunks = self._create_hierarchical_chunks(pages, structure, filepath, metadata)
        
        # Create document summary for document-level retrieval
        doc_summary = self._create_document_summary(pages, metadata, structure)
        
        self.stats['documents_processed'] += 1
        self.stats['chunks_created'] += len(chunks)
        self.stats['sections_detected'] += len(structure.get('sections', []))
        
        return {
            'path': str(path),
            'metadata': metadata,
            'structure': structure,
            'chunks': chunks,
            'document_summary': doc_summary,
        }
    
    def _extract_pdf(self, filepath: str) -> List[str]:
        """Extract text from PDF with page preservation"""
        try:
            doc = fitz.open(filepath)
            pages = []
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text("text")
                if text.strip():
                    pages.append(text)
            doc.close()
            return pages
        except Exception as e:
            print(f"Error extracting PDF {filepath}: {e}")
            return []
    
    def _extract_html(self, filepath: str) -> List[str]:
        """Extract text from HTML"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                html = f.read()
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove scripts and styles
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            
            text = soup.get_text(separator='\n')
            text = clean_text(text)
            return [text] if text else []
        except Exception as e:
            print(f"Error extracting HTML {filepath}: {e}")
            return []
    
    def _extract_markdown(self, filepath: str) -> List[str]:
        """Extract text from Markdown"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Remove code blocks
            text = re.sub(r'```[\s\S]*?```', '', content)
            text = re.sub(r'`[^`]+`', '', text)
            # Remove images
            text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
            # Convert links to text
            text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
            # Remove formatting
            text = re.sub(r'[*_~]+', '', text)
            
            return [text] if text.strip() else []
        except Exception as e:
            print(f"Error extracting Markdown {filepath}: {e}")
            return []
    
    def _extract_text(self, filepath: str) -> List[str]:
        """Extract plain text"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return [content] if content.strip() else []
        except Exception as e:
            print(f"Error extracting text {filepath}: {e}")
            return []
    
    def _extract_structure(self, pages: List[str]) -> Dict:
        """Extract document structure (sections, hierarchy)"""
        structure = {
            'sections': [],
            'page_sections': {},
            'section_pages': defaultdict(list),
        }
        
        current_section = "Introduction"
        
        for page_num, page_text in enumerate(pages):
            lines = page_text.split('\n')
            
            # Check first 30 lines of each page for section headers
            for line in lines[:30]:
                section = detect_section(line)
                if section:
                    current_section = section['text']
                    structure['sections'].append({
                        'page': page_num + 1,
                        'title': current_section,
                        'type': section['type']
                    })
            
            structure['page_sections'][page_num + 1] = current_section
            structure['section_pages'][current_section].append(page_num + 1)
        
        return structure
    
    def _create_hierarchical_chunks(
        self, 
        pages: List[str], 
        structure: Dict, 
        filepath: str,
        metadata: Dict
    ) -> List[Dict]:
        """
        Create chunks with proper boundaries:
        1. Respect section boundaries
        2. Keep chunks within CHUNK_SIZE
        3. Include overlap for context
        4. Attach rich metadata
        """
        chunks = []
        current_chunk = ""
        current_pages = []
        current_section = None
        chunk_idx = 0
        
        for page_num, page_text in enumerate(pages):
            page_id = page_num + 1
            section = structure['page_sections'].get(page_id, "Unknown")
            
            # Section change - save current chunk and start new
            if section != current_section and current_chunk:
                if len(current_chunk) >= settings.MIN_CHUNK_SIZE:
                    chunks.append(self._create_chunk_dict(
                        current_chunk, current_pages, current_section,
                        filepath, metadata, chunk_idx
                    ))
                    chunk_idx += 1
                current_chunk = ""
                current_pages = []
            
            current_section = section
            
            # Split page text into smaller pieces if needed
            page_chunks = self._split_text(page_text, settings.CHUNK_SIZE)
            
            for piece in page_chunks:
                if len(current_chunk) + len(piece) <= settings.MAX_CHUNK_SIZE:
                    current_chunk += f"\n[Page {page_id}]\n{piece}" if current_chunk else piece
                    if page_id not in current_pages:
                        current_pages.append(page_id)
                else:
                    # Save current chunk
                    if len(current_chunk) >= settings.MIN_CHUNK_SIZE:
                        chunks.append(self._create_chunk_dict(
                            current_chunk, current_pages, current_section,
                            filepath, metadata, chunk_idx
                        ))
                        chunk_idx += 1
                    
                    # Start new chunk with overlap
                    overlap = current_chunk[-settings.CHUNK_OVERLAP:] if len(current_chunk) > settings.CHUNK_OVERLAP else ""
                    current_chunk = overlap + piece
                    current_pages = [page_id]
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) >= settings.MIN_CHUNK_SIZE:
            chunks.append(self._create_chunk_dict(
                current_chunk, current_pages, current_section,
                filepath, metadata, chunk_idx
            ))
        
        return chunks
    
    def _split_text(self, text: str, max_size: int) -> List[str]:
        """Split text into pieces respecting sentence boundaries"""
        if len(text) <= max_size:
            return [text]
        
        pieces = []
        
        # Try to split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_piece = ""
        for sentence in sentences:
            if len(current_piece) + len(sentence) <= max_size:
                current_piece += " " + sentence if current_piece else sentence
            else:
                if current_piece:
                    pieces.append(current_piece.strip())
                
                # Handle very long sentences
                if len(sentence) > max_size:
                    # Force split
                    for i in range(0, len(sentence), max_size):
                        pieces.append(sentence[i:i+max_size])
                    current_piece = ""
                else:
                    current_piece = sentence
        
        if current_piece:
            pieces.append(current_piece.strip())
        
        return pieces
    
    def _create_chunk_dict(
        self,
        text: str,
        pages: List[int],
        section: str,
        filepath: str,
        doc_metadata: Dict,
        idx: int
    ) -> Dict:
        """Create a chunk dictionary with all metadata"""
        # Extract chunk-level keywords
        keywords_upper, keywords_lower = extract_keywords(text)
        phrases = extract_phrases(text)
        
        return {
            'text': clean_text(text),
            'pages': pages.copy(),
            'page_start': min(pages) if pages else 0,
            'page_end': max(pages) if pages else 0,
            'section': section,
            'source': Path(filepath).name,
            'source_path': filepath,
            'chunk_id': f"{Path(filepath).stem}_chunk_{idx}",
            'char_count': len(text),
            'word_count': len(text.split()),
            'keywords': keywords_upper[:15],
            'keywords_lower': keywords_lower[:15],
            'phrases': phrases[:10],
            # Document-level metadata (for filtering)
            'vendor': doc_metadata.get('vendor'),
            'version': doc_metadata.get('version'),
            'release': doc_metadata.get('release'),
            'doc_type': doc_metadata.get('doc_type'),
            'topics': doc_metadata.get('topics', []),
        }
    
    def _create_document_summary(
        self, 
        pages: List[str], 
        metadata: Dict, 
        structure: Dict
    ) -> str:
        """
        Create a document-level summary for 2-stage retrieval.
        This is indexed separately to find relevant documents first.
        """
        parts = []
        
        # Title/name
        if metadata.get('title'):
            parts.append(f"Title: {metadata['title']}")
        
        # Vendor
        if metadata.get('vendor'):
            parts.append(f"Vendor: {metadata['vendor']}")
        
        # Version
        if metadata.get('version'):
            parts.append(f"Version: {metadata['version']}")
        
        # Spec number
        if metadata.get('spec_number'):
            parts.append(f"Specification: {metadata['spec_number']}")
        
        # Topics
        if metadata.get('topics'):
            parts.append(f"Topics: {', '.join(metadata['topics'])}")
        
        # Section titles
        section_titles = [s['title'] for s in structure.get('sections', [])[:20]]
        if section_titles:
            parts.append(f"Sections: {'; '.join(section_titles)}")
        
        # First page content (usually contains abstract/overview)
        if pages:
            first_page = pages[0][:1000]
            parts.append(f"Content preview: {first_page}")
        
        return '\n'.join(parts)


def process_directory(
    docs_dir: str,
    extensions: List[str] = None,
    enable_version_selection: bool = True,
    enable_deduplication: bool = True,
    enable_noise_filtering: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Process all documents in a directory with pre-cleaning.
    
    Cleaning steps:
    1. Version selection (keep newest per document family)
    2. Document deduplication (MinHash similarity)
    3. Noise page filtering (TOC, legal, revision)
    
    Returns:
        (list of processed documents, statistics dict)
    """
    if extensions is None:
        extensions = settings.SUPPORTED_EXTENSIONS
    
    processor = DocumentProcessor(enable_noise_filtering=enable_noise_filtering)
    cleaner = DocumentCleaner(
        enable_version_selection=enable_version_selection,
        enable_deduplication=enable_deduplication,
        enable_noise_filtering=False,  # Done at page level
        dedup_threshold=0.80
    )
    
    docs_path = Path(docs_dir)
    
    # Find all files
    all_files = []
    for ext in extensions:
        all_files.extend(list(docs_path.rglob(f"*{ext}")))
    
    print(f"📂 Found {len(all_files):,} files")
    
    # Phase 1: Version selection (if enabled)
    if enable_version_selection:
        print(f"\n🔄 Phase 1: Version Selection...")
        for filepath in tqdm(all_files, desc="Scanning versions"):
            cleaner.should_process_file(str(filepath))
        
        files_to_process = cleaner.select_files([str(f) for f in all_files])
        version_stats = cleaner.stats.get('version_selection', {})
        
        print(f"   ✓ Document families: {version_stats.get('total_families', 0):,}")
        print(f"   ✓ Keeping newest: {version_stats.get('documents_kept', 0):,}")
        print(f"   ✓ Older versions removed: {version_stats.get('documents_removed', 0):,}")
    else:
        files_to_process = [str(f) for f in all_files]
    
    # Phase 2: Process files with deduplication
    print(f"\n📄 Phase 2: Processing {len(files_to_process):,} files...")
    results = []
    duplicates_skipped = 0
    
    for filepath in tqdm(files_to_process, desc="Processing documents"):
        try:
            result = processor.process_file(filepath)
            if result:
                # Check for duplicate content
                if enable_deduplication:
                    full_text = '\n'.join([c['text'] for c in result['chunks']])
                    is_dup, dup_of = cleaner.is_duplicate_document(filepath, full_text)
                    if is_dup:
                        duplicates_skipped += 1
                        continue
                
                results.append(result)
        except Exception as e:
            print(f"\n⚠️ Error processing {filepath}: {e}")
    
    # Collect stats
    cleaner_stats = cleaner.get_stats()
    
    combined_stats = {
        **processor.stats,
        'files_found': len(all_files),
        'files_after_version_selection': len(files_to_process),
        'duplicates_skipped': duplicates_skipped,
        'cleaning_stats': cleaner_stats,
    }
    
    print(f"\n✅ Processing Complete")
    print(f"   Documents processed: {len(results):,}")
    print(f"   Total chunks: {processor.stats['chunks_created']:,}")
    print(f"   Sections detected: {processor.stats['sections_detected']:,}")
    print(f"   Noise pages removed: {processor.stats['pages_removed_noise']:,}")
    print(f"   Duplicates skipped: {duplicates_skipped:,}")
    
    return results, combined_stats


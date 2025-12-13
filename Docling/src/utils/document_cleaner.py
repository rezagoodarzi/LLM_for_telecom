"""
Document Cleaner - Pre-indexing Cleanup
========================================

Implements:
1. Noise page filtering (TOC, legal, revision history)
2. Document-level deduplication (MinHash)
3. Version selection (keep newest only)

This runs BEFORE chunking to eliminate duplicates and noise.
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path


# =============================================================================
# NOISE PAGE DETECTION
# =============================================================================

# Patterns that indicate noise pages (should be filtered out)
NOISE_PATTERNS = [
    # Table of Contents
    r'^table\s+of\s+contents',
    r'^contents$',
    r'^\d+\.\s+.*\.{3,}\s*\d+$',  # TOC entry pattern: "1. Introduction....... 5"
    
    # Legal/Copyright
    r'copyright\s*©?\s*\d{4}',
    r'all\s+rights\s+reserved',
    r'confidential',
    r'proprietary',
    r'trademark',
    r'®|™',
    
    # Revision/Change history
    r'^revision\s+history',
    r'^change\s+history',
    r'^document\s+history',
    r'^version\s+history',
    r'^\s*rev\s+date\s+author',
    
    # Company boilerplate
    r'ericsson\s+ab',
    r'nokia\s+(solutions|networks|corporation)',
    r'huawei\s+technologies',
    r'samsung\s+electronics',
    
    # Glossary (when standalone)
    r'^glossary$',
    r'^abbreviations$',
    r'^acronyms$',
    r'^definitions$',
    
    # Empty/placeholder
    r'^this\s+page\s+(is\s+)?intentionally\s+(left\s+)?blank',
    r'^\s*page\s+\d+\s+of\s+\d+\s*$',
]

# Compile patterns for efficiency
NOISE_REGEX = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in NOISE_PATTERNS]

# Minimum content threshold (pages with less than this are likely noise)
MIN_PAGE_CONTENT_CHARS = 200
MIN_PAGE_CONTENT_WORDS = 30


def is_noise_page(page_text: str) -> Tuple[bool, str]:
    """
    Determine if a page is noise (TOC, legal, revision, etc.)
    
    Returns:
        (is_noise: bool, reason: str)
    """
    text_lower = page_text.lower().strip()
    
    # Check if page is too short
    if len(text_lower) < MIN_PAGE_CONTENT_CHARS:
        return True, "too_short"
    
    word_count = len(text_lower.split())
    if word_count < MIN_PAGE_CONTENT_WORDS:
        return True, "few_words"
    
    # Check noise patterns
    for pattern in NOISE_REGEX:
        if pattern.search(text_lower):
            # Get the pattern name for debugging
            pattern_str = pattern.pattern[:30]
            return True, f"pattern:{pattern_str}"
    
    # Check if mostly dots/numbers (TOC-like)
    dots_and_nums = len(re.findall(r'[.\d]', text_lower))
    if dots_and_nums / len(text_lower) > 0.3:
        # More than 30% dots/numbers = likely TOC
        lines = text_lower.split('\n')
        toc_like_lines = sum(1 for l in lines if re.match(r'.*\.{3,}\s*\d+\s*$', l))
        if toc_like_lines > len(lines) * 0.5:
            return True, "toc_format"
    
    return False, ""


def filter_noise_pages(pages: List[str]) -> Tuple[List[str], Dict]:
    """
    Filter noise pages from a document.
    
    Returns:
        (filtered_pages, statistics)
    """
    filtered = []
    stats = {
        'total_pages': len(pages),
        'kept_pages': 0,
        'removed_pages': 0,
        'removal_reasons': defaultdict(int)
    }
    
    for i, page in enumerate(pages):
        is_noise, reason = is_noise_page(page)
        
        if is_noise:
            stats['removed_pages'] += 1
            stats['removal_reasons'][reason] += 1
        else:
            filtered.append(page)
            stats['kept_pages'] += 1
    
    return filtered, dict(stats)


# =============================================================================
# DOCUMENT-LEVEL DEDUPLICATION (MinHash)
# =============================================================================

class MinHasher:
    """
    MinHash implementation for document deduplication.
    
    Uses k-shingle hashing to create document fingerprints,
    then compares Jaccard similarity.
    """
    
    def __init__(self, num_hashes: int = 100, shingle_size: int = 5):
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        # Generate hash functions (using different seeds)
        self.hash_seeds = [i * 31337 for i in range(num_hashes)]
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Extract k-shingles from text"""
        # Normalize text
        text = re.sub(r'\s+', ' ', text.lower().strip())
        words = text.split()
        
        if len(words) < self.shingle_size:
            return {text}
        
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = ' '.join(words[i:i + self.shingle_size])
            shingles.add(shingle)
        
        return shingles
    
    def _hash_shingle(self, shingle: str, seed: int) -> int:
        """Hash a shingle with a specific seed"""
        h = hashlib.md5(f"{seed}{shingle}".encode()).hexdigest()
        return int(h, 16)
    
    def compute_signature(self, text: str) -> List[int]:
        """Compute MinHash signature for text"""
        shingles = self._get_shingles(text)
        
        if not shingles:
            return [0] * self.num_hashes
        
        signature = []
        for seed in self.hash_seeds:
            min_hash = min(self._hash_shingle(s, seed) for s in shingles)
            signature.append(min_hash)
        
        return signature
    
    def estimate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """Estimate Jaccard similarity from signatures"""
        if len(sig1) != len(sig2):
            return 0.0
        
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)


class DocumentDeduplicator:
    """
    Document-level deduplication using MinHash.
    
    Identifies and removes duplicate documents BEFORE chunking.
    """
    
    def __init__(self, similarity_threshold: float = 0.80):
        self.threshold = similarity_threshold
        self.hasher = MinHasher(num_hashes=100, shingle_size=5)
        self.signatures: Dict[str, List[int]] = {}
        self.document_groups: Dict[str, List[str]] = defaultdict(list)
    
    def add_document(self, doc_id: str, text: str) -> Tuple[bool, Optional[str]]:
        """
        Add document and check for duplicates.
        
        Returns:
            (is_unique: bool, duplicate_of: Optional[str])
        """
        # Compute signature
        signature = self.hasher.compute_signature(text)
        
        # Check against existing documents
        for existing_id, existing_sig in self.signatures.items():
            similarity = self.hasher.estimate_similarity(signature, existing_sig)
            
            if similarity >= self.threshold:
                self.document_groups[existing_id].append(doc_id)
                return False, existing_id
        
        # Unique document
        self.signatures[doc_id] = signature
        self.document_groups[doc_id] = [doc_id]
        return True, None
    
    def get_duplicate_groups(self) -> Dict[str, List[str]]:
        """Get groups of duplicate documents"""
        return {k: v for k, v in self.document_groups.items() if len(v) > 1}


# =============================================================================
# VERSION SELECTION (Keep Newest Only)
# =============================================================================

# Version patterns with their relative ordering
VERSION_PATTERNS = [
    # Ericsson format: L14A, L15B, L16A, etc.
    (r'L(\d{2})([A-Z])', 'ericsson_l', lambda m: int(m.group(1)) * 100 + ord(m.group(2))),
    
    # Release format: Release 10, Rel-16, R17
    (r'[Rr]el(?:ease)?[-_\s]?(\d{1,2})', 'release', lambda m: int(m.group(1)) * 100),
    
    # RU format: RU20, RU21
    (r'RU(\d{2})', 'ru', lambda m: int(m.group(1)) * 100),
    
    # Version format: v1.2.3, V2.0
    (r'[Vv](\d+)\.(\d+)(?:\.(\d+))?', 'version', 
     lambda m: int(m.group(1)) * 10000 + int(m.group(2)) * 100 + int(m.group(3) or 0)),
    
    # Year format: 2023-06, 202306
    (r'(20\d{2})[-_]?(\d{2})', 'date', lambda m: int(m.group(1)) * 100 + int(m.group(2))),
    
    # Revision: Rev1, Revision 2
    (r'[Rr]ev(?:ision)?[-_\s]?(\d+)', 'revision', lambda m: int(m.group(1))),
]


def extract_version_info(filepath: str, text: str = "") -> Dict:
    """
    Extract version information from filepath and content.
    
    Returns dict with version details and a comparable score.
    """
    search_text = f"{filepath} {text[:2000]}".lower()
    
    result = {
        'raw_versions': [],
        'version_type': None,
        'version_score': 0,
        'version_string': None,
    }
    
    for pattern, version_type, scorer in VERSION_PATTERNS:
        matches = re.finditer(pattern, search_text, re.IGNORECASE)
        for match in matches:
            try:
                score = scorer(match)
                if score > result['version_score']:
                    result['version_score'] = score
                    result['version_type'] = version_type
                    result['version_string'] = match.group(0)
                result['raw_versions'].append(match.group(0))
            except:
                pass
    
    return result


def extract_document_family(filepath: str) -> str:
    """
    Extract document family identifier (without version).
    
    Examples:
        "Ericsson_MIMO_L16A.pdf" → "ericsson_mimo"
        "TS38.331_v15.6.0.pdf" → "ts38.331"
    """
    filename = Path(filepath).stem.lower()
    
    # Remove version patterns
    family = filename
    for pattern, _, _ in VERSION_PATTERNS:
        family = re.sub(pattern, '', family, flags=re.IGNORECASE)
    
    # Clean up
    family = re.sub(r'[-_\s]+', '_', family)
    family = re.sub(r'_+', '_', family)
    family = family.strip('_')
    
    return family


class VersionSelector:
    """
    Select best version per document family.
    
    Keeps only the newest version of each document.
    """
    
    def __init__(self):
        self.families: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_document(self, filepath: str, text_preview: str = "") -> None:
        """Add document for version comparison"""
        family = extract_document_family(filepath)
        version_info = extract_version_info(filepath, text_preview)
        
        self.families[family].append({
            'filepath': filepath,
            'version_info': version_info,
        })
    
    def select_best_versions(self) -> Tuple[List[str], Dict]:
        """
        Select best version per family.
        
        Returns:
            (list of filepaths to keep, statistics)
        """
        keep = []
        stats = {
            'total_families': len(self.families),
            'documents_kept': 0,
            'documents_removed': 0,
            'families_with_versions': 0,
        }
        
        for family, docs in self.families.items():
            if len(docs) == 1:
                # Only one document in family
                keep.append(docs[0]['filepath'])
                stats['documents_kept'] += 1
            else:
                # Multiple versions - keep highest score
                stats['families_with_versions'] += 1
                sorted_docs = sorted(
                    docs, 
                    key=lambda d: d['version_info']['version_score'],
                    reverse=True
                )
                
                # Keep the best version
                keep.append(sorted_docs[0]['filepath'])
                stats['documents_kept'] += 1
                stats['documents_removed'] += len(sorted_docs) - 1
        
        return keep, stats


# =============================================================================
# COMBINED CLEANER
# =============================================================================

class DocumentCleaner:
    """
    Combined document cleaning pipeline.
    
    1. Version selection (keep newest)
    2. Document deduplication (MinHash)
    3. Noise page filtering
    """
    
    def __init__(
        self, 
        enable_version_selection: bool = True,
        enable_deduplication: bool = True,
        enable_noise_filtering: bool = True,
        dedup_threshold: float = 0.80
    ):
        self.enable_version_selection = enable_version_selection
        self.enable_deduplication = enable_deduplication
        self.enable_noise_filtering = enable_noise_filtering
        
        self.version_selector = VersionSelector() if enable_version_selection else None
        self.deduplicator = DocumentDeduplicator(dedup_threshold) if enable_deduplication else None
        
        self.stats = {
            'version_selection': {},
            'deduplication': {},
            'noise_filtering': {},
        }
    
    def should_process_file(self, filepath: str, text_preview: str = "") -> Tuple[bool, str]:
        """
        Check if file should be processed (version selection).
        Call this first for all files, then call select_files().
        """
        if self.enable_version_selection:
            self.version_selector.add_document(filepath, text_preview)
        return True, ""
    
    def select_files(self, all_files: List[str]) -> List[str]:
        """
        After adding all files, select which to process.
        """
        if not self.enable_version_selection:
            return all_files
        
        keep, stats = self.version_selector.select_best_versions()
        self.stats['version_selection'] = stats
        
        return [f for f in all_files if f in keep]
    
    def is_duplicate_document(self, doc_id: str, full_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if document is duplicate of existing.
        """
        if not self.enable_deduplication:
            return False, None
        
        is_unique, duplicate_of = self.deduplicator.add_document(doc_id, full_text)
        return not is_unique, duplicate_of
    
    def clean_pages(self, pages: List[str]) -> List[str]:
        """
        Filter noise pages from document.
        """
        if not self.enable_noise_filtering:
            return pages
        
        filtered, stats = filter_noise_pages(pages)
        
        # Accumulate stats
        for key, value in stats.items():
            if key not in self.stats['noise_filtering']:
                self.stats['noise_filtering'][key] = 0
            if isinstance(value, int):
                self.stats['noise_filtering'][key] += value
        
        return filtered
    
    def get_stats(self) -> Dict:
        """Get cleaning statistics"""
        if self.enable_deduplication:
            dup_groups = self.deduplicator.get_duplicate_groups()
            self.stats['deduplication'] = {
                'duplicate_groups': len(dup_groups),
                'total_duplicates': sum(len(v) - 1 for v in dup_groups.values()),
            }
        
        return self.stats


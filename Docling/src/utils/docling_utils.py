"""
Docling Utilities
==================

Helper functions and utilities for Docling document processing.

Features:
- Content similarity detection
- Advanced duplicate handling
- Document fingerprinting
- Format conversion utilities
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONTENT FINGERPRINTING
# =============================================================================

class ContentFingerprinter:
    """
    Generate and compare content fingerprints for duplicate detection.
    
    Uses multiple techniques:
    - SimHash for near-duplicate detection
    - MinHash for set similarity
    - Content hash for exact duplicates
    """
    
    def __init__(
        self,
        shingle_size: int = 5,
        num_hashes: int = 128,
        similarity_threshold: float = 0.85
    ):
        """
        Initialize the fingerprinter.
        
        Args:
            shingle_size: Size of text shingles for hashing
            num_hashes: Number of hash functions for MinHash
            similarity_threshold: Threshold for considering documents similar
        """
        self.shingle_size = shingle_size
        self.num_hashes = num_hashes
        self.similarity_threshold = similarity_threshold
        
        # Storage for fingerprints
        self._fingerprints: Dict[str, Dict] = {}
    
    def compute_fingerprint(self, text: str) -> Dict:
        """
        Compute comprehensive fingerprint for text.
        
        Returns:
            Dictionary containing:
            - content_hash: SHA-256 hash of normalized content
            - simhash: SimHash for near-duplicate detection
            - minhash: MinHash signature
            - shingles: Set of text shingles
        """
        # Normalize text
        normalized = self._normalize_text(text)
        
        # Compute hashes
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        
        # Get shingles
        shingles = self._get_shingles(normalized)
        
        # Compute MinHash signature
        minhash = self._compute_minhash(shingles)
        
        # Compute SimHash
        simhash = self._compute_simhash(normalized)
        
        return {
            'content_hash': content_hash,
            'simhash': simhash,
            'minhash': minhash,
            'shingle_count': len(shingles),
            'char_count': len(normalized),
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove punctuation except for technical identifiers
        text = re.sub(r'[^\w\s\-\.]', '', text)
        return text.strip()
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Generate text shingles (k-grams)."""
        words = text.split()
        if len(words) < self.shingle_size:
            return {text}
        
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = ' '.join(words[i:i + self.shingle_size])
            shingles.add(shingle)
        
        return shingles
    
    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature."""
        if not shingles:
            return [0] * self.num_hashes
        
        signature = [float('inf')] * self.num_hashes
        
        for shingle in shingles:
            for i in range(self.num_hashes):
                # Use different seeds for each hash function
                h = int(hashlib.md5(
                    f"{i}:{shingle}".encode()
                ).hexdigest()[:16], 16)
                signature[i] = min(signature[i], h)
        
        return [int(h) if h != float('inf') else 0 for h in signature]
    
    def _compute_simhash(self, text: str) -> int:
        """Compute SimHash for text."""
        # Tokenize
        tokens = text.split()
        
        if not tokens:
            return 0
        
        # 64-bit hash
        bits = 64
        v = [0] * bits
        
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest()[:16], 16)
            for i in range(bits):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Generate final hash
        simhash = 0
        for i in range(bits):
            if v[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def register_document(self, doc_id: str, text: str) -> Dict:
        """Register a document and return its fingerprint."""
        fingerprint = self.compute_fingerprint(text)
        self._fingerprints[doc_id] = fingerprint
        return fingerprint
    
    def find_similar(self, text: str) -> List[Tuple[str, float]]:
        """
        Find similar documents in the registry.
        
        Returns:
            List of (doc_id, similarity_score) tuples
        """
        fp = self.compute_fingerprint(text)
        similar = []
        
        for doc_id, stored_fp in self._fingerprints.items():
            # Check exact match first
            if fp['content_hash'] == stored_fp['content_hash']:
                similar.append((doc_id, 1.0))
                continue
            
            # Check MinHash similarity
            mh_sim = self._minhash_similarity(fp['minhash'], stored_fp['minhash'])
            if mh_sim >= self.similarity_threshold:
                similar.append((doc_id, mh_sim))
                continue
            
            # Check SimHash similarity
            sh_sim = self._simhash_similarity(fp['simhash'], stored_fp['simhash'])
            if sh_sim >= self.similarity_threshold:
                similar.append((doc_id, sh_sim))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def _minhash_similarity(self, mh1: List[int], mh2: List[int]) -> float:
        """Compute Jaccard similarity from MinHash signatures."""
        if not mh1 or not mh2:
            return 0.0
        
        matches = sum(1 for a, b in zip(mh1, mh2) if a == b)
        return matches / len(mh1)
    
    def _simhash_similarity(self, sh1: int, sh2: int) -> float:
        """Compute similarity from SimHash (based on Hamming distance)."""
        # XOR to find differing bits
        diff = sh1 ^ sh2
        # Count differing bits
        hamming = bin(diff).count('1')
        # Convert to similarity (64 bits)
        return 1 - (hamming / 64)
    
    def is_duplicate(self, text: str, threshold: float = None) -> Tuple[bool, Optional[str]]:
        """
        Check if text is a duplicate of any registered document.
        
        Returns:
            (is_duplicate, matching_doc_id or None)
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        similar = self.find_similar(text)
        
        for doc_id, score in similar:
            if score >= threshold:
                return True, doc_id
        
        return False, None


# =============================================================================
# VERSION SELECTION
# =============================================================================

class VersionSelector:
    """
    Select the newest version of each document family.
    
    Handles versioning patterns like:
    - document_v1.0.pdf, document_v2.0.pdf
    - TS38.331_rel15.pdf, TS38.331_rel16.pdf
    - manual_2021.pdf, manual_2023.pdf
    """
    
    # Version patterns to detect
    VERSION_PATTERNS = [
        r'[_\-]v(\d+(?:\.\d+)*)(?:[_\-\.]|$)',     # _v1.0, -v2.3
        r'[_\-]r(\d+)(?:[_\-\.]|$)',              # _r15, -r16
        r'[_\-]rel(\d+)(?:[_\-\.]|$)',            # _rel15, _rel16
        r'[_\-](\d{4})(?:[_\-\.]|$)',             # _2023, _2024 (years)
        r'[_\-]rev(\d+)(?:[_\-\.]|$)',            # _rev1, _rev2
        r'_(\d+\.\d+\.\d+)(?:[_\-\.]|$)',         # _1.2.3 (semver)
    ]
    
    def __init__(self):
        self.document_families: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_document(self, filepath: str) -> None:
        """Register a document for version selection."""
        path = Path(filepath)
        filename = path.stem
        
        # Extract version and base name
        version, base_name = self._extract_version(filename)
        
        self.document_families[base_name].append({
            'path': filepath,
            'filename': filename,
            'version': version,
            'version_tuple': self._version_to_tuple(version),
        })
    
    def _extract_version(self, filename: str) -> Tuple[str, str]:
        """Extract version string and base document name."""
        base_name = filename
        version = "0"
        
        for pattern in self.VERSION_PATTERNS:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                version = match.group(1)
                # Remove version from base name
                base_name = re.sub(pattern, '', filename, flags=re.IGNORECASE)
                break
        
        # Normalize base name
        base_name = re.sub(r'[_\-]+$', '', base_name)
        base_name = base_name.lower()
        
        return version, base_name
    
    def _version_to_tuple(self, version: str) -> Tuple:
        """Convert version string to sortable tuple."""
        parts = re.split(r'[.\-_]', version)
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                result.append(part)
        return tuple(result)
    
    def select_newest(self) -> List[str]:
        """
        Select newest version of each document family.
        
        Returns:
            List of file paths to keep
        """
        selected = []
        
        for base_name, versions in self.document_families.items():
            if len(versions) == 1:
                selected.append(versions[0]['path'])
            else:
                # Sort by version (newest first)
                sorted_versions = sorted(
                    versions,
                    key=lambda x: x['version_tuple'],
                    reverse=True
                )
                selected.append(sorted_versions[0]['path'])
                
                # Log what was skipped
                for skipped in sorted_versions[1:]:
                    logger.info(
                        f"Skipping older version: {skipped['filename']} "
                        f"(keeping {sorted_versions[0]['filename']})"
                    )
        
        return selected
    
    def get_statistics(self) -> Dict:
        """Return version selection statistics."""
        total_docs = sum(len(v) for v in self.document_families.values())
        families = len(self.document_families)
        
        return {
            'total_documents': total_docs,
            'document_families': families,
            'documents_to_keep': families,
            'documents_to_skip': total_docs - families,
        }


# =============================================================================
# FORMAT CONVERSION
# =============================================================================

def convert_table_to_text(table_data: Dict) -> str:
    """
    Convert table data to readable text format.
    
    Args:
        table_data: Dictionary with table structure
        
    Returns:
        Text representation of the table
    """
    if 'markdown' in table_data:
        return table_data['markdown']
    
    parts = []
    
    if 'caption' in table_data:
        parts.append(f"Table: {table_data['caption']}")
    
    if 'headers' in table_data and 'rows' in table_data:
        headers = table_data['headers']
        rows = table_data['rows']
        
        # Header row
        parts.append(' | '.join(str(h) for h in headers))
        parts.append('-' * 50)
        
        # Data rows
        for row in rows:
            parts.append(' | '.join(str(cell) for cell in row))
    
    return '\n'.join(parts)


def extract_text_from_image_result(image_data: Dict) -> str:
    """
    Extract searchable text from image data.
    
    Args:
        image_data: Dictionary with image information
        
    Returns:
        Text content from image
    """
    parts = []
    
    if 'caption' in image_data:
        parts.append(f"Figure: {image_data['caption']}")
    
    if 'ocr_text' in image_data:
        parts.append(image_data['ocr_text'])
    
    return '\n'.join(parts)


def merge_chunks_by_section(chunks: List[Dict]) -> List[Dict]:
    """
    Merge small chunks within the same section.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Merged chunks list
    """
    if not chunks:
        return chunks
    
    merged = []
    current = None
    
    for chunk in chunks:
        if current is None:
            current = chunk.copy()
            continue
        
        # Check if should merge
        same_section = chunk.get('section') == current.get('section')
        small_enough = (
            len(current.get('text', '')) + len(chunk.get('text', '')) < 3000
        )
        
        if same_section and small_enough:
            # Merge
            current['text'] += '\n' + chunk.get('text', '')
            current['pages'] = list(set(current.get('pages', []) + chunk.get('pages', [])))
            current['page_end'] = max(current.get('page_end', 0), chunk.get('page_end', 0))
            current['char_count'] = len(current['text'])
            current['word_count'] = len(current['text'].split())
        else:
            merged.append(current)
            current = chunk.copy()
    
    if current:
        merged.append(current)
    
    return merged


def deduplicate_chunks(chunks: List[Dict], threshold: float = 0.9) -> List[Dict]:
    """
    Remove near-duplicate chunks from a list.
    
    Args:
        chunks: List of chunk dictionaries
        threshold: Similarity threshold for deduplication
        
    Returns:
        Deduplicated chunks list
    """
    if not chunks:
        return chunks
    
    fingerprinter = ContentFingerprinter(similarity_threshold=threshold)
    unique_chunks = []
    
    for i, chunk in enumerate(chunks):
        text = chunk.get('text', '')
        is_dup, _ = fingerprinter.is_duplicate(text)
        
        if not is_dup:
            fingerprinter.register_document(f"chunk_{i}", text)
            unique_chunks.append(chunk)
    
    return unique_chunks


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_json(results: List[Dict], output_path: str) -> None:
    """Export processing results to JSON."""
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)


def export_chunks_to_jsonl(chunks: List[Dict], output_path: str) -> None:
    """Export chunks to JSONL format (one JSON per line)."""
    import json
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, default=str) + '\n')


def export_tables_to_csv(tables: List[Dict], output_dir: str) -> None:
    """Export extracted tables to CSV files."""
    import csv
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for table in tables:
        if 'headers' in table and 'rows' in table:
            filename = f"table_{table.get('index', 0)}.csv"
            filepath = output_path / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(table['headers'])
                writer.writerows(table['rows'])

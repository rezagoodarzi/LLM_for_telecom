"""
Utility Functions for RAG System
================================

Common helper functions used across the system.
"""

import re
import hashlib
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict

import sys
sys.path.append('..')
from config import settings


def compute_text_hash(text: str) -> str:
    """Compute MD5 hash for text deduplication"""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def compute_shingle_hash(text: str, k: int = 5) -> Set[int]:
    """Compute k-shingle hashes for similarity detection"""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    words = normalized.split()
    
    if len(words) < k:
        return {hash(normalized)}
    
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = ' '.join(words[i:i+k])
        shingles.add(hash(shingle))
    
    return shingles


def compute_jaccard_similarity(set1: Set[int], set2: Set[int]) -> float:
    """Compute Jaccard similarity between two sets"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def extract_keywords(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract technical keywords from text
    Returns: (uppercase_keywords, lowercase_keywords)
    """
    keywords = set()
    keywords_lower = set()
    
    text_upper = text.upper()
    text_lower = text.lower()
    
    # Apply telecom-specific patterns
    for pattern in settings.TELECOM_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            keywords.add(m.upper())
            keywords_lower.add(m.lower())
    
    # Capitalized terms (3+ chars)
    cap_terms = re.findall(r'\b[A-Z]{3,}\b', text)
    for t in cap_terms:
        keywords.add(t.upper())
        keywords_lower.add(t.lower())
    
    # Quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    for q in quoted:
        if len(q) > 2:
            keywords.add(q.upper())
            keywords_lower.add(q.lower())
    
    # Version numbers
    versions = re.findall(r'\bv?\d+\.\d+(?:\.\d+)*\b', text, re.IGNORECASE)
    for v in versions:
        keywords.add(v.upper())
        keywords_lower.add(v.lower())
    
    return list(keywords), list(keywords_lower)


def extract_phrases(text: str) -> List[str]:
    """Extract important multi-word phrases"""
    phrases = []
    text_lower = text.lower()
    
    for phrase in settings.TECHNICAL_PHRASES:
        if phrase in text_lower:
            phrases.append(phrase)
    
    # Section titles (ending with colon)
    section_titles = re.findall(r'^([A-Za-z][^:]{5,50}):$', text, re.MULTILINE)
    phrases.extend([t.lower().strip() for t in section_titles])
    
    return phrases


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 indexing"""
    text_lower = text.lower()
    
    # Protect technical terms
    protected = []
    for pattern in settings.TELECOM_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        protected.extend([m.lower() for m in matches])
    
    # Clean and split
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', text_lower)
    tokens = cleaned.split()
    tokens.extend(protected)
    
    # Filter short tokens (except important short ones)
    important_short = {'5g', 'nr', 'lte', 'rf', 'ip', 'tx', 'rx', 'dl', 'ul', 'f1', 'f2', 'f3', 'f4', 'f5'}
    tokens = [t for t in tokens if len(t) >= 2 or t in important_short]
    
    # Deduplicate while preserving order
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    
    return unique_tokens


def detect_section(text: str) -> Optional[Dict]:
    """Detect if text is a section header"""
    for pattern, section_type in settings.SECTION_PATTERNS:
        match = re.match(pattern, text.strip())
        if match:
            return {
                'type': section_type,
                'text': text.strip(),
                'match': match.groups()
            }
    return None


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: 1 token ≈ 4 chars for English)"""
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens"""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


class TextDeduplicator:
    """Efficient text deduplication using hashing"""
    
    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.hashes: Dict[int, str] = {}
        self.shingles: Dict[int, Set[int]] = {}
        self.duplicate_groups: Dict[str, List[int]] = defaultdict(list)
    
    def add(self, idx: int, text: str) -> bool:
        """
        Add text and check for duplicates.
        Returns True if text is unique, False if duplicate.
        """
        text_hash = compute_text_hash(text)
        
        # Exact duplicate check
        if text_hash in [h for h in self.hashes.values()]:
            self.duplicate_groups[text_hash].append(idx)
            return False
        
        self.hashes[idx] = text_hash
        self.duplicate_groups[text_hash].append(idx)
        
        # Shingle-based similarity (only for recent items to avoid O(n²))
        shingles = compute_shingle_hash(text)
        self.shingles[idx] = shingles
        
        return True
    
    def get_duplicate_map(self) -> Dict[int, int]:
        """Get mapping of duplicates to primary chunk"""
        duplicate_map = {}
        for group in self.duplicate_groups.values():
            if len(group) > 1:
                primary = group[0]
                for dup in group[1:]:
                    duplicate_map[dup] = primary
        return duplicate_map


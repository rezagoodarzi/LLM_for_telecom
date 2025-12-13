"""
Similarity Handler
===================

Advanced document similarity detection for handling:
- Exact duplicates (hash-based)
- Near-duplicates (fuzzy matching)
- Similar document versions
- Content overlap detection

Uses multiple algorithms for robust detection:
- SHA-256 for exact matches
- MinHash for Jaccard similarity
- SimHash for Hamming distance
- N-gram analysis for partial matches
"""

import re
import hashlib
from typing import List, Dict, Set, Tuple, Optional, NamedTuple
from collections import defaultdict
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SimilarityResult:
    """Result of similarity comparison"""
    doc_id: str
    similarity_score: float
    match_type: str  # 'exact', 'near_duplicate', 'similar', 'partial'
    details: Dict = field(default_factory=dict)


class DocumentSignature(NamedTuple):
    """Compact document signature for comparison"""
    content_hash: str
    simhash: int
    minhash: Tuple[int, ...]
    word_count: int
    char_count: int


# =============================================================================
# SIMILARITY HANDLER
# =============================================================================

class SimilarityHandler:
    """
    Comprehensive similarity detection for documents.
    
    Handles:
    - Exact duplicate detection
    - Near-duplicate detection (90%+ similar)
    - Similar document detection (70-90% similar)
    - Version detection (same base document, different versions)
    """
    
    def __init__(
        self,
        exact_threshold: float = 1.0,
        near_dup_threshold: float = 0.90,
        similar_threshold: float = 0.70,
        shingle_size: int = 5,
        num_hashes: int = 128
    ):
        """
        Initialize the similarity handler.
        
        Args:
            exact_threshold: Threshold for exact duplicates (1.0)
            near_dup_threshold: Threshold for near-duplicates (0.90)
            similar_threshold: Threshold for similar docs (0.70)
            shingle_size: Size of text shingles
            num_hashes: Number of MinHash functions
        """
        self.exact_threshold = exact_threshold
        self.near_dup_threshold = near_dup_threshold
        self.similar_threshold = similar_threshold
        self.shingle_size = shingle_size
        self.num_hashes = num_hashes
        
        # Document registry
        self._documents: Dict[str, DocumentSignature] = {}
        self._content_hashes: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self.stats = {
            'documents_registered': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'similar_documents': 0,
        }
    
    def register_document(
        self, 
        doc_id: str, 
        content: str
    ) -> Tuple[bool, Optional[SimilarityResult]]:
        """
        Register a document and check for duplicates.
        
        Args:
            doc_id: Unique document identifier
            content: Document text content
            
        Returns:
            (is_unique, similarity_result if duplicate else None)
        """
        # Compute signature
        signature = self._compute_signature(content)
        
        # Check for exact duplicate
        if signature.content_hash in self._content_hashes:
            existing_ids = self._content_hashes[signature.content_hash]
            if existing_ids:
                self.stats['exact_duplicates'] += 1
                return False, SimilarityResult(
                    doc_id=list(existing_ids)[0],
                    similarity_score=1.0,
                    match_type='exact',
                    details={'hash': signature.content_hash}
                )
        
        # Check for near-duplicates
        for existing_id, existing_sig in self._documents.items():
            similarity = self._compute_similarity(signature, existing_sig)
            
            if similarity >= self.near_dup_threshold:
                self.stats['near_duplicates'] += 1
                return False, SimilarityResult(
                    doc_id=existing_id,
                    similarity_score=similarity,
                    match_type='near_duplicate',
                    details={'algorithm': 'combined'}
                )
        
        # Register as unique
        self._documents[doc_id] = signature
        self._content_hashes[signature.content_hash].add(doc_id)
        self.stats['documents_registered'] += 1
        
        return True, None
    
    def find_similar(
        self, 
        content: str, 
        top_k: int = 5
    ) -> List[SimilarityResult]:
        """
        Find similar documents to the given content.
        
        Args:
            content: Text content to compare
            top_k: Maximum number of results
            
        Returns:
            List of SimilarityResult sorted by score
        """
        signature = self._compute_signature(content)
        results = []
        
        for doc_id, existing_sig in self._documents.items():
            similarity = self._compute_similarity(signature, existing_sig)
            
            if similarity >= self.similar_threshold:
                match_type = 'exact' if similarity >= self.exact_threshold else \
                            'near_duplicate' if similarity >= self.near_dup_threshold else \
                            'similar'
                
                results.append(SimilarityResult(
                    doc_id=doc_id,
                    similarity_score=similarity,
                    match_type=match_type,
                    details={}
                ))
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def _compute_signature(self, content: str) -> DocumentSignature:
        """Compute document signature."""
        # Normalize
        normalized = self._normalize_text(content)
        
        # Content hash
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:32]
        
        # SimHash
        simhash = self._compute_simhash(normalized)
        
        # MinHash
        shingles = self._get_shingles(normalized)
        minhash = tuple(self._compute_minhash(shingles))
        
        return DocumentSignature(
            content_hash=content_hash,
            simhash=simhash,
            minhash=minhash,
            word_count=len(normalized.split()),
            char_count=len(normalized)
        )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\-\.]', '', text)
        return text.strip()
    
    def _get_shingles(self, text: str) -> Set[str]:
        """Generate text shingles."""
        words = text.split()
        if len(words) < self.shingle_size:
            return {text}
        
        return {
            ' '.join(words[i:i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        }
    
    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        """Compute MinHash signature."""
        if not shingles:
            return [0] * self.num_hashes
        
        signature = [float('inf')] * self.num_hashes
        
        for shingle in shingles:
            for i in range(self.num_hashes):
                h = int(hashlib.md5(
                    f"{i}:{shingle}".encode()
                ).hexdigest()[:16], 16)
                signature[i] = min(signature[i], h)
        
        return [int(h) if h != float('inf') else 0 for h in signature]
    
    def _compute_simhash(self, text: str) -> int:
        """Compute SimHash."""
        tokens = text.split()
        if not tokens:
            return 0
        
        bits = 64
        v = [0] * bits
        
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode()).hexdigest()[:16], 16)
            for i in range(bits):
                if token_hash & (1 << i):
                    v[i] += 1
                else:
                    v[i] -= 1
        
        simhash = 0
        for i in range(bits):
            if v[i] > 0:
                simhash |= (1 << i)
        
        return simhash
    
    def _compute_similarity(
        self, 
        sig1: DocumentSignature, 
        sig2: DocumentSignature
    ) -> float:
        """Compute similarity between two signatures."""
        # Exact match check
        if sig1.content_hash == sig2.content_hash:
            return 1.0
        
        # MinHash Jaccard similarity
        minhash_sim = self._minhash_jaccard(sig1.minhash, sig2.minhash)
        
        # SimHash similarity
        simhash_sim = self._simhash_similarity(sig1.simhash, sig2.simhash)
        
        # Length similarity (penalize very different lengths)
        len_ratio = min(sig1.char_count, sig2.char_count) / max(sig1.char_count, sig2.char_count, 1)
        
        # Combined score (weighted average)
        combined = (
            0.5 * minhash_sim +
            0.3 * simhash_sim +
            0.2 * len_ratio
        )
        
        return combined
    
    def _minhash_jaccard(self, mh1: Tuple[int, ...], mh2: Tuple[int, ...]) -> float:
        """Compute Jaccard similarity from MinHash."""
        if not mh1 or not mh2:
            return 0.0
        matches = sum(1 for a, b in zip(mh1, mh2) if a == b)
        return matches / len(mh1)
    
    def _simhash_similarity(self, sh1: int, sh2: int) -> float:
        """Compute SimHash similarity."""
        diff = sh1 ^ sh2
        hamming = bin(diff).count('1')
        return 1 - (hamming / 64)
    
    def get_stats(self) -> Dict:
        """Return statistics."""
        return dict(self.stats)
    
    def clear(self) -> None:
        """Clear all registered documents."""
        self._documents.clear()
        self._content_hashes.clear()
        for key in self.stats:
            self.stats[key] = 0


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def deduplicate_documents(
    documents: List[Dict],
    content_key: str = 'text',
    id_key: str = 'path',
    threshold: float = 0.90
) -> Tuple[List[Dict], List[Dict]]:
    """
    Deduplicate a list of documents.
    
    Args:
        documents: List of document dictionaries
        content_key: Key for text content
        id_key: Key for document identifier
        threshold: Similarity threshold
        
    Returns:
        (unique_documents, duplicate_documents)
    """
    handler = SimilarityHandler(near_dup_threshold=threshold)
    
    unique = []
    duplicates = []
    
    for doc in documents:
        content = doc.get(content_key, '')
        doc_id = doc.get(id_key, str(len(unique)))
        
        is_unique, result = handler.register_document(doc_id, content)
        
        if is_unique:
            unique.append(doc)
        else:
            doc['duplicate_of'] = result.doc_id if result else None
            doc['similarity_score'] = result.similarity_score if result else 0
            duplicates.append(doc)
    
    logger.info(
        f"Deduplication: {len(unique)} unique, {len(duplicates)} duplicates "
        f"({len(duplicates) / len(documents) * 100:.1f}% removed)"
    )
    
    return unique, duplicates


def find_document_clusters(
    documents: List[Dict],
    content_key: str = 'text',
    threshold: float = 0.70
) -> List[List[Dict]]:
    """
    Cluster documents by similarity.
    
    Args:
        documents: List of document dictionaries
        content_key: Key for text content
        threshold: Similarity threshold for clustering
        
    Returns:
        List of document clusters
    """
    handler = SimilarityHandler(similar_threshold=threshold)
    
    # Register all documents
    for i, doc in enumerate(documents):
        content = doc.get(content_key, '')
        handler.register_document(f"doc_{i}", content)
    
    # Build similarity graph
    clusters = []
    assigned = set()
    
    for i, doc in enumerate(documents):
        if i in assigned:
            continue
        
        content = doc.get(content_key, '')
        similar = handler.find_similar(content, top_k=100)
        
        cluster = [doc]
        assigned.add(i)
        
        for result in similar:
            idx = int(result.doc_id.split('_')[1])
            if idx not in assigned:
                cluster.append(documents[idx])
                assigned.add(idx)
        
        clusters.append(cluster)
    
    return clusters

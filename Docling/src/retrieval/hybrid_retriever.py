"""
Hybrid Retriever with 2-Stage Retrieval
========================================

Implements:
1. Document-level retrieval (find relevant documents first)
2. Chunk-level retrieval (find chunks within documents)
3. Hybrid scoring (semantic + BM25 + keyword)
4. Vendor/metadata filtering
5. Cross-encoder reranking
6. Deduplication

Key improvements:
- 2-stage retrieval prevents cross-document contamination
- Vendor filtering keeps results from single source
- Better keyword weighting for technical terms
"""

import os
import json
import pickle
import numpy as np
import faiss
import re
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set
from sentence_transformers import SentenceTransformer, CrossEncoder

import sys
sys.path.append('..')
from config import settings
from utils.helpers import tokenize_for_bm25, extract_keywords


class HybridRetriever:
    """
    Advanced hybrid retrieval with document filtering.
    
    Features:
    - 2-stage retrieval (document → chunk)
    - Semantic + BM25 + Keyword fusion
    - Vendor/version filtering
    - Cross-encoder reranking
    - Deduplication
    """
    
    def __init__(self, store_path: str = None):
        self.store_path = store_path or settings.OUTPUT_DIR
        self._load_components()
    
    def _load_components(self):
        """Load all retrieval components"""
        print("📥 Loading Hybrid Retriever...")
        
        # Load embedding model
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        print(f"   ✓ Embedding model loaded")
        
        # Load cross-encoder
        self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL)
        print(f"   ✓ Cross-encoder loaded")
        
        # Load chunk FAISS index
        index_path = os.path.join(self.store_path, "faiss.index")
        self.chunk_index = faiss.read_index(index_path)
        print(f"   ✓ Chunk index: {self.chunk_index.ntotal:,} vectors")
        
        # Load document FAISS index
        doc_index_path = os.path.join(self.store_path, "faiss_documents.index")
        if os.path.exists(doc_index_path):
            self.doc_index = faiss.read_index(doc_index_path)
            print(f"   ✓ Document index: {self.doc_index.ntotal:,} vectors")
        else:
            self.doc_index = None
            print(f"   ⚠️ Document index not found")
        
        # Load metadata
        meta_path = os.path.join(self.store_path, "metadata.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.texts = data['texts']
        self.metadatas = data['metadatas']
        self.keyword_index = data.get('keywords_index', {})
        self.keyword_index_lower = data.get('keywords_index_lower', {})
        self.phrase_index = data.get('phrases_index', {})
        self.section_index = data.get('section_index', {})
        self.vendor_index = data.get('vendor_index', {})
        self.adjacency_map = data.get('adjacency_map', {})
        self.duplicate_map = data.get('duplicate_map', {})
        
        # Load document summaries
        doc_path = os.path.join(self.store_path, "documents.json")
        if os.path.exists(doc_path):
            with open(doc_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"   ✓ Documents loaded: {len(self.documents):,}")
        else:
            self.documents = []
        
        # Load BM25
        bm25_path = os.path.join(self.store_path, "bm25.pkl")
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                self.bm25 = pickle.load(f)
            print(f"   ✓ BM25 index loaded")
        else:
            self.bm25 = None
            print(f"   ⚠️ BM25 index not found")
        
        print(f"   ✓ Total chunks: {len(self.texts):,}")
    
    def retrieve(
        self,
        query: str,
        filters: Dict = None,
        top_k: int = None,
        use_document_retrieval: bool = True
    ) -> List[Dict]:
        """
        Main retrieval method with all features.
        
        Args:
            query: User query
            filters: {'vendor': 'ericsson', 'release': 17, etc.}
            top_k: Number of results (default from settings)
            use_document_retrieval: Whether to use 2-stage retrieval
        
        Returns:
            List of result dicts with text, metadata, scores
        """
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        # Extract query features
        features = self._analyze_query(query)
        
        if settings.VERBOSE:
            self._print_query_analysis(features)
        
        # Determine which chunks to search
        candidate_indices = None
        
        # Stage 1: Document-level retrieval (if enabled)
        if use_document_retrieval and self.doc_index and settings.ENABLE_DOCUMENT_RETRIEVAL:
            candidate_indices = self._retrieve_by_documents(query, filters)
            if settings.VERBOSE:
                print(f"   📄 Narrowed to {len(candidate_indices):,} chunks from top documents")
        
        # Apply vendor filter
        if filters and filters.get('vendor'):
            vendor = filters['vendor'].lower()
            if vendor in self.vendor_index:
                vendor_chunks = set(self.vendor_index[vendor])
                if candidate_indices is not None:
                    candidate_indices = candidate_indices.intersection(vendor_chunks)
                else:
                    candidate_indices = vendor_chunks
                if settings.VERBOSE:
                    print(f"   🏢 Filtered to vendor '{vendor}': {len(candidate_indices):,} chunks")
        
        # Stage 2: Hybrid chunk retrieval
        candidates = self._hybrid_retrieve(query, features, candidate_indices)
        
        # Rerank with cross-encoder
        if len(candidates) > 5:
            candidates = self._rerank(query, candidates)
        
        # Deduplicate
        results = self._deduplicate(candidates, top_k)
        
        return results
    
    def _analyze_query(self, query: str) -> Dict:
        """Analyze query for retrieval strategy"""
        features = {
            'raw': query,
            'quoted_terms': re.findall(r'"([^"]+)"', query),
            'keywords': [],
            'technical_ids': [],
            'is_continuation': False,
        }
        
        # Extract keywords
        kw_upper, kw_lower = extract_keywords(query)
        features['keywords'] = kw_upper
        
        # Extract technical IDs (e.g., SN0012, TS38.331)
        features['technical_ids'] = re.findall(
            r'\b[A-Za-z]{2,10}\d{2,10}\b|\b\d{2,10}[A-Za-z]{2,10}\b',
            query
        )
        
        # Check for continuation query
        continuation_words = ['more', 'continue', 'also', 'what else', 'details']
        if any(w in query.lower() for w in continuation_words):
            features['is_continuation'] = True
        
        return features
    
    def _print_query_analysis(self, features: Dict):
        """Print query analysis for debugging"""
        print(f"\n🔎 Query Analysis:")
        if features['quoted_terms']:
            print(f"   📌 Quoted: {features['quoted_terms']}")
        if features['keywords']:
            print(f"   🔑 Keywords: {features['keywords'][:10]}")
        if features['technical_ids']:
            print(f"   🆔 Technical IDs: {features['technical_ids']}")
    
    def _retrieve_by_documents(
        self,
        query: str,
        filters: Dict = None
    ) -> Set[int]:
        """
        Stage 1: Find relevant documents first.
        Returns set of chunk indices belonging to top documents.
        """
        # Embed query
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search document index
        scores, doc_indices = self.doc_index.search(
            query_emb.astype('float32'),
            settings.DOCUMENT_TOP_K
        )
        
        # Collect chunk indices from top documents
        chunk_indices = set()
        for doc_idx in doc_indices[0]:
            if 0 <= doc_idx < len(self.documents):
                doc = self.documents[doc_idx]
                
                # Apply filters at document level
                if filters:
                    meta = doc.get('metadata', {})
                    if filters.get('vendor') and meta.get('vendor') != filters['vendor']:
                        continue
                    if filters.get('release') and meta.get('release') != filters['release']:
                        continue
                
                chunk_indices.update(doc.get('chunk_indices', []))
        
        return chunk_indices
    
    def _hybrid_retrieve(
        self,
        query: str,
        features: Dict,
        candidate_indices: Set[int] = None
    ) -> List[Dict]:
        """
        Stage 2: Hybrid retrieval combining semantic + BM25 + keyword.
        """
        candidates = {}
        
        # 1. Semantic search
        semantic_results = self._semantic_search(query, settings.TOP_K * 2)
        semantic_max = max([s for _, s in semantic_results]) if semantic_results else 1.0
        
        for idx, score in semantic_results:
            if candidate_indices is not None and idx not in candidate_indices:
                continue
            
            normalized = score / semantic_max if semantic_max > 0 else 0
            candidates[idx] = {
                'idx': idx,
                'semantic_score': normalized,
                'bm25_score': 0.0,
                'keyword_score': 0.0,
                'type': 'semantic'
            }
        
        # 2. BM25 search
        if self.bm25 is not None:
            bm25_results = self._bm25_search(query, settings.BM25_TOP_K)
            bm25_max = max([s for _, s in bm25_results]) if bm25_results else 1.0
            
            for idx, score in bm25_results:
                if candidate_indices is not None and idx not in candidate_indices:
                    continue
                
                normalized = score / bm25_max if bm25_max > 0 else 0
                
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': normalized,
                        'keyword_score': 0.0,
                        'type': 'bm25'
                    }
                else:
                    candidates[idx]['bm25_score'] = normalized
                    candidates[idx]['type'] += '+bm25'
        
        # 3. Keyword search
        all_terms = features['keywords'] + features['quoted_terms'] + features['technical_ids']
        if all_terms:
            kw_results = self._keyword_search(all_terms)
            
            for idx, score in kw_results.items():
                if candidate_indices is not None and idx not in candidate_indices:
                    continue
                
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': 0.0,
                        'keyword_score': score,
                        'type': 'keyword'
                    }
                else:
                    candidates[idx]['keyword_score'] = score
                    candidates[idx]['type'] += '+kw'
        
        # 4. Exact text search for quoted terms
        if features['quoted_terms']:
            exact_results = self._exact_search(features['quoted_terms'], candidate_indices)
            
            for idx, score in exact_results.items():
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': 0.0,
                        'keyword_score': score,
                        'type': 'exact'
                    }
                else:
                    candidates[idx]['keyword_score'] = max(candidates[idx]['keyword_score'], score)
                    candidates[idx]['type'] += '+exact'
        
        # Calculate hybrid scores
        for cand in candidates.values():
            hybrid = (
                settings.HYBRID_WEIGHTS['semantic'] * cand['semantic_score'] +
                settings.HYBRID_WEIGHTS['bm25'] * cand['bm25_score'] +
                settings.HYBRID_WEIGHTS['keyword'] * cand['keyword_score']
            )
            cand['hybrid_score'] = hybrid
        
        # Sort by hybrid score
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        return sorted_candidates[:settings.TOP_K * 2]
    
    def _semantic_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """Semantic search using FAISS"""
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        scores, indices = self.chunk_index.search(query_emb.astype('float32'), k)
        return list(zip(indices[0], scores[0]))
    
    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        """BM25 search for term matching"""
        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []
        
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        
        return results
    
    def _keyword_search(self, keywords: List[str], limit: int = 30) -> Dict[int, float]:
        """Keyword search using indices"""
        results = defaultdict(float)
        
        for kw in keywords:
            # Uppercase index
            kw_upper = kw.upper()
            if kw_upper in self.keyword_index:
                for idx in self.keyword_index[kw_upper][:limit]:
                    results[idx] += 1.0
            
            # Lowercase index
            kw_lower = kw.lower()
            if kw_lower in self.keyword_index_lower:
                for idx in self.keyword_index_lower[kw_lower][:limit]:
                    results[idx] += 0.5  # Lower weight for case-insensitive
        
        # Normalize
        if results:
            max_score = max(results.values())
            for idx in results:
                results[idx] /= max_score
        
        return dict(results)
    
    def _exact_search(
        self,
        terms: List[str],
        candidate_indices: Set[int] = None
    ) -> Dict[int, float]:
        """Exact text search for quoted terms"""
        results = defaultdict(float)
        
        search_range = candidate_indices if candidate_indices else range(len(self.texts))
        
        for term in terms:
            term_lower = term.lower()
            
            for idx in search_range:
                text_lower = self.texts[idx].lower()
                count = text_lower.count(term_lower)
                if count > 0:
                    results[idx] += count * settings.EXACT_MATCH_BOOST
                    
                    # Bonus for word boundary match
                    if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                        results[idx] += settings.EXACT_MATCH_BOOST
        
        # Normalize
        if results:
            max_score = max(results.values())
            for idx in results:
                results[idx] /= max_score
        
        return dict(results)
    
    def _rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank using cross-encoder"""
        if settings.VERBOSE:
            print(f"   🎯 Reranking {len(candidates)} candidates...")
        
        # Prepare pairs
        pairs = []
        for cand in candidates:
            text = self.texts[cand['idx']][:1500]  # Truncate for cross-encoder
            pairs.append([query, text])
        
        # Score
        scores = self.cross_encoder.predict(pairs)
        
        # Update candidates
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)
            # Combine with hybrid score
            candidates[i]['final_score'] = (
                candidates[i]['hybrid_score'] * 0.3 +
                max(0, score) * 0.7
            )
        
        # Sort by final score
        candidates.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return candidates
    
    def _deduplicate(self, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Remove duplicates and prepare final results"""
        results = []
        seen_hashes = set()
        seen_primary = set()
        
        for cand in candidates:
            if len(results) >= top_k:
                break
            
            idx = cand['idx']
            text = self.texts[idx]
            
            # Hash-based dedup
            text_hash = hash(re.sub(r'\s+', ' ', text[:300].lower()))
            if text_hash in seen_hashes:
                continue
            
            # Check duplicate map
            if settings.ENABLE_DEDUPLICATION:
                idx_str = str(idx)
                if idx_str in self.duplicate_map:
                    primary = self.duplicate_map[idx_str]
                    if primary in seen_primary:
                        continue
                    seen_primary.add(primary)
            
            seen_hashes.add(text_hash)
            seen_primary.add(idx)
            
            # Create result
            result = {
                'idx': idx,
                'text': text,
                'metadata': self.metadatas[idx],
                'score': cand.get('final_score', cand.get('hybrid_score', 0)),
                'retrieval_type': cand['type'],
                'scores': {
                    'semantic': cand.get('semantic_score', 0),
                    'bm25': cand.get('bm25_score', 0),
                    'keyword': cand.get('keyword_score', 0),
                    'hybrid': cand.get('hybrid_score', 0),
                    'rerank': cand.get('rerank_score', 0),
                }
            }
            results.append(result)
        
        if settings.VERBOSE:
            print(f"   ✅ Retrieved {len(results)} unique chunks")
        
        return results
    
    def get_adjacent_chunks(self, chunk_ids: List[int], n: int = 2) -> List[int]:
        """Get adjacent chunks for context expansion"""
        adjacent = set()
        for chunk_id in chunk_ids:
            chunk_id_str = str(chunk_id)
            if chunk_id_str in self.adjacency_map:
                adjacent.update(self.adjacency_map[chunk_id_str][:n])
        return list(adjacent)
    
    def get_section_chunks(self, chunk_ids: List[int], max_per_section: int = 5) -> List[int]:
        """
        Get all chunks from the same sections (section agglomeration).
        This helps retrieve complete context for a topic.
        """
        section_chunks = set()
        
        for chunk_id in chunk_ids:
            if chunk_id < len(self.metadatas):
                section = self.metadatas[chunk_id].get('section')
                if section and section in self.section_index:
                    # Get chunks from same section (limited)
                    same_section = self.section_index[section][:max_per_section]
                    section_chunks.update(same_section)
        
        return list(section_chunks)
    
    def apply_confidence_penalties(
        self, 
        candidates: List[Dict],
        query_vendor: Optional[str] = None
    ) -> List[Dict]:
        """
        Apply confidence penalties for:
        - Cross-vendor results (when vendor is specified)
        - Version mismatches
        - Duplicate content
        
        This helps prevent mixing information from different sources.
        """
        if not candidates:
            return candidates
        
        # Count vendors in results
        vendor_counts = defaultdict(int)
        for cand in candidates:
            idx = cand['idx']
            vendor = self.metadatas[idx].get('vendor')
            if vendor:
                vendor_counts[vendor] += 1
        
        # Determine primary vendor (most common)
        primary_vendor = max(vendor_counts.items(), key=lambda x: x[1])[0] if vendor_counts else None
        
        for cand in candidates:
            idx = cand['idx']
            meta = self.metadatas[idx]
            
            penalty = 1.0
            
            # Vendor mismatch penalty
            if query_vendor:
                if meta.get('vendor') and meta.get('vendor') != query_vendor:
                    penalty *= 0.5  # 50% penalty for wrong vendor
            elif primary_vendor:
                if meta.get('vendor') and meta.get('vendor') != primary_vendor:
                    penalty *= 0.8  # 20% penalty for minority vendor
            
            # Apply penalty
            if 'final_score' in cand:
                cand['final_score'] *= penalty
            if 'hybrid_score' in cand:
                cand['hybrid_score'] *= penalty
            
            cand['confidence_penalty'] = penalty
        
        return candidates
    
    def retrieve_with_context(
        self,
        query: str,
        filters: Dict = None,
        top_k: int = None,
        include_adjacent: bool = True,
        include_section: bool = True
    ) -> List[Dict]:
        """
        Enhanced retrieval with context expansion.
        
        Includes:
        - Normal hybrid retrieval
        - Adjacent chunk expansion
        - Section agglomeration
        - Confidence penalties
        """
        if top_k is None:
            top_k = settings.RERANK_TOP_K
        
        # Get base results
        results = self.retrieve(query, filters, top_k)
        
        if not results:
            return results
        
        # Get chunk IDs
        base_ids = [r['idx'] for r in results]
        
        # Expand with adjacent chunks
        if include_adjacent and settings.ENABLE_ADJACENT_RETRIEVAL:
            adjacent_ids = self.get_adjacent_chunks(base_ids, n=settings.ADJACENT_CHUNKS)
            for adj_id in adjacent_ids:
                if adj_id not in base_ids and adj_id < len(self.texts):
                    results.append({
                        'idx': adj_id,
                        'text': self.texts[adj_id],
                        'metadata': self.metadatas[adj_id],
                        'score': 0.5,  # Lower score for adjacent
                        'retrieval_type': 'adjacent',
                        'scores': {}
                    })
        
        # Expand with section chunks
        if include_section and settings.ENABLE_SECTION_EXPANSION:
            section_ids = self.get_section_chunks(base_ids[:3], max_per_section=3)
            for sec_id in section_ids:
                if sec_id not in base_ids and sec_id < len(self.texts):
                    # Check if not already in results
                    if not any(r['idx'] == sec_id for r in results):
                        results.append({
                            'idx': sec_id,
                            'text': self.texts[sec_id],
                            'metadata': self.metadatas[sec_id],
                            'score': 0.4,  # Lower score for section
                            'retrieval_type': 'section',
                            'scores': {}
                        })
        
        # Apply confidence penalties
        query_vendor = filters.get('vendor') if filters else None
        results = self.apply_confidence_penalties(results, query_vendor)
        
        # Re-sort by final score
        results.sort(key=lambda x: x.get('final_score', x.get('score', 0)), reverse=True)
        
        return results[:top_k + 4]  # Return a few extra for context


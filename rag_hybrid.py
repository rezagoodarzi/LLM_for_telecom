#!/usr/bin/env python3
"""
HYBRID RAG Query System with Text-Generation-WebUI API
Combines: Semantic + BM25 + Keyword search with optimized LLM API
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import requests
import re
from collections import defaultdict
from typing import List, Dict, Tuple, Set
import warnings
import pickle

warnings.filterwarnings("ignore")

# ==================== CONFIG ====================
# API settings (text-generation-webui)
VLLM_URL = "http://localhost:5000/v1/chat/completions"
VLLM_MODEL_NAME = "qwen3-4b-bnb4"  # Model name as loaded in webui

# RAG settings
MODEL_NAME = "/home/fatemebookanian/models/BGE-m3"
CROSS_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
#CROSS_MODEL = "cross-encoder/qnli-electra-base"

RAG_STORE = "./rag_store_3gpp_bem"

# Retrieval parameters
TOP_K = 12                    # Initial retrieval
RERANK_TOP_K = 8              # After reranking

# Smart retrieval features
ENABLE_ADJACENT_RETRIEVAL = True
ENABLE_SECTION_EXPANSION = True
ADJACENT_CHUNKS = 2

# ============ HYBRID SEARCH CONFIG ============
HYBRID_WEIGHTS = {
    'semantic': 0.50,
    'bm25': 0.35,
    'keyword': 0.15,
}

# BM25 parameters
ENABLE_BM25_SEARCH = True
BM25_TOP_K = 30

# Keyword search parameters
ENABLE_CASE_INSENSITIVE = True
EXACT_MATCH_BOOST = 3.0

# Deduplication parameters
ENABLE_DEDUPLICATION = True
SIMILARITY_PENALTY = 0.5
# ==============================================

# LLM parameters (for API)
MODEL_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 2048,
    "top_k": 20,
    "repetition_penalty": 1.1,
}

DEFAULT_CONFIG = MODEL_CONFIG.copy()

# Global state
CONVERSATION_HISTORY = []
LAST_RETRIEVED = []
LAST_QUERY_INFO = {}
# ==============================================


def tokenize_query_for_bm25(query: str) -> List[str]:
    """Tokenize query for BM25 search"""
    query_lower = query.lower()
    
    protected = []
    tech_patterns = [
        r'\b[a-z]{2,10}\d{2,8}\b',
        r'\b\d{2,8}[a-z]{2,10}\b',
        r'\b\d+\.\d+(?:\.\d+)*\b',
        r'\b[a-z]+-[a-z0-9]+\b',
        r'\b[a-z]+_[a-z0-9]+\b',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, query_lower)
        protected.extend(matches)
    
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', query_lower)
    tokens = cleaned.split()
    tokens.extend(protected)
    tokens = [t for t in tokens if len(t) >= 2 or t.upper() in ['5G', 'NR', 'LTE', 'RF', 'IP', 'TX', 'RX']]
    
    return tokens


class HybridRAGRetriever:
    """Hybrid RAG retrieval: Semantic + BM25 + Keywords"""
    
    def __init__(self):
        self.load_components()
        
    def load_components(self):
        """Load all RAG components"""
        print("📥 Loading HYBRID RAG components...")
        
        # Load embedder
        self.embedder = SentenceTransformer(MODEL_NAME)
        
        # Load cross-encoder
        self.cross_encoder = CrossEncoder(CROSS_MODEL)
        
        # Load FAISS index
        index_path = os.path.join(RAG_STORE, "faiss.index")
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        meta_path = os.path.join(RAG_STORE, "metadata.json")
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.texts = self.data['texts']
        self.metadatas = self.data['metadatas']
        self.keyword_index = self.data.get('keywords_index', {})
        self.keyword_index_lower = self.data.get('keywords_index_lower', {})
        self.phrase_index = self.data.get('phrases_index', {})
        self.section_index = self.data.get('section_index', {})
        self.adjacency_map = self.data.get('adjacency_map', {})
        self.doc_structures = self.data.get('document_structures', {})
        self.duplicate_map = self.data.get('duplicate_map', {})
        self.similarity_map = self.data.get('similarity_map', {})
        self.config = self.data.get('config', {})
        
        # Load BM25 index if available
        self.bm25_index = None
        self.bm25_corpus = None
        bm25_path = os.path.join(RAG_STORE, "bm25.pkl")
        bm25_corpus_path = os.path.join(RAG_STORE, "bm25_corpus.pkl")
        
        if os.path.exists(bm25_path) and ENABLE_BM25_SEARCH:
            try:
                with open(bm25_path, 'rb') as f:
                    self.bm25_index = pickle.load(f)
                with open(bm25_corpus_path, 'rb') as f:
                    self.bm25_corpus = pickle.load(f)
                print(f"✅ Loaded BM25 index with {len(self.bm25_corpus)} documents")
            except Exception as e:
                print(f"⚠️  Could not load BM25 index: {e}")
                print("   Run word_embbeding_smart.py to create BM25 index")
        else:
            print("⚠️  BM25 index not found - using semantic + keyword search only")
            print("   Run word_embbeding_smart.py to enable full hybrid search")
        
        self._build_text_hash_index()
        
        print(f"✅ Loaded {len(self.texts)} chunks")
        print(f"   ✓ Semantic search (FAISS)")
        if self.bm25_index:
            print(f"   ✓ BM25 exact term matching")
        print(f"   ✓ Keyword search ({len(self.keyword_index)} uppercase, {len(self.keyword_index_lower)} lowercase)")
    
    def _build_text_hash_index(self):
        """Build hash index for deduplication"""
        self.text_hashes = {}
        for idx, text in enumerate(self.texts):
            normalized = re.sub(r'\s+', ' ', text[:200].lower().strip())
            hash_key = hash(normalized)
            if hash_key not in self.text_hashes:
                self.text_hashes[hash_key] = []
            self.text_hashes[hash_key].append(idx)
    
    def extract_query_features(self, query: str) -> Dict:
        """Extract features from query"""
        features = {
            'raw': query,
            'quoted_terms': [],
            'keywords': [],
            'is_continuation': False,
            'section_request': None,
        }
        
        # Extract quoted terms
        quoted_matches = re.findall(r'"([^"]+)"', query)
        features['quoted_terms'] = quoted_matches
        
        # Check for section requests
        section_patterns = [
            r'all (?:text|content|information) (?:in|about|on|from)\s+["\']?([^"\']+)["\']?',
            r'(?:full|complete|entire)\s+(?:section|chapter|part)\s+(?:on|about|of)?\s*["\']?([^"\']+)["\']?',
            r'everything (?:about|on|regarding)\s+["\']?([^"\']+)["\']?'
        ]
        
        for pattern in section_patterns:
            match = re.search(pattern, query.lower())
            if match:
                features['section_request'] = match.group(1)
                break
        
        # Check if continuation query
        continuation_words = [
            'more', 'additional', 'further', 'elaborate', 'detail', 'details',
            'also', 'besides', 'furthermore', 'moreover', 'what else',
            'continue', 'expand', 'example', 'examples',
            'previous', 'earlier', 'that', 'this', 'those', 'these',
        ]
        
        query_lower = query.lower()
        query_words = query_lower.split()
        
        if len(query.split()) < 4 and any(pronoun in query_lower for pronoun in ['it', 'this', 'that', 'they', 'them']):
            features['is_continuation'] = True
        elif any(word in query_words[:5] for word in continuation_words):
            features['is_continuation'] = True
        
        # Extract technical keywords
        keywords = re.findall(r'\b[A-Z]{2,}\b', query.upper())
        features['keywords'] = keywords
        
        return features
    
    def semantic_search(self, query: str, k: int = TOP_K) -> List[Tuple[int, float]]:
        """Semantic search using embeddings"""
        query_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb.astype('float32'), k)
        return list(zip(indices[0], scores[0]))
    
    def bm25_search(self, query: str, k: int = BM25_TOP_K) -> List[Tuple[int, float]]:
        """BM25 search for exact term matching"""
        if self.bm25_index is None:
            return []
        
        query_tokens = tokenize_query_for_bm25(query)
        if not query_tokens:
            return []
        
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
        
        return results
    
    def keyword_search(self, keywords: List[str], limit: int = 20) -> Dict[int, float]:
        """Case-insensitive keyword search"""
        results = defaultdict(float)
        
        for kw in keywords:
            kw_upper = kw.upper()
            if kw_upper in self.keyword_index:
                for idx in self.keyword_index[kw_upper][:limit]:
                    results[idx] += 1.0
            
            if ENABLE_CASE_INSENSITIVE:
                kw_lower = kw.lower()
                if kw_lower in self.keyword_index_lower:
                    for idx in self.keyword_index_lower[kw_lower][:limit]:
                        results[idx] += 1.0
        
        max_score = max(results.values()) if results else 1.0
        for idx in results:
            results[idx] /= max_score
        
        return dict(results)
    
    def exact_text_search(self, terms: List[str], limit: int = 30) -> Dict[int, float]:
        """Search for exact terms in text"""
        results = defaultdict(float)
        
        for term in terms:
            term_lower = term.lower()
            
            for idx, text in enumerate(self.texts):
                text_lower = text.lower()
                count = text_lower.count(term_lower)
                if count > 0:
                    results[idx] += count * EXACT_MATCH_BOOST
                    if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                        results[idx] += EXACT_MATCH_BOOST
        
        if results:
            max_score = max(results.values())
            for idx in results:
                results[idx] /= max_score
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:limit]
        return dict(sorted_results)
    
    def get_adjacent_chunks(self, chunk_ids: List[int]) -> List[int]:
        """Get adjacent chunks"""
        adjacent = set()
        for chunk_id in chunk_ids:
            chunk_id_str = str(chunk_id)
            if chunk_id_str in self.adjacency_map:
                adjacent.update(self.adjacency_map[chunk_id_str])
        return list(adjacent)
    
    def expand_to_section(self, chunk_ids: List[int]) -> List[int]:
        """Expand to include full sections"""
        expanded = set(chunk_ids)
        for chunk_id in chunk_ids:
            section = self.metadatas[chunk_id].get('section')
            if section and section in self.section_index:
                expanded.update(self.section_index[section])
        return list(expanded)
    
    def smart_retrieve(self, query: str) -> List[Dict]:
        """HYBRID multi-strategy retrieval"""
        global LAST_QUERY_INFO
        
        features = self.extract_query_features(query)
        LAST_QUERY_INFO = features
        
        print(f"\n🔎 Query Analysis:")
        if features['quoted_terms']:
            print(f"  📌 Quoted: {features['quoted_terms']}")
        if features['keywords']:
            print(f"  🔑 Keywords: {features['keywords']}")
        if features['section_request']:
            print(f"  📂 Section request: {features['section_request']}")
        if features['is_continuation']:
            print(f"  🔗 Continuation query detected")
        
        candidates = {}
        
        # 0. Continuation context
        global LAST_RETRIEVED, CONVERSATION_HISTORY
        if features['is_continuation'] and LAST_RETRIEVED:
            print(f"  📚 Including context from previous retrieval...")
            for doc in LAST_RETRIEVED[:5]:
                idx = doc['idx']
                candidates[idx] = {
                    'idx': idx,
                    'semantic_score': 0.0,
                    'bm25_score': 0.0,
                    'keyword_score': 0.7,
                    'type': 'continuation',
                    'boost': 1.5
                }
        
        # 1. SEMANTIC SEARCH
        search_query = query
        if features['is_continuation'] and CONVERSATION_HISTORY:
            last_question = CONVERSATION_HISTORY[-1].get('question', '')
            search_query = f"{last_question} {query}"
            print(f"  🔄 Enhanced query with previous context")
        
        semantic_results = self.semantic_search(search_query, k=TOP_K)
        semantic_max = max([s for _, s in semantic_results]) if semantic_results else 1.0
        
        for idx, score in semantic_results:
            normalized_score = score / semantic_max if semantic_max > 0 else 0
            if idx not in candidates:
                candidates[idx] = {
                    'idx': idx,
                    'semantic_score': normalized_score,
                    'bm25_score': 0.0,
                    'keyword_score': 0.0,
                    'type': 'semantic',
                    'boost': 1.0
                }
            else:
                candidates[idx]['semantic_score'] = normalized_score
        
        # 2. BM25 SEARCH
        if self.bm25_index is not None:
            print(f"  📊 Running BM25 search...")
            bm25_results = self.bm25_search(query, k=BM25_TOP_K)
            bm25_max = max([s for _, s in bm25_results]) if bm25_results else 1.0
            
            for idx, score in bm25_results:
                normalized_score = score / bm25_max if bm25_max > 0 else 0
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': normalized_score,
                        'keyword_score': 0.0,
                        'type': 'bm25',
                        'boost': 1.0
                    }
                else:
                    candidates[idx]['bm25_score'] = normalized_score
                    if candidates[idx]['type'] == 'semantic':
                        candidates[idx]['type'] = 'semantic+bm25'
        
        # 3. KEYWORD SEARCH
        all_terms = features['keywords'] + features['quoted_terms']
        query_ids = re.findall(r'\b[a-zA-Z]{2,10}\d{2,8}\b', query, re.IGNORECASE)
        query_ids += re.findall(r'\b\d{2,8}[a-zA-Z]{2,10}\b', query, re.IGNORECASE)
        all_terms.extend(query_ids)
        
        if all_terms:
            print(f"  🔑 Searching keywords: {all_terms[:5]}{'...' if len(all_terms) > 5 else ''}")
            keyword_results = self.keyword_search(all_terms, limit=20)
            
            for idx, score in keyword_results.items():
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': 0.0,
                        'keyword_score': score,
                        'type': 'keyword',
                        'boost': 1.2
                    }
                else:
                    candidates[idx]['keyword_score'] = score
                    candidates[idx]['boost'] *= 1.5
        
        # 4. EXACT TEXT SEARCH for quoted terms
        if features['quoted_terms']:
            print(f"  🎯 Exact text search for: {features['quoted_terms']}")
            exact_results = self.exact_text_search(features['quoted_terms'], limit=15)
            
            for idx, score in exact_results.items():
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': 0.0,
                        'keyword_score': score,
                        'type': 'exact_match',
                        'boost': 2.0
                    }
                else:
                    candidates[idx]['keyword_score'] = max(candidates[idx]['keyword_score'], score)
                    candidates[idx]['boost'] *= 2.0
                    candidates[idx]['type'] += '+exact'
        
        # 5. Section expansion
        if features['section_request'] and ENABLE_SECTION_EXPANSION:
            section_results = self.semantic_search(features['section_request'], k=5)
            if section_results:
                best_section_idx = section_results[0][0]
                section_name = self.metadatas[best_section_idx].get('section')
                if section_name:
                    print(f"  📚 Expanding section: {section_name}")
                    section_chunks = self.expand_to_section([best_section_idx])
                    for idx in section_chunks:
                        if idx not in candidates:
                            candidates[idx] = {
                                'idx': idx,
                                'semantic_score': 0.0,
                                'bm25_score': 0.0,
                                'keyword_score': 0.3,
                                'type': 'section',
                                'boost': 1.0
                            }
        
        # 6. Adjacent chunks
        if ENABLE_ADJACENT_RETRIEVAL and candidates:
            top_ids = sorted(
                candidates.keys(),
                key=lambda x: candidates[x]['semantic_score'] + candidates[x]['bm25_score'],
                reverse=True
            )[:RERANK_TOP_K]
            
            adjacent_ids = self.get_adjacent_chunks(top_ids)
            for idx in adjacent_ids:
                if idx not in candidates:
                    candidates[idx] = {
                        'idx': idx,
                        'semantic_score': 0.0,
                        'bm25_score': 0.0,
                        'keyword_score': 0.1,
                        'type': 'adjacent',
                        'boost': 0.8
                    }
        
        # Calculate HYBRID scores
        print(f"  ⚖️  Computing hybrid scores (S={HYBRID_WEIGHTS['semantic']:.0%}, "
              f"B={HYBRID_WEIGHTS['bm25']:.0%}, K={HYBRID_WEIGHTS['keyword']:.0%})...")
        
        for cand in candidates.values():
            hybrid_score = (
                HYBRID_WEIGHTS['semantic'] * cand['semantic_score'] +
                HYBRID_WEIGHTS['bm25'] * cand['bm25_score'] +
                HYBRID_WEIGHTS['keyword'] * cand['keyword_score']
            )
            cand['hybrid_score'] = hybrid_score
            cand['final_score'] = hybrid_score * cand['boost']
        
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        # Re-rank with cross-encoder
        if len(sorted_candidates) > 5:
            print(f"  🎯 Re-ranking {min(len(sorted_candidates), TOP_K * 2)} candidates...")
            
            pairs = []
            for cand in sorted_candidates[:TOP_K * 2]:
                pairs.append([query, self.texts[cand['idx']]])
            
            if pairs:
                rerank_scores = self.cross_encoder.predict(pairs)
                
                for i, score in enumerate(rerank_scores):
                    sorted_candidates[i]['rerank_score'] = float(score)
                    sorted_candidates[i]['final_score'] = (
                        sorted_candidates[i]['final_score'] * 0.35 +
                        max(0, score) * 0.65
                    )
                
                sorted_candidates = sorted(
                    sorted_candidates,
                    key=lambda x: x.get('final_score', 0),
                    reverse=True
                )
        
        # Deduplicate and prepare results
        results = []
        seen_hashes = set()
        seen_primary = set()
        
        for cand in sorted_candidates:
            if len(results) >= RERANK_TOP_K:
                break
            
            idx = cand['idx']
            text = self.texts[idx]
            
            text_hash = hash(re.sub(r'\s+', ' ', text[:300].lower().strip()))
            if text_hash in seen_hashes:
                continue
            
            if ENABLE_DEDUPLICATION:
                idx_str = str(idx)
                if idx_str in self.duplicate_map:
                    primary = self.duplicate_map[idx_str]
                    if primary in seen_primary:
                        continue
                    seen_primary.add(primary)
            
            seen_hashes.add(text_hash)
            seen_primary.add(idx)
            
            result = {
                'idx': idx,
                'text': text,
                'metadata': self.metadatas[idx],
                'score': cand.get('final_score', 0),
                'retrieval_type': cand['type'],
                'scores': {
                    'semantic': cand.get('semantic_score', 0),
                    'bm25': cand.get('bm25_score', 0),
                    'keyword': cand.get('keyword_score', 0),
                    'hybrid': cand.get('hybrid_score', 0),
                    'rerank': cand.get('rerank_score', 0),
                    'boost': cand.get('boost', 1.0)
                }
            }
            results.append(result)
        
        print(f"  ✅ Retrieved {len(results)} unique chunks")
        return results


# Initialize retriever
print("=" * 70)
print("🚀 HYBRID RAG System with WebUI API")
print("=" * 70)

retriever = HybridRAGRetriever()


def build_context(retrieved: List[Dict]) -> str:
    """Build context string from retrieved documents"""
    pieces = []
    for i, r in enumerate(retrieved[:5], 1):
        meta = r['metadata']
        source = meta.get('source', 'unknown')
        page = meta.get('page', meta.get('pages', 'N/A'))
        if isinstance(page, list):
            page = f"{page[0]}-{page[-1]}" if page else 'N/A'
        section = meta.get('section', '')
        
        header = f"[Document {i}] Source: {source} | Page: {page}"
        if section:
            header += f" | Section: {section}"
        header += f" | Score: {r['score']:.3f}\n"
        
        pieces.append(header + r["text"][:2000])
    
    return "\n\n---\n\n".join(pieces)


def ask_llm_with_context(question: str, retrieved: List[Dict]) -> str:
    """Generate answer using text-generation-webui API"""
    system = """You are a helpful AI assistant with access to technical documentation.
Answer questions based on the provided context. Be concise and cite sources (filename:page).
If the answer is not in context, say you don't know."""
    
    context = build_context(retrieved)
    
    # Include conversation history for continuation queries
    history_context = ""
    if LAST_QUERY_INFO.get('is_continuation') and CONVERSATION_HISTORY:
        recent = CONVERSATION_HISTORY[-2:]
        if recent:
            history_parts = []
            for turn in recent:
                history_parts.append(f"Previous Q: {turn['question']}")
                history_parts.append(f"Previous A: {turn['answer'][:300]}...")
            history_context = "\n\nPrevious conversation:\n" + "\n".join(history_parts) + "\n"
    
    user_message = f"""{history_context}Context:
{context}

Question: {question}
Answer concisely and cite sources."""

    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message}
        ],
        **MODEL_CONFIG
    }
    
    try:
        r = requests.post(VLLM_URL, json=payload, timeout=300)
        r.raise_for_status()
        j = r.json()
        
        text = ""
        if "choices" in j and len(j["choices"]) > 0:
            c = j["choices"][0]
            if "message" in c and "content" in c["message"]:
                text = c["message"]["content"]
            elif "text" in c:
                text = c["text"]
        
        return text
    except requests.exceptions.ConnectionError:
        return "❌ Error: Cannot connect to text-generation-webui. Make sure it's running on port 5000."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def show_retrieved_details():
    """Show detailed information about retrieved chunks"""
    if not LAST_RETRIEVED:
        print("❌ No documents retrieved yet!")
        return
    
    print("\n" + "=" * 70)
    print("📚 RETRIEVED DOCUMENTS - DETAILED VIEW")
    print("=" * 70)
    
    for i, doc in enumerate(LAST_RETRIEVED, 1):
        meta = doc['metadata']
        scores = doc.get('scores', {})
        
        print(f"\n📌 Document {i}/{len(LAST_RETRIEVED)}")
        print("-" * 70)
        print(f"📁 Source: {meta.get('source', 'unknown')}")
        
        pages = meta.get('pages', [])
        if pages:
            print(f"📄 Pages: {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"📄 Page: {pages[0]}")
        
        print(f"📂 Section: {meta.get('section', 'N/A')}")
        print(f"🔍 Type: {doc['retrieval_type']}")
        
        print(f"\n⚖️  Scores:")
        print(f"   Semantic: {scores.get('semantic', 0):.3f} | BM25: {scores.get('bm25', 0):.3f} | Keyword: {scores.get('keyword', 0):.3f}")
        print(f"   Rerank: {scores.get('rerank', 0):.3f} | Boost: {scores.get('boost', 1.0):.1f}x | Final: {doc['score']:.3f}")
        
        print(f"\n📝 Preview (500 chars):")
        print("-" * 70)
        print(doc['text'][:500] + "..." if len(doc['text']) > 500 else doc['text'])


def show_config():
    """Display current model configuration"""
    print("\n" + "=" * 70)
    print("⚙️  MODEL CONFIGURATION")
    print("=" * 70)
    print(f"\n🔗 API: {VLLM_URL}")
    print(f"🤖 Model: {VLLM_MODEL_NAME}")
    print(f"\n📊 Parameters:")
    for key, value in MODEL_CONFIG.items():
        print(f"   {key}: {value}")
    print(f"\n⚖️  Hybrid Weights:")
    print(f"   Semantic: {HYBRID_WEIGHTS['semantic']:.0%}")
    print(f"   BM25: {HYBRID_WEIGHTS['bm25']:.0%}")
    print(f"   Keyword: {HYBRID_WEIGHTS['keyword']:.0%}")


def change_config():
    """Interactive menu to change model parameters"""
    print("\n=== Change Model Configuration ===")
    print("Available parameters:")
    for i, (key, value) in enumerate(MODEL_CONFIG.items(), 1):
        print(f"{i}. {key}: {value}")
    print("0. Back")
    
    try:
        choice = int(input("\nSelect parameter (0-{}): ".format(len(MODEL_CONFIG))))
        if choice == 0:
            return
        
        keys = list(MODEL_CONFIG.keys())
        if 1 <= choice <= len(keys):
            param = keys[choice - 1]
            current = MODEL_CONFIG[param]
            
            if param in ["temperature", "top_p"]:
                new_value = float(input(f"New value for {param} (0-1, current={current}): "))
                MODEL_CONFIG[param] = max(0.0, min(1.0, new_value))
            elif param in ["max_tokens", "top_k"]:
                new_value = int(input(f"New value for {param} (current={current}): "))
                MODEL_CONFIG[param] = max(1, new_value)
            elif param == "repetition_penalty":
                new_value = float(input(f"New value for {param} (1.0-2.0, current={current}): "))
                MODEL_CONFIG[param] = max(1.0, min(2.0, new_value))
            
            print(f"✅ {param} = {MODEL_CONFIG[param]}")
    except (ValueError, IndexError):
        print("Invalid input.")


def adjust_weights():
    """Adjust hybrid search weights"""
    global HYBRID_WEIGHTS
    
    print("\n" + "=" * 70)
    print("⚖️  HYBRID SEARCH WEIGHTS")
    print("=" * 70)
    print(f"\nCurrent: Semantic={HYBRID_WEIGHTS['semantic']:.0%}, BM25={HYBRID_WEIGHTS['bm25']:.0%}, Keyword={HYBRID_WEIGHTS['keyword']:.0%}")
    
    print("\nPresets:")
    print("  1. Balanced:        50% semantic, 35% BM25, 15% keyword")
    print("  2. Semantic-heavy:  70% semantic, 20% BM25, 10% keyword")
    print("  3. Exact-match:     30% semantic, 50% BM25, 20% keyword")
    print("  4. Technical IDs:   20% semantic, 40% BM25, 40% keyword")
    print("  0. Cancel")
    
    try:
        choice = input("\nSelect preset (0-4): ").strip()
        
        presets = {
            '1': {'semantic': 0.50, 'bm25': 0.35, 'keyword': 0.15},
            '2': {'semantic': 0.70, 'bm25': 0.20, 'keyword': 0.10},
            '3': {'semantic': 0.30, 'bm25': 0.50, 'keyword': 0.20},
            '4': {'semantic': 0.20, 'bm25': 0.40, 'keyword': 0.40},
        }
        
        if choice in presets:
            HYBRID_WEIGHTS = presets[choice]
            print(f"✅ Updated: S={HYBRID_WEIGHTS['semantic']:.0%}, B={HYBRID_WEIGHTS['bm25']:.0%}, K={HYBRID_WEIGHTS['keyword']:.0%}")
        elif choice != '0':
            print("Invalid choice")
    except Exception as e:
        print(f"Error: {e}")


def show_stats():
    """Show system statistics"""
    print("\n" + "=" * 70)
    print("📊 SYSTEM STATISTICS")
    print("=" * 70)
    
    print(f"\n📚 RAG Store:")
    print(f"   Chunks: {len(retriever.texts)}")
    print(f"   Documents: {len(set(m.get('source', 'unknown') for m in retriever.metadatas))}")
    print(f"   Keywords (upper): {len(retriever.keyword_index)}")
    print(f"   Keywords (lower): {len(retriever.keyword_index_lower)}")
    print(f"   BM25 enabled: {retriever.bm25_index is not None}")
    
    if CONVERSATION_HISTORY:
        print(f"\n💬 Session:")
        print(f"   Conversation turns: {len(CONVERSATION_HISTORY)}")


def print_help():
    """Print help message"""
    print("\n" + "=" * 70)
    print("📖 HYBRID RAG SYSTEM - HELP")
    print("=" * 70)
    print("\n🔍 QUERY TIPS:")
    print('   Use "quotes" for exact matches (e.g., "SN0012" or "pucch")')
    print('   Case-insensitive: sn0012 = SN0012')
    print('   Say "more" or "continue" for follow-ups')
    print("\n📊 COMMANDS:")
    print("   !help      - Show this help")
    print("   !show      - Show retrieved documents details")
    print("   !config    - Show current configuration")
    print("   !change    - Change model parameters")
    print("   !weights   - Change hybrid search weights")
    print("   !stats     - Show statistics")
    print("   !clear     - Clear conversation history")
    print("   exit       - Exit")
    print("=" * 70)


# Main loop
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎉 HYBRID RAG System Ready!")
    print("=" * 70)
    print("🔀 Search: Semantic + BM25 + Keywords")
    print("🤖 LLM: text-generation-webui API")
    print("💡 Type !help for commands")
    print("=" * 70)
    
    while True:
        try:
            query = input("\n🔎 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'exit':
                print("\n👋 Goodbye!")
                break
            
            # Handle commands
            if query.startswith('!'):
                cmd = query[1:].lower().split()[0]
                
                if cmd == 'help':
                    print_help()
                elif cmd == 'show':
                    show_retrieved_details()
                elif cmd == 'config':
                    show_config()
                elif cmd == 'change':
                    change_config()
                elif cmd == 'weights':
                    adjust_weights()
                elif cmd == 'stats':
                    show_stats()
                elif cmd == 'clear':
                    CONVERSATION_HISTORY = []
                    LAST_RETRIEVED = []
                    print("✅ Conversation cleared")
                elif cmd == 'history':
                    for i, h in enumerate(CONVERSATION_HISTORY[-5:], 1):
                        print(f"\n{i}. Q: {h['question']}")
                        print(f"   A: {h['answer'][:200]}...")
                else:
                    print(f"❌ Unknown command: {cmd}. Type !help")
                continue
            
            # Process query
            print(f"\n🔍 Searching...")
            retrieved = retriever.smart_retrieve(query)
            LAST_RETRIEVED = retrieved
            
            # Show results
            print(f"\n📚 Retrieved {len(retrieved)} chunks:")
            for i, doc in enumerate(retrieved[:5], 1):
                meta = doc['metadata']
                scores = doc.get('scores', {})
                
                source = meta.get('source', 'unknown')
                pages = meta.get('pages', [])
                page_str = f"p{pages[0]}-{pages[-1]}" if pages and len(pages) > 1 else f"p{pages[0]}" if pages else "p?"
                
                score_parts = []
                if scores.get('semantic', 0) > 0:
                    score_parts.append(f"S:{scores['semantic']:.2f}")
                if scores.get('bm25', 0) > 0:
                    score_parts.append(f"B:{scores['bm25']:.2f}")
                if scores.get('keyword', 0) > 0:
                    score_parts.append(f"K:{scores['keyword']:.2f}")
                
                print(f"  {i}. [{doc['retrieval_type']}] {source} {page_str} | {doc['score']:.3f} ({' '.join(score_parts)})")
            
            if len(retrieved) > 5:
                print(f"  ... and {len(retrieved) - 5} more (!show for details)")
            
            # Generate response
            print(f"\n🤖 Generating response...")
            answer = ask_llm_with_context(query, retrieved)
            
            print(f"\n💬 ANSWER:\n{answer}")
            
            # Store history
            CONVERSATION_HISTORY.append({
                'question': query,
                'answer': answer,
                'retrieved': [r['idx'] for r in retrieved]
            })
            
            print("\n" + "-" * 70)
            print("💡 Commands: !show | !weights | !config | !help")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


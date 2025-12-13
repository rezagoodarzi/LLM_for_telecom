"""
Index Builder for RAG System
============================

Creates and manages:
1. FAISS index for semantic search
2. BM25 index for keyword/term matching
3. Metadata indices for filtering
4. Document-level index for 2-stage retrieval

With checkpoint support for resumable indexing.
"""

import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import torch
import gc
from contextlib import nullcontext

import sys
sys.path.append('..')
from config import settings
from utils.helpers import (
    tokenize_for_bm25, extract_keywords, extract_phrases,
    TextDeduplicator, compute_text_hash
)


class IndexBuilder:
    """Build and manage RAG indices"""
    
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or settings.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.embedder = None
        self.device = None
        
    def _init_embedder(self):
        """Initialize embedding model"""
        if self.embedder is None:
            print(f"📦 Loading embedding model: {settings.EMBEDDING_MODEL}")
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL, device=self.device)
            
            if settings.USE_FP16 and self.device == "cuda":
                self.embedder.half()
                print("   ✓ FP16 mode enabled")
            
            print(f"   ✓ Using device: {self.device}")
    
    def build_from_documents(self, documents: List[Dict]) -> Dict:
        """
        Build all indices from processed documents.
        
        Args:
            documents: List from DocumentProcessor
        
        Returns:
            Statistics dict
        """
        print("\n" + "=" * 70)
        print("🔨 Building RAG Indices")
        print("=" * 70)
        
        # Collect all chunks
        all_chunks = []
        document_summaries = []
        
        for doc in documents:
            all_chunks.extend(doc['chunks'])
            document_summaries.append({
                'path': doc['path'],
                'metadata': doc['metadata'],
                'summary': doc['document_summary'],
                'chunk_indices': list(range(len(all_chunks) - len(doc['chunks']), len(all_chunks)))
            })
        
        print(f"   Total chunks: {len(all_chunks):,}")
        print(f"   Total documents: {len(document_summaries):,}")
        
        # Build metadata structures
        texts, metadatas, indices = self._build_metadata(all_chunks)
        
        # Build embeddings with checkpoint support
        embeddings = self._build_embeddings(texts)
        
        # Build FAISS index
        faiss_index = self._build_faiss_index(embeddings)
        
        # Build BM25 index
        bm25_index, bm25_corpus = self._build_bm25_index(texts)
        
        # Build document-level index (for 2-stage retrieval)
        doc_embeddings = self._build_document_embeddings(document_summaries)
        doc_faiss_index = self._build_faiss_index(doc_embeddings)
        
        # Save everything
        self._save_indices(
            faiss_index, bm25_index, bm25_corpus,
            texts, metadatas, indices,
            document_summaries, doc_faiss_index
        )
        
        stats = {
            'total_chunks': len(texts),
            'total_documents': len(document_summaries),
            'unique_keywords': len(indices['keyword_index']),
            'unique_sections': len(indices['section_index']),
        }
        
        print("\n" + "=" * 70)
        print("✅ Index Building Complete!")
        print("=" * 70)
        print(f"   Chunks indexed: {stats['total_chunks']:,}")
        print(f"   Documents indexed: {stats['total_documents']:,}")
        print(f"   Unique keywords: {stats['unique_keywords']:,}")
        
        return stats
    
    def _build_metadata(self, chunks: List[Dict]) -> tuple:
        """Build metadata structures and indices"""
        print("\n📋 Building metadata indices...")
        
        texts = []
        metadatas = []
        
        keyword_index = defaultdict(list)
        keyword_index_lower = defaultdict(list)
        phrase_index = defaultdict(list)
        section_index = defaultdict(list)
        vendor_index = defaultdict(list)
        
        deduplicator = TextDeduplicator(settings.SIMILARITY_THRESHOLD)
        
        for idx, chunk in enumerate(tqdm(chunks, desc="Processing metadata")):
            texts.append(chunk['text'])
            
            # Track duplicates
            deduplicator.add(idx, chunk['text'])
            
            # Build keyword indices
            for kw in chunk.get('keywords', []):
                keyword_index[kw].append(idx)
            
            for kw in chunk.get('keywords_lower', []):
                keyword_index_lower[kw].append(idx)
            
            # Build phrase index
            for phrase in chunk.get('phrases', []):
                phrase_index[phrase].append(idx)
            
            # Build section index
            section = chunk.get('section')
            if section:
                section_index[section].append(idx)
            
            # Build vendor index
            vendor = chunk.get('vendor')
            if vendor:
                vendor_index[vendor].append(idx)
            
            # Create metadata entry
            meta = {
                'source': chunk['source'],
                'source_path': chunk.get('source_path', ''),
                'pages': chunk['pages'],
                'page_start': chunk['page_start'],
                'page_end': chunk['page_end'],
                'section': chunk['section'],
                'chunk_id': chunk['chunk_id'],
                'keywords': chunk.get('keywords', [])[:10],
                'char_count': chunk['char_count'],
                'word_count': chunk['word_count'],
                # Filtering metadata
                'vendor': chunk.get('vendor'),
                'version': chunk.get('version'),
                'release': chunk.get('release'),
                'doc_type': chunk.get('doc_type'),
                'topics': chunk.get('topics', []),
            }
            metadatas.append(meta)
        
        # Build adjacency map
        adjacency_map = self._build_adjacency_map(metadatas)
        
        # Get duplicate map
        duplicate_map = deduplicator.get_duplicate_map()
        
        indices = {
            'keyword_index': dict(keyword_index),
            'keyword_index_lower': dict(keyword_index_lower),
            'phrase_index': dict(phrase_index),
            'section_index': dict(section_index),
            'vendor_index': dict(vendor_index),
            'adjacency_map': adjacency_map,
            'duplicate_map': duplicate_map,
        }
        
        print(f"   ✓ Keywords (uppercase): {len(keyword_index):,}")
        print(f"   ✓ Keywords (lowercase): {len(keyword_index_lower):,}")
        print(f"   ✓ Sections: {len(section_index):,}")
        print(f"   ✓ Vendors: {len(vendor_index):,}")
        print(f"   ✓ Duplicates found: {len(duplicate_map):,}")
        
        return texts, metadatas, indices
    
    def _build_adjacency_map(self, metadatas: List[Dict]) -> Dict[int, List[int]]:
        """Build adjacency map for chunk relationships"""
        print("   Building adjacency map...")
        
        # Group by source document
        doc_chunks = defaultdict(list)
        for idx, meta in enumerate(metadatas):
            doc_chunks[meta['source']].append((idx, meta['pages']))
        
        adjacency_map = defaultdict(list)
        
        for source, chunks in doc_chunks.items():
            # Build page → chunk mapping
            page_to_chunk = {}
            for idx, pages in chunks:
                for p in pages:
                    page_to_chunk[p] = idx
            
            # Find adjacent chunks
            for idx, pages in chunks:
                max_page = max(pages) if pages else 0
                min_page = min(pages) if pages else 0
                
                # Next chunk
                if max_page + 1 in page_to_chunk:
                    next_idx = page_to_chunk[max_page + 1]
                    if next_idx != idx and next_idx not in adjacency_map[idx]:
                        adjacency_map[idx].append(next_idx)
                
                # Previous chunk
                if min_page - 1 in page_to_chunk:
                    prev_idx = page_to_chunk[min_page - 1]
                    if prev_idx != idx and prev_idx not in adjacency_map[idx]:
                        adjacency_map[idx].append(prev_idx)
        
        return {str(k): v for k, v in adjacency_map.items()}
    
    def _build_embeddings(self, texts: List[str]) -> np.ndarray:
        """Build embeddings with checkpoint support"""
        self._init_embedder()
        
        print(f"\n🧠 Creating embeddings...")
        print(f"   Total texts: {len(texts):,}")
        print(f"   Batch size: {settings.BATCH_SIZE}")
        
        # Check for checkpoint
        start_idx = 0
        embeddings = np.zeros((len(texts), settings.EMBEDDING_DIM), dtype=np.float32)
        
        if os.path.exists(settings.EMBEDDING_CHECKPOINT):
            try:
                checkpoint = np.load(settings.EMBEDDING_CHECKPOINT)
                saved_emb = checkpoint['embeddings']
                start_idx = int(checkpoint['last_idx'])
                
                if saved_emb.shape[0] == len(texts):
                    embeddings[:] = saved_emb
                    print(f"   ✓ Resuming from index {start_idx:,}")
            except Exception as e:
                print(f"   ⚠️ Could not load checkpoint: {e}")
                start_idx = 0
        
        # Process remaining
        batch_size = settings.BATCH_SIZE
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        batch_count = 0
        for i in tqdm(range(start_idx, len(texts), batch_size), desc="Embedding"):
            batch_end = min(i + batch_size, len(texts))
            batch = texts[i:batch_end]
            
            try:
                with torch.cuda.amp.autocast() if settings.USE_FP16 and self.device == "cuda" else nullcontext():
                    embs = self.embedder.encode(
                        batch,
                        batch_size=batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                
                embeddings[i:batch_end] = embs.astype(np.float32)
                del embs
                
            except Exception as e:
                print(f"\n⚠️ Error at batch {i}: {e}")
                # Save checkpoint
                np.savez(settings.EMBEDDING_CHECKPOINT, embeddings=embeddings, last_idx=i)
                print(f"   💾 Checkpoint saved at index {i}")
                raise
            
            batch_count += 1
            
            # Periodic checkpoint
            if batch_count % settings.EMBEDDING_SAVE_INTERVAL == 0:
                np.savez(settings.EMBEDDING_CHECKPOINT, embeddings=embeddings, last_idx=batch_end)
            
            # Memory cleanup
            if batch_count % 5 == 0:
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
        
        # Remove checkpoint on success
        if os.path.exists(settings.EMBEDDING_CHECKPOINT):
            os.remove(settings.EMBEDDING_CHECKPOINT)
        
        print(f"   ✓ Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _build_document_embeddings(self, doc_summaries: List[Dict]) -> np.ndarray:
        """Build document-level embeddings for 2-stage retrieval"""
        self._init_embedder()
        
        print("\n📄 Creating document-level embeddings...")
        
        summaries = [d['summary'] for d in doc_summaries]
        
        embeddings = self.embedder.encode(
            summaries,
            batch_size=settings.BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"   ✓ Document embeddings: {embeddings.shape}")
        return embeddings.astype(np.float32)
    
    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index"""
        print("   Building FAISS index...")
        
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        print(f"   ✓ FAISS index: {index.ntotal:,} vectors")
        return index
    
    def _build_bm25_index(self, texts: List[str]) -> tuple:
        """Build BM25 index"""
        print("\n📊 Building BM25 index...")
        
        corpus = []
        for text in tqdm(texts, desc="Tokenizing"):
            tokens = tokenize_for_bm25(text)
            corpus.append(tokens)
        
        bm25 = BM25Okapi(corpus, k1=settings.BM25_K1, b=settings.BM25_B)
        
        print(f"   ✓ BM25 index: {len(corpus):,} docs, avg length {bm25.avgdl:.1f}")
        return bm25, corpus
    
    def _save_indices(
        self,
        faiss_index: faiss.Index,
        bm25_index,
        bm25_corpus: List[List[str]],
        texts: List[str],
        metadatas: List[Dict],
        indices: Dict,
        doc_summaries: List[Dict],
        doc_faiss_index: faiss.Index
    ):
        """Save all indices to disk"""
        print("\n💾 Saving indices...")
        
        # FAISS chunk index
        faiss_path = os.path.join(self.output_dir, "faiss.index")
        faiss.write_index(faiss_index, faiss_path)
        print(f"   ✓ FAISS index: {os.path.getsize(faiss_path) / 1024 / 1024:.1f} MB")
        
        # FAISS document index
        doc_faiss_path = os.path.join(self.output_dir, "faiss_documents.index")
        faiss.write_index(doc_faiss_index, doc_faiss_path)
        print(f"   ✓ Document FAISS: {os.path.getsize(doc_faiss_path) / 1024 / 1024:.1f} MB")
        
        # BM25 index
        bm25_path = os.path.join(self.output_dir, "bm25.pkl")
        with open(bm25_path, 'wb') as f:
            pickle.dump(bm25_index, f)
        print(f"   ✓ BM25 index: {os.path.getsize(bm25_path) / 1024 / 1024:.1f} MB")
        
        # BM25 corpus
        corpus_path = os.path.join(self.output_dir, "bm25_corpus.pkl")
        with open(corpus_path, 'wb') as f:
            pickle.dump(bm25_corpus, f)
        
        # Main metadata file
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        save_data = {
            'texts': texts,
            'metadatas': metadatas,
            'keywords_index': indices['keyword_index'],
            'keywords_index_lower': indices['keyword_index_lower'],
            'phrases_index': indices['phrase_index'],
            'section_index': indices['section_index'],
            'vendor_index': indices['vendor_index'],
            'adjacency_map': indices['adjacency_map'],
            'duplicate_map': indices['duplicate_map'],
            'config': {
                'chunk_size': settings.CHUNK_SIZE,
                'chunk_overlap': settings.CHUNK_OVERLAP,
                'model': settings.EMBEDDING_MODEL,
                'embedding_dim': settings.EMBEDDING_DIM,
                'bm25_k1': settings.BM25_K1,
                'bm25_b': settings.BM25_B,
                'similarity_threshold': settings.SIMILARITY_THRESHOLD,
            }
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False)
        print(f"   ✓ Metadata: {os.path.getsize(metadata_path) / 1024 / 1024:.1f} MB")
        
        # Document summaries
        doc_path = os.path.join(self.output_dir, "documents.json")
        with open(doc_path, 'w', encoding='utf-8') as f:
            json.dump(doc_summaries, f, ensure_ascii=False)
        print(f"   ✓ Documents: {os.path.getsize(doc_path) / 1024 / 1024:.1f} MB")


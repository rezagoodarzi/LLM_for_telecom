# word_embbeding_optimized.py - OPTIMIZED for High-End Hardware
# Designed for: RTX 3080 Ti (12GB), 64GB RAM, i9-12900
# Handles large datasets (10GB+) efficiently

import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
import hashlib
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict, Set, Tuple, Optional
import string
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import torch
import gc
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HARDWARE-OPTIMIZED CONFIGURATION
# ============================================================================

DOCS_DIR = "/home/fatemebookanian/qwen_code/PDF"
MODEL_NAME = "/home/fatemebookanian/models/BGE-m3"
OUT_DIR = "./rag_store_3gpp_bem"
CHECKPOINT_FILE = "./rag_store_3gpp_bem/metadata_checkpoint.pkl"  # Checkpoint for resuming
EMBEDDING_CHECKPOINT_FILE = "./rag_store_3gpp_bem/embedding_checkpoint.npz"  # Embedding progress
EMBEDDING_SAVE_INTERVAL = 50  # Save embedding progress every N batches

# GPU Optimization (RTX 3080 Ti with 16GB VRAM)
# BGE-M3 needs smaller batches due to large model size (2.2GB)
BATCH_SIZE = 8               # Safe for BGE-M3 on 16GB VRAM
USE_FP16 = True               # Half precision for 2x speed
DEVICE = None                 # Auto-detect

# CPU Optimization (i9-12900 with 16 cores)
NUM_WORKERS = 12             # For parallel file processing
NUM_IO_THREADS = 8            # For I/O operations

# Memory Optimization (64GB RAM)
CHUNK_SAVE_INTERVAL = 50000   # Save progress every N chunks
MAX_MEMORY_GB = 50            # Max memory to use before flushing

# Chunking Parameters
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 800
PAGE_MERGE_THRESHOLD = 500
MAX_CHUNK_SIZE = 12000
EMB_DIM = 1024

# File Types
SUPPORTED_EXTENSIONS = ['.pdf', '.html', '.htm', '.md']

# BM25 Configuration
BM25_K1 = 1.5
BM25_B = 0.80

# Deduplication
SIMILARITY_THRESHOLD = 0.85
MIN_HASH_SHINGLES = 5

# Section Detection Patterns
SECTION_PATTERNS = [
    (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'numbered'),
    (r'^([A-Z](?:\.\d+)*)\s+(.+)$', 'letter'),
    (r'^((?:I|II|III|IV|V|VI|VII|VIII|IX|X)+(?:\.\d+)*)\s+(.+)$', 'roman'),
    (r'^(Chapter|Section|Part|Module)\s+(\d+|\w+):\s*(.+)$', 'keyword'),
    (r'^([A-Z][A-Z\s]{10,})$', 'caps'),
    (r'^(#{1,6})\s+(.+)$', 'markdown'),
]

# ============================================================================
# INITIALIZATION
# ============================================================================

os.makedirs(OUT_DIR, exist_ok=True)

def print_header():
    print("\n" + "=" * 70)
    print("🚀 OPTIMIZED RAG INDEXING SYSTEM v3.0")
    print("   Designed for High-End Hardware")
    print("=" * 70)

def detect_hardware():
    """Detect and configure hardware"""
    global DEVICE, BATCH_SIZE, USE_FP16
    
    print("\n🖥️  Hardware Detection:")
    
    # CPU info
    cpu_count = multiprocessing.cpu_count()
    print(f"   CPU Cores: {cpu_count}")
    
    # RAM info (approximate)
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   RAM: {ram_gb:.1f} GB")
    except:
        ram_gb = 64
        print(f"   RAM: (psutil not installed, assuming 64GB)")
    
    # GPU info
    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name}")
        print(f"   VRAM: {gpu_mem:.1f} GB")
        

        print(f"   ✓ Using CUDA with batch size {BATCH_SIZE}")
    else:
        DEVICE = "cpu"
        BATCH_SIZE = 64
        USE_FP16 = False
        print(f"   ⚠️  No GPU detected, using CPU (slower)")
    
    print(f"   FP16 Mode: {'Enabled' if USE_FP16 else 'Disabled'}")
    return DEVICE

# ============================================================================
# TEXT EXTRACTION FUNCTIONS (Optimized for parallel processing)
# ============================================================================

def extract_text_from_pdf(path: str) -> List[str]:
    """Extract text from PDF with page preservation"""
    try:
        doc = fitz.open(path)
        pages = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return pages
    except Exception as e:
        return []

def extract_text_from_html(path: str) -> List[str]:
    """Extract text from HTML files"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return [text] if text.strip() else []
    except Exception as e:
        return []

def extract_text_from_markdown(path: str) -> List[str]:
    """Extract text from Markdown files"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            md_content = f.read()
        text = md_content
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        text = re.sub(r'[*_~]+', '', text)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        lines = (line.strip() for line in text.splitlines())
        text = '\n'.join(line for line in lines if line)
        return [text] if text.strip() else []
    except Exception as e:
        return []

# ============================================================================
# DOCUMENT PROCESSOR CLASS
# ============================================================================

class DocumentProcessor:
    """Advanced document processing with smart chunking"""
    
    def __init__(self):
        self.current_section_stack = []
        self.section_hierarchy = defaultdict(list)
    
    def detect_section(self, text: str) -> Optional[Dict]:
        """Detect if text is a section header"""
        for pattern, section_type in SECTION_PATTERNS:
            match = re.match(pattern, text.strip())
            if match:
                return {
                    'type': section_type,
                    'text': text.strip(),
                    'match': match.groups()
                }
        return None
    
    def extract_document_structure(self, pages: List[str]) -> Dict:
        """Build document structure with sections"""
        structure = {
            'sections': [],
            'page_sections': {},
            'section_pages': defaultdict(list)
        }
        
        current_section = "Document Start"
        
        for page_num, page_text in enumerate(pages):
            lines = page_text.split('\n')
            for line in lines[:20]:
                section = self.detect_section(line)
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
    
    def smart_chunk_pages(self, pages: List[str], structure: Dict, source_file: str) -> List[Dict]:
        """Create smart chunks respecting section boundaries"""
        chunks = []
        current_chunk = ""
        current_pages = []
        current_section = None
        
        for page_num, page_text in enumerate(pages):
            page_id = page_num + 1
            section = structure['page_sections'].get(page_id, "Unknown")
            
            if section != current_section and current_chunk:
                if len(current_chunk) > 100:
                    chunks.append({
                        'text': current_chunk,
                        'pages': current_pages.copy(),
                        'section': current_section,
                        'source': source_file
                    })
                current_chunk = ""
                current_pages = []
            
            current_section = section
            
            if len(current_chunk) + len(page_text) <= MAX_CHUNK_SIZE:
                current_chunk += f"\n[Page {page_id}]\n{page_text}"
                current_pages.append(page_id)
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'pages': current_pages.copy(),
                        'section': current_section,
                        'source': source_file
                    })
                
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap_text + f"\n[Page {page_id}]\n{page_text}"
                current_pages = [page_id]
        
        if current_chunk and len(current_chunk) > 100:
            chunks.append({
                'text': current_chunk,
                'pages': current_pages.copy(),
                'section': current_section,
                'source': source_file
            })
        
        return chunks

# ============================================================================
# PARALLEL FILE PROCESSING
# ============================================================================

def process_single_file(args) -> Optional[Dict]:
    """Process a single file - for parallel execution"""
    file_path, base_path = args
    try:
        relative_path = file_path.relative_to(base_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            pages = extract_text_from_pdf(str(file_path))
        elif suffix in ['.html', '.htm']:
            pages = extract_text_from_html(str(file_path))
        elif suffix == '.md':
            pages = extract_text_from_markdown(str(file_path))
        else:
            return None
        
        if not pages:
            return None
        
        return {
            'path': str(relative_path),
            'pages': pages,
            'suffix': suffix
        }
    except Exception as e:
        return None

def process_files_parallel(files: List[Path], base_path: Path) -> List[Dict]:
    """Process files in parallel using multiple workers"""
    print(f"\n📁 Processing {len(files)} files with {NUM_WORKERS} workers...")
    
    file_args = [(f, base_path) for f in files]
    results = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_file, arg): arg for arg in file_args}
        
        with tqdm(total=len(futures), desc="Reading files", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                pbar.update(1)
    
    return results

# ============================================================================
# KEYWORD AND PHRASE EXTRACTION
# ============================================================================

def extract_keywords_advanced(text: str) -> Tuple[List[str], List[str]]:
    """Extract keywords - returns (uppercase, lowercase) lists"""
    keywords = set()
    keywords_lower = set()
    
    text_upper = text.upper()
    text_lower = text.lower()
    
    patterns = [
        r'\b[A-Z]{2,10}\d{2,8}\b',
        r'\b\d{2,8}[A-Z]{2,10}\b',
        r'\b[A-Z]{2,10}\d+[A-Z]+\b',
        r'\b\d+[A-Z]+\d+\b',
        r'\b\d+\.\d+(?:\.\d+)*\b',
        r'\bV\d+\.\d+\b',
        r'\bREV\s*\d+\b',
        r'\b[A-Z]{2,10}-\d{2,10}\b',
        r'\b\d{2,10}-[A-Z]{2,10}\b',
        r'\b[A-Z]+-[A-Z]+-\d+\b',
        r'\b[A-Z]{2,}\.\d+(?:\.\d+)*\b',
        r'\b[A-Z]{3,15}\b',
        r'\b\d+(?:\.\d+)?\s*(?:MHz|GHz|kHz|dB|dBm|ms|us|ns|Mbps|Gbps|KB|MB|GB)\b',
        r'\b3GPP\b',
        r'\bTS\s*\d+\.\d+\b',
        r'\bTR\s*\d+\.\d+\b',
        r'\b5G\s*NR\b',
        r'\bLTE\b',
        r'\bNR\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        for m in matches:
            keywords.add(m.upper())
            keywords_lower.add(m.lower())
    
    cap_terms = re.findall(r'\b[A-Z]{3,}\b', text)
    for t in cap_terms:
        if len(t) >= 3:
            keywords.add(t.upper())
            keywords_lower.add(t.lower())
    
    quoted = re.findall(r'"([^"]+)"', text)
    for q in quoted:
        if len(q) > 2:
            keywords.add(q.upper())
            keywords_lower.add(q.lower())
    
    lower_patterns = [
        r'\b[a-z]{2,10}\d{2,8}\b',
        r'\b\d{2,8}[a-z]{2,10}\b',
    ]
    
    for pattern in lower_patterns:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            keywords.add(m.upper())
            keywords_lower.add(m.lower())
    
    return list(keywords), list(keywords_lower)

def extract_phrases(text: str) -> List[str]:
    """Extract important multi-word phrases"""
    phrases = []
    
    technical_phrases = [
        "gap length", "site to site", "coverage area", "maximum distance",
        "minimum distance", "transmission distance", "coverage radius",
        "cell radius", "propagation distance", "link distance", "hop distance",
        "basic safety", "safety aspects", "safety requirements", "safety precautions",
        "installation guide", "installation procedure", "installation requirements",
        "technical specifications", "system requirements", "minimum requirements",
        "frequency band", "frequency range", "carrier frequency", "bandwidth requirements",
        "remote radio head", "base station", "antenna system", "feeder cable",
        "configuration parameters", "default settings", "factory settings",
        "maintenance procedure", "preventive maintenance", "corrective maintenance",
        "troubleshooting guide", "fault diagnosis", "error codes", "alarm messages",
        "operating temperature", "storage temperature", "humidity range",
        "regulatory compliance", "certification requirements", "type approval",
        "key performance indicators", "kpi metrics", "availability target",
        "reference manual", "user manual", "operation manual", "service manual",
    ]
    
    text_lower = text.lower()
    for phrase in technical_phrases:
        if phrase in text_lower:
            phrases.append(phrase)
    
    section_titles = re.findall(r'^([A-Za-z][^:]{5,50}):$', text, re.MULTILINE)
    phrases.extend([t.lower().strip() for t in section_titles])
    
    return phrases

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 indexing"""
    text_lower = text.lower()
    
    protected = []
    tech_patterns = [
        r'\b[a-z]{2,10}\d{2,8}\b',
        r'\b\d{2,8}[a-z]{2,10}\b',
        r'\b\d+\.\d+(?:\.\d+)*\b',
        r'\b[a-z]+-[a-z0-9]+\b',
        r'\b[a-z]+_[a-z0-9]+\b',
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_lower)
        protected.extend(matches)
    
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', text_lower)
    tokens = cleaned.split()
    tokens.extend(protected)
    tokens = [t for t in tokens if len(t) >= 2 or t.upper() in ['5G', 'NR', 'LTE', 'RF', 'IP', 'TX', 'RX']]
    
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    
    return unique_tokens

# ============================================================================
# DEDUPLICATION
# ============================================================================

def compute_text_hash(text: str) -> str:
    """Compute hash for deduplication"""
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
    """Compute Jaccard similarity"""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

class DuplicateDetector:
    """Detect duplicate/similar chunks"""
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = similarity_threshold
        self.chunk_hashes: Dict[int, str] = {}
        self.chunk_shingles: Dict[int, Set[int]] = {}
        self.duplicate_groups: Dict[str, List[int]] = defaultdict(list)
        self.similar_pairs: List[Tuple[int, int, float]] = []
    
    def add_chunk(self, idx: int, text: str):
        """Add chunk and detect duplicates"""
        text_hash = compute_text_hash(text)
        shingles = compute_shingle_hash(text)
        
        self.chunk_hashes[idx] = text_hash
        self.chunk_shingles[idx] = shingles
        self.duplicate_groups[text_hash].append(idx)
        
        # Only check against recent chunks for efficiency
        recent_chunks = list(self.chunk_shingles.items())[-50:]
        for other_idx, other_shingles in recent_chunks:
            if other_idx != idx:
                similarity = compute_jaccard_similarity(shingles, other_shingles)
                if similarity >= self.threshold:
                    self.similar_pairs.append((idx, other_idx, similarity))
    
    def get_duplicate_map(self) -> Dict[int, int]:
        """Get mapping of duplicates to primary chunk"""
        duplicate_map = {}
        for group in self.duplicate_groups.values():
            if len(group) > 1:
                primary = group[0]
                for dup in group[1:]:
                    duplicate_map[dup] = primary
        return duplicate_map
    
    def get_similarity_map(self) -> Dict[int, List[Tuple[int, float]]]:
        """Get mapping of similar chunks"""
        similarity_map = defaultdict(list)
        for idx1, idx2, sim in self.similar_pairs:
            similarity_map[idx1].append((idx2, sim))
            similarity_map[idx2].append((idx1, sim))
        return dict(similarity_map)

# ============================================================================
# OPTIMIZED ADJACENCY MAP (O(n) instead of O(n²))
# ============================================================================

def build_adjacency_map_fast(metadatas: List[Dict]) -> Dict[int, List[int]]:
    """Build adjacency map in O(n) time"""
    print("🔗 Building adjacency map (optimized O(n))...")
    
    # Group by source
    doc_chunks = defaultdict(list)
    for idx, meta in enumerate(metadatas):
        doc_chunks[meta['source']].append((idx, meta['pages']))
    
    adjacency_map = defaultdict(list)
    
    for source, chunks in tqdm(doc_chunks.items(), desc="Building adjacency", leave=False):
        # Build page -> chunk mapping
        page_to_chunk = {}
        for idx, pages in chunks:
            for p in pages:
                page_to_chunk[p] = idx
        
        # Find adjacent chunks
        for idx, pages in chunks:
            max_page = max(pages)
            min_page = min(pages)
            
            if max_page + 1 in page_to_chunk:
                next_idx = page_to_chunk[max_page + 1]
                if next_idx != idx and next_idx not in adjacency_map[idx]:
                    adjacency_map[idx].append(next_idx)
            
            if min_page - 1 in page_to_chunk:
                prev_idx = page_to_chunk[min_page - 1]
                if prev_idx != idx and prev_idx not in adjacency_map[idx]:
                    adjacency_map[idx].append(prev_idx)
    
    return dict(adjacency_map)

# ============================================================================
# CHECKPOINT SAVE/LOAD (Resume capability)
# ============================================================================

def save_metadata_checkpoint(texts, metadatas, bm25_corpus, keyword_index, keyword_index_lower,
                              phrase_index, section_index, adjacency_map, document_structures,
                              duplicate_map, similarity_map, total_docs):
    """Save metadata checkpoint to resume later without rebuilding"""
    print("\n💾 Saving metadata checkpoint...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    checkpoint = {
        'texts': texts,
        'metadatas': metadatas,
        'bm25_corpus': bm25_corpus,
        'keyword_index': dict(keyword_index),
        'keyword_index_lower': dict(keyword_index_lower),
        'phrase_index': dict(phrase_index),
        'section_index': dict(section_index),
        'adjacency_map': adjacency_map,
        'document_structures': document_structures,
        'duplicate_map': duplicate_map,
        'similarity_map': similarity_map,
        'total_docs': total_docs,
    }
    
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    size_mb = os.path.getsize(CHECKPOINT_FILE) / (1024 * 1024)
    print(f"   ✓ Checkpoint saved: {CHECKPOINT_FILE} ({size_mb:.1f} MB)")
    print(f"   ✓ Contains {len(texts):,} chunks - can resume from here!")


def load_metadata_checkpoint():
    """Load metadata checkpoint if it exists"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    
    print(f"\n📂 Found checkpoint: {CHECKPOINT_FILE}")
    print("   Loading saved metadata (skipping rebuild)...")
    
    try:
        with open(CHECKPOINT_FILE, 'rb') as f:
            checkpoint = pickle.load(f)
        
        print(f"   ✓ Loaded {len(checkpoint['texts']):,} chunks from checkpoint")
        return checkpoint
    except Exception as e:
        print(f"   ⚠️ Failed to load checkpoint: {e}")
        print("   Will rebuild metadata from scratch...")
        return None


# ============================================================================
# OPTIMIZED EMBEDDING CREATION (Memory-Safe with Checkpointing)
# ============================================================================

def create_embeddings_gpu(embedder, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Create embeddings with GPU optimization - WITH CHECKPOINT RESUME"""
    print(f"\n🧠 Creating embeddings...")
    print(f"   Total texts: {len(texts):,}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {DEVICE}")
    print(f"   FP16: {USE_FP16}")
    
    # Always use streaming/batched approach to avoid memory issues
    # BGE-M3 needs small batches to avoid OOM on 16GB VRAM
    effective_batch_size = min(batch_size, 8)
    num_batches = (len(texts) + effective_batch_size - 1) // effective_batch_size
    
    # ========== CHECK FOR EMBEDDING CHECKPOINT ==========
    start_batch = 0
    embeddings = np.zeros((len(texts), EMB_DIM), dtype=np.float32)
    
    if os.path.exists(EMBEDDING_CHECKPOINT_FILE):
        try:
            print(f"   📂 Found embedding checkpoint!")
            checkpoint = np.load(EMBEDDING_CHECKPOINT_FILE)
            saved_embeddings = checkpoint['embeddings']
            start_batch = int(checkpoint['last_batch'])
            
            if saved_embeddings.shape[0] == len(texts):
                # Copy saved embeddings
                embeddings[:] = saved_embeddings
                start_idx = start_batch * effective_batch_size
                progress_pct = (start_batch / num_batches) * 100
                print(f"   ✓ Resuming from batch {start_batch}/{num_batches} ({progress_pct:.1f}% complete)")
                print(f"   ✓ Skipping {start_idx:,} already processed texts")
            else:
                print(f"   ⚠️ Checkpoint size mismatch, starting fresh")
                start_batch = 0
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint: {e}")
            start_batch = 0
    
    # Force garbage collection before starting
    gc.collect()
    if DEVICE == "cuda":
        try:
            torch.cuda.empty_cache()
        except:
            pass
    
    start_time = time.time()
    print(f"   Using memory-safe streaming mode (effective batch: {effective_batch_size})...")
    print(f"   💾 Saving progress every {EMBEDDING_SAVE_INTERVAL} batches")
    
    # Create progress bar starting from resume point
    start_idx = start_batch * effective_batch_size
    pbar = tqdm(range(start_idx, len(texts), effective_batch_size), 
                desc="Batches", unit="batch", 
                initial=start_batch, total=num_batches)
    
    batch_count = start_batch
    
    for i in pbar:
        batch_end = min(i + effective_batch_size, len(texts))
        batch = texts[i:batch_end]
        
        try:
            with torch.cuda.amp.autocast() if USE_FP16 and DEVICE == "cuda" else nullcontext():
                embs = embedder.encode(
                    batch,
                    batch_size=effective_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            # Store directly in pre-allocated array
            embeddings[i:batch_end] = embs.astype(np.float32)
            del embs
            
        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"\n⚠️  GPU OOM at batch {batch_count}, saving checkpoint and falling back to CPU...")
                
                # SAVE CHECKPOINT BEFORE ATTEMPTING RECOVERY
                try:
                    np.savez(EMBEDDING_CHECKPOINT_FILE, embeddings=embeddings, last_batch=batch_count)
                    print(f"   💾 Checkpoint saved at batch {batch_count}")
                except Exception as save_e:
                    print(f"   ⚠️ Failed to save checkpoint: {save_e}")
                
                # Clear GPU memory aggressively
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
                
                # Wait a moment for memory to clear
                time.sleep(2)
                
                # Try again on CPU with smaller batch
                try:
                    embs = embedder.encode(
                        batch,
                        batch_size=8,  # Very small batch for CPU
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        device='cpu'
                    )
                    embeddings[i:batch_end] = embs.astype(np.float32)
                    del embs
                    print(f"   ✓ CPU fallback successful")
                except Exception as cpu_e:
                    print(f"\n❌ CPU fallback also failed: {cpu_e}")
                    print(f"   💾 Progress saved! Run script again to resume from batch {batch_count}")
                    raise
            else:
                # Save checkpoint before raising other errors
                try:
                    np.savez(EMBEDDING_CHECKPOINT_FILE, embeddings=embeddings, last_batch=batch_count)
                    print(f"\n💾 Checkpoint saved at batch {batch_count} before error")
                except:
                    pass
                raise e
        
        batch_count += 1
        
        # Save checkpoint periodically
        if batch_count % EMBEDDING_SAVE_INTERVAL == 0:
            try:
                np.savez(EMBEDDING_CHECKPOINT_FILE, embeddings=embeddings, last_batch=batch_count)
                pbar.set_postfix_str(f"checkpoint saved")
            except Exception as e:
                pbar.set_postfix_str(f"checkpoint failed")
        
        # Free memory every 5 batches
        if batch_count % 5 == 0:
            gc.collect()
            if DEVICE == "cuda":
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
    
    pbar.close()
    
    # Remove checkpoint file after successful completion
    if os.path.exists(EMBEDDING_CHECKPOINT_FILE):
        try:
            os.remove(EMBEDDING_CHECKPOINT_FILE)
            print(f"   ✓ Removed checkpoint file (completed successfully)")
        except:
            pass
    
    elapsed = time.time() - start_time
    if len(texts) - start_idx > 0:
        texts_per_sec = (len(texts) - start_idx) / elapsed
        print(f"   ✓ Completed in {elapsed:.1f}s ({texts_per_sec:.0f} texts/sec)")
    else:
        print(f"   ✓ All embeddings loaded from checkpoint!")
    
    return embeddings

# Context manager for FP16
from contextlib import nullcontext

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main():
    """Main processing pipeline"""
    
    print_header()
    device = detect_hardware()
    
    # Load model
    print(f"\n📦 Loading embedding model: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME, device=device)
    
    if USE_FP16 and device == "cuda":
        embedder.half()
        print("   ✓ FP16 mode enabled")
    
    # ========== CHECK FOR CHECKPOINT ==========
    checkpoint = load_metadata_checkpoint()
    
    if checkpoint:
        # Load from checkpoint - skip metadata building!
        texts = checkpoint['texts']
        metadatas = checkpoint['metadatas']
        bm25_corpus = checkpoint['bm25_corpus']
        keyword_index = checkpoint['keyword_index']
        keyword_index_lower = checkpoint['keyword_index_lower']
        phrase_index = checkpoint['phrase_index']
        section_index = checkpoint['section_index']
        adjacency_map = checkpoint['adjacency_map']
        document_structures = checkpoint['document_structures']
        duplicate_map = checkpoint['duplicate_map']
        similarity_map = checkpoint['similarity_map']
        total_docs = checkpoint['total_docs']
        
        print(f"   ✓ Skipping to embedding step!")
    else:
        # No checkpoint - build from scratch
        # Initialize processor
        processor = DocumentProcessor()
        
        # Find all files
        print(f"\n📂 Scanning {DOCS_DIR}...")
        docs_dir_path = Path(DOCS_DIR)
        
        if not docs_dir_path.exists():
            print(f"❌ Directory not found: {DOCS_DIR}")
            return
        
        all_files = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(list(docs_dir_path.rglob(f"*{ext}")))
        
        print(f"   Found {len(all_files):,} files")
        
        if not all_files:
            print("❌ No files found!")
            return
        
        # Process files in parallel
        processed_docs = process_files_parallel(all_files, docs_dir_path)
        print(f"✅ Successfully read {len(processed_docs):,}/{len(all_files):,} files")
        
        # Create chunks
        print("\n📝 Creating smart chunks...")
        all_chunks = []
        document_structures = {}
        
        for doc in tqdm(processed_docs, desc="Chunking", unit="doc"):
            structure = processor.extract_document_structure(doc['pages'])
            document_structures[doc['path']] = structure
            chunks = processor.smart_chunk_pages(doc['pages'], structure, doc['path'])
            all_chunks.extend(chunks)
        
        print(f"✅ Created {len(all_chunks):,} chunks")
        
        if not all_chunks:
            print("❌ No chunks created!")
            return
        
        # Build metadata and indices
        print("\n🔨 Building metadata and indices...")
        texts = []
        metadatas = []
        keyword_index = defaultdict(list)
        keyword_index_lower = defaultdict(list)
        phrase_index = defaultdict(list)
        section_index = defaultdict(list)
        bm25_corpus = []
        
        duplicate_detector = DuplicateDetector()
        
        for idx, chunk in enumerate(tqdm(all_chunks, desc="Building metadata", unit="chunk")):
            texts.append(chunk['text'])
            
            # Detect duplicates (only check every 10th chunk for speed)
            if idx % 10 == 0:
                duplicate_detector.add_chunk(idx, chunk['text'])
            
            # Tokenize for BM25
            bm25_tokens = tokenize_for_bm25(chunk['text'])
            bm25_corpus.append(bm25_tokens)
            
            # Extract keywords
            keywords_upper, keywords_lower = extract_keywords_advanced(chunk['text'])
            phrases = extract_phrases(chunk['text'])
            
            # Build indices
            for kw in keywords_upper:
                keyword_index[kw].append(idx)
            for kw in keywords_lower:
                keyword_index_lower[kw].append(idx)
            for ph in phrases:
                phrase_index[ph].append(idx)
            if chunk['section']:
                section_index[chunk['section']].append(idx)
            
            # Metadata
            meta = {
                'source': chunk['source'],
                'pages': chunk['pages'],
                'page_start': min(chunk['pages']),
                'page_end': max(chunk['pages']),
                'section': chunk['section'],
                'chunk_id': f"{chunk['source']}_p{min(chunk['pages'])}-{max(chunk['pages'])}_idx{idx}",
                'keywords': keywords_upper[:10],
                'keywords_lower': keywords_lower[:10],
                'phrases': phrases[:5],
                'char_count': len(chunk['text']),
                'word_count': len(chunk['text'].split()),
                'bm25_token_count': len(bm25_tokens)
            }
            metadatas.append(meta)
        
        # Get deduplication info
        duplicate_map = duplicate_detector.get_duplicate_map()
        similarity_map = duplicate_detector.get_similarity_map()
        print(f"   Duplicates: {len(duplicate_map):,}, Similar pairs: {len(similarity_map):,}")
        
        # Build adjacency map (optimized O(n))
        adjacency_map = build_adjacency_map_fast(metadatas)
        print(f"   Adjacency connections: {sum(len(v) for v in adjacency_map.values()):,}")
        
        # Store counts before deleting
        total_docs = len(processed_docs)
        del all_chunks  # No longer needed
        del processed_docs  # No longer needed
        
        # ========== SAVE CHECKPOINT ==========
        save_metadata_checkpoint(texts, metadatas, bm25_corpus, keyword_index, keyword_index_lower,
                                 phrase_index, section_index, adjacency_map, document_structures,
                                 duplicate_map, similarity_map, total_docs)
    
    # ========== CONTINUE WITH EMBEDDINGS (both paths converge here) ==========
    
    # FREE MEMORY before embedding
    print("\n🧹 Freeing memory before embedding...")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Check available memory
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"   Available RAM: {mem.available / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB")
    except:
        pass
    
    # Create embeddings
    embeddings = create_embeddings_gpu(embedder, texts, BATCH_SIZE)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Build FAISS index
    print("\n🔍 Building FAISS index...")
    index = faiss.IndexFlatIP(EMB_DIM)
    
    # Embeddings are already normalized from encode()
    index.add(embeddings)
    print(f"   ✓ FAISS index built with {index.ntotal:,} vectors")
    
    # Build BM25 index
    print("🔍 Building BM25 index...")
    bm25_index = BM25Okapi(bm25_corpus, k1=BM25_K1, b=BM25_B)
    print(f"   ✓ BM25 index: {len(bm25_corpus):,} docs, avg length {bm25_index.avgdl:.1f}")
    
    # Save everything
    print("\n💾 Saving to disk...")
    
    index_path = os.path.join(OUT_DIR, "faiss.index")
    meta_path = os.path.join(OUT_DIR, "metadata.json")
    bm25_path = os.path.join(OUT_DIR, "bm25.pkl")
    bm25_corpus_path = os.path.join(OUT_DIR, "bm25_corpus.pkl")
    
    faiss.write_index(index, index_path)
    
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25_index, f)
    with open(bm25_corpus_path, 'wb') as f:
        pickle.dump(bm25_corpus, f)
    
    save_data = {
        'texts': texts,
        'metadatas': metadatas,
        'keywords_index': dict(keyword_index),
        'keywords_index_lower': dict(keyword_index_lower),
        'phrases_index': dict(phrase_index),
        'section_index': dict(section_index),
        'adjacency_map': adjacency_map,
        'document_structures': document_structures,
        'duplicate_map': duplicate_map,
        'similarity_map': {str(k): v for k, v in similarity_map.items()},
        'config': {
            'chunk_size': CHUNK_SIZE,
            'overlap': CHUNK_OVERLAP,
            'model': MODEL_NAME,
            'total_chunks': len(texts),
            'total_docs': total_docs,
            'bm25_k1': BM25_K1,
            'bm25_b': BM25_B,
            'similarity_threshold': SIMILARITY_THRESHOLD,
            'has_bm25': True,
            'has_duplicate_detection': True,
            'device_used': DEVICE,
            'batch_size': BATCH_SIZE,
            'fp16': USE_FP16
        }
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 INDEXING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"✓ Documents processed: {total_docs:,}")
    print(f"✓ Total chunks: {len(texts):,}")
    print(f"✓ Average chunk size: {np.mean([m['word_count'] for m in metadatas]):.0f} words")
    print(f"✓ Unique sections: {len(section_index):,}")
    print(f"✓ Keywords (uppercase): {len(keyword_index):,}")
    print(f"✓ Keywords (lowercase): {len(keyword_index_lower):,}")
    print(f"✓ Phrases: {len(phrase_index):,}")
    print(f"✓ Adjacency connections: {sum(len(v) for v in adjacency_map.values()):,}")
    print(f"✓ Duplicates found: {len(duplicate_map):,}")
    print("-" * 70)
    print(f"✓ FAISS index: {os.path.getsize(index_path) / 1024 / 1024:.1f} MB")
    print(f"✓ BM25 index: {os.path.getsize(bm25_path) / 1024 / 1024:.1f} MB")
    print(f"✓ Metadata: {os.path.getsize(meta_path) / 1024 / 1024:.1f} MB")
    print("=" * 70)
    print("🎉 OPTIMIZED RAG SYSTEM READY!")
    print("   ✓ GPU-accelerated embeddings")
    print("   ✓ Parallel file processing")
    print("   ✓ Hybrid search (FAISS + BM25)")
    print("=" * 70)


if __name__ == "__main__":
    main()


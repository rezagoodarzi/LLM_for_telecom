"""
RAG System Configuration - All settings in one place
=====================================================

This file contains all configurable parameters for the RAG system.
Modify these values based on your hardware and document types.

Key Optimizations Applied:
- Smaller chunks (2000 chars) for better embedding accuracy
- Higher similarity threshold (0.92) to preserve technical content
- Vendor/document filtering for multi-source documents
- Hierarchical retrieval (document → chunk)
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

# Base directories
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = "/home/fatemebookanian/qwen_code/PDF"
OUTPUT_DIR = "./rag_store_v2"

# Model paths
EMBEDDING_MODEL = "/home/fatemebookanian/models/BGE-m3"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Faster, good for technical

# Checkpoint files
METADATA_CHECKPOINT = os.path.join(OUTPUT_DIR, "metadata_checkpoint.pkl")
EMBEDDING_CHECKPOINT = os.path.join(OUTPUT_DIR, "embedding_checkpoint.npz")

# =============================================================================
# CHUNKING PARAMETERS (CRITICAL FOR RAG QUALITY)
# =============================================================================

# Chunk size in characters
# BGE-M3 optimal: 512-1024 tokens ≈ 2000-4000 characters
# Smaller chunks = better precision, more chunks to retrieve
# Larger chunks = more context per chunk, but less precision
CHUNK_SIZE = 2000              # chars (~500 tokens) - REDUCED from 6000

# Overlap between chunks (helps maintain context across boundaries)
# Rule: 10-20% of chunk size
CHUNK_OVERLAP = 300            # chars - REDUCED from 800

# Maximum chunk size (hard limit)
MAX_CHUNK_SIZE = 3000          # chars - REDUCED from 12000

# Minimum chunk size (skip tiny chunks)
MIN_CHUNK_SIZE = 100           # chars

# Page merge threshold - when to merge short pages
# Set higher to prevent over-merging of unrelated content
PAGE_MERGE_THRESHOLD = 150     # chars - REDUCED from 500

# =============================================================================
# EMBEDDING PARAMETERS
# =============================================================================

EMBEDDING_DIM = 1024           # BGE-M3 output dimension
BATCH_SIZE = 8                 # For GPU embedding (adjust for VRAM)
USE_FP16 = True                # Half precision for speed
EMBEDDING_SAVE_INTERVAL = 25   # Save progress every N batches

# =============================================================================
# BM25 PARAMETERS
# =============================================================================

BM25_K1 = 1.5                  # Term frequency saturation
BM25_B = 0.75                  # Length normalization (adjusted for smaller chunks)

# =============================================================================
# DEDUPLICATION PARAMETERS
# =============================================================================

# Similarity threshold for near-duplicate detection
# Higher = more conservative (keeps more content)
# 0.85 was too aggressive for technical docs with similar structure
SIMILARITY_THRESHOLD = 0.92    # INCREASED from 0.85

# Min-hash shingles for similarity computation
MIN_HASH_SHINGLES = 5

# =============================================================================
# RETRIEVAL PARAMETERS
# =============================================================================

# Initial retrieval count (before reranking)
TOP_K = 20                     # INCREASED from 12 (more candidates with smaller chunks)

# After reranking
RERANK_TOP_K = 12              # INCREASED from 8

# BM25 retrieval count
BM25_TOP_K = 40                # INCREASED from 30

# Context for LLM (number of chunks to include)
CONTEXT_CHUNKS = 8             # Final chunks sent to LLM

# =============================================================================
# HYBRID SEARCH WEIGHTS
# =============================================================================

# Weights for combining search methods
# Increased keyword weight for technical IDs (f1, AMF, PDCP, etc.)
HYBRID_WEIGHTS = {
    'semantic': 0.40,          # Meaning-based (REDUCED from 0.50)
    'bm25': 0.30,              # Exact term matching
    'keyword': 0.30,           # Technical keywords (INCREASED from 0.15)
}

# Exact match boost for quoted terms
EXACT_MATCH_BOOST = 3.0

# =============================================================================
# SMART RETRIEVAL FEATURES
# =============================================================================

# Enable adjacent chunk retrieval
ENABLE_ADJACENT_RETRIEVAL = True
ADJACENT_CHUNKS = 2

# Enable section expansion
ENABLE_SECTION_EXPANSION = True

# Enable deduplication in results
ENABLE_DEDUPLICATION = True
SIMILARITY_PENALTY = 0.5

# Enable case-insensitive keyword search
ENABLE_CASE_INSENSITIVE = True

# =============================================================================
# PRE-INDEXING CLEANING
# =============================================================================

# Version selection - keep only newest version per document family
ENABLE_VERSION_SELECTION = True

# Document-level deduplication (MinHash)
ENABLE_DOCUMENT_DEDUP = True
DOCUMENT_DEDUP_THRESHOLD = 0.80  # Similarity threshold for duplicates

# Noise page filtering (TOC, legal, revision history)
ENABLE_NOISE_FILTERING = True

# =============================================================================
# DOCUMENT FILTERING (RETRIEVAL)
# =============================================================================

# Enable vendor-based filtering
ENABLE_VENDOR_FILTERING = True

# Known vendors (extracted from filenames/content)
KNOWN_VENDORS = [
    'ericsson', 'nokia', 'huawei', 'samsung', 'zte',
    '3gpp', 'etsi', 'itu', 'gsma'
]

# Enable document-level retrieval (2-stage)
ENABLE_DOCUMENT_RETRIEVAL = True
DOCUMENT_TOP_K = 5             # Top documents before chunk retrieval

# Enable version filtering
ENABLE_VERSION_FILTERING = True

# =============================================================================
# LLM API CONFIGURATION
# =============================================================================

LLM_API_URL = "http://localhost:5000/v1/chat/completions"
LLM_MODEL_NAME = "qwen3-4b-bnb4"

LLM_PARAMS = {
    "temperature": 0.1,        # Low for factual answers
    "top_p": 0.95,
    "max_tokens": 2048,
    "top_k": 20,
    "repetition_penalty": 1.1,
}

# =============================================================================
# HARDWARE OPTIMIZATION
# =============================================================================

NUM_WORKERS = 10               # Parallel file processing
NUM_IO_THREADS = 8             # I/O operations

# =============================================================================
# SUPPORTED FILE TYPES
# =============================================================================

SUPPORTED_EXTENSIONS = ['.pdf', '.html', '.htm', '.md', '.txt']

# =============================================================================
# SECTION DETECTION PATTERNS (for hierarchical chunking)
# =============================================================================

SECTION_PATTERNS = [
    (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'numbered'),           # 1.2.3 Title
    (r'^([A-Z](?:\.\d+)*)\s+(.+)$', 'letter'),           # A.1 Title
    (r'^((?:I|II|III|IV|V|VI|VII|VIII|IX|X)+)\s+', 'roman'),
    (r'^(Chapter|Section|Part|Annex)\s+', 'keyword'),
    (r'^([A-Z][A-Z\s]{10,})$', 'caps'),                  # ALL CAPS HEADER
    (r'^(#{1,6})\s+(.+)$', 'markdown'),                  # # Markdown
]

# =============================================================================
# TELECOM-SPECIFIC PATTERNS (for better keyword extraction)
# =============================================================================

TELECOM_PATTERNS = [
    r'\b[A-Z]{2,10}\d{2,8}\b',           # TS36.331, NR17
    r'\b\d{2,8}[A-Z]{2,10}\b',           # 3GPP
    r'\b[A-Z]+-[A-Z]+-\d+\b',            # MAC-I, AMF-ID
    r'\bTS\s*\d+\.\d+\b',                # TS 36.331
    r'\bTR\s*\d+\.\d+\b',                # TR 38.913
    r'\b5G\s*NR\b',
    r'\bLTE\b', r'\bNR\b', r'\bRRC\b', r'\bPDCP\b', r'\bRLC\b',
    r'\bMAC\b', r'\bPHY\b', r'\bSDAP\b', r'\bNAS\b',
    r'\bAMF\b', r'\bSMF\b', r'\bUPF\b', r'\bNRF\b',
    r'\bPUCCH\b', r'\bPUSCH\b', r'\bPDSCH\b', r'\bPDCCH\b',
    r'\bMIMO\b', r'\bMU-MIMO\b', r'\bSU-MIMO\b',
    r'\bQoS\b', r'\bQCI\b', r'\b5QI\b',
    r'\bSN\d+\b',                         # SN0012
    r'\bf[1-5]\b',                        # f1, f2, f3 (Milenage)
    r'\bK_?[A-Z]+\b',                     # KASME, KNASint
    r'\bOPc?\b', r'\bSQN\b', r'\bAK\b',  # Security
]

# Technical phrases for phrase extraction
TECHNICAL_PHRASES = [
    "gap length", "coverage area", "maximum distance",
    "frequency band", "carrier frequency", "bandwidth",
    "remote radio head", "base station", "antenna system",
    "reference signal", "channel estimation", "beamforming",
    "handover", "mobility management", "paging",
    "security mode", "integrity protection", "ciphering",
    "key derivation", "authentication", "registration",
]

# =============================================================================
# LOGGING
# =============================================================================

VERBOSE = True
LOG_RETRIEVAL_DETAILS = True

# =============================================================================
# Ensure output directory exists
# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


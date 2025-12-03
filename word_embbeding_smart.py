# word_embbeding_smart.py - Advanced RAG Indexing System with Hybrid Search
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
from typing import List, Dict, Set, Tuple
import string

# -------- CONFIG --------
DOCS_DIR = "/home/fatemebookanian/qwen_code/PDF"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
OUT_DIR = "./rag_store"

# Smart Chunking Parameters
CHUNK_SIZE = 5000        # Target chunk size
CHUNK_OVERLAP = 600      # Overlap between chunks
PAGE_MERGE_THRESHOLD = 500  # Min chars to trigger page merge
MAX_CHUNK_SIZE = 8000    # Max size even with merging

# Model Configuration
EMB_DIM = 768
BATCH_SIZE = 64

# File Types
SUPPORTED_EXTENSIONS = ['.pdf', '.html', '.htm']

# BM25 Configuration
BM25_K1 = 1.5           # Term frequency saturation parameter
BM25_B = 0.75           # Length normalization parameter

# Deduplication Configuration
SIMILARITY_THRESHOLD = 0.85  # Threshold for considering chunks as duplicates
MIN_HASH_SHINGLES = 5        # Number of shingles for MinHash

# Section Detection Patterns
SECTION_PATTERNS = [
    # Numbered sections (1.2.3, 7.3.6, etc.)
    (r'^(\d+(?:\.\d+)*)\s+(.+)$', 'numbered'),
    # Letter sections (A. B. C. or A.1.2)
    (r'^([A-Z](?:\.\d+)*)\s+(.+)$', 'letter'),
    # Roman numerals
    (r'^((?:I|II|III|IV|V|VI|VII|VIII|IX|X)+(?:\.\d+)*)\s+(.+)$', 'roman'),
    # Chapter/Section keywords
    (r'^(Chapter|Section|Part|Module)\s+(\d+|\w+):\s*(.+)$', 'keyword'),
    # All caps headers (min 3 words)
    (r'^([A-Z][A-Z\s]{10,})$', 'caps'),
    # Markdown-style headers
    (r'^(#{1,6})\s+(.+)$', 'markdown'),
]
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)
print("=" * 70)
print("🚀 SMART RAG INDEXING SYSTEM v2.0")
print("=" * 70)

print(f"Loading embedding model: {MODEL_NAME}")
embedder = SentenceTransformer(MODEL_NAME)

class DocumentProcessor:
    """Advanced document processing with smart chunking and section tracking"""
    
    def __init__(self):
        self.current_section_stack = []
        self.section_hierarchy = defaultdict(list)
    
    def detect_section(self, text):
        """Detect if text is a section header and return its level"""
        for pattern, section_type in SECTION_PATTERNS:
            match = re.match(pattern, text.strip())
            if match:
                return {
                    'type': section_type,
                    'text': text.strip(),
                    'match': match.groups()
                }
        return None
    
    def extract_document_structure(self, pages):
        """Build document structure with sections and hierarchy"""
        structure = {
            'sections': [],
            'page_sections': {},  # Map page to current section
            'section_pages': defaultdict(list)  # Map section to pages
        }
        
        current_section = "Document Start"
        section_level = 0
        
        for page_num, page_text in enumerate(pages):
            lines = page_text.split('\n')
            for line in lines[:20]:  # Check first 20 lines for headers
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
    
    def smart_chunk_pages(self, pages, structure, source_file):
        """Create smart chunks that respect section boundaries"""
        chunks = []
        current_chunk = ""
        current_pages = []
        current_section = None
        
        for page_num, page_text in enumerate(pages):
            page_id = page_num + 1
            section = structure['page_sections'].get(page_id, "Unknown")
            
            # Check if we're starting a new section
            if section != current_section and current_chunk:
                # Save current chunk before starting new section
                if len(current_chunk) > 100:  # Minimum chunk size
                    chunks.append({
                        'text': current_chunk,
                        'pages': current_pages.copy(),
                        'section': current_section,
                        'source': source_file
                    })
                current_chunk = ""
                current_pages = []
            
            current_section = section
            
            # Add page to current chunk
            if len(current_chunk) + len(page_text) <= MAX_CHUNK_SIZE:
                current_chunk += f"\n[Page {page_id}]\n{page_text}"
                current_pages.append(page_id)
            else:
                # Need to split within the page
                if current_chunk:
                    chunks.append({
                        'text': current_chunk,
                        'pages': current_pages.copy(),
                        'section': current_section,
                        'source': source_file
                    })
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
                current_chunk = overlap_text + f"\n[Page {page_id}]\n{page_text}"
                current_pages = [page_id]
        
        # Don't forget the last chunk
        if current_chunk and len(current_chunk) > 100:
            chunks.append({
                'text': current_chunk,
                'pages': current_pages.copy(),
                'section': current_section,
                'source': source_file
            })
        
        return chunks

def extract_text_from_pdf(path):
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
        print(f"Error reading PDF {path}: {e}")
        return []

def extract_text_from_html(path):
    """Extract text from HTML files"""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return [text] if text.strip() else []
    except Exception as e:
        print(f"Error reading HTML {path}: {e}")
        return []

def extract_keywords_advanced(text):
    """Extract keywords, IDs, and important terms - CASE INSENSITIVE"""
    keywords = set()
    keywords_lower = set()  # For case-insensitive matching
    
    # Normalize text for pattern matching
    text_upper = text.upper()
    text_lower = text.lower()
    
    # Technical IDs and codes - COMPREHENSIVE PATTERNS
    patterns = [
        # Alphanumeric IDs (like SN0012, SR001225, RAN1795, PUCCH)
        r'\b[A-Z]{2,10}\d{2,8}\b',     # XX00 to XXXXXXXXXX00000000
        r'\b\d{2,8}[A-Z]{2,10}\b',     # 00XX format
        r'\b[A-Z]{2,10}\d+[A-Z]+\b',   # XX00XX format
        r'\b\d+[A-Z]+\d+\b',           # 00XX00 format
        
        # Version and reference numbers
        r'\b\d+\.\d+(?:\.\d+)*\b',     # Version numbers (1.2.3)
        r'\bV\d+\.\d+\b',              # V1.2 format
        r'\bREV\s*\d+\b',              # REV 1 format
        
        # Hyphenated codes
        r'\b[A-Z]{2,10}-\d{2,10}\b',   # ABC-123 format
        r'\b\d{2,10}-[A-Z]{2,10}\b',   # 123-ABC format
        r'\b[A-Z]+-[A-Z]+-\d+\b',      # ABC-XYZ-123 format
        
        # Dotted formats
        r'\b[A-Z]{2,}\.\d+(?:\.\d+)*\b',  # XX.123.456 format
        
        # Pure technical codes (all caps, 3+ chars)
        r'\b[A-Z]{3,15}\b',            # PUCCH, PDSCH, NR, LTE etc.
        
        # Numbers with units
        r'\b\d+(?:\.\d+)?\s*(?:MHz|GHz|kHz|dB|dBm|ms|us|ns|Mbps|Gbps|KB|MB|GB)\b',
        
        # Specific telecom patterns
        r'\b3GPP\b',
        r'\bTS\s*\d+\.\d+\b',          # TS 38.211 format
        r'\bTR\s*\d+\.\d+\b',          # TR 38.901 format
        r'\b5G\s*NR\b',
        r'\bLTE\b',
        r'\bNR\b',
    ]
    
    for pattern in patterns:
        # Match in original text (preserve case for display)
        matches = re.findall(pattern, text_upper, re.IGNORECASE)
        for m in matches:
            keywords.add(m.upper())  # Store uppercase for indexing
            keywords_lower.add(m.lower())  # Store lowercase for matching
    
    # Important capitalized terms (min 3 chars) - acronyms
    cap_terms = re.findall(r'\b[A-Z]{3,}\b', text)
    for t in cap_terms:
        if len(t) >= 3:
            keywords.add(t.upper())
            keywords_lower.add(t.lower())
    
    # Extract quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    for q in quoted:
        if len(q) > 2:
            keywords.add(q.upper())
            keywords_lower.add(q.lower())
    
    # Also extract lowercase versions of potential IDs
    lower_patterns = [
        r'\b[a-z]{2,10}\d{2,8}\b',     # sn0012 format
        r'\b\d{2,8}[a-z]{2,10}\b',     # 0012sn format
    ]
    
    for pattern in lower_patterns:
        matches = re.findall(pattern, text_lower)
        for m in matches:
            keywords.add(m.upper())
            keywords_lower.add(m.lower())
    
    return list(keywords), list(keywords_lower)

def extract_phrases(text):
    """Extract important multi-word phrases"""
    phrases = []
    
    # Common technical phrases
    technical_phrases = [
        # Distance and Coverage
        "gap length", "site to site", "coverage area", "maximum distance", 
        "minimum distance", "transmission distance", "coverage radius",
        "cell radius", "propagation distance", "link distance", "hop distance",
        
        # Safety and Compliance
        "basic safety", "safety aspects", "safety requirements", "safety precautions",
        "safety measures", "safety instructions", "hazard warning", "risk assessment",
        "personal protective equipment", "ppe requirements", "safety compliance",
        "electromagnetic exposure", "radiation safety", "electrical safety",
        
        # Installation and Setup
        "installation guide", "installation procedure", "installation requirements",
        "installation manual", "mounting instructions", "cable installation",
        "grounding requirements", "site preparation", "commissioning procedure",
        "pre-installation", "post-installation", "installation checklist",
        
        # Technical Specifications
        "technical specifications", "system requirements", "minimum requirements",
        "hardware requirements", "software requirements", "performance specifications",
        "electrical specifications", "mechanical specifications", "environmental specifications",
        "operating conditions", "storage conditions", "power consumption",
        "power requirements", "voltage requirements", "current consumption",
        
        # Network and RF Terms
        "frequency band", "frequency range", "carrier frequency", "bandwidth requirements",
        "channel bandwidth", "modulation scheme", "antenna gain", "transmission power",
        "output power", "input power", "power amplifier", "signal strength",
        "signal quality", "signal to noise", "bit error rate", "packet loss",
        "latency requirements", "throughput requirements", "qos parameters",
        
        # Equipment and Hardware
        "remote radio head", "base station", "antenna system", "feeder cable",
        "jumper cable", "connector type", "cable type", "fiber optic",
        "power supply unit", "backup battery", "cooling system", "fan unit",
        "control unit", "processing unit", "interface module", "transceiver module",
        
        # Configuration and Parameters
        "configuration parameters", "default settings", "factory settings",
        "network configuration", "ip configuration", "vlan configuration",
        "port configuration", "parameter settings", "optimization parameters",
        "calibration procedure", "alignment procedure", "test parameters",
        
        # Maintenance and Troubleshooting
        "maintenance procedure", "preventive maintenance", "corrective maintenance",
        "troubleshooting guide", "fault diagnosis", "error codes", "alarm messages",
        "status indicators", "led indicators", "diagnostic tools", "test equipment",
        "replacement procedure", "spare parts", "service manual",
        
        # Environmental and Physical
        "operating temperature", "storage temperature", "humidity range",
        "altitude requirements", "wind load", "ice load", "seismic requirements",
        "ingress protection", "ip rating", "corrosion resistance", "weather protection",
        "thermal management", "heat dissipation", "ventilation requirements",
        
        # Regulatory and Standards
        "regulatory compliance", "certification requirements", "type approval",
        "fcc compliance", "ce marking", "rohs compliance", "safety standards",
        "emission limits", "immunity requirements", "emc requirements",
        "quality standards", "iso standards", "industry standards",
        
        # Performance Metrics
        "key performance indicators", "kpi metrics", "availability target",
        "reliability metrics", "mean time between failures", "mtbf", "mttr",
        "service level agreement", "sla requirements", "uptime requirements",
        "capacity planning", "traffic capacity", "user capacity",
        
        # Documentation References
        "reference manual", "user manual", "operation manual", "service manual",
        "quick start guide", "release notes", "technical bulletin", "application note",
        "white paper", "best practices", "design guide", "deployment guide",
        
        # Common Abbreviations as Phrases
        "rrh installation", "bts configuration", "rf optimization", "vswr measurement",
        "rssi level", "sinr ratio", "ber measurement", "evm measurement",
        "pim testing", "otdr measurement", "link budget", "path loss"
    ]
    
    text_lower = text.lower()
    for phrase in technical_phrases:
        if phrase in text_lower:
            phrases.append(phrase)
    
    # Extract section titles (lines ending with :)
    section_titles = re.findall(r'^([A-Za-z][^:]{5,50}):$', text, re.MULTILINE)
    phrases.extend([t.lower().strip() for t in section_titles])
    
    return phrases


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text for BM25 indexing - preserves technical terms"""
    # Convert to lowercase for consistent matching
    text_lower = text.lower()
    
    # Replace punctuation with spaces (except hyphens and dots in technical terms)
    # First protect technical patterns
    protected = []
    
    # Protect version numbers and technical IDs
    tech_patterns = [
        r'\b[a-z]{2,10}\d{2,8}\b',     # sn0012
        r'\b\d{2,8}[a-z]{2,10}\b',     # 0012sn
        r'\b\d+\.\d+(?:\.\d+)*\b',     # 1.2.3
        r'\b[a-z]+-[a-z0-9]+\b',       # abc-123
        r'\b[a-z]+_[a-z0-9]+\b',       # abc_123
    ]
    
    for pattern in tech_patterns:
        matches = re.findall(pattern, text_lower)
        protected.extend(matches)
    
    # Standard tokenization
    # Remove most punctuation but keep alphanumeric and spaces
    cleaned = re.sub(r'[^\w\s\-\.]', ' ', text_lower)
    
    # Split on whitespace
    tokens = cleaned.split()
    
    # Add protected terms that might have been split
    tokens.extend(protected)
    
    # Remove very short tokens (except known technical abbreviations)
    tokens = [t for t in tokens if len(t) >= 2 or t.upper() in ['5G', 'NR', 'LTE', 'RF', 'IP', 'TX', 'RX']]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    
    return unique_tokens


def compute_text_hash(text: str) -> str:
    """Compute hash for deduplication"""
    # Normalize text
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def compute_shingle_hash(text: str, k: int = 5) -> Set[int]:
    """Compute k-shingle hashes for similarity detection"""
    # Normalize text
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


class DuplicateDetector:
    """Detect and track duplicate/similar chunks"""
    
    def __init__(self, similarity_threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = similarity_threshold
        self.chunk_hashes: Dict[int, str] = {}
        self.chunk_shingles: Dict[int, Set[int]] = {}
        self.duplicate_groups: Dict[str, List[int]] = defaultdict(list)
        self.similar_pairs: List[Tuple[int, int, float]] = []
    
    def add_chunk(self, idx: int, text: str):
        """Add a chunk and detect duplicates"""
        text_hash = compute_text_hash(text)
        shingles = compute_shingle_hash(text)
        
        self.chunk_hashes[idx] = text_hash
        self.chunk_shingles[idx] = shingles
        
        # Check for exact duplicates
        self.duplicate_groups[text_hash].append(idx)
        
        # Check for similar chunks (only against recent chunks for efficiency)
        recent_chunks = list(self.chunk_shingles.items())[-100:]
        for other_idx, other_shingles in recent_chunks:
            if other_idx != idx:
                similarity = compute_jaccard_similarity(shingles, other_shingles)
                if similarity >= self.threshold:
                    self.similar_pairs.append((idx, other_idx, similarity))
    
    def get_duplicate_map(self) -> Dict[int, int]:
        """Get mapping of duplicate chunks to their primary chunk"""
        duplicate_map = {}
        for group in self.duplicate_groups.values():
            if len(group) > 1:
                primary = group[0]
                for dup in group[1:]:
                    duplicate_map[dup] = primary
        return duplicate_map
    
    def get_similarity_map(self) -> Dict[int, List[Tuple[int, float]]]:
        """Get mapping of chunks to their similar chunks"""
        similarity_map = defaultdict(list)
        for idx1, idx2, sim in self.similar_pairs:
            similarity_map[idx1].append((idx2, sim))
            similarity_map[idx2].append((idx1, sim))
        return dict(similarity_map)

# Initialize processor
processor = DocumentProcessor()

# Process all documents
print(f"\n📂 Scanning {DOCS_DIR} for documents...")
docs_dir_path = Path(DOCS_DIR)
all_files = []
for ext in SUPPORTED_EXTENSIONS:
    all_files.extend(list(docs_dir_path.rglob(f"*{ext}")))

print(f"📊 Found {len(all_files)} files to process\n")

# Process documents and create smart chunks
all_chunks = []
document_structures = {}
processed_count = 0

for doc_file in tqdm(all_files, desc="Processing documents"):
    try:
        relative_path = doc_file.relative_to(docs_dir_path)
        
        # Extract pages based on file type
        if doc_file.suffix.lower() == '.pdf':
            pages = extract_text_from_pdf(str(doc_file))
            file_type = "PDF"
        elif doc_file.suffix.lower() in ['.html', '.htm']:
            pages = extract_text_from_html(str(doc_file))
            file_type = "HTML"
        else:
            continue
        
        if not pages:
            continue
        
        # Build document structure
        structure = processor.extract_document_structure(pages)
        document_structures[str(relative_path)] = structure
        
        # Create smart chunks
        chunks = processor.smart_chunk_pages(pages, structure, str(relative_path))
        all_chunks.extend(chunks)
        
        processed_count += 1
        
    except Exception as e:
        print(f"\n⚠️  Error processing {doc_file.name}: {e}")

print(f"\n✅ Successfully processed {processed_count}/{len(all_files)} documents")
print(f"📦 Created {len(all_chunks)} smart chunks\n")

if len(all_chunks) == 0:
    print("❌ No chunks created! Check your documents.")
    exit(1)

# Build metadata with enhanced information
print("🔨 Building enhanced metadata with hybrid search indices...")
texts = []
metadatas = []
keyword_index = defaultdict(list)           # UPPERCASE keywords -> chunk indices
keyword_index_lower = defaultdict(list)     # lowercase keywords -> chunk indices (for case-insensitive)
phrase_index = defaultdict(list)
section_index = defaultdict(list)
adjacency_map = defaultdict(list)  # Maps chunk to adjacent chunks

# Initialize BM25 tokenized corpus
bm25_corpus = []

# Initialize duplicate detector
duplicate_detector = DuplicateDetector(similarity_threshold=SIMILARITY_THRESHOLD)

print("📝 Processing chunks and building indices...")
for idx, chunk in enumerate(tqdm(all_chunks, desc="Building metadata")):
    texts.append(chunk['text'])
    
    # Detect duplicates/similar chunks
    duplicate_detector.add_chunk(idx, chunk['text'])
    
    # Tokenize for BM25
    bm25_tokens = tokenize_for_bm25(chunk['text'])
    bm25_corpus.append(bm25_tokens)
    
    # Extract keywords and phrases (now returns two lists)
    keywords_upper, keywords_lower = extract_keywords_advanced(chunk['text'])
    phrases = extract_phrases(chunk['text'])
    
    # Build BOTH uppercase and lowercase keyword indices for case-insensitive matching
    for kw in keywords_upper:
        keyword_index[kw].append(idx)
    for kw in keywords_lower:
        keyword_index_lower[kw].append(idx)
    
    # Also add the original text tokens to keyword index for exact matching
    text_tokens = set(chunk['text'].lower().split())
    for token in text_tokens:
        # Clean token
        clean_token = re.sub(r'[^\w]', '', token)
        if len(clean_token) >= 2:
            keyword_index_lower[clean_token].append(idx)
    
    for ph in phrases:
        phrase_index[ph].append(idx)
    if chunk['section']:
        section_index[chunk['section']].append(idx)
    
    # Create rich metadata
    meta = {
        'source': chunk['source'],
        'pages': chunk['pages'],
        'page_start': min(chunk['pages']),
        'page_end': max(chunk['pages']),
        'section': chunk['section'],
        'chunk_id': f"{chunk['source']}_p{min(chunk['pages'])}-{max(chunk['pages'])}_idx{idx}",
        'keywords': keywords_upper[:10],  # Top 10 keywords
        'keywords_lower': keywords_lower[:10],  # Lowercase versions
        'phrases': phrases[:5],      # Top 5 phrases
        'char_count': len(chunk['text']),
        'word_count': len(chunk['text'].split()),
        'bm25_token_count': len(bm25_tokens)
    }
    metadatas.append(meta)

# Get duplicate and similarity information
duplicate_map = duplicate_detector.get_duplicate_map()
similarity_map = duplicate_detector.get_similarity_map()
print(f"📊 Found {len(duplicate_map)} exact duplicates and {len(similarity_map)} similar chunk pairs")

# Build adjacency map for smart retrieval
print("🔗 Building adjacency map...")
for idx, meta in enumerate(metadatas):
    source = meta['source']
    pages = meta['pages']
    
    # Find adjacent chunks (same document, adjacent pages)
    for other_idx, other_meta in enumerate(metadatas):
        if idx == other_idx:
            continue
        if other_meta['source'] != source:
            continue
        
        # Check if pages are adjacent
        other_pages = other_meta['pages']
        if (max(pages) + 1 == min(other_pages) or  # Next pages
            min(pages) - 1 == max(other_pages)):     # Previous pages
            adjacency_map[idx].append(other_idx)

print(f"📊 Built adjacency map with {sum(len(v) for v in adjacency_map.values())} connections\n")

# Create embeddings
print(f"🧠 Creating embeddings with {MODEL_NAME}...")
embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding batches"):
    batch = texts[i:i+BATCH_SIZE]
    embs = embedder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
    embeddings.append(embs)

embeddings = np.vstack(embeddings).astype("float32")
print(f"✅ Created embeddings with shape: {embeddings.shape}\n")

# Build FAISS index
print("🔍 Building FAISS index...")
index = faiss.IndexFlatIP(EMB_DIM)  # Inner product for cosine similarity
faiss.normalize_L2(embeddings)       # Normalize for cosine similarity
index.add(embeddings)

# Build BM25 index for hybrid search
print("🔍 Building BM25 index for exact term matching...")
bm25_index = BM25Okapi(bm25_corpus, k1=BM25_K1, b=BM25_B)
print(f"✅ BM25 index created with {len(bm25_corpus)} documents")
print(f"   Average document length: {bm25_index.avgdl:.1f} tokens")

# Save everything
print("💾 Saving to disk...")
index_path = os.path.join(OUT_DIR, "faiss.index")
meta_path = os.path.join(OUT_DIR, "metadata.json")
bm25_path = os.path.join(OUT_DIR, "bm25.pkl")
bm25_corpus_path = os.path.join(OUT_DIR, "bm25_corpus.pkl")

faiss.write_index(index, index_path)

# Save BM25 index and corpus
with open(bm25_path, 'wb') as f:
    pickle.dump(bm25_index, f)
with open(bm25_corpus_path, 'wb') as f:
    pickle.dump(bm25_corpus, f)

# Save all metadata and indices
save_data = {
    'texts': texts,
    'metadatas': metadatas,
    'keywords_index': dict(keyword_index),              # UPPERCASE keywords
    'keywords_index_lower': dict(keyword_index_lower),  # lowercase keywords (NEW)
    'phrases_index': dict(phrase_index),
    'section_index': dict(section_index),
    'adjacency_map': dict(adjacency_map),
    'document_structures': document_structures,
    'duplicate_map': duplicate_map,                     # Exact duplicates (NEW)
    'similarity_map': {str(k): v for k, v in similarity_map.items()},  # Similar chunks (NEW)
    'config': {
        'chunk_size': CHUNK_SIZE,
        'overlap': CHUNK_OVERLAP,
        'model': MODEL_NAME,
        'total_chunks': len(texts),
        'total_docs': processed_count,
        'bm25_k1': BM25_K1,
        'bm25_b': BM25_B,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'has_bm25': True,
        'has_duplicate_detection': True
    }
}

with open(meta_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False)

print(f"✅ Saved FAISS index to {index_path}")
print(f"✅ Saved BM25 index to {bm25_path}")
print(f"✅ Saved metadata to {meta_path}")

# Print summary statistics
print("\n" + "=" * 70)
print("📊 INDEXING COMPLETE - SUMMARY")
print("=" * 70)
print(f"✓ Documents processed: {processed_count}")
print(f"✓ Total chunks created: {len(texts)}")
print(f"✓ Average chunk size: {np.mean([m['word_count'] for m in metadatas]):.0f} words")
print(f"✓ Unique sections detected: {len(section_index)}")
print(f"✓ Keywords indexed (uppercase): {len(keyword_index)}")
print(f"✓ Keywords indexed (lowercase): {len(keyword_index_lower)}")
print(f"✓ Phrases indexed: {len(phrase_index)}")
print(f"✓ Adjacency connections: {sum(len(v) for v in adjacency_map.values())}")
print(f"✓ Exact duplicates found: {len(duplicate_map)}")
print(f"✓ Similar chunk pairs: {len(similarity_map)}")
print(f"✓ BM25 avg doc length: {bm25_index.avgdl:.1f} tokens")
print("-" * 70)
print(f"✓ FAISS index size: {os.path.getsize(index_path) / 1024 / 1024:.1f} MB")
print(f"✓ BM25 index size: {os.path.getsize(bm25_path) / 1024 / 1024:.1f} MB")
print(f"✓ Metadata size: {os.path.getsize(meta_path) / 1024 / 1024:.1f} MB")
print("=" * 70)
print("🎉 Your HYBRID RAG system is ready!")
print("   ✓ Semantic search (FAISS embeddings)")
print("   ✓ Exact term matching (BM25)")
print("   ✓ Case-insensitive keyword search")
print("   ✓ Duplicate/similarity detection")
print("=" * 70)

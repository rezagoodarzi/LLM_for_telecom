# ingest_pdfs_and_html.py
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

# -------- CONFIG --------
DOCS_DIR = "/home/fatemebookanian/qwen_code/PDF"   # <--- put your documents here (PDFs and HTMLs)
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # More powerful model (768 dims)
# Alternative: "sentence-transformers/all-MiniLM-L12-v2" (384 dims, faster)
OUT_DIR = "./rag_store"
CHUNK_SIZE = 5000     # Larger chunks for better context retention
CHUNK_OVERLAP = 600   # More overlap to prevent info loss at boundaries
EMB_DIM = 768         # embedding dim for all-mpnet-base-v2
TOP_N = 5
SUPPORTED_EXTENSIONS = ['.pdf', '.html', '.htm']  # Supported file types
# ------------------------

os.makedirs(OUT_DIR, exist_ok=True)
index_path = os.path.join(OUT_DIR, "faiss.index")
meta_path = os.path.join(OUT_DIR, "metadata.json")

print("Loading embedding model:", MODEL_NAME)
embedder = SentenceTransformer(MODEL_NAME)

def extract_section_headers(text):
    """Extract section headers from text (e.g., 7.3.6 Site to Site Distance)"""
    lines = text.split('\n')
    headers = []
    
    for line in lines:
        line = line.strip()
        # Pattern 1: Numbered sections (7.3.6, 1.2.3, etc.)
        if re.match(r'^\d+\.[\d\.]+\s+.{3,}', line):
            headers.append(line)
        # Pattern 2: All caps headers with at least 3 words
        elif re.match(r'^[A-Z][A-Z\s]{10,}$', line) and len(line.split()) >= 3:
            headers.append(line)
    
    return headers

def extract_text_from_pdf(path):
    """Extract text from PDF files"""
    doc = fitz.open(path)
    texts = []
    for pageno in range(len(doc)):
        page = doc.load_page(pageno)
        txt = page.get_text("text")
        headers = extract_section_headers(txt)
        texts.append({"page": pageno+1, "text": txt, "headers": headers})
    return texts

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
        
        # Clean up: remove multiple whitespaces and newlines
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract headers from cleaned text
        headers = extract_section_headers(text)
        
        # HTML files don't have pages, so we treat the whole file as "page 1"
        if text.strip():
            return [{"page": 1, "text": text, "headers": headers}]
        return []
    except Exception as e:
        print(f"Error reading HTML {path}: {e}")
        return []

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

# Walk document dir recursively for PDFs and HTMLs
documents = []
processed_files = []
docs_dir_path = Path(DOCS_DIR)

# Process all supported file types
all_files = []
for ext in SUPPORTED_EXTENSIONS:
    all_files.extend(list(docs_dir_path.rglob(f"*{ext}")))

print(f"Found {len(all_files)} document files to process...")

for doc_file in all_files:
    try:
        # Get relative path from DOCS_DIR for better source tracking
        relative_path = doc_file.relative_to(docs_dir_path)
        
        # Extract text based on file type
        if doc_file.suffix.lower() == '.pdf':
            docs = extract_text_from_pdf(str(doc_file))
            file_type = "PDF"
        elif doc_file.suffix.lower() in ['.html', '.htm']:
            docs = extract_text_from_html(str(doc_file))
            file_type = "HTML"
        else:
            continue
        for p in docs:
            # skip empty pages
            if p["text"].strip():
                documents.append({
                    "source": str(relative_path), 
                    "page": p["page"], 
                    "text": p["text"],
                    "headers": p.get("headers", [])
                })
        
        processed_files.append(f"{file_type}: {relative_path}")
        
    except Exception as e:
        print(f"⚠️  Skipping {doc_file.name}: {str(e)}")

# Build chunks & metadata
metadatas = []
texts = []
for doc in tqdm(documents, desc="Creating chunks"):
    chunks = chunk_text(doc["text"])
    # Find most relevant header for this page
    page_header = " | ".join(doc["headers"][:2]) if doc["headers"] else ""
    
    for i, c in enumerate(chunks):
        meta = {
            "source": doc["source"],
            "page": doc["page"],
            "chunk_id": f"{doc['source']}_p{doc['page']}_c{i}",
            "text_snippet": c[:200],
            "section": page_header  # Add section/chapter context
        }
        metadatas.append(meta)
        texts.append(c)

print(f"Total chunks: {len(texts)}")

if len(texts) == 0:
    print("❌ ERROR: No text chunks were created! Check if:")
    print("  1. Documents (PDF/HTML) exist in the directory")
    print("  2. PDFs are not corrupted or encrypted")
    print("  3. Documents contain extractable text")
    print(f"  4. Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
    exit(1)
    
print(f"\n✓ Processed {len(processed_files)} files:")
for pf in processed_files[:10]:  # Show first 10
    print(f"  - {pf}")
if len(processed_files) > 10:
    print(f"  ... and {len(processed_files) - 10} more")

# Embed all chunks in batches
BATCH = 64
embeddings = []
for i in tqdm(range(0, len(texts), BATCH), desc="Embedding batches"):
    batch_texts = texts[i:i+BATCH]
    embs = embedder.encode(batch_texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings.append(embs)
embeddings = np.vstack(embeddings).astype("float32")
print("Embeddings shape:", embeddings.shape)

# Build FAISS index
dim = embeddings.shape[1]
if os.path.exists(index_path):
    print("Removing existing index at", index_path)
    os.remove(index_path)
index = faiss.IndexFlatIP(dim)  # inner product (use cosine with normalized vectors)
# normalize for cosine similarity
faiss.normalize_L2(embeddings)
index.add(embeddings)
faiss.write_index(index, index_path)
print("Saved FAISS index to", index_path)

# Extract keywords/IDs from all chunks for faster exact search
keywords_index = {}
phrases_index = {}  # New: store multi-word phrases

for idx, text in enumerate(texts):
    # Extract potential IDs and keywords
    # Pattern for IDs like SR001225, RAN1795, etc.
    id_patterns = [
        r'\b[A-Z]{2,5}\d{4,6}\b',  # SR001225, RAN1795
        r'\b\d{4,6}\b',             # Pure numbers like 2470
        r'\b[A-Z]+-\d+\b',          # Format like ABC-123
        r'\b\d+\.\d+\.\d+\b',       # Version numbers like 7.3.6
        r'\b[A-Z]{2,}\.\d+\b',      # Format like ABC.123
    ]
    
    keywords = set()
    for pattern in id_patterns:
        matches = re.findall(pattern, text.upper())
        keywords.update(matches)
    
    # Also add capitalized words (potential technical terms)
    cap_words = re.findall(r'\b[A-Z][A-Za-z]{3,}\b', text)
    keywords.update([w.upper() for w in cap_words[:10]])  # Limit to avoid noise
    
    # Extract important phrases (2-4 word combinations)
    text_lower = text.lower()
    # Common technical phrases
    for n in [2, 3, 4]:  # 2-word, 3-word, 4-word phrases
        words = text_lower.split()
        for i in range(len(words) - n + 1):
            phrase = ' '.join(words[i:i+n])
            # Only store phrases with meaningful words (not all stopwords)
            if len(phrase) > 10 and not all(w in ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at'] for w in words[i:i+n]):
                if phrase not in phrases_index:
                    phrases_index[phrase] = []
                if len(phrases_index[phrase]) < 100:  # Limit occurrences per phrase
                    phrases_index[phrase].append(idx)
    
    # Also index section headers if available
    if metadatas[idx].get("section"):
        section = metadatas[idx]["section"].upper()
        keywords.add(section)
        # Also add section number (e.g., "7.3.6" from "7.3.6 Site to Site")
        section_nums = re.findall(r'\b\d+\.[\d\.]+\b', section)
        keywords.update(section_nums)
    
    # Store mapping from keyword to chunk indices
    for kw in keywords:
        if kw not in keywords_index:
            keywords_index[kw] = []
        keywords_index[kw].append(idx)

# Save metadata (text + meta + keywords + phrases)
store = {
    "metadatas": metadatas, 
    "texts": texts, 
    "model": MODEL_NAME,
    "keywords_index": keywords_index,
    "phrases_index": phrases_index
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(store, f, ensure_ascii=False, indent=2)
print("Saved metadata to", meta_path)
print(f"Extracted {len(keywords_index)} unique keywords/IDs")
print(f"Extracted {len(phrases_index)} unique phrases")

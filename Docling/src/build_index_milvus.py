"""
Build Index with Docling and Milvus
====================================

Main script with both FAISS and Milvus vector store options.
Shows all downloads with progress bars.

Usage:
    python build_index_milvus.py --docs-dir  /home/fatemebookanian/qwen_code/PD --use-milvus
    python build_index_milvus.py --docs-dir /path/to/docs --use-faiss
"""

import os
import sys
import argparse
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent))

# Enable HuggingFace progress
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['TRANSFORMERS_VERBOSITY'] = 'info'

from config import settings
from utils.download_progress import print_download_info, setup_huggingface_progress, check_model_cache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Build RAG index with Docling + Milvus/FAISS')
    parser.add_argument('--docs-dir', type=str, default=settings.DOCS_DIR)
    parser.add_argument('--output-dir', type=str, default=settings.OUTPUT_DIR)
    parser.add_argument('--use-milvus', action='store_true', help='Use Milvus vector store')
    parser.add_argument('--use-faiss', action='store_true', help='Use FAISS vector store (default)')
    parser.add_argument('--milvus-host', type=str, default='localhost')
    parser.add_argument('--milvus-port', type=int, default=19530)
    parser.add_argument('--collection-name', type=str, default='rag_documents')
    parser.add_argument('--enable-ocr', action='store_true', default=True)
    parser.add_argument('--no-ocr', action='store_true')
    parser.add_argument('--fast-mode', action='store_true')
    parser.add_argument('--show-downloads', action='store_true', help='Show download info')
    parser.add_argument('--skip-duplicates', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=8)
    return parser.parse_args()

def show_banner():
    print("\n" + "="*60)
    print("ðŸš€ DOCLING RAG INDEX BUILDER WITH MILVUS")
    print("="*60)

def load_embedding_model():
    """Load BGE-M3 with visible progress."""
    print("\nðŸ“¥ Loading embedding model (BGE-M3)...")
    print("   This may download ~1.5GB on first run...")
    setup_huggingface_progress()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.EMBEDDING_MODEL, device='cuda' if settings.USE_FP16 else 'cpu')
    print("âœ… Embedding model loaded")
    return model

def initialize_docling():
    """Initialize Docling with visible progress."""
    print("\nðŸ“¥ Initializing Docling...")
    print("   Downloading document processing models...")
    from indexing.docling_processor import DoclingProcessor, check_docling_installation
    features = check_docling_installation()
    print(f"   Docling: {'âœ…' if features.get('docling_available') else 'âŒ'}")
    print(f"   EasyOCR: {'âœ…' if features.get('easyocr_available') else 'âŒ'}")
    print(f"   PyTorch: {'âœ…' if features.get('torch_available') else 'âŒ'}")
    return DoclingProcessor

def process_with_docling(files, args):
    """Process documents with Docling."""
    from indexing.docling_processor import DoclingProcessor
    from tqdm import tqdm
    
    processor = DoclingProcessor(
        enable_ocr=args.enable_ocr and not args.no_ocr,
        enable_table_extraction=True,
        table_mode='fast' if args.fast_mode else 'accurate',
    )
    
    results = []
    print(f"\nðŸ“„ Processing {len(files)} documents...")
    
    for filepath in tqdm(files, desc="Processing"):
        try:
            result = processor.process_file(str(filepath))
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error: {filepath}: {e}")
    
    return results, processor.get_stats()

def embed_chunks(chunks, model, batch_size=8):
    """Embed chunks with progress."""
    from tqdm import tqdm
    print(f"\nðŸ§® Embedding {len(chunks):,} chunks...")
    texts = [c.get('text', '') for c in chunks]
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        batch_emb = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())
    return embeddings

def store_in_milvus(chunks, embeddings, args):
    """Store in Milvus."""
    from indexing.milvus_store import MilvusVectorStore, MilvusConfig
    
    config = MilvusConfig(
        host=args.milvus_host,
        port=args.milvus_port,
        collection_name=args.collection_name,
        embedding_dim=settings.EMBEDDING_DIM,
    )
    
    store = MilvusVectorStore(config=config)
    store.connect()
    store.create_collection(drop_existing=True)
    store.insert_chunks(chunks, embeddings)
    
    stats = store.get_collection_stats()
    print(f"âœ… Milvus: {stats['num_entities']:,} vectors stored")
    return store

def store_in_faiss(chunks, embeddings, output_dir):
    """Store in FAISS."""
    import numpy as np
    try:
        import faiss
    except ImportError:
        print("âŒ FAISS not installed. Run: pip install faiss-cpu")
        return None
    
    print(f"\nðŸ’¾ Building FAISS index...")
    embeddings_np = np.array(embeddings).astype('float32')
    dim = embeddings_np.shape[1]
    
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    faiss.write_index(index, str(output_path / 'faiss_index.bin'))
    with open(output_path / 'chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"âœ… FAISS: {len(chunks):,} vectors stored at {output_dir}")
    return index

def main():
    args = parse_args()
    show_banner()
    
    if args.show_downloads:
        print_download_info()
    
    # Check cache
    cache = check_model_cache()
    print("\nðŸ“¦ Model Cache Status:")
    for name, info in cache.items():
        status = "âœ…" if info['exists'] else "ðŸ“­"
        print(f"   {status} {name}: {info['size']}")
    
    # Find files
    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        print(f"âŒ Directory not found: {args.docs_dir}")
        sys.exit(1)
    
    all_files = []
    for ext in settings.SUPPORTED_EXTENSIONS:
        all_files.extend(list(docs_path.rglob(f"*{ext}")))
    
    print(f"\nðŸ“ Found {len(all_files)} files")
    
    if not all_files:
        print("No files to process!")
        sys.exit(1)
    
    # Initialize Docling
    initialize_docling()
    
    # Process documents
    results, stats = process_with_docling(all_files, args)
    
    # Collect chunks
    chunks = []
    for doc in results:
        chunks.extend(doc.get('chunks', []))
    
    print(f"\nðŸ“Š Total chunks: {len(chunks):,}")
    
    # Load embedding model
    model = load_embedding_model()
    
    # Embed
    embeddings = embed_chunks(chunks, model, args.batch_size)
    
    # Store
    if args.use_milvus:
        store_in_milvus(chunks, embeddings, args)
    else:
        store_in_faiss(chunks, embeddings, args.output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("âœ… INDEXING COMPLETE")
    print("="*60)
    print(f"   Documents: {len(results):,}")
    print(f"   Chunks: {len(chunks):,}")
    print(f"   Tables: {stats.get('tables_extracted', 0):,}")
    print(f"   Vector Store: {'Milvus' if args.use_milvus else 'FAISS'}")
    print("="*60)

if __name__ == '__main__':
    main()
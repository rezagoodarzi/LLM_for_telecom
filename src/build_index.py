#!/usr/bin/env python3
"""
RAG Index Builder
=================

Build the RAG index from your documents.

Usage:
    python build_index.py [--docs-dir PATH] [--output-dir PATH]

This will:
1. Process all documents (PDF, HTML, MD, TXT)
2. Extract hierarchical structure and metadata
3. Create optimized chunks (2000 chars)
4. Build FAISS semantic index
5. Build BM25 keyword index
6. Build document-level index for 2-stage retrieval
7. Save everything with checkpoint support
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from indexing.document_processor import process_directory
from indexing.index_builder import IndexBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index from documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build from default directory
    python build_index.py
    
    # Specify custom directories
    python build_index.py --docs-dir /path/to/pdfs --output-dir ./my_index
    
    # Process only specific file types
    python build_index.py --extensions .pdf .html
"""
    )
    
    parser.add_argument(
        '--docs-dir', '-d',
        default=settings.DOCS_DIR,
        help=f'Directory containing documents (default: {settings.DOCS_DIR})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default=settings.OUTPUT_DIR,
        help=f'Output directory for index (default: {settings.OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--extensions', '-e',
        nargs='+',
        default=settings.SUPPORTED_EXTENSIONS,
        help=f'File extensions to process (default: {settings.SUPPORTED_EXTENSIONS})'
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("🚀 RAG INDEX BUILDER v2.0")
    print("   Optimized for Technical Documents")
    print("=" * 70)
    
    print(f"\n📁 Configuration:")
    print(f"   Documents: {args.docs_dir}")
    print(f"   Output: {args.output_dir}")
    print(f"   Extensions: {args.extensions}")
    
    print(f"\n⚙️  Chunking Settings:")
    print(f"   Chunk size: {settings.CHUNK_SIZE} chars (~{settings.CHUNK_SIZE//4} tokens)")
    print(f"   Overlap: {settings.CHUNK_OVERLAP} chars")
    print(f"   Max chunk: {settings.MAX_CHUNK_SIZE} chars")
    
    print(f"\n🔧 Index Settings:")
    print(f"   Embedding model: {settings.EMBEDDING_MODEL}")
    print(f"   Embedding dim: {settings.EMBEDDING_DIM}")
    print(f"   BM25 k1={settings.BM25_K1}, b={settings.BM25_B}")
    print(f"   Similarity threshold: {settings.SIMILARITY_THRESHOLD}")
    
    # Check documents directory
    if not os.path.exists(args.docs_dir):
        print(f"\n❌ Error: Documents directory not found: {args.docs_dir}")
        sys.exit(1)
    
    # Process documents
    print("\n" + "=" * 70)
    print("📄 PHASE 1: Document Processing")
    print("=" * 70)
    
    documents, proc_stats = process_directory(args.docs_dir, args.extensions)
    
    if not documents:
        print("\n❌ No documents processed!")
        sys.exit(1)
    
    # Build indices
    print("\n" + "=" * 70)
    print("🔨 PHASE 2: Index Building")
    print("=" * 70)
    
    builder = IndexBuilder(args.output_dir)
    index_stats = builder.build_from_documents(documents)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 FINAL SUMMARY")
    print("=" * 70)
    print(f"   Documents processed: {proc_stats['documents_processed']:,}")
    print(f"   Chunks created: {index_stats['total_chunks']:,}")
    print(f"   Avg chunk size: ~{settings.CHUNK_SIZE} chars")
    print(f"   Sections detected: {proc_stats['sections_detected']:,}")
    print(f"   Keywords indexed: {index_stats['unique_keywords']:,}")
    print(f"\n   Output directory: {args.output_dir}")
    print("=" * 70)
    print("🎉 Index building complete!")
    print("   Run 'python query.py' to start querying.")
    print("=" * 70)


if __name__ == "__main__":
    main()


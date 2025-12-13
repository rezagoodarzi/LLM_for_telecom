"""
Build Index with Docling
=========================

Main script for building RAG index using Docling-enhanced document processing.

Usage:
    python build_index_docling.py --docs-dir /path/to/docs
    python build_index_docling.py --docs-dir /home/fatemebookanian/qwen_code/PDF --enable-ocr
    python build_index_docling.py --docs-dir /path/to/docs --fast-mode

Features:
- Advanced document parsing with Docling
- Table-aware chunking
- Image OCR extraction
- Duplicate detection
- Version selection
- Progress tracking with checkpoints                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from indexing.docling_processor import (
    DoclingProcessor,
    process_directory_with_docling,
    check_docling_installation
)
from utils.docling_utils import (
    VersionSelector,
    ContentFingerprinter,
    deduplicate_chunks,
    export_to_json,
    export_chunks_to_jsonl
)
from utils.similarity_handler import SimilarityHandler, deduplicate_documents
from utils.image_processor import ImageProcessor, images_to_chunks

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import gc
try:
    import torch
except:
    torch = None

def clear_memory():
    """Clear GPU and CPU memory after each document."""
    gc.collect()
    if torch and torch.cuda.is_available():
        torch.cuda.empty_cache()
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build RAG index using Docling document processing'
    )
    
    parser.add_argument(
        '--docs-dir',
        type=str,
        default=settings.DOCS_DIR,
        help='Directory containing documents'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=settings.OUTPUT_DIR,
        help='Output directory for index'
    )
    
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=settings.SUPPORTED_EXTENSIONS,
        help='File extensions to process'
    )
    
    parser.add_argument(
        '--enable-ocr',
        action='store_true',
        default=settings.DOCLING_CONFIG.get('enable_ocr', True),
        help='Enable OCR for images and scanned PDFs'
    )
    
    parser.add_argument(
        '--no-ocr',
        action='store_true',
        help='Disable OCR processing'
    )
    
    parser.add_argument(
        '--enable-tables',
        action='store_true',
        default=settings.DOCLING_CONFIG.get('enable_table_extraction', True),
        help='Enable table extraction'
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='Use fast mode (less accurate but faster)'
    )
    
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        default=True,
        help='Skip duplicate documents'
    )
    
    parser.add_argument(
        '--keep-all-versions',
        action='store_true',
        help='Keep all document versions (default: keep newest only)'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=settings.SIMILARITY_CONFIG.get('near_dup_threshold', 0.90),
        help='Similarity threshold for duplicate detection'
    )
    
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=25,
        help='Save checkpoint every N documents'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    
    parser.add_argument(
        '--export-json',
        action='store_true',
        help='Export results to JSON'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def check_installation():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("DOCLING RAG INDEX BUILDER")
    print("="*60)
    
    features = check_docling_installation()
    
    print("\nInstallation Status:")
    for feature, available in features.items():
        status = "OK" if available else "MISSING"
        symbol = "[+]" if available else "[-]"
        print(f"  {symbol} {feature}: {status}")
    
    if not features.get('docling_available', False):
        print("\nERROR: Docling is not installed!")
        print("Please run: pip install docling")
        sys.exit(1)
    
    return features


def select_versions(files: List[str], keep_all: bool = False) -> List[str]:
    """Select newest version of each document family."""
    if keep_all:
        return files
    
    print("\nPhase 1: Version Selection...")
    selector = VersionSelector()
    
    for filepath in files:
        selector.add_document(filepath)
    
    selected = selector.select_newest()
    stats = selector.get_statistics()
    
    print(f"  Document families: {stats['document_families']}")
    print(f"  Documents to keep: {stats['documents_to_keep']}")
    print(f"  Older versions skipped: {stats['documents_to_skip']}")
    
    return selected


def process_documents(
    files: List[str],
    args,
    checkpoint_path: str = None
) -> Tuple[List[Dict], Dict]:
    """Process documents with Docling."""
    print(f"\nPhase 2: Processing {len(files)} documents with Docling...")
    
    # Initialize processor
    processor = DoclingProcessor(
        enable_ocr=args.enable_ocr and not args.no_ocr,
        enable_table_extraction=args.enable_tables,
        ocr_engine=settings.DOCLING_CONFIG.get('ocr_engine', 'easyocr'),
        table_mode='fast' if args.fast_mode else 'accurate',
        extract_images=settings.DOCLING_CONFIG.get('extract_images', True),
        similarity_threshold=args.similarity_threshold,
    )
    
    # Load checkpoint if resuming
    processed_files = set()
    results = []
    
    if args.resume and checkpoint_path and os.path.exists(checkpoint_path):
        print("  Resuming from checkpoint...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
            results = checkpoint.get('results', [])
            processed_files = set(checkpoint.get('processed_files', []))
        print(f"  Loaded {len(results)} previously processed documents")
    
    # Process remaining files
    from tqdm import tqdm
    
    for i, filepath in enumerate(tqdm(files, desc="Processing")):
        if filepath in processed_files:
            continue
        
        try:
            result = processor.process_file(filepath)
            if result:
                results.append(result)
                clear_memory()
            processed_files.add(filepath)
            
            # Save checkpoint
            if checkpoint_path and (i + 1) % args.checkpoint_interval == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({
                        'results': results,
                        'processed_files': list(processed_files),
                        'timestamp': datetime.now().isoformat()
                    }, f)
                    
        except Exception as e:
            logger.error(f"Error processing {filepath}: {e}")
    
    stats = processor.get_stats()
    
    print(f"\n  Documents processed: {stats['documents_processed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Tables extracted: {stats['tables_extracted']}")
    print(f"  Images processed: {stats['images_processed']}")
    print(f"  Duplicates found: {stats['duplicates_found']}")
    
    return results, stats


def deduplicate_results(
    results: List[Dict],
    threshold: float = 0.90
) -> List[Dict]:
    """Remove duplicate documents from results."""
    print(f"\nPhase 3: Deduplication (threshold: {threshold})...")
    
    unique, duplicates = deduplicate_documents(
        results,
        content_key='document_summary',
        id_key='path',
        threshold=threshold
    )
    
    print(f"  Unique documents: {len(unique)}")
    print(f"  Duplicates removed: {len(duplicates)}")
    
    return unique


def collect_all_chunks(results: List[Dict]) -> List[Dict]:
    """Collect all chunks from processed documents."""
    print("\nPhase 4: Collecting chunks...")
    
    all_chunks = []
    
    for doc in results:
        # Add document chunks
        chunks = doc.get('chunks', [])
        all_chunks.extend(chunks)
        
        # Add image-based chunks if available
        images = doc.get('images', [])
        if images:
            # Convert images with text to chunks
            from utils.image_processor import ImageInfo, images_to_chunks
            
            image_infos = []
            for img in images:
                info = ImageInfo(
                    index=img.get('index', 0),
                    page=img.get('page', 0),
                    ocr_text=img.get('ocr_text', ''),
                    caption=img.get('caption', ''),
                )
                image_infos.append(info)
            
            img_chunks = images_to_chunks(image_infos)
            
            # Add source info
            for chunk in img_chunks:
                chunk['source'] = doc.get('path', '')
                chunk['vendor'] = doc.get('metadata', {}).get('vendor')
            
            all_chunks.extend(img_chunks)
    
    print(f"  Total chunks: {len(all_chunks)}")
    
    # Deduplicate chunks
    unique_chunks = deduplicate_chunks(all_chunks, threshold=0.95)
    print(f"  After deduplication: {len(unique_chunks)}")
    
    return unique_chunks


def save_results(
    results: List[Dict],
    chunks: List[Dict],
    stats: Dict,
    output_dir: str,
    export_json: bool = False
):
    """Save processed results."""
    print(f"\nPhase 5: Saving results to {output_dir}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save chunks for indexing
    chunks_path = output_path / 'chunks.pkl'
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"  Saved chunks to: {chunks_path}")
    
    # Save document metadata
    metadata_path = output_path / 'documents_metadata.pkl'
    doc_metadata = [
        {
            'path': doc['path'],
            'metadata': doc['metadata'],
            'structure': doc.get('structure', {}),
            'tables_count': len(doc.get('tables', [])),
            'images_count': len(doc.get('images', [])),
        }
        for doc in results
    ]
    with open(metadata_path, 'wb') as f:
        pickle.dump(doc_metadata, f)
    print(f"  Saved metadata to: {metadata_path}")
    
    # Save statistics
    stats_path = output_path / 'processing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved statistics to: {stats_path}")
    
    # Export to JSON if requested
    if export_json:
        json_path = output_path / 'chunks.json'
        export_to_json(chunks, str(json_path))
        print(f"  Exported JSON to: {json_path}")
        
        jsonl_path = output_path / 'chunks.jsonl'
        export_chunks_to_jsonl(chunks, str(jsonl_path))
        print(f"  Exported JSONL to: {jsonl_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check installation
    check_installation()
    
    # Find files
    docs_path = Path(args.docs_dir)
    if not docs_path.exists():
        print(f"\nERROR: Documents directory not found: {args.docs_dir}")
        sys.exit(1)
    
    all_files = []
    for ext in args.extensions:
        all_files.extend([str(f) for f in docs_path.rglob(f"*{ext}")])
    
    print(f"\nFound {len(all_files)} files in {args.docs_dir}")
    
    if not all_files:
        print("No files found to process!")
        sys.exit(1)
    
    # Version selection
    files_to_process = select_versions(all_files, args.keep_all_versions)
    
    # Process documents
    checkpoint_path = os.path.join(args.output_dir, 'docling_checkpoint.pkl')
    results, stats = process_documents(files_to_process, args, checkpoint_path)
    
    # Deduplicate
    if args.skip_duplicates:
        results = deduplicate_results(results, args.similarity_threshold)
    
    # Collect chunks
    chunks = collect_all_chunks(results)
    
    # Save results
    save_results(
        results=results,
        chunks=chunks,
        stats=stats,
        output_dir=args.output_dir,
        export_json=args.export_json
    )
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"  Documents processed: {len(results)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Output directory: {args.output_dir}")
    print("\nNext step: Run the embedding and index building script")
    print("="*60)


if __name__ == '__main__':
    main()

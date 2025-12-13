# Docling Integration Guide

## Overview

This guide explains the Docling integration for the RAG system, providing advanced document processing capabilities including:

- **Superior PDF parsing** with layout analysis
- **Table extraction** with structure preservation
- **OCR** for scanned documents and images
- **Hierarchical section detection**
- **Duplicate detection** with multiple algorithms
- **Image processing** with text extraction

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Features](#features)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [File Structure](#file-structure)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for OCR)
- 16GB+ RAM recommended

### Install Docling

```bash
# Install Docling with all extras
pip install docling[all]

# Or minimal installation
pip install docling

# Install OCR support
pip install easyocr

# Install image processing
pip install pillow
```

### Install Project Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Add Docling to requirements (already included)
pip install docling easyocr pillow
```

### Verify Installation

```bash
cd src
python -c "from indexing.docling_processor import check_docling_installation; print(check_docling_installation())"
```

Expected output:
```
{'docling_available': True, 'easyocr_available': True, 'torch_available': True, 'cuda_available': True}
```

---

## Quick Start

### 1. Process Documents with Docling

```bash
cd src
python build_index_docling.py --docs-dir /path/to/your/pdfs
```

### 2. With OCR Enabled (for scanned documents)

```bash
python build_index_docling.py --docs-dir /path/to/pdfs --enable-ocr
```

### 3. Fast Mode (less accurate, faster processing)

```bash
python build_index_docling.py --docs-dir /path/to/pdfs --fast-mode
```

### 4. Process Single Document

```python
from indexing.docling_processor import convert_single_document

result = convert_single_document(
    "your_document.pdf",
    enable_ocr=True,
    enable_table_extraction=True
)

print(f"Chunks: {len(result['chunks'])}")
print(f"Tables: {len(result['tables'])}")
print(f"Images: {len(result['images'])}")
```

---

## Features

### 1. Table Extraction

Docling extracts tables with full structure preservation:

```python
# Tables are extracted as markdown and structured data
for table in result['tables']:
    print(f"Table on page {table['page']}")
    print(f"Headers: {table.get('headers', [])}")
    print(f"Rows: {table.get('row_count', 0)}")
    print(f"Markdown:\n{table['markdown']}")
```

**Benefits:**
- Tables are kept intact as single chunks
- Row/column structure preserved
- Captions extracted when available
- Better retrieval for tabular data

### 2. Section Hierarchy

Docling detects document structure automatically:

```python
structure = result['structure']

# Access sections
for section in structure['sections']:
    print(f"Level {section['level']}: {section['title']} (page {section['page']})")
```

**Detected Elements:**
- Headings (H1-H6)
- Numbered sections (1.2.3 Title)
- Lettered sections (A.1 Title)
- Markdown headers (# Title)

### 3. OCR for Images and Scanned PDFs

```python
# Images with OCR text
for image in result['images']:
    if image.get('ocr_text'):
        print(f"Image on page {image['page']}: {image['ocr_text'][:100]}...")
```

**Supported:**
- Scanned PDF pages
- Embedded images with text
- Diagrams and figures
- Multi-language OCR

### 4. Duplicate Detection

Multiple algorithms for robust detection:

```python
from utils.similarity_handler import SimilarityHandler

handler = SimilarityHandler(near_dup_threshold=0.90)

# Register documents
is_unique, match = handler.register_document("doc1", content1)
is_unique, match = handler.register_document("doc2", content2)

if not is_unique:
    print(f"Duplicate of {match.doc_id} (similarity: {match.similarity_score:.2%})")
```

**Algorithms Used:**
- **SHA-256**: Exact duplicates
- **SimHash**: Near-duplicates (Hamming distance)
- **MinHash**: Jaccard similarity

### 5. Version Selection

Automatically keeps newest version per document family:

```python
from utils.docling_utils import VersionSelector

selector = VersionSelector()

# Add documents
selector.add_document("manual_v1.0.pdf")
selector.add_document("manual_v2.0.pdf")
selector.add_document("manual_v3.0.pdf")

# Get newest versions only
newest = selector.select_newest()
# Returns: ["manual_v3.0.pdf"]
```

**Detected Patterns:**
- `_v1.0`, `-v2.3` (version numbers)
- `_r15`, `_r16` (releases)
- `_rel15`, `_rel16` (3GPP releases)
- `_2023`, `_2024` (years)
- `_rev1`, `_rev2` (revisions)

---

## Configuration

### Settings (src/config/settings.py)

```python
# Enable Docling processing
USE_DOCLING = True

# Docling configuration
DOCLING_CONFIG = {
    # OCR settings
    'enable_ocr': True,
    'ocr_engine': 'easyocr',     # 'easyocr' or 'tesseract'
    'ocr_languages': ['en'],     # Languages for OCR
    
    # Table extraction
    'enable_table_extraction': True,
    'table_mode': 'accurate',    # 'accurate' or 'fast'
    
    # Image processing
    'extract_images': True,
    'save_images': True,
    'max_image_size': 2048,
    
    # Processing options
    'batch_size': 4,
    'timeout_per_doc': 300,
}

# Similarity detection
SIMILARITY_CONFIG = {
    'exact_threshold': 1.0,
    'near_dup_threshold': 0.90,
    'similar_threshold': 0.70,
}
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--docs-dir` | Documents directory | settings.DOCS_DIR |
| `--output-dir` | Output directory | settings.OUTPUT_DIR |
| `--enable-ocr` | Enable OCR | True |
| `--no-ocr` | Disable OCR | False |
| `--enable-tables` | Extract tables | True |
| `--fast-mode` | Fast processing | False |
| `--skip-duplicates` | Skip duplicates | True |
| `--similarity-threshold` | Dup threshold | 0.90 |
| `--resume` | Resume from checkpoint | False |

---

## Usage Examples

### Example 1: Full Processing Pipeline

```python
from indexing.docling_processor import DoclingProcessor

# Initialize with all features
processor = DoclingProcessor(
    enable_ocr=True,
    enable_table_extraction=True,
    ocr_engine="easyocr",
    table_mode="accurate",
    extract_images=True,
    similarity_threshold=0.92
)

# Process a document
result = processor.process_file("technical_spec.pdf")

# Access results
print(f"Title: {result['metadata']['title']}")
print(f"Vendor: {result['metadata'].get('vendor', 'Unknown')}")
print(f"Pages: {result['metadata']['page_count']}")
print(f"Chunks: {len(result['chunks'])}")
print(f"Tables: {len(result['tables'])}")
print(f"Images: {len(result['images'])}")

# Get statistics
stats = processor.get_stats()
print(f"Processing stats: {stats}")
```

### Example 2: Batch Processing with Progress

```python
from indexing.docling_processor import process_directory_with_docling

results, stats = process_directory_with_docling(
    docs_dir="/path/to/documents",
    extensions=['.pdf', '.html', '.md'],
    enable_ocr=True,
    enable_table_extraction=True,
    similarity_threshold=0.90
)

print(f"Processed {len(results)} documents")
print(f"Created {stats['chunks_created']} chunks")
print(f"Found {stats['duplicates_found']} duplicates")
```

### Example 3: Working with Tables

```python
# Extract and process tables
for doc in results:
    for table in doc['tables']:
        print(f"\n--- Table {table['index']+1} (Page {table['page']}) ---")
        
        # Access markdown format
        print(table['markdown'])
        
        # Access structured data
        if 'headers' in table:
            print(f"Headers: {table['headers']}")
            for row in table['rows'][:3]:
                print(f"  {row}")
```

### Example 4: Image OCR Text

```python
# Get searchable text from images
from utils.image_processor import ImageInfo, images_to_chunks

for doc in results:
    for img in doc['images']:
        if img.get('ocr_text'):
            print(f"Image {img['index']} (page {img['page']}):")
            print(f"  OCR: {img['ocr_text'][:200]}...")
            
        if img.get('caption'):
            print(f"  Caption: {img['caption']}")
```

### Example 5: Custom Duplicate Detection

```python
from utils.similarity_handler import SimilarityHandler, find_document_clusters

# Find similar document clusters
clusters = find_document_clusters(
    documents=results,
    content_key='document_summary',
    threshold=0.70
)

print(f"Found {len(clusters)} document clusters")
for i, cluster in enumerate(clusters):
    if len(cluster) > 1:
        print(f"\nCluster {i+1}: {len(cluster)} similar documents")
        for doc in cluster:
            print(f"  - {doc['path']}")
```

---

## File Structure

```
src/
+-- config/
|   +-- settings.py              # Configuration (updated with Docling settings)
|
+-- indexing/
|   +-- docling_processor.py     # Main Docling processor
|   +-- document_processor.py    # Original processor (still available)
|
+-- utils/
|   +-- docling_utils.py         # Docling utilities
|   +-- similarity_handler.py    # Duplicate detection
|   +-- image_processor.py       # Image processing
|
+-- build_index_docling.py       # Main build script with Docling
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `docling_processor.py` | Core Docling document processing |
| `docling_utils.py` | Version selection, fingerprinting, export |
| `similarity_handler.py` | Duplicate/similarity detection |
| `image_processor.py` | Image extraction and OCR |
| `build_index_docling.py` | Command-line build script |

---

## API Reference

### DoclingProcessor

```python
class DoclingProcessor:
    def __init__(
        self,
        enable_ocr: bool = True,
        enable_table_extraction: bool = True,
        ocr_engine: str = "easyocr",
        table_mode: str = "accurate",
        extract_images: bool = True,
        max_image_size: int = 1024,
        similarity_threshold: float = 0.92
    )
    
    def process_file(self, filepath: str) -> Optional[Dict]
    def get_stats(self) -> Dict
    def reset_stats(self) -> None
```

### SimilarityHandler

```python
class SimilarityHandler:
    def __init__(
        self,
        exact_threshold: float = 1.0,
        near_dup_threshold: float = 0.90,
        similar_threshold: float = 0.70,
        shingle_size: int = 5,
        num_hashes: int = 128
    )
    
    def register_document(self, doc_id: str, content: str) -> Tuple[bool, Optional[SimilarityResult]]
    def find_similar(self, content: str, top_k: int = 5) -> List[SimilarityResult]
    def get_stats(self) -> Dict
```

### ImageProcessor

```python
class ImageProcessor:
    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_languages: List[str] = None,
        save_images: bool = True,
        output_dir: str = ".docling_images"
    )
    
    def process_image(self, image, page_number: int, index: int, ...) -> ImageInfo
    def process_batch(self, images: List[Dict], document_path: str) -> List[ImageInfo]
```

---

## Troubleshooting

### Common Issues

#### 1. Docling Not Found

```
ImportError: Docling is not installed
```

**Solution:**
```bash
pip install docling[all]
```

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce batch size in settings
- Use `--fast-mode` for smaller models
- Set `DOCLING_CONFIG['batch_size'] = 2`

#### 3. OCR Not Working

```
EasyOCR failed to initialize
```

**Solutions:**
```bash
pip install easyocr
# Or use Tesseract
pip install pytesseract
# Update settings: ocr_engine='tesseract'
```

#### 4. Slow Processing

**Solutions:**
- Use `--fast-mode` for faster (less accurate) processing
- Disable OCR if not needed: `--no-ocr`
- Reduce image size: `max_image_size = 1024`
- Use GPU: ensure CUDA is available

#### 5. Missing Dependencies

```bash
# Install all optional dependencies
pip install torch torchvision
pip install easyocr
pip install pytesseract
pip install pillow
pip install opencv-python
```

### Performance Tips

1. **GPU Acceleration**: Ensure PyTorch with CUDA is installed
2. **Batch Processing**: Process documents in batches to maximize GPU utilization
3. **Checkpoint Resume**: Use `--resume` to continue interrupted processing
4. **Fast Mode**: Use `--fast-mode` for initial testing
5. **Disable OCR**: Skip OCR for text-based PDFs: `--no-ocr`

---

## Comparison: Original vs Docling

| Feature | Original | With Docling |
|---------|----------|--------------|
| PDF Text Extraction | PyMuPDF | Docling AI |
| Table Detection | Basic | Advanced |
| Section Detection | Regex | AI-powered |
| OCR Support | None | EasyOCR/Tesseract |
| Image Processing | None | Full |
| Duplicate Detection | MinHash | Multi-algorithm |
| Processing Speed | Fast | Moderate |
| Accuracy | Good | Excellent |

---

## License

MIT License - Same as the main project.

---

## Support

For issues with:
- **Docling**: https://github.com/DS4SD/docling
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR
- **This integration**: Create an issue in this repository

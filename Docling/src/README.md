# RAG System v2.0 - Optimized for Technical Documents

A professional RAG (Retrieval-Augmented Generation) system designed for technical documents like 3GPP specifications, vendor manuals, and telecom documentation.

## 🎯 Key Improvements Over v1

| Issue | Old Value | New Value | Impact |
|-------|-----------|-----------|--------|
| Chunk size | 6000 chars | 2000 chars | Better retrieval precision |
| Chunk overlap | 800 chars | 300 chars | Proportional overlap |
| Max chunk | 12000 chars | 3000 chars | Prevents oversized chunks |
| Similarity threshold | 0.85 | 0.92 | Preserves similar technical content |
| Keyword weight | 15% | 30% | Better technical ID matching |
| TOP_K | 12 | 20 | More candidates before reranking |
| RERANK_TOP_K | 8 | 12 | Better final selection |

## ✨ Features

### Before Indexing (Document Cleaning)
1. **Version Selection**: Automatically keeps only the newest version per document family
2. **Document Deduplication**: MinHash-based detection removes ~50-70% duplicate content
3. **Noise Page Filtering**: Removes TOC, legal, revision history, copyright pages

### During Indexing
4. **Hierarchical Chunking**: Respects section boundaries, no mid-section splits
5. **Rich Metadata**: Vendor, version, release, section, topics extracted
6. **2-Stage Index**: Both document-level and chunk-level FAISS indices

### During Retrieval
7. **2-Stage Retrieval**: Document → Chunk prevents cross-document contamination
8. **Vendor Filtering**: Filter results by vendor (Ericsson, Nokia, Huawei)
9. **Section Agglomeration**: Retrieves complete sections for full context
10. **Confidence Penalties**: Penalizes cross-vendor/version mixing

### During Generation
11. **Anti-Hallucination Prompts**: Strict instructions to cite and not guess
12. **Vendor Preference**: When filtered, LLM only uses that vendor's content
13. **Multi-Vendor Warnings**: Alerts when mixing vendors in response
14. **Checkpoint Resume**: Interrupted indexing can be resumed

## 📁 Directory Structure

```
rag_system_v2/
├── config/
│   ├── __init__.py
│   └── settings.py         # All configuration in one place
├── indexing/
│   ├── __init__.py
│   ├── document_processor.py   # Document parsing & chunking
│   └── index_builder.py        # FAISS & BM25 index building
├── retrieval/
│   ├── __init__.py
│   └── hybrid_retriever.py     # Hybrid search + reranking
├── utils/
│   ├── __init__.py
│   ├── helpers.py              # Common utilities
│   └── metadata_extractor.py   # Vendor/version extraction
├── build_index.py              # Main indexing script
├── query.py                    # Interactive query interface
└── README.md
```

## 🚀 Quick Start

### 1. Build the Index

```bash
cd rag_system_v2
python build_index.py --docs-dir /path/to/your/pdfs
```

Options:
- `--docs-dir`: Directory containing documents
- `--output-dir`: Where to save the index
- `--extensions`: File types to process (default: .pdf .html .md .txt)

### 2. Query the System

```bash
python query.py
```

## 💡 Query Tips

### Exact Matches
Use quotes for exact term matching:
```
"SN0012"           # Find exact ID
"f1 function"      # Find exact phrase
"KASME derivation" # Find exact concept
```

### Vendor Filtering
Filter by vendor to prevent cross-vendor contamination:
```
!filter
> Select: 1. Set vendor filter
> Enter vendor: ericsson
```

Then query normally - only Ericsson documents will be searched.

### Technical IDs
The system automatically detects technical IDs like:
- Spec numbers: TS38.331, TR23.501
- Security keys: KASME, KNASint, KUPenc
- Functions: f1, f2, f3, f4, f5
- IDs: SN0012, AMF-ID

### Commands
```
!help      - Show help
!show      - Show retrieved document details
!filter    - Set vendor/release filters
!clear     - Clear filters and history
!config    - Show current configuration
!weights   - Adjust hybrid search weights
exit       - Exit
```

## ⚙️ Configuration

All settings are in `config/settings.py`. Key parameters:

### Chunking
```python
CHUNK_SIZE = 2000          # Characters per chunk (~500 tokens)
CHUNK_OVERLAP = 300        # Overlap between chunks
MAX_CHUNK_SIZE = 3000      # Maximum chunk size
MIN_CHUNK_SIZE = 100       # Minimum chunk size
```

### Retrieval
```python
TOP_K = 20                 # Initial retrieval candidates
RERANK_TOP_K = 12          # After cross-encoder reranking
CONTEXT_CHUNKS = 8         # Chunks sent to LLM
```

### Hybrid Weights
```python
HYBRID_WEIGHTS = {
    'semantic': 0.40,      # Meaning-based search
    'bm25': 0.30,          # Exact term matching
    'keyword': 0.30,       # Technical keyword matching
}
```

### Deduplication
```python
SIMILARITY_THRESHOLD = 0.92   # Higher = keep more similar content
ENABLE_DEDUPLICATION = True   # Remove near-duplicates from results
```

## 🔧 Customization

### Adding New Vendors
Edit `utils/metadata_extractor.py`:
```python
VENDOR_PATTERNS = {
    'ericsson': [r'ericsson', r'eric\.'],
    'your_vendor': [r'pattern1', r'pattern2'],
}
```

### Adding Technical Patterns
Edit `config/settings.py`:
```python
TELECOM_PATTERNS = [
    r'\bYOUR_PATTERN\b',
    # ...
]
```

### Adjusting Weights
Use the `!weights` command at runtime, or edit `settings.py`:
```python
HYBRID_WEIGHTS = {
    'semantic': 0.30,
    'bm25': 0.40,
    'keyword': 0.30,
}
```

## 📊 Performance Tips

1. **For exact technical IDs**: Use quotes and increase keyword weight
2. **For conceptual queries**: Increase semantic weight
3. **For vendor-specific**: Always use vendor filter
4. **For formula lookup**: Use quotes around function names

## 🐛 Troubleshooting

### "Index not found"
Run `build_index.py` first to create the index.

### "LLM API error"
Make sure text-generation-webui is running on port 5000.

### "Out of memory during indexing"
Reduce `BATCH_SIZE` in settings.py (try 4 or 8).

### "Poor retrieval quality"
1. Check chunk sizes are appropriate
2. Try different hybrid weights
3. Enable vendor filtering
4. Use exact quotes for technical terms

## 📝 License

MIT License - Feel free to use and modify.


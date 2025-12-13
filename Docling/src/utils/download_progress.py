"""
Download Progress Tracker
==========================

Provides visible download progress for models and dependencies.
Shows download size, speed, and progress bar.
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, Callable
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_with_progress(url: str, output_path: str, desc: str = "Downloading") -> bool:
    """Download file with visible progress bar."""
    try:
        print(f"\nðŸ“¥ {desc}")
        print(f"   URL: {url}")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        print(f"âœ… Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"âŒ Download failed: {e}")
        return False

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"

def get_cache_size(cache_dir: str) -> str:
    """Get total size of cache directory."""
    total = 0
    path = Path(cache_dir)
    if path.exists():
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
    return format_size(total)

def check_model_cache() -> dict:
    """Check cached models and their sizes."""
    home = Path.home()
    caches = {
        'huggingface': home / '.cache' / 'huggingface',
        'easyocr': home / '.EasyOCR',
        'torch': home / '.cache' / 'torch',
    }
    result = {}
    for name, path in caches.items():
        if path.exists():
            result[name] = {'path': str(path), 'size': get_cache_size(str(path)), 'exists': True}
        else:
            result[name] = {'path': str(path), 'size': '0 B', 'exists': False}
    return result

def setup_huggingface_progress():
    """Enable progress bars for Hugging Face downloads."""
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'info'
    try:
        from huggingface_hub import logging as hf_logging
        hf_logging.set_verbosity_info()
    except:
        pass

def print_download_info():
    """Print information about what will be downloaded."""
    print("\n" + "="*60)
    print("ðŸ“¦ MODEL DOWNLOAD INFORMATION")
    print("="*60)
    print("\nEstimated Downloads:")
    print("  â€¢ Docling models:     ~1.5-2 GB")
    print("  â€¢ EasyOCR (English):  ~200-300 MB")
    print("  â€¢ Total first run:    ~2-3 GB")
    print("\nCache Locations:")
    cache = check_model_cache()
    for name, info in cache.items():
        status = "âœ…" if info['exists'] else "ðŸ“­"
        print(f"  {status} {name}: {info['size']} ({info['path']})")
    print("\n" + "="*60)

if __name__ == "__main__":
    print_download_info()
#!/bin/bash
# BGE-M3 RAG System - Automatic Setup Script
# Usage: bash setup.sh

set -e  # Exit on error

echo "========================================"
echo "BGE-M3 RAG System - Installation"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
echo "Found Python $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.10" | bc) -eq 1 ]]; then
    echo "❌ Python 3.10+ required. Current version: $PYTHON_VERSION"
    exit 1
fi
echo "✓ Python version OK"
echo ""

# Check if CUDA is available
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✓ CUDA detected"
    USE_GPU=true
else
    echo "⚠️  No CUDA detected. Will install CPU-only version."
    USE_GPU=false
fi
echo ""

# Create virtual environment
ENV_NAME="qwen_gpu_env"
echo "Creating virtual environment: $ENV_NAME"
if [ -d "$ENV_NAME" ]; then
    echo "⚠️  Environment already exists. Removing..."
    rm -rf "$ENV_NAME"
fi
python3 -m venv "$ENV_NAME"
echo "✓ Virtual environment created"
echo ""

# Activate environment
echo "Activating environment..."
source "$ENV_NAME/bin/activate"
echo "✓ Environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install PyTorch
echo "Installing PyTorch..."
if [ "$USE_GPU" = true ]; then
    # Detect CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
    echo "CUDA Version: $CUDA_VERSION"
    
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "Installing PyTorch CPU-only..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo "✓ PyTorch installed"
echo ""

# Install FAISS
echo "Installing FAISS..."
if [ "$USE_GPU" = true ]; then
    pip install faiss-gpu
else
    pip install faiss-cpu
fi
echo "✓ FAISS installed"
echo ""

# Install remaining dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo "✓ All dependencies installed"
echo ""

# Verify installation
echo "========================================"
echo "Verifying Installation..."
echo "========================================"
python << 'VERIFY'
import sys
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    
    import faiss
    print(f"✓ FAISS: {faiss.__version__}")
    
    import fitz
    print(f"✓ PyMuPDF: {fitz.__version__}")
    
    from sentence_transformers import SentenceTransformer
    print("✓ SentenceTransformers: OK")
    
    from rank_bm25 import BM25Okapi
    print("✓ BM25: OK")
    
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
    
    print("\n🎉 All packages installed successfully!")
    sys.exit(0)
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
VERIFY

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Installation Complete!"
    echo "========================================"
    echo ""
    echo "To activate the environment, run:"
    echo "  source $ENV_NAME/bin/activate"
    echo ""
    echo "Then run your script:"
    echo "  python word_embbeding_optimized.py"
    echo ""
else
    echo ""
    echo "❌ Installation failed. Check errors above."
    exit 1
fi


#!/usr/bin/env python3
"""
Environment verification script for Qwen setup
Run this to diagnose any issues with your setup.
"""

import sys
import subprocess

def check_item(name, check_func, success_msg, fail_msg, critical=False):
    """Helper function to check and report status"""
    try:
        result = check_func()
        if result:
            print(f"✓ {name}: {success_msg}")
            return True
        else:
            print(f"{'❌' if critical else '⚠'} {name}: {fail_msg}")
            return False
    except Exception as e:
        print(f"{'❌' if critical else '⚠'} {name}: {fail_msg} - {str(e)}")
        return False

def main():
    print("=" * 70)
    print("🔍 Qwen Environment Check")
    print("=" * 70 + "\n")
    
    all_good = True
    
    # Python version
    print("📋 Checking Python...")
    py_version = sys.version_info
    if py_version >= (3, 8):
        print(f"✓ Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"❌ Python version: {py_version.major}.{py_version.minor}.{py_version.micro} (need 3.8+)")
        all_good = False
    
    # PyTorch
    print("\n📋 Checking PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: Yes")
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  - GPU {i}: {name} ({memory:.2f} GB)")
        else:
            print("❌ CUDA available: No")
            print("   Model will run very slowly on CPU")
            all_good = False
    except ImportError:
        print("❌ PyTorch not installed")
        all_good = False
    
    # Transformers
    print("\n📋 Checking Transformers...")
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major >= 4 and minor >= 37:
            print("✓ Version is compatible (4.37.0+)")
        else:
            print(f"⚠ Version might have issues. Recommended: 4.37.0+")
    except ImportError:
        print("❌ Transformers not installed")
        all_good = False
    
    # Other dependencies
    print("\n📋 Checking other dependencies...")
    deps = [
        ('accelerate', 'Accelerate'),
        ('tiktoken', 'Tiktoken'),
        ('einops', 'Einops'),
        ('scipy', 'SciPy'),
        ('transformers_stream_generator', 'Stream Generator'),
        ('peft', 'PEFT'),
        ('optimum', 'Optimum'),
        ('auto_gptq', 'Auto-GPTQ'),
    ]
    
    for module, name in deps:
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} not installed")
            all_good = False
    
    # Optional: Flash Attention
    print("\n📋 Checking optional dependencies...")
    try:
        import flash_attn
        print("✓ Flash Attention installed (recommended)")
    except ImportError:
        print("⚠ Flash Attention not installed (optional but recommended)")
    
    # NVIDIA driver check
    print("\n📋 Checking NVIDIA driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ nvidia-smi accessible")
            # Extract driver version
            for line in result.stdout.split('\n'):
                if 'Driver Version' in line:
                    print(f"✓ {line.strip()}")
                    break
        else:
            print("❌ nvidia-smi not working")
            all_good = False
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
        all_good = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✅ All checks passed! You're ready to use Qwen-7B-Chat-Int8")
        print("\nNext steps:")
        print("  python test_qwen_simple.py   # Quick test")
        print("  python test_qwen.py          # Interactive chat")
    else:
        print("⚠ Some issues detected. Please review the output above.")
        print("\nRecommended actions:")
        print("1. Run the setup script: bash setup_qwen.sh")
        print("2. See QWEN_SETUP_GUIDE.md for detailed instructions")
        print("3. Make sure NVIDIA drivers are installed on Windows")
    print("=" * 70)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Setup script to prepare the environment and download required models.
"""

import os
import sys
import subprocess
import torch
from pathlib import Path


def check_gpu():
    """Check GPU availability and CUDA setup."""
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ CUDA available with {gpu_count} GPU(s)")
        print(f"✓ Primary GPU: {gpu_name}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("⚠ CUDA not available - will use CPU")
        return False


def install_requirements():
    """Install required packages."""
    print("\nInstalling required packages...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False
    return True


def download_models():
    """Download and cache required models."""
    print("\nDownloading and caching models...")
    
    try:
        # Download sentence transformer model
        print("Downloading sentence-transformers model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Sentence transformer model cached")
        
        # Download language model
        print("Downloading language model...")
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        llm = pipeline(
            "text-generation",
            model="google/flan-t5-small",
            device=device
        )
        print("✓ Language model cached")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download models: {str(e)}")
        return False


def create_sample_data():
    """Create sample knowledge base if it doesn't exist."""
    print("\nChecking knowledge base...")
    
    if not os.path.exists("knowledge_base.md"):
        print("⚠ knowledge_base.md not found")
        print("Please ensure the knowledge_base.md file is present")
        return False
    else:
        print("✓ Knowledge base file found")
        return True


def run_quick_test():
    """Run a quick test to verify everything works."""
    print("\nRunning quick system test...")
    
    try:
        from retrieval import setup_retrieval_system
        from simulation import XRaySimulator
        
        # Test simulator
        print("Testing X-ray simulator...")
        simulator = XRaySimulator()
        phantom, projection = simulator.simulate_xray('shepp_logan', size=64)
        print("✓ X-ray simulation working")
        
        # Test retrieval (if knowledge base exists)
        if os.path.exists("knowledge_base.md"):
            print("Testing RAG system...")
            retriever, rag_chain = setup_retrieval_system(['knowledge_base.md'])
            response = rag_chain.generate_response("What is X-ray imaging?")
            print("✓ RAG system working")
        
        print("✓ All systems operational!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {str(e)}")
        return False


def main():
    """Main setup function."""
    print("GPU-Accelerated Scientific Research Assistant")
    print("Setup and Configuration")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Download models
    if not download_models():
        print("⚠ Model download failed - you may need to run this later")
    
    # Check knowledge base
    if not create_sample_data():
        print("⚠ Please create or provide knowledge_base.md")
    
    # Run test
    if run_quick_test():
        print("\n" + "=" * 50)
        print("✓ Setup completed successfully!")
        print("\nYou can now run the assistant with:")
        print("  python main.py                    # Interactive mode")
        print("  python main.py -q 'your question' # Single query")
        print("  python main.py --benchmark        # GPU benchmark")
        print("  python main.py --help             # Show all options")
    else:
        print("\n" + "=" * 50)
        print("⚠ Setup completed with warnings")
        print("Some components may not work correctly")
        print("Check the error messages above")


if __name__ == "__main__":
    main()
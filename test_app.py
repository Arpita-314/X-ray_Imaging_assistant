#!/usr/bin/env python3
"""
Test script to verify the application works correctly.
Run this first before using the main application.
"""

import os
import sys
import traceback

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, using CPU")
    except ImportError:
        print("❌ PyTorch not found - install with: pip install torch")
        return False
    
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found - install with: pip install transformers")
        return False
    
    try:
        import sentence_transformers
        print(f"✓ Sentence Transformers available")
    except ImportError:
        print("❌ Sentence Transformers not found - install with: pip install sentence-transformers")
        return False
    
    try:
        import langchain
        print(f"✓ LangChain {langchain.__version__}")
    except ImportError:
        print("❌ LangChain not found - install with: pip install langchain langchain-community")
        return False
    
    try:
        import faiss
        print(f"✓ FAISS available")
    except ImportError:
        print("❌ FAISS not found - install with: pip install faiss-cpu")
        return False
    
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("❌ Matplotlib not found - install with: pip install matplotlib")
        return False
    
    return True

def test_knowledge_base():
    """Test if knowledge base file exists."""
    print("\nTesting knowledge base...")
    
    if os.path.exists("knowledge_base.md"):
        with open("knowledge_base.md", 'r') as f:
            content = f.read()
        if len(content) > 100:
            print("✓ Knowledge base file exists and has content")
            return True
        else:
            print("⚠ Knowledge base file is too short")
            return False
    else:
        print("❌ Knowledge base file not found")
        return False

def test_simulation():
    """Test X-ray simulation functionality."""
    print("\nTesting X-ray simulation...")
    
    try:
        from simulation import XRaySimulator
        
        simulator = XRaySimulator()
        phantom, projection = simulator.simulate_xray('shepp_logan', size=64)
        
        if phantom.shape == (64, 64) and projection.shape == (64,):
            print("✓ X-ray simulation working correctly")
            return True
        else:
            print(f"❌ Unexpected output shapes: phantom {phantom.shape}, projection {projection.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Simulation test failed: {str(e)}")
        traceback.print_exc()
        return False

def test_retrieval():
    """Test retrieval system functionality."""
    print("\nTesting retrieval system...")
    
    try:
        from retrieval import DocumentRetriever
        import torch
        
        # Test basic retriever initialization
        retriever = DocumentRetriever()
        
        # Test document loading
        if os.path.exists("knowledge_base.md"):
            documents = retriever.load_documents(["knowledge_base.md"])
            if documents:
                print("✓ Document loading working")
                
                # Test vectorstore creation (small test)
                split_docs = retriever.split_documents(documents[:1])  # Just first document
                if split_docs:
                    print("✓ Document splitting working")
                    return True
                else:
                    print("❌ Document splitting failed")
                    return False
            else:
                print("❌ No documents loaded")
                return False
        else:
            print("⚠ No knowledge base to test with")
            return False
            
    except Exception as e:
        print(f"❌ Retrieval test failed: {str(e)}")
        traceback.print_exc()
        return False

def create_minimal_knowledge_base():
    """Create a minimal knowledge base for testing."""
    print("\nCreating minimal knowledge base...")
    
    content = """# X-ray Imaging Basics

## What is X-ray Imaging?

X-ray imaging is a medical imaging technique that uses electromagnetic radiation to create images of internal body structures. X-rays can penetrate soft tissues but are absorbed by denser materials like bone.

## How X-rays Work

When X-rays pass through the body, they are attenuated (reduced in intensity) by different tissues according to their density and atomic composition. This creates contrast in the resulting image.

## Applications

X-ray imaging is used for:
- Bone fracture diagnosis
- Chest imaging for lung conditions
- Dental examinations
- Security screening

## Safety

X-ray procedures should follow the ALARA principle - As Low As Reasonably Achievable - to minimize radiation exposure while maintaining diagnostic quality.
"""
    
    with open("knowledge_base.md", 'w') as f:
        f.write(content)
    
    print("✓ Created minimal knowledge base: knowledge_base.md")

def main():
    """Run all tests."""
    print("GPU-Accelerated Scientific Research Assistant")
    print("Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install missing packages.")
        sys.exit(1)
    
    # Check/create knowledge base
    if not test_knowledge_base():
        create_minimal_knowledge_base()
    
    # Test simulation
    sim_ok = test_simulation()
    
    # Test retrieval
    ret_ok = test_retrieval()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY:")
    print("=" * 50)
    print(f"Imports: ✓")
    print(f"Knowledge Base: ✓")
    print(f"Simulation: {'✓' if sim_ok else '❌'}")
    print(f"Retrieval: {'✓' if ret_ok else '❌'}")
    
    if sim_ok and ret_ok:
        print("\n✓ All tests passed! You can now run:")
        print("  python main_fixed.py")
        print("  python main_fixed.py -q 'What is X-ray imaging?'")
    else:
        print("\n⚠ Some tests failed. The app may work with limited functionality.")
        print("  You can still try: python main_fixed.py")

if __name__ == "__main__":
    main()
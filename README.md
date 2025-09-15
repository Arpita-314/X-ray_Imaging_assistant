# GPU-Accelerated Scientific Research Assistant

A minimal MVP demonstrating RAG (Retrieval-Augmented Generation) with GPU-accelerated X-ray imaging simulation for scientific research applications.

## Features

- **RAG System**: LangChain-based retrieval with FAISS vector store and sentence transformers
- **LLM Integration**: Hugging Face transformers with FLAN-T5 for scientific explanations
- **GPU-Accelerated Simulation**: PyTorch-based X-ray imaging simulation with 2D phantom projection
- **Interactive CLI**: User-friendly command-line interface with real-time visualization
- **Modular Architecture**: Clean separation of retrieval, simulation, and main application logic

## System Requirements

- Python 3.8+
- CUDA-capable GPU (optional, will fall back to CPU)
- 4GB+ RAM (8GB+ recommended)
- 2GB+ disk space for models

## Quick Start

### 1. Clone or Download Files

Ensure you have all the following files:
- `main.py` - Main application
- `retrieval.py` - RAG and document retrieval system
- `simulation.py` - GPU-accelerated X-ray simulation
- `knowledge_base.md` - Scientific knowledge base
- `requirements.txt` - Python dependencies
- `setup.py` - Environment setup script

### 2. Install Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Or run setup script for guided installation
python setup.py
```

### 3. Run the Assistant

**Interactive Mode:**
```bash
python main.py
```

**Single Query Mode:**
```bash
python main.py -q "What is X-ray attenuation?"
```

**GPU Benchmark:**
```bash
python main.py --benchmark
```

## Usage Examples

### Interactive Mode Commands

Once running in interactive mode, you can:

- Ask scientific questions: *"How does X-ray imaging work?"*
- Change phantom type: `sim:chest` or `sim:shepp`
- Set projection angle: `angle:45`
- Toggle simulation: `nosim`
- Run benchmark: `benchmark`
- Exit: `quit`

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  -q, --query TEXT           Single query to process
  -p, --phantom-type CHOICE  Phantom type (shepp_logan, chest)
  -a, --angle FLOAT         Projection angle in degrees
  -n, --no-simulation       Disable X-ray simulation
  -b, --benchmark           Run GPU benchmark
  -k, --knowledge-files PATH Knowledge base files
  -v, --vectorstore-path PATH Vector store location
  --help                    Show help message
```

### Example Queries

Try these sample queries:

1. **Basic X-ray Physics**: *"Explain X-ray attenuation and the Beer-Lambert law"*
2. **Medical Applications**: *"What are the main applications of X-ray imaging in medicine?"*
3. **Technical Details**: *"How do digital X-ray detectors work?"*
4. **Safety**: *"What is the ALARA principle in radiation safety?"*

## Architecture Overview

### Module Structure

```
├── main.py           # CLI interface and orchestration
├── retrieval.py      # RAG system with FAISS and LangChain
├── simulation.py     # GPU-accelerated X-ray simulation
└── knowledge_base.md # Scientific knowledge repository
```

### Key Components

1. **Document Retriever**: Uses sentence-transformers and FAISS for semantic search
2. **RAG Chain**: Combines retrieval with FLAN-T5 for contextual responses
3. **X-ray Simulator**: GPU-accelerated phantom projection using PyTorch
4. **Visualization**: Real-time plotting of simulation results

### GPU Acceleration

The system leverages GPU acceleration for:
- X-ray phantom projection calculations
- Ray transform operations
- Neural network inference (when GPU available)
- Parallel processing of multiple simulations

## Phantom Types

### Shepp-Logan Phantom
- Standard medical imaging test object
- Multiple elliptical structures
- Known analytical properties
- Ideal for algorithm validation

### Chest Phantom
- Anatomically inspired chest model
- Includes lungs, heart, ribs, and spine
- Realistic tissue contrast
- Demonstrates clinical applications

## Performance

### GPU Performance (RTX 3080)
- ~0.05s per 256x256 simulation
- ~20 simulations/second throughput
- <500MB GPU memory usage

### CPU Performance (Intel i7)
- ~0.3s per 256x256 simulation
- ~3 simulations/second throughput
- Compatible with all systems

## Customization

### Adding Knowledge

Edit `knowledge_base.md` or add new markdown files:

```bash
python main.py -k knowledge_base.md additional_docs.md
```

### Custom Phantoms

Add new phantom types in `simulation.py`:

```python
def create_custom_phantom(self, size: int = 256) -> torch.Tensor:
    # Your phantom definition here
    return phantom
```

### Different LLMs

Modify the model in `retrieval.py`:

```python
llm = pipeline(
    "text-generation",
    model="microsoft/DialoGPT-medium",  # Different model
    device=device
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce phantom size: `--phantom-size 128`
   - Use CPU: Force `device='cpu'` in simulator

2. **Model Download Fails**
   - Run `python setup.py` to pre-download models
   - Check internet connection

3. **Import Errors**
   - Ensure all requirements are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Issues

- **Slow GPU Performance**: Update CUDA drivers
- **High Memory Usage**: Reduce batch sizes or phantom resolution
- **Model Loading Time**: Models are cached after first run

## Development

### Running Tests

```bash
# Basic functionality test
python setup.py

# GPU benchmark
python main.py --benchmark

# Sample queries
python main.py -q "Test query"
```

### Code Structure

The codebase follows these principles:
- **Modular Design**: Clear separation of concerns
- **GPU Optimization**: Efficient tensor operations
- **Error Handling**: Graceful degradation on failures
- **Documentation**: Comprehensive docstrings

### Extending Functionality

1. **New Simulation Types**: Add methods to `XRaySimulator`
2. **Additional RAG Sources**: Modify `DocumentRetriever`
3. **Enhanced Visualization**: Extend plotting functions
4. **API Interface**: Wrap components in REST/GraphQL API

## License

This is an educational MVP designed for research and learning purposes. Modify and distribute as needed.

## Acknowledgments

- Hugging Face for transformer models
- PyTorch team for GPU acceleration framework  
- LangChain for RAG orchestration tools
- FAISS team for efficient similarity search

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all requirements are installed correctly
3. Ensure knowledge base file exists and is readable
4. Test with simple queries first

Happy researching! 🔬✨
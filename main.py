#!/usr/bin/env python3
"""
Fixed main application for the GPU-accelerated scientific research assistant.
Provides CLI interface for querying the RAG system and generating X-ray simulations.
"""

import argparse
import os
import sys
import torch
import matplotlib
matplotlib.use('TkAgg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from typing import Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import our modules
try:
    from retrieval import setup_retrieval_system, RAGChain
    from simulation import XRaySimulator, plot_simulation_results, run_gpu_benchmark
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all required files are in the same directory:")
    print("- retrieval.py")
    print("- simulation.py") 
    print("- knowledge_base.md")
    sys.exit(1)


class ScientificAssistant:
    """Main application class for the scientific research assistant."""
    
    def __init__(self, knowledge_files: list, vectorstore_path: str = "vectorstore"):
        """
        Initialize the scientific assistant.
        
        Args:
            knowledge_files: List of paths to knowledge base files
            vectorstore_path: Path to vector store
        """
        print("Initializing Scientific Research Assistant...")
        print("=" * 50)
        
        # Setup retrieval system
        try:
            self.retriever, self.rag_chain = setup_retrieval_system(
                knowledge_files, vectorstore_path
            )
            print("✓ RAG system initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize RAG system: {str(e)}")
            print("Continuing without RAG functionality...")
            self.retriever = None
            self.rag_chain = None
        
        # Setup X-ray simulator
        try:
            self.simulator = XRaySimulator()
            print("✓ X-ray simulator initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize simulator: {str(e)}")
            print("Continuing without simulation functionality...")
            self.simulator = None
        
        print("=" * 50)
        if self.retriever or self.simulator:
            print("System ready! (Some features may be limited)")
        else:
            print("❌ Critical error: No functional components available")
            sys.exit(1)
    
    def answer_query(self, query: str, include_simulation: bool = True, 
                    phantom_type: str = 'shepp_logan', angle: float = 0.0) -> dict:
        """
        Answer a query using RAG and optionally generate simulation.
        
        Args:
            query: User query
            include_simulation: Whether to generate X-ray simulation
            phantom_type: Type of phantom for simulation
            angle: Projection angle for simulation
            
        Returns:
            Dictionary with response and simulation results
        """
        results = {}
        
        # Generate text response using RAG
        if self.rag_chain:
            print("Generating response using RAG system...")
            try:
                response = self.rag_chain.generate_response(query)
                results['response'] = response
                print("✓ Response generated successfully")
            except Exception as e:
                print(f"✗ Failed to generate response: {str(e)}")
                results['response'] = f"Sorry, I encountered an error: {str(e)}"
        else:
            results['response'] = "RAG system not available. Please check your setup."
        
        # Generate X-ray simulation if requested
        if include_simulation and self.simulator:
            print("Running X-ray simulation...")
            try:
                phantom, projection = self.simulator.simulate_xray(
                    phantom_type=phantom_type,
                    size=256,
                    angle=angle
                )
                
                # Create plot
                fig = plot_simulation_results(phantom, projection, angle)
                
                results['phantom'] = phantom
                results['projection'] = projection
                results['figure'] = fig
                results['simulation_params'] = {
                    'phantom_type': phantom_type,
                    'angle': angle,
                    'size': 256
                }
                print("✓ Simulation completed successfully")
                
            except Exception as e:
                print(f"✗ Failed to run simulation: {str(e)}")
                results['simulation_error'] = str(e)
        elif include_simulation:
            results['simulation_error'] = "Simulator not available"
        
        return results
    
    def interactive_mode(self):
        """Run the assistant in interactive CLI mode."""
        print("\n" + "="*60)
        print("GPU-Accelerated Scientific Research Assistant")
        print("Interactive Mode")
        print("="*60)
        print("\nCommands:")
        print("  - Ask any question about X-ray imaging or medical physics")
        print("  - Type 'sim:chest' or 'sim:shepp' to specify phantom type")
        print("  - Type 'angle:X' to set projection angle (degrees)")
        print("  - Type 'nosim' to toggle simulation on/off")
        print("  - Type 'benchmark' to run GPU performance test")
        print("  - Type 'quit' to exit")
        print("-"*60)
        
        # Default parameters
        phantom_type = 'shepp_logan'
        angle = 0.0
        include_simulation = True
        
        while True:
            try:
                query = input("\n🔬 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'benchmark':
                    if self.simulator:
                        run_gpu_benchmark(self.simulator)
                    else:
                        print("❌ Simulator not available")
                    continue
                
                # Parse commands
                if query.startswith('sim:'):
                    phantom_type = query.split(':')[1].strip()
                    if phantom_type not in ['chest', 'shepp_logan', 'shepp']:
                        print("❌ Invalid phantom type. Use 'chest' or 'shepp_logan'")
                        continue
                    if phantom_type == 'shepp':
                        phantom_type = 'shepp_logan'
                    print(f"✓ Phantom type set to: {phantom_type}")
                    continue
                
                if query.startswith('angle:'):
                    try:
                        angle = float(query.split(':')[1].strip())
                        print(f"✓ Projection angle set to: {angle}°")
                        continue
                    except ValueError:
                        print("❌ Invalid angle. Please enter a number.")
                        continue
                
                if query.lower() == 'nosim':
                    include_simulation = not include_simulation
                    status = "enabled" if include_simulation else "disabled"
                    print(f"✓ Simulation {status}")
                    continue
                
                if not query:
                    print("Please enter a query.")
                    continue
                
                # Process query
                print(f"\nProcessing query: '{query}'")
                print(f"Simulation: {'ON' if include_simulation else 'OFF'}")
                if include_simulation:
                    print(f"Phantom: {phantom_type}, Angle: {angle}°")
                
                results = self.answer_query(
                    query=query,
                    include_simulation=include_simulation,
                    phantom_type=phantom_type,
                    angle=angle
                )
                
                # Display results
                print("\n" + "="*50)
                print("RESPONSE:")
                print("="*50)
                print(results['response'])
                
                if include_simulation and 'figure' in results:
                    print("\n" + "="*50)
                    print("SIMULATION RESULTS:")
                    print("="*50)
                    params = results['simulation_params']
                    print(f"Phantom type: {params['phantom_type']}")
                    print(f"Projection angle: {params['angle']}°")
                    print(f"Image size: {params['size']}x{params['size']}")
                    
                    # Save plot
                    import random
                    timestamp = str(random.randint(1000, 9999))
                    filename = f"xray_simulation_{timestamp}.png"
                    results['figure'].savefig(filename, dpi=150, bbox_inches='tight')
                    print(f"✓ Simulation plot saved as: {filename}")
                    
                    # Show plot (non-blocking)
                    plt.show(block=False)
                
                elif include_simulation and 'simulation_error' in results:
                    print(f"❌ Simulation error: {results['simulation_error']}")
                
                print("\n" + "-"*50)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
                import traceback
                traceback.print_exc()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Scientific Research Assistant"
    )
    
    parser.add_argument(
        '--knowledge-files', '-k',
        nargs='+',
        default=['knowledge_base.md'],
        help='Paths to knowledge base files'
    )
    
    parser.add_argument(
        '--vectorstore-path', '-v',
        default='vectorstore',
        help='Path to vector store'
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to process (non-interactive mode)'
    )
    
    parser.add_argument(
        '--phantom-type', '-p',
        choices=['shepp_logan', 'chest'],
        default='shepp_logan',
        help='Type of phantom for simulation'
    )
    
    parser.add_argument(
        '--angle', '-a',
        type=float,
        default=0.0,
        help='Projection angle in degrees'
    )
    
    parser.add_argument(
        '--no-simulation', '-n',
        action='store_true',
        help='Disable X-ray simulation'
    )
    
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run GPU benchmark and exit'
    )
    
    args = parser.parse_args()
    
    # Check if knowledge files exist
    for file_path in args.knowledge_files:
        if not os.path.exists(file_path):
            print(f"❌ Knowledge file not found: {file_path}")
            print("Creating a minimal knowledge base...")
            with open(file_path, 'w') as f:
                f.write("# X-ray Imaging\n\nX-ray imaging uses electromagnetic radiation to create images of internal body structures.")
            print(f"✓ Created basic knowledge file: {file_path}")
    
    # Initialize assistant
    try:
        assistant = ScientificAssistant(
            knowledge_files=args.knowledge_files,
            vectorstore_path=args.vectorstore_path
        )
    except Exception as e:
        print(f"❌ Failed to initialize assistant: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Handle benchmark mode
    if args.benchmark:
        print("\nRunning GPU benchmark...")
        if assistant.simulator:
            run_gpu_benchmark(assistant.simulator, size=512, num_runs=20)
        else:
            print("❌ Simulator not available for benchmark")
        return
    
    # Handle single query mode
    if args.query:
        print(f"\nProcessing query: '{args.query}'")
        
        results = assistant.answer_query(
            query=args.query,
            include_simulation=not args.no_simulation,
            phantom_type=args.phantom_type,
            angle=args.angle
        )
        
        # Display results
        print("\n" + "="*60)
        print("RESPONSE:")
        print("="*60)
        print(results['response'])
        
        if not args.no_simulation and 'figure' in results:
            print("\n" + "="*60)
            print("SIMULATION RESULTS:")
            print("="*60)
            params = results['simulation_params']
            print(f"Phantom type: {params['phantom_type']}")
            print(f"Projection angle: {params['angle']}°")
            print(f"Image size: {params['size']}x{params['size']}")
            
            # Save and show plot
            filename = "xray_simulation_output.png"
            results['figure'].savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✓ Simulation plot saved as: {filename}")
            plt.show()
        
        return
    
    # Run in interactive mode
    assistant.interactive_mode()


if __name__ == "__main__":
    main()

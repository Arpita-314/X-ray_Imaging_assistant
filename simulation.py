"""
GPU-accelerated X-ray imaging simulation module.
Implements 2D phantom projection using PyTorch for GPU acceleration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import math


class XRaySimulator:
    """GPU-accelerated X-ray imaging simulator."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the X-ray simulator.
        
        Args:
            device: Device to run simulation on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
    
    def create_shepp_logan_phantom(self, size: int = 256) -> torch.Tensor:
        """
        Create a Shepp-Logan phantom for X-ray simulation.
        
        Args:
            size: Image size (square image)
            
        Returns:
            2D phantom tensor with attenuation coefficients
        """
        # Shepp-Logan phantom parameters (x, y, a, b, angle, intensity)
        ellipses = [
            [0,     0,     0.69,  0.92,  0,      2.0],   # Outer skull
            [0,    -0.0184, 0.6624, 0.874, 0,     -0.98], # Inner skull  
            [0.22,  0,     0.11,  0.31,  -18,    -0.02], # Right ventricle
            [-0.22, 0,     0.16,  0.41,  18,     -0.02], # Left ventricle
            [0,     0.35,  0.21,  0.25,  0,      0.01],  # Corpus callosum
            [0,     0.1,   0.046, 0.046, 0,      0.01],  # Small structure 1
            [0,    -0.1,   0.046, 0.046, 0,      0.01],  # Small structure 2
            [-0.08, -0.605, 0.046, 0.023, 0,     0.01],  # Small structure 3
            [0,    -0.605, 0.023, 0.023, 0,      0.01],  # Small structure 4
            [0.06, -0.605, 0.046, 0.023, 90,     0.01],  # Small structure 5
        ]
        
        # Create coordinate grids
        x = torch.linspace(-1, 1, size, device=self.device)
        y = torch.linspace(-1, 1, size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        phantom = torch.zeros(size, size, device=self.device)
        
        for params in ellipses:
            x0, y0, a, b, angle, intensity = params
            
            # Rotation transformation
            angle_rad = math.radians(angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            
            # Rotate coordinates
            X_rot = (X - x0) * cos_angle + (Y - y0) * sin_angle
            Y_rot = -(X - x0) * sin_angle + (Y - y0) * cos_angle
            
            # Ellipse equation: (X_rot/a)^2 + (Y_rot/b)^2 <= 1
            ellipse_mask = (X_rot/a)**2 + (Y_rot/b)**2 <= 1
            phantom[ellipse_mask] += intensity
            
        return phantom
    
    def create_chest_phantom(self, size: int = 256) -> torch.Tensor:
        """
        Create a simple chest phantom with lungs, heart, and ribs.
        
        Args:
            size: Image size (square image)
            
        Returns:
            2D chest phantom tensor
        """
        x = torch.linspace(-1, 1, size, device=self.device)
        y = torch.linspace(-1, 1, size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        phantom = torch.zeros(size, size, device=self.device)
        
        # Soft tissue background
        body_mask = X**2 + Y**2 <= 0.8**2
        phantom[body_mask] = 0.2  # Soft tissue attenuation
        
        # Lungs (lower attenuation)
        lung_left = (X + 0.3)**2 + (Y - 0.1)**2 <= 0.25**2
        lung_right = (X - 0.3)**2 + (Y - 0.1)**2 <= 0.25**2
        phantom[lung_left | lung_right] = 0.05  # Air/lung tissue
        
        # Heart (higher attenuation)
        heart_mask = (X + 0.1)**2 + (Y + 0.2)**2 <= 0.15**2
        phantom[heart_mask] = 0.3  # Heart muscle
        
        # Ribs (high attenuation)
        for i in range(-2, 3):
            rib_y = i * 0.2
            rib_mask = (torch.abs(Y - rib_y) <= 0.02) & (torch.abs(X) >= 0.15) & (torch.abs(X) <= 0.7)
            phantom[rib_mask] = 1.0  # Bone
            
        # Spine
        spine_mask = (torch.abs(X) <= 0.05) & (torch.abs(Y) <= 0.4)
        phantom[spine_mask] = 0.8  # Vertebrae
        
        return phantom
    
    def ray_transform(self, phantom: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """
        Perform ray transform (Radon transform) for X-ray projection.
        
        Args:
            phantom: 2D phantom tensor
            angles: Projection angles in degrees
            
        Returns:
            Projection sinogram
        """
        size = phantom.shape[0]
        num_angles = len(angles)
        
        # Initialize sinogram
        sinogram = torch.zeros(num_angles, size, device=self.device)
        
        for i, angle in enumerate(angles):
            # Rotate phantom
            angle_rad = torch.tensor(math.radians(angle.item()), device=self.device)
            
            # Create rotation matrix
            cos_a = torch.cos(angle_rad)
            sin_a = torch.sin(angle_rad)
            rotation_matrix = torch.tensor([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ], device=self.device)
            
            # Create affine transformation matrix
            affine_matrix = rotation_matrix.unsqueeze(0)  # Add batch dimension
            
            # Create grid for rotation
            grid = F.affine_grid(
                affine_matrix, 
                phantom.unsqueeze(0).unsqueeze(0).shape, 
                align_corners=False
            )
            
            # Apply rotation using bilinear interpolation
            rotated_phantom = F.grid_sample(
                phantom.unsqueeze(0).unsqueeze(0), 
                grid, 
                align_corners=False,
                padding_mode='zeros'
            ).squeeze()
            
            # Sum along one axis to get projection
            projection = torch.sum(rotated_phantom, dim=0)
            sinogram[i] = projection
            
        return sinogram
    
    def forward_project(self, phantom: torch.Tensor, angle: float = 0.0) -> torch.Tensor:
        """
        Generate X-ray projection at a single angle.
        
        Args:
            phantom: 2D phantom with attenuation coefficients
            angle: Projection angle in degrees
            
        Returns:
            X-ray projection image
        """
        # Convert to intensity using Beer-Lambert law: I = I0 * exp(-∫μ dx)
        # For simplicity, we use the ray sum as path integral
        angles = torch.tensor([angle], device=self.device)
        sinogram = self.ray_transform(phantom, angles)
        projection = sinogram[0]
        
        # Apply Beer-Lambert law
        I0 = 1.0  # Incident intensity
        intensity = I0 * torch.exp(-projection)
        
        # Add some noise for realism
        noise = torch.randn_like(intensity, device=self.device) * 0.02
        intensity = torch.clamp(intensity + noise, 0, 1)
        
        return intensity
    
    def simulate_xray(self, phantom_type: str = 'shepp_logan', size: int = 256, 
                     angle: float = 0.0, add_noise: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete X-ray simulation pipeline.
        
        Args:
            phantom_type: Type of phantom ('shepp_logan' or 'chest')
            size: Image size
            angle: Projection angle in degrees
            add_noise: Whether to add noise to the projection
            
        Returns:
            Tuple of (phantom, projection)
        """
        # Create phantom
        if phantom_type == 'shepp_logan':
            phantom = self.create_shepp_logan_phantom(size)
        elif phantom_type == 'chest':
            phantom = self.create_chest_phantom(size)
        else:
            raise ValueError(f"Unknown phantom type: {phantom_type}")
        
        # Generate projection
        projection = self.forward_project(phantom, angle)
        
        if add_noise:
            # Add quantum noise (Poisson-like)
            noise_level = 0.01
            noise = torch.randn_like(projection, device=self.device) * noise_level
            projection = torch.clamp(projection + noise, 0, 1)
        
        return phantom, projection


def plot_simulation_results(phantom: torch.Tensor, projection: torch.Tensor, 
                          angle: float = 0.0, save_path: str = None) -> plt.Figure:
    """
    Plot simulation results showing phantom and X-ray projection.
    
    Args:
        phantom: 2D phantom tensor
        projection: X-ray projection tensor  
        angle: Projection angle
        save_path: Optional path to save the plot
        
    Returns:
        Matplotlib figure object
    """
    # Convert tensors to numpy for plotting
    phantom_np = phantom.cpu().numpy()
    projection_np = projection.cpu().numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot phantom
    im1 = axes[0].imshow(phantom_np, cmap='gray', origin='upper')
    axes[0].set_title('Phantom (Attenuation Map)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='μ (cm⁻¹)')
    
    # Plot X-ray projection as 2D image
    projection_2d = projection_np.reshape(-1, 1).repeat(32, axis=1)
    im2 = axes[1].imshow(projection_2d.T, cmap='gray', aspect='auto', origin='upper')
    axes[1].set_title(f'X-ray Projection (angle={angle}°)')
    axes[1].set_xlabel('Detector Position')
    axes[1].set_ylabel('Projection Width')
    plt.colorbar(im2, ax=axes[1], label='Intensity')
    
    # Plot projection profile
    axes[2].plot(projection_np)
    axes[2].set_title('Projection Profile')
    axes[2].set_xlabel('Detector Position')
    axes[2].set_ylabel('Transmitted Intensity')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


def run_gpu_benchmark(simulator: XRaySimulator, size: int = 512, num_runs: int = 10):
    """
    Run GPU performance benchmark for the X-ray simulator.
    
    Args:
        simulator: XRaySimulator instance
        size: Phantom size for benchmark
        num_runs: Number of simulation runs
    """
    import time
    
    print(f"Running GPU benchmark with {size}x{size} phantom...")
    print(f"Device: {simulator.device}")
    
    # Warm up GPU
    phantom, _ = simulator.simulate_xray('shepp_logan', size=size)
    
    # Benchmark
    start_time = time.time()
    for i in range(num_runs):
        phantom, projection = simulator.simulate_xray('shepp_logan', size=size, angle=i*18)
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    print(f"Average simulation time: {avg_time:.4f} seconds")
    print(f"Throughput: {1/avg_time:.2f} simulations/second")
    
    # Memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak GPU memory usage: {memory_mb:.2f} MB")
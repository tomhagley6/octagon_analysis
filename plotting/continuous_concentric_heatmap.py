"""
Continuous Concentric Heatmap Generator

This script creates smooth, continuous concentric heatmaps from spatial position data
using kernel density estimation and radial basis function interpolation.

It provides better results than discrete ring binning with interpolation by:
1. Using true continuous density estimation (KDE)
2. Computing radial density profiles directly
3. Applying proper smoothing in polar/radial coordinates
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RBFInterpolator
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import pandas as pd
from typing import Tuple, Optional


class ContinuousConcentricHeatmap:
    """
    Creates continuous concentric heatmaps from 2D spatial data.
    
    Methods:
    1. Kernel Density Estimation (KDE) - Best for general density visualization
    2. Radial Binning with Interpolation - Your current method, but optimized
    3. Radial Basis Function (RBF) - Best for smooth continuous fields
    """
    
    def __init__(self, center: Tuple[float, float] = (0, 0)):
        """
        Initialize the heatmap generator.
        
        Args:
            center: (x, y) coordinates of the arena center
        """
        self.center = np.array(center)
        
    
    def compute_radial_distances(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute radial distances from center for all positions.
        
        Args:
            positions: Nx2 array of (x, z) positions
            
        Returns:
            N-length array of radial distances
        """
        centered = positions - self.center
        return np.sqrt(np.sum(centered**2, axis=1))
    
    def method_1_kde_continuous(self, 
                                positions: np.ndarray,
                                grid_size: int = 200,
                                bandwidth: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 1: Kernel Density Estimation - True continuous density
        
        This is the most statistically sound approach for creating smooth density maps.
        
        Args:
            positions: Nx2 array of (x, z) positions
            grid_size: Resolution of the output grid
            bandwidth: KDE bandwidth (None = auto via Scott's rule)
            
        Returns:
            X grid, Z grid, density values
        """
        # Create KDE from positions
        kde = gaussian_kde(positions.T, bw_method=bandwidth)
        
        # Create evaluation grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        z_min, z_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Add padding
        padding = 0.1 * max(x_max - x_min, z_max - z_min)
        x = np.linspace(x_min - padding, x_max + padding, grid_size)
        z = np.linspace(z_min - padding, z_max + padding, grid_size)
        X, Z = np.meshgrid(x, z)
        
        # Evaluate KDE on grid
        positions_grid = np.vstack([X.ravel(), Z.ravel()])
        density = kde(positions_grid).reshape(X.shape)
        
        return X, Z, density
    
    def method_2_radial_kde(self,
                           positions: np.ndarray,
                           n_radial_bins: int = 100,
                           bandwidth: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method 2: Radial KDE - Continuous density as function of radius only
        
        This collapses the spatial information to radial distance from center,
        then applies KDE in 1D for a smooth radial profile.
        
        Args:
            positions: Nx2 array of (x, z) positions
            n_radial_bins: Number of radial evaluation points
            bandwidth: KDE bandwidth
            
        Returns:
            radii, density values
        """
        # Compute radial distances
        radii = self.compute_radial_distances(positions)
        
        # Create 1D KDE
        kde = gaussian_kde(radii, bw_method=bandwidth)
        
        # Evaluate on radial grid
        r_eval = np.linspace(0, radii.max() * 1.1, n_radial_bins)
        density = kde(r_eval)
        
        return r_eval, density
    
    def method_3_rbf_interpolation(self,
                                   positions: np.ndarray,
                                   grid_size: int = 200,
                                   kernel: str = 'thin_plate_spline',
                                   smoothing: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 3: Radial Basis Function Interpolation
        
        Creates a smooth continuous field by interpolating count density.
        Best for creating smooth, visually appealing heatmaps.
        
        Args:
            positions: Nx2 array of (x, z) positions
            grid_size: Resolution of output grid
            kernel: RBF kernel type ('thin_plate_spline', 'cubic', 'quintic', 'multiquadric')
            smoothing: Smoothing parameter (0 = exact interpolation, >0 = smoothed)
            
        Returns:
            X grid, Z grid, interpolated density values
        """
        # Bin positions into coarse grid to get counts
        H, x_edges, z_edges = np.histogram2d(
            positions[:, 0], positions[:, 1],
            bins=max(20, grid_size // 10)
        )
        
        # Get center positions and counts for non-zero bins
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2
        X_coarse, Z_coarse = np.meshgrid(x_centers, z_centers)
        
        # Flatten and keep only non-zero bins
        mask = H.T > 0
        points = np.column_stack([X_coarse[mask], Z_coarse[mask]])
        values = H.T[mask]
        
        # Add center point with zero if not present
        if not np.any(np.all(np.abs(points - self.center) < 1e-6, axis=1)):
            points = np.vstack([points, self.center])
            values = np.append(values, 0)
        
        # Create RBF interpolator
        rbf = RBFInterpolator(
            points, values,
            kernel=kernel,
            smoothing=smoothing
        )
        
        # Create fine evaluation grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        z_min, z_max = positions[:, 1].min(), positions[:, 1].max()
        padding = 0.1 * max(x_max - x_min, z_max - z_min)
        
        x = np.linspace(x_min - padding, x_max + padding, grid_size)
        z = np.linspace(z_min - padding, z_max + padding, grid_size)
        X, Z = np.meshgrid(x, z)
        
        # Interpolate
        grid_points = np.column_stack([X.ravel(), Z.ravel()])
        density = rbf(grid_points).reshape(X.shape)
        
        # Ensure non-negative
        density = np.maximum(density, 0)
        
        return X, Z, density
    
    def method_4_improved_ring_binning(self,
                                       positions: np.ndarray,
                                       n_rings: int = 50,
                                       grid_size: int = 200,
                                       sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method 4: Improved ring binning with proper smoothing
        
        This is your current approach but optimized with:
        - More bins for finer resolution
        - Gaussian smoothing instead of linear interpolation
        - Proper normalization
        
        Args:
            positions: Nx2 array of (x, z) positions
            n_rings: Number of concentric rings
            grid_size: Resolution of output grid
            sigma: Gaussian smoothing parameter (pixels)
            
        Returns:
            X grid, Z grid, smoothed density values
        """
        # Compute radial distances
        radii = self.compute_radial_distances(positions)
        max_radius = radii.max()
        
        # Create ring bins
        ring_edges = np.linspace(0, max_radius * 1.1, n_rings + 1)
        ring_centers = (ring_edges[:-1] + ring_edges[1:]) / 2
        
        # Count positions in each ring
        counts, _ = np.histogram(radii, bins=ring_edges)
        
        # Normalize by ring area to get density
        ring_areas = np.pi * (ring_edges[1:]**2 - ring_edges[:-1]**2)
        densities = counts / ring_areas
        
        # Create 2D grid
        x_range = max_radius * 1.1
        x = np.linspace(-x_range, x_range, grid_size)
        z = np.linspace(-x_range, x_range, grid_size)
        X, Z = np.meshgrid(x, z)
        
        # Compute radius for each grid point
        R = np.sqrt((X - self.center[0])**2 + (Z - self.center[1])**2)
        
        # Map radii to densities using interpolation
        density_map = np.interp(R, ring_centers, densities, left=0, right=0)
        
        # Apply Gaussian smoothing for continuous appearance
        density_smooth = gaussian_filter(density_map, sigma=sigma)
        
        return X, Z, density_smooth
    
    def plot_comparison(self,
                       positions: np.ndarray,
                       save_path: Optional[str] = None):
        """
        Create a comparison plot of all methods.
        
        Args:
            positions: Nx2 array of (x, z) positions
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Method 1: KDE
        X, Z, density = self.method_1_kde_continuous(positions)
        im1 = axes[0, 0].contourf(X, Z, density, levels=50, cmap='hot')
        axes[0, 0].set_title('Method 1: 2D KDE (Continuous)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Z Position')
        axes[0, 0].scatter(self.center[0], self.center[1], c='cyan', s=100, marker='x', label='Center')
        axes[0, 0].legend()
        plt.colorbar(im1, ax=axes[0, 0], label='Density')
        
        # Method 2: Radial KDE (show as line plot)
        r_eval, density_radial = self.method_2_radial_kde(positions)
        axes[0, 1].plot(r_eval, density_radial, 'r-', linewidth=2)
        axes[0, 1].fill_between(r_eval, density_radial, alpha=0.3)
        axes[0, 1].set_title('Method 2: Radial KDE Profile', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Radial Distance from Center')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Method 3: RBF
        X, Z, density = self.method_3_rbf_interpolation(positions)
        im3 = axes[0, 2].contourf(X, Z, density, levels=50, cmap='hot')
        axes[0, 2].set_title('Method 3: RBF Interpolation', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('X Position')
        axes[0, 2].set_ylabel('Z Position')
        axes[0, 2].scatter(self.center[0], self.center[1], c='cyan', s=100, marker='x', label='Center')
        axes[0, 2].legend()
        plt.colorbar(im3, ax=axes[0, 2], label='Count Density')
        
        # Method 4: Improved ring binning
        X, Z, density = self.method_4_improved_ring_binning(positions)
        im4 = axes[1, 0].contourf(X, Z, density, levels=50, cmap='hot')
        axes[1, 0].set_title('Method 4: Ring Binning + Smoothing', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Z Position')
        axes[1, 0].scatter(self.center[0], self.center[1], c='cyan', s=100, marker='x', label='Center')
        axes[1, 0].legend()
        plt.colorbar(im4, ax=axes[1, 0], label='Density')
        
        # Scatter plot of raw data
        axes[1, 1].scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.5, c='blue')
        axes[1, 1].scatter(self.center[0], self.center[1], c='red', s=100, marker='x', label='Center')
        axes[1, 1].set_title('Raw Position Data', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('X Position')
        axes[1, 1].set_ylabel('Z Position')
        axes[1, 1].legend()
        axes[1, 1].set_aspect('equal')
        
        # Radial histogram
        radii = self.compute_radial_distances(positions)
        axes[1, 2].hist(radii, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Radial Distance Distribution', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Radial Distance from Center')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()


def example_usage():
    """
    Example usage with synthetic data and real agent logs.
    """
    # Example 1: Synthetic concentric data
    print("Generating synthetic concentric data...")
    n_points = 5000
    
    # Generate points with density decreasing with radius
    angles = np.random.uniform(0, 2*np.pi, n_points)
    # Bias toward center
    radii = np.random.exponential(scale=3, size=n_points)
    
    x = radii * np.cos(angles)
    z = radii * np.sin(angles)
    positions = np.column_stack([x, z])
    
    # Create heatmap
    heatmap = ContinuousConcentricHeatmap(center=(0, 0))
    heatmap.plot_comparison(positions, save_path='synthetic_heatmap_comparison.png')
    
    # Example 2: Load real agent data
    print("\nTo use with real agent data:")
    print("```python")
    print("# Load CSV data")
    print("df = pd.read_csv('path/to/log_PlayerAgent_*.csv')")
    print("")
    print("# Extract positions (adjust column names as needed)")
    print("positions = df[['PosX', 'PosZ']].values")
    print("")
    print("# Create heatmap with arena center")
    print("heatmap = ContinuousConcentricHeatmap(center=(0, 0))  # adjust center")
    print("heatmap.plot_comparison(positions, save_path='agent_heatmap.png')")
    print("```")
    
    # Example 3: Generate single heatmap with preferred method
    print("\n\nGenerating high-quality single heatmap...")
    X, Z, density = heatmap.method_1_kde_continuous(positions, grid_size=300)
    
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.contourf(X, Z, density, levels=100, cmap='hot')
    ax.set_title('High-Resolution Continuous Concentric Heatmap\n(KDE Method)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Z Position', fontsize=12)
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, label='Density')
    cbar.set_label('Density', fontsize=12)
    
    # Add concentric reference circles
    for r in [2, 4, 6, 8]:
        circle = plt.Circle((0, 0), r, fill=False, color='cyan', 
                           linestyle='--', alpha=0.3, linewidth=1)
        ax.add_patch(circle)
    
    plt.tight_layout()
    plt.savefig('high_quality_concentric_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDone! Check the output images.")


if __name__ == "__main__":
    example_usage()

"""
L1-Ball Projection Algorithm - Core Innovation
===============================================

This module implements the exact projection onto the L1-ball using Duchi's algorithm.
This is the key technical innovation that enables robust handling of impulse noise.

Reference: 
Duchi et al. "Efficient Projections onto the L1-Ball for Learning in High Dimensions"
"""

import numpy as np
from typing import Tuple, Dict, Optional
import time

def project_l1_ball(v: np.ndarray, radius: float, validate: bool = False) -> np.ndarray:
    """
    Project vector v onto L1-ball of given radius using Duchi's algorithm.
    
    Solves: argmin_z ||z - v||₂² subject to ||z||₁ ≤ radius
    
    Args:
        v: Input vector (any shape, will be flattened internally)
        radius: L1-ball radius (must be positive)
        validate: Whether to validate the projection result
        
    Returns:
        Projected vector with same shape as input
        
    Algorithm:
        1. Handle trivial case (already inside ball)
        2. Store signs and work with absolute values  
        3. Sort absolute values in descending order
        4. Find optimal threshold via cumulative sum
        5. Apply soft thresholding with threshold
        6. Restore original signs and shape
    """
    if radius <= 0:
        raise ValueError("Radius must be positive")
    
    original_shape = v.shape
    v_flat = v.flatten().astype(np.float64)
    n = len(v_flat)
    
    # Quick exit if already inside ball
    current_l1_norm = np.sum(np.abs(v_flat))
    if current_l1_norm <= radius:
        return v
    
    # Store signs and work with absolute values
    signs = np.sign(v_flat)
    u = np.abs(v_flat)
    
    # Sort in descending order for Duchi's algorithm
    u_sorted = np.sort(u)[::-1]
    
    # Find optimal threshold using cumulative sum
    # We need: sum(max(u_i - τ, 0)) = radius
    cumsum = np.cumsum(u_sorted)
    indices = np.arange(1, n + 1)
    
    # Threshold candidates: τ_j = (sum_{i=1}^j u_sorted[i] - radius) / j
    threshold_candidates = (cumsum - radius) / indices
    
    # Find largest j such that τ_j < u_sorted[j-1] (using 0-based indexing)
    # Add small tolerance for numerical stability with floating-point comparison
    valid_indices = np.where(threshold_candidates < u_sorted + 1e-10)[0]
    
    if len(valid_indices) > 0:
        # Take the largest valid index
        j = valid_indices[-1]
        threshold = threshold_candidates[j]
    else:
        # Degenerate case - project to zero
        threshold = np.inf
    
    # Apply soft thresholding: max(|v_i| - τ, 0) * sign(v_i)
    projected_abs = np.maximum(u - threshold, 0)
    projected = signs * projected_abs
    
    # Reshape to original form
    result = projected.reshape(original_shape)
    
    # Optional validation
    if validate:
        _validate_projection(v, result, radius)
    
    return result

def project_l2_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """
    Project vector v onto L2-ball of given radius.
    
    This is the baseline method from Benfenati 2024 for comparison.
    Much simpler than L1 projection - just scaling.
    
    Args:
        v: Input vector  
        radius: L2-ball radius
        
    Returns:
        Projected vector
    """
    if radius <= 0:
        raise ValueError("Radius must be positive")
    
    norm = np.linalg.norm(v)
    if norm <= radius:
        return v

    # Handle zero vector edge case
    if norm == 0:
        return v  # Zero vector projects to itself

    return v * (radius / norm)

def _validate_projection(original: np.ndarray, projected: np.ndarray, radius: float, 
                        tolerance: float = 1e-10) -> None:
    """Validate that projection is correct"""
    l1_norm = np.sum(np.abs(projected))
    
    # Check constraint satisfaction
    if l1_norm > radius + tolerance:
        raise ValueError(f"Projection failed: L1 norm {l1_norm} > radius {radius}")
    
    # Check optimality conditions (simplified KKT check)
    residual = projected - original
    
    # For points inside the ball, gradient should be zero
    if l1_norm < radius - tolerance:
        if np.linalg.norm(residual) > tolerance:
            raise ValueError("Interior point optimality violated")

def benchmark_projection_algorithms(sizes: list = [100, 1000, 10000], 
                                  num_trials: int = 10) -> Dict:
    """
    Benchmark L1 vs L2 projection performance.
    
    Args:
        sizes: Vector sizes to test
        num_trials: Number of trials per size
        
    Returns:
        Performance comparison results
    """
    results = {'l1': {}, 'l2': {}}
    
    for size in sizes:
        l1_times = []
        l2_times = []
        
        for _ in range(num_trials):
            # Generate random test vector
            v = np.random.randn(size)
            radius = np.sum(np.abs(v)) * 0.5  # Ensure projection needed
            
            # Time L1 projection
            start = time.perf_counter()
            project_l1_ball(v, radius)
            l1_times.append(time.perf_counter() - start)
            
            # Time L2 projection  
            start = time.perf_counter()
            project_l2_ball(v, radius)
            l2_times.append(time.perf_counter() - start)
        
        results['l1'][size] = {
            'mean_time': np.mean(l1_times),
            'std_time': np.std(l1_times)
        }
        results['l2'][size] = {
            'mean_time': np.mean(l2_times), 
            'std_time': np.std(l2_times)
        }
    
    return results

def test_projection_correctness():
    """Test suite for projection algorithms"""
    print("Testing L1-ball projection correctness...")
    
    # Test 1: Interior point (should remain unchanged)
    v1 = np.array([0.1, 0.2, -0.1])
    radius1 = 1.0
    proj1 = project_l1_ball(v1, radius1, validate=True)
    assert np.allclose(v1, proj1), "Interior point test failed"
    print("✓ Interior point test passed")
    
    # Test 2: Boundary projection
    v2 = np.array([1.0, 1.0, 1.0])
    radius2 = 2.0
    proj2 = project_l1_ball(v2, radius2, validate=True)
    assert abs(np.sum(np.abs(proj2)) - radius2) < 1e-10, "Boundary projection failed"
    print("✓ Boundary projection test passed")
    
    # Test 3: Zero vector
    v3 = np.zeros(5)
    radius3 = 1.0
    proj3 = project_l1_ball(v3, radius3, validate=True)
    assert np.allclose(v3, proj3), "Zero vector test failed"
    print("✓ Zero vector test passed")
    
    # Test 4: Random vector consistency
    np.random.seed(42)
    for i in range(10):
        v = np.random.randn(50)
        radius = np.random.uniform(0.1, 10.0)
        proj = project_l1_ball(v, radius, validate=True)
        
        # Check constraint satisfaction
        assert np.sum(np.abs(proj)) <= radius + 1e-10, f"Random test {i} failed"
    
    print("✓ Random vector tests passed")
    print("All projection tests completed successfully!")

if __name__ == "__main__":
    # Run correctness tests
    test_projection_correctness()
    
    # Run performance benchmark
    print("\nRunning performance benchmark...")
    benchmark_results = benchmark_projection_algorithms()
    
    for size, data in benchmark_results['l1'].items():
        l1_time = data['mean_time'] * 1000  # Convert to ms
        l2_time = benchmark_results['l2'][size]['mean_time'] * 1000
        slowdown = l1_time / l2_time
        print(f"Size {size:5d}: L1 = {l1_time:6.2f}ms, L2 = {l2_time:6.2f}ms, "
              f"Slowdown = {slowdown:.1f}x")

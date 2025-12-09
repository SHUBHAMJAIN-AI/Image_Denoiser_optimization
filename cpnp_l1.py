"""
Robust Constrained Plug-and-Play ADMM Implementation
===================================================

This module implements the core CPnP-ADMM algorithm with both L1 and L2 constraints.
The L1 constraint version is the novel contribution for robust impulse noise handling.

Key Features:
- Plug-and-Play framework with pre-trained denoisers
- L1-ball constraints for impulse noise robustness
- L2-ball constraints for baseline comparison
- Comprehensive convergence monitoring
- Flexible denoiser integration
"""

import numpy as np
import time
from typing import Tuple, Dict, Optional, Callable, Any
from dataclasses import dataclass
from .projections import project_l1_ball, project_l2_ball

@dataclass
class CPnPConfig:
    """Configuration for CPnP-ADMM algorithm"""
    constraint_type: str = 'l1'  # 'l1' or 'l2'
    rho: float = 1.0             # ADMM penalty parameter
    max_iter: int = 50           # Maximum iterations
    tolerance: float = 1e-6      # Convergence tolerance
    verbose: bool = True         # Print progress
    store_history: bool = True   # Store convergence history

@dataclass
class ConvergenceInfo:
    """Information about algorithm convergence"""
    iterations: int
    converged: bool
    final_primal_residual: float
    final_dual_residual: float
    final_constraint_violation: float
    runtime: float
    history: Dict[str, list]

class RobustCPnP:
    """
    Robust Constrained Plug-and-Play ADMM for image restoration.
    
    Solves the constrained optimization problem:
        minimize   g(x)  [implicit via denoiser]
        subject to ||y - x||_p ≤ ε
    
    where p=1 (L1 constraint, novel) or p=2 (L2 constraint, baseline)
    """
    
    def __init__(self, denoiser: Callable, config: CPnPConfig = None):
        """
        Initialize CPnP solver.
        
        Args:
            denoiser: Function that takes noisy image and returns denoised version
            config: Algorithm configuration
        """
        self.denoiser = denoiser
        self.config = config or CPnPConfig()
        self.history = self._init_history() if config.store_history else None
    
    def _init_history(self) -> Dict[str, list]:
        """Initialize convergence history tracking"""
        return {
            'primal_residuals': [],
            'dual_residuals': [],
            'constraint_violations': [],
            'objectives': [],
            'l1_norms': [],
            'l2_norms': []
        }
    
    def solve(self, y: np.ndarray, epsilon: float, x_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, ConvergenceInfo]:
        """
        Solve the constrained image restoration problem.
        
        Args:
            y: Noisy observed image
            epsilon: Constraint radius (noise tolerance)
            x_init: Initial guess for x (default: y)
            
        Returns:
            x_restored: Restored image
            info: Convergence information
        """
        if epsilon <= 0:
            raise ValueError("Constraint radius epsilon must be positive")
        
        start_time = time.perf_counter()
        
        # Initialize variables
        x = x_init.copy() if x_init is not None else y.copy()
        z = np.zeros_like(y)
        u = np.zeros_like(y)
        z_prev = None
        
        if self.config.verbose:
            print(f"Starting CPnP-ADMM with {self.config.constraint_type.upper()} constraint")
            print(f"Image shape: {y.shape}, epsilon: {epsilon:.4f}, rho: {self.config.rho}")
        
        # Main ADMM iteration loop
        for iteration in range(self.config.max_iter):
            # Store previous iterate for dual residual computation
            x_prev = x.copy()
            if z_prev is not None:
                z_old = z_prev.copy()
            else:
                z_old = z.copy()
            
            # 1. x-update: Plug-and-Play step (Proximal operator of g)
            #    x^{k+1} = argmin_x [g(x) + (ρ/2)||y - x - z^k + u^k||²]
            #             = Denoiser(y - z^k + u^k)
            denoiser_input = y - z + u
            x = self.denoiser(denoiser_input)
            
            # 2. z-update: Constraint projection step (THE KEY INNOVATION)
            #    z^{k+1} = Proj_{||·||_p ≤ ε}(y - x^{k+1} + u^k)
            residual_vector = y - x + u
            
            if self.config.constraint_type == 'l1':
                # Novel L1-ball projection for impulse noise robustness
                z = project_l1_ball(residual_vector, epsilon)
            elif self.config.constraint_type == 'l2':
                # Baseline L2-ball projection (Benfenati 2024)
                z = project_l2_ball(residual_vector, epsilon)
            else:
                raise ValueError(f"Unknown constraint type: {self.config.constraint_type}")
            
            # 3. u-update: Dual variable update (Lagrange multiplier)
            #    u^{k+1} = u^k + (y - x^{k+1} - z^{k+1})
            u = u + (y - x - z)
            
            # Compute convergence metrics
            primal_residual = np.linalg.norm(y - x - z)
            dual_residual = self.config.rho * np.linalg.norm(z - z_old)
            
            # Compute constraint violation
            if self.config.constraint_type == 'l1':
                constraint_violation = max(0, np.sum(np.abs(z)) - epsilon)
            else:
                constraint_violation = max(0, np.linalg.norm(z) - epsilon)
            
            # Store history
            if self.history is not None:
                self.history['primal_residuals'].append(primal_residual)
                self.history['dual_residuals'].append(dual_residual)
                self.history['constraint_violations'].append(constraint_violation)
                self.history['l1_norms'].append(np.sum(np.abs(z)))
                self.history['l2_norms'].append(np.linalg.norm(z))
            
            # Check convergence
            converged = (primal_residual < self.config.tolerance and 
                        dual_residual < self.config.tolerance and
                        constraint_violation < self.config.tolerance)
            
            if self.config.verbose and (iteration % 10 == 0 or converged):
                print(f"Iter {iteration:3d}: "
                      f"Primal={primal_residual:.2e}, "
                      f"Dual={dual_residual:.2e}, "
                      f"Constraint={constraint_violation:.2e}")
            
            if converged:
                if self.config.verbose:
                    print(f"Converged after {iteration + 1} iterations!")
                break
            
            z_prev = z.copy()
        
        runtime = time.perf_counter() - start_time
        
        # Create convergence info
        info = ConvergenceInfo(
            iterations=iteration + 1,
            converged=converged,
            final_primal_residual=primal_residual,
            final_dual_residual=dual_residual,
            final_constraint_violation=constraint_violation,
            runtime=runtime,
            history=self.history.copy() if self.history is not None else {}
        )
        
        if self.config.verbose:
            print(f"Algorithm completed in {runtime:.2f}s")
            if not converged:
                print("Warning: Algorithm did not converge within maximum iterations")
        
        return x, info
    
    def compute_objective_value(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the implicit objective value.
        
        Note: Since g(x) is implicit (defined by the denoiser), we use
        a proxy objective: ||x - Denoiser(x)||² + constraint_penalty
        """
        denoised = self.denoiser(x)
        data_fidelity = 0.5 * np.sum((x - denoised)**2)
        return data_fidelity

def create_denoiser_wrapper(denoiser_func: Callable) -> Callable:
    """
    Create a standardized wrapper for denoising functions.
    
    Args:
        denoiser_func: Raw denoising function
        
    Returns:
        Wrapped denoiser with consistent interface
    """
    def wrapped_denoiser(noisy_image: np.ndarray) -> np.ndarray:
        """Wrapped denoiser with input validation and normalization"""
        # Ensure proper data type and range
        if noisy_image.dtype != np.float64:
            noisy_image = noisy_image.astype(np.float64)
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0.0, 1.0)
        
        # Apply denoising
        denoised = denoiser_func(noisy_image)
        
        # Ensure output is in valid range
        denoised = np.clip(denoised, 0.0, 1.0)
        
        return denoised.astype(np.float64)
    
    return wrapped_denoiser

def compare_constraint_methods(y: np.ndarray, epsilon: float, denoiser: Callable, 
                             config: CPnPConfig = None) -> Dict[str, Tuple[np.ndarray, ConvergenceInfo]]:
    """
    Compare L1 vs L2 constraint methods on the same input.
    
    Args:
        y: Noisy input image
        epsilon: Constraint radius
        denoiser: Denoising function
        config: Base configuration (will be modified for each method)
        
    Returns:
        Dictionary with results for each constraint type
    """
    if config is None:
        config = CPnPConfig(verbose=False)  # Suppress individual outputs
    
    results = {}
    
    for constraint_type in ['l1', 'l2']:
        method_config = CPnPConfig(
            constraint_type=constraint_type,
            rho=config.rho,
            max_iter=config.max_iter,
            tolerance=config.tolerance,
            verbose=False,
            store_history=True
        )
        
        solver = RobustCPnP(denoiser, method_config)
        x_restored, info = solver.solve(y, epsilon)
        
        results[constraint_type] = (x_restored, info)
        
        print(f"{constraint_type.upper()} method: "
              f"{info.iterations} iterations, "
              f"converged: {info.converged}, "
              f"runtime: {info.runtime:.3f}s")
    
    return results

# Utility function for parameter selection
def estimate_noise_level(y: np.ndarray, method: str = 'robust_mad') -> float:
    """
    Estimate noise level in image for epsilon selection.
    
    Args:
        y: Noisy image
        method: Estimation method ('robust_mad', 'std', 'percentile')
        
    Returns:
        Estimated noise level
    """
    if method == 'robust_mad':
        # Median Absolute Deviation (robust to outliers)
        median = np.median(y)
        mad = np.median(np.abs(y - median))
        return mad * 1.4826  # Scale factor for normal distribution
    
    elif method == 'std':
        return np.std(y)
    
    elif method == 'percentile':
        # Use percentile-based estimation
        return np.percentile(np.abs(y - np.median(y)), 84.1)
    
    else:
        raise ValueError(f"Unknown noise estimation method: {method}")

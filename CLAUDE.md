# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Robust Constrained Plug-and-Play ADMM** implementation for image restoration that uses **L¹-ball constraints** instead of traditional L²-ball constraints. The key innovation is superior robustness against impulse noise while maintaining competitive performance on Gaussian noise.

**Problem Formulation:**
```
minimize   g(x)  [implicit via pre-trained denoiser]
subject to ||y - x||₁ ≤ ε
```

Where:
- `x`: Clean image (unknown)
- `y`: Noisy observed image
- `g(x)`: Implicit regularization via plug-and-play denoiser
- `ε`: L¹-ball radius (noise tolerance)

## Running the Code

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Advanced denoisers (if available)
pip install bm3d deepinv
```

### Quick Test (2-3 minutes)
```bash
python main_demo.py --mode quick
```

### Full Demonstration (5-10 minutes)
```bash
python main_demo.py --mode full --save-results
```

## Core Architecture

### The Three-Module Architecture

The implementation is organized around three core modules, even though they're currently in the root directory rather than a `src/` structure:

1. **projections.py** - L¹-ball projection (the technical core innovation)
2. **cpnp_l1.py** - Main CPnP-ADMM algorithm with both L¹ and L² implementations
3. **pretrained.py** - Denoiser implementations (Plug-and-Play framework)

### ADMM Update Steps

The algorithm uses a three-step ADMM iteration:

**Step 1 - x-update (Plug-and-Play):**
```python
x^(k+1) = Denoiser(y - z^k + u^k)
```

**Step 2 - z-update (L¹-Ball Projection - THE KEY INNOVATION):**
```python
z^(k+1) = Proj_L1_Ball(y - x^(k+1) + u^k, ε)
```

**Step 3 - u-update (Dual Variable):**
```python
u^(k+1) = u^k + (y - x^(k+1) - z^(k+1))
```

### L¹-Ball Projection Algorithm

The core technical innovation is in `projections.py` using Duchi's algorithm:

1. Check if vector is already inside ball (early exit)
2. Store signs and work with absolute values
3. Sort absolute values in descending order
4. Find optimal threshold via cumulative sum
5. Apply soft thresholding with the threshold
6. Restore original signs and shape

**Key Insight:** L¹ constraints ignore outliers (impulse noise) whereas L² constraints average them out, causing blur.

## Code Structure Notes

### Current Directory Layout

The code files are currently flat in the root directory, not organized in a `src/` hierarchy:
- Root contains: `main_demo.py`, `cpnp_l1.py`, `projections.py`, `pretrained.py`
- The imports in code reference `src.algorithms.*` and `src.denoisers.*` but actual files are in root
- When importing, use: `from src.algorithms.projections import ...` (the main_demo.py shows this pattern)

### Module Dependencies

```
main_demo.py
  └─ imports from: src.algorithms.projections
  └─ imports from: src.algorithms.cpnp_l1
  └─ imports from: src.denoisers.pretrained

cpnp_l1.py (RobustCPnP class)
  └─ imports from: .projections (relative)
  └─ takes a denoiser callable as input

projections.py
  └─ standalone module with L¹ and L² projection implementations
```

### Key Classes and Functions

**RobustCPnP** (`cpnp_l1.py`):
- Main solver class
- Constructor: `RobustCPnP(denoiser, config)`
- Main method: `solve(y, epsilon, x_init=None)` returns `(x_restored, info)`
- Supports both 'l1' and 'l2' constraint types via `CPnPConfig`

**CPnPConfig** (`cpnp_l1.py`):
- Dataclass for algorithm configuration
- Key parameters: `constraint_type`, `rho`, `max_iter`, `tolerance`, `verbose`, `store_history`

**Denoiser Creation** (`pretrained.py`):
- Factory function: `create_denoiser(type, **kwargs)`
- Types: 'identity', 'gaussian', 'nlm', 'tv', 'bm3d', 'dncnn'
- All denoisers inherit from `BaseDenoiser` and are callable

**Comparison Helper** (`cpnp_l1.py`):
- Function: `compare_constraint_methods(y, epsilon, denoiser, config)`
- Returns dict with 'l1' and 'l2' keys, each containing (restored_image, info)

## Expected Experimental Results

### Gaussian Noise (Control Test)
- L¹ advantage: ~0-5% PSNR improvement
- Validates that L¹ robustness doesn't hurt standard performance

### Salt-and-Pepper Noise (Stress Test) ⭐
- L¹ advantage: ~10-30% PSNR improvement
- This is where L¹ constraints dramatically outperform L²
- Demonstrates core innovation: L² averaging fails on impulse noise, L¹ ignores outliers

### Performance Characteristics
- Runtime: L¹ projection is ~3-5x slower than L² (due to sorting in Duchi's algorithm)
- Convergence: Both methods converge reliably
- Memory: Similar footprint

## Important Implementation Details

### Convergence Monitoring

The algorithm tracks three key metrics:
1. **Primal residual**: `||y - x - z||₂` (should decrease to zero)
2. **Dual residual**: `ρ||z^(k+1) - z^k||₂` (should decrease to zero)
3. **Constraint violation**: `max(0, ||z||₁ - ε)` (should remain at zero)

### Epsilon Selection Guidelines

From the code:
- **Gaussian noise**: `ε = 2.0 * σ * sqrt(N)` where N is image size
- **Impulse noise**: `ε = 0.8 * density * N` where density is corruption fraction

### Testing and Validation

The `projections.py` module includes built-in tests:
```bash
python projections.py  # Runs correctness tests and benchmarks
```

Use `test_projection_correctness()` to validate the L¹ projection implementation.

## Optimization Techniques Demonstrated

This project demonstrates four key optimization techniques:

1. **Constrained Optimization** - Lagrange multipliers for ||·||₁ ≤ ε constraint
2. **Operator Splitting** - ADMM for non-convex problem decomposition
3. **Geometric Projections** - Exact L¹-ball projection (Duchi's algorithm)
4. **Implicit Regularization** - Plug-and-Play denoiser framework

## Academic Context

**Novel Contribution:** This extends Benfenati et al. (2024) by replacing L² with L¹ constraints for impulse robustness while maintaining ADMM convergence guarantees.

**Related Work:**
- Benfenati et al. (2024): Constrained PnP with L² constraints (baseline)
- Venkatakrishnan et al. (2013): Original Plug-and-Play ADMM
- Duchi et al. (2008): Efficient L¹-ball projection algorithm
- Boyd et al. (2011): ADMM theory

## Troubleshooting

### Module Import Issues
If you get import errors about `src.algorithms` or `src.denoisers`, ensure the repository structure matches what the imports expect, or modify imports to match the actual file locations.

### Missing Dependencies
Core dependencies: `numpy`, `scipy`, `matplotlib`, `scikit-image`
Optional: `torch`, `torchvision`, `bm3d`, `deepinv`

For basic testing without optional dependencies, use `create_denoiser('identity')` or `create_denoiser('gaussian')`.

### Memory Issues
For large images or memory constraints:
- Reduce `max_iter` in `CPnPConfig`
- Use smaller test images
- Use `verbose=False` to reduce output overhead

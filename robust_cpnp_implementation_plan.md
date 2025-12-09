# Robust Constrained Plug-and-Play ADMM Implementation Plan
## L¹-Ball Geometry for Blind Image Restoration

### 🎯 Project Overview
**Title:** Robust Automation: Blind Image Restoration via Constrained Plug-and-Play ADMM with L¹-Ball Geometry  
**Innovation:** Using L¹-ball constraints instead of L²-ball for superior robustness against impulse noise  
**Timeline:** 4 weeks (3 phases)  
**Expected Outcome:** Demonstrate superior performance on salt-and-pepper noise while maintaining Gaussian noise performance  

---

## 📋 Phase 1: Mathematical Foundation & Theory (Weeks 1-2)

### 1.1 Problem Formulation

**Constrained Optimization Problem:**
```
minimize   g(x)  [implicit via denoiser]
subject to ||y - x||₁ ≤ ε
```

Where:
- `x`: Clean image (unknown)
- `y`: Noisy observed image  
- `g(x)`: Implicit regularization via pre-trained denoiser
- `ε`: L¹-ball radius (noise tolerance)

**ADMM Formulation with Variable Splitting:**
```
minimize   g(x) + ι_C(z)
subject to y - x = z
```

Where:
- `ι_C(z)`: Indicator function of L¹-ball C = {z : ||z||₁ ≤ ε}
- `z`: Residual vector (auxiliary variable)

### 1.2 Augmented Lagrangian
```
L_ρ(x,z,u) = g(x) + ι_C(z) + u^T(y-x-z) + (ρ/2)||y-x-z||²₂
```

### 1.3 ADMM Update Steps

**Step 1: x-update (Plug-and-Play)**
```
x^(k+1) = argmin_x [g(x) + (ρ/2)||y-x-z^k+u^k||²₂]
        = Denoiser(y - z^k + u^k)
```

**Step 2: z-update (L¹-Ball Projection) - THE NOVELTY**
```
z^(k+1) = argmin_z [ι_C(z) + (ρ/2)||y-x^(k+1)-z+u^k||²₂]
        = Proj_L1_Ball(y - x^(k+1) + u^k, ε)
```

**Step 3: u-update (Dual Variable)**
```
u^(k+1) = u^k + (y - x^(k+1) - z^(k+1))
```

### 1.4 L¹-Ball Projection Algorithm (Duchi et al.)

**Core Innovation:** Efficient projection onto L¹-ball of radius ε

```python
def project_l1_ball(v, radius):
    """
    Project vector v onto L1-ball of given radius
    Solves: argmin_z ||z - v||₂² subject to ||z||₁ ≤ radius
    """
    # Algorithm:
    # 1. Take absolute values: u = |v|
    # 2. Sort in descending order
    # 3. Find threshold τ via cumulative sum
    # 4. Apply soft thresholding with τ
    # 5. Restore original signs
```

---

## 🔧 Phase 2: Implementation Architecture (Week 2-3)

### 2.1 Project Structure
```
robust_cpnp/
├── src/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── cpnp_l1.py          # Novel L¹-CPnP algorithm
│   │   ├── cpnp_l2.py          # Baseline L²-CPnP (Benfenati 2024)
│   │   ├── tv_admm.py          # Traditional TV-ADMM baseline
│   │   └── projections.py      # L¹ and L² ball projections
│   ├── denoisers/
│   │   ├── __init__.py
│   │   ├── pretrained.py       # Pre-trained network interface
│   │   └── classical.py        # Classical denoisers (NLM, BM3D)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py          # PSNR, SSIM, LPIPS
│   │   ├── noise.py            # Noise generation utilities
│   │   └── visualization.py    # Plotting and image display
│   └── experiments/
│       ├── __init__.py
│       ├── gaussian_comparison.py
│       ├── impulse_comparison.py
│       └── convergence_analysis.py
├── data/
│   ├── test_images/           # Standard test images
│   └── results/               # Output directory
├── notebooks/
│   ├── 01_theory_validation.ipynb
│   ├── 02_algorithm_comparison.ipynb
│   └── 03_results_analysis.ipynb
├── requirements.txt
└── README.md
```

### 2.2 Core Algorithm Implementation

#### 2.2.1 L¹-Ball Projection (Critical Component)

```python
import numpy as np
from typing import Tuple

def project_l1_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """
    Exact projection onto L1-ball using Duchi's algorithm
    
    Args:
        v: Input vector (can be multidimensional)
        radius: L1-ball radius
    
    Returns:
        Projected vector onto L1-ball
    """
    original_shape = v.shape
    v_flat = v.flatten()
    n = len(v_flat)
    
    # If already inside ball, return as-is
    if np.sum(np.abs(v_flat)) <= radius:
        return v
    
    # Store signs and work with absolute values
    signs = np.sign(v_flat)
    u = np.abs(v_flat)
    
    # Sort in descending order
    u_sorted = np.sort(u)[::-1]
    
    # Find threshold via cumulative sum
    cumsum = np.cumsum(u_sorted)
    threshold_candidates = (cumsum - radius) / (np.arange(n) + 1)
    
    # Find largest valid threshold
    valid_mask = threshold_candidates < u_sorted
    if np.any(valid_mask):
        k = np.where(valid_mask)[0][-1]
        threshold = threshold_candidates[k]
    else:
        threshold = 0.0
    
    # Apply soft thresholding
    projected = signs * np.maximum(u - threshold, 0)
    
    return projected.reshape(original_shape)

def verify_l1_projection(v: np.ndarray, projected: np.ndarray, radius: float) -> dict:
    """Verify projection correctness"""
    l1_norm = np.sum(np.abs(projected))
    distance = np.sum((projected - v)**2)
    
    return {
        'l1_norm': l1_norm,
        'constraint_satisfied': l1_norm <= radius + 1e-10,
        'distance_to_original': distance,
        'projection_valid': True  # Additional KKT checks can be added
    }
```

#### 2.2.2 Main CPnP-ADMM Algorithm

```python
class RobustCPnP:
    def __init__(self, denoiser, constraint_type='l1', rho=1.0):
        self.denoiser = denoiser
        self.constraint_type = constraint_type
        self.rho = rho
        self.history = {'residuals': [], 'objectives': [], 'constraint_violations': []}
    
    def solve(self, y: np.ndarray, epsilon: float, max_iter: int = 50, 
              tolerance: float = 1e-6) -> Tuple[np.ndarray, dict]:
        """
        Solve constrained image restoration problem
        
        Args:
            y: Noisy image
            epsilon: Constraint radius
            max_iter: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            x: Restored image
            info: Convergence information
        """
        # Initialize variables
        x = y.copy()
        z = np.zeros_like(y)
        u = np.zeros_like(y)
        
        for k in range(max_iter):
            # Store previous iterate for convergence check
            x_prev = x.copy()
            
            # 1. x-update: Plug-and-Play step
            v = y - z + u
            x = self.denoiser(v)
            
            # 2. z-update: Constraint projection (THE NOVELTY)
            residual = y - x + u
            if self.constraint_type == 'l1':
                z = project_l1_ball(residual, epsilon)
            elif self.constraint_type == 'l2':
                z = project_l2_ball(residual, epsilon)
            
            # 3. u-update: Dual variable
            u = u + (y - x - z)
            
            # Compute convergence metrics
            primal_residual = np.linalg.norm(y - x - z)
            dual_residual = self.rho * np.linalg.norm(z - z if k == 0 else z - z_prev)
            constraint_violation = max(0, np.sum(np.abs(z)) - epsilon)
            
            # Store history
            self.history['residuals'].append(primal_residual)
            self.history['constraint_violations'].append(constraint_violation)
            
            # Check convergence
            if (primal_residual < tolerance and 
                dual_residual < tolerance and
                constraint_violation < tolerance):
                break
            
            z_prev = z.copy()
        
        info = {
            'iterations': k + 1,
            'converged': k < max_iter - 1,
            'final_residual': primal_residual,
            'history': self.history
        }
        
        return x, info

def project_l2_ball(v: np.ndarray, radius: float) -> np.ndarray:
    """L2-ball projection for comparison"""
    norm = np.linalg.norm(v)
    if norm <= radius:
        return v
    return v * (radius / norm)
```

#### 2.2.3 Denoiser Interface

```python
class PretrainedDenoiser:
    def __init__(self, model_type='dncnn'):
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()
    
    def _load_model(self):
        """Load pre-trained denoising model"""
        if self.model_type == 'dncnn':
            # Use deepinv or download pretrained DnCNN
            try:
                import deepinv
                return deepinv.models.DnCNN()
            except ImportError:
                # Fallback to skimage
                return None
        
    def __call__(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply denoising"""
        if self.model is None:
            # Fallback to classical denoising
            from skimage.restoration import denoise_nl_means
            return denoise_nl_means(noisy_image, h=0.1, fast_mode=True)
        
        # Deep learning denoising
        # Convert to tensor, apply model, convert back
        # Implementation details depend on specific model
        pass
```

### 2.3 Validation Framework

```python
class ValidationSuite:
    def __init__(self):
        self.metrics = {}
    
    def run_convergence_tests(self, algorithm, test_cases):
        """Test ADMM convergence properties"""
        results = {}
        for case in test_cases:
            # Run algorithm
            x_restored, info = algorithm.solve(**case)
            
            # Check convergence
            results[case['name']] = {
                'converged': info['converged'],
                'iterations': info['iterations'],
                'final_residual': info['final_residual']
            }
        return results
    
    def run_constraint_validation(self, algorithm, epsilon_values):
        """Verify constraint satisfaction"""
        results = {}
        for eps in epsilon_values:
            # Test with known inputs
            results[eps] = self._check_constraint_satisfaction(algorithm, eps)
        return results
    
    def run_kkt_checks(self, x, z, u, y, epsilon):
        """Verify KKT conditions for optimality"""
        # Stationarity: gradient conditions
        # Primal feasibility: constraint satisfaction  
        # Dual feasibility: multiplier signs
        # Complementary slackness: active constraints
        pass
```

---

## 🧪 Phase 3: Experimental Design & Validation (Weeks 3-4)

### 3.1 Experimental Protocol

#### 3.1.1 Test Images
```python
TEST_IMAGES = {
    'standard': ['cameraman', 'lena', 'barbara', 'house'],
    'texture': ['fabric', 'wood', 'stone'],
    'edges': ['buildings', 'text', 'line_drawings']
}
```

#### 3.1.2 Noise Models
```python
NOISE_SCENARIOS = {
    'gaussian': {
        'sigma_values': [10, 20, 30, 40, 50],
        'description': 'Control test - both methods should work'
    },
    'salt_pepper': {
        'density_values': [0.05, 0.1, 0.15, 0.2, 0.3],
        'description': 'Stress test - L1 should outperform L2'
    },
    'mixed': {
        'combinations': [(10, 0.05), (20, 0.1), (30, 0.15)],
        'description': 'Real-world scenario'
    }
}
```

#### 3.1.3 Comparison Matrix
```python
ALGORITHMS = {
    'tv_admm': 'Traditional TV-ADMM baseline',
    'cpnp_l2': 'Benfenati 2024 method (L2 constraint)',
    'cpnp_l1': 'Our novel method (L1 constraint)',
    'cpnp_l1_adaptive': 'L1 with adaptive epsilon'
}
```

### 3.2 Performance Metrics

#### 3.2.1 Image Quality Metrics
```python
def compute_all_metrics(clean, restored):
    return {
        'psnr': peak_signal_noise_ratio(clean, restored),
        'ssim': structural_similarity(clean, restored),
        'lpips': learned_perceptual_distance(clean, restored),
        'rmse': np.sqrt(np.mean((clean - restored)**2)),
        'edge_preservation': compute_edge_preservation(clean, restored)
    }
```

#### 3.2.2 Optimization Metrics
```python
def compute_optimization_metrics(history, epsilon):
    return {
        'convergence_rate': estimate_convergence_rate(history['residuals']),
        'constraint_satisfaction': check_constraint_satisfaction(history, epsilon),
        'dual_gap': compute_duality_gap(history),
        'computational_cost': measure_runtime_complexity()
    }
```

### 3.3 Key Experimental Results (Expected)

#### 3.3.1 Gaussian Noise Results
- **Expected:** L¹ and L² CPnP perform similarly
- **Insight:** L¹ is robust enough for Gaussian noise
- **Validation:** Both outperform classical TV-ADMM

#### 3.3.2 Salt-and-Pepper Noise Results  
- **Expected:** L¹ CPnP significantly outperforms L² CPnP
- **Key insight:** L² constraint averages out outliers → blur
- **L¹ advantage:** Ignores outliers → preserves sharp edges

#### 3.3.3 Convergence Analysis
- **ADMM convergence:** Prove monotonic decrease in residuals
- **Constraint satisfaction:** Show ||z||₁ ≤ ε maintained
- **Computational complexity:** Compare iteration counts

---

## 📊 Implementation Timeline & Deliverables

### Week 1: Mathematical Foundation
- [ ] Derive complete ADMM formulation
- [ ] Implement L¹-ball projection algorithm  
- [ ] Validate projection with unit tests
- [ ] Theoretical convergence analysis

### Week 2: Core Implementation
- [ ] Build CPnP-ADMM framework
- [ ] Integrate pre-trained denoisers
- [ ] Implement L² baseline for comparison
- [ ] Create validation test suite

### Week 3: Experimental Framework
- [ ] Design comprehensive experiments
- [ ] Implement noise generation utilities
- [ ] Build evaluation metrics pipeline
- [ ] Run initial validation tests

### Week 4: Results & Analysis
- [ ] Execute full experimental protocol
- [ ] Generate comparison visualizations
- [ ] Analyze convergence properties
- [ ] Prepare final deliverables

### Final Deliverables
1. **Jupyter Notebook:** Complete implementation with comparisons
2. **Mathematical Appendix:** L¹-projection derivation and proofs
3. **Results Gallery:** Before/after images showing L¹ advantage
4. **Performance Report:** Quantitative metrics and analysis
5. **Code Repository:** Clean, documented implementation

---

## 🔍 Success Criteria

### Technical Success
- [ ] L¹-ball projection algorithm passes all validation tests
- [ ] ADMM convergence demonstrated for all test cases
- [ ] Constraint satisfaction verified (||z||₁ ≤ ε)
- [ ] Superior performance on salt-and-pepper noise proven

### Academic Success
- [ ] Clear optimization content satisfying course requirements
- [ ] Novel contribution beyond existing literature
- [ ] Rigorous experimental validation
- [ ] Professional presentation and documentation

### Innovation Impact
- [ ] Demonstrate practical advantage of L¹ geometry
- [ ] Provide efficient, implementable algorithm
- [ ] Show robustness across multiple noise scenarios
- [ ] Establish foundation for future robust PnP methods

---

## 🚀 Getting Started

1. **Set up environment:**
   ```bash
   pip install numpy scipy matplotlib torch torchvision
   pip install scikit-image deepinv
   pip install jupyter ipywidgets
   ```

2. **Download test images:**
   ```python
   from skimage import data
   test_images = {
       'cameraman': data.camera(),
       'astronaut': data.astronaut(),
       'coins': data.coins()
   }
   ```

3. **Implement core projection:**
   Start with the L¹-ball projection as it's the technical cornerstone

4. **Build incrementally:**
   Test each component before integration

This implementation plan provides a clear roadmap for creating a novel, technically sound, and experimentally validated contribution to image restoration using convex optimization.

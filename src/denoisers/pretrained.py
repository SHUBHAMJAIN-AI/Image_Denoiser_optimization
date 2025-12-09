"""
Denoiser Implementations for Plug-and-Play Framework
===================================================

This module provides various denoising algorithms that can be "plugged in" 
to the CPnP-ADMM framework. Includes both classical and deep learning methods.
"""

import numpy as np
from typing import Callable, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings

try:
    from skimage import restoration, filters
    from skimage.restoration import denoise_nl_means, denoise_tv_chambolle
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available. Classical denoisers will not work.")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    warnings.warn("PyTorch not available. Deep learning denoisers will not work.")

class BaseDenoiser(ABC):
    """Abstract base class for all denoisers"""
    
    @abstractmethod
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply denoising to input image"""
        pass
    
    def __call__(self, noisy_image: np.ndarray) -> np.ndarray:
        """Make denoiser callable"""
        return self.denoise(noisy_image)
    
    def validate_input(self, image: np.ndarray) -> np.ndarray:
        """Validate and normalize input image"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
        
        if image.ndim not in [2, 3]:
            raise ValueError("Input must be 2D (grayscale) or 3D (color) image")
        
        # Ensure float64 and [0,1] range
        if image.dtype != np.float64:
            if np.issubdtype(image.dtype, np.integer):
                image = image.astype(np.float64) / np.iinfo(image.dtype).max
            else:
                image = image.astype(np.float64)
        
        # Clip to [0,1] range
        image = np.clip(image, 0.0, 1.0)
        
        return image

class NonLocalMeansDenoiser(BaseDenoiser):
    """Non-Local Means denoising (Buades et al.)"""
    
    def __init__(self, h: float = 0.1, patch_size: int = 7, patch_distance: int = 11, 
                 fast_mode: bool = True):
        """
        Initialize Non-Local Means denoiser.
        
        Args:
            h: Filter strength. Higher h removes more noise but removes detail too
            patch_size: Size of patches used for comparison
            patch_distance: Maximum distance to search for patches
            fast_mode: Use fast approximate version
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for Non-Local Means denoising")
        
        self.h = h
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.fast_mode = fast_mode
    
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply Non-Local Means denoising"""
        image = self.validate_input(noisy_image)
        
        if image.ndim == 2:
            # Grayscale image
            denoised = denoise_nl_means(
                image, 
                h=self.h,
                fast_mode=self.fast_mode,
                patch_size=self.patch_size,
                patch_distance=self.patch_distance
            )
        else:
            # Color image - process each channel
            denoised = np.zeros_like(image)
            for i in range(image.shape[2]):
                denoised[:, :, i] = denoise_nl_means(
                    image[:, :, i],
                    h=self.h,
                    fast_mode=self.fast_mode,
                    patch_size=self.patch_size,
                    patch_distance=self.patch_distance
                )
        
        return np.clip(denoised, 0.0, 1.0)

class TVDenoiser(BaseDenoiser):
    """Total Variation denoising"""
    
    def __init__(self, weight: float = 0.1, max_iter: int = 50):
        """
        Initialize TV denoiser.
        
        Args:
            weight: Denoising weight (higher = more smoothing)
            max_iter: Maximum iterations
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for TV denoising")
        
        self.weight = weight
        self.max_iter = max_iter
    
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply Total Variation denoising"""
        image = self.validate_input(noisy_image)
        
        denoised = denoise_tv_chambolle(
            image,
            weight=self.weight,
            max_num_iter=self.max_iter
        )
        
        return np.clip(denoised, 0.0, 1.0)

class BM3DDenoiser(BaseDenoiser):
    """BM3D denoising (requires bm3d package)"""
    
    def __init__(self, sigma: float = 0.1):
        """
        Initialize BM3D denoiser.
        
        Args:
            sigma: Noise standard deviation
        """
        try:
            import bm3d
            self.bm3d = bm3d
        except ImportError:
            raise ImportError("bm3d package required. Install with: pip install bm3d")
        
        self.sigma = sigma
    
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply BM3D denoising"""
        image = self.validate_input(noisy_image)
        
        # BM3D expects [0,1] float32
        image_32 = image.astype(np.float32)
        
        if image.ndim == 2:
            denoised = self.bm3d.bm3d(image_32, self.sigma)
        else:
            # Color image
            denoised = self.bm3d.bm3d(image_32, self.sigma)
        
        return np.clip(denoised.astype(np.float64), 0.0, 1.0)

class GaussianDenoiser(BaseDenoiser):
    """Simple Gaussian filtering denoiser"""
    
    def __init__(self, sigma: float = 1.0):
        """
        Initialize Gaussian denoiser.
        
        Args:
            sigma: Gaussian kernel standard deviation
        """
        if not SKIMAGE_AVAILABLE:
            raise ImportError("scikit-image required for Gaussian denoising")
        
        self.sigma = sigma
    
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Apply Gaussian filtering"""
        image = self.validate_input(noisy_image)
        
        denoised = filters.gaussian(image, sigma=self.sigma)
        return np.clip(denoised, 0.0, 1.0)

class IdentityDenoiser(BaseDenoiser):
    """Identity denoiser (no denoising) for testing"""
    
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """Return input unchanged"""
        return self.validate_input(noisy_image)

if TORCH_AVAILABLE:
    class DnCNNDenoiser(BaseDenoiser):
        """DnCNN deep learning denoiser"""
        
        def __init__(self, model_path: Optional[str] = None, device: str = 'auto'):
            """
            Initialize DnCNN denoiser.
            
            Args:
                model_path: Path to pre-trained model (None for built-in)
                device: Device to use ('cpu', 'cuda', or 'auto')
            """
            if device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device
            
            self.model = self._load_model(model_path)
            self.model.eval()
        
        def _load_model(self, model_path: Optional[str]) -> nn.Module:
            """Load DnCNN model"""
            try:
                # Try to use deepinv library
                import deepinv
                model = deepinv.models.DnCNN(depth=17, n_channels=64)
                
                if model_path:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                
                return model.to(self.device)
                
            except ImportError:
                # Fallback: create simple CNN architecture
                print("deepinv not available, using simplified CNN")
                return self._create_simple_cnn()
        
        def _create_simple_cnn(self) -> nn.Module:
            """Create simplified CNN for denoising"""
            class SimpleDnCNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
                    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                    self.conv3 = nn.Conv2d(64, 1, 3, padding=1)
                    self.relu = nn.ReLU(inplace=True)
                
                def forward(self, x):
                    residual = x
                    out = self.relu(self.conv1(x))
                    out = self.relu(self.conv2(out))
                    out = self.conv3(out)
                    return residual - out  # Residual learning
            
            model = SimpleDnCNN()
            # Initialize with small random weights
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, 0, 0.01)
            
            return model.to(self.device)
        
        def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
            """Apply DnCNN denoising"""
            image = self.validate_input(noisy_image)
            
            # Handle color images by processing each channel
            if image.ndim == 3:
                denoised = np.zeros_like(image)
                for c in range(image.shape[2]):
                    denoised[:, :, c] = self._denoise_channel(image[:, :, c])
                return denoised
            else:
                return self._denoise_channel(image)
        
        def _denoise_channel(self, channel: np.ndarray) -> np.ndarray:
            """Denoise single channel"""
            # Convert to torch tensor
            input_tensor = torch.from_numpy(channel).float().unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                denoised = output.squeeze().cpu().numpy()
            
            return np.clip(denoised, 0.0, 1.0)
else:
    # Dummy class when PyTorch is not available
    class DnCNNDenoiser(BaseDenoiser):
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Cannot use DnCNN denoiser.")
        
        def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
            raise ImportError("PyTorch not available. Cannot use DnCNN denoiser.")

# Factory functions for easy denoiser creation
def create_denoiser(denoiser_type: str, **kwargs) -> BaseDenoiser:
    """
    Factory function to create denoisers.
    
    Args:
        denoiser_type: Type of denoiser ('nlm', 'tv', 'bm3d', 'gaussian', 'dncnn', 'identity')
        **kwargs: Arguments for specific denoiser
        
    Returns:
        Configured denoiser instance
    """
    denoiser_map = {
        'nlm': NonLocalMeansDenoiser,
        'nonlocal': NonLocalMeansDenoiser,
        'tv': TVDenoiser,
        'bm3d': BM3DDenoiser,
        'gaussian': GaussianDenoiser,
        'identity': IdentityDenoiser
    }
    
    # Only add DnCNN if PyTorch is available
    if TORCH_AVAILABLE:
        denoiser_map['dncnn'] = DnCNNDenoiser
    
    if denoiser_type.lower() not in denoiser_map:
        available = list(denoiser_map.keys())
        raise ValueError(f"Unknown denoiser type '{denoiser_type}'. Available: {available}")
    
    DenoiserClass = denoiser_map[denoiser_type.lower()]
    return DenoiserClass(**kwargs)

def get_default_denoiser() -> BaseDenoiser:
    """Get a good default denoiser for general use"""
    try:
        # Try Non-Local Means first (good general purpose)
        return NonLocalMeansDenoiser(h=0.08, fast_mode=True)
    except ImportError:
        try:
            # Fallback to TV denoising
            return TVDenoiser(weight=0.1)
        except ImportError:
            # Last resort - identity
            warnings.warn("No denoising libraries available, using identity denoiser")
            return IdentityDenoiser()

# Adaptive denoiser selection based on noise characteristics
def adaptive_denoiser_selection(noisy_image: np.ndarray, 
                              noise_type: str = 'auto') -> BaseDenoiser:
    """
    Select appropriate denoiser based on noise characteristics.
    
    Args:
        noisy_image: Noisy input image
        noise_type: Type of noise ('gaussian', 'impulse', 'mixed', 'auto')
        
    Returns:
        Appropriate denoiser for the noise type
    """
    if noise_type == 'auto':
        # Simple heuristic for noise type detection
        # Could be improved with more sophisticated analysis
        gradient_norm = np.mean(np.abs(np.gradient(noisy_image)))
        local_variance = np.var(noisy_image)
        
        if gradient_norm > 0.1:  # High gradient suggests impulse noise
            noise_type = 'impulse'
        elif local_variance > 0.05:  # High variance suggests Gaussian
            noise_type = 'gaussian'
        else:
            noise_type = 'mixed'
    
    if noise_type == 'gaussian':
        # BM3D is excellent for Gaussian noise
        try:
            return BM3DDenoiser(sigma=0.1)
        except ImportError:
            return NonLocalMeansDenoiser(h=0.08)
    
    elif noise_type == 'impulse':
        # Non-Local Means handles impulse noise well
        return NonLocalMeansDenoiser(h=0.12, fast_mode=True)
    
    elif noise_type == 'mixed':
        # TV denoising is robust to various noise types
        try:
            return TVDenoiser(weight=0.15)
        except ImportError:
            return NonLocalMeansDenoiser(h=0.1)
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

# Testing and validation utilities
def test_denoiser_performance(denoiser: BaseDenoiser, 
                            test_images: Dict[str, np.ndarray],
                            noise_params: Dict[str, Any]) -> Dict[str, float]:
    """Test denoiser performance on various images and noise levels"""
    results = {}
    
    for image_name, clean_image in test_images.items():
        # Add noise
        noisy_image = add_noise(clean_image, **noise_params)
        
        # Denoise
        denoised = denoiser.denoise(noisy_image)
        
        # Compute metrics
        psnr = compute_psnr(clean_image, denoised)
        ssim = compute_ssim(clean_image, denoised)
        
        results[f"{image_name}_psnr"] = psnr
        results[f"{image_name}_ssim"] = ssim
    
    return results

def add_noise(image: np.ndarray, noise_type: str = 'gaussian', **params) -> np.ndarray:
    """Add noise to clean image for testing"""
    if noise_type == 'gaussian':
        sigma = params.get('sigma', 0.1)
        noise = np.random.normal(0, sigma, image.shape)
        return np.clip(image + noise, 0, 1)
    
    elif noise_type == 'salt_pepper':
        density = params.get('density', 0.1)
        noisy = image.copy()
        # Salt noise
        salt_mask = np.random.random(image.shape) < density/2
        noisy[salt_mask] = 1.0
        # Pepper noise  
        pepper_mask = np.random.random(image.shape) < density/2
        noisy[pepper_mask] = 0.0
        return noisy
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

def compute_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio"""
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Compute Structural Similarity Index"""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(image1, image2, data_range=1.0)
    except ImportError:
        # Simplified SSIM approximation
        mu1, mu2 = np.mean(image1), np.mean(image2)
        sigma1, sigma2 = np.var(image1), np.var(image2)
        sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim

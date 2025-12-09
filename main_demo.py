"""
Main Execution Script for Robust CPnP Implementation
=================================================

This script demonstrates the complete implementation of the Robust Constrained
Plug-and-Play ADMM algorithm with L¹-ball constraints for impulse noise robustness.

Usage:
    python main_demo.py [--mode {quick|full}] [--save-results]
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from pathlib import Path
import json

# Import our algorithms
from src.algorithms.projections import project_l1_ball, project_l2_ball, test_projection_correctness
from src.algorithms.cpnp_l1 import RobustCPnP, CPnPConfig, compare_constraint_methods
from src.denoisers.pretrained import create_denoiser

class RobustCPnPDemo:
    """Main demonstration class for Robust CPnP algorithm"""
    
    def __init__(self, save_results=True):
        self.save_results = save_results
        self.output_dir = Path("demo_results")
        if save_results:
            self.output_dir.mkdir(exist_ok=True)
    
    def run_complete_demo(self):
        """Run complete demonstration of the Robust CPnP algorithm"""
        print("🚀 ROBUST CPNP ALGORITHM DEMONSTRATION")
        print("=" * 60)
        print("Comparing L¹ vs L² constraints for image denoising")
        print("Key Innovation: L¹ constraints for impulse noise robustness\n")
        
        # Phase 1: Algorithm Validation
        print("📋 PHASE 1: Algorithm Validation")
        print("-" * 40)
        self.validate_core_algorithms()
        
        # Phase 2: The Key Experiment - Noise Robustness
        print("\n🎯 PHASE 2: Noise Robustness Analysis (KEY EXPERIMENT)")
        print("-" * 40)
        gaussian_results = self.test_gaussian_noise()
        impulse_results = self.test_impulse_noise()
        
        # Phase 3: Performance Analysis
        print("\n📊 PHASE 3: Performance Analysis") 
        print("-" * 40)
        self.analyze_performance(gaussian_results, impulse_results)
        
        # Phase 4: Convergence Study
        print("\n📈 PHASE 4: Convergence Analysis")
        print("-" * 40)
        convergence_results = self.analyze_convergence()
        
        print("\n🎉 DEMONSTRATION COMPLETED!")
        self.print_final_summary(gaussian_results, impulse_results)
        
        return {
            'gaussian_results': gaussian_results,
            'impulse_results': impulse_results,
            'convergence_results': convergence_results
        }
    
    def validate_core_algorithms(self):
        """Validate core algorithmic components"""
        print("  Testing L¹-ball projection algorithm...")
        
        try:
            # Test projection correctness
            test_projection_correctness()
            print("    ✅ L¹-ball projection: All tests passed")
            
            # Test constraint satisfaction
            test_vectors = [np.random.randn(100), np.random.randn(50)]
            epsilon_values = [0.5, 1.0, 2.0]
            
            all_satisfied = True
            for v in test_vectors:
                for eps in epsilon_values:
                    proj = project_l1_ball(v, eps)
                    l1_norm = np.sum(np.abs(proj))
                    if l1_norm > eps + 1e-10:
                        all_satisfied = False
                        break
            
            if all_satisfied:
                print("    ✅ Constraint satisfaction: All tests passed")
            else:
                print("    ❌ Constraint satisfaction: Some tests failed")
            
            print("  Testing denoiser interface...")
            denoiser = create_denoiser('identity')
            test_image = np.random.rand(32, 32)
            result = denoiser(test_image)
            
            if result.shape == test_image.shape:
                print("    ✅ Denoiser interface: Working correctly")
            else:
                print("    ❌ Denoiser interface: Shape mismatch")
        
        except Exception as e:
            print(f"    ❌ Validation error: {e}")
    
    def test_gaussian_noise(self):
        """Test algorithm performance on Gaussian noise (control experiment)"""
        print("  🔬 Control Test: Gaussian Noise")
        print("    Expected: Both L¹ and L² should perform similarly")
        
        # Generate test image
        clean_image = self.generate_test_image()
        
        # Add Gaussian noise
        sigma = 0.15
        noise = np.random.normal(0, sigma, clean_image.shape)
        noisy_image = np.clip(clean_image + noise, 0, 1)
        
        # Create denoiser
        denoiser = create_denoiser('gaussian', sigma=1.0)
        
        # Set epsilon based on noise level
        epsilon = 2.0 * sigma * np.sqrt(clean_image.size)
        
        print(f"    Noise level: σ = {sigma}")
        print(f"    Constraint radius: ε = {epsilon:.2f}")
        
        # Compare L1 vs L2 methods
        config = CPnPConfig(max_iter=30, verbose=False, tolerance=1e-5)
        results = compare_constraint_methods(noisy_image, epsilon, denoiser, config)
        
        # Compute metrics
        metrics = {}
        for method, (restored, info) in results.items():
            psnr = self.compute_psnr(clean_image, restored)
            ssim = self.compute_ssim(clean_image, restored)
            
            metrics[method] = {
                'psnr': psnr,
                'ssim': ssim,
                'runtime': info.runtime,
                'iterations': info.iterations,
                'converged': info.converged
            }
            
            print(f"    {method.upper()}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.3f}, "
                  f"Runtime = {info.runtime:.3f}s, Iterations = {info.iterations}")
        
        # Save results if requested
        if self.save_results:
            self.save_test_images(clean_image, noisy_image, results, "gaussian")
        
        return {
            'clean_image': clean_image,
            'noisy_image': noisy_image,
            'results': results,
            'metrics': metrics,
            'noise_params': {'type': 'gaussian', 'sigma': sigma},
            'epsilon': epsilon
        }
    
    def test_impulse_noise(self):
        """Test algorithm performance on impulse noise (stress test)"""
        print("\n  🎯 Stress Test: Salt-and-Pepper Noise")
        print("    Expected: L¹ should significantly outperform L²")
        
        # Generate test image  
        clean_image = self.generate_test_image()
        
        # Add salt-and-pepper noise
        density = 0.1  # 10% of pixels corrupted
        noisy_image = clean_image.copy()
        
        # Salt noise (white pixels)
        salt_coords = np.random.random(clean_image.shape) < density/2
        noisy_image[salt_coords] = 1.0
        
        # Pepper noise (black pixels)
        pepper_coords = np.random.random(clean_image.shape) < density/2
        noisy_image[pepper_coords] = 0.0
        
        # Create denoiser
        denoiser = create_denoiser('gaussian', sigma=1.0)
        
        # Set epsilon based on impulse noise characteristics
        epsilon = 0.8 * density * clean_image.size
        
        print(f"    Noise density: {density*100}% pixels corrupted")
        print(f"    Constraint radius: ε = {epsilon:.2f}")
        
        # Compare L1 vs L2 methods
        config = CPnPConfig(max_iter=30, verbose=False, tolerance=1e-5)
        results = compare_constraint_methods(noisy_image, epsilon, denoiser, config)
        
        # Compute metrics
        metrics = {}
        for method, (restored, info) in results.items():
            psnr = self.compute_psnr(clean_image, restored)
            ssim = self.compute_ssim(clean_image, restored)
            
            metrics[method] = {
                'psnr': psnr,
                'ssim': ssim,
                'runtime': info.runtime,
                'iterations': info.iterations,
                'converged': info.converged
            }
            
            print(f"    {method.upper()}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.3f}, "
                  f"Runtime = {info.runtime:.3f}s, Iterations = {info.iterations}")
        
        # Save results if requested
        if self.save_results:
            self.save_test_images(clean_image, noisy_image, results, "impulse")
        
        return {
            'clean_image': clean_image,
            'noisy_image': noisy_image,
            'results': results,
            'metrics': metrics,
            'noise_params': {'type': 'impulse', 'density': density},
            'epsilon': epsilon
        }
    
    def analyze_performance(self, gaussian_results, impulse_results):
        """Analyze and compare performance across noise types"""
        print("  Comparing L¹ vs L² performance...")
        
        # Gaussian noise comparison
        g_l1_psnr = gaussian_results['metrics']['l1']['psnr']
        g_l2_psnr = gaussian_results['metrics']['l2']['psnr']
        g_advantage = (g_l1_psnr - g_l2_psnr) / g_l2_psnr * 100
        
        print(f"    Gaussian Noise - L¹ advantage: {g_advantage:+.1f}% PSNR")
        
        # Impulse noise comparison
        i_l1_psnr = impulse_results['metrics']['l1']['psnr']
        i_l2_psnr = impulse_results['metrics']['l2']['psnr']
        i_advantage = (i_l1_psnr - i_l2_psnr) / i_l2_psnr * 100
        
        print(f"    Impulse Noise - L¹ advantage: {i_advantage:+.1f}% PSNR")
        
        # Runtime comparison
        g_l1_time = gaussian_results['metrics']['l1']['runtime']
        g_l2_time = gaussian_results['metrics']['l2']['runtime']
        time_ratio_g = g_l1_time / g_l2_time
        
        i_l1_time = impulse_results['metrics']['l1']['runtime']
        i_l2_time = impulse_results['metrics']['l2']['runtime']
        time_ratio_i = i_l1_time / i_l2_time
        
        print(f"    Runtime ratio (L¹/L²): Gaussian = {time_ratio_g:.1f}x, Impulse = {time_ratio_i:.1f}x")
        
        # Key validation check
        if i_advantage > g_advantage + 5:  # At least 5% better advantage for impulse
            print("    ✅ VALIDATION PASSED: L¹ shows superior robustness to impulse noise")
        else:
            print("    ⚠️  VALIDATION INCONCLUSIVE: Advantage not clearly demonstrated")
    
    def analyze_convergence(self):
        """Analyze ADMM convergence properties"""
        print("  Testing convergence for different penalty parameters...")
        
        # Generate test problem
        clean_image = self.generate_test_image(size=(64, 64))  # Smaller for faster convergence
        noisy_image = clean_image + 0.1 * np.random.randn(*clean_image.shape)
        denoiser = create_denoiser('identity')  # Simple denoiser for analysis
        
        epsilon = 0.5
        rho_values = [0.5, 1.0, 2.0]
        results = {}
        
        for rho in rho_values:
            config = CPnPConfig(
                constraint_type='l1',
                rho=rho,
                max_iter=50,
                tolerance=1e-8,
                verbose=False,
                store_history=True
            )
            
            solver = RobustCPnP(denoiser, config)
            start_time = time.perf_counter()
            x_restored, info = solver.solve(noisy_image, epsilon)
            
            results[rho] = {
                'converged': info.converged,
                'iterations': info.iterations,
                'final_residual': info.final_primal_residual,
                'runtime': info.runtime,
                'history': info.history
            }
            
            print(f"    ρ = {rho}: {info.iterations} iterations, "
                  f"residual = {info.final_primal_residual:.2e}, "
                  f"converged = {info.converged}")
        
        return results
    
    def generate_test_image(self, size=(128, 128)):
        """Generate synthetic test image for experiments"""
        try:
            # Try to use skimage if available
            from skimage import data
            img = data.camera()
            if img.shape[:2] != size:
                from skimage.transform import resize
                img = resize(img, size)
            return img.astype(np.float64) / 255.0
        except ImportError:
            # Generate synthetic image
            H, W = size
            x, y = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H))
            
            # Create interesting pattern with edges
            image = 0.3 * (np.sin(3*x) * np.cos(3*y)) + 0.5
            image += 0.3 * np.exp(-((x-0.5)**2 + (y-0.5)**2) * 4)
            image += 0.2 * np.exp(-((x+0.5)**2 + (y+0.5)**2) * 6)
            
            return np.clip(image, 0, 1)
    
    def compute_psnr(self, image1, image2):
        """Compute Peak Signal-to-Noise Ratio"""
        mse = np.mean((image1 - image2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    def compute_ssim(self, image1, image2):
        """Compute Structural Similarity Index (simplified version)"""
        mu1, mu2 = np.mean(image1), np.mean(image2)
        sigma1, sigma2 = np.var(image1), np.var(image2)
        sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
        
        c1, c2 = (0.01)**2, (0.03)**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        return max(0, min(1, ssim))
    
    def save_test_images(self, clean, noisy, results, noise_type):
        """Save test images for visual comparison"""
        try:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Clean image
            axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Clean Image')
            axes[0].axis('off')
            
            # Noisy image
            axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title(f'Noisy ({noise_type.title()})')
            axes[1].axis('off')
            
            # L2 result
            l2_restored, _ = results['l2']
            axes[2].imshow(l2_restored, cmap='gray', vmin=0, vmax=1)
            l2_psnr = self.compute_psnr(clean, l2_restored)
            axes[2].set_title(f'L² CPnP\nPSNR: {l2_psnr:.1f} dB')
            axes[2].axis('off')
            
            # L1 result
            l1_restored, _ = results['l1']
            axes[3].imshow(l1_restored, cmap='gray', vmin=0, vmax=1)
            l1_psnr = self.compute_psnr(clean, l1_restored)
            axes[3].set_title(f'L¹ CPnP (Novel)\nPSNR: {l1_psnr:.1f} dB')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'comparison_{noise_type}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    💾 Saved comparison images to {self.output_dir}/comparison_{noise_type}.png")
        
        except ImportError:
            print("    ⚠️  Matplotlib not available - skipping image saving")
    
    def print_final_summary(self, gaussian_results, impulse_results):
        """Print final summary of results"""
        print("\n📋 FINAL SUMMARY")
        print("=" * 50)
        
        # Extract key metrics
        g_l1_psnr = gaussian_results['metrics']['l1']['psnr']
        g_l2_psnr = gaussian_results['metrics']['l2']['psnr']
        g_advantage = (g_l1_psnr - g_l2_psnr) / g_l2_psnr * 100
        
        i_l1_psnr = impulse_results['metrics']['l1']['psnr']
        i_l2_psnr = impulse_results['metrics']['l2']['psnr']
        i_advantage = (i_l1_psnr - i_l2_psnr) / i_l2_psnr * 100
        
        print("🔬 EXPERIMENTAL VALIDATION:")
        print(f"   Gaussian Noise: L¹ vs L² advantage = {g_advantage:+.1f}% PSNR")
        print(f"   Impulse Noise:  L¹ vs L² advantage = {i_advantage:+.1f}% PSNR")
        
        if i_advantage > g_advantage + 3:
            print("\n✅ HYPOTHESIS CONFIRMED:")
            print("   L¹ constraints provide superior robustness to impulse noise")
            print("   while maintaining competitive performance on Gaussian noise.")
        else:
            print("\n⚠️  RESULTS INCONCLUSIVE:")
            print("   May need parameter tuning or different test conditions.")
        
        print("\n🎯 KEY INNOVATION:")
        print("   Novel use of L¹-ball constraints in CPnP-ADMM framework")
        print("   enables robust handling of non-Gaussian impulse noise.")
        
        print("\n🔧 OPTIMIZATION TECHNIQUES DEMONSTRATED:")
        print("   1. Constrained optimization with Lagrange multipliers")
        print("   2. ADMM operator splitting for non-convex problems")
        print("   3. Geometric projections onto L¹-ball") 
        print("   4. Plug-and-Play implicit regularization")
        
        if self.save_results:
            print(f"\n💾 Results saved to: {self.output_dir.absolute()}")

def quick_test():
    """Run a quick test to verify the implementation works"""
    print("🚀 QUICK IMPLEMENTATION TEST")
    print("=" * 40)
    
    try:
        # Test basic components
        print("Testing core algorithms...")
        
        # Test projection
        v = np.array([1, 2, -1, -2])
        proj = project_l1_ball(v, 3.0)
        print(f"  L¹ projection: ||{proj}||₁ = {np.sum(np.abs(proj)):.3f}")
        
        # Test denoiser
        denoiser = create_denoiser('identity')
        test_img = np.random.rand(32, 32)
        denoised = denoiser(test_img)
        print(f"  Denoiser: {test_img.shape} -> {denoised.shape}")
        
        # Test CPnP solver
        config = CPnPConfig(max_iter=3, verbose=False)
        solver = RobustCPnP(denoiser, config)
        noisy = test_img + 0.1 * np.random.randn(*test_img.shape)
        restored, info = solver.solve(noisy, 0.5)
        print(f"  CPnP solver: {info.iterations} iterations, converged: {info.converged}")
        
        print("\n✅ All components working! Ready for full demonstration.")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check dependencies and implementation.")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Robust CPnP Demonstration')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Run quick test or full demonstration')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save results and images')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        success = quick_test()
        if success:
            print("\n🎉 Implementation verified! Use --mode full for complete demo.")
    else:
        demo = RobustCPnPDemo(save_results=args.save_results)
        results = demo.run_complete_demo()
        
        if args.save_results:
            # Save detailed results
            timestamp = int(time.time())
            results_file = demo.output_dir / f'detailed_results_{timestamp}.json'
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_results = convert_for_json({
                'summary': 'Robust CPnP Demonstration Results',
                'timestamp': timestamp,
                'gaussian_metrics': results['gaussian_results']['metrics'],
                'impulse_metrics': results['impulse_results']['metrics']
            })
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n💾 Detailed results saved to {results_file}")

if __name__ == "__main__":
    main()

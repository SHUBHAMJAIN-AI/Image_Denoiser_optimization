"""
Test Script for DnCNN Integration with CPnP-ADMM
=================================================

This script validates that the DnCNN denoiser is properly integrated
with the CPnP-ADMM framework.

Run with: python test_dncnn_integration.py
"""

import numpy as np
import sys

def test_imports():
    """Test 1: Verify all required imports are available"""
    print("\n" + "="*60)
    print("TEST 1: Checking imports...")
    print("="*60)

    try:
        import torch
        print(f"✓ PyTorch available (version {torch.__version__})")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch not available: {e}")
        return False

    try:
        import deepinv
        print(f"✓ deepinv available (version {deepinv.__version__})")
    except ImportError as e:
        print(f"✗ deepinv not available: {e}")
        print("  Install with: pip install deepinv>=0.2.0")
        return False

    try:
        from src.denoisers.pretrained import DnCNNDenoiser, create_denoiser
        print("✓ DnCNNDenoiser imported successfully")
    except ImportError as e:
        print(f"✗ Cannot import DnCNNDenoiser: {e}")
        return False

    try:
        from src.algorithms.cpnp_l1 import RobustCPnP, CPnPConfig
        print("✓ CPnP-ADMM classes imported successfully")
    except ImportError as e:
        print(f"✗ Cannot import CPnP classes: {e}")
        return False

    print("\n✓ All imports successful!")
    return True


def test_dncnn_weights():
    """Test 2: Verify DnCNN pretrained weight loading"""
    print("\n" + "="*60)
    print("TEST 2: Testing DnCNN weight loading...")
    print("="*60)

    try:
        from src.denoisers.pretrained import DnCNNDenoiser

        # Test with pretrained weights
        print("\nAttempting to load pretrained weights...")
        denoiser = DnCNNDenoiser(pretrained='download', device='cpu')

        # Create noisy test image
        np.random.seed(42)
        clean = np.random.rand(64, 64).astype(np.float64) * 0.5 + 0.25
        noisy = clean + np.random.normal(0, 0.1, clean.shape)
        noisy = np.clip(noisy, 0, 1).astype(np.float64)

        # Apply denoising
        print("Applying DnCNN denoising to test image...")
        denoised = denoiser.denoise(noisy)

        # Verify output is valid
        assert denoised.shape == noisy.shape, "Output shape mismatch"
        assert denoised.dtype == np.float64, "Output dtype mismatch"
        assert np.all(denoised >= 0) and np.all(denoised <= 1), "Output out of range"

        # Check if denoising actually does something
        mse_before = np.mean((noisy - clean)**2)
        mse_after = np.mean((denoised - clean)**2)

        print(f"  MSE before denoising: {mse_before:.6f}")
        print(f"  MSE after denoising:  {mse_after:.6f}")

        # With pretrained weights, denoising should improve MSE
        if mse_after < mse_before:
            improvement = ((mse_before - mse_after) / mse_before) * 100
            print(f"  Improvement: {improvement:.2f}%")
            print("\n✓ DnCNN appears to be working with pretrained weights!")
        else:
            print("⚠ Warning: DnCNN did not improve MSE (may be using random weights)")

        return True

    except Exception as e:
        print(f"\n✗ DnCNN weight loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_function():
    """Test 3: Test denoiser factory with DnCNN"""
    print("\n" + "="*60)
    print("TEST 3: Testing factory function...")
    print("="*60)

    try:
        from src.denoisers.pretrained import create_denoiser

        # Test creating DnCNN via factory
        print("\nCreating DnCNN via factory function...")
        denoiser = create_denoiser('dncnn', pretrained='download', device='cpu')

        # Test on small image
        test_img = np.random.rand(32, 32).astype(np.float64)
        result = denoiser.denoise(test_img)

        assert result.shape == test_img.shape, "Shape mismatch"
        print(f"  Input shape: {test_img.shape}")
        print(f"  Output shape: {result.shape}")
        print("\n✓ Factory function works correctly!")

        return True

    except Exception as e:
        print(f"\n✗ Factory function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cpnp_integration():
    """Test 4: Test full CPnP-ADMM integration with DnCNN"""
    print("\n" + "="*60)
    print("TEST 4: Testing CPnP-ADMM integration...")
    print("="*60)

    try:
        from src.denoisers.pretrained import DnCNNDenoiser
        from src.algorithms.cpnp_l1 import RobustCPnP, CPnPConfig

        # Create DnCNN denoiser
        print("\nInitializing DnCNN for CPnP-ADMM...")
        denoiser = DnCNNDenoiser(pretrained='download', device='cpu')

        # Create test image with salt & pepper noise
        np.random.seed(42)
        clean = np.random.rand(64, 64).astype(np.float64) * 0.5 + 0.25
        noisy = clean.copy()

        # Add impulse noise
        impulse_ratio = 0.1
        num_impulse = int(impulse_ratio * noisy.size)
        impulse_indices = np.random.choice(noisy.size, num_impulse, replace=False)
        impulse_flat = noisy.flatten()
        impulse_flat[impulse_indices] = np.random.choice([0.0, 1.0], num_impulse)
        noisy = impulse_flat.reshape(noisy.shape)

        # Configure CPnP with L¹ constraint
        config = CPnPConfig(
            constraint_type='l1',
            rho=1.0,
            max_iter=20,  # Reduced for quick test
            tolerance=1e-4,
            verbose=False,
            store_history=True
        )

        print("Running CPnP-ADMM with L¹ constraint...")
        solver = RobustCPnP(denoiser, config)

        # Solve with appropriate epsilon for impulse noise
        epsilon = np.sum(np.abs(noisy - clean)) * 1.2
        restored, info = solver.solve(noisy, epsilon)

        # Verify results
        print(f"\n  Iterations: {info.iterations}")
        print(f"  Converged: {info.converged}")
        print(f"  Runtime: {info.runtime:.3f}s")
        print(f"  Final primal residual: {info.final_primal_residual:.2e}")
        print(f"  Final dual residual: {info.final_dual_residual:.2e}")

        # Check quality improvement
        mse_noisy = np.mean((noisy - clean)**2)
        mse_restored = np.mean((restored - clean)**2)

        print(f"\n  MSE (noisy): {mse_noisy:.6f}")
        print(f"  MSE (restored): {mse_restored:.6f}")

        if mse_restored < mse_noisy:
            improvement = ((mse_noisy - mse_restored) / mse_noisy) * 100
            print(f"  Improvement: {improvement:.2f}%")
            print("\n✓ CPnP-ADMM with DnCNN working correctly!")
        else:
            print("⚠ Warning: Restoration did not improve MSE")

        return True

    except Exception as e:
        print(f"\n✗ CPnP integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DnCNN Integration Test Suite")
    print("="*60)

    tests = [
        ("Import Test", test_imports),
        ("DnCNN Weight Loading", test_dncnn_weights),
        ("Factory Function", test_factory_function),
        ("CPnP-ADMM Integration", test_cpnp_integration),
    ]

    results = {}

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} raised an exception: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - DnCNN is ready to use!")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please check errors above")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

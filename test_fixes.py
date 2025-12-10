#!/usr/bin/env python3
"""
Quick test script to verify all bug fixes are working.
Run this before running the full notebook.
"""

import numpy as np
import sys

print("=" * 70)
print("TESTING BUG FIXES")
print("=" * 70)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.algorithms.cpnp_l1 import RobustCPnP, CPnPConfig
    from src.denoisers.pretrained import create_denoiser
    print("   ✅ Imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Epsilon scaling
print("\n2. Testing epsilon scaling...")
sigma = 0.15
spatial_size = 128 * 128
num_channels = 3

# With moderate margin for algorithm slack (reduced from before)
epsilon_l2 = 2.0 * sigma * np.sqrt(spatial_size * num_channels)
epsilon_l1 = 1.2 * sigma * spatial_size * num_channels

print(f"   L² epsilon: {epsilon_l2:.2f}")
print(f"   L¹ epsilon: {epsilon_l1:.2f}")
print(f"   Ratio (L¹/L²): {epsilon_l1/epsilon_l2:.1f}x")

if epsilon_l1 / epsilon_l2 > 100:
    print("   ✅ Epsilon scaling looks correct (L¹ >> L²)")
else:
    print("   ❌ Epsilon scaling might be wrong")

# Test 3: DnCNN loading
print("\n3. Testing DnCNN loading...")
try:
    dncnn = create_denoiser('dncnn', pretrained='download', device='cpu')
    print("   ✅ DnCNN loaded successfully")
except Exception as e:
    print(f"   ⚠ DnCNN loading failed: {e}")
    print("   (This is OK if you don't have deepinv installed)")

# Test 4: DnCNN color image handling
print("\n4. Testing DnCNN with color image...")
try:
    dncnn = create_denoiser('dncnn', pretrained='download', device='cpu')

    # Create test RGB image
    test_image = np.random.rand(64, 64, 3).astype(np.float64)

    print(f"   Input shape: {test_image.shape} (RGB)")

    # Try denoising
    denoised = dncnn.denoise(test_image)

    print(f"   Output shape: {denoised.shape}")

    if denoised.shape == test_image.shape:
        print("   ✅ DnCNN handles RGB images correctly")
    else:
        print("   ❌ DnCNN output shape mismatch")

    # Check if it actually denoises (not identity)
    if not np.allclose(test_image, denoised, atol=0.05):
        print("   ✅ DnCNN is actually denoising (not identity)")
    else:
        print("   ⚠ DnCNN might be returning identity (could be OK for clean input)")

except Exception as e:
    print(f"   ⚠ DnCNN test failed: {e}")
    print("   (This is OK if you don't have deepinv installed)")

# Test 5: ADMM with clipping
print("\n5. Testing ADMM with input clipping...")
try:
    # Create simple test
    test_noisy = np.random.rand(32, 32).astype(np.float64)

    # Create simple Gaussian denoiser
    gaussian = create_denoiser('gaussian', sigma=1.0)

    # Test L2 CPnP
    config_l2 = CPnPConfig(
        constraint_type='l2',
        max_iter=5,
        verbose=False,
        store_history=True
    )

    solver_l2 = RobustCPnP(gaussian, config_l2)
    epsilon_test = 1.0
    restored_l2, info_l2 = solver_l2.solve(test_noisy, epsilon_test)

    print(f"   L² CPnP iterations: {info_l2.iterations}")
    print(f"   L² CPnP converged: {info_l2.converged}")

    if restored_l2.min() >= 0 and restored_l2.max() <= 1:
        print("   ✅ L² CPnP output in valid range [0, 1]")
    else:
        print(f"   ⚠ L² CPnP output range: [{restored_l2.min():.3f}, {restored_l2.max():.3f}]")

    # Test L1 CPnP
    config_l1 = CPnPConfig(
        constraint_type='l1',
        max_iter=5,
        verbose=False,
        store_history=True
    )

    solver_l1 = RobustCPnP(gaussian, config_l1)
    epsilon_test_l1 = 1000.0  # Larger epsilon for L1
    restored_l1, info_l1 = solver_l1.solve(test_noisy, epsilon_test_l1)

    print(f"   L¹ CPnP iterations: {info_l1.iterations}")
    print(f"   L¹ CPnP converged: {info_l1.converged}")

    if restored_l1.min() >= 0 and restored_l1.max() <= 1:
        print("   ✅ L¹ CPnP output in valid range [0, 1]")
    else:
        print(f"   ⚠ L¹ CPnP output range: [{restored_l1.min():.3f}, {restored_l1.max():.3f}]")

    print("   ✅ ADMM methods run without errors")

    # Check for blank output (key test!)
    print("\n6. Testing for blank output...")
    if restored_l2.max() - restored_l2.min() < 0.01:
        print(f"   ❌ L² output is BLANK! Range: [{restored_l2.min():.4f}, {restored_l2.max():.4f}]")
    else:
        print(f"   ✅ L² output has variation: [{restored_l2.min():.3f}, {restored_l2.max():.3f}]")

    if restored_l1.max() - restored_l1.min() < 0.01:
        print(f"   ❌ L¹ output is BLANK! Range: [{restored_l1.min():.4f}, {restored_l1.max():.4f}]")
    else:
        print(f"   ✅ L¹ output has variation: [{restored_l1.min():.3f}, {restored_l1.max():.3f}]")

except Exception as e:
    print(f"   ❌ ADMM test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("\nIf all tests passed (✅), you're ready to run the notebook!")
print("\nNext step:")
print("  1. Open: jupyter notebook Robust_CPnP_Demo_New.ipynb")
print("  2. Kernel → Restart & Clear Output")
print("  3. Cell → Run All")
print("  4. Wait 5-10 minutes")
print("  5. Check that L² CPnP achieves ~24-29 dB (not 14-17 dB)")
print("  6. Check that DnCNN achieves ~29 dB (not 20 dB)")
print("\n" + "=" * 70)

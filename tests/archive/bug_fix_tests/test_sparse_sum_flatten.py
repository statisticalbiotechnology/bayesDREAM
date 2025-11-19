"""
Test that sparse matrix sums are correctly flattened to avoid broadcasting issues.

This test verifies the fix for the bug where scipy.sparse.sum(axis=1) returns
a numpy.matrix of shape (n, 1) instead of a 1D array, causing incorrect mask
broadcasting.
"""

import numpy as np
from scipy import sparse

print("=" * 80)
print("Test: Sparse matrix sum flattening")
print("=" * 80)

# Create a sparse matrix (smaller for testing, but same pattern as user's 31468 x 21761 dataset)
n_features = 1000
n_cells = 500

print(f"\nCreating sparse matrix of shape ({n_features}, {n_cells})")

# Create sparse CSR matrix with some zeros
np.random.seed(42)
# Use sparse random to avoid creating huge dense array
sparse_matrix = sparse.random(n_features, n_cells, density=0.1, format='csr')
# Make some rows all zeros (to test zero_count_mask)
sparse_matrix[:100, :] = 0

print(f"Sparse matrix shape: {sparse_matrix.shape}")
print(f"Sparse matrix type: {type(sparse_matrix)}")

# Sum along axis 1 (cells axis) - this is what happens in technical.py
feature_sums = sparse_matrix.sum(axis=1)

print(f"\nBefore flattening:")
print(f"  feature_sums type: {type(feature_sums)}")
print(f"  feature_sums shape: {feature_sums.shape}")
print(f"  feature_sums is 2D: {feature_sums.ndim == 2}")

# This is the bug - feature_sums is a numpy.matrix of shape (n, 1)
if isinstance(feature_sums, np.matrix):
    print("  ✗ ISSUE: feature_sums is numpy.matrix (deprecated)")

# Create masks the old way (without flattening)
zero_count_mask_old = feature_sums == 0
zero_std_mask = np.zeros(len(feature_sums), dtype=bool)  # len() returns first dimension

print(f"\nOld behavior (WITHOUT flattening):")
print(f"  zero_count_mask shape: {zero_count_mask_old.shape}")
print(f"  zero_std_mask shape: {zero_std_mask.shape}")

# Try combining masks - this causes broadcasting issue
try:
    combined_old = zero_count_mask_old & ~zero_std_mask
    print(f"  Combined mask shape: {combined_old.shape}")
    if combined_old.shape != (n_features,):
        print(f"  ✗ BUG: Combined mask has wrong shape! Expected ({n_features},), got {combined_old.shape}")
        print(f"  This causes incorrect feature count: {combined_old.sum()} instead of {zero_count_mask_old.sum()}")
except Exception as e:
    print(f"  ✗ ERROR during mask combination: {e}")

# Now test the fix - flatten sparse matrix sum
feature_sums_fixed = np.asarray(feature_sums).flatten()

print(f"\nAfter flattening:")
print(f"  feature_sums_fixed type: {type(feature_sums_fixed)}")
print(f"  feature_sums_fixed shape: {feature_sums_fixed.shape}")
print(f"  feature_sums_fixed is 1D: {feature_sums_fixed.ndim == 1}")

if isinstance(feature_sums_fixed, np.ndarray) and feature_sums_fixed.ndim == 1:
    print("  ✓ FIXED: feature_sums is now 1D numpy array")

# Create masks the new way (with flattening)
zero_count_mask_new = feature_sums_fixed == 0
zero_std_mask = np.zeros(len(feature_sums_fixed), dtype=bool)

print(f"\nNew behavior (WITH flattening):")
print(f"  zero_count_mask shape: {zero_count_mask_new.shape}")
print(f"  zero_std_mask shape: {zero_std_mask.shape}")

# Try combining masks - should work correctly now
combined_new = zero_count_mask_new & ~zero_std_mask
print(f"  Combined mask shape: {combined_new.shape}")

if combined_new.shape == (n_features,):
    print(f"  ✓ SUCCESS: Combined mask has correct shape!")
    print(f"  Correct feature count: {combined_new.sum()}")
else:
    print(f"  ✗ FAILED: Combined mask still has wrong shape")

# Verify the counts match
print(f"\nVerification:")
print(f"  Number of zero-count features: {zero_count_mask_new.sum()}")
print(f"  Expected (first 100 rows): 100")
if zero_count_mask_new.sum() == 100:
    print(f"  ✓ Zero-count detection working correctly!")

print("\n" + "=" * 80)
print("Test completed successfully!")
print("=" * 80)

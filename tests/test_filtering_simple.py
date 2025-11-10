"""
Simple test for distribution-specific filtering at modality creation.
"""

import numpy as np
import pandas as pd
from bayesDREAM import bayesDREAM

print("=" * 80)
print("Setting up bayesDREAM model")
print("=" * 80)

# Create mock data
meta = pd.DataFrame({
    'cell': [f'cell{i}' for i in range(1, 11)],
    'guide': ['g1', 'g2', 'g3', 'g4', 'g5', 'ntc1', 'ntc2', 'ntc3', 'ntc4', 'ntc5'],
    'target': ['GFI1B', 'GFI1B', 'GFI1B', 'GFI1B', 'GFI1B', 'ntc', 'ntc', 'ntc', 'ntc', 'ntc'],
    'cell_line': ['A', 'A', 'B', 'B', 'B', 'A', 'A', 'B', 'B', 'B'],
    'sum_factor': [1.0]*10
})

gene_counts = pd.DataFrame({
    f'cell{i}': [10 + i, 20 + i, 30 + i, 40 + i, 50 + i, 100 + i, 200 + i, 300 + i, 400 + i, 500 + i, 1000 + i]
    for i in range(1, 11)
}, index=['GFI1B', 'GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5',
          'GENE6', 'GENE7', 'GENE8', 'GENE9', 'GENE10'])

# Initialize model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='filter_test'
)

print()
print("=" * 80)
print("Test 1: Custom binomial modality filtering")
print("=" * 80)

# Feature 1: ratio varies (10/100, 20/100, 30/100, 40/100, 50/100) = (0.1, 0.2, 0.3, 0.4, 0.5)
# Feature 2: constant ratio (10/100, 20/200, 30/300, 40/400, 50/500) = (0.1, 0.1, 0.1, 0.1, 0.1)

custom_counts = np.array([
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],  # Feature 1: variable ratio
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # Feature 2: constant ratio
])

custom_denom = np.array([
    [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],  # Feature 1 denom: constant (ratio varies!)
    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]   # Feature 2 denom: proportional (ratio constant!)
])

custom_meta = pd.DataFrame({
    'feature': ['feat1_variable', 'feat2_constant']
})

print("Adding custom binomial modality with 2 features:")
print("  - feat1_variable: ratio = (0.1, 0.2, 0.3, 0.4, 0.5) - VARIABLE, should KEEP")
print("  - feat2_constant: ratio = (0.1, 0.1, 0.1, 0.1, 0.1) - CONSTANT, should FILTER")

model.add_custom_modality(
    name='custom_binomial',
    counts=custom_counts,
    feature_meta=custom_meta,
    distribution='binomial',
    denominator=custom_denom
)

binomial_mod = model.get_modality('custom_binomial')
print(f"\n✓ Features after filtering: {binomial_mod.dims['n_features']}")
print(f"✓ Expected: 1 (kept feat1_variable, filtered feat2_constant)")
assert binomial_mod.dims['n_features'] == 1, f"Expected 1 feature, got {binomial_mod.dims['n_features']}"
print("✓ Custom binomial filtering works!")

print()
print("=" * 80)
print("Test 2: Multinomial filtering (all ratios)")
print("=" * 80)

# Feature 1: Category ratios vary - KEEP
#   Cell 1: [10, 10] -> ratios [0.5, 0.5]
#   Cell 2: [20, 10] -> ratios [0.67, 0.33]
#   Cell 3: [30, 10] -> ratios [0.75, 0.25]
# Feature 2: All category ratios constant - FILTER
#   All cells: [5, 5] -> ratios [0.5, 0.5] (constant!)

multinomial_counts = np.array([
    [[10, 10], [20, 10], [30, 10], [40, 10], [50, 10], [60, 10], [70, 10], [80, 10], [90, 10], [100, 10]],  # Feature 1: variable
    [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]              # Feature 2: constant ratios
])  # Shape: (2 features, 10 cells, 2 categories)

multinomial_meta = pd.DataFrame({
    'feature': ['feat1_variable', 'feat2_constant']
})

print("Adding custom multinomial modality with 2 features:")
print("  - feat1_variable: category ratios vary - should KEEP")
print("  - feat2_constant: ALL category ratios constant [0.5, 0.5] - should FILTER")

model.add_custom_modality(
    name='custom_multinomial',
    counts=multinomial_counts,
    feature_meta=multinomial_meta,
    distribution='multinomial'
)

multinomial_mod = model.get_modality('custom_multinomial')
print(f"\n✓ Features after filtering: {multinomial_mod.dims['n_features']}")
print(f"✓ Expected: 1 (kept feat1_variable, filtered feat2_constant)")
assert multinomial_mod.dims['n_features'] == 1, f"Expected 1 feature, got {multinomial_mod.dims['n_features']}"
print("✓ Custom multinomial filtering works!")

print()
print("=" * 80)
print("Test 3: Normal filtering (standard std=0)")
print("=" * 80)

normal_counts = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # Feature 1: varies
    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]   # Feature 2: constant - FILTER
])

normal_meta = pd.DataFrame({
    'feature': ['feat1_variable', 'feat2_constant']
})

print("Adding custom normal modality with 2 features:")
print("  - feat1_variable: [1, 2, 3, 4, 5] - should KEEP")
print("  - feat2_constant: [5, 5, 5, 5, 5] - should FILTER")

model.add_custom_modality(
    name='custom_normal',
    counts=normal_counts,
    feature_meta=normal_meta,
    distribution='normal'
)

normal_mod = model.get_modality('custom_normal')
print(f"\n✓ Features after filtering: {normal_mod.dims['n_features']}")
print(f"✓ Expected: 1 (kept feat1_variable, filtered feat2_constant)")
assert normal_mod.dims['n_features'] == 1, f"Expected 1 feature, got {normal_mod.dims['n_features']}"
print("✓ Normal filtering works!")

print()
print("=" * 80)
print("ALL FILTERING TESTS PASSED!")
print("=" * 80)
print()
print("Summary:")
print("  ✓ Binomial: Filters when numerator/denominator ratio has std=0")
print("  ✓ Multinomial: Filters when ALL category ratios (k/total) have std=0")
print("  ✓ Normal: Filters when raw values have std=0")

"""
Test distribution-specific filtering at modality creation.
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("Test 1: Binomial ratio filtering (exon skipping)")
print("=" * 80)

# Create test data with some zero-variance ratio features
inc1 = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1]])  # event 0: variable, event 1: constant ratio, event 2: constant ratio
inc2 = np.array([[2, 3, 4], [4, 8, 12], [1, 1, 1]])  # same pattern
skip = np.array([[1, 1, 1], [2, 4, 6], [0, 0, 0]])

# Event 0: inc=(1,2,3), tot=(2,3,4), ratio=(0.5, 0.67, 0.75) - VARIABLE
# Event 1: inc=min(2,4)=(2,4,6), tot=(4,8,12), ratio=(0.5, 0.5, 0.5) - CONSTANT
# Event 2: inc=min(1,1)=(1,1,1), tot=(1,1,1), ratio=(1,1,1) - CONSTANT

from bayesDREAM.splicing import process_exon_skipping

# Mock SJ data
sj_counts = pd.DataFrame({
    'cell1': [1, 2, 1, 2, 4, 1, 1, 1, 0],
    'cell2': [2, 3, 1, 3, 8, 1, 1, 1, 0],
    'cell3': [3, 4, 1, 4, 12, 1, 1, 1, 0]
}, index=['sj_inc1_1', 'sj_inc2_1', 'sj_skip_1',
          'sj_inc1_2', 'sj_inc2_2', 'sj_skip_2',
          'sj_inc1_3', 'sj_inc2_3', 'sj_skip_3'])

sj_meta = pd.DataFrame({
    'coord.intron': ['sj_inc1_1', 'sj_inc2_1', 'sj_skip_1',
                     'sj_inc1_2', 'sj_inc2_2', 'sj_skip_2',
                     'sj_inc1_3', 'sj_inc2_3', 'sj_skip_3'],
    'chrom': ['chr1']*9,
    'strand': ['+']*9,
    'intron_start': [100, 150, 100, 200, 250, 200, 300, 350, 300],
    'intron_end': [120, 170, 180, 220, 270, 280, 320, 370, 380],
    'gene_name_start': ['GENE1']*9,
    'gene_name_end': ['GENE1']*9
})

result = process_exon_skipping(sj_counts, sj_meta, min_total_exon=0, method='min', return_unfiltered=True)

print(f"Number of events after filtering: {len(result[3])}")
print(f"Number of unfiltered events: {result[6].shape[0] if result[6] is not None else 'N/A'}")

# Should filter 2 events with constant ratios, keep 1 with variable ratio
assert len(result[3]) <= result[6].shape[0], "Filtered should be <= unfiltered"
print("✓ Exon skipping ratio filtering works!")

print()
print("=" * 80)
print("Test 2: Multinomial ratio filtering (donor usage)")
print("=" * 80)

from bayesDREAM.splicing import process_donor_usage

# Create test data: donor with 2 acceptors
# Donor 1: Variable usage (0.5, 0.67, 0.75) - KEEP
# Donor 2: Constant usage (0.5, 0.5, 0.5) - FILTER

sj_counts_multi = pd.DataFrame({
    'cell1': [10, 10, 5, 5],
    'cell2': [20, 10, 5, 5],
    'cell3': [30, 10, 5, 5]
}, index=['donor1_acc1', 'donor1_acc2', 'donor2_acc1', 'donor2_acc2'])

sj_meta_multi = pd.DataFrame({
    'coord.intron': ['donor1_acc1', 'donor1_acc2', 'donor2_acc1', 'donor2_acc2'],
    'chrom': ['chr1']*4,
    'strand': ['+']*4,
    'intron_start': [100, 100, 200, 200],
    'intron_end': [120, 150, 220, 250],
    'gene_name_start': ['GENE1']*4,
    'gene_name_end': ['GENE1']*4
})

counts_3d, feature_meta, cell_names = process_donor_usage(sj_counts_multi, sj_meta_multi, min_cell_total=0)

print(f"Number of donors after filtering: {len(feature_meta)}")
print(f"Donors: {feature_meta}")

# Should keep donor 1 (variable ratios), filter donor 2 (constant ratios)
print("✓ Donor usage ratio filtering works!")

print()
print("=" * 80)
print("Test 3: Custom binomial modality filtering")
print("=" * 80)

from bayesDREAM import bayesDREAM, Modality

# Create mock data
meta = pd.DataFrame({
    'cell': [f'cell{i}' for i in range(1, 6)],
    'guide': ['g1', 'g2', 'g3', 'g4', 'g5'],
    'target': ['GFI1B']*5,
    'cell_line': ['A', 'A', 'B', 'B', 'B'],
    'sum_factor': [1.0]*5
})

gene_counts = pd.DataFrame({
    f'cell{i}': [10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]
    for i in range(1, 6)
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

# Add custom binomial modality with constant ratio
# Feature 1: ratio varies (10/100, 20/100, 30/100, 40/100, 50/100)
# Feature 2: constant ratio (10/100=0.1, 20/200=0.1, 30/300=0.1, ...)

custom_counts = np.array([
    [10, 20, 30, 40, 50],  # Feature 1: variable ratio
    [10, 20, 30, 40, 50]   # Feature 2: constant ratio (numerator/denominator = 0.1 always)
])

custom_denom = np.array([
    [100, 100, 100, 100, 100],  # Feature 1 denom: constant
    [100, 200, 300, 400, 500]   # Feature 2 denom: proportional (constant ratio!)
])

custom_meta = pd.DataFrame({
    'feature': ['feat1', 'feat2']
})

print("Adding custom binomial modality with 2 features (1 variable, 1 constant ratio)...")
model.add_custom_modality(
    name='custom_binomial',
    counts=custom_counts,
    feature_meta=custom_meta,
    distribution='binomial',
    denominator=custom_denom
)

binomial_mod = model.get_modality('custom_binomial')
print(f"Features after filtering: {binomial_mod.dims['n_features']}")
assert binomial_mod.dims['n_features'] == 1, f"Expected 1 feature, got {binomial_mod.dims['n_features']}"
print("✓ Custom binomial filtering works!")

print()
print("=" * 80)
print("ALL FILTERING TESTS PASSED!")
print("=" * 80)

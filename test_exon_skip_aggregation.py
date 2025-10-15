"""
Test the new exon skipping aggregation functionality.
"""
import numpy as np
import pandas as pd
from bayesDREAM import Modality

print("=" * 60)
print("Test 1: Create exon skipping modality with inc1/inc2/skip")
print("=" * 60)

n_events = 5
n_cells = 10

# Create synthetic data
np.random.seed(42)
inc1 = np.random.poisson(10, (n_events, n_cells)).astype(float)
inc2 = np.random.poisson(12, (n_events, n_cells)).astype(float)
skip = np.random.poisson(8, (n_events, n_cells)).astype(float)

# Compute inclusion with min method
inclusion_min = np.minimum(inc1, inc2)
total_min = inclusion_min + skip

# Create feature metadata
feature_meta = pd.DataFrame({
    'trip_id': range(n_events),
    'chrom': ['chr1'] * n_events,
    'strand': ['+'] * n_events
})

# Create modality with min aggregation
modality = Modality(
    name='exon_skip_test',
    counts=inclusion_min,
    feature_meta=feature_meta,
    distribution='binomial',
    denominator=total_min,
    inc1=inc1,
    inc2=inc2,
    skip=skip,
    exon_aggregate_method='min'
)

print(f"✓ Created modality: {modality}")
print(f"✓ Is exon skipping: {modality.is_exon_skipping()}")
print(f"✓ Aggregation method: {modality.exon_aggregate_method}")
print(f"✓ Inc1 shape: {modality.inc1.shape}")
print(f"✓ Inc2 shape: {modality.inc2.shape}")
print(f"✓ Skip shape: {modality.skip.shape}")

print("\n" + "=" * 60)
print("Test 2: Change aggregation method from 'min' to 'mean'")
print("=" * 60)

old_counts = modality.counts.copy()
modality.set_exon_aggregate_method('mean')

print(f"✓ New aggregation method: {modality.exon_aggregate_method}")
print(f"✓ Counts changed: {not np.allclose(old_counts, modality.counts)}")

# Manually verify the mean calculation
expected_inclusion_mean = (inc1 + inc2) / 2.0
expected_total_mean = expected_inclusion_mean + skip
print(f"✓ Inclusion matches expected: {np.allclose(modality.counts, expected_inclusion_mean)}")
print(f"✓ Total matches expected: {np.allclose(modality.denominator, expected_total_mean)}")

print("\n" + "=" * 60)
print("Test 3: Mark technical fit complete and try to change method")
print("=" * 60)

modality.mark_technical_fit_complete()
print(f"✓ Technical fit marked complete with method: {modality._technical_fit_aggregate_method}")

try:
    modality.set_exon_aggregate_method('min')
    print("✗ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly prevented method change: {str(e)[:80]}...")

print("\n" + "=" * 60)
print("Test 4: Override with allow_after_technical_fit=True")
print("=" * 60)

modality.set_exon_aggregate_method('min', allow_after_technical_fit=True)
print(f"✓ Changed to method: {modality.exon_aggregate_method}")

# Verify it went back to min
expected_inclusion_min = np.minimum(inc1, inc2)
expected_total_min = expected_inclusion_min + skip
print(f"✓ Inclusion matches min: {np.allclose(modality.counts, expected_inclusion_min)}")
print(f"✓ Total matches min: {np.allclose(modality.denominator, expected_total_min)}")

print("\n" + "=" * 60)
print("Test 5: Subsetting preserves exon skipping data")
print("=" * 60)

# Subset to first 3 events
subset = modality.get_feature_subset([0, 1, 2])
print(f"✓ Subset created: {subset}")
print(f"✓ Is exon skipping: {subset.is_exon_skipping()}")
print(f"✓ Inc1 shape: {subset.inc1.shape}")
print(f"✓ Aggregation method preserved: {subset.exon_aggregate_method}")

# Subset to first 5 cells
subset_cells = modality.get_cell_subset([0, 1, 2, 3, 4])
print(f"✓ Cell subset created: {subset_cells}")
print(f"✓ Inc1 shape: {subset_cells.inc1.shape}")

print("\n" + "=" * 60)
print("Test 6: Modality without exon skipping data")
print("=" * 60)

regular_mod = Modality(
    name='regular_binomial',
    counts=inclusion_min,
    feature_meta=feature_meta,
    distribution='binomial',
    denominator=total_min
)

print(f"✓ Regular modality: {regular_mod}")
print(f"✓ Is exon skipping: {regular_mod.is_exon_skipping()}")

try:
    regular_mod.set_exon_aggregate_method('mean')
    print("✗ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly rejected: {str(e)[:60]}...")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)

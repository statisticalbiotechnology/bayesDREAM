"""
Simple test script for multi-modal bayesDREAM.

Run this to verify basic functionality without needing real data.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add bayesDREAM to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bayesDREAM import Modality, MultiModalBayesDREAM

print("=" * 60)
print("Testing Multi-Modal bayesDREAM")
print("=" * 60)

# =============================================================================
# Test 1: Create Modality objects
# =============================================================================
print("\n[Test 1] Creating Modality objects...")

# Gene counts (negbinom)
gene_counts_array = np.random.poisson(10, (100, 50))
gene_counts = pd.DataFrame(
    gene_counts_array,
    index=[f'GENE{i}' for i in range(100)],
    columns=[f'CELL{i}' for i in range(50)]
)
gene_meta = pd.DataFrame({'gene': gene_counts.index})

gene_modality = Modality(
    name='gene',
    counts=gene_counts,
    feature_meta=gene_meta,
    distribution='negbinom',
    cells_axis=1
)

print(f"✓ Created gene modality: {gene_modality}")
assert gene_modality.dims['n_features'] == 100
assert gene_modality.dims['n_cells'] == 50
print(f"  Dimensions: {gene_modality.dims}")

# Multinomial modality (e.g., donor usage)
donor_counts = np.random.multinomial(100, [0.3, 0.5, 0.2], size=(20, 50))  # 20 donors, 50 cells, 3 acceptors
donor_meta = pd.DataFrame({
    'chrom': ['chr1'] * 20,
    'strand': ['+'] * 20,
    'donor': np.arange(1000, 1020),
    'acceptors': [['acc1', 'acc2', 'acc3']] * 20,
    'n_acceptors': [3] * 20
})

donor_modality = Modality(
    name='donor',
    counts=donor_counts,
    feature_meta=donor_meta,
    distribution='multinomial',
    cells_axis=1
)

print(f"✓ Created donor modality: {donor_modality}")
assert donor_modality.dims['n_features'] == 20
assert donor_modality.dims['n_cells'] == 50
assert donor_modality.dims['n_categories'] == 3
print(f"  Dimensions: {donor_modality.dims}")

# Binomial modality (e.g., exon skipping)
exon_inclusion = np.random.binomial(100, 0.5, size=(15, 50))
exon_total = np.random.binomial(200, 0.5, size=(15, 50))
exon_meta = pd.DataFrame({
    'trip_id': np.arange(15),
    'chrom': ['chr1'] * 15,
    'strand': ['+'] * 15,
    'd1': np.arange(2000, 2015),
    'a2': np.arange(3000, 3015),
    'd2': np.arange(4000, 4015),
    'a3': np.arange(5000, 5015)
})

exon_modality = Modality(
    name='exon_skip',
    counts=exon_inclusion,
    feature_meta=exon_meta,
    distribution='binomial',
    denominator=exon_total,
    cells_axis=1
)

print(f"✓ Created exon skip modality: {exon_modality}")
print(f"  Has denominator: {exon_modality.denominator is not None}")

# Normal modality (e.g., SpliZ)
spliz_scores = np.random.normal(0, 1, size=(100, 50))
spliz_meta = pd.DataFrame({'gene': [f'GENE{i}' for i in range(100)]})

spliz_modality = Modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=spliz_meta,
    distribution='normal',
    cells_axis=1
)

print(f"✓ Created SpliZ modality: {spliz_modality}")

# Multivariate normal (e.g., SpliZVD)
splizvd_array = np.random.normal(0, 1, size=(100, 50, 3))  # 3D: genes x cells x (z0, z1, z2)
splizvd_meta = pd.DataFrame({'gene': [f'GENE{i}' for i in range(100)]})

splizvd_modality = Modality(
    name='splizvd',
    counts=splizvd_array,
    feature_meta=splizvd_meta,
    distribution='mvnormal',
    cells_axis=1
)

print(f"✓ Created SpliZVD modality: {splizvd_modality}")
assert splizvd_modality.dims['n_dimensions'] == 3

print("\n✓ All Modality objects created successfully!")

# =============================================================================
# Test 2: Modality subsetting
# =============================================================================
print("\n[Test 2] Testing modality subsetting...")

# Feature subset
subset_genes = [f'GENE{i}' for i in [0, 5, 10]]
gene_subset = gene_modality.get_feature_subset(subset_genes)
print(f"✓ Feature subset: {gene_subset.dims['n_features']} features (expected 3)")
assert gene_subset.dims['n_features'] == 3

# Cell subset
subset_cells = [f'CELL{i}' for i in range(10)]
cell_subset = gene_modality.get_cell_subset(subset_cells)
print(f"✓ Cell subset: {cell_subset.dims['n_cells']} cells (expected 10)")
assert cell_subset.dims['n_cells'] == 10

print("\n✓ Subsetting works correctly!")

# =============================================================================
# Test 3: Create MultiModalBayesDREAM (without actual fitting)
# =============================================================================
print("\n[Test 3] Creating MultiModalBayesDREAM...")

# Create minimal metadata
meta = pd.DataFrame({
    'cell': [f'CELL{i}' for i in range(50)],
    'guide': ['guide1'] * 25 + ['guide2'] * 25,
    'target': ['ntc'] * 25 + ['GENE0'] * 25,
    'cell_line': ['line1'] * 50,
    'sum_factor': np.random.uniform(0.8, 1.2, 50)
})

try:
    # Initialize with gene counts
    model = MultiModalBayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GENE0',
        output_dir='./test_output',
        label='test_run',
        cores=1
    )
    print(f"✓ Created model: {model}")
    print(f"  Primary modality: {model.primary_modality}")

    # Add other modalities
    model.add_modality('donor', donor_modality)
    model.add_modality('exon_skip', exon_modality)
    model.add_modality('spliz', spliz_modality)
    model.add_modality('splizvd', splizvd_modality)

    print(f"✓ Added 4 additional modalities")

    # List modalities
    modality_summary = model.list_modalities()
    print(f"\n  Modality summary:\n{modality_summary}")

    assert len(model.modalities) == 5
    assert 'gene' in model.modalities
    assert 'donor' in model.modalities

    # Access specific modality
    retrieved_donor = model.get_modality('donor')
    print(f"\n✓ Retrieved modality: {retrieved_donor}")

    print("\n✓ MultiModalBayesDREAM created successfully!")

except Exception as e:
    print(f"\n✗ Error creating MultiModalBayesDREAM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Test 4: Custom modality addition
# =============================================================================
print("\n[Test 4] Testing custom modality addition...")

try:
    # Add via add_custom_modality
    custom_data = np.random.normal(0, 1, size=(100, 50))
    custom_meta = pd.DataFrame({'feature': [f'FEATURE{i}' for i in range(100)]})

    model.add_custom_modality(
        name='custom',
        counts=custom_data,
        feature_meta=custom_meta,
        distribution='normal'
    )

    print(f"✓ Added custom modality")
    assert 'custom' in model.modalities

    print("\n✓ Custom modality addition works!")

except Exception as e:
    print(f"\n✗ Error adding custom modality: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Test 5: Tensor conversion
# =============================================================================
print("\n[Test 5] Testing tensor conversion...")

try:
    gene_tensor = gene_modality.to_tensor()
    print(f"✓ Converted to tensor: shape {gene_tensor.shape}")
    assert gene_tensor.shape == (100, 50)

    donor_tensor = donor_modality.to_tensor()
    print(f"✓ Converted donor to tensor: shape {donor_tensor.shape}")
    assert donor_tensor.shape == (20, 50, 3)

    print("\n✓ Tensor conversion works!")

except Exception as e:
    print(f"\n✗ Error in tensor conversion: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nMulti-modal bayesDREAM is working correctly.")
print("\nNext steps:")
print("  1. See examples/multimodal_example.py for real-world usage")
print("  2. See QUICKSTART_MULTIMODAL.md for documentation")
print("  3. See MULTIMODAL_IMPLEMENTATION.md for technical details")

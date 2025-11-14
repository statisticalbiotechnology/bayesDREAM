"""
Test that gene_meta is auto-created from DataFrame index and plotting works.

This test addresses the user's requirement:
a) gene_meta mandatory when counts has no index, else create from index
b) plotting should work in all cases (gene names or Ensembl IDs)
"""

import pandas as pd
import numpy as np
import sys

# Test 1: Initialize with DataFrame having gene names as index (NO gene_meta)
print("=" * 80)
print("Test 1: DataFrame with gene names as index, no gene_meta provided")
print("=" * 80)

# Load toy data
meta = pd.read_csv('toydata/cell_meta.csv')  # CELL metadata, not gene metadata
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print(f"\nCounts shape: {counts.shape}")
print(f"Counts index (first 5): {counts.index[:5].tolist()}")
print(f"Counts index dtype: {counts.index.dtype}")

# Initialize WITHOUT providing gene_meta (should auto-create from index)
from bayesDREAM import bayesDREAM

try:
    model = bayesDREAM(
        meta=meta,
        counts=counts,  # DataFrame with gene names as index
        # gene_meta=None,  # Not provided - should auto-create
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='autocreate_test'
    )
    print("\n✓ Model initialized successfully")
except Exception as e:
    print(f"\n✗ FAILED to initialize model: {e}")
    sys.exit(1)

# Verify gene modality has correct feature_meta
gene_mod = model.get_modality('gene')
print(f"\nGene modality feature_meta shape: {gene_mod.feature_meta.shape}")
print(f"Gene modality feature_meta columns: {gene_mod.feature_meta.columns.tolist()}")

# Check that gene_name and gene columns exist
if 'gene_name' not in gene_mod.feature_meta.columns:
    print("\n✗ FAILED: 'gene_name' column not found in feature_meta")
    sys.exit(1)
if 'gene' not in gene_mod.feature_meta.columns:
    print("\n✗ FAILED: 'gene' column not found in feature_meta")
    sys.exit(1)

print("\n✓ Both 'gene_name' and 'gene' columns present in feature_meta")

# Verify gene names match original index
print(f"\nFirst 5 genes in feature_meta['gene_name']: {gene_mod.feature_meta['gene_name'].head().tolist()}")
print(f"First 5 genes in original index: {counts.index[:5].tolist()}")

# Check that they match (excluding cis gene which was extracted)
original_genes = set(counts.index.tolist())
feature_genes = set(gene_mod.feature_meta['gene_name'].tolist())
cis_mod = model.get_modality('cis')
cis_genes = set(cis_mod.feature_meta['gene_name'].tolist()) if 'gene_name' in cis_mod.feature_meta.columns else set()

# Gene modality should have all genes except the cis gene
expected_gene_count = len(original_genes) - len(cis_genes)
actual_gene_count = len(feature_genes)

if actual_gene_count != expected_gene_count:
    print(f"\n✗ FAILED: Expected {expected_gene_count} genes in gene modality, got {actual_gene_count}")
    sys.exit(1)

print(f"\n✓ Gene count correct: {actual_gene_count} genes (excluding cis gene)")

# Test 2: Verify plotting works with gene names
print("\n" + "=" * 80)
print("Test 2: Plotting with gene names")
print("=" * 80)

try:
    from bayesDREAM.plotting.xy_plots import _resolve_features

    # Try to resolve a gene name that should exist
    test_genes = ['GAPDH', 'TET2', 'TBPL1']
    for gene_name in test_genes:
        if gene_name in feature_genes:
            try:
                indices, names, is_gene = _resolve_features(gene_name, gene_mod)
                print(f"\n✓ Successfully resolved '{gene_name}' -> indices: {indices}, names: {names}, is_gene: {is_gene}")
            except Exception as e:
                print(f"\n✗ FAILED to resolve '{gene_name}': {e}")
                sys.exit(1)
        else:
            print(f"\n  Skipping '{gene_name}' (not in gene modality - may be cis gene)")

except Exception as e:
    print(f"\n✗ FAILED during plotting test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: DataFrame with numeric index should raise error
print("\n" + "=" * 80)
print("Test 3: DataFrame with numeric index, no gene_meta (should FAIL)")
print("=" * 80)

counts_numeric = counts.copy()
counts_numeric.index = range(len(counts_numeric))  # Replace with numeric index

try:
    model_fail = bayesDREAM(
        meta=meta,
        counts=counts_numeric,  # Numeric index
        # gene_meta=None,  # Not provided
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='should_fail'
    )
    print("\n✗ FAILED: Should have raised ValueError for numeric index without gene_meta")
    sys.exit(1)
except ValueError as e:
    if "numeric index" in str(e).lower():
        print(f"\n✓ Correctly raised ValueError: {e}")
    else:
        print(f"\n✗ FAILED: Raised ValueError but wrong message: {e}")
        sys.exit(1)
except Exception as e:
    print(f"\n✗ FAILED: Raised wrong exception type: {type(e).__name__}: {e}")
    sys.exit(1)

# Test 4: Array without gene_meta should raise error
print("\n" + "=" * 80)
print("Test 4: Array without gene_meta (should FAIL)")
print("=" * 80)

counts_array = counts.values  # Convert to numpy array

try:
    model_fail2 = bayesDREAM(
        meta=meta,
        counts=counts_array,  # Array
        # gene_meta=None,  # Not provided
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='should_fail2'
    )
    print("\n✗ FAILED: Should have raised ValueError for array without gene_meta")
    sys.exit(1)
except ValueError as e:
    if "not a dataframe" in str(e).lower() or "gene_meta must be provided" in str(e).lower():
        print(f"\n✓ Correctly raised ValueError: {e}")
    else:
        print(f"\n✗ FAILED: Raised ValueError but wrong message: {e}")
        sys.exit(1)
except Exception as e:
    print(f"\n✗ FAILED: Raised wrong exception type: {type(e).__name__}: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("ALL TESTS PASSED!")
print("=" * 80)
print("\nSummary:")
print("✓ Auto-created gene_meta from DataFrame index")
print("✓ Plotting can resolve gene names")
print("✓ Numeric index without gene_meta raises error")
print("✓ Array without gene_meta raises error")

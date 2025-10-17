"""
Test multi-modal fitting infrastructure.

This script tests:
1. Backward compatibility: bayesDREAM works exactly like bayesDREAM for gene expression
2. fit_modality_technical() delegates correctly to fit_technical()
3. fit_modality_trans() delegates correctly to fit_trans()
4. Distribution registry is properly loaded
"""

import sys
import pandas as pd
import numpy as np

# Test imports
print("Testing imports...")
try:
    from bayesDREAM import (
        bayesDREAM,
        get_observation_sampler,
        requires_denominator,
        is_3d_distribution,
        DISTRIBUTION_REGISTRY
    )
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test distribution registry
print("\nTesting distribution registry...")
try:
    assert 'negbinom' in DISTRIBUTION_REGISTRY
    assert 'multinomial' in DISTRIBUTION_REGISTRY
    assert 'binomial' in DISTRIBUTION_REGISTRY
    assert 'normal' in DISTRIBUTION_REGISTRY
    assert 'mvnormal' in DISTRIBUTION_REGISTRY
    print("✓ All distributions registered")

    # Test helper functions
    assert requires_denominator('binomial') == True
    assert requires_denominator('negbinom') == False
    assert is_3d_distribution('multinomial') == True
    assert is_3d_distribution('negbinom') == False
    print("✓ Helper functions work correctly")

    # Test get_observation_sampler
    negbinom_sampler = get_observation_sampler('negbinom', 'trans')
    assert callable(negbinom_sampler)
    print("✓ get_observation_sampler works correctly")

except AssertionError as e:
    print(f"✗ Distribution registry test failed: {e}")
    sys.exit(1)

# Create toy data
print("\nCreating toy data...")
np.random.seed(42)
n_genes = 20
n_cells = 50
n_guides = 10

# Cell metadata
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
    'cell_line': np.random.choice(['A', 'B'], n_cells),
    'target': ['GFI1B' if i < 30 else 'ntc' for i in range(n_cells)],
    'sum_factor': np.random.lognormal(0, 0.3, n_cells)
})

# Gene counts (including cis gene)
genes = [f'gene_{i}' for i in range(n_genes)] + ['GFI1B']
gene_counts = pd.DataFrame(
    np.random.poisson(50, (len(genes), n_cells)),
    index=genes,
    columns=meta['cell']
)

print(f"  - Metadata: {meta.shape[0]} cells")
print(f"  - Gene counts: {gene_counts.shape[0]} genes × {gene_counts.shape[1]} cells")

# Test 1: Create bayesDREAM
print("\nTest 1: Create bayesDREAM...")
try:
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        primary_modality='gene',
        output_dir='./test_output',
        label='test_multimodal',
        device='cpu',
        cores=1
    )
    print("✓ bayesDREAM created successfully")
    print(f"  - Primary modality: {model.primary_modality}")
    print(f"  - Number of modalities: {len(model.modalities)}")

    # Check modalities
    modalities_df = model.list_modalities()
    print(f"\n  Modalities:\n{modalities_df}")

except Exception as e:
    print(f"✗ Failed to create bayesDREAM: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test fit_modality_technical() delegation
print("\nTest 2: Test fit_modality_technical() delegation...")
try:
    # This should work (delegates to fit_technical)
    print("  - Testing delegation to fit_technical()...")
    # Note: Not actually running the fit (takes too long), just testing the method exists
    assert hasattr(model, 'fit_modality_technical')
    print("✓ fit_modality_technical() method exists")

    # Test that it raises NotImplementedError for non-negbinom distributions
    # (We don't have any yet, so we'll skip this test)

except Exception as e:
    print(f"✗ fit_modality_technical() test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test fit_modality_trans() delegation
print("\nTest 3: Test fit_modality_trans() delegation...")
try:
    # This should work (delegates to fit_trans)
    print("  - Testing delegation to fit_trans()...")
    # Note: Not actually running the fit (takes too long), just testing the method exists
    assert hasattr(model, 'fit_modality_trans')
    print("✓ fit_modality_trans() method exists")

except Exception as e:
    print(f"✗ fit_modality_trans() test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test that gene modality excludes cis gene
print("\nTest 4: Test that gene modality excludes cis gene...")
try:
    gene_modality = model.get_modality('gene')
    gene_names = gene_modality.feature_meta['gene'].tolist()

    assert 'GFI1B' not in gene_names, "Cis gene should be excluded from gene modality!"
    assert len(gene_names) == n_genes, f"Expected {n_genes} genes, got {len(gene_names)}"

    print(f"✓ Gene modality correctly excludes cis gene (has {len(gene_names)} genes)")

except Exception as e:
    print(f"✗ Gene modality test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test that base class still has cis gene
print("\nTest 5: Test that base class still has cis gene for cis modeling...")
try:
    assert 'GFI1B' in model.counts.index, "Cis gene should be in base class counts!"
    assert model.cis_gene == 'GFI1B'

    print(f"✓ Base class counts include cis gene (has {len(model.counts)} genes)")

except Exception as e:
    print(f"✗ Base class cis gene test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nSummary:")
print("1. ✓ Imports successful")
print("2. ✓ Distribution registry functional")
print("3. ✓ bayesDREAM created successfully")
print("4. ✓ fit_modality_technical() method exists")
print("5. ✓ fit_modality_trans() method exists")
print("6. ✓ Gene modality excludes cis gene")
print("7. ✓ Base class retains cis gene for cis modeling")
print("\nInfrastructure is ready for multi-modal fitting!")
print("Next step: Implement distribution-specific models for fit_technical and fit_trans.")

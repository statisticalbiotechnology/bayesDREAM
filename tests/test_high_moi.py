"""
Test high MOI (multiple guides per cell) functionality.

This test verifies:
1. Initialization with guide_assignment matrix and guide_meta
2. Backward compatibility (single-guide mode still works)
3. Additive guide effects in fit_cis
4. Proper handling of NTC cells
"""

import numpy as np
import pandas as pd
import torch
import pyro
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesDREAM import bayesDREAM


def test_high_moi_initialization():
    """Test that high MOI initialization works correctly."""
    print("\n" + "="*80)
    print("TEST: High MOI Initialization")
    print("="*80)

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    pyro.set_rng_seed(42)

    # Create synthetic data
    n_cells = 100
    n_genes = 50
    n_guides = 6  # 2 targeting GFI1B, 2 targeting MYB, 2 NTC

    # Cell metadata (no 'guide' or 'target' columns for high MOI)
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'cell_line': np.random.choice(['K562', 'MOLM13'], n_cells),
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells)
    })

    # Gene counts
    gene_names = [f'gene_{i}' for i in range(n_genes)]
    gene_names[0] = 'GFI1B'  # Make sure cis_gene exists
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (n_genes, n_cells)),
        index=gene_names,
        columns=[f'cell_{i}' for i in range(n_cells)]
    )

    # Guide assignment matrix (binary, cells × guides)
    # Each cell has 0-2 guides
    guide_assignment = np.zeros((n_cells, n_guides), dtype=int)

    # Cells 0-29: guide 0 + guide 1 (both target GFI1B)
    guide_assignment[0:30, 0] = 1
    guide_assignment[0:30, 1] = 1

    # Cells 30-59: guide 2 only (targets MYB)
    guide_assignment[30:60, 2] = 1

    # Cells 60-79: guide 3 + guide 4 (both target MYB)
    guide_assignment[60:80, 3] = 1
    guide_assignment[60:80, 4] = 1

    # Cells 80-99: NTC guides only
    guide_assignment[80:100, 5] = 1

    # Guide metadata
    guide_meta = pd.DataFrame({
        'guide': ['guide_A', 'guide_B', 'guide_C', 'guide_D', 'guide_E', 'ntc_1'],
        'target': ['GFI1B', 'GFI1B', 'MYB', 'MYB', 'MYB', 'ntc']
    }, index=['guide_A', 'guide_B', 'guide_C', 'guide_D', 'guide_E', 'ntc_1'])

    # Initialize model with high MOI
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        guide_assignment=guide_assignment,
        guide_meta=guide_meta,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='test_high_moi',
        device='cpu'
    )

    # Verify high MOI mode is active
    assert model.is_high_moi, "Model should be in high MOI mode"
    assert hasattr(model, 'guide_assignment'), "Model should have guide_assignment attribute"
    assert hasattr(model, 'guide_meta'), "Model should have guide_meta attribute"
    assert hasattr(model, 'guide_assignment_tensor'), "Model should have guide_assignment_tensor attribute"

    # Verify guide_meta shape
    assert len(model.guide_meta) == n_guides, \
        f"guide_meta should have {n_guides} guides, got {len(model.guide_meta)}"

    # Verify target column was created in meta
    assert 'target' in model.meta.columns, "meta should have 'target' column"

    # Verify NTC cells were identified correctly
    ntc_cells = (model.meta['target'] == 'ntc').sum()
    print(f"NTC cells identified: {ntc_cells}")
    assert ntc_cells == 20, f"Should have 20 NTC cells, found {ntc_cells}"

    # Verify targeting cells were identified correctly
    targeting_cells = (model.meta['target'] == 'GFI1B').sum()
    print(f"GFI1B-targeting cells: {targeting_cells}")
    assert targeting_cells == 30, f"Should have 30 GFI1B-targeting cells, found {targeting_cells}"

    # Verify model subsetted to NTC + GFI1B-targeting cells only (excludes 'other' cells)
    n_cells_subsetted = len(model.meta)
    expected_cells = ntc_cells + targeting_cells
    assert n_cells_subsetted == expected_cells, \
        f"Model should have {expected_cells} cells after subsetting, got {n_cells_subsetted}"

    # Verify guide_assignment shape after subsetting
    assert model.guide_assignment.shape == (expected_cells, n_guides), \
        f"guide_assignment shape should be ({expected_cells}, {n_guides}), got {model.guide_assignment.shape}"

    # Verify guide_code is set to -1 (placeholder)
    assert (model.meta['guide_code'] == -1).all(), "guide_code should be -1 for high MOI mode"

    print("✓ High MOI initialization test PASSED")
    return model


def test_high_moi_cis_fitting():
    """Test that cis fitting works with high MOI."""
    print("\n" + "="*80)
    print("TEST: High MOI Cis Fitting")
    print("="*80)

    # Get model from initialization test
    model = test_high_moi_initialization()

    # Run fit_cis with reduced iterations for testing
    print("\nRunning fit_cis...")
    model.fit_cis(
        sum_factor_col='sum_factor',
        lr=1e-2,
        niters=500,  # Reduced for testing
        nsamples=10,  # Reduced for testing
        tolerance=1e-3
    )

    # Verify posterior samples exist
    assert hasattr(model, 'posterior_samples_cis'), "Model should have posterior_samples_cis"
    assert 'x_true' in model.posterior_samples_cis, "Posterior should contain x_true"
    assert 'x_eff_g' in model.posterior_samples_cis, "Posterior should contain x_eff_g (per-guide effects)"

    # Verify x_true shape (samples × cells)
    x_true = model.posterior_samples_cis['x_true']
    print(f"x_true shape: {x_true.shape}")
    assert x_true.shape[1] == len(model.meta), \
        f"x_true should have {len(model.meta)} cells, got {x_true.shape[1]}"

    # Verify x_eff_g shape (samples × guides)
    x_eff_g = model.posterior_samples_cis['x_eff_g']
    print(f"x_eff_g shape: {x_eff_g.shape}")
    assert x_eff_g.shape[1] == model.guide_assignment.shape[1], \
        f"x_eff_g should have {model.guide_assignment.shape[1]} guides, got {x_eff_g.shape[1]}"

    print("✓ High MOI cis fitting test PASSED")
    return model


def test_additive_guide_effects():
    """Test that guide effects are additive for cells with multiple guides."""
    print("\n" + "="*80)
    print("TEST: Additive Guide Effects")
    print("="*80)

    # Get fitted model
    model = test_high_moi_cis_fitting()

    # Get posterior mean of x_true and x_eff_g
    x_true_mean = model.posterior_samples_cis['x_true'].mean(dim=0).numpy()
    x_eff_g_mean = model.posterior_samples_cis['x_eff_g'].mean(dim=0).numpy()

    print(f"\nPer-guide effects (posterior mean):")
    for i, guide_name in enumerate(model.guide_meta['guide']):
        print(f"  {guide_name}: {x_eff_g_mean[i]:.4f}")

    # For cells with multiple guides, x_true should approximately equal sum of guide effects
    # Check cells 0-29 (have guides 0 and 1)
    cells_with_guides_0_1 = np.where((model.guide_assignment[:, 0] == 1) &
                                      (model.guide_assignment[:, 1] == 1))[0]

    if len(cells_with_guides_0_1) > 0:
        expected_sum = x_eff_g_mean[0] + x_eff_g_mean[1]
        actual_mean = x_true_mean[cells_with_guides_0_1].mean()

        print(f"\nCells with guides 0+1:")
        print(f"  Expected sum: {expected_sum:.4f}")
        print(f"  Actual mean x_true: {actual_mean:.4f}")
        print(f"  Difference: {abs(expected_sum - actual_mean):.4f}")

        # Allow for some variance due to cell-level noise (sigma_eff)
        # Just check that they're in the same ballpark (within 50%)
        assert abs(expected_sum - actual_mean) / expected_sum < 0.5, \
            f"x_true should be approximately additive (within 50%), but got {abs(expected_sum - actual_mean) / expected_sum:.2%} difference"

    print("✓ Additive guide effects test PASSED")


def test_backward_compatibility():
    """Test that single-guide mode still works (backward compatibility)."""
    print("\n" + "="*80)
    print("TEST: Backward Compatibility (Single-Guide Mode)")
    print("="*80)

    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    pyro.set_rng_seed(42)

    # Create synthetic data in traditional format (with 'guide' and 'target' in meta)
    n_cells = 50
    n_genes = 30

    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': ['guide_A'] * 25 + ['ntc'] * 25,
        'target': ['GFI1B'] * 25 + ['ntc'] * 25,
        'cell_line': np.random.choice(['K562'], n_cells),
        'sum_factor': np.random.uniform(0.8, 1.2, n_cells)
    })

    gene_names = [f'gene_{i}' for i in range(n_genes)]
    gene_names[0] = 'GFI1B'
    counts = pd.DataFrame(
        np.random.negative_binomial(20, 0.3, (n_genes, n_cells)),
        index=gene_names,
        columns=[f'cell_{i}' for i in range(n_cells)]
    )

    # Initialize model WITHOUT guide_assignment (single-guide mode)
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='test_single_guide',
        device='cpu'
    )

    # Verify NOT in high MOI mode
    assert not model.is_high_moi, "Model should NOT be in high MOI mode"
    assert not hasattr(model, 'guide_assignment'), "Model should NOT have guide_assignment in single-guide mode"

    # Verify guide_code exists and is not -1
    assert 'guide_code' in model.meta.columns, "meta should have guide_code column"
    assert not (model.meta['guide_code'] == -1).all(), "guide_code should not be -1 in single-guide mode"

    # Run fit_cis to verify it still works
    print("\nRunning fit_cis in single-guide mode...")
    model.fit_cis(
        sum_factor_col='sum_factor',
        lr=1e-2,
        niters=500,
        nsamples=10,
        tolerance=1e-3
    )

    assert hasattr(model, 'posterior_samples_cis'), "Model should have posterior_samples_cis"
    assert 'x_true' in model.posterior_samples_cis, "Posterior should contain x_true"

    print("✓ Backward compatibility test PASSED")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING HIGH MOI TESTS")
    print("="*80)

    try:
        # Run all tests
        test_high_moi_initialization()
        test_high_moi_cis_fitting()
        test_additive_guide_effects()
        test_backward_compatibility()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)

    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

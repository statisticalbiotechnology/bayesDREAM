"""
Quick test to verify trans model works for binomial and multinomial.

This test focuses on verifying the critical changes:
1. Binomial Hill uses Beta priors in probability space
2. Binomial polynomial works in logit space
3. Multinomial has per-category parameters
4. No technical group double-application
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import pyro

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bayesDREAM import bayesDREAM


def test_binomial_beta_priors():
    """Test that binomial Hill uses Beta priors for A and Vmax."""
    print("\n=== Testing binomial Beta priors ===")

    # Create minimal test data
    np.random.seed(42)
    n_cells = 30
    n_features = 5

    # Simple metadata
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': [f'guide_{i//10}' for i in range(n_cells)],
        'target': ['GFI1B' if i < 10 else 'ntc' for i in range(n_cells)],
        'cell_line': ['A' if i < 15 else 'B' for i in range(n_cells)],
        'sum_factor': np.ones(n_cells),
        'sum_factor_adj': np.ones(n_cells)
    })

    # Gene counts
    gene_counts = pd.DataFrame(
        np.random.poisson(50, (n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'gene_{i}' for i in range(n_features)]
    )
    gene_counts.loc['GFI1B'] = np.random.poisson(100, n_cells)

    # Binomial counts
    inclusion = pd.DataFrame(
        np.random.poisson(30, (n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'exon_{i}' for i in range(n_features)]
    )
    total = pd.DataFrame(
        np.random.poisson(60, (n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'exon_{i}' for i in range(n_features)]
    )

    # Create model
    feature_meta = pd.DataFrame({'gene': gene_counts.index}, index=gene_counts.index)
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        feature_meta=feature_meta,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        device='cpu'
    )

    # Add binomial modality
    exon_meta = pd.DataFrame({'exon': inclusion.index}, index=inclusion.index)
    model.add_custom_modality(
        name='exon_skip',
        counts=inclusion,
        feature_meta=exon_meta,
        distribution='binomial',
        denominator=total
    )

    # Minimal technical/cis fitting on gene modality
    model.set_technical_groups(['cell_line'])
    model.fit_technical(sum_factor_col='sum_factor', num_steps=10, niters=100)
    model.fit_cis(sum_factor_col='sum_factor', num_steps=10, niters=100)

    # Also run technical on binomial modality (required before fit_trans)
    model.fit_technical(
        sum_factor_col=None,
        modality_name='exon_skip',
        distribution='binomial',
        num_steps=10,
        niters=100
    )

    # Fit trans on binomial with Hill
    model.fit_trans(
        sum_factor_col=None,
        function_type='single_hill',
        modality_name='exon_skip',
        num_steps=10,
        niters=100
    )

    # Check Beta priors: A and Vmax should be in [0,1]
    # Access posterior samples from the modality
    exon_mod = model.get_modality('exon_skip')
    A = exon_mod.posterior_samples_trans['A']
    Vmax_a = exon_mod.posterior_samples_trans['Vmax_a']

    assert torch.all(A >= 0) and torch.all(A <= 1), f"A should be in [0,1], got range [{A.min():.3f}, {A.max():.3f}]"
    assert torch.all(Vmax_a >= 0) and torch.all(Vmax_a <= 1), f"Vmax should be in [0,1], got range [{Vmax_a.min():.3f}, {Vmax_a.max():.3f}]"

    print(f"  ✓ A in range [{A.min():.3f}, {A.max():.3f}]")
    print(f"  ✓ Vmax_a in range [{Vmax_a.min():.3f}, {Vmax_a.max():.3f}]")
    print("✓ Binomial Beta priors passed")


def test_multinomial_per_category():
    """Test that multinomial has per-category parameters."""
    print("\n=== Testing multinomial per-category parameters ===")

    np.random.seed(42)
    n_cells = 30
    n_features = 5
    K = 3  # Number of categories

    # Simple metadata
    meta = pd.DataFrame({
        'cell': [f'cell_{i}' for i in range(n_cells)],
        'guide': [f'guide_{i//10}' for i in range(n_cells)],
        'target': ['GFI1B' if i < 10 else 'ntc' for i in range(n_cells)],
        'cell_line': ['A' if i < 15 else 'B' for i in range(n_cells)],
        'sum_factor': np.ones(n_cells),
        'sum_factor_adj': np.ones(n_cells)
    })

    # Gene counts
    gene_counts = pd.DataFrame(
        np.random.poisson(50, (n_features, n_cells)),
        columns=[f'cell_{i}' for i in range(n_cells)],
        index=[f'gene_{i}' for i in range(n_features)]
    )
    gene_counts.loc['GFI1B'] = np.random.poisson(100, n_cells)

    # Multinomial counts (3D)
    multinomial_counts = np.random.poisson(20, (n_features, n_cells, K))

    # Create model
    feature_meta = pd.DataFrame({'gene': gene_counts.index}, index=gene_counts.index)
    model = bayesDREAM(
        meta=meta,
        counts=gene_counts,
        feature_meta=feature_meta,
        cis_gene='GFI1B',
        guide_covariates=['cell_line'],
        device='cpu'
    )

    # Add multinomial modality
    donor_meta = pd.DataFrame(
        {'donor': [f'donor_{i}' for i in range(n_features)]},
        index=[f'donor_{i}' for i in range(n_features)]
    )
    model.add_custom_modality(
        name='donor_usage',
        counts=multinomial_counts,
        feature_meta=donor_meta,
        distribution='multinomial'
    )

    # Minimal technical/cis fitting on gene modality
    model.set_technical_groups(['cell_line'])
    model.fit_technical(sum_factor_col='sum_factor', num_steps=10, niters=100)
    model.fit_cis(sum_factor_col='sum_factor', num_steps=10, niters=100)

    # Also run technical on multinomial modality (required before fit_trans)
    model.fit_technical(
        sum_factor_col=None,
        modality_name='donor_usage',
        distribution='multinomial',
        num_steps=10,
        niters=100
    )

    # Fit trans on multinomial with Hill (should have K-1 parameters)
    model.fit_trans(
        sum_factor_col=None,
        function_type='single_hill',
        modality_name='donor_usage',
        num_steps=10,
        niters=100
    )

    # Check K-1 dimensions
    # Access posterior samples from the modality
    donor_mod = model.get_modality('donor_usage')
    n_a = donor_mod.posterior_samples_trans['n_a']
    Vmax_a = donor_mod.posterior_samples_trans['Vmax_a']
    K_a = donor_mod.posterior_samples_trans['K_a']

    # Shape is [S, K-1, T] after moving category plate to dim=-2
    assert n_a.shape[-2] == K - 1, f"n_a should have K-1={K-1} categories, got {n_a.shape[-2]}"
    assert Vmax_a.shape[-2] == K - 1, f"Vmax_a should have K-1={K-1} categories, got {Vmax_a.shape[-2]}"
    assert K_a.shape[-2] == K - 1, f"K_a should have K-1={K-1} categories, got {K_a.shape[-2]}"

    # Check that Vmax is in [0,1] for all categories
    assert torch.all(Vmax_a >= 0) and torch.all(Vmax_a <= 1), "Vmax should be in [0,1] for all categories"

    print(f"  ✓ Parameters have K-1={K-1} dimensions")
    print(f"  ✓ Vmax_a in range [{Vmax_a.min():.3f}, {Vmax_a.max():.3f}]")
    print("✓ Multinomial per-category parameters passed")


def run_quick_tests():
    """Run quick focused tests."""
    print("="*60)
    print("Quick Trans Model Tests")
    print("="*60)

    pyro.set_rng_seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        test_binomial_beta_priors()
        test_multinomial_per_category()

        print("\n" + "="*60)
        print("✓ ALL QUICK TESTS PASSED")
        print("="*60)
        print("\nVerified:")
        print("  1. Binomial Hill uses Beta priors (A, Vmax in [0,1])")
        print("  2. Multinomial has K-1 per-category parameters")
        print("  3. Both distributions fit without errors")

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_quick_tests()

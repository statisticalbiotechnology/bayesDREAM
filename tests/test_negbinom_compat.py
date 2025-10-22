"""
Test backward compatibility with negbinom distribution.

This ensures the refactored code still works with the default negative binomial.
"""

import pandas as pd
import numpy as np
from bayesDREAM import bayesDREAM

print("Creating toy data...")

# Create minimal toy data
np.random.seed(42)
n_cells = 30
n_genes = 15

# Metadata (with 2 cell lines, both having NTC and gRNA cells)
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 8 + ['gRNA1'] * 7 + ['ntc'] * 7 + ['gRNA1'] * 8,  # Mixed NTC and gRNA in both cell lines
    'target': ['ntc'] * 8 + ['GFI1B'] * 7 + ['ntc'] * 7 + ['GFI1B'] * 8,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 15 + ['Jurkat'] * 15  # 2 cell lines
})

# Gene counts (including cis gene GFI1B)
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(50, (n_genes, n_cells)),
    index=gene_names,
    columns=meta['cell']
)

print(f"  - Metadata: {n_cells} cells")
print(f"  - Gene counts: {n_genes} genes × {n_cells} cells")

# Create model
print("\nCreating bayesDREAM...")
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='negbinom_compat_test'
)
print(f"✓ Model created successfully")

print("\nTesting fit_trans functionality...")
try:
    import torch

    # Set dummy technical groups (simpler test without correction)
    print("  Setting up technical groups...")
    model.set_technical_groups(['cell_line'])
    print("  ✓ Technical groups set")

    # Run minimal technical fit to get alpha_y_prefit
    print("  Running minimal technical fit (10 iterations)...")
    model.fit_technical(
        sum_factor_col='sum_factor',
        niters=10,  # Very short
        nsamples=10
    )
    print("  ✓ Technical fit complete")

    # Set dummy x_true so we can test the trans function logic
    print("  Setting dummy x_true for testing...")
    model.x_true = torch.ones(len(model.meta), device=model.device)
    model.x_true_type = 'point'
    model.log2_x_true = torch.log2(model.x_true)
    model.log2_x_true_type = 'point'

    print("  ✓ x_true set")

    print("\n  Testing fit_trans with negbinom (short run)...")
    model.fit_trans(
        sum_factor_col='sum_factor',  # Required for negbinom
        distribution='negbinom',  # Explicit
        function_type='single_hill',
        niters=50,  # Very short for testing
        lr=1e-2,
        p0=0.01,  # Required gamma parameters
        gamma_threshold=0.01,
        nsamples=10  # Very few samples for testing
    )
    print("  ✓ fit_trans completed successfully with negbinom")

    # Verify that posterior samples were created
    gene_modality = model.get_modality('gene')
    assert hasattr(gene_modality, 'posterior_samples_trans'), "Missing posterior_samples_trans in modality"
    assert gene_modality.posterior_samples_trans is not None, "posterior_samples_trans is None"
    print("  ✓ Posterior samples created")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Backward compatibility test completed!")
print("="*60)

"""
Test backward compatibility of fit_technical with negbinom distribution.

This ensures the refactored code still works with the default negative binomial.
"""

import pandas as pd
import numpy as np
from bayesDREAM import bayesDREAM

print("Creating toy data...")

# Create minimal toy data
np.random.seed(42)
n_cells = 40
n_genes = 15

# Metadata - include both NTC and perturbed cells
# IMPORTANT: All cell lines must have NTC cells for fit_technical
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 20 + ['gRNA1'] * 20,
    'target': ['ntc'] * 20 + ['GFI1B'] * 20,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 10 + ['HEK293T'] * 10 + ['K562'] * 10 + ['HEK293T'] * 10
})

# Gene counts (including cis gene GFI1B)
# Use higher counts to avoid numerical issues
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, n_cells)),  # Higher counts for stability
    index=gene_names,
    columns=meta['cell']
)

print(f"  - Metadata: {n_cells} cells")
print(f"  - Gene counts: {n_genes} genes × {n_cells} cells")
print(f"  - Cell lines: {meta['cell_line'].unique()}")
print(f"  - NTC cells: {(meta['target'] == 'ntc').sum()}")

# Create model
print("\nCreating bayesDREAM...")
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='technical_compat_test'
)
print(f"✓ Model created successfully")

print("\nTesting fit_technical functionality...")
try:
    import torch

    # Set up technical groups first (required)
    print("  Setting up technical groups...")
    model.set_technical_groups(['cell_line'])
    print("  ✓ Technical groups set")

    print("\n  Testing fit_technical with negbinom (short run)...")
    model.fit_technical(
        sum_factor_col='sum_factor',  # Required for negbinom
        distribution='negbinom',  # Explicit
        niters=50,  # Very short for testing
        nsamples=10,  # Few samples for testing
        lr=1e-2
    )
    print("  ✓ fit_technical completed successfully with negbinom")

    # Verify alpha_y_prefit was set in the modality
    gene_modality = model.get_modality('gene')
    assert hasattr(gene_modality, 'alpha_y_prefit'), "alpha_y_prefit not set in modality"
    assert gene_modality.alpha_y_prefit is not None, "alpha_y_prefit is None"
    print(f"  ✓ alpha_y_prefit shape: {gene_modality.alpha_y_prefit.shape}")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Technical fit backward compatibility test completed!")
print("="*60)

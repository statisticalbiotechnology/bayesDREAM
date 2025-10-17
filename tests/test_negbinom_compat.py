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

# Metadata
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 15 + ['gRNA1'] * 15,
    'target': ['ntc'] * 15 + ['GFI1B'] * 15,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * n_cells
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

print("\nTesting fit_trans signature compatibility...")
try:
    import torch
    import inspect

    # Check that fit_trans accepts the new parameters
    sig = inspect.signature(model.fit_trans)
    params = list(sig.parameters.keys())
    print(f"  fit_trans parameters: {params[:5]}...")  # Show first 5

    assert 'distribution' in params, "Missing 'distribution' parameter"
    assert 'denominator' in params, "Missing 'denominator' parameter"
    assert sig.parameters['distribution'].default == 'negbinom', "Wrong default for distribution"
    assert sig.parameters['sum_factor_col'].default is None, "sum_factor_col should default to None"

    print("✓ Signature is correct")

    # Set dummy x_true so we can test the function logic
    print("\nSetting dummy x_true for testing...")
    model.x_true = torch.ones(len(model.meta), device=model.device)
    model.x_true_type = 'point'
    model.log2_x_true = torch.log2(model.x_true)
    model.log2_x_true_type = 'point'
    model.alpha_y_prefit = None
    model.alpha_y_type = None

    print("✓ x_true set")

    print("\nTesting fit_trans with negbinom (short run)...")
    model.fit_trans(
        sum_factor_col='sum_factor',  # Required for negbinom
        distribution='negbinom',  # Explicit
        function_type='single_hill',
        niters=50,  # Very short for testing
        lr=1e-2,
        p0=0.01,  # Required gamma parameters
        gamma_threshold=0.01
    )
    print("✓ fit_trans completed successfully with negbinom")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Backward compatibility test completed!")
print("="*60)

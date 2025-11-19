"""
Test case: What if there are NO NTC cells in the data?
This would cause df_ntc to be empty, leading to KeyError.
"""
import pandas as pd
import numpy as np
from bayesDREAM import bayesDREAM

print("Creating data with NO NTC cells...")
np.random.seed(42)
n_cells = 20
n_genes = 15

# Metadata - NO ntc cells!
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['gRNA1'] * 20,
    'target': ['GFI1B'] * 20,  # All target GFI1B, no 'ntc'
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 10 + ['HEK293T'] * 10,
    'lane': ['lane1'] * 10 + ['lane2'] * 10
})

# Gene counts
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, n_cells)),
    index=gene_names,
    columns=meta['cell']
)

print("\n=== Creating model (should fail or warn) ===")
try:
    model = bayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene='GFI1B'
    )
    print("✓ Model created (unexpectedly!)")

    print("\n=== Now calling adjust_ntc_sum_factor ===")
    try:
        model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
        print("✓ adjust_ntc_sum_factor completed successfully!")
    except KeyError as e:
        print(f"✗ KeyError: {e}")
        print("This is the expected failure!")
except ValueError as e:
    print(f"✗ ValueError during init: {e}")
    print("Model initialization correctly rejected data without NTC cells.")

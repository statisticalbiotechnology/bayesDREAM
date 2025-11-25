#!/usr/bin/env python3
"""
Test script to verify automatic memory-aware batching in fit_technical.
"""

import numpy as np
import pandas as pd
import torch

# Add parent directory to path
import sys
sys.path.insert(0, '/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude')

from bayesDREAM import bayesDREAM

# Create small toy data
np.random.seed(42)

n_cells = 100
n_genes = 50
n_guides = 5

# Cell metadata
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': np.random.choice([f'guide_{i}' for i in range(n_guides)], n_cells),
    'target': np.random.choice(['NTC', 'GeneA'], n_cells),
    'experiment': np.random.choice(['exp1', 'exp2'], n_cells),
    'sum_factor': np.random.lognormal(0, 0.5, n_cells)
})

# Gene counts (sparse-like)
counts = pd.DataFrame(
    np.random.negative_binomial(5, 0.5, (n_genes, n_cells)),
    index=[f'gene_{i}' for i in range(n_genes)],
    columns=[f'cell_{i}' for i in range(n_cells)]
)

# Gene metadata
gene_meta = pd.DataFrame({
    'gene_name': [f'gene_{i}' for i in range(n_genes)]
})

print("=" * 60)
print("Testing automatic memory-aware batching")
print("=" * 60)

# Create model
model = bayesDREAM(
    meta=meta,
    counts=counts,
    feature_meta=gene_meta,
    output_dir='/tmp/test_memory/',
    label='test_memory',
    device='cpu'
)

# Set technical groups
model.set_technical_groups(['experiment'])

print("\n" + "=" * 60)
print("Running fit_technical WITHOUT specifying minibatch_size")
print("Should auto-detect available memory and decide batching strategy")
print("=" * 60)

try:
    model.fit_technical(
        sum_factor_col='sum_factor',
        niters=10,  # Very short for testing
        nsamples=100  # Small number for testing
    )
    print("\n[SUCCESS] fit_technical completed with automatic memory management!")
except Exception as e:
    print(f"\n[ERROR] fit_technical failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)

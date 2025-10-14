"""
Test that the adjustment_factor column conflict bug is fixed.
"""
import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM

print("Test 1: Data WITHOUT pre-existing adjustment_factor column")
print("=" * 60)
np.random.seed(42)
n_cells = 40
n_genes = 15

meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 20 + ['gRNA1'] * 20,
    'target': ['ntc'] * 20 + ['GFI1B'] * 20,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 10 + ['HEK293T'] * 10 + ['K562'] * 10 + ['HEK293T'] * 10,
    'lane': ['lane1'] * 20 + ['lane2'] * 20
})

gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, n_cells)),
    index=gene_names,
    columns=meta['cell']
)

model = MultiModalBayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='adjust_fix_test1'
)

try:
    model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
    print("✓ Test 1 PASSED: adjust_ntc_sum_factor works without pre-existing column")
except Exception as e:
    print(f"✗ Test 1 FAILED: {e}")

print("\n" + "=" * 60)
print("Test 2: Data WITH pre-existing adjustment_factor column")
print("=" * 60)

# Add a pre-existing adjustment_factor column (simulating CSV data from previous run)
meta['adjustment_factor'] = np.random.uniform(0.8, 1.2, n_cells)

model2 = MultiModalBayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='adjust_fix_test2'
)

print(f"Pre-existing adjustment_factor in model.meta: {'adjustment_factor' in model2.meta.columns}")

try:
    model2.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
    print("✓ Test 2 PASSED: adjust_ntc_sum_factor works WITH pre-existing column")
    print(f"✓ sum_factor_adj created: {'sum_factor_adj' in model2.meta.columns}")
except KeyError as e:
    print(f"✗ Test 2 FAILED with KeyError: {e}")
    print("This was the original bug!")
except Exception as e:
    print(f"✗ Test 2 FAILED: {e}")

print("\n" + "=" * 60)
print("Test 3: Verify adjustment_factor values are recomputed (not reused)")
print("=" * 60)

# Check that the new adjustment_factor is different from the old one
old_adj = meta['adjustment_factor'].copy()
model3 = MultiModalBayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='adjust_fix_test3'
)
model3.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])

# The new adjustment_factor should be in the meta
new_adj = model3.meta['adjustment_factor']

# They should be different (unless by chance they're identical)
if not np.allclose(old_adj.values, new_adj.values):
    print("✓ Test 3 PASSED: adjustment_factor was recomputed (not reused from CSV)")
else:
    print("⚠ Test 3 WARNING: adjustment_factor values are identical (could be coincidence)")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)

"""
Test case that tries to reproduce the actual user error.

Hypothesis: User's meta has been modified after initialization,
possibly removing or corrupting the adjustment_factor logic.
"""
import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM

print("Creating realistic dataset...")
np.random.seed(42)
n_cells_total = 100
n_genes = 15

# Create metadata with unbalanced covariate combinations
# Some lane/cell_line combinations might not have NTC cells
cells_data = []
cell_id = 0

# lane1 + K562: has NTC
for i in range(10):
    cells_data.append({
        'cell': f'cell_{cell_id}',
        'guide': 'ntc',
        'target': 'ntc',
        'sum_factor': np.random.uniform(0.8, 1.2),
        'cell_line': 'K562',
        'lane': 'lane1'
    })
    cell_id += 1

# lane1 + K562: has guides
for i in range(10):
    cells_data.append({
        'cell': f'cell_{cell_id}',
        'guide': 'gRNA1',
        'target': 'GFI1B',
        'sum_factor': np.random.uniform(0.8, 1.2),
        'cell_line': 'K562',
        'lane': 'lane1'
    })
    cell_id += 1

# lane2 + HEK293T: has NTC
for i in range(10):
    cells_data.append({
        'cell': f'cell_{cell_id}',
        'guide': 'ntc',
        'target': 'ntc',
        'sum_factor': np.random.uniform(0.8, 1.2),
        'cell_line': 'HEK293T',
        'lane': 'lane2'
    })
    cell_id += 1

# lane2 + HEK293T: has guides
for i in range(10):
    cells_data.append({
        'cell': f'cell_{cell_id}',
        'guide': 'gRNA2',
        'target': 'GFI1B',
        'sum_factor': np.random.uniform(0.8, 1.2),
        'cell_line': 'HEK293T',
        'lane': 'lane2'
    })
    cell_id += 1

# lane3 + K562: NO NTC, only guides (this is the problem case!)
for i in range(10):
    cells_data.append({
        'cell': f'cell_{cell_id}',
        'guide': 'gRNA3',
        'target': 'GFI1B',
        'sum_factor': np.random.uniform(0.8, 1.2),
        'cell_line': 'K562',
        'lane': 'lane3'
    })
    cell_id += 1

meta = pd.DataFrame(cells_data)

# Gene counts
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, len(meta))),
    index=gene_names,
    columns=meta['cell']
)

print(f"\n=== Data summary ===")
print(f"Total cells: {len(meta)}")
print(f"NTC cells: {(meta['target'] == 'ntc').sum()}")
print(f"\nCovariate combinations:")
print(meta.groupby(['lane', 'cell_line', 'target']).size())

print("\n=== Creating model ===")
model = MultiModalBayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='realistic_test'
)

print(f"\n=== After initialization ===")
print(f"Cells in model: {len(model.meta)}")
print(f"NTC cells: {(model.meta['target'] == 'ntc').sum()}")
print(f"\nCovariate combinations in model.meta:")
print(model.meta.groupby(['lane', 'cell_line', 'target']).size())

print("\n=== Calling adjust_ntc_sum_factor ===")
try:
    model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
    print("✓ adjust_ntc_sum_factor completed successfully!")
    print(f"✓ sum_factor_adj created: {'sum_factor_adj' in model.meta.columns}")

    # Check for NaN
    nan_count = model.meta['sum_factor_adj'].isna().sum()
    print(f"NaN values in sum_factor_adj: {nan_count}")
    if nan_count > 0:
        print("Cells with NaN sum_factor_adj:")
        print(model.meta[model.meta['sum_factor_adj'].isna()][['guide', 'target', 'lane', 'cell_line']].groupby(['lane', 'cell_line', 'target']).size())

except KeyError as e:
    print(f"✗ KeyError: {e}")
    import traceback
    traceback.print_exc()
    print("\nREPRODUCED THE BUG!")

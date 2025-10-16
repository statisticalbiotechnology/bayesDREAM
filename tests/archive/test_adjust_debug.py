"""
Debug test for adjust_ntc_sum_factor() KeyError issue.
"""
import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM

print("Creating toy data...")
np.random.seed(42)
n_cells = 60
n_genes = 15

# Metadata with both lane and cell_line covariates
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 20 + ['gRNA1'] * 20 + ['gRNA2'] * 20,
    'target': ['ntc'] * 20 + ['GFI1B'] * 20 + ['gene_1'] * 20,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 15 + ['HEK293T'] * 15 + ['K562'] * 15 + ['HEK293T'] * 15,
    'lane': ['lane1'] * 30 + ['lane2'] * 30
})

# Gene counts
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, n_cells)),
    index=gene_names,
    columns=meta['cell']
)

print("\n=== Creating model ===")
model = MultiModalBayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_output',
    label='adjust_debug_test'
)

print("\n=== Checking meta columns ===")
print(f"Columns in model.meta: {model.meta.columns.tolist()}")
print(f"'guide_used' in meta: {'guide_used' in model.meta.columns}")
print(f"'lane' in meta: {'lane' in model.meta.columns}")
print(f"'cell_line' in meta: {'cell_line' in model.meta.columns}")

print("\n=== Checking NTC cells ===")
ntc_rows = model.meta[model.meta['target'] == 'ntc']
print(f"Number of NTC cells: {len(ntc_rows)}")
print(f"NTC covariates (lane, cell_line):")
print(ntc_rows[['lane', 'cell_line', 'sum_factor']].head())

print("\n=== Manually testing the logic ===")
meta_out = model.meta.copy()
covariates = ["lane", "cell_line"]
sum_factor_col_old = "sum_factor"

# Step 1: Mean sum_factor among NTC rows, grouped by covariates
df_ntc = (
    meta_out.loc[meta_out["target"] == "ntc"]
    .groupby(covariates)[sum_factor_col_old]
    .mean()
    .reset_index(name="mean_SumFacs_ntc")
)
print("\ndf_ntc (NTC means by lane/cell_line):")
print(df_ntc)

# Step 2: Mean sum_factor among all guides, grouped by covariates + guide_used
df_guide = (
    meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
    .mean()
    .reset_index(name="mean_SumFacs_guide")
)
print("\ndf_guide (guide means by lane/cell_line/guide_used):")
print(df_guide)

# Step 3: Merge and compute adjustment_factor
merged = pd.merge(df_guide, df_ntc, on=covariates, how="left")
print("\nmerged (after joining df_guide and df_ntc):")
print(merged)
print(f"\nColumns in merged: {merged.columns.tolist()}")

# Check if adjustment_factor can be created
merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
print("\nmerged with adjustment_factor:")
print(merged)

# Step 4: Merge back onto meta_out
merge_cols = covariates + ["guide_used", "adjustment_factor"]
print(f"\nAttempting merge with cols: {merge_cols}")
meta_out_merged = pd.merge(meta_out, merged[merge_cols], on=covariates + ["guide_used"], how="left")
print(f"\nColumns in meta_out after merge: {meta_out_merged.columns.tolist()}")
print(f"'adjustment_factor' in meta_out: {'adjustment_factor' in meta_out_merged.columns}")

if 'adjustment_factor' in meta_out_merged.columns:
    print("\n✓ adjustment_factor column exists!")
    print(f"Number of NaN values: {meta_out_merged['adjustment_factor'].isna().sum()}")
else:
    print("\n✗ adjustment_factor column is MISSING!")

print("\n=== Now calling the actual method ===")
try:
    model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
    print("✓ adjust_ntc_sum_factor completed successfully!")
    print(f"✓ sum_factor_adj created: {'sum_factor_adj' in model.meta.columns}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

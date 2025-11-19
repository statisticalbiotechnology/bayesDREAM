"""
Test case that reproduces the KeyError in adjust_ntc_sum_factor().

Scenario: Model initialized with guide_covariates=['cell_line'] (default),
but adjust_ntc_sum_factor called with covariates=['lane', 'cell_line'].
"""
import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM

print("Creating toy data with mismatched covariates...")
np.random.seed(42)
n_cells = 40
n_genes = 15

# Metadata - only target='ntc' or target=cis_gene will be kept after initialization
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 20 + ['gRNA1'] * 20,
    'target': ['ntc'] * 20 + ['GFI1B'] * 20,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 10 + ['HEK293T'] * 10 + ['K562'] * 10 + ['HEK293T'] * 10,
    'lane': ['lane1'] * 20 + ['lane2'] * 20
})

# Gene counts
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, n_cells)),
    index=gene_names,
    columns=meta['cell']
)

print("\n=== Creating model with guide_covariates=['cell_line'] (DEFAULT) ===")
model = MultiModalBayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    guide_covariates=['cell_line'],  # Explicitly set to default
    output_dir='./test_output',
    label='adjust_failure_test'
)

print(f"\n=== Checking guide_used construction ===")
print(f"Sample guide_used values:")
print(model.meta[['guide', 'target', 'cell_line', 'lane', 'guide_used']].head(10))

print("\n=== Now calling adjust_ntc_sum_factor with covariates=['lane', 'cell_line'] ===")
try:
    model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
    print("✓ adjust_ntc_sum_factor completed successfully!")
    print(f"✓ sum_factor_adj created: {'sum_factor_adj' in model.meta.columns}")
except KeyError as e:
    print(f"✗ KeyError: {e}")
    import traceback
    traceback.print_exc()

    # Additional debugging
    print("\n=== Debugging the issue ===")
    meta_out = model.meta.copy()
    covariates = ["lane", "cell_line"]
    sum_factor_col_old = "sum_factor"

    print("\nStep 1: df_ntc")
    df_ntc = (
        meta_out.loc[meta_out["target"] == "ntc"]
        .groupby(covariates)[sum_factor_col_old]
        .mean()
        .reset_index(name="mean_SumFacs_ntc")
    )
    print(df_ntc)

    print("\nStep 2: df_guide")
    df_guide = (
        meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
        .mean()
        .reset_index(name="mean_SumFacs_guide")
    )
    print(df_guide)

    print("\nStep 3: merged")
    merged = pd.merge(df_guide, df_ntc, on=covariates, how="left")
    merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
    print(merged)

    print("\nStep 4: Attempting merge back onto meta_out")
    merge_cols = covariates + ["guide_used", "adjustment_factor"]
    print(f"Merge keys: {covariates + ['guide_used']}")
    print(f"Columns to merge: {merge_cols}")

    # Check if guide_used values align
    guide_used_in_meta = set(meta_out['guide_used'].unique())
    guide_used_in_merged = set(merged['guide_used'].unique())
    print(f"\nguide_used in meta_out: {guide_used_in_meta}")
    print(f"guide_used in merged: {guide_used_in_merged}")
    print(f"Overlap: {guide_used_in_meta & guide_used_in_merged}")

    meta_out_test = pd.merge(meta_out, merged[merge_cols], on=covariates + ["guide_used"], how="left")
    print(f"\nColumns after merge: {meta_out_test.columns.tolist()}")
    print(f"'adjustment_factor' in meta_out: {'adjustment_factor' in meta_out_test.columns}")

    if 'adjustment_factor' not in meta_out_test.columns:
        print("\n✗ CONFIRMED: adjustment_factor column is NOT created by the merge!")
        print("This confirms the bug: merge is failing due to mismatched keys.")

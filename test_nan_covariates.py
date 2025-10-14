"""
Test case: What if covariates have NaN values?
This could cause the merge to fail.
"""
import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM

print("Creating data with NaN in 'lane' covariate...")
np.random.seed(42)
n_cells = 40
n_genes = 15

# Metadata with NaN in 'lane' column for some cells
meta = pd.DataFrame({
    'cell': [f'cell_{i}' for i in range(n_cells)],
    'guide': ['ntc'] * 20 + ['gRNA1'] * 20,
    'target': ['ntc'] * 20 + ['GFI1B'] * 20,
    'sum_factor': np.random.uniform(0.5, 1.5, n_cells),
    'cell_line': ['K562'] * 10 + ['HEK293T'] * 10 + ['K562'] * 10 + ['HEK293T'] * 10,
    'lane': [np.nan] * 10 + ['lane1'] * 10 + [np.nan] * 10 + ['lane2'] * 10  # NaN for half the cells
})

# Gene counts
gene_names = ['GFI1B'] + [f'gene_{i}' for i in range(n_genes - 1)]
counts = pd.DataFrame(
    np.random.poisson(200, (n_genes, n_cells)),
    index=gene_names,
    columns=meta['cell']
)

print("\n=== Creating model ===")
try:
    model = MultiModalBayesDREAM(
        meta=meta,
        counts=counts,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='nan_covariate_test'
    )
    print("✓ Model created successfully")

    print("\n=== Checking meta ===")
    print(f"Number of NaN in 'lane': {model.meta['lane'].isna().sum()}")
    print(f"Sample rows:")
    print(model.meta[['guide', 'target', 'cell_line', 'lane', 'guide_used']].head(15))

    print("\n=== Now calling adjust_ntc_sum_factor with covariates=['lane', 'cell_line'] ===")
    try:
        model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
        print("✓ adjust_ntc_sum_factor completed successfully!")
        print(f"✓ sum_factor_adj created: {'sum_factor_adj' in model.meta.columns}")

        # Check for NaN in sum_factor_adj
        nan_count = model.meta['sum_factor_adj'].isna().sum()
        print(f"Number of NaN in 'sum_factor_adj': {nan_count}")
        if nan_count > 0:
            print("⚠ Warning: sum_factor_adj contains NaN values!")

    except KeyError as e:
        print(f"✗ KeyError: {e}")
        import traceback
        traceback.print_exc()

        # Debug
        print("\n=== Debugging ===")
        meta_out = model.meta.copy()
        covariates = ["lane", "cell_line"]
        sum_factor_col_old = "sum_factor"

        df_ntc = (
            meta_out.loc[meta_out["target"] == "ntc"]
            .groupby(covariates)[sum_factor_col_old]
            .mean()
            .reset_index(name="mean_SumFacs_ntc")
        )
        print(f"\ndf_ntc shape: {df_ntc.shape}")
        print(df_ntc)

        df_guide = (
            meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
            .mean()
            .reset_index(name="mean_SumFacs_guide")
        )
        print(f"\ndf_guide shape: {df_guide.shape}")
        print(df_guide)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

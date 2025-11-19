"""
Test adjust_ntc_sum_factor with actual toy data.
"""
import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM

print("Loading toy data...")
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print(f"Original data: {len(meta)} cells, {len(counts)} genes")
print(f"Targets: {meta['target'].unique()}")

# Subset to GFI1B and ntc only (as the user described)
print("\nSubsetting to GFI1B and ntc only...")
meta_subset = meta[meta['target'].isin(['GFI1B', 'ntc'])].copy()
print(f"After subsetting: {len(meta_subset)} cells")
print(f"\nLane/cell_line/target combinations:")
print(meta_subset.groupby(['lane', 'cell_line', 'target']).size())

# Subset counts to match (use 'cell' column which should already exist)
print("\nPreparing data...")
print(f"'cell' column exists: {'cell' in meta_subset.columns}")
print(f"First few cell values: {meta_subset['cell'].head().tolist()}")
print(f"First few count column names: {counts.columns[:5].tolist()}")

# Check overlap
cells_in_meta = set(meta_subset['cell'])
cells_in_counts = set(counts.columns)
overlap = cells_in_meta & cells_in_counts
print(f"Cells in meta: {len(cells_in_meta)}")
print(f"Cells in counts: {len(cells_in_counts)}")
print(f"Overlap: {len(overlap)}")

if len(overlap) == 0:
    print("ERROR: No overlap between meta['cell'] and counts.columns!")
    print("This means the test data doesn't match properly.")
    import sys
    sys.exit(1)

counts_subset = counts[list(overlap)]
meta_subset = meta_subset[meta_subset['cell'].isin(overlap)].copy()
print(f"After matching: {len(meta_subset)} cells, counts shape: {counts_subset.shape}")

print("\n=== Creating model ===")
try:
    model = MultiModalBayesDREAM(
        meta=meta_subset,
        counts=counts_subset,
        cis_gene='GFI1B',
        output_dir='./test_output',
        label='toydata_test'
    )
    print("✓ Model created successfully")
    print(f"Cells in model: {len(model.meta)}")
    print(f"Genes in model: {len(model.counts)}")

    print(f"\n=== Checking model.meta ===")
    print(f"Columns: {model.meta.columns.tolist()}")
    print(f"NTC cells: {(model.meta['target'] == 'ntc').sum()}")
    print(f"\nLane/cell_line/target in model:")
    print(model.meta.groupby(['lane', 'cell_line', 'target']).size())

    print("\n=== Calling adjust_ntc_sum_factor(covariates=['lane', 'cell_line']) ===")
    try:
        model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
        print("✓ adjust_ntc_sum_factor completed successfully!")
        print(f"✓ 'sum_factor_adj' in meta: {'sum_factor_adj' in model.meta.columns}")

        # Check for NaN
        if 'sum_factor_adj' in model.meta.columns:
            nan_count = model.meta['sum_factor_adj'].isna().sum()
            print(f"\nNaN values in sum_factor_adj: {nan_count}")
            if nan_count > 0:
                print("⚠ Warning: Some cells have NaN sum_factor_adj")
                print(model.meta[model.meta['sum_factor_adj'].isna()][['lane', 'cell_line', 'target', 'guide_used']].head(20))

    except KeyError as e:
        print(f"✗ KeyError: {e}")
        import traceback
        traceback.print_exc()

        print("\n=== REPRODUCED THE BUG! Debugging... ===")

        # Manual debug
        meta_out = model.meta.copy()
        covariates = ["lane", "cell_line"]
        sum_factor_col_old = "sum_factor"

        print(f"\nStep 1: Create df_ntc")
        df_ntc = (
            meta_out.loc[meta_out["target"] == "ntc"]
            .groupby(covariates)[sum_factor_col_old]
            .mean()
            .reset_index(name="mean_SumFacs_ntc")
        )
        print(f"df_ntc shape: {df_ntc.shape}")
        print(df_ntc)

        print(f"\nStep 2: Create df_guide")
        df_guide = (
            meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
            .mean()
            .reset_index(name="mean_SumFacs_guide")
        )
        print(f"df_guide shape: {df_guide.shape}")
        print(df_guide.head(20))

        print(f"\nStep 3: Merge df_guide and df_ntc")
        merged = pd.merge(df_guide, df_ntc, on=covariates, how="left")
        print(f"merged shape: {merged.shape}")
        print(f"Columns in merged: {merged.columns.tolist()}")
        print(merged.head(20))

        print(f"\nStep 4: Create adjustment_factor")
        if 'mean_SumFacs_ntc' in merged.columns:
            merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
            print("✓ adjustment_factor created")
            print(f"adjustment_factor in merged: {'adjustment_factor' in merged.columns}")
        else:
            print("✗ mean_SumFacs_ntc column is missing!")

        print(f"\nStep 5: Merge back onto meta_out")
        merge_cols = covariates + ["guide_used", "adjustment_factor"]
        print(f"merge_cols: {merge_cols}")

        try:
            meta_out_test = pd.merge(meta_out, merged[merge_cols], on=covariates + ["guide_used"], how="left")
            print(f"✓ Merge succeeded")
            print(f"adjustment_factor in meta_out: {'adjustment_factor' in meta_out_test.columns}")
        except KeyError as e2:
            print(f"✗ Merge failed with KeyError: {e2}")
            print("This means merged doesn't have the expected columns!")

except Exception as e:
    print(f"✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

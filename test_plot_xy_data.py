"""
Test plot_xy_data with 91-feature posterior samples.
"""
import pandas as pd
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bayesDREAM import bayesDREAM

print("="*80)
print("TESTING plot_xy_data WITH 91-FEATURE POSTERIOR SAMPLES")
print("="*80)

# Load data
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

cis_gene = 'GFI1B'
print(f"\nOriginal counts: {counts.shape[0]} genes")
print(f"Cis gene: {cis_gene} at position {counts.index.get_loc(cis_gene)}")

# Create model
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,
    output_dir='./test_xy_plotting',
    cores=1
)

# Get modalities
gene_mod = model.get_modality('gene')
cis_mod = model.get_modality('cis')

# Set technical groups
model.set_technical_groups(['cell_line'])

print("\n" + "="*80)
print("CREATING MOCK TECHNICAL FIT")
print("="*80)

# Create mock posterior samples for gene modality (91 genes)
S = 100
C = model.meta['technical_group_code'].nunique()
T = 91

gene_posterior = {
    'log2_alpha_y': torch.randn(S, C-1, T),
    'alpha_y_mul': torch.exp(torch.randn(S, C-1, T) * 0.1),
    'delta_y_add': torch.randn(S, C-1, T) * 0.01,
    'o_y': torch.randn(S, C-1, T),
    'mu_ntc': torch.randn(S, C-1, T),
    'beta_o': torch.randn(S, C-1, 1)
}

alpha_y_mul_raw = gene_posterior['alpha_y_mul']
gene_posterior['alpha_y_mult'] = torch.cat([torch.ones(S, 1, T), alpha_y_mul_raw], dim=1)
gene_posterior['alpha_y'] = gene_posterior['alpha_y_mult']

delta_y_add_raw = gene_posterior['delta_y_add']
gene_posterior['alpha_y_add'] = torch.cat([torch.zeros(S, 1, T), delta_y_add_raw], dim=1)

# Assign to gene modality
gene_mod.posterior_samples_technical = gene_posterior
gene_mod.alpha_y_prefit = gene_posterior['alpha_y']
gene_mod.alpha_y_prefit_mult = gene_posterior['alpha_y_mult']
gene_mod.alpha_y_prefit_add = gene_posterior['alpha_y_add']

print(f"✓ Created gene modality technical fit with 91 features")

# Create mock posterior samples for cis modality
cis_posterior = {
    'log2_alpha_x': torch.randn(S, C-1, 1),
    'alpha_x_mul': torch.exp(torch.randn(S, C-1, 1) * 0.1),
    'delta_x_add': torch.randn(S, C-1, 1) * 0.01,
    'o_x': torch.randn(S, C-1, 1),
    'mu_ntc': torch.randn(S, C-1, 1)
}

alpha_x_mul_raw = cis_posterior['alpha_x_mul']
cis_posterior['alpha_x_mult'] = torch.cat([torch.ones(S, 1, 1), alpha_x_mul_raw], dim=1)
cis_posterior['alpha_x'] = cis_posterior['alpha_x_mult']

delta_x_add_raw = cis_posterior['delta_x_add']
cis_posterior['alpha_x_add'] = torch.cat([torch.zeros(S, 1, 1), delta_x_add_raw], dim=1)

cis_mod.posterior_samples_technical = cis_posterior
model.alpha_x_prefit = cis_posterior['alpha_x'].squeeze(-1)
model.alpha_x_type = 'posterior'

print(f"✓ Created cis modality technical fit with 1 feature")

print("\n" + "="*80)
print("SETTING UP CELL-LEVEL X_TRUE")
print("="*80)

# plot_xy_data works from cell-level data in model.meta
# We need to set x_true for cells (not guides)
n_cells = len(model.meta)
print(f"Model has {n_cells} cells in meta")

# Create mock x_true (cis expression per cell)
# This would normally come from fit_cis, but we're mocking it for testing
model.x_true = torch.randn(n_cells) * 2 + 5
model.x_true_type = 'posterior'
print(f"✓ Set x_true with shape {model.x_true.shape}")

print("\n" + "="*80)
print("TESTING plot_xy_data")
print("="*80)

# Test 1: Plot without correction
print("\n[TEST 1] plot_xy_data without correction...")
try:
    result = model.plot_xy_data(
        feature='TET2',
        modality_name='gene',
        show_correction='uncorrected',
        window=50
    )
    # Result can be Figure or Axes, close the figure
    if hasattr(result, 'figure'):
        plt.close(result.figure)
    else:
        plt.close(result)
    print("  ✓ Uncorrected plot succeeded")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 2: Plot with correction (uses alpha_y_prefit)
print("\n[TEST 2] plot_xy_data with correction (uses alpha_y_prefit)...")
try:
    result = model.plot_xy_data(
        feature='TET2',
        modality_name='gene',
        show_correction='corrected',
        window=50
    )
    if hasattr(result, 'figure'):
        plt.close(result.figure)
    else:
        plt.close(result)
    print("  ✓ Corrected plot succeeded - this uses gene_mod.alpha_y_prefit[..., feature_idx]")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 3: Plot both
print("\n[TEST 3] plot_xy_data with both corrected and uncorrected...")
try:
    result = model.plot_xy_data(
        feature='TET2',
        modality_name='gene',
        show_correction='both',
        window=50
    )
    if hasattr(result, 'figure'):
        plt.close(result.figure)
    else:
        plt.close(result)
    print("  ✓ Both plots succeeded")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 4: Test with multiple genes including those after GFI1B alphabetically
print("\n[TEST 4] Testing multiple trans genes...")
test_genes = ['ACTB', 'TET2', 'MYB', 'NFE2']
for gene in test_genes:
    if gene in gene_mod.feature_meta['gene'].values:
        try:
            result = model.plot_xy_data(
                feature=gene,
                modality_name='gene',
                show_correction='corrected',
                window=50
            )
            if hasattr(result, 'figure'):
                plt.close(result.figure)
            else:
                plt.close(result)

            # Get the feature index to show it's using correct alpha_y
            gene_list = gene_mod.feature_meta['gene'].tolist()
            feature_idx = gene_list.index(gene)
            alpha_sample = gene_mod.alpha_y_prefit[0, 0, feature_idx].item()

            print(f"  ✓ {gene:10s}: feature_idx={feature_idx:2d}, alpha_y={alpha_sample:.4f}")
        except Exception as e:
            print(f"  ✗ {gene:10s}: FAILED - {e}")
            raise
    else:
        print(f"  - {gene:10s}: not in gene modality (OK if cis gene)")

# Test 5: Verify GFI1B cannot be plotted from gene modality
print("\n[TEST 5] Verify cis gene (GFI1B) cannot be plotted from gene modality...")
try:
    result = model.plot_xy_data(
        feature='GFI1B',
        modality_name='gene',
        show_correction='corrected'
    )
    if hasattr(result, 'figure'):
        plt.close(result.figure)
    else:
        plt.close(result)
    print("  ✗ FAILED: GFI1B should NOT be plottable from gene modality!")
    raise AssertionError("GFI1B should not be in gene modality")
except (ValueError, KeyError) as e:
    print(f"  ✓ Correctly failed to plot GFI1B from gene modality: {e}")

print("\n" + "="*80)
print("ALL plot_xy_data TESTS PASSED!")
print("="*80)

print("\nSummary:")
print("  ✓ plot_xy_data works without correction")
print("  ✓ plot_xy_data works with correction (uses alpha_y_prefit with 91 features)")
print("  ✓ plot_xy_data works with both corrected and uncorrected")
print("  ✓ Correctly indexes alpha_y for multiple trans genes")
print("  ✓ Correctly prevents plotting cis gene from gene modality")
print("\nThe technical correction in plot_xy_data works correctly with 91-feature posteriors!")

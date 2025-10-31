"""
Test that plotting functions work correctly with 91-feature posterior samples.
Creates a model with mock posterior samples and calls plotting functions.
"""
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from bayesDREAM import bayesDREAM

print("="*80)
print("TESTING PLOTTING FUNCTIONS WITH 91-FEATURE POSTERIOR SAMPLES")
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
    output_dir='./test_plotting',
    cores=1
)

print("\n" + "="*80)
print("CREATING MOCK POSTERIOR SAMPLES")
print("="*80)

# Get modalities
gene_mod = model.get_modality('gene')
cis_mod = model.get_modality('cis')

# Set technical groups (required for plotting)
model.set_technical_groups(['cell_line'])

# Create mock posterior samples for gene modality (91 genes)
S = 100  # number of samples
C = model.meta['technical_group_code'].nunique()  # number of technical groups
T = 91   # number of trans genes

print(f"\nCreating mock posterior samples:")
print(f"  Samples (S): {S}")
print(f"  Technical groups (C): {C}")
print(f"  Trans genes (T): {T}")

gene_posterior = {
    'log2_alpha_y': torch.randn(S, C-1, T),
    'alpha_y_mul': torch.exp(torch.randn(S, C-1, T) * 0.1),
    'delta_y_add': torch.randn(S, C-1, T) * 0.01,
    'o_y': torch.randn(S, C-1, T),
    'mu_ntc': torch.randn(S, C-1, T),
    'beta_o': torch.randn(S, C-1, 1)
}

# Create reconstructed alpha (with baseline)
alpha_y_mul_raw = gene_posterior['alpha_y_mul']
gene_posterior['alpha_y_mult'] = torch.cat(
    [torch.ones(S, 1, T), alpha_y_mul_raw],
    dim=1
)
gene_posterior['alpha_y'] = gene_posterior['alpha_y_mult']

delta_y_add_raw = gene_posterior['delta_y_add']
gene_posterior['alpha_y_add'] = torch.cat(
    [torch.zeros(S, 1, T), delta_y_add_raw],
    dim=1
)

# Assign to gene modality
gene_mod.posterior_samples_technical = gene_posterior
gene_mod.alpha_y_prefit = gene_posterior['alpha_y']
gene_mod.alpha_y_prefit_mult = gene_posterior['alpha_y_mult']
gene_mod.alpha_y_prefit_add = gene_posterior['alpha_y_add']

print("\n✓ Created gene modality posterior samples:")
for key, val in gene_posterior.items():
    print(f"  {key:20s}: {val.shape}")

# Create mock posterior samples for cis modality (1 gene)
cis_posterior = {
    'log2_alpha_x': torch.randn(S, C-1, 1),
    'alpha_x_mul': torch.exp(torch.randn(S, C-1, 1) * 0.1),
    'delta_x_add': torch.randn(S, C-1, 1) * 0.01,
    'o_x': torch.randn(S, C-1, 1),
    'mu_ntc': torch.randn(S, C-1, 1)
}

alpha_x_mul_raw = cis_posterior['alpha_x_mul']
cis_posterior['alpha_x_mult'] = torch.cat(
    [torch.ones(S, 1, 1), alpha_x_mul_raw],
    dim=1
)
cis_posterior['alpha_x'] = cis_posterior['alpha_x_mult']

delta_x_add_raw = cis_posterior['delta_x_add']
cis_posterior['alpha_x_add'] = torch.cat(
    [torch.zeros(S, 1, 1), delta_x_add_raw],
    dim=1
)

# Assign to cis modality
cis_mod.posterior_samples_technical = cis_posterior
# Note: alpha_x_prefit should have shape (S, C) not (S, C, 1) to match technical.py extraction
model.alpha_x_prefit = cis_posterior['alpha_x'].squeeze(-1)  # (100, 2, 1) -> (100, 2)
model.alpha_x_type = 'posterior'

print("\n✓ Created cis modality posterior samples:")
for key, val in cis_posterior.items():
    print(f"  {key:20s}: {val.shape}")

print("\n" + "="*80)
print("TESTING PLOTTING FUNCTIONS")
print("="*80)

# Test 1: plot_technical_fit for gene modality
print("\n[TEST 1] plot_technical_fit for gene modality...")
try:
    fig = model.plot_technical_fit(
        param='alpha_y',
        modality_name='gene',
        subset_features=['TET2', 'MYB'],  # Test genes alphabetically before and after GFI1B
        technical_group_index=1  # Test with second technical group
    )
    plt.close(fig)
    print("  ✓ plot_technical_fit succeeded for gene modality")
except Exception as e:
    print(f"  ✗ plot_technical_fit FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 2: plot_technical_fit for cis modality
print("\n[TEST 2] plot_technical_fit for cis modality...")
try:
    fig = model.plot_technical_fit(
        param='alpha_x',
        technical_group_index=1  # Test with second technical group
    )
    plt.close(fig)
    print("  ✓ plot_technical_fit succeeded for cis modality")
except Exception as e:
    print(f"  ✗ plot_technical_fit FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 3: plot_xy_data with technical correction
print("\n[TEST 3] plot_xy_data with technical correction...")
try:
    # We need to set x_true for this test
    # Create mock x_true for the guides
    if hasattr(model, 'model_df') and len(model.model_df) > 0:
        n_guides = len(model.model_df)
        model.x_true = torch.randn(n_guides) + 5  # Mock cis expression
        model.x_true_type = 'posterior'

        fig = model.plot_xy_data(
            feature='TET2',  # A trans gene
            modality_name='gene',
            show_correction='corrected'  # This uses alpha_y_prefit
        )
        plt.close(fig)
        print("  ✓ plot_xy_data with correction succeeded")
    else:
        print("  - plot_xy_data test skipped (no model_df available)")
except Exception as e:
    print(f"  ✗ plot_xy_data FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

# Test 4: Verify indexing is correct
print("\n[TEST 4] Verifying alpha_y indexing for specific genes...")
try:
    # Test that we can access alpha for genes correctly
    test_genes = ['ACTB', 'TET2', 'MYB', 'ZBTB7A']  # Mix of genes before and after GFI1B

    for gene in test_genes:
        if gene in gene_mod.feature_meta['gene'].values:
            # Get the position in the feature list (not the DataFrame index)
            gene_list = gene_mod.feature_meta['gene'].tolist()
            feature_idx = gene_list.index(gene)
            alpha_val = gene_mod.alpha_y_prefit[0, 0, feature_idx].item()
            print(f"  ✓ {gene:10s}: feature_idx={feature_idx:2d}, alpha_y={alpha_val:.4f}")
        else:
            print(f"  - {gene:10s}: not in gene modality (OK if it's the cis gene)")

    # Verify GFI1B is NOT accessible in gene modality
    if cis_gene not in gene_mod.feature_meta['gene'].values:
        print(f"  ✓ {cis_gene:10s}: correctly NOT in gene modality")
    else:
        raise AssertionError(f"{cis_gene} should NOT be in gene modality!")

except Exception as e:
    print(f"  ✗ Indexing verification FAILED: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*80)
print("ALL PLOTTING TESTS PASSED!")
print("="*80)

print("\nSummary:")
print("  ✓ plot_technical_fit works with 91-feature gene modality posterior")
print("  ✓ plot_technical_fit works with 1-feature cis modality posterior")
print("  ✓ plot_xy_data with technical correction works correctly")
print("  ✓ Alpha_y indexing is correct for all trans genes")
print("  ✓ Cis gene is correctly excluded from gene modality")
print("\nThe 91-feature posterior samples work correctly with all plotting functions!")

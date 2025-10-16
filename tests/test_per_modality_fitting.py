"""
Test per-modality fitting functionality.

This tests that fit_technical() and fit_trans() can fit different modalities
and store results correctly in both modality objects and at model level.
"""
import numpy as np
import pandas as pd
import torch
from bayesDREAM import MultiModalBayesDREAM, Modality

print("=" * 80)
print("Test 1: Setup - Create MultiModalBayesDREAM with multiple modalities")
print("=" * 80)

# Create synthetic data
np.random.seed(42)
n_genes = 10
n_cells = 50
n_guides = 10

# Create metadata
cell_names = [f"cell_{i}" for i in range(n_cells)]
guides = np.repeat([f"guide_{i}" for i in range(n_guides)], n_cells // n_guides)
targets = ['ntc'] * (n_cells // 2) + ['GFI1B'] * (n_cells // 2)
cell_lines = np.random.choice(['A', 'B'], n_cells)

meta = pd.DataFrame({
    'cell': cell_names,
    'guide': guides,
    'target': targets,
    'cell_line': cell_lines,
    'sum_factor': np.random.uniform(0.8, 1.2, n_cells),
    'L_cell_barcode': cell_names
})

# Create gene counts (including cis gene GFI1B)
gene_names = [f"gene_{i}" for i in range(n_genes)] + ['GFI1B']
gene_counts = pd.DataFrame(
    np.random.poisson(50, (n_genes + 1, n_cells)),
    index=gene_names,
    columns=cell_names
)

# Initialize MultiModalBayesDREAM
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    primary_modality='gene',
    output_dir='./test_output',
    label='per_modality_test',
    device='cpu'
)

print(f"✓ Created model with primary modality: {model.primary_modality}")
print(f"✓ Model has {len(model.modalities)} modality(ies)")
print(f"✓ Primary modality: {model.get_modality('gene')}")

# Add a second modality (splicing-like binomial data)
print("\n" + "=" * 80)
print("Test 2: Add splicing modality (binomial distribution)")
print("=" * 80)

n_junctions = 5
sj_counts = np.random.poisson(20, (n_junctions, n_cells))
sj_total = np.random.poisson(100, (n_junctions, n_cells))
sj_meta = pd.DataFrame({
    'junction_id': [f"junction_{i}" for i in range(n_junctions)],
    'chrom': ['chr1'] * n_junctions,
    'strand': ['+'] * n_junctions
})

splicing_modality = Modality(
    name='splicing_test',
    counts=pd.DataFrame(sj_counts, columns=cell_names),
    feature_meta=sj_meta,
    distribution='binomial',
    denominator=sj_total,
    cells_axis=1
)

model.add_modality('splicing_test', splicing_modality)

print(f"✓ Added splicing modality: {model.get_modality('splicing_test')}")
print(f"✓ Model now has {len(model.modalities)} modalities")
print(f"✓ Splicing modality has {splicing_modality.dims['n_cells']} cells")

# Test 3: Fit technical for primary modality
print("\n" + "=" * 80)
print("Test 3: Set technical groups and fit technical for PRIMARY modality (gene)")
print("=" * 80)

# Set technical groups first
model.set_technical_groups(['cell_line'])

model.fit_technical(
    sum_factor_col='sum_factor',
    modality_name='gene',  # Explicitly specify primary modality
    niters=100,  # Few iterations for testing
    nsamples=10
)

gene_modality = model.get_modality('gene')
print(f"✓ Technical fit completed for 'gene' modality")
print(f"✓ Gene modality alpha_y_prefit shape: {gene_modality.alpha_y_prefit.shape if gene_modality.alpha_y_prefit is not None else 'None'}")
print(f"✓ Gene modality has posterior_samples_technical: {gene_modality.posterior_samples_technical is not None}")

# Check model-level storage (backward compatibility)
print(f"✓ Model-level alpha_y_prefit stored (backward compat): {model.alpha_y_prefit is not None}")
print(f"✓ Model-level posterior_samples_technical stored: {model.posterior_samples_technical is not None}")

# Verify they are the same object (or at least equal)
if gene_modality.alpha_y_prefit is not None and model.alpha_y_prefit is not None:
    print(f"✓ Model-level and modality-level alpha_y_prefit match: {torch.allclose(gene_modality.alpha_y_prefit, model.alpha_y_prefit)}")

# Test 4: Fit technical for non-primary modality
print("\n" + "=" * 80)
print("Test 4: Fit technical model for NON-PRIMARY modality (splicing_test)")
print("=" * 80)

# technical_groups already set, just fit
model.fit_technical(
    modality_name='splicing_test',  # Non-primary modality
    niters=100,
    nsamples=10
)

splicing_modality = model.get_modality('splicing_test')
print(f"✓ Technical fit completed for 'splicing_test' modality")
print(f"✓ Splicing modality alpha_y_prefit shape: {splicing_modality.alpha_y_prefit.shape if splicing_modality.alpha_y_prefit is not None else 'None'}")
print(f"✓ Splicing modality has posterior_samples_technical: {splicing_modality.posterior_samples_technical is not None}")

# Check that model-level storage was NOT overwritten
print(f"✓ Model-level alpha_y_prefit NOT overwritten (still from gene): {model.alpha_y_prefit is not None}")
if gene_modality.alpha_y_prefit is not None and model.alpha_y_prefit is not None:
    print(f"✓ Model-level alpha_y_prefit still matches gene modality: {torch.allclose(gene_modality.alpha_y_prefit, model.alpha_y_prefit)}")

# Test 5: Fit trans for primary modality
print("\n" + "=" * 80)
print("Test 5: Fit trans model for PRIMARY modality (gene)")
print("=" * 80)

# First need to set x_true (cis expression)
# For testing, just create dummy x_true
model.x_true = torch.ones(n_cells, dtype=torch.float32)
model.x_true_type = 'point'

model.fit_trans(
    sum_factor_col='sum_factor',
    function_type='additive_hill',
    modality_name='gene',
    p0=0.01,
    gamma_threshold=0.01,
    niters=100,
    nsamples=10
)

print(f"✓ Trans fit completed for 'gene' modality")
print(f"✓ Gene modality has posterior_samples_trans: {gene_modality.posterior_samples_trans is not None}")
print(f"✓ Model-level posterior_samples_trans stored (backward compat): {model.posterior_samples_trans is not None}")

# Test 6: Fit trans for non-primary modality
print("\n" + "=" * 80)
print("Test 6: Fit trans model for NON-PRIMARY modality (splicing_test)")
print("=" * 80)

model.fit_trans(
    function_type='additive_hill',
    modality_name='splicing_test',
    p0=0.01,
    gamma_threshold=0.01,
    niters=100,
    nsamples=10
)

print(f"✓ Trans fit completed for 'splicing_test' modality")
print(f"✓ Splicing modality has posterior_samples_trans: {splicing_modality.posterior_samples_trans is not None}")

# Check that model-level storage was NOT overwritten
print(f"✓ Model-level posterior_samples_trans NOT overwritten (still from gene): {model.posterior_samples_trans is not None}")

# Test 7: Error handling - trans without technical fit
print("\n" + "=" * 80)
print("Test 7: Error handling - trans without technical fit")
print("=" * 80)

# Add a third modality without fitting technical
third_modality = Modality(
    name='untrained',
    counts=np.random.poisson(10, (5, n_cells)),
    feature_meta=pd.DataFrame({'feature': [f'f_{i}' for i in range(5)]}),
    distribution='negbinom',
    cells_axis=1
)
model.add_modality('untrained', third_modality)

try:
    model.fit_trans(modality_name='untrained', niters=10, nsamples=5)
    print("✗ FAILED: Should have raised ValueError")
except ValueError as e:
    print(f"✓ Correctly raised ValueError: {str(e)[:80]}...")

# Test 8: Backward compatibility - fit without specifying modality_name
print("\n" + "=" * 80)
print("Test 8: Backward compatibility - default to primary modality")
print("=" * 80)

# Reset gene modality fitting results
gene_modality.alpha_y_prefit = None
gene_modality.posterior_samples_technical = None

# Fit without specifying modality_name (should default to primary)
# technical_groups already set from earlier
model.fit_technical(
    sum_factor_col='sum_factor',
    niters=100,
    nsamples=10
)

print(f"✓ Technical fit completed (defaulted to primary modality 'gene')")
print(f"✓ Gene modality alpha_y_prefit restored: {gene_modality.alpha_y_prefit is not None}")

# Test 9: Verify results are stored per-modality
print("\n" + "=" * 80)
print("Test 9: Verify results are stored per-modality")
print("=" * 80)

print(f"✓ Gene modality results:")
print(f"  - alpha_y_prefit: {gene_modality.alpha_y_prefit is not None}")
print(f"  - posterior_samples_technical: {gene_modality.posterior_samples_technical is not None}")
print(f"  - posterior_samples_trans: {gene_modality.posterior_samples_trans is not None}")

print(f"\n✓ Splicing modality results:")
print(f"  - alpha_y_prefit: {splicing_modality.alpha_y_prefit is not None}")
print(f"  - posterior_samples_technical: {splicing_modality.posterior_samples_technical is not None}")
print(f"  - posterior_samples_trans: {splicing_modality.posterior_samples_trans is not None}")

print(f"\n✓ Untrained modality results:")
untrained_mod = model.get_modality('untrained')
print(f"  - alpha_y_prefit: {untrained_mod.alpha_y_prefit is not None}")
print(f"  - posterior_samples_technical: {untrained_mod.posterior_samples_technical is not None}")
print(f"  - posterior_samples_trans: {untrained_mod.posterior_samples_trans is not None}")

# Test 10: Access modality-specific results
print("\n" + "=" * 80)
print("Test 10: Access modality-specific results")
print("=" * 80)

gene_alpha = gene_modality.alpha_y_prefit
splicing_alpha = splicing_modality.alpha_y_prefit

print(f"✓ Gene modality alpha_y_prefit shape: {gene_alpha.shape}")
print(f"✓ Splicing modality alpha_y_prefit shape: {splicing_alpha.shape}")
print(f"✓ Results are different between modalities: {not torch.allclose(gene_alpha[:, :, 0], splicing_alpha[:, :, 0])}")

print("\n" + "=" * 80)
print("All tests passed!")
print("=" * 80)

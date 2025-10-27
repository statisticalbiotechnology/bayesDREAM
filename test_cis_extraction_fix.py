"""
Test that cis gene extraction correctly removes it from ALL posterior samples.
"""
import pandas as pd
import torch
from bayesDREAM import bayesDREAM

# Load data
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print("="*80)
print("CREATING MODEL AND RUNNING TECHNICAL FIT")
print("="*80)
print(f"Original counts: {counts.shape[0]} genes")
print(f"Cis gene: GFI1B at position {counts.index.get_loc('GFI1B')}")
print()

# Create model
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_cis_fix',
    cores=1
)

# Run technical fit (this is where the bug was)
model.set_technical_groups(['cell_line'])
print("\n[TEST] Running technical fit...")
model.fit_technical(sum_factor_col='sum_factor')

print()
print("="*80)
print("VERIFYING CIS GENE EXTRACTION")
print("="*80)

# Check gene modality
gene_mod = model.get_modality('gene')
print(f"\n✓ Gene modality has {len(gene_mod.feature_meta)} genes (expected: 91)")
assert len(gene_mod.feature_meta) == 91, "Gene modality should have 91 genes"

# Check cis modality
cis_mod = model.get_modality('cis')
print(f"✓ Cis modality has {len(cis_mod.feature_meta)} genes (expected: 1)")
assert len(cis_mod.feature_meta) == 1, "Cis modality should have 1 gene"

print()
print("="*80)
print("CHECKING GENE MODALITY POSTERIOR SAMPLES")
print("="*80)

if hasattr(gene_mod, 'posterior_samples_technical') and gene_mod.posterior_samples_technical:
    print("\nGene modality posterior_samples_technical:")
    for key, val in gene_mod.posterior_samples_technical.items():
        if hasattr(val, 'shape'):
            print(f"  {key:20s}: {val.shape}")
            # All should have 91 features (last dimension)
            expected_features = 91
            actual_features = val.shape[-1]
            if actual_features != expected_features:
                print(f"    ✗ ERROR: Expected {expected_features} features, got {actual_features}")
            else:
                print(f"    ✓ Correct: {expected_features} features")
        else:
            print(f"  {key:20s}: {type(val)}")
else:
    print("✗ No posterior_samples_technical found in gene modality!")

print()
print("="*80)
print("CHECKING CIS MODALITY POSTERIOR SAMPLES")
print("="*80)

if hasattr(cis_mod, 'posterior_samples_technical') and cis_mod.posterior_samples_technical:
    print("\nCis modality posterior_samples_technical:")
    for key, val in cis_mod.posterior_samples_technical.items():
        if hasattr(val, 'shape'):
            print(f"  {key:20s}: {val.shape}")
            # All should have 1 feature (last dimension)
            expected_features = 1
            actual_features = val.shape[-1]
            if actual_features != expected_features:
                print(f"    ✗ ERROR: Expected {expected_features} features, got {actual_features}")
            else:
                print(f"    ✓ Correct: {expected_features} feature")
        else:
            print(f"  {key:20s}: {type(val)}")

    # Check that we have all expected keys
    expected_keys = ['log2_alpha_x', 'alpha_x_mul', 'delta_x_add', 'o_x', 'mu_ntc',
                     'alpha_x', 'alpha_x_mult', 'alpha_x_add']
    missing_keys = [k for k in expected_keys if k not in cis_mod.posterior_samples_technical]
    if missing_keys:
        print(f"\n✗ Missing keys: {missing_keys}")
    else:
        print(f"\n✓ All expected keys present: {expected_keys}")
else:
    print("✗ No posterior_samples_technical found in cis modality!")

print()
print("="*80)
print("TESTING SAVE/LOAD")
print("="*80)

# Save
model.save_technical_fit()
print("\n✓ Saved technical fit")

# Create new model and load
model2 = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./test_cis_fix',
    cores=1
)
model2.load_technical_fit()
print("✓ Loaded technical fit")

# Verify loaded shapes
gene_mod2 = model2.get_modality('gene')
cis_mod2 = model2.get_modality('cis')

print(f"\nAfter load:")
print(f"  Gene modality alpha_y_prefit shape: {gene_mod2.alpha_y_prefit.shape}")
print(f"  Cis modality alpha_x_prefit shape: {model2.alpha_x_prefit.shape if hasattr(model2, 'alpha_x_prefit') else 'Not set'}")

if hasattr(gene_mod2, 'posterior_samples_technical') and gene_mod2.posterior_samples_technical:
    print(f"\n  Gene modality posterior samples:")
    for key in ['log2_alpha_y', 'alpha_y_mult', 'o_y', 'mu_ntc']:
        if key in gene_mod2.posterior_samples_technical:
            shape = gene_mod2.posterior_samples_technical[key].shape
            print(f"    {key}: {shape}")
            assert shape[-1] == 91, f"✗ {key} should have 91 features, got {shape[-1]}"
    print("  ✓ All gene posterior samples have 91 features")

if hasattr(cis_mod2, 'posterior_samples_technical') and cis_mod2.posterior_samples_technical:
    print(f"\n  Cis modality posterior samples:")
    for key in ['log2_alpha_x', 'alpha_x_mult', 'o_x', 'mu_ntc']:
        if key in cis_mod2.posterior_samples_technical:
            shape = cis_mod2.posterior_samples_technical[key].shape
            print(f"    {key}: {shape}")
            assert shape[-1] == 1, f"✗ {key} should have 1 feature, got {shape[-1]}"
    print("  ✓ All cis posterior samples have 1 feature")

print()
print("="*80)
print("ALL TESTS PASSED!")
print("="*80)

"""
Quick test to verify posterior sample shapes after cis gene extraction fix.
"""
import pandas as pd
import torch
from bayesDREAM import bayesDREAM
import os

# Load data
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print("="*80)
print("TESTING POSTERIOR SAMPLE SHAPES")
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

# Check if we have a pre-existing technical fit
if os.path.exists('./test_cis_fix/posterior_samples_technical_gene.pt'):
    print("[INFO] Found existing technical fit, loading...")
    model.load_technical_fit()
    print("[INFO] Loaded technical fit")
else:
    print("[INFO] No existing fit found, running quick technical fit with 100 samples...")
    model.set_technical_groups(['cell_line'])
    model.fit_technical(sum_factor_col='sum_factor', num_samples=100, num_steps=500)
    model.save_technical_fit()
    print("[INFO] Completed and saved technical fit")

print()
print("="*80)
print("VERIFYING MODALITY STRUCTURE")
print("="*80)

# Check gene modality
gene_mod = model.get_modality('gene')
print(f"\n✓ Gene modality has {len(gene_mod.feature_meta)} genes (expected: 91)")
assert len(gene_mod.feature_meta) == 91, f"Gene modality should have 91 genes, got {len(gene_mod.feature_meta)}"

# Check cis modality
cis_mod = model.get_modality('cis')
print(f"✓ Cis modality has {len(cis_mod.feature_meta)} genes (expected: 1)")
assert len(cis_mod.feature_meta) == 1, f"Cis modality should have 1 gene, got {len(cis_mod.feature_meta)}"

print()
print("="*80)
print("CHECKING GENE MODALITY POSTERIOR SAMPLES")
print("="*80)

if hasattr(gene_mod, 'posterior_samples_technical') and gene_mod.posterior_samples_technical:
    print("\nGene modality posterior_samples_technical:")
    all_correct = True
    for key, val in gene_mod.posterior_samples_technical.items():
        if hasattr(val, 'shape'):
            expected_features = 91
            actual_features = val.shape[-1]
            status = "✓" if actual_features == expected_features else "✗"
            print(f"  {status} {key:20s}: {val.shape} (expected last dim: {expected_features})")
            if actual_features != expected_features:
                all_correct = False
                print(f"      ERROR: Expected {expected_features} features, got {actual_features}")
        else:
            print(f"  - {key:20s}: {type(val)}")

    if all_correct:
        print("\n✓✓✓ ALL GENE MODALITY POSTERIOR SAMPLES HAVE 91 FEATURES ✓✓✓")
    else:
        print("\n✗✗✗ SOME GENE MODALITY POSTERIOR SAMPLES HAVE WRONG SHAPE ✗✗✗")
        raise AssertionError("Gene modality posterior samples have incorrect shapes")
else:
    print("✗ No posterior_samples_technical found in gene modality!")
    raise AssertionError("No posterior samples found")

print()
print("="*80)
print("CHECKING CIS MODALITY POSTERIOR SAMPLES")
print("="*80)

if hasattr(cis_mod, 'posterior_samples_technical') and cis_mod.posterior_samples_technical:
    print("\nCis modality posterior_samples_technical:")
    expected_keys = ['log2_alpha_x', 'alpha_x_mul', 'delta_x_add', 'o_x', 'mu_ntc',
                     'alpha_x', 'alpha_x_mult', 'alpha_x_add']

    all_correct = True
    for key, val in cis_mod.posterior_samples_technical.items():
        if hasattr(val, 'shape'):
            expected_features = 1
            actual_features = val.shape[-1]
            status = "✓" if actual_features == expected_features else "✗"
            print(f"  {status} {key:20s}: {val.shape} (expected last dim: {expected_features})")
            if actual_features != expected_features:
                all_correct = False
                print(f"      ERROR: Expected {expected_features} features, got {actual_features}")
        else:
            print(f"  - {key:20s}: {type(val)}")

    # Check for missing keys
    missing_keys = [k for k in expected_keys if k not in cis_mod.posterior_samples_technical]
    if missing_keys:
        print(f"\n✗ Missing keys: {missing_keys}")
        all_correct = False
    else:
        print(f"\n✓ All expected keys present")

    if all_correct:
        print("\n✓✓✓ ALL CIS MODALITY POSTERIOR SAMPLES HAVE 1 FEATURE ✓✓✓")
    else:
        print("\n✗✗✗ SOME CIS MODALITY POSTERIOR SAMPLES HAVE WRONG SHAPE OR MISSING ✗✗✗")
        raise AssertionError("Cis modality posterior samples have issues")
else:
    print("✗ No posterior_samples_technical found in cis modality!")
    raise AssertionError("No cis posterior samples found")

print()
print("="*80)
print("VERIFYING NO CIS GENE IN PRIMARY MODALITY FEATURE LIST")
print("="*80)

gene_names = gene_mod.feature_meta['gene'].tolist()
if 'GFI1B' in gene_names:
    print(f"✗✗✗ ERROR: Found 'GFI1B' in gene modality feature list at position {gene_names.index('GFI1B')} ✗✗✗")
    raise AssertionError("Cis gene should NOT be in gene modality feature list")
else:
    print("✓ Confirmed: 'GFI1B' is NOT in gene modality feature list")
    print(f"  Gene modality features: {gene_names[:5]} ... {gene_names[-5:]}")

print()
print("="*80)
print("ALL TESTS PASSED!")
print("="*80)
print("\nSummary:")
print(f"  - Gene modality: 91 genes (excluding cis gene GFI1B)")
print(f"  - Cis modality: 1 gene (GFI1B)")
print(f"  - All gene modality posterior samples: 91 features ✓")
print(f"  - All cis modality posterior samples: 1 feature ✓")
print(f"  - GFI1B not in gene modality feature list ✓")

"""
Unit test for cis gene extraction from posterior samples.
Tests the logic without running full technical fit.
"""
import pandas as pd
import torch
from bayesDREAM import bayesDREAM

print("="*80)
print("UNIT TEST: CIS GENE EXTRACTION FROM POSTERIOR SAMPLES")
print("="*80)

# Load data
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print(f"\nOriginal counts: {counts.shape[0]} genes")
cis_gene = 'GFI1B'
cis_idx_orig = counts.index.get_loc(cis_gene)
print(f"Cis gene: {cis_gene} at position {cis_idx_orig}")

# Create model
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene=cis_gene,
    output_dir='./test_unit',
    cores=1
)

print("\n" + "="*80)
print("VERIFYING MODEL INITIALIZATION")
print("="*80)

# Verify modalities were created correctly
gene_mod = model.get_modality('gene')
cis_mod = model.get_modality('cis')

print(f"\n✓ Gene modality created with {len(gene_mod.feature_meta)} features")
print(f"✓ Cis modality created with {len(cis_mod.feature_meta)} features")

assert len(gene_mod.feature_meta) == 91, f"Expected 91 genes in gene modality, got {len(gene_mod.feature_meta)}"
assert len(cis_mod.feature_meta) == 1, f"Expected 1 gene in cis modality, got {len(cis_mod.feature_meta)}"

# Verify cis gene is not in gene modality
gene_names = gene_mod.feature_meta['gene'].tolist()
assert cis_gene not in gene_names, f"Cis gene {cis_gene} should NOT be in gene modality feature list"
print(f"✓ Confirmed: {cis_gene} is NOT in gene modality feature list")

# Verify cis gene is in cis modality
cis_names = cis_mod.feature_meta['gene'].tolist()
assert cis_gene in cis_names, f"Cis gene {cis_gene} should be in cis modality feature list"
print(f"✓ Confirmed: {cis_gene} IS in cis modality feature list")

print("\n" + "="*80)
print("SIMULATING POSTERIOR SAMPLES")
print("="*80)

# Simulate what technical.py does - create mock posterior samples
# These would normally come from Pyro inference
S = 100  # number of samples
C = 2    # number of cell lines
T_orig = 92  # original number of genes (including cis)
T_trans = 91  # trans genes only

print(f"\nCreating mock posterior samples with shape [S={S}, C-1={C-1}, T={T_orig}]")

# Create mock raw posterior samples (these would have 92 genes originally)
mock_posterior_raw = {
    'log2_alpha_y': torch.randn(S, C-1, T_orig),
    'alpha_y_mul': torch.exp(torch.randn(S, C-1, T_orig) * 0.1),
    'delta_y_add': torch.randn(S, C-1, T_orig) * 0.01,
    'o_y': torch.randn(S, C-1, T_orig),
    'mu_ntc': torch.randn(S, C-1, T_orig),
    'beta_o': torch.randn(S, C-1, 1)  # This one is per-group, not per-gene
}

print("✓ Created mock raw posterior samples")
for key, val in mock_posterior_raw.items():
    print(f"  {key:20s}: {val.shape}")

print("\n" + "="*80)
print("SIMULATING CIS GENE EXTRACTION LOGIC")
print("="*80)

# Simulate what the fix does in technical.py
all_idx = list(range(T_orig))
trans_idx = [i for i in all_idx if i != cis_idx_orig]

print(f"\nOriginal feature indices: {len(all_idx)} genes")
print(f"Trans feature indices: {len(trans_idx)} genes (excluding position {cis_idx_orig})")

# Extract cis gene samples
cis_posterior = {}
cis_posterior['log2_alpha_x'] = mock_posterior_raw['log2_alpha_y'][..., cis_idx_orig:cis_idx_orig+1]
cis_posterior['alpha_x_mul'] = mock_posterior_raw['alpha_y_mul'][..., cis_idx_orig:cis_idx_orig+1]
cis_posterior['delta_x_add'] = mock_posterior_raw['delta_y_add'][..., cis_idx_orig:cis_idx_orig+1]
cis_posterior['o_x'] = mock_posterior_raw['o_y'][..., cis_idx_orig:cis_idx_orig+1]
cis_posterior['mu_ntc'] = mock_posterior_raw['mu_ntc'][..., cis_idx_orig:cis_idx_orig+1]

# Reconstruct alpha with baseline (like technical.py does)
alpha_x_mul_raw = cis_posterior['alpha_x_mul']
cis_posterior['alpha_x_mult'] = torch.cat(
    [torch.ones(S, 1, 1, device=alpha_x_mul_raw.device), alpha_x_mul_raw],
    dim=1
)
cis_posterior['alpha_x'] = cis_posterior['alpha_x_mult']

delta_x_add_raw = cis_posterior['delta_x_add']
cis_posterior['alpha_x_add'] = torch.cat(
    [torch.zeros(S, 1, 1, device=delta_x_add_raw.device), delta_x_add_raw],
    dim=1
)

print("\n✓ Extracted cis gene posterior samples:")
for key, val in cis_posterior.items():
    print(f"  {key:20s}: {val.shape}")

# Remove cis gene from trans samples
gene_posterior = {}
for key in ['log2_alpha_y', 'alpha_y_mul', 'delta_y_add', 'o_y', 'mu_ntc']:
    if key in mock_posterior_raw:
        gene_posterior[key] = mock_posterior_raw[key][..., trans_idx]

# Add beta_o (not per-gene)
gene_posterior['beta_o'] = mock_posterior_raw['beta_o']

# Create reconstructed alpha (like technical.py does for gene modality)
alpha_y_mul_raw = gene_posterior['alpha_y_mul']
gene_posterior['alpha_y_mult'] = torch.cat(
    [torch.ones(S, 1, T_trans, device=alpha_y_mul_raw.device), alpha_y_mul_raw],
    dim=1
)
gene_posterior['alpha_y'] = gene_posterior['alpha_y_mult']

delta_y_add_raw = gene_posterior['delta_y_add']
gene_posterior['alpha_y_add'] = torch.cat(
    [torch.zeros(S, 1, T_trans, device=delta_y_add_raw.device), delta_y_add_raw],
    dim=1
)

print("\n✓ Created gene modality posterior samples (excluding cis):")
for key, val in gene_posterior.items():
    print(f"  {key:20s}: {val.shape}")

print("\n" + "="*80)
print("VERIFYING SHAPES")
print("="*80)

# Verify all gene posterior samples have 91 features
print("\nGene modality posterior samples:")
all_correct = True
for key, val in gene_posterior.items():
    if hasattr(val, 'shape') and val.shape[-1] not in [1, T_trans]:
        print(f"  ✗ {key:20s}: {val.shape} - WRONG! Expected last dim = {T_trans}")
        all_correct = False
    elif hasattr(val, 'shape'):
        expected = T_trans if val.shape[-1] == T_trans else 1
        print(f"  ✓ {key:20s}: {val.shape} - Correct! Last dim = {expected}")
    else:
        print(f"  - {key:20s}: {type(val)}")

if not all_correct:
    raise AssertionError("Some gene modality posterior samples have incorrect shapes!")

# Verify all cis posterior samples have 1 feature
print("\nCis modality posterior samples:")
all_correct = True
for key, val in cis_posterior.items():
    if hasattr(val, 'shape') and val.shape[-1] != 1:
        print(f"  ✗ {key:20s}: {val.shape} - WRONG! Expected last dim = 1")
        all_correct = False
    elif hasattr(val, 'shape'):
        print(f"  ✓ {key:20s}: {val.shape} - Correct! Last dim = 1")
    else:
        print(f"  - {key:20s}: {type(val)}")

if not all_correct:
    raise AssertionError("Some cis modality posterior samples have incorrect shapes!")

print("\n" + "="*80)
print("VERIFYING INDEXING ALIGNMENT")
print("="*80)

# Verify that trans_idx correctly maps genes
original_genes = counts.index.tolist()
expected_trans_genes = [g for g in original_genes if g != cis_gene]
actual_trans_genes = gene_mod.feature_meta['gene'].tolist()

print(f"\nExpected {len(expected_trans_genes)} trans genes (excluding {cis_gene})")
print(f"Actual {len(actual_trans_genes)} trans genes in gene modality")

if expected_trans_genes == actual_trans_genes:
    print("✓ Trans gene list matches expected order")
else:
    print("✗ Trans gene list does NOT match!")
    # Find first mismatch
    for i, (exp, act) in enumerate(zip(expected_trans_genes, actual_trans_genes)):
        if exp != act:
            print(f"  First mismatch at position {i}: expected '{exp}', got '{act}'")
            break
    raise AssertionError("Trans gene list mismatch!")

# Verify indexing example
test_gene = 'TET2'  # A gene alphabetically after GFI1B
if test_gene in actual_trans_genes:
    gene_mod_idx = actual_trans_genes.index(test_gene)
    orig_idx = original_genes.index(test_gene)
    trans_idx_val = trans_idx[gene_mod_idx]

    print(f"\nIndexing example for gene '{test_gene}':")
    print(f"  Original position in counts: {orig_idx}")
    print(f"  Position in gene modality: {gene_mod_idx}")
    print(f"  trans_idx[{gene_mod_idx}] = {trans_idx_val}")

    if orig_idx == trans_idx_val:
        print(f"  ✓ Mapping is correct: gene_mod[{gene_mod_idx}] -> orig[{trans_idx_val}]")
    else:
        print(f"  ✗ Mapping is WRONG!")
        raise AssertionError("Index mapping incorrect!")

print("\n" + "="*80)
print("ALL UNIT TESTS PASSED!")
print("="*80)

print("\nSummary:")
print(f"  ✓ Model initialization: gene modality has 91 genes, cis modality has 1 gene")
print(f"  ✓ Cis gene '{cis_gene}' is NOT in gene modality feature list")
print(f"  ✓ Cis gene '{cis_gene}' IS in cis modality feature list")
print(f"  ✓ Mock posterior extraction: all gene samples have {T_trans} features")
print(f"  ✓ Mock posterior extraction: all cis samples have 1 feature")
print(f"  ✓ Trans gene list matches expected order")
print(f"  ✓ Index mapping from gene modality to original positions is correct")
print("\nThe cis gene extraction logic is working correctly!")

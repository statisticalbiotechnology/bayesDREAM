"""
Verify that the extraction mapping matches the gene modality order.
"""
import pandas as pd
import torch
from bayesDREAM import bayesDREAM

# Load counts
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)
all_genes_orig = counts.index.tolist()
cis_gene = 'GFI1B'
cis_idx_orig = all_genes_orig.index(cis_gene)

print(f'GFI1B is at original position {cis_idx_orig}')
print()

# Simulate what technical.py does
full_shape = 92  # from log2_alpha_y.shape[-1]
all_idx = list(range(full_shape))
trans_idx = [i for i in all_idx if i != cis_idx_orig]

print(f'full_shape: {full_shape}')
print(f'cis_idx_orig: {cis_idx_orig}')
print(f'len(trans_idx): {len(trans_idx)}')
print()

# Show what mapping this creates
print('Extraction mapping (first few):')
for i in range(5):
    orig_idx = trans_idx[i]
    print(f'  alpha_y_prefit[:, :, {i}] = full_alpha_y[:, :, {orig_idx}] ({all_genes_orig[orig_idx]})')

print()
print('Around GFI1B:')
for i in range(max(0, cis_idx_orig - 2), min(cis_idx_orig + 3, len(trans_idx))):
    orig_idx = trans_idx[i]
    print(f'  alpha_y_prefit[:, :, {i}] = full_alpha_y[:, :, {orig_idx}] ({all_genes_orig[orig_idx]})')

print()
print('Last few:')
for i in range(len(trans_idx) - 5, len(trans_idx)):
    orig_idx = trans_idx[i]
    print(f'  alpha_y_prefit[:, :, {i}] = full_alpha_y[:, :, {orig_idx}] ({all_genes_orig[orig_idx]})')

print()
print('='*80)
print('NOW CHECK: Does gene modality order match this?')
print('='*80)

# Load actual gene modality order
meta = pd.read_csv('toydata/cell_meta.csv')

model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./testing/output',
    cores=1
)

gene_mod = model.get_modality('gene')
gene_names = gene_mod.feature_meta.index.tolist()

print()
print('Gene modality order:')
for i in range(5):
    orig_idx = all_genes_orig.index(gene_names[i])
    print(f'  gene_mod index {i}: {gene_names[i]} (original position {orig_idx})')

print()
print('Around GFI1B:')
for i in range(max(0, cis_idx_orig - 2), min(cis_idx_orig + 3, len(gene_names))):
    orig_idx = all_genes_orig.index(gene_names[i])
    print(f'  gene_mod index {i}: {gene_names[i]} (original position {orig_idx})')

print()
print('Last few:')
for i in range(len(gene_names) - 5, len(gene_names)):
    orig_idx = all_genes_orig.index(gene_names[i])
    print(f'  gene_mod index {i}: {gene_names[i]} (original position {orig_idx})')

print()
print('='*80)
print('VERIFICATION')
print('='*80)
mismatch = False
for i in range(len(gene_names)):
    expected_orig_idx = trans_idx[i]
    actual_orig_idx = all_genes_orig.index(gene_names[i])
    if expected_orig_idx != actual_orig_idx:
        print(f'✗ MISMATCH at i={i}: expected orig {expected_orig_idx}, got {actual_orig_idx}')
        print(f'   Expected gene: {all_genes_orig[expected_orig_idx]}')
        print(f'   Actual gene:   {gene_names[i]}')
        mismatch = True

if not mismatch:
    print('✓ All genes map correctly!')
    print()
    print('This means alpha_y_prefit indexing is theoretically correct.')
    print('The user-reported bug must be elsewhere!')

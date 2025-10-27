"""
Debug script to verify alpha_y indexing for gene modality.
"""
import pandas as pd
import numpy as np
import torch
from bayesDREAM import bayesDREAM

# Load data
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print("="*80)
print("ORIGINAL COUNTS")
print("="*80)
print(f"Shape: {counts.shape}")
print(f"Genes: {counts.index.tolist()}")
print()

# Create model
model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./testing/output',
    cores=1
)

# Load technical fit
try:
    model.load_technical_fit()
    has_technical = True
    print("="*80)
    print("TECHNICAL FIT LOADED")
    print("="*80)
except Exception as e:
    print(f"Could not load technical fit: {e}")
    has_technical = False

if has_technical:
    gene_mod = model.get_modality('gene')

    # Get feature names from feature_meta
    gene_names = gene_mod.feature_meta.index.tolist()
    print(f"\nGene modality features: {len(gene_names)}")
    print(f"Gene modality features: {gene_names}")
    print()

    print(f"alpha_y_prefit shape: {gene_mod.alpha_y_prefit.shape}")
    print()

    # Check original counts metadata (if it was stored during technical fit)
    if hasattr(model, 'counts_meta'):
        print("="*80)
        print("ORIGINAL COUNTS METADATA (from technical fit)")
        print("="*80)
        print(f"counts_meta index: {model.counts_meta.index.tolist()}")
        print()

    # Test a few genes
    test_genes = ['AAAA', 'GFI1B', 'TET2', 'ZZZZ']

    print("="*80)
    print("INDEXING TEST")
    print("="*80)

    for gene in test_genes:
        # Check in original counts
        if gene in counts.index:
            orig_idx = counts.index.get_loc(gene)
            print(f"{gene}:")
            print(f"  - Original counts index: {orig_idx}")
        else:
            print(f"{gene}: NOT IN ORIGINAL COUNTS")
            continue

        # Check in gene modality
        if gene in gene_names:
            gene_mod_idx = gene_names.index(gene)
            print(f"  - Gene modality index: {gene_mod_idx}")

            # Get alpha_y
            alpha_y_mean = gene_mod.alpha_y_prefit[:, :, gene_mod_idx].mean().item()
            print(f"  - alpha_y_prefit mean: {alpha_y_mean:.4f}")
        elif gene == 'GFI1B':
            print(f"  - Not in gene modality (is cis gene)")
        else:
            print(f"  - NOT IN GENE MODALITY (unexpected!)")
        print()

    # Check if there's an off-by-one error for genes after GFI1B
    print("="*80)
    print("OFF-BY-ONE CHECK")
    print("="*80)

    # Find GFI1B position in original counts
    gfi1b_orig_idx = counts.index.get_loc('GFI1B')
    print(f"GFI1B original index: {gfi1b_orig_idx}")
    print()

    # Check genes around GFI1B
    for i in range(max(0, gfi1b_orig_idx - 2), min(len(counts.index), gfi1b_orig_idx + 3)):
        gene = counts.index[i]
        print(f"Original index {i}: {gene}")

        if gene == 'GFI1B':
            print(f"  -> CIS GENE (excluded from gene modality)")
        else:
            if gene in gene_names:
                gene_mod_idx = gene_names.index(gene)
                print(f"  -> Gene modality index: {gene_mod_idx}")

                # What should the expected index be?
                expected_idx = i if i < gfi1b_orig_idx else i - 1
                print(f"  -> Expected index: {expected_idx}")
                print(f"  -> Match: {gene_mod_idx == expected_idx}")
            else:
                print(f"  -> NOT IN GENE MODALITY")
        print()

    # Check if counts_meta was stored and how it maps
    if hasattr(model, 'counts_meta'):
        print("="*80)
        print("ALPHA EXTRACTION CHECK")
        print("="*80)

        # The technical fit was done on all 92 genes
        # alpha_y_prefit should exclude the cis gene

        print(f"counts_meta has {len(model.counts_meta)} genes (from technical fit)")
        print(f"gene modality has {len(gene_names)} genes (trans only)")
        print(f"alpha_y_prefit has {gene_mod.alpha_y_prefit.shape[-1]} features")
        print()

        # Check if the extraction was done correctly
        # For each gene in gene modality, verify its alpha comes from the right original index
        print("Checking a few genes:")
        for gene in ['AAAA', 'TET2', 'MYB']:
            if gene not in gene_names:
                continue

            orig_idx = counts.index.get_loc(gene)
            gene_mod_idx = gene_names.index(gene)

            print(f"\n{gene}:")
            print(f"  Original index in counts: {orig_idx}")
            print(f"  Index in gene modality: {gene_mod_idx}")
            print(f"  Index in alpha_y_prefit: {gene_mod_idx} (used for lookup)")

            # The alpha should come from original index orig_idx in the technical fit
            # But alpha_y_prefit[gene_mod_idx] corresponds to...?
            if gene_mod_idx < gfi1b_orig_idx:
                expected_orig = gene_mod_idx
            else:
                expected_orig = gene_mod_idx + 1

            print(f"  Expected to correspond to original index: {expected_orig}")
            print(f"  Does it match? {expected_orig == orig_idx}")

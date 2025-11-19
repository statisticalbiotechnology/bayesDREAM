"""
Test if alpha_y values are correctly assigned to genes.
"""
import pandas as pd
import torch
from bayesDREAM import bayesDREAM

# Load data
meta = pd.read_csv('toydata/cell_meta.csv')
counts = pd.read_csv('toydata/gene_counts.csv', index_col=0)

print("="*80)
print("LOADING MODEL AND TECHNICAL FIT")
print("="*80)

model = bayesDREAM(
    meta=meta,
    counts=counts,
    cis_gene='GFI1B',
    output_dir='./testing/output',
    cores=1
)

model.load_technical_fit()

gene_mod = model.get_modality('gene')
gene_names = gene_mod.feature_meta.index.tolist()

# Find GFI1B position in original counts
gfi1b_orig_idx = counts.index.get_loc('GFI1B')
print(f"\nGFI1B is at original position {gfi1b_orig_idx}")
print()

# Check if we have counts_meta (stored during technical fit)
if hasattr(model, 'counts_meta'):
    print("="*80)
    print("VERIFYING ALPHA EXTRACTION")
    print("="*80)
    print("Original counts_meta index (from technical fit):")
    print(f"  {model.counts_meta.index.tolist()}")
    print()

    # Load the full posterior from technical fit
    if hasattr(gene_mod, 'posterior_samples_technical') and gene_mod.posterior_samples_technical is not None:
        alpha_y_mult = gene_mod.posterior_samples_technical.get('alpha_y_mult')
        if alpha_y_mult is not None:
            print(f"posterior_samples_technical['alpha_y_mult'] shape: {alpha_y_mult.shape}")
            print(f"gene_mod.alpha_y_prefit shape: {gene_mod.alpha_y_prefit.shape}")
            print()

            # The posterior should have alpha for all 92 genes
            # But gene_mod.alpha_y_prefit should have only 91 (excluding GFI1B)

            print("="*80)
            print("ALPHA VALUE COMPARISON")
            print("="*80)
            print("Comparing alpha values for genes before and after GFI1B\n")

            # Sample a few genes before and after GFI1B
            test_genes_before = [counts.index[i] for i in [20, 30, 40] if i < gfi1b_orig_idx]
            test_genes_after = [counts.index[i] for i in [45, 50, 60, 70] if i < len(counts.index)]

            for gene in test_genes_before + test_genes_after:
                orig_idx = counts.index.get_loc(gene)
                gene_mod_idx = gene_names.index(gene)

                # Get alpha from gene_mod.alpha_y_prefit (what plotting uses)
                alpha_from_prefit = gene_mod.alpha_y_prefit[0, 1, gene_mod_idx].item()  # Sample 0, group 1

                # Get alpha from full posterior (what should be the source)
                if alpha_y_mult.shape[-1] == 92:
                    # Posterior has all 92 genes
                    alpha_from_full = alpha_y_mult[0, 1, orig_idx].item()
                else:
                    alpha_from_full = "N/A (unexpected shape)"

                print(f"{gene:12s} (orig_idx={orig_idx:2d}, gene_mod_idx={gene_mod_idx:2d}):")
                print(f"  alpha from prefit:     {alpha_from_prefit:.6f}")
                print(f"  alpha from full (exp): {alpha_from_full if isinstance(alpha_from_full, str) else f'{alpha_from_full:.6f}'}")
                print(f"  Match: {abs(alpha_from_prefit - alpha_from_full) < 1e-6 if not isinstance(alpha_from_full, str) else False}")
                print()

print("="*80)
print("CRITICAL TEST: Check if extraction preserves ordering")
print("="*80)

# The key question: when we extract trans_idx = [0, ..., 42, 44, ..., 91],
# does modality.alpha_y_prefit = full_alpha_y_mult[..., trans_idx] preserve the right mapping?

# Let's manually reconstruct what should happen:
print(f"\nOriginal counts has {len(counts.index)} genes")
print(f"GFI1B at position {gfi1b_orig_idx}")
print(f"Gene modality has {len(gene_names)} genes")
print()

print("Expected mapping:")
print("  gene_mod.alpha_y_prefit[i] should correspond to:")
for i in range(min(5, len(gene_names))):
    gene = gene_names[i]
    orig_idx = counts.index.get_loc(gene)
    print(f"    i={i:2d}: {gene:12s} (original position {orig_idx})")

print("  ...")

for i in range(max(0, gfi1b_orig_idx - 2), min(gfi1b_orig_idx + 3, len(gene_names))):
    gene = gene_names[i]
    orig_idx = counts.index.get_loc(gene)
    print(f"    i={i:2d}: {gene:12s} (original position {orig_idx})")

print("  ...")

for i in range(max(0, len(gene_names) - 5), len(gene_names)):
    gene = gene_names[i]
    orig_idx = counts.index.get_loc(gene)
    print(f"    i={i:2d}: {gene:12s} (original position {orig_idx})")

print()
print("The extraction code does:")
print(f"  trans_idx = [i for i in range(92) if i != {gfi1b_orig_idx}]")
print(f"  modality.alpha_y_prefit = full_alpha_y_mult[..., trans_idx]")
print()
print("This means:")
for i in [0, gfi1b_orig_idx-1, gfi1b_orig_idx, len(gene_names)-1]:
    if i < len(gene_names):
        gene = gene_names[i]
        orig_idx = counts.index.get_loc(gene)
        trans_idx_val = orig_idx
        print(f"  alpha_y_prefit[:, :, {i}] = full_alpha_y_mult[:, :, {trans_idx_val}] ({gene})")

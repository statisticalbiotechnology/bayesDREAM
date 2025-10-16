"""
Example: Using MultiModalBayesDREAM with multiple molecular modalities.

This script demonstrates how to:
1. Load gene counts, transcript counts, and splice junction data
2. Create a multi-modal bayesDREAM model
3. Add different types of modalities (transcripts, splicing, custom)
4. Run the standard 3-step pipeline (technical, cis, trans)
"""

import pandas as pd
import numpy as np
from bayesDREAM import MultiModalBayesDREAM, Modality

# =============================================================================
# Example 1: Gene counts only (backward compatible)
# =============================================================================

def example_gene_only():
    """Standard single-modality usage (genes only)."""
    print("=" * 60)
    print("Example 1: Gene counts only (backward compatible)")
    print("=" * 60)

    # Load data
    meta = pd.read_csv('path/to/meta.csv')
    gene_counts = pd.read_csv('path/to/gene_counts.csv', index_col=0)

    # Create model (works exactly like original bayesDREAM)
    model = MultiModalBayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./output',
        label='gene_only_run'
    )

    print(model.list_modalities())

    # Set technical groups first (required before fit_technical)
    model.set_technical_groups(['cell_line'])

    # Run pipeline as usual
    model.fit_technical()
    model.fit_cis(sum_factor_col='sum_factor')
    model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')


# =============================================================================
# Example 2: Genes + Transcripts (isoform usage)
# =============================================================================

def example_with_transcripts():
    """Adding transcript-level data for isoform usage analysis."""
    print("=" * 60)
    print("Example 2: Genes + Transcript isoform usage")
    print("=" * 60)

    # Load data
    meta = pd.read_csv('path/to/meta.csv')
    gene_counts = pd.read_csv('path/to/gene_counts.csv', index_col=0)
    transcript_counts = pd.read_csv('path/to/transcript_counts.csv', index_col=0)
    transcript_meta = pd.read_csv('path/to/transcript_meta.csv')  # must have: transcript_id, gene

    # Create model with gene counts
    model = MultiModalBayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./output',
        label='gene_transcript_run'
    )

    # Add transcript modality as isoform usage (multinomial distribution)
    model.add_transcript_modality(
        transcript_counts=transcript_counts,
        transcript_meta=transcript_meta,
        use_isoform_usage=True,  # Model as proportional usage within gene
        name='transcript_usage'
    )

    # OR: Add as independent transcripts (negative binomial)
    model.add_transcript_modality(
        transcript_counts=transcript_counts,
        transcript_meta=transcript_meta,
        use_isoform_usage=False,  # Model each transcript independently
        name='transcript_counts'
    )

    print(model.list_modalities())

    # Access modalities
    transcript_mod = model.get_modality('transcript_usage')
    print(f"Transcript modality: {transcript_mod}")
    print(f"Feature metadata:\n{transcript_mod.feature_meta.head()}")

    # Set technical groups first (required before fit_technical)
    model.set_technical_groups(['cell_line'])

    # Run pipeline on primary (gene) modality
    model.fit_technical()
    model.fit_cis(sum_factor_col='sum_factor')
    model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')


# =============================================================================
# Example 3: Genes + Splicing (donor/acceptor/exon skipping)
# =============================================================================

def example_with_splicing():
    """Adding splicing junction data."""
    print("=" * 60)
    print("Example 3: Genes + Splicing modalities")
    print("=" * 60)

    # Load data
    meta = pd.read_csv('path/to/meta.csv')
    gene_counts = pd.read_csv('path/to/gene_counts.csv', index_col=0)
    sj_counts = pd.read_csv('path/to/SJ_counts.csv', index_col=0)
    sj_meta = pd.read_csv('path/to/SJ_meta.csv')

    # SJ metadata must have: coord.intron, chrom, intron_start, intron_end, strand
    # Optionally: gene_short_name.start, gene_short_name.end

    # Create model
    model = MultiModalBayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./output',
        label='gene_splicing_run'
    )

    # Add all splicing modalities
    model.add_splicing_modality(
        sj_counts=sj_counts,
        sj_meta=sj_meta,
        splicing_types=['donor', 'acceptor', 'exon_skip'],  # Which metrics to compute
        gene_of_interest='GFI1B',  # Filter to specific gene
        min_cell_total=1,          # Min reads for donor/acceptor
        min_total_exon=2,          # Min reads for exon skipping
        r_code_path='path/to/splicing code/CodeDump.R'  # Optional: specify R code location
    )

    print(model.list_modalities())

    # Access splicing modalities
    donor_mod = model.get_modality('splicing_donor')
    print(f"\nDonor usage modality: {donor_mod}")
    print(f"Distribution: {donor_mod.distribution}")  # 'multinomial'
    print(f"Dimensions: {donor_mod.dims}")
    print(f"Feature metadata:\n{donor_mod.feature_meta.head()}")

    acceptor_mod = model.get_modality('splicing_acceptor')
    print(f"\nAcceptor usage modality: {acceptor_mod}")

    exon_skip_mod = model.get_modality('splicing_exon_skip')
    print(f"\nExon skipping modality: {exon_skip_mod}")
    print(f"Distribution: {exon_skip_mod.distribution}")  # 'binomial'
    print(f"Has denominator: {exon_skip_mod.denominator is not None}")

    # Set technical groups first (required before fit_technical)
    model.set_technical_groups(['cell_line'])

    # Run pipeline
    model.fit_technical()
    model.fit_cis(sum_factor_col='sum_factor')
    model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')


# =============================================================================
# Example 4: Custom modalities (SpliZ, SpliZVD)
# =============================================================================

def example_custom_modalities():
    """Adding custom user-defined modalities."""
    print("=" * 60)
    print("Example 4: Custom modalities (SpliZ, SpliZVD)")
    print("=" * 60)

    # Load data
    meta = pd.read_csv('path/to/meta.csv')
    gene_counts = pd.read_csv('path/to/gene_counts.csv', index_col=0)

    # Create model
    model = MultiModalBayesDREAM(
        meta=meta,
        counts=gene_counts,
        cis_gene='GFI1B',
        output_dir='./output',
        label='custom_modalities_run'
    )

    # -------------------------------------------------------------------------
    # Add SpliZ scores (normal distribution)
    # -------------------------------------------------------------------------
    spliz_scores = pd.read_csv('path/to/SpliZ_counts.csv', index_col=0)  # genes × cells
    spliz_meta = pd.read_csv('path/to/SpliZ_meta.csv')  # gene metadata

    model.add_custom_modality(
        name='spliz',
        counts=spliz_scores,
        feature_meta=spliz_meta,
        distribution='normal'  # SpliZ scores follow normal distribution
    )

    # -------------------------------------------------------------------------
    # Add SpliZVD (3D multivariate normal: z0, z1, z2 per gene-cell)
    # -------------------------------------------------------------------------
    # Load 3 separate files and combine into 3D array
    z0 = pd.read_csv('path/to/spliZVD_z0.csv', index_col=0)
    z1 = pd.read_csv('path/to/spliZVD_z1.csv', index_col=0)
    z2 = pd.read_csv('path/to/spliZVD_z2.csv', index_col=0)

    # Stack into 3D array: (genes, cells, 3)
    genes = z0.index
    cells = z0.columns
    splizvd_array = np.stack([z0.values, z1.values, z2.values], axis=2)

    splizvd_meta = pd.DataFrame({'gene': genes})

    model.add_custom_modality(
        name='splizvd',
        counts=splizvd_array,
        feature_meta=splizvd_meta,
        distribution='mvnormal'  # Multivariate normal (3D)
    )

    print(model.list_modalities())

    # Access custom modalities
    spliz_mod = model.get_modality('spliz')
    print(f"\nSpliZ modality: {spliz_mod}")

    splizvd_mod = model.get_modality('splizvd')
    print(f"\nSpliZVD modality: {splizvd_mod}")
    print(f"Dimensions per feature: {splizvd_mod.dims['n_dimensions']}")

    # Set technical groups first (required before fit_technical)
    model.set_technical_groups(['cell_line'])

    # Run pipeline
    model.fit_technical()
    model.fit_cis(sum_factor_col='sum_factor')
    model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')


# =============================================================================
# Example 5: Pre-constructed modalities
# =============================================================================

def example_preconstruced_modalities():
    """Create modalities manually before initializing model."""
    print("=" * 60)
    print("Example 5: Pre-constructed modalities")
    print("=" * 60)

    # Load data
    meta = pd.read_csv('path/to/meta.csv')
    gene_counts = pd.read_csv('path/to/gene_counts.csv', index_col=0)
    transcript_counts = pd.read_csv('path/to/transcript_counts.csv', index_col=0)

    # Create modalities manually
    gene_meta = pd.DataFrame({'gene': gene_counts.index})
    gene_modality = Modality(
        name='gene',
        counts=gene_counts,
        feature_meta=gene_meta,
        distribution='negbinom',
        cells_axis=1
    )

    transcript_meta = pd.DataFrame({
        'transcript_id': transcript_counts.index,
        'gene': ['GENE1', 'GENE1', 'GENE2', 'GENE2']  # Example mapping
    })
    transcript_modality = Modality(
        name='transcript',
        counts=transcript_counts,
        feature_meta=transcript_meta,
        distribution='negbinom',
        cells_axis=1
    )

    # Initialize model with pre-constructed modalities
    modalities_dict = {
        'gene': gene_modality,
        'transcript': transcript_modality
    }

    model = MultiModalBayesDREAM(
        meta=meta,
        modalities=modalities_dict,
        cis_gene='GFI1B',
        primary_modality='gene',  # Which modality to use for cis/trans
        output_dir='./output',
        label='preconstructed_run'
    )

    print(model.list_modalities())

    # Set technical groups first (required before fit_technical)
    model.set_technical_groups(['cell_line'])

    # Run pipeline
    model.fit_technical()
    model.fit_cis(sum_factor_col='sum_factor')
    model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')


# =============================================================================
# Example 6: Subsetting and filtering modalities
# =============================================================================

def example_subsetting():
    """Demonstrate subsetting modalities by features or cells."""
    print("=" * 60)
    print("Example 6: Subsetting modalities")
    print("=" * 60)

    # Create a simple modality
    gene_counts = pd.DataFrame(
        np.random.poisson(10, (100, 50)),
        index=[f'GENE{i}' for i in range(100)],
        columns=[f'CELL{i}' for i in range(50)]
    )
    gene_meta = pd.DataFrame({'gene': gene_counts.index})

    modality = Modality(
        name='gene',
        counts=gene_counts,
        feature_meta=gene_meta,
        distribution='negbinom',
        cells_axis=1
    )

    print(f"Original modality: {modality}")

    # Subset to specific genes
    subset_genes = ['GENE1', 'GENE5', 'GENE10']
    gene_subset = modality.get_feature_subset(subset_genes)
    print(f"Gene subset: {gene_subset}")

    # Subset to specific cells
    subset_cells = [f'CELL{i}' for i in range(10)]
    cell_subset = modality.get_cell_subset(subset_cells)
    print(f"Cell subset: {cell_subset}")


# =============================================================================
# Run examples
# =============================================================================

if __name__ == '__main__':
    # Uncomment to run specific examples:

    # example_gene_only()
    # example_with_transcripts()
    # example_with_splicing()
    # example_custom_modalities()
    # example_preconstruced_modalities()
    # example_subsetting()

    print("\nSee function definitions for usage examples.")
    print("Update file paths and uncomment examples to run.")

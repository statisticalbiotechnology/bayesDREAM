"""
Transcript modality methods for bayesDREAM.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Union

from ..modality import Modality
from ..splicing import create_splicing_modality



class TranscriptModalityMixin:
    """Mixin for transcript modality support."""

def add_transcript_modality(
        self,
        transcript_counts: pd.DataFrame,
        transcript_meta: pd.DataFrame,
        modality_types: Union[str, List[str]] = 'counts',
        counts_name: str = 'transcript_counts',
        usage_name: str = 'transcript_usage'
):
    """
    Add transcript-level data as counts and/or isoform usage.

    Parameters
    ----------
    transcript_counts : pd.DataFrame
        Transcript counts (transcripts × cells)
    transcript_meta : pd.DataFrame
        Transcript metadata with required columns: 'transcript_id' and either
        'gene' or 'gene_name' or 'gene_id' (gene symbol or Ensembl ID)
    modality_types : str or list of str
        Which modalities to add: 'counts', 'usage', or both.
        Options: 'counts' (negbinom), 'usage' (multinomial), or ['counts', 'usage']
    counts_name : str
        Name for transcript counts modality (default: 'transcript_counts')
    usage_name : str
        Name for isoform usage modality (default: 'transcript_usage')
    """
    # Validate required columns
    if 'transcript_id' not in transcript_meta.columns:
        raise ValueError("transcript_meta must have 'transcript_id' column")

    # Flexible gene column detection (gene, gene_name, gene_id)
    gene_col = None
    for col in ['gene', 'gene_name', 'gene_id']:
        if col in transcript_meta.columns:
            gene_col = col
            break

    if gene_col is None:
        raise ValueError("transcript_meta must have one of: 'gene', 'gene_name', or 'gene_id' column")

    print(f"[INFO] Using '{gene_col}' column for gene identifiers")

    # Standardize to 'gene' column for internal processing
    if gene_col != 'gene':
        transcript_meta = transcript_meta.copy()
        transcript_meta['gene'] = transcript_meta[gene_col]

    # Validate transcript_id values are in transcript_counts
    meta_transcripts = set(transcript_meta['transcript_id'])
    count_transcripts = set(transcript_counts.index)
    missing_in_counts = meta_transcripts - count_transcripts

    if missing_in_counts:
        print(f"[INFO] {len(missing_in_counts)} transcript(s) in metadata not found in counts (will be skipped)")

    # Subset transcript_counts to valid cells
    valid_cells = self.meta['cell'].tolist()
    tx_cells = transcript_counts.columns.tolist()
    common_tx_cells = [c for c in tx_cells if c in valid_cells]

    if len(common_tx_cells) == 0:
        raise ValueError("No overlapping cells between transcript_counts and model cells")

    if len(common_tx_cells) < len(tx_cells):
        print(f"[INFO] Subsetting transcript_counts from {len(tx_cells)} to {len(common_tx_cells)} cells to match model")
        transcript_counts_subset = transcript_counts[common_tx_cells].copy()
    else:
        transcript_counts_subset = transcript_counts

    # Normalize modality_types to list
    if isinstance(modality_types, str):
        modality_types = [modality_types]

    # Validate modality types
    valid_types = {'counts', 'usage'}
    invalid = set(modality_types) - valid_types
    if invalid:
        raise ValueError(f"Invalid modality_types: {invalid}. Must be 'counts', 'usage', or both.")

    # Add counts modality
    if 'counts' in modality_types:
        # Subset metadata to transcripts present in counts
        valid_tx = [tx for tx in transcript_meta['transcript_id'] if tx in transcript_counts_subset.index]
        tx_meta_subset = transcript_meta[transcript_meta['transcript_id'].isin(valid_tx)].copy()

        if len(valid_tx) == 0:
            warnings.warn(f"No transcripts found in counts. Skipping '{counts_name}' modality.")
        else:
            # Ensure counts are in same order as metadata
            transcript_counts_ordered = transcript_counts_subset.loc[valid_tx]

            # Filter transcripts with zero standard deviation across ALL cells
            # (transcripts that are constant across all cells can't be modeled)
            tx_stds = transcript_counts_ordered.std(axis=1)
            zero_std_mask = tx_stds == 0
            num_zero_std = zero_std_mask.sum()

            if num_zero_std > 0:
                print(f"[INFO] Filtering {num_zero_std} transcript(s) with zero standard deviation across all cells")
                transcript_counts_ordered = transcript_counts_ordered.loc[~zero_std_mask]
                tx_meta_subset = tx_meta_subset[tx_meta_subset['transcript_id'].isin(transcript_counts_ordered.index)].copy()

            if len(transcript_counts_ordered) == 0:
                warnings.warn(f"No transcripts left after filtering zero-variance transcripts. Skipping '{counts_name}' modality.")
            else:
                modality = Modality(
                    name=counts_name,
                    counts=transcript_counts_ordered,
                    feature_meta=tx_meta_subset.reset_index(drop=True),
                    distribution='negbinom',
                    cells_axis=1
                )
                self.add_modality(counts_name, modality)

    # Add usage modality
    if 'usage' in modality_types:
        # Group transcripts by gene and create 3D array
        # Shape: (genes, cells, max_transcripts_per_gene)
        gene_to_transcripts = transcript_meta.groupby('gene')['transcript_id'].apply(list).to_dict()

        # Filter to transcripts present in counts
        gene_to_transcripts_filtered = {
            gene: [tx for tx in txs if tx in transcript_counts_subset.index]
            for gene, txs in gene_to_transcripts.items()
        }

        # Remove genes with no transcripts
        gene_to_transcripts_filtered = {
            gene: txs for gene, txs in gene_to_transcripts_filtered.items() if len(txs) > 0
        }

        if len(gene_to_transcripts_filtered) == 0:
            warnings.warn(f"No genes with transcripts found in counts. Skipping '{usage_name}' modality.")
        else:
            genes = sorted(gene_to_transcripts_filtered.keys())
            n_cells = len(common_tx_cells)
            max_transcripts = max(len(txs) for txs in gene_to_transcripts_filtered.values())

            counts_3d = np.zeros((len(genes), n_cells, max_transcripts))

            gene_meta_rows = []
            n_genes_dropped = 0

            for gene_idx, gene in enumerate(genes):
                transcripts = gene_to_transcripts_filtered[gene]

                # Skip genes with only one transcript (no isoform variation)
                if len(transcripts) < 2:
                    n_genes_dropped += 1
                    continue

                gene_meta_rows.append({
                    'gene': gene,
                    'transcripts': transcripts,
                    'n_transcripts': len(transcripts)
                })

                for tx_idx, tx in enumerate(transcripts):
                    if tx in transcript_counts_subset.index:
                        counts_3d[gene_idx, :, tx_idx] = transcript_counts_subset.loc[tx].values

            if n_genes_dropped > 0:
                print(f"[INFO] Dropped {n_genes_dropped} gene(s) with only 1 transcript (no isoform usage to model)")

            if len(gene_meta_rows) == 0:
                warnings.warn(f"No genes with multiple transcripts found. Skipping '{usage_name}' modality.")
            else:
                # Filter counts_3d to genes with multiple transcripts
                valid_gene_indices = [i for i, gene in enumerate(genes)
                                     if len(gene_to_transcripts_filtered[gene]) >= 2]
                counts_3d = counts_3d[valid_gene_indices, :, :]

                gene_meta_df = pd.DataFrame(gene_meta_rows)

                modality = Modality(
                    name=usage_name,
                    counts=counts_3d,
                    feature_meta=gene_meta_df,
                    distribution='multinomial',
                    cells_axis=1
                )
                self.add_modality(usage_name, modality)


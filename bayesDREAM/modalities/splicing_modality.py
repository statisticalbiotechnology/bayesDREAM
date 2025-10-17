"""
Splicing modality methods for bayesDREAM.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Union

from ..modality import Modality
from ..splicing import create_splicing_modality



class SplicingModalityMixin:
    """Mixin for splicing modality support."""

def add_splicing_modality(
        self,
        sj_counts: pd.DataFrame,
        sj_meta: pd.DataFrame,
        splicing_types: Union[str, List[str]] = ['donor', 'acceptor', 'exon_skip'],
        gene_counts: Optional[pd.DataFrame] = None,
        min_cell_total: int = 1,
        min_total_exon: int = 2
):
    """
    Add splicing modalities (raw SJ counts, donor usage, acceptor usage, exon skipping).

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with required columns: coord.intron, chrom, intron_start,
        intron_end, strand, gene_name_start, gene_name_end
        Optional columns: gene_id_start, gene_id_end (for Ensembl ID support)
    splicing_types : str or list
        Which splicing metrics to compute: 'sj', 'donor', 'acceptor', 'exon_skip', or list
    gene_counts : pd.DataFrame, optional
        Gene-level counts for SJ denominator (genes × cells).
        If not provided, will use primary gene counts from model (self.counts).
    min_cell_total : int
        Minimum reads for donor/acceptor
    min_total_exon : int
        Minimum reads for exon skipping
    """
    if isinstance(splicing_types, str):
        splicing_types = [splicing_types]

    # Get gene_counts (use model's counts if not provided)
    if gene_counts is None:
        if hasattr(self, 'counts') and self.counts is not None:
            gene_counts_to_use = self.counts
        else:
            raise ValueError("gene_counts must be provided or model must have been initialized with counts")
    else:
        gene_counts_to_use = gene_counts

    # Subset sj_counts to valid cells (cells in model.meta)
    # Allow sj_counts to have extra cells (will be discarded)
    # Also allow sj_counts to be missing some cells (that's OK)
    valid_cells = self.meta['cell'].tolist()
    sj_cells = sj_counts.columns.tolist()
    common_sj_cells = [c for c in sj_cells if c in valid_cells]

    if len(common_sj_cells) == 0:
        raise ValueError("No overlapping cells between sj_counts and model cells")

    if len(common_sj_cells) < len(sj_cells):
        print(f"[INFO] Subsetting sj_counts from {len(sj_cells)} to {len(common_sj_cells)} cells to match model")
        sj_counts_subset = sj_counts[common_sj_cells].copy()
    else:
        sj_counts_subset = sj_counts

    for stype in splicing_types:
        modality = create_splicing_modality(
            sj_counts=sj_counts_subset,
            sj_meta=sj_meta,
            splicing_type=stype,
            gene_counts=gene_counts_to_use,
            min_cell_total=min_cell_total,
            min_total_exon=min_total_exon
        )
        self.add_modality(f'splicing_{stype}', modality)


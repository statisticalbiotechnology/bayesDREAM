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
        sj_counts: Union[np.ndarray, pd.DataFrame],
        sj_meta: pd.DataFrame,
        splicing_types: Union[str, List[str]] = ['donor', 'acceptor', 'exon_skip'],
        gene_counts: Optional[pd.DataFrame] = None,
        min_cell_total: int = 1,
        min_total_exon: int = 2,
        cell_names: Optional[List[str]] = None,
        overwrite: bool = False
    ):
        """
        Add splicing modalities (raw SJ counts, donor usage, acceptor usage, exon skipping).

        Parameters
        ----------
        sj_counts : array or DataFrame
            Splice junction counts (junctions × cells).
            If DataFrame, cell names come from columns.
            If ndarray, use cell_names parameter to specify cell identifiers.
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
        cell_names : list of str, optional
            Cell names/identifiers (only used when sj_counts is ndarray, not DataFrame).
            Should match number of cells (axis 1).
        overwrite : bool, default=False
            Whether to overwrite existing modalities with the same name(s)
        """
        if isinstance(splicing_types, str):
            splicing_types = [splicing_types]

        # Handle DataFrame vs numpy array input
        if isinstance(sj_counts, pd.DataFrame):
            is_dataframe = True
            extracted_cell_names = sj_counts.columns.tolist()
            sj_index = sj_counts.index
        else:
            is_dataframe = False
            extracted_cell_names = cell_names
            # For numpy array, we need junction IDs from metadata
            if len(sj_meta) != sj_counts.shape[0]:
                raise ValueError(
                    f"sj_counts has {sj_counts.shape[0]} rows but "
                    f"sj_meta has {len(sj_meta)} rows. They must match."
                )
            sj_index = sj_meta['coord.intron'].tolist()

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

        if extracted_cell_names is None:
            # No cell names provided, use all cells
            common_sj_cells_idx = list(range(sj_counts.shape[1]))
            common_sj_cells = extracted_cell_names if extracted_cell_names else common_sj_cells_idx
            n_sj_cells = sj_counts.shape[1]
        else:
            # Have cell names, subset to valid cells
            sj_cells = extracted_cell_names
            common_sj_cells = [c for c in sj_cells if c in valid_cells]
            common_sj_cells_idx = [i for i, c in enumerate(sj_cells) if c in valid_cells]
            n_sj_cells = len(sj_cells)

            if len(common_sj_cells) == 0:
                raise ValueError("No overlapping cells between sj_counts and model cells")

            if len(common_sj_cells) < n_sj_cells:
                print(f"[INFO] Subsetting sj_counts from {n_sj_cells} to {len(common_sj_cells)} cells to match model")

        # Subset to common cells
        if is_dataframe:
            if extracted_cell_names is None or len(common_sj_cells) == n_sj_cells:
                sj_counts_subset = sj_counts
            else:
                sj_counts_subset = sj_counts[common_sj_cells].copy()
            # Update extracted_cell_names to reflect subset
            if extracted_cell_names is not None:
                extracted_cell_names = common_sj_cells
        else:
            # Numpy array - subset by indices
            if extracted_cell_names is None or len(common_sj_cells_idx) == sj_counts.shape[1]:
                sj_counts_subset = sj_counts
            else:
                sj_counts_subset = sj_counts[:, common_sj_cells_idx]
            # Update extracted_cell_names to reflect subset
            if extracted_cell_names is not None:
                extracted_cell_names = common_sj_cells
            # Convert to DataFrame for create_splicing_modality
            sj_counts_subset = pd.DataFrame(
                sj_counts_subset,
                index=sj_index,
                columns=extracted_cell_names if extracted_cell_names else list(range(sj_counts_subset.shape[1]))
            )

        for stype in splicing_types:
            modality = create_splicing_modality(
                sj_counts=sj_counts_subset,
                sj_meta=sj_meta,
                splicing_type=stype,
                gene_counts=gene_counts_to_use,
                min_cell_total=min_cell_total,
                min_total_exon=min_total_exon,
                cell_names=extracted_cell_names
            )
            self.add_modality(f'splicing_{stype}', modality, overwrite=overwrite)

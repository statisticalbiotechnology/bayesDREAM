"""
ATAC-seq modality methods for bayesDREAM.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Union

from ..modality import Modality
from ..splicing import create_splicing_modality



class ATACModalityMixin:
    """Mixin for ATAC-seq modality support."""

def add_atac_modality(
        self,
        atac_counts: Union[np.ndarray, pd.DataFrame],
        region_meta: pd.DataFrame,
        name: str = 'atac',
        cis_region: Optional[str] = None
):
    """
    Add ATAC-seq modality with genomic region annotations.

    ATAC fragment counts are treated as negative binomial data (like gene expression).
    Regions can be promoters, gene bodies, or distal elements (enhancers).

    Parameters
    ----------
    atac_counts : array or DataFrame
        Fragment counts per region per cell. Shape: (n_regions, n_cells)
    region_meta : pd.DataFrame
        Region metadata with required columns:
        - region_id : str, unique region identifier
        - region_type : str, one of ['promoter', 'gene_body', 'distal']
        - chrom : str, chromosome
        - start : int, start coordinate (0-based)
        - end : int, end coordinate
        - gene : str, associated gene (NA for distal regions)
        Optional columns: gene_name, gene_id, strand, tss_distance
    name : str, optional
        Name for this ATAC modality (default: 'atac')
    cis_region : str, optional
        Region ID to use as cis gene proxy (e.g., 'chr9:132283881-132284881')
        If provided, this region will be used in fit_cis() instead of gene expression

    Examples
    --------
    >>> # Load ATAC data
    >>> atac_counts = pd.read_csv('atac_counts.csv', index_col=0)
    >>> region_meta = pd.read_csv('region_meta.csv')
    >>>
    >>> # Add ATAC modality
    >>> model.add_atac_modality(
    ...     atac_counts=atac_counts,
    ...     region_meta=region_meta,
    ...     cis_region='chr9:132283881-132284881'  # GFI1B promoter
    ... )
    >>>
    >>> # Fit using ATAC
    >>> model.fit_technical(modality_name='atac')
    >>> model.fit_cis(modality_name='atac', cis_feature='chr9:132283881-132284881')
    """
    # Validate required columns
    required_cols = ['region_id', 'region_type', 'chrom', 'start', 'end', 'gene']
    missing_cols = set(required_cols) - set(region_meta.columns)
    if missing_cols:
        raise ValueError(
            f"region_meta missing required columns: {missing_cols}\n"
            f"Required: {required_cols}"
        )

    # Validate region_type values
    valid_types = {'promoter', 'gene_body', 'distal'}
    invalid_types = set(region_meta['region_type'].unique()) - valid_types
    if invalid_types:
        raise ValueError(
            f"Invalid region_type values: {invalid_types}\n"
            f"Must be one of: {valid_types}"
        )

    # Validate gene column: should be non-null for promoter/gene_body
    for region_type in ['promoter', 'gene_body']:
        type_mask = region_meta['region_type'] == region_type
        if type_mask.any():
            null_genes = region_meta[type_mask]['gene'].isnull().sum()
            if null_genes > 0:
                raise ValueError(
                    f"Found {null_genes} {region_type} regions with null gene annotations.\n"
                    f"All {region_type} regions must have an associated gene."
                )

    # Set region_id as index if not already
    if 'region_id' in region_meta.columns and region_meta.index.name != 'region_id':
        region_meta = region_meta.set_index('region_id')

    # Store cis_region if provided
    if cis_region is not None:
        if cis_region not in region_meta.index:
            raise ValueError(
                f"cis_region '{cis_region}' not found in region_meta.index.\n"
                f"Available regions: {region_meta.index[:5].tolist()}..."
            )
        # Store for later use in fit_cis
        if not hasattr(self, 'cis_feature_map'):
            self.cis_feature_map = {}
        self.cis_feature_map[name] = cis_region
        print(f"[INFO] Set cis region for '{name}' modality: {cis_region}")

    # Convert to array if DataFrame
    if isinstance(atac_counts, pd.DataFrame):
        counts_array = atac_counts.values
        counts_index = atac_counts.index
        is_dataframe = True
    else:
        counts_array = np.asarray(atac_counts)
        counts_index = None
        is_dataframe = False

    # Apply negbinom filtering (zero std)
    feature_stds = counts_array.std(axis=1)
    valid_features = feature_stds != 0
    num_filtered = (~valid_features).sum()

    if num_filtered > 0:
        print(f"[INFO] Filtering {num_filtered} ATAC region(s) with zero std in '{name}' modality")
        counts_array = counts_array[valid_features, :]
        region_meta = region_meta.iloc[valid_features].copy()

    if counts_array.shape[0] == 0:
        raise ValueError(f"No ATAC regions left after filtering in '{name}' modality!")

    # Convert back to DataFrame if input was DataFrame
    if is_dataframe:
        counts_final = pd.DataFrame(
            counts_array,
            index=region_meta.index,
            columns=atac_counts.columns
        )
    else:
        counts_final = counts_array

    # Create modality
    modality = Modality(
        name=name,
        counts=counts_final,
        feature_meta=region_meta,
        distribution='negbinom',  # ATAC uses negbinom like gene expression
        cells_axis=1
    )

    self.add_modality(name, modality)
    print(f"[INFO] Added ATAC modality '{name}' with {modality.dims['n_features']} regions")

    # Print region type summary
    region_summary = region_meta['region_type'].value_counts()
    print(f"[INFO] Region types: {region_summary.to_dict()}")


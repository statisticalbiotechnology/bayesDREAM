"""
Core bayesDREAM implementation.

This module contains the _BayesDREAMCore base class with delegation to
specialized fitters for technical, cis, and trans modeling.
"""

import os
import subprocess
import warnings
from typing import Dict, Optional, List, Union
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer
import h5py
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import gc
from scipy import sparse

# Import utility functions and modules
from .utils import (
    set_max_threads,
    find_beta,
    calculate_mu_x_guide,
    Hill_based_positive,
    Hill_based_negative,
    Hill_based_piecewise,
    Polynomial_function,
    cutoff_sigmoid,
    sample_or_use_point,
    check_tensor
)
from .modality import Modality
from .splicing import create_splicing_modality
from .fitting.distributions import get_observation_sampler, requires_denominator, is_3d_distribution

# Import fitters
from .fitting import TechnicalFitter, CisFitter, TransFitter
from .io import ModelSaver, ModelLoader
from .plotting.model_plots import PlottingMixin

warnings.simplefilter(action="ignore", category=FutureWarning)


class _BayesDREAMCore(PlottingMixin):
    """
    Internal core class for the three-step Bayesian Dosage Response Effects Across Modalities framework:

    1) Optional technical group prefit (modeling alpha_y for NTC),
    2) Fitting cis effects (model_x),
    3) Fitting trans effects (model_y).
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts: pd.DataFrame,
        gene_meta: pd.DataFrame = None,
        cis_gene: str = None,
        guide_covariates: list[str] = None,
        guide_covariates_ntc: list[str] = None,
        sum_factor_col: str = 'sum_factor',
        output_dir: str = "./model_out",
        label: str = None,
        device: str = None,
        random_seed: int = 2402,
        cores: int = 1,
        guide_assignment: np.ndarray = None,
        guide_meta: pd.DataFrame = None,
        guide_target: pd.DataFrame = None,
        exclude_targets: list[str] = None,
        require_ntc: bool = True
    ):
        """
        Initialize the model with the metadata and count matrices.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell metadata DataFrame. For single-guide mode: includes columns cell, guide, target, sum_factor, etc.
            For high MOI mode: includes columns cell, sum_factor, etc. (NO guide or target columns)
            May optionally include technical group identifiers like 'cell_line', 'batch', 'lane', etc.
        counts : pd.DataFrame
            Counts DataFrame (genes as rows, cell barcodes as columns)
        gene_meta : pd.DataFrame, optional
            Gene metadata DataFrame with genes as rows. Required to have at least one identifier column.
            Recommended columns: 'gene' (or use index), 'gene_name', 'gene_id'
            If not provided, will create minimal metadata from counts.index
        cis_gene : str
            The 'X' gene for cis modeling
        guide_covariates : list of str
            List of columns used to construct guide_used for non-NTC guides (single-guide mode only).
        guide_covariates_ntc : list of str or None
            List of columns used to construct guide_used for NTC guides (single-guide mode only).
        output_dir : str
            Where to save results
        label : str
            A label to prefix output files
        device : str or None
            "cuda" or "cpu" or None. If None, auto-detect.
        random_seed : int
            Random seed for reproducibility
        cores : int
            Number of CPU cores for Pyro to use
        guide_assignment : np.ndarray, optional
            Binary matrix [N, G] for high MOI mode. Each row represents a cell, each column a guide.
            guide_assignment[i, j] = 1 if cell i has guide j, else 0.
            If provided, must also provide guide_meta. Activates high MOI mode.
            Note: If dimensions are [G, N], will auto-transpose with a warning.
        guide_meta : pd.DataFrame, optional
            Guide metadata DataFrame for high MOI mode. Must have column 'guide'.
            Can optionally have 'target' column for simple one-to-one guide-target mapping.
            Index must match the column order of guide_assignment matrix.
            If provided, must also provide guide_assignment.
        guide_target : pd.DataFrame, optional
            High MOI only: Many-to-many guide-target relationship DataFrame.
            Must have columns 'guide' and 'target'. Multiple rows can have the same guide
            (one guide can target multiple genes). If provided, overrides guide_meta['target'].
            This allows flexible specification of guides with multiple possible targets.
            NTC guides can be specified with 'ntc', 'NTC', 'non-targeting', or 'non-targeting-control'.
        exclude_targets : list[str], optional
            High MOI only: List of target gene names to exclude. Cells with ANY guide targeting
            a gene in this list will be removed from analysis, regardless of other guides present.
            Example: exclude_targets=['MYB'] will remove cells with guides targeting MYB,
            even if they also have NTC or cis-targeting guides.
        require_ntc : bool, optional
            If True (default), requires NTC cells in meta for single-guide mode.
            Set to False when subsetting a model that has already had technical fitting done,
            or when NTC cells are not needed (e.g., stress testing without NTC).
        """
        
        if label is None and cis_gene is not None:
            label = cis_gene
        elif label is None:
            label = ""

        # Basic assignments
        self.meta = meta.copy()

        # Handle counts - can be DataFrame or sparse matrix
        self.is_sparse_counts = sparse.issparse(counts)
        if self.is_sparse_counts:
            # Keep sparse matrix as-is
            self.counts = counts.copy() if hasattr(counts, 'copy') else counts.tocsr()
            # Extract gene and cell names from DataFrame metadata
            if isinstance(counts, pd.DataFrame):
                self._gene_names = counts.index.tolist()
                self._cell_names = counts.columns.tolist()
            else:
                # counts is sparse array - will extract names from gene_meta and meta later
                self._gene_names = None
                self._cell_names = None
            print(f"[SPARSE] Keeping counts as sparse matrix (shape: {counts.shape}, sparsity: {1 - counts.nnz / (counts.shape[0] * counts.shape[1]):.2%} zeros)")
        else:
            # DataFrame or dense array
            if isinstance(counts, pd.DataFrame):
                self.counts = counts.copy()
                self._gene_names = counts.index.tolist()
                self._cell_names = counts.columns.tolist()
            else:
                # Dense numpy array - store as-is, extract names later
                self.counts = counts.copy() if hasattr(counts, 'copy') else counts
                self._gene_names = None
                self._cell_names = None

        self.cis_gene = cis_gene
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.label = label

        # ==============================================================================
        # Detect high MOI mode and validate
        # ==============================================================================
        if guide_assignment is not None or guide_meta is not None:
            if guide_assignment is None or guide_meta is None:
                raise ValueError(
                    "Both guide_assignment and guide_meta must be provided for high MOI mode. "
                    "Got guide_assignment={}, guide_meta={}".format(
                        type(guide_assignment).__name__ if guide_assignment is not None else None,
                        type(guide_meta).__name__ if guide_meta is not None else None
                    )
                )
            self.is_high_moi = True
            print("[INFO] High MOI mode detected")

            # Validate guide_assignment shape
            if guide_assignment.ndim != 2:
                raise ValueError(
                    f"guide_assignment must be a 2D matrix (cells × guides), "
                    f"but got shape {guide_assignment.shape} with {guide_assignment.ndim} dimensions"
                )

            # Auto-detect and transpose if dimensions are swapped
            # Expected: (n_cells, n_guides)
            # If user provides (n_guides, n_cells), transpose it
            dim0, dim1 = guide_assignment.shape
            n_guides_meta = len(guide_meta)
            n_cells_meta = len(meta)

            # Check if dimensions match expected orientation
            if dim1 == n_guides_meta and dim0 == n_cells_meta:
                # Correct orientation: (cells, guides)
                N_cells_assignment, G_guides = dim0, dim1
            elif dim0 == n_guides_meta and dim1 == n_cells_meta:
                # Transposed: (guides, cells) - auto-fix
                warnings.warn(
                    f"[HIGH MOI] guide_assignment appears to be transposed (shape {guide_assignment.shape} = guides × cells). "
                    f"Expected (cells × guides). Auto-transposing to ({dim1}, {dim0}).",
                    UserWarning
                )
                guide_assignment = guide_assignment.T
                N_cells_assignment, G_guides = guide_assignment.shape
            else:
                # Cannot determine orientation - provide helpful error
                raise ValueError(
                    f"guide_assignment shape {guide_assignment.shape} does not match expected dimensions:\n"
                    f"  - guide_meta has {n_guides_meta} guides\n"
                    f"  - meta has {n_cells_meta} cells\n"
                    f"Expected guide_assignment shape: ({n_cells_meta}, {n_guides_meta}) [cells × guides]\n"
                    f"Got: {guide_assignment.shape}\n"
                    f"Please check your guide_assignment matrix orientation."
                )

            # Validate guide_meta matches resolved dimensions
            if len(guide_meta) != G_guides:
                raise ValueError(
                    f"guide_meta has {len(guide_meta)} rows but guide_assignment has {G_guides} guides (columns). "
                    f"These dimensions must match."
                )

            # Validate guide_meta has 'guide' column
            if 'guide' not in guide_meta.columns:
                raise ValueError(
                    f"guide_meta missing required column 'guide'. "
                    f"Available columns: {list(guide_meta.columns)}"
                )

            # Store guide assignment and metadata
            self.guide_assignment = guide_assignment.copy()
            self.guide_meta = guide_meta.copy()

            # Create guide_code mapping for guide_meta
            self.guide_meta['guide_code'] = range(G_guides)

            # Process guide-target relationships
            # Priority: guide_target > guide_meta['target']
            if guide_target is not None:
                # Validate guide_target DataFrame
                required_gt_cols = {'guide', 'target'}
                missing_gt_cols = required_gt_cols - set(guide_target.columns)
                if missing_gt_cols:
                    raise ValueError(
                        f"guide_target missing required columns: {missing_gt_cols}. "
                        f"Available columns: {list(guide_target.columns)}"
                    )

                # Create guide -> list of targets mapping
                guide_targets_dict = {}
                for _, row in guide_target.iterrows():
                    guide_name = row['guide']
                    target = row['target']
                    if guide_name not in guide_targets_dict:
                        guide_targets_dict[guide_name] = []
                    guide_targets_dict[guide_name].append(target)

                # Store for later use
                self.guide_targets_dict = guide_targets_dict
                print(f"[INFO] Using guide_target DataFrame: {len(guide_target)} guide-target relationships")

            elif 'target' in guide_meta.columns:
                # Use simple one-to-one mapping from guide_meta
                guide_targets_dict = {
                    row['guide']: [row['target']]
                    for _, row in guide_meta.iterrows()
                }
                self.guide_targets_dict = guide_targets_dict
                print(f"[INFO] Using guide_meta['target'] for one-to-one guide-target mapping")
            else:
                raise ValueError(
                    "Either guide_target DataFrame or guide_meta['target'] column must be provided "
                    "to specify guide-target relationships in high MOI mode."
                )

            print(f"[INFO] High MOI: {N_cells_assignment} cells (from guide_assignment), {G_guides} guides")
            print(f"[INFO] Average guides per cell: {guide_assignment.sum(axis=1).mean():.2f}")

        else:
            self.is_high_moi = False

        # Handle gene metadata and extract gene names
        if gene_meta is None:
            # Create minimal gene metadata from counts
            print("[INFO] No gene_meta provided - creating minimal metadata")
            if isinstance(counts, pd.DataFrame):
                # DataFrame: use string index from DataFrame
                gene_names = counts.index.tolist()
                self.gene_meta = pd.DataFrame({
                    'gene': gene_names
                }, index=gene_names)
                self._gene_names = gene_names
            else:
                # Matrix/array: use NUMERIC index [0, 1, 2, ...]
                n_features = counts.shape[0]
                self.gene_meta = pd.DataFrame({
                    'feature_id': range(n_features)
                }, index=range(n_features))
                self._gene_names = None  # No string names for matrices
                print(f"[INFO] Using numeric feature indices [0..{n_features-1}] for matrix/array")
        else:
            # Validate and process provided gene_meta
            gene_meta = gene_meta.copy()

            # Check that gene_meta has at least one identifier column
            has_gene_col = 'gene' in gene_meta.columns
            has_gene_name = 'gene_name' in gene_meta.columns
            has_gene_id = 'gene_id' in gene_meta.columns
            has_feature_id = 'feature_id' in gene_meta.columns

            if not has_gene_col and not has_gene_name and not has_gene_id and not has_feature_id:
                raise ValueError(
                    "gene_meta must have at least one identifier column: "
                    "'gene', 'gene_name', 'gene_id', or 'feature_id'"
                )

            # Handle based on counts type
            if isinstance(counts, pd.DataFrame):
                # DataFrame: gene_meta should use same string index as counts
                # Ensure we have a 'gene' column for internal use
                if not has_gene_col:
                    if gene_meta.index.name is not None:
                        gene_meta['gene'] = gene_meta.index
                        print(f"[INFO] Using gene_meta index ('{gene_meta.index.name}') as 'gene' column")
                    elif has_gene_name:
                        gene_meta['gene'] = gene_meta['gene_name']
                        print("[INFO] Using 'gene_name' column as 'gene' identifier")
                    elif has_gene_id:
                        gene_meta['gene'] = gene_meta['gene_id']
                        print("[INFO] Using 'gene_id' column as 'gene' identifier")

                counts_gene_names = counts.index.tolist()

                # Ensure gene_meta index matches counts index
                if not gene_meta.index.equals(pd.Index(counts_gene_names)):
                    # Try to reindex by matching identifiers
                    if set(counts_gene_names).issubset(set(gene_meta.index)):
                        gene_meta = gene_meta.loc[counts_gene_names]
                    else:
                        warnings.warn(
                            f"[WARNING] gene_meta index does not match counts index. "
                            f"Attempting to match by identifier columns.",
                            UserWarning
                        )
                        # Try matching by columns
                        matched_indices = []
                        for gene in counts_gene_names:
                            found = False
                            for col in ['gene', 'gene_name', 'gene_id']:
                                if col in gene_meta.columns and gene in gene_meta[col].values:
                                    matched_indices.append(gene_meta[gene_meta[col] == gene].index[0])
                                    found = True
                                    break
                            if not found:
                                # Add minimal metadata for missing gene
                                matched_indices.append(gene)

                        # Add any missing genes
                        missing = set(counts_gene_names) - set(gene_meta.index)
                        if missing:
                            missing_df = pd.DataFrame({'gene': list(missing)}, index=list(missing))
                            gene_meta = pd.concat([gene_meta, missing_df])

                        gene_meta = gene_meta.loc[counts_gene_names]

                self._gene_names = counts_gene_names
            else:
                # Matrix/array: ensure gene_meta has NUMERIC index [0, 1, 2, ...]
                n_features = counts.shape[0]

                # Check if gene_meta already has numeric index of correct length
                if isinstance(gene_meta.index, pd.RangeIndex) and len(gene_meta) == n_features:
                    # Already has correct numeric index
                    pass
                elif all(isinstance(idx, (int, np.integer)) for idx in gene_meta.index) and len(gene_meta) == n_features:
                    # Has integer index - ensure it's a proper range
                    gene_meta.index = range(n_features)
                else:
                    # Reset to numeric index
                    if len(gene_meta) != n_features:
                        raise ValueError(
                            f"gene_meta has {len(gene_meta)} rows but counts has {n_features} features. "
                            f"For matrix/array counts, gene_meta must have exactly {n_features} rows."
                        )
                    gene_meta.index = range(n_features)
                    print(f"[INFO] Reset gene_meta to numeric index [0..{n_features-1}] to match matrix/array")

                self._gene_names = None  # No string names for matrices

            self.gene_meta = gene_meta

            # Print summary
            meta_cols = [c for c in ['gene', 'gene_name', 'gene_id', 'feature_id'] if c in self.gene_meta.columns]
            print(f"[INFO] Feature metadata loaded with {len(self.gene_meta)} features and columns: {meta_cols}")

        # Ensure guide_covariates and guide_covariates_ntc are always lists
        if guide_covariates is None:
            guide_covariates = []

        if guide_covariates_ntc is None:
            guide_covariates_ntc = []

        # Input checks - different requirements for single-guide vs high MOI mode
        if self.is_high_moi:
            # High MOI mode: do NOT require 'guide' or 'target' in meta
            required_cols = {"cell", sum_factor_col} | set(guide_covariates) | set(guide_covariates_ntc)
            missing_cols = required_cols - set(self.meta.columns)
            if missing_cols:
                raise ValueError(f"[High MOI] Missing required columns in meta: {missing_cols}")

            # Validate guide_assignment matches meta length
            if len(self.meta) != N_cells_assignment:
                raise ValueError(
                    f"[High MOI] guide_assignment has {N_cells_assignment} rows but meta has {len(self.meta)} rows. "
                    f"These dimensions must match."
                )

        else:
            # Single-guide mode: require 'guide' and 'target' in meta
            required_cols = {"target", "cell", sum_factor_col, "guide"} | set(guide_covariates) | set(guide_covariates_ntc)
            missing_cols = required_cols - set(self.meta.columns)
            if missing_cols:
                raise ValueError(f"[Single-guide] Missing required columns in meta: {missing_cols}")

            if require_ntc and "ntc" not in self.meta["target"].values:
                raise ValueError(
                    "The 'target' column in meta must contain 'ntc'. "
                    "If you have already run fit_technical() and want to subset without NTC cells, "
                    "use require_ntc=False."
                )

        # Populate cell names if not already set
        if self._cell_names is None:
            if isinstance(counts, pd.DataFrame):
                self._cell_names = counts.columns.tolist()
            else:
                # Use meta['cell'] as cell names
                self._cell_names = self.meta['cell'].tolist()

        if not set(self.meta["cell"]).issubset(set(self._cell_names)):
            raise ValueError("The 'cell' column in meta must correspond 1:1 with the cell names in counts.")

        if (self.meta[sum_factor_col] <= 0).any():
            raise ValueError(f"All values in sum_factor_col={sum_factor_col} column must be strictly greater than 0.")

        # For high MOI mode, create 'target' column based on guide assignment
        if self.is_high_moi:
            # Helper function to normalize NTC target names (case-insensitive)
            def is_ntc_target(target_name):
                """Check if target name is NTC (flexible matching)."""
                ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
                return target_name in ntc_variants

            # Classify each guide based on its targets (from guide_targets_dict)
            # A guide can have multiple targets, so we check if any match NTC, cis, or excluded
            ntc_guide_indices = []
            cis_guide_indices = []
            exclude_guide_indices = []

            for guide_idx, guide_row in self.guide_meta.iterrows():
                guide_name = guide_row['guide']
                targets = self.guide_targets_dict.get(guide_name, [])

                # Check if this guide has ANY NTC target
                if any(is_ntc_target(t) for t in targets):
                    ntc_guide_indices.append(guide_idx)

                # Check if this guide has ANY cis_gene target
                if self.cis_gene in targets:
                    cis_guide_indices.append(guide_idx)

                # Check if this guide has ANY excluded target
                if exclude_targets is not None and any(t in exclude_targets for t in targets):
                    exclude_guide_indices.append(guide_idx)

            ntc_guide_indices = np.array(ntc_guide_indices)
            cis_guide_indices = np.array(cis_guide_indices)
            exclude_guide_indices = np.array(exclude_guide_indices)

            # Determine which cells have these guide types
            if len(exclude_guide_indices) > 0:
                has_excluded_guide = self.guide_assignment[:, exclude_guide_indices].sum(axis=1) > 0
            else:
                has_excluded_guide = np.zeros(len(self.guide_assignment), dtype=bool)

            if len(ntc_guide_indices) > 0:
                has_ntc_guide = self.guide_assignment[:, ntc_guide_indices].sum(axis=1) > 0
            else:
                has_ntc_guide = np.zeros(len(self.guide_assignment), dtype=bool)

            if len(cis_guide_indices) > 0:
                has_cis_guide = self.guide_assignment[:, cis_guide_indices].sum(axis=1) > 0
            else:
                has_cis_guide = np.zeros(len(self.guide_assignment), dtype=bool)

            # Cell classification:
            # - If cell has ANY excluded guides -> target = 'excluded' (will be removed)
            # - Else if cell has ANY cis guides -> target = cis_gene
            # - Else if cell has ANY NTC guides (but no cis) -> target = 'ntc'
            # - Else -> target = 'other' (will be removed)
            targets = []
            for i in range(len(self.guide_assignment)):
                if has_excluded_guide[i]:
                    # Cell has guide(s) targeting excluded gene(s) - remove
                    targets.append('excluded')
                elif has_cis_guide[i]:
                    # Cell has cis guide(s) - regardless of other guides
                    targets.append(self.cis_gene)
                elif has_ntc_guide[i]:
                    # Cell has NTC guide(s) but no cis guides
                    # (may also have "other" guides - these are ignored)
                    targets.append('ntc')
                else:
                    # Cell has ONLY "other" guides (no NTC, no cis, no excluded)
                    targets.append('other')

            self.meta['target'] = targets

            # Add guide_code column (not meaningful in high MOI, marked as -1)
            self.meta['guide_code'] = -1

            ntc_count = (np.array(targets) == 'ntc').sum()
            cis_count = (np.array(targets) == self.cis_gene).sum()
            other_count = (np.array(targets) == 'other').sum()
            excluded_count = (np.array(targets) == 'excluded').sum()
            print(f"[INFO] Cell classification before subsetting:")
            print(f"  NTC cells (NTC guides, no cis): {ntc_count}")
            print(f"  {self.cis_gene}-targeting cells (any cis guides): {cis_count}")
            print(f"  Other-only cells (will be removed): {other_count}")
            if exclude_targets is not None:
                print(f"  Excluded cells (guides targeting {exclude_targets}): {excluded_count}")

        # Subset meta and counts to relevant cells
        valid_cells = self.meta[self.meta["target"].isin(["ntc", self.cis_gene])]["cell"].unique()
        if len(valid_cells) < len(self.meta["cell"].unique()):
            warnings.warn(
                f"Subsetting reduced the number of cells in the metadata from {len(self.meta['cell'].unique())} to {len(valid_cells)}. "
                "This may impact downstream analysis.",
                UserWarning
            )
        self.meta = self.meta[self.meta["cell"].isin(valid_cells)].copy()

        # Subset counts by cells - works for both DataFrame and sparse
        if isinstance(self.counts, pd.DataFrame):
            self.counts = self.counts[valid_cells].copy()
        else:
            # Sparse or dense array - subset by column indices
            cell_indices = [i for i, cell in enumerate(self._cell_names) if cell in valid_cells]
            if self.is_sparse_counts:
                self.counts = self.counts[:, cell_indices]
            else:
                self.counts = self.counts[:, cell_indices]
            # Update cell names
            self._cell_names = [self._cell_names[i] for i in cell_indices]

        # For high MOI: subset guide_assignment to remove "other"-targeting guide columns
        if self.is_high_moi:
            # Keep only NTC and cis-gene targeting guides
            # A guide is kept if it has ANY NTC or cis target among its possible targets
            keep_guide_indices = []

            def is_ntc_target(target_name):
                """Check if target name is NTC (flexible matching)."""
                ntc_variants = {'ntc', 'NTC', 'non-targeting', 'non-targeting-control', 'Non-Targeting'}
                return target_name in ntc_variants

            for guide_idx, guide_row in self.guide_meta.iterrows():
                guide_name = guide_row['guide']
                targets = self.guide_targets_dict.get(guide_name, [])

                # Keep if ANY target is NTC or cis_gene
                if any(is_ntc_target(t) for t in targets) or self.cis_gene in targets:
                    keep_guide_indices.append(guide_idx)

            keep_guide_indices = np.array(keep_guide_indices)

            n_guides_before = self.guide_assignment.shape[1]
            n_guides_after = len(keep_guide_indices)

            # Subset guide_assignment columns
            self.guide_assignment = self.guide_assignment[:, keep_guide_indices]

            # Subset guide_meta rows
            self.guide_meta = self.guide_meta.iloc[keep_guide_indices].copy()

            # Update guide_code to match new indices
            self.guide_meta['guide_code'] = range(len(self.guide_meta))

            # Also update guide_targets_dict to only include kept guides
            kept_guide_names = set(self.guide_meta['guide'].values)
            self.guide_targets_dict = {
                guide: targets
                for guide, targets in self.guide_targets_dict.items()
                if guide in kept_guide_names
            }

            print(f"[INFO] Subsetted guides from {n_guides_before} to {n_guides_after} (keeping NTC + {self.cis_gene} guides only)")

        # Remove genes with zero total counts - works for DataFrame, dense, and sparse
        if isinstance(self.counts, pd.DataFrame):
            gene_sums = self.counts.sum(axis=1).values
        elif self.is_sparse_counts:
            # Sparse matrix - sum along axis 1 (cells)
            gene_sums = np.array(self.counts.sum(axis=1)).flatten()
        else:
            # Dense array
            gene_sums = self.counts.sum(axis=1)

        detected_mask = gene_sums > 0
        num_removed = (~detected_mask).sum()

        # Raise error if cis gene is undetected
        if self.cis_gene is not None:
            if isinstance(self.counts, pd.DataFrame):
                # DataFrame: check if cis_gene is in index
                if self.cis_gene not in self.counts.index:
                    raise ValueError(f"[ERROR] The cis gene '{self.cis_gene}' not found in counts index!")
                cis_idx = self.counts.index.get_loc(self.cis_gene)
            else:
                # Matrix/array: search gene_meta columns for cis_gene
                # Try columns in order: gene_name, gene, gene_id, feature_id
                cis_idx = None
                for col in ['gene_name', 'gene', 'gene_id', 'feature_id']:
                    if col in self.gene_meta.columns:
                        matches = self.gene_meta[self.gene_meta[col] == self.cis_gene]
                        if len(matches) > 0:
                            # Found cis gene - use its numeric index for validation
                            cis_idx = matches.index[0]
                            break

                if cis_idx is None:
                    available_cols = [c for c in ['gene_name', 'gene', 'gene_id', 'feature_id'] if c in self.gene_meta.columns]
                    raise ValueError(
                        f"[ERROR] The cis gene '{self.cis_gene}' not found in gene_meta columns {available_cols}!"
                    )

            # Check if cis gene has zero counts
            if not detected_mask[cis_idx]:
                raise ValueError(f"[ERROR] The cis gene '{self.cis_gene}' has zero counts after subsetting!")

        # Subset counts to detected genes only - works for DataFrame, dense, and sparse
        if isinstance(self.counts, pd.DataFrame):
            self.counts = self.counts.loc[detected_mask]
            self._gene_names = self.counts.index.tolist()
            self.gene_meta = self.gene_meta.loc[self._gene_names]
        else:
            # Sparse or dense array - subset by row indices
            gene_indices = np.where(detected_mask)[0]
            if self.is_sparse_counts:
                self.counts = self.counts[gene_indices, :]
            else:
                self.counts = self.counts[gene_indices, :]

            # Subset gene_meta by numeric indices
            self.gene_meta = self.gene_meta.iloc[gene_indices].copy()

            # CRITICAL: Reset index to [0, 1, 2, ..., n_remaining-1]
            # This ensures row i in counts corresponds to row i in gene_meta
            self.gene_meta.index = range(len(self.gene_meta))
            print(f"[INFO] After filtering: reset gene_meta index to [0..{len(self.gene_meta)-1}]")

            # _gene_names remains None for matrices
            self._gene_names = None

        # Set trans genes
        if isinstance(self.counts, pd.DataFrame):
            # DataFrame: use string names
            self.trans_genes = [g for g in self._gene_names if g != self.cis_gene]
        else:
            # Matrix/array: trans_genes is list of numeric indices excluding cis
            # Find which row has the cis gene after filtering
            cis_row_after_filter = None
            if self.cis_gene is not None:
                for col in ['gene_name', 'gene', 'gene_id', 'feature_id']:
                    if col in self.gene_meta.columns:
                        matches = self.gene_meta[self.gene_meta[col] == self.cis_gene]
                        if len(matches) > 0:
                            cis_row_after_filter = matches.index[0]
                            break

            if cis_row_after_filter is not None:
                # trans_genes = all indices except cis row
                self.trans_genes = [i for i in range(len(self.gene_meta)) if i != cis_row_after_filter]
            else:
                # No cis gene - all rows are trans
                self.trans_genes = list(range(len(self.gene_meta)))

        if num_removed > 0:
            warnings.warn(
                f"[WARNING] {num_removed} gene(s) had zero counts after subsetting and were removed from the counts matrix.",
                UserWarning
            )
        
        # Ensure same order of meta and counts
        # Use _cell_names for sparse/dense array compatibility
        if isinstance(self.counts, pd.DataFrame):
            self.meta = self.meta.set_index("cell", drop=False).loc[self.counts.columns]
        else:
            # For sparse/dense arrays, reorder meta to match _cell_names
            self.meta = self.meta.set_index("cell", drop=False).loc[self._cell_names]

        # Construct guide_used column (single-guide mode only)
        if not self.is_high_moi:
            self.meta["guide_used"] = self.meta.apply(
                lambda row: f"{row['guide']}_{'_'.join(str(row[cov]) for cov in (guide_covariates_ntc if row['target'] == 'ntc' else set(guide_covariates_ntc + guide_covariates)))}",
                axis=1
            )
            # one-hot encode guides
            self.meta['guide_code'] = pd.Categorical(self.meta['guide_used']).codes
        else:
            # High MOI: guide_used not needed, guide_code already set to -1
            self.meta["guide_used"] = "highmoi"  # Placeholder for compatibility
            # Convert guide_assignment to tensor and store after subsetting
            # Need to subset guide_assignment to match the subsetted cells
            if isinstance(self.counts, pd.DataFrame):
                cell_indices = [list(self.counts.columns).index(cell) for cell in self.meta['cell']]
            else:
                # Use _cell_names for sparse/dense array compatibility
                cell_indices = [self._cell_names.index(cell) for cell in self.meta['cell']]
            self.guide_assignment = self.guide_assignment[cell_indices, :]
            self.guide_assignment_tensor = torch.tensor(
                self.guide_assignment,
                dtype=torch.float32,
                device='cpu'  # Will move to device below
            )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Move guide_assignment_tensor to device if in high MOI mode
        if self.is_high_moi:
            self.guide_assignment_tensor = self.guide_assignment_tensor.to(self.device)

        # Set random seeds & threads
        pyro.set_rng_seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        set_max_threads(cores)

        # Bookkeeping for results
        self.alpha_x_prefit = None    # from step1
        self.alpha_x_type = None    # from step1
        # NOTE: alpha_y_prefit is stored per-modality (modality.alpha_y_prefit), not on the model
        self.alpha_y_type = None    # from step1 (type is still tracked at model level for backward compat)
        self.trace_cellline = None    # from step1
        self.trace_x = None          # from step2
        self.trace_y = None          # from step3

        # Initialize fitter objects and helpers
        self._technical_fitter = TechnicalFitter(self)
        self._cis_fitter = CisFitter(self)
        self._trans_fitter = TransFitter(self)
        self._saver = ModelSaver(self)
        self._loader = ModelLoader(self)

        # Import here to avoid circular dependency
        from .io.summary import ModelSummarizer
        self._summarizer = ModelSummarizer(self)

        print(f"[INIT] bayesDREAM core: label={self.label}, device={self.device}")

    def cis_init_loc_fn(
        self,
        mu_init: torch.Tensor,
        sigma_init: torch.Tensor,
        beta_o_init: float,
        o_x_init: float,
        sigma_eff_init: float,
        sigma_eff_alpha_init: float,
        sigma_eff_beta_init: float,
        alpha_x_init: torch.Tensor = None,
    ):
        """
        Returns a function that uses custom data-driven initial values for certain sites:
            - 'mu' initialized to `mu_init`
            - 'sigma' initialized to `sigma_init`
            - 'alpha_x' (if present) initialized to `alpha_x_init`
            - 'beta_o' initialized to `beta_o_init`
            - 'o_x' initialized to `o_x_init`
            - 'sigma_eff' (the gamma shape or scale for per-guide sigmas) initialized to `sigma_eff_init`
            - 'sigma_eff_alpha' (the gamma shape or scale for per-guide sigmas) initialized to `sigma_eff_alpha_init`
            - 'sigma_eff_beta' (the gamma shape or scale for per-guide sigmas) initialized to `sigma_eff_beta_init`
        Fallback to `init_to_median` for all other sites.
        """

        def to_tensor(value, device):
            """Helper function to ensure value is a tensor."""
            if isinstance(value, torch.Tensor):
                return value.detach().clone().to(device)
            return torch.tensor(value, dtype=torch.float32, device=device)
    
        def _init_fun(site):
            name = site["name"]
            device = self.device
    
            if name == "mu":
                return to_tensor(mu_init, device)
    
            elif name == "sigma":
                return to_tensor(sigma_init, device)
    
            elif name == "alpha_x" and alpha_x_init is not None:
                return to_tensor(alpha_x_init, device)
    
            elif name == "beta_o":
                return to_tensor(beta_o_init, device)
    
            elif name == "o_x":
                return to_tensor(o_x_init, device)
    
            elif name == "sigma_eff_alpha":
                return to_tensor(sigma_eff_alpha_init, device)
    
            elif name == "sigma_eff_beta":
                return to_tensor(sigma_eff_beta_init, device)
    
            elif name == "sigma_eff":
                shape = site["fn"].sample().shape  # Get shape from distribution
                return to_tensor(sigma_eff_init, device).expand(shape)
    
            # Otherwise, fall back to a default
            return pyro.infer.autoguide.initialization.init_to_sample(site)
    
        return _init_fun

    def set_alpha_x(
        self,
        alpha_x: float, # expected to be C x 1 point estimate or S x C x 1 posterior
        is_posterior: bool,
        covariates: list[str] = None # Technical group covariates (e.g., ["cell_line"]). NOT empty. The point is to fit to the covariates. Lane is typically not included as this tends to be corrected by sum factor adjustment alone
    ):
        """
        Sets alpha_x either as a point estimate (float/tensor) or as a Pyro posterior.
        """
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif not "technical_group_code" in self.meta.columns:
            raise ValueError(f"No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming alpha_x corresponds.")
        self.alpha_x_prefit = sample_or_use_point("alpha_x_posterior", alpha_x, self.device)
        
        if is_posterior:
            if not (self.alpha_x_prefit.ndim == 2 or (self.alpha_x_prefit.ndim == 3 and self.alpha_x_prefit.shape[2] == 1)):
                raise ValueError(
                    f"when it is a posterior, alpha_x_prefit is expected to be of shape S x C or S x C x 1, "
                    f"but got {self.alpha_x_prefit.shape}."
                )
            self.alpha_x_type = 'posterior'
        else:
            if not (self.alpha_x_prefit.ndim == 0 or  # Scalar case
                    (self.alpha_x_prefit.ndim == 2 and self.alpha_x_prefit.shape[1] == 1)):  # C x 1 case
                raise ValueError(
                    f"when it is a point estimate, alpha_x_prefit is expected to be a scalar or of shape C x 1, "
                    f"but got {self.alpha_x_prefit.shape}."
                )
            self.alpha_x_type = 'point'

    def set_alpha_y(
        self,
        alpha_y,
        is_posterior: bool,
        covariates: list[str] = None # Technical group covariates (e.g., ["cell_line"]). NOT empty. The point is to fit to the covariates. Lane is typically not included as this tends to be corrected by sum factor adjustment alone
    ):
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif not "technical_group_code" in self.meta.columns:
            raise ValueError(f"No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming alpha_xy corresponds.")

        # Determine T (number of response variables)
        if self.cis_gene is not None:
            # Count trans genes (all genes except cis) - works for DataFrame and arrays
            if isinstance(self.counts, pd.DataFrame):
                T = self.counts.drop([self.cis_gene]).shape[0]
            else:
                # For matrices/arrays: use self.trans_genes (already computed as numeric indices)
                T = len(self.trans_genes)
        else:
            T = self.counts.shape[0]
        C = self.meta["technical_group_code"].nunique() - 1
    
        # Convert alpha_y to tensor
        alpha_y = sample_or_use_point("alpha_y_posterior", alpha_y, self.device)
    
        if is_posterior:
            if not (alpha_y.ndim == 3 and alpha_y.shape[1] == C and alpha_y.shape[2] == T):
                raise ValueError(
                    f"When it is a posterior, alpha_y is expected to have shape S x C-1 x T, but got {alpha_y.shape}."
                )
            self.alpha_y_type = 'posterior'
        else:
            if not (alpha_y.ndim == 2 and alpha_y.shape[0] == C and alpha_y.shape[1] == T):
                raise ValueError(
                    f"When it is a point estimate, alpha_y must have shape C-1 x T, but got {alpha_y.shape}."
                )
            self.alpha_y_type = 'point'
    
        # Store in primary modality (not model-level)
        primary_mod = self.get_modality(self.primary_modality)
        primary_mod.alpha_y_prefit = alpha_y
        primary_mod.alpha_y_type = self.alpha_y_type

    def set_o_x_grouped(
        self,
        o_x,  # Expected to be C x 1 point estimate or S x C x 1 posterior
        is_posterior: bool,
        covariates: list[str] = None,  # Technical group covariates (e.g., ["cell_line"]) NOT empty if using
    ):
        """
        Sets o_x either as a point estimate (float/tensor) or as a Pyro posterior.
        """
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif "technical_group_code" not in self.meta.columns:
            raise ValueError("No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming o_x corresponds.")
    
        self.o_x_prefit = sample_or_use_point("o_x_posterior", o_x, self.device)
    
        if is_posterior:
            if not (self.o_x_prefit.ndim == 2 or
                    (self.o_x_prefit.ndim == 3 and self.o_x_prefit.shape[2] == 1)):
                raise ValueError(
                    f"When it is a posterior, o_x_prefit must be S x C or S x C x 1, "
                    f"but got {self.o_x_prefit.shape}."
                )
            self.o_x_type = 'posterior'
        else:
            if not (self.o_x_prefit.ndim == 0 or
                    (self.o_x_prefit.ndim == 2 and self.o_x_prefit.shape[1] == 1)):  # C x 1 case
                raise ValueError(
                    f"When it is a point estimate, o_x_prefit must be a scalar or of shape C x 1, "
                    f"but got {self.o_x_prefit.shape}."
                )
            self.o_x_type = 'point'


    def set_o_x(
        self,
        o_x,
        is_posterior: bool
    ):
        """
        Sets o_x either as a point estimate (float/tensor) or as a posterior.
        
        - If posterior: o_x should have shape S, S x 1, or S x 1 x 1.
        - If point estimate: o_x should be a scalar or [scalar] (1-element tensor).
        """
        self.o_x = sample_or_use_point("o_x_posterior", o_x, self.device)
    
        if is_posterior:
            if not (
                (self.o_x.ndim == 1) or  # S
                (self.o_x.ndim == 2 and self.o_x.shape[1] == 1) or  # S x 1
                (self.o_x.ndim == 3 and self.o_x.shape[1] == 1 and self.o_x.shape[2] == 1)  # S x 1 x 1
            ):
                raise ValueError(
                    f"When it is a posterior, o_x must have shape S, S x 1, or S x 1 x 1, but got {self.o_x.shape}."
                )
            self.o_x_type = "posterior"
        else:
            if not (self.o_x.ndim == 0 or (self.o_x.ndim == 1 and self.o_x.numel() == 1)):
                raise ValueError(
                    f"When it is a point estimate, o_x must be a scalar or a single-element tensor, but got {self.o_x.shape}."
                )
            self.o_x_type = "point"

    def set_x_true(
        self,
        x_true,
        is_posterior: bool
    ):

        """
        Sets x_true either as a point estimate or as a posterior.
        
        - If posterior: x_true should have shape S x N or S x 1 x N.
        - If point estimate: x_true should have shape N.
        """
        # Determine N (number of observations)
        N = self.counts.shape[1]
    
        # Convert x_true to tensor
        x_true = sample_or_use_point("x_true_posterior", x_true, self.device)
    
        if is_posterior:
            if not ((x_true.ndim == 2 and x_true.shape[1] == N) or
                    (x_true.ndim == 3 and x_true.shape[1] == 1 and x_true.shape[2] == N)):
                raise ValueError(
                    f"When it is a posterior, x_true must have shape S x N or S x 1 x N, but got {x_true.shape}."
                )
            self.x_true_type = "posterior"
        else:
            if not (x_true.ndim == 1 and x_true.shape[0] == N):
                raise ValueError(
                    f"When it is a point estimate, x_true must have shape N ({N},), but got {x_true.shape}."
                )
            self.x_true_type = "point"
    
        self.x_true = x_true  # Store validated tensor


    def adjust_ntc_sum_factor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_adj: str = "sum_factor_adj",
        covariates: list[str] = None # Technical group covariates (e.g., ["lane", "cell_line"]) or could be empty
    ):
        """
        Step 1 of sum factor adjustment: Normalize guides to NTC controls.

        Use BEFORE fit_cis() to account for guide-level technical variation.
        Computes adjustment factor = mean_ntc_sum_factor / mean_guide_sum_factor
        within covariate groups (e.g., technical groups like cell_line, lane).

        Typical workflow:
            1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
            2. fit_cis(sum_factor_col='sum_factor_adj')
            3. refit_sumfactor() -> creates 'sum_factor_refit'
            4. fit_trans(sum_factor_col='sum_factor_refit')

        Parameters
        ----------
        sum_factor_col_old : str
            Name of existing sum factor column in meta (default: 'sum_factor')
        sum_factor_col_adj : str
            Name for adjusted sum factor column to create (default: 'sum_factor_adj')
        covariates : list of str, optional
            Technical group covariates to group by for adjustment (e.g., ['cell_line', 'lane']).
            If None or empty, uses global mean across all cells.

        Returns
        -------
        None
            Creates new column sum_factor_col_adj in self.meta

        Notes
        -----
        For each guide within each covariate group:
        - Compute mean NTC sum factor: mean_ntc
        - Compute mean guide sum factor: mean_guide
        - Adjustment factor = mean_ntc / mean_guide
        - Adjusted sum factor = sum_factor * adjustment_factor
        """
        meta_out = self.meta.copy()
        meta_out["original_index"] = np.arange(len(meta_out))

        if sum_factor_col_old not in meta_out.columns:
            raise ValueError(f"No column '{sum_factor_col_old}' found in meta. Provide a precomputed sum_factor.")

        # Drop existing adjustment_factor column if it exists (prevents merge conflicts)
        if "adjustment_factor" in meta_out.columns:
            meta_out = meta_out.drop(columns=["adjustment_factor"])

        # Make sure all covariates are actually in meta_out
        if covariates:
            missing_cols = [c for c in covariates if c not in meta_out.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing covariate columns: {missing_cols}. "
                    f"Available columns are: {list(meta_out.columns)}"
            )
        
        if not covariates:
            # (1) Mean sum_factor among NTC rows, grouped by covariates (e.g. lane, cell_line)
            mean_ntc_value = meta_out.loc[meta_out['target'] == 'ntc', sum_factor_col_old].mean()
    
            # (2) Mean sum_factor among *all* guides, grouped by covariates + [guide_col]
            df_guide = (
                meta_out.groupby(["guide_used"])[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_guide")
            )
    
            # (3) Merge them and compute ratio = mean_NTC / mean_guide
            df_guide["adjustment_factor"] = (
                mean_ntc_value / (df_guide["mean_SumFacs_guide"])
            )
    
            # (4) Merge that ratio back onto meta_out
            meta_out = pd.merge(
                meta_out,
                df_guide[["guide_used", "adjustment_factor"]],
                on="guide_used",
                how="left"
            )

        else:
            # (1) Mean sum_factor among NTC rows, grouped by covariates (e.g. lane, cell_line)
            df_ntc = (
                meta_out.loc[meta_out["target"] == "ntc"]
                .groupby(covariates)[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_ntc")
            )
    
            # (2) Mean sum_factor among *all* guides, grouped by covariates + [guide_col]
            df_guide = (
                meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_guide")
            )
    
            # (3) Merge them and compute ratio = mean_NTC / mean_guide
            merged = pd.merge(df_guide, df_ntc, on=covariates, how="left")
            merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
    
            # (4) Merge that ratio back onto meta_out
            merge_cols = covariates + ["guide_used", "adjustment_factor"]
            meta_out = pd.merge(meta_out, merged[merge_cols], on=covariates + ["guide_used"], how="left")
            
        # (5) Multiply original sum_factor by ratio
        meta_out[sum_factor_col_adj] = meta_out[sum_factor_col_old] * meta_out["adjustment_factor"]

        meta_out.sort_values("original_index", inplace=True)
        meta_out.drop(columns="original_index", inplace=True)
        self.meta = meta_out
        print(f"[INFO] Created '{sum_factor_col_adj}' in meta with NTC-based guide-level adjustment.")

    def refit_sumfactor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_refit: str = "sum_factor_new",
        covariates: list[str] = None, # Technical group covariates (e.g., ["lane", "cell_line"]) or could be empty
        n_knots: int = 5,
        degree: int = 3,
        alpha: float = 0.1
    ):
        """
        Step 2 of sum factor adjustment: Remove cis gene contribution.

        Use AFTER fit_cis() and BEFORE fit_trans().
        Fits a spline regression: (sum_factor - baseline_ntc) ~ f(x_true)
        Then removes the predicted x_true contribution from sum factors.

        This ensures trans modeling isn't confounded by cis expression levels.

        Typical workflow:
            1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
            2. fit_cis(sum_factor_col='sum_factor_adj')
            3. refit_sumfactor() -> creates 'sum_factor_refit'  <-- This step
            4. fit_trans(sum_factor_col='sum_factor_refit')

        Parameters
        ----------
        sum_factor_col_old : str
            Name of existing sum factor column (typically from adjust_ntc_sum_factor)
        sum_factor_col_refit : str
            Name for refitted sum factor column to create (default: 'sum_factor_new')
        covariates : list of str, optional
            Technical group covariates to group by for baseline NTC calculation (e.g., ['cell_line', 'lane'])
        n_knots : int
            Number of spline knots for regression (default: 5)
        degree : int
            Polynomial degree of spline pieces (default: 3)
        alpha : float
            Ridge regression regularization parameter (default: 0.1)

        Returns
        -------
        None
            Creates new column sum_factor_col_refit in self.meta

        Notes
        -----
        Algorithm:
        1. Compute baseline NTC sum factor for each covariate group
        2. Compute leftover = sum_factor - baseline_ntc
        3. Fit spline model: leftover ~ f(x_true)
        4. Predict x_true contribution: y_pred = model(x_true)
        5. Adjusted sum factor = max(0, leftover - y_pred + baseline_ntc)

        Requires:
        - self.x_true must be set (from fit_cis())
        - self.x_true_type must be 'posterior' or 'point'
        """
        sum_factor_data = self.meta[sum_factor_col_old].values  # shape (N,)
        
        if covariates is None:
            covariates = []

        if covariates:
            # Create a single group identifier by concatenating covariate values
            tech_group = self.meta[covariates].astype(str).agg('_'.join, axis=1)
            groups, group_id = np.unique(tech_group, return_inverse=True)
            n_groups = len(groups)
        else:
            # No grouping, treat all samples as a single group
            groups, group_id = np.array(["all"]), np.zeros(len(self.meta), dtype=int)
            n_groups = 1
        
        baseline_ntc_of_group = np.zeros(n_groups)
        if covariates:
            for grp, grp_name in enumerate(groups):
                mask_grp = (tech_group == grp_name)
                # Among that group, pick rows with gene == 'ntc'
                mask_ntc = (self.meta['target'] == 'ntc') & mask_grp
            
                # If a group has no NTC, you must decide on a fallback
                # For example, use the overall mean or 1.0, or skip that group
                if not np.any(mask_ntc):
                    baseline_ntc_of_group[grp] = 1.0  # fallback
                else:
                    # The mean of sum_factor_data among the NTC rows
                    baseline_ntc_of_group[grp] = np.mean(sum_factor_data[mask_ntc])
        else:
            grp = 0
            mask_ntc = (self.meta['target'] == 'ntc')
            # If a group has no NTC, you must decide on a fallback
            # For example, use the overall mean or 1.0, or skip that group
            if not np.any(mask_ntc):
                baseline_ntc_of_group[grp] = 1.0  # fallback
            else:
                # The mean of sum_factor_data among the NTC rows
                baseline_ntc_of_group[grp] = np.mean(sum_factor_data[mask_ntc])
        
        # leftover_data[i] = sum_factor[i] - baseline_ntc_of_group[group_id[i]]
        leftover_data = sum_factor_data - baseline_ntc_of_group[group_id]
    
        # ------------------------------------------------------------------------------
        # Build a pipeline with a Spline transformer + Linear regression
        # ------------------------------------------------------------------------------
        # 'n_knots' is how many spline knots to use;
        # 'degree' is the polynomial degree of each spline piece.
        model_spline_ridge = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree),
            Ridge(alpha=alpha)  # alpha > 0 adds penalty
        )
        
        # ------------------------------------------------------------------------------
        # Fit the model on training data
        # ------------------------------------------------------------------------------
        if self.x_true_type == 'posterior':
            X_true = self.x_true.mean(dim=0)
        else:
            X_true = self.x_true
        model_spline_ridge.fit(X_true.reshape(-1, 1), leftover_data)
        
        # ------------------------------------------------------------------------------
        # Predict on train & test
        # ------------------------------------------------------------------------------
        y_pred = model_spline_ridge.predict(X_true.reshape(-1, 1))
        self.meta[sum_factor_col_refit] = np.max(np.vstack((leftover_data - y_pred + baseline_ntc_of_group[group_id], np.zeros_like(leftover_data))), axis=0)
        print(f"[INFO] Created '{sum_factor_col_refit}' in meta with xtrue-based adjustment.")

    def permute_genes(
        self,
        genes2permute: list[str] = None,
        covariates: list[str] = ["cell_line", "lane"],
        sum_factor_col: str = 'sum_factor_adj',
        permute_ntc_x: bool = True
    ):
        """
        Permute specified genes within technical covariates while ensuring consistency with NTC cells.
        After permutation, the sum factors should be adjusted.

        Parameters
        ----------
        genes2permute : list of str
            List of gene names to permute. If 'All', all genes except the cis gene are permuted.
        covariates : list of str
            Technical group covariates used to group cells for permutation (e.g., ['cell_line', 'lane']).
        """

        if sum_factor_col not in self.meta.columns:
            raise ValueError(f"No column '{sum_factor_col}' found in meta. Provide a precomputed sum_factor.")
            
        print("Running gene permutation...")

        # If genes2permute is 'All', include all genes except the cis_gene
        if genes2permute == 'All' or genes2permute == ['All']:
            if isinstance(self.counts, pd.DataFrame):
                genes2permute = list(self.counts.index.values[self.counts.index.values != self.cis_gene])
            else:
                # For matrices/arrays: use self.trans_genes (already computed as numeric indices)
                genes2permute = self.trans_genes.copy()

        if isinstance(genes2permute, str):
            genes2permute = [genes2permute]

        meta_sub = self.meta.copy()

        # Handle counts copying based on type
        if isinstance(self.counts, pd.DataFrame):
            counts_sub = self.counts.copy()
        elif self.is_sparse_counts:
            counts_sub = self.counts.copy()
        else:
            counts_sub = self.counts.copy()

        # Create cell name to column index mapping for matrices/arrays
        if isinstance(self.counts, pd.DataFrame):
            cell_to_col_idx = {cell: idx for idx, cell in enumerate(counts_sub.columns)}
        else:
            cell_to_col_idx = {cell: idx for idx, cell in enumerate(self._cell_names)}

        for gene in genes2permute:
            # Check if gene exists - works for both DataFrame and arrays
            if isinstance(self.counts, pd.DataFrame):
                if gene not in counts_sub.index:
                    continue
                gene_idx = gene  # Use gene name directly for DataFrame
            else:
                # For matrices/arrays: gene is a numeric index
                if gene >= counts_sub.shape[0]:
                    continue
                gene_idx = gene  # Use numeric index directly

            for cov_values, group in meta_sub.groupby(covariates):
                mycells = group.loc[group["target"] != "ntc", "cell"]
                my_ntc_cells = group.loc[group["target"] == "ntc", "cell"]

                if len(mycells) > 0 and len(my_ntc_cells) > 0:
                    # Get column indices for cells
                    mycell_indices = [cell_to_col_idx[cell] for cell in mycells]
                    my_ntc_indices = [cell_to_col_idx[cell] for cell in my_ntc_cells]

                    # Get count data - works for DataFrame, dense, and sparse
                    if isinstance(self.counts, pd.DataFrame):
                        gene_counts_ntc = counts_sub.loc[gene_idx, my_ntc_cells].values
                    elif self.is_sparse_counts:
                        # Sparse: extract row, convert to dense array
                        gene_counts_ntc = np.array(counts_sub[gene_idx, my_ntc_indices].todense()).flatten()
                    else:
                        # Dense array
                        gene_counts_ntc = counts_sub[gene_idx, my_ntc_indices]

                    # Get sum factors
                    meta_indexed = meta_sub.set_index("cell")
                    ntc_sum_factors = meta_indexed.loc[my_ntc_cells, sum_factor_col].values
                    mycell_sum_factors = meta_indexed.loc[mycells, sum_factor_col].values

                    # Sample values from NTC distribution
                    sampled_values = np.random.choice(
                        gene_counts_ntc / ntc_sum_factors,
                        size=len(mycells),
                        replace=True
                    ) * mycell_sum_factors

                    # Assign back to counts - works for DataFrame, dense, and sparse
                    if isinstance(self.counts, pd.DataFrame):
                        counts_sub.loc[gene_idx, mycells] = np.round(sampled_values)
                    elif self.is_sparse_counts:
                        # For sparse: convert to LIL format for efficient assignment
                        counts_sub = counts_sub.tolil()
                        for i, cell_idx in enumerate(mycell_indices):
                            counts_sub[gene_idx, cell_idx] = np.round(sampled_values[i])
                        counts_sub = counts_sub.tocsr()
                    else:
                        # Dense array
                        counts_sub[gene_idx, mycell_indices] = np.round(sampled_values)

        if permute_ntc_x:
            # Find cis gene index
            if isinstance(self.counts, pd.DataFrame):
                cis_gene_idx = self.cis_gene
            else:
                # For matrices/arrays: find cis gene row
                cis_gene_idx = None
                for col in ['gene_name', 'gene', 'gene_id', 'feature_id']:
                    if col in self.gene_meta.columns:
                        matches = self.gene_meta[self.gene_meta[col] == self.cis_gene]
                        if len(matches) > 0:
                            cis_gene_idx = matches.index[0]
                            break
                if cis_gene_idx is None:
                    raise ValueError(f"Cannot find cis gene '{self.cis_gene}' for permutation")

            for cov_values, group in meta_sub.groupby(covariates):
                my_ntc_cells = group.loc[group["target"] == "ntc", "cell"]

                if len(my_ntc_cells) > 0:
                    # Get column indices for NTC cells
                    my_ntc_indices = [cell_to_col_idx[cell] for cell in my_ntc_cells]

                    # Get sum factors
                    meta_indexed = meta_sub.set_index("cell")
                    ntc_sum_factor = meta_indexed.loc[my_ntc_cells, sum_factor_col].values

                    # Get cis gene expression for NTC cells
                    if isinstance(self.counts, pd.DataFrame):
                        ntc_expr = counts_sub.loc[cis_gene_idx, my_ntc_cells].values
                    elif self.is_sparse_counts:
                        ntc_expr = np.array(counts_sub[cis_gene_idx, my_ntc_indices].todense()).flatten()
                    else:
                        ntc_expr = counts_sub[cis_gene_idx, my_ntc_indices]

                    ntc_expr_norm = ntc_expr / ntc_sum_factor

                    # Sample indices with replacement
                    permuted_indices = np.random.choice(len(my_ntc_cells), size=len(my_ntc_cells), replace=True)

                    # Permute counts
                    new_counts = ntc_expr_norm[permuted_indices] * ntc_sum_factor

                    # Assign back
                    if isinstance(self.counts, pd.DataFrame):
                        counts_sub.loc[cis_gene_idx, my_ntc_cells] = np.round(new_counts)
                    elif self.is_sparse_counts:
                        counts_sub = counts_sub.tolil()
                        for i, cell_idx in enumerate(my_ntc_indices):
                            counts_sub[cis_gene_idx, cell_idx] = np.round(new_counts[i])
                        counts_sub = counts_sub.tocsr()
                    else:
                        counts_sub[cis_gene_idx, my_ntc_indices] = np.round(new_counts)
        
                    # Permute x_true in the same order
                    ntc_idx = meta_sub.index[meta_sub["cell"].isin(my_ntc_cells)].tolist()
                    assert len(ntc_idx) == len(my_ntc_cells)  # Sanity check
        
                    # Apply permutation to x_true
                    if isinstance(self.x_true, torch.Tensor):
                        x_true_np = self.x_true.detach().cpu().numpy()
                    else:
                        x_true_np = np.array(self.x_true)
        
                    x_true_ntc = x_true_np[ntc_idx]
                    x_true_np[ntc_idx] = x_true_ntc[permuted_indices]
        
                    # Assign back
                    self.x_true = torch.tensor(x_true_np, dtype=self.x_true.dtype, device=self.x_true.device)

        self.counts = counts_sub

    def subset_cells(
        self,
        cell_mask: Union[np.ndarray, pd.Series, list] = None,
        query: str = None,
        preserve_fits: bool = True
    ):
        """
        Create a new model instance with a subset of cells.

        Useful for testing without technical correction by subsetting to a single
        cell_line (e.g., CRISPRi or CRISPRa only).

        Parameters
        ----------
        cell_mask : np.ndarray, pd.Series, or list, optional
            Boolean mask or list of cell names to keep. If None, must provide query.
        query : str, optional
            Pandas query string to filter cells (e.g., "cell_line == 'CRISPRa'").
            Applied to self.meta. If None, must provide cell_mask.
        preserve_fits : bool
            If True (default), copy fitted parameters (alpha_x_prefit, alpha_y_prefit,
            x_true, etc.) to the new model. Set to False to start fresh with the subset.

        Returns
        -------
        bayesDREAM
            New model instance with subsetted cells

        Examples
        --------
        # Subset by query
        model_crispra = model.subset_cells(query="cell_line == 'CRISPRa'")

        # Subset by mask
        mask = model.meta['cell_line'].str.contains('CRISPRa')
        model_crispra = model.subset_cells(cell_mask=mask)

        # Subset by cell list
        cells = model.meta[model.meta['cell_line'] == 'CRISPRa']['cell'].tolist()
        model_crispra = model.subset_cells(cell_mask=cells)
        """
        if cell_mask is None and query is None:
            raise ValueError("Must provide either cell_mask or query")

        if query is not None:
            # Filter meta using query
            meta_subset = self.meta.query(query).copy()
            cells_to_keep = meta_subset['cell'].values
        elif isinstance(cell_mask, (list, np.ndarray, pd.Series)):
            # Handle list of cell names
            if isinstance(cell_mask, list) and len(cell_mask) > 0 and isinstance(cell_mask[0], str):
                cells_to_keep = cell_mask
                meta_subset = self.meta[self.meta['cell'].isin(cells_to_keep)].copy()
            # Handle boolean mask
            elif isinstance(cell_mask, (np.ndarray, pd.Series)):
                if cell_mask.dtype == bool:
                    meta_subset = self.meta[cell_mask].copy()
                    cells_to_keep = meta_subset['cell'].values
                else:
                    # Assume it's a list of cell names
                    cells_to_keep = cell_mask
                    meta_subset = self.meta[self.meta['cell'].isin(cells_to_keep)].copy()
            else:
                raise ValueError("cell_mask must be boolean array/Series or list of cell names")
        else:
            raise ValueError("Invalid cell_mask type")

        if len(meta_subset) == 0:
            raise ValueError("Subset resulted in zero cells")

        print(f"[INFO] Subsetting from {len(self.meta)} to {len(meta_subset)} cells")

        # Subset counts
        if isinstance(self.counts, pd.DataFrame):
            counts_subset = self.counts[cells_to_keep].copy()
        else:
            # Sparse or dense array - subset by column indices
            cell_indices = [i for i, cell in enumerate(self._cell_names) if cell in cells_to_keep]
            if self.is_sparse_counts:
                counts_subset = self.counts[:, cell_indices]
            else:
                counts_subset = self.counts[:, cell_indices]

        # Subset high MOI guide_assignment if applicable
        guide_assignment_subset = None
        if self.is_high_moi:
            # Get indices of subsetted cells in original meta
            cell_indices = [i for i, cell in enumerate(self.meta['cell']) if cell in cells_to_keep]
            guide_assignment_subset = self.guide_assignment[cell_indices, :]

        # Create new model instance
        from .model import bayesDREAM

        model_new = bayesDREAM(
            meta=meta_subset,
            counts=counts_subset,
            modality_name=self.primary_modality,  # Use same primary modality name
            feature_meta=self.gene_meta.copy() if hasattr(self, 'gene_meta') else None,
            cis_gene=self.cis_gene,
            output_dir=self.output_dir,
            label=f"{self.label}_subset",
            device=str(self.device),
            random_seed=2402,
            cores=1,
            guide_assignment=guide_assignment_subset,
            guide_meta=self.guide_meta.copy() if self.is_high_moi else None,
            guide_target=None,  # Already encoded in guide_meta
            require_ntc=False  # Allow subsetting without NTC cells
        )

        print(f"[DEBUG] Original modalities: {list(self.modalities.keys())}")
        print(f"[DEBUG] New model modalities: {list(model_new.modalities.keys())}")

        # Copy additional modalities (beyond the primary 'gene' modality)
        if hasattr(self, 'modalities'):
            for mod_name, modality in self.modalities.items():
                if mod_name not in ['gene', 'cis']:  # Skip primary and cis modalities (already handled)
                    # Subset the modality to the selected cells using cell names
                    subset_modality = modality.get_cell_subset(cells_to_keep)
                    model_new.modalities[mod_name] = subset_modality
            print(f"[INFO] Copied {len(self.modalities) - 2} additional modalities to subset model")

        # Optionally preserve fitted parameters
        if preserve_fits:
            # Copy technical fit parameters if they exist
            if self.alpha_x_prefit is not None:
                model_new.alpha_x_prefit = self.alpha_x_prefit.clone() if isinstance(self.alpha_x_prefit, torch.Tensor) else self.alpha_x_prefit
                model_new.alpha_x_type = self.alpha_x_type

            # Copy all fit attributes from original modalities to new modalities
            # (including 'gene' and 'cis' which were recreated during initialization)
            if self.alpha_y_type is not None:
                model_new.alpha_y_type = self.alpha_y_type

            # Helper to clone tensors
            def _clone_attr(val):
                if hasattr(val, 'clone'):
                    return val.clone()
                return val

            # Modality-level attributes to copy from fit_technical and fit_trans
            modality_attrs = [
                'alpha_y_prefit', 'alpha_y_type', 'alpha_y_prefit_mult', 'alpha_y_prefit_add',
                'posterior_samples_technical', 'posterior_samples_trans', 'losses_trans'
            ]
            for mod_name in self.modalities:
                if mod_name in model_new.modalities:
                    orig_mod = self.modalities[mod_name]
                    new_mod = model_new.modalities[mod_name]
                    copied_attrs = []
                    for attr in modality_attrs:
                        orig_val = getattr(orig_mod, attr, None)
                        if orig_val is not None:
                            setattr(new_mod, attr, _clone_attr(orig_val))
                            copied_attrs.append(attr)
                    if copied_attrs:
                        print(f"[INFO] Copied {copied_attrs} from '{mod_name}' modality")
                    else:
                        print(f"[DEBUG] No attributes to copy from '{mod_name}' modality")
                        # Debug: show which attrs exist
                        for attr in modality_attrs:
                            val = getattr(orig_mod, attr, "MISSING")
                            print(f"  {attr}: {type(val).__name__ if val != 'MISSING' else 'MISSING'}")
                else:
                    print(f"[WARN] Modality '{mod_name}' not found in new model")

            # Copy cis fit parameters if they exist
            # Get cell indices for subsetting cell-indexed tensors
            cell_indices_torch = torch.tensor(
                [i for i, cell in enumerate(self.meta['cell']) if cell in cells_to_keep],
                dtype=torch.long
            )

            if hasattr(self, 'x_true') and self.x_true is not None:
                # x_true needs to be subsetted to match new cells
                if isinstance(self.x_true, torch.Tensor):
                    if self.x_true_type == 'posterior':
                        # x_true shape: [S, N] or [S, 1, N]
                        if self.x_true.ndim == 2:
                            model_new.x_true = self.x_true[:, cell_indices_torch].clone()
                        else:
                            model_new.x_true = self.x_true[:, :, cell_indices_torch].clone()
                    else:
                        # Point estimate: shape [N]
                        model_new.x_true = self.x_true[cell_indices_torch].clone()
                    model_new.x_true_type = self.x_true_type

            # Copy log2_x_true (also needs subsetting)
            if hasattr(self, 'log2_x_true') and self.log2_x_true is not None:
                if isinstance(self.log2_x_true, torch.Tensor):
                    if hasattr(self, 'log2_x_true_type') and self.log2_x_true_type == 'posterior':
                        if self.log2_x_true.ndim == 2:
                            model_new.log2_x_true = self.log2_x_true[:, cell_indices_torch].clone()
                        else:
                            model_new.log2_x_true = self.log2_x_true[:, :, cell_indices_torch].clone()
                    else:
                        model_new.log2_x_true = self.log2_x_true[cell_indices_torch].clone()
                    if hasattr(self, 'log2_x_true_type'):
                        model_new.log2_x_true_type = self.log2_x_true_type

            # Copy posterior_samples_cis with cell-indexed tensors subsetted
            if hasattr(self, 'posterior_samples_cis') and self.posterior_samples_cis is not None:
                n_cells_orig = len(self.meta)
                subsetted_cis = {}
                for key, val in self.posterior_samples_cis.items():
                    if isinstance(val, torch.Tensor) and val.shape[-1] == n_cells_orig:
                        # This tensor has cell dimension - subset it
                        subsetted_cis[key] = val[..., cell_indices_torch].clone()
                    elif isinstance(val, torch.Tensor):
                        subsetted_cis[key] = val.clone()
                    else:
                        subsetted_cis[key] = val
                model_new.posterior_samples_cis = subsetted_cis
            if hasattr(self, 'loss_x') and self.loss_x is not None:
                model_new.loss_x = self.loss_x
            if hasattr(self, 'posterior_samples_trans') and self.posterior_samples_trans is not None:
                model_new.posterior_samples_trans = self.posterior_samples_trans
            if hasattr(self, 'losses_trans') and self.losses_trans is not None:
                model_new.losses_trans = self.losses_trans

            # Copy traces if they exist
            if self.trace_cellline is not None:
                model_new.trace_cellline = self.trace_cellline
            if self.trace_x is not None:
                model_new.trace_x = self.trace_x
            if self.trace_y is not None:
                model_new.trace_y = self.trace_y

            print("[INFO] Preserved fitted parameters in subset model")

        return model_new


    # ========================================================================
    # Delegation methods to fitters
    # ========================================================================

    def _model_technical(self, *args, **kwargs):
        """Delegate to TechnicalFitter."""
        return self._technical_fitter._model_technical(*args, **kwargs)

    def set_technical_groups(self, *args, **kwargs):
        """Delegate to TechnicalFitter."""
        return self._technical_fitter.set_technical_groups(*args, **kwargs)

    def fit_technical(self, *args, **kwargs):
        """Delegate to TechnicalFitter."""
        return self._technical_fitter.fit_technical(*args, **kwargs)

    def _model_x(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter._model_x(*args, **kwargs)

    def cis_init_loc_fn(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter.cis_init_loc_fn(*args, **kwargs)

    def fit_cis(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter.fit_cis(*args, **kwargs)

    def _model_y(self, *args, **kwargs):
        """Delegate to TransFitter."""
        return self._trans_fitter._model_y(*args, **kwargs)

    def fit_trans(self, *args, **kwargs):
        """Delegate to TransFitter."""
        return self._trans_fitter.fit_trans(*args, **kwargs)

    def save_technical_fit(self, *args, **kwargs):
        """Delegate to ModelSaver."""
        return self._saver.save_technical_fit(*args, **kwargs)

    def save_cis_fit(self, *args, **kwargs):
        """Delegate to ModelSaver."""
        return self._saver.save_cis_fit(*args, **kwargs)

    def save_trans_fit(self, *args, **kwargs):
        """Delegate to ModelSaver."""
        return self._saver.save_trans_fit(*args, **kwargs)

    def load_technical_fit(self, *args, **kwargs):
        """Delegate to ModelLoader."""
        return self._loader.load_technical_fit(*args, **kwargs)

    def load_cis_fit(self, *args, **kwargs):
        """Delegate to ModelLoader."""
        return self._loader.load_cis_fit(*args, **kwargs)

    def load_trans_fit(self, *args, **kwargs):
        """Delegate to ModelLoader."""
        return self._loader.load_trans_fit(*args, **kwargs)

    # Summary export methods
    def save_technical_summary(self, *args, **kwargs):
        """Delegate to ModelSummarizer."""
        return self._summarizer.save_technical_summary(*args, **kwargs)

    def save_cis_summary(self, *args, **kwargs):
        """Delegate to ModelSummarizer."""
        return self._summarizer.save_cis_summary(*args, **kwargs)

    def save_trans_summary(self, *args, **kwargs):
        """Delegate to ModelSummarizer."""
        return self._summarizer.save_trans_summary(*args, **kwargs)

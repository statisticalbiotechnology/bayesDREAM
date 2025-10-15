"""
Modality data structure for multi-modal bayesDREAM.

This module defines the Modality class for storing and validating
different types of molecular measurements (genes, transcripts, splicing, etc.)
"""

import numpy as np
import pandas as pd
import torch
from typing import Literal, Optional, Union, Dict, Any


class Modality:
    """
    Stores data and metadata for a single molecular modality.

    Attributes
    ----------
    name : str
        Name of the modality (e.g., 'gene', 'transcript', 'splicing')
    counts : np.ndarray or pd.DataFrame
        The count/measurement data. Shape depends on distribution:
        - negbinom/normal: (features, cells) or (cells, features)
        - multinomial: (features, cells, categories)
        - binomial: (features, cells) with separate denominator
        - mvnormal: (features, cells, dimensions)
    feature_meta : pd.DataFrame
        Feature-level metadata (genes, junctions, etc.)
    distribution : str
        Distribution family: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
    denominator : np.ndarray, optional
        For binomial distribution: (features, cells) array of trial counts
    """

    VALID_DISTRIBUTIONS = {'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'}

    def __init__(
        self,
        name: str,
        counts: Union[np.ndarray, pd.DataFrame],
        feature_meta: pd.DataFrame,
        distribution: Literal['negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'],
        denominator: Optional[np.ndarray] = None,
        cells_axis: int = 1,  # 0 if cells are rows, 1 if cells are columns
        # Exon skipping specific parameters
        inc1: Optional[np.ndarray] = None,
        inc2: Optional[np.ndarray] = None,
        skip: Optional[np.ndarray] = None,
        exon_aggregate_method: Optional[str] = None,
    ):
        """
        Initialize a Modality.

        Parameters
        ----------
        name : str
            Modality name
        counts : np.ndarray or pd.DataFrame
            Count/measurement data
        feature_meta : pd.DataFrame
            Feature metadata with index matching features in counts
        distribution : str
            Distribution type
        denominator : np.ndarray, optional
            For binomial: denominator counts (e.g., total gene expression for SJ usage)
        cells_axis : int
            Which axis represents cells (0 or 1 for 2D data)
        inc1 : np.ndarray, optional
            For exon skipping: inclusion counts from first junction (d1->a2)
        inc2 : np.ndarray, optional
            For exon skipping: inclusion counts from second junction (d2->a3)
        skip : np.ndarray, optional
            For exon skipping: skipping counts (d1->a3)
        exon_aggregate_method : str, optional
            For exon skipping: how inc1 and inc2 were aggregated ('min' or 'mean')
        """
        if distribution not in self.VALID_DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {self.VALID_DISTRIBUTIONS}, got {distribution}")

        self.name = name
        self.distribution = distribution
        self.cells_axis = cells_axis

        # Convert DataFrame to array if needed
        if isinstance(counts, pd.DataFrame):
            self.counts = counts.values
            self.count_df = counts
            # Feature names from index or columns depending on orientation
            if cells_axis == 1:
                self.feature_names = counts.index.tolist()
                self.cell_names = counts.columns.tolist()
            else:
                self.feature_names = counts.columns.tolist()
                self.cell_names = counts.index.tolist()
        else:
            self.counts = np.asarray(counts)
            self.count_df = None
            self.feature_names = None
            self.cell_names = None

        self.feature_meta = feature_meta.copy()
        self.denominator = np.asarray(denominator) if denominator is not None else None

        # Exon skipping specific storage
        self.inc1 = np.asarray(inc1) if inc1 is not None else None
        self.inc2 = np.asarray(inc2) if inc2 is not None else None
        self.skip = np.asarray(skip) if skip is not None else None
        self.exon_aggregate_method = exon_aggregate_method
        self._technical_fit_aggregate_method = None  # Track method used during technical fit

        # Validate shapes
        self._validate()

        # Store dimensionality info
        self.dims = self._compute_dims()

    def _validate(self):
        """Validate data shapes and requirements."""
        if self.distribution in ['negbinom', 'normal', 'binomial']:
            if self.counts.ndim != 2:
                raise ValueError(f"{self.distribution} modality requires 2D counts, got shape {self.counts.shape}")
            n_features = self.counts.shape[1 - self.cells_axis]

        elif self.distribution == 'multinomial':
            if self.counts.ndim != 3:
                raise ValueError(f"multinomial modality requires 3D counts (features, cells, categories), got shape {self.counts.shape}")
            n_features = self.counts.shape[0]

        elif self.distribution == 'mvnormal':
            if self.counts.ndim != 3:
                raise ValueError(f"mvnormal modality requires 3D counts (features, cells, dimensions), got shape {self.counts.shape}")
            n_features = self.counts.shape[0]

        # Validate feature_meta matches
        if len(self.feature_meta) != n_features:
            raise ValueError(f"feature_meta has {len(self.feature_meta)} rows but counts has {n_features} features")

        # Validate denominator for binomial
        if self.distribution == 'binomial':
            if self.denominator is None:
                raise ValueError("binomial distribution requires denominator")
            if self.denominator.shape != self.counts.shape:
                raise ValueError(f"denominator shape {self.denominator.shape} must match counts shape {self.counts.shape}")

        # Validate exon skipping data if provided
        if self.inc1 is not None or self.inc2 is not None or self.skip is not None:
            if self.distribution != 'binomial':
                raise ValueError("inc1/inc2/skip are only valid for binomial distribution (exon skipping)")
            if self.inc1 is None or self.inc2 is None or self.skip is None:
                raise ValueError("For exon skipping, must provide all three: inc1, inc2, skip")
            if self.inc1.shape != self.counts.shape or self.inc2.shape != self.counts.shape or self.skip.shape != self.counts.shape:
                raise ValueError(f"inc1, inc2, skip must all have same shape as counts: {self.counts.shape}")
            if self.exon_aggregate_method not in ['min', 'mean']:
                raise ValueError(f"exon_aggregate_method must be 'min' or 'mean', got: {self.exon_aggregate_method}")

    def _compute_dims(self) -> Dict[str, int]:
        """Compute dimensionality information."""
        dims = {}

        if self.distribution in ['negbinom', 'normal', 'binomial']:
            dims['n_features'] = self.counts.shape[1 - self.cells_axis]
            dims['n_cells'] = self.counts.shape[self.cells_axis]

        elif self.distribution == 'multinomial':
            dims['n_features'] = self.counts.shape[0]
            dims['n_cells'] = self.counts.shape[1]
            dims['n_categories'] = self.counts.shape[2]

        elif self.distribution == 'mvnormal':
            dims['n_features'] = self.counts.shape[0]
            dims['n_cells'] = self.counts.shape[1]
            dims['n_dimensions'] = self.counts.shape[2]

        return dims

    def to_tensor(self, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Convert counts to torch tensor."""
        return torch.tensor(self.counts, dtype=torch.float32, device=device)

    def get_feature_subset(self, feature_indices: Union[list, np.ndarray]) -> 'Modality':
        """
        Return a new Modality with subset of features.

        Parameters
        ----------
        feature_indices : list or array
            Indices or names of features to keep

        Returns
        -------
        Modality
            New modality with subset of features
        """
        if isinstance(feature_indices, (list, np.ndarray)) and len(feature_indices) > 0:
            if isinstance(feature_indices[0], str):
                # Convert names to indices
                if self.feature_names is None:
                    raise ValueError("Cannot subset by name: feature_names not available")
                feature_indices = [self.feature_names.index(n) for n in feature_indices]

        # Subset counts
        if self.distribution in ['negbinom', 'normal', 'binomial']:
            if self.cells_axis == 1:
                new_counts = self.counts[feature_indices, :]
                new_denom = self.denominator[feature_indices, :] if self.denominator is not None else None
                new_inc1 = self.inc1[feature_indices, :] if self.inc1 is not None else None
                new_inc2 = self.inc2[feature_indices, :] if self.inc2 is not None else None
                new_skip = self.skip[feature_indices, :] if self.skip is not None else None
            else:
                new_counts = self.counts[:, feature_indices]
                new_denom = self.denominator[:, feature_indices] if self.denominator is not None else None
                new_inc1 = self.inc1[:, feature_indices] if self.inc1 is not None else None
                new_inc2 = self.inc2[:, feature_indices] if self.inc2 is not None else None
                new_skip = self.skip[:, feature_indices] if self.skip is not None else None
        else:
            # multinomial or mvnormal: features on axis 0
            new_counts = self.counts[feature_indices, :, :]
            new_denom = None
            new_inc1 = None
            new_inc2 = None
            new_skip = None

        # Subset metadata
        new_feature_meta = self.feature_meta.iloc[feature_indices].copy()

        return Modality(
            name=self.name,
            counts=new_counts,
            feature_meta=new_feature_meta,
            distribution=self.distribution,
            denominator=new_denom,
            cells_axis=self.cells_axis,
            inc1=new_inc1,
            inc2=new_inc2,
            skip=new_skip,
            exon_aggregate_method=self.exon_aggregate_method
        )

    def get_cell_subset(self, cell_indices: Union[list, np.ndarray]) -> 'Modality':
        """
        Return a new Modality with subset of cells.

        Parameters
        ----------
        cell_indices : list or array
            Indices or names of cells to keep

        Returns
        -------
        Modality
            New modality with subset of cells
        """
        if isinstance(cell_indices, (list, np.ndarray)) and len(cell_indices) > 0:
            if isinstance(cell_indices[0], str):
                # Convert names to indices
                if self.cell_names is None:
                    raise ValueError("Cannot subset by name: cell_names not available")
                cell_indices = [self.cell_names.index(n) for n in cell_indices]

        # Subset counts
        if self.distribution in ['negbinom', 'normal', 'binomial']:
            if self.cells_axis == 1:
                new_counts = self.counts[:, cell_indices]
                new_denom = self.denominator[:, cell_indices] if self.denominator is not None else None
                new_inc1 = self.inc1[:, cell_indices] if self.inc1 is not None else None
                new_inc2 = self.inc2[:, cell_indices] if self.inc2 is not None else None
                new_skip = self.skip[:, cell_indices] if self.skip is not None else None
            else:
                new_counts = self.counts[cell_indices, :]
                new_denom = self.denominator[cell_indices, :] if self.denominator is not None else None
                new_inc1 = self.inc1[cell_indices, :] if self.inc1 is not None else None
                new_inc2 = self.inc2[cell_indices, :] if self.inc2 is not None else None
                new_skip = self.skip[cell_indices, :] if self.skip is not None else None
        else:
            # multinomial or mvnormal: cells on axis 1
            new_counts = self.counts[:, cell_indices, :]
            new_denom = None
            new_inc1 = None
            new_inc2 = None
            new_skip = None

        return Modality(
            name=self.name,
            counts=new_counts,
            feature_meta=self.feature_meta.copy(),
            distribution=self.distribution,
            denominator=new_denom,
            cells_axis=self.cells_axis,
            inc1=new_inc1,
            inc2=new_inc2,
            skip=new_skip,
            exon_aggregate_method=self.exon_aggregate_method
        )

    def set_exon_aggregate_method(self, method: str, allow_after_technical_fit: bool = False):
        """
        Change the exon skipping aggregation method and recompute inclusion counts.

        This recomputes self.counts (inclusion) and self.denominator (total) based on
        the new aggregation method.

        Parameters
        ----------
        method : str
            Aggregation method: 'min' or 'mean'
        allow_after_technical_fit : bool
            If True, allow changing method even after technical fit. Default: False.
            WARNING: Changing aggregation after technical fit invalidates the prefit
            overdispersion parameters.

        Raises
        ------
        ValueError
            If method is invalid, modality is not exon skipping, or changing after
            technical fit without explicit permission.
        """
        if self.inc1 is None:
            raise ValueError("This modality does not have exon skipping data (inc1/inc2/skip)")

        if method not in ['min', 'mean']:
            raise ValueError(f"method must be 'min' or 'mean', got: {method}")

        # Check if technical fit was done with different method
        if self._technical_fit_aggregate_method is not None:
            if self._technical_fit_aggregate_method != method and not allow_after_technical_fit:
                raise ValueError(
                    f"Technical fit was performed with '{self._technical_fit_aggregate_method}' aggregation. "
                    f"Cannot change to '{method}' without invalidating prefit parameters. "
                    f"Set allow_after_technical_fit=True to override (not recommended)."
                )

        # Recompute inclusion
        if method == 'min':
            inclusion = np.minimum(self.inc1, self.inc2)
        elif method == 'mean':
            inclusion = (self.inc1 + self.inc2) / 2.0

        # Recompute total
        total = inclusion + self.skip

        # Update
        self.counts = inclusion
        self.denominator = total
        self.exon_aggregate_method = method

        print(f"[INFO] Recomputed exon skipping inclusion with method='{method}'")

    def is_exon_skipping(self) -> bool:
        """Check if this is an exon skipping modality with inc1/inc2/skip data."""
        return self.inc1 is not None

    def mark_technical_fit_complete(self):
        """
        Mark that technical fit has been performed with current aggregation method.

        This locks the aggregation method to prevent accidental changes that would
        invalidate the prefit overdispersion parameters.
        """
        if self.is_exon_skipping():
            self._technical_fit_aggregate_method = self.exon_aggregate_method

    def __repr__(self) -> str:
        exon_info = f", exon_agg='{self.exon_aggregate_method}'" if self.is_exon_skipping() else ""
        return f"Modality(name='{self.name}', distribution='{self.distribution}', dims={self.dims}{exon_info})"

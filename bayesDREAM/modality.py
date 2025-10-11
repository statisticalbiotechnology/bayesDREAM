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
            else:
                new_counts = self.counts[:, feature_indices]
                new_denom = self.denominator[:, feature_indices] if self.denominator is not None else None
        else:
            # multinomial or mvnormal: features on axis 0
            new_counts = self.counts[feature_indices, :, :]
            new_denom = None

        # Subset metadata
        new_feature_meta = self.feature_meta.iloc[feature_indices].copy()

        return Modality(
            name=self.name,
            counts=new_counts,
            feature_meta=new_feature_meta,
            distribution=self.distribution,
            denominator=new_denom,
            cells_axis=self.cells_axis
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
            else:
                new_counts = self.counts[cell_indices, :]
                new_denom = self.denominator[cell_indices, :] if self.denominator is not None else None
        else:
            # multinomial or mvnormal: cells on axis 1
            new_counts = self.counts[:, cell_indices, :]
            new_denom = None

        return Modality(
            name=self.name,
            counts=new_counts,
            feature_meta=self.feature_meta.copy(),
            distribution=self.distribution,
            denominator=new_denom,
            cells_axis=self.cells_axis
        )

    def __repr__(self) -> str:
        return f"Modality(name='{self.name}', distribution='{self.distribution}', dims={self.dims})"

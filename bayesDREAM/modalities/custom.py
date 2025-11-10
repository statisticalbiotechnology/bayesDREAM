"""
Custom modality methods for bayesDREAM.
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, List, Union

from ..modality import Modality
from ..splicing import create_splicing_modality



class CustomModalityMixin:
    """Mixin for custom modality support."""
    
    def add_custom_modality(
        self,
        name: str,
        counts: Union[np.ndarray, pd.DataFrame],
        feature_meta: pd.DataFrame,
        distribution: str,
        denominator: Optional[np.ndarray] = None,
        cell_names: Optional[List[str]] = None,
        overwrite: bool = False
    ):
        """
        Add a custom user-defined modality with distribution-specific filtering.

        Parameters
        ----------
        name : str
            Modality name
        counts : array or DataFrame
            Measurement data. If DataFrame, cell names come from columns.
            If ndarray, use cell_names parameter to specify cell identifiers.
        feature_meta : pd.DataFrame
            Feature metadata
        distribution : str
            'negbinom', 'multinomial', 'binomial', 'normal', or 'studentt'
        denominator : array, optional
            For binomial: denominator counts
        cell_names : list of str, optional
            Cell names/identifiers (only used when counts is ndarray, not DataFrame).
            Should match number of cells (axis 1 for 2D data, axis 1 for 3D data).
        overwrite : bool, default=False
            Whether to overwrite existing modality with the same name
        """
        # Convert counts to ndarray for consistent filtering
        if isinstance(counts, pd.DataFrame):
            counts_array = counts.values
            is_dataframe = True
            counts_index = counts.columns  # Get column names (cells), not row index
            # Get cell_names from DataFrame columns (will be used later)
            extracted_cell_names = counts.columns.tolist()
        else:
            counts_array = np.asarray(counts)
            is_dataframe = False
            counts_index = None
            # Use provided cell_names (can be None)
            extracted_cell_names = cell_names

        # Apply distribution-specific filtering
        valid_features = None

        if distribution in ['negbinom', 'normal', 'studentt']:
            # Filter features with zero standard deviation
            if counts_array.ndim == 2:
                feature_stds = counts_array.std(axis=1)
                valid_features = feature_stds != 0
                num_filtered = (~valid_features).sum()
                if num_filtered > 0:
                    print(f"[INFO] Filtering {num_filtered} feature(s) with zero std in '{name}' modality ({distribution})")

        elif distribution == 'binomial':
            # Filter features with zero variance in numerator/denominator ratio
            if denominator is None:
                raise ValueError(f"denominator required for binomial distribution in '{name}' modality")

            if isinstance(denominator, pd.DataFrame):
                denom_array = denominator.values
            else:
                denom_array = np.asarray(denominator)

            if counts_array.shape != denom_array.shape:
                raise ValueError(f"counts and denominator must have same shape for binomial in '{name}' modality")

            if counts_array.ndim == 2:
                n_features = counts_array.shape[0]
                valid_features = np.ones(n_features, dtype=bool)

                for i in range(n_features):
                    numer = counts_array[i, :]
                    denom = denom_array[i, :]

                    # Compute ratios, excluding cells where denominator is 0
                    valid_mask = denom > 0
                    if valid_mask.sum() == 0:
                        valid_features[i] = False
                        continue

                    ratios = numer[valid_mask] / denom[valid_mask]
                    if ratios.std() == 0:
                        valid_features[i] = False

                num_filtered = (~valid_features).sum()
                if num_filtered > 0:
                    print(f"[INFO] Filtering {num_filtered} feature(s) with zero ratio variance in '{name}' modality (binomial)")

        elif distribution == 'multinomial':
            # Filter features where ALL category ratios have zero variance
            if counts_array.ndim != 3:
                raise ValueError(f"multinomial requires 3D counts (features, cells, categories) in '{name}' modality")

            n_features = counts_array.shape[0]
            valid_features = np.ones(n_features, dtype=bool)

            for i in range(n_features):
                feature_counts = counts_array[i, :, :]  # (cells, categories)
                totals = feature_counts.sum(axis=1, keepdims=True)  # (cells, 1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(totals > 0, feature_counts / totals, 0)  # (cells, categories)

                # Check if ALL ratios have zero std across cells
                ratio_stds = ratios.std(axis=0)  # std for each category across cells
                if np.all(ratio_stds == 0):
                    valid_features[i] = False

            num_filtered = (~valid_features).sum()
            if num_filtered > 0:
                print(f"[INFO] Filtering {num_filtered} feature(s) with zero variance in ALL category ratios in '{name}' modality (multinomial)")

        # Apply filtering if necessary
        if valid_features is not None:
            if not np.any(valid_features):
                raise ValueError(f"No features left after filtering zero-variance features in '{name}' modality!")

            if not np.all(valid_features):
                # Apply mask
                counts_array = counts_array[valid_features]
                feature_meta = feature_meta.iloc[valid_features].copy()
                if denominator is not None:
                    if isinstance(denominator, pd.DataFrame):
                        denominator = denominator.iloc[valid_features]
                    else:
                        denominator = denom_array[valid_features]

        # Convert back to DataFrame if original was DataFrame
        if is_dataframe:
            if counts_array.ndim == 2:
                counts_final = pd.DataFrame(
                    counts_array,
                    index=feature_meta.index,
                    columns=counts_index if counts_index is not None else range(counts_array.shape[1])
                )
            else:
                # For 3D data, keep as ndarray
                counts_final = counts_array
        else:
            counts_final = counts_array

        modality = Modality(
            name=name,
            counts=counts_final,
            feature_meta=feature_meta,
            distribution=distribution,
            denominator=denominator,
            cells_axis=1,
            cell_names=extracted_cell_names
        )
        self.add_modality(name, modality, overwrite=overwrite)


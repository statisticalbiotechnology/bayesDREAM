"""
Raw x-y data plotting for bayesDREAM.

Plot relationships between cis gene expression (x_true) and modality values
with k-NN smoothing, technical group coloring, and optional technical correction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from scipy.spatial import cKDTree
from typing import Optional, Dict, List, Tuple, Union, Any
import warnings
import torch


# ============================================================================
# Cell Alignment Utilities
# ============================================================================

def _align_cells_to_modality(
    model,
    modality,
    x_true: np.ndarray,
    y_data: np.ndarray,
    subset_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Align model.meta cells with modality cells, handling the case where
    the modality has fewer cells than model.meta.

    Parameters
    ----------
    model : bayesDREAM
        The model instance
    modality : Modality
        The modality instance
    x_true : np.ndarray
        x_true values for ALL cells in model.meta
    y_data : np.ndarray
        y values for cells in the modality (may be fewer than model.meta)
    subset_mask : Optional[np.ndarray]
        Optional boolean mask for subsetting (length = len(model.meta))

    Returns
    -------
    x_true_aligned : np.ndarray
        x_true values aligned to final cell set
    y_data_aligned : np.ndarray
        y values aligned to final cell set
    meta_aligned : pd.DataFrame
        Metadata aligned to final cell set
    """
    # Get cell identifiers
    if 'cell' in model.meta.columns:
        model_cells = model.meta['cell'].values
    else:
        model_cells = model.meta.index.values

    modality_cells = modality.cell_names
    if modality_cells is None:
        raise ValueError(f"Modality '{modality.name}' does not have cell_names set. Cannot align cells.")

    # Create mask for cells in model.meta that are also in modality
    modality_mask = np.isin(model_cells, modality_cells)

    # Combine with subset_mask if provided
    if subset_mask is not None:
        final_mask = modality_mask & subset_mask
    else:
        final_mask = modality_mask

    # Get final set of cells
    final_cells = model_cells[final_mask]

    # Subset x_true and metadata using final_mask
    x_true_aligned = x_true[final_mask]
    meta_aligned = model.meta[final_mask].copy()

    # Map final_cells to indices in modality
    # Create a lookup dictionary for efficiency
    modality_cell_to_idx = {cell: idx for idx, cell in enumerate(modality_cells)}
    y_indices = np.array([modality_cell_to_idx[cell] for cell in final_cells])

    # Subset y_data using these indices
    y_data_aligned = y_data[y_indices]

    return x_true_aligned, y_data_aligned, meta_aligned


# ============================================================================
# k-NN Smoothing Utilities
# ============================================================================

def _knn_k(n: int, window: Union[int, float]) -> int:
    """
    Compute k for k-NN smoothing.

    Parameters
    ----------
    n : int
        Number of data points
    window : int or float
        Window size (if int: exact k, if float: proportion of n)

    Returns
    -------
    int
        k value for k-NN (at least 1, at most n)
    """
    if n <= 0:
        return 1
    k = int(window) if isinstance(window, (int, np.integer)) else int(np.ceil(float(window) * n))
    return max(1, min(k, n))


def _smooth_knn(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    is_ntc: Optional[np.ndarray] = None
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    k-NN smoothing along x-axis.

    Parameters
    ----------
    x : np.ndarray
        X values (1D)
    y : np.ndarray
        Y values (1D)
    k : int
        Number of nearest neighbors
    is_ntc : np.ndarray, optional
        Boolean array indicating NTC cells (1D, same length as x)
        If provided, returns NTC proportions in k-NN windows

    Returns
    -------
    x_sorted : np.ndarray
        Sorted x values
    y_hat : np.ndarray
        Smoothed y values
    ntc_prop : np.ndarray, optional
        If is_ntc provided: proportion of NTC cells in each k-NN window
    """
    if len(x) == 0:
        if is_ntc is not None:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])

    order = np.argsort(x)
    x_sorted = np.asarray(x)[order].reshape(-1, 1)
    y_sorted = np.asarray(y)[order]

    k = max(1, min(k, len(x_sorted)))
    tree = cKDTree(x_sorted)

    y_hat = np.empty_like(y_sorted, dtype=float)

    if is_ntc is not None:
        is_ntc_sorted = np.asarray(is_ntc)[order]
        ntc_prop = np.empty_like(y_sorted, dtype=float)

        for i in range(len(x_sorted)):
            _, idx = tree.query(x_sorted[i], k=k)
            y_hat[i] = np.nanmean(y_sorted[idx])
            ntc_prop[i] = np.mean(is_ntc_sorted[idx])  # Proportion of NTC in window

        return x_sorted.ravel(), y_hat, ntc_prop
    else:
        for i in range(len(x_sorted)):
            _, idx = tree.query(x_sorted[i], k=k)
            y_hat[i] = np.nanmean(y_sorted[idx])

        return x_sorted.ravel(), y_hat


def _smooth_knn_counts(
    x: np.ndarray,
    numerators: np.ndarray,
    denominators: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    k-NN smoothing for count data by binning.

    Aggregates numerators and denominators within k-NN windows,
    allowing proportion calculation from aggregated counts.

    Parameters
    ----------
    x : np.ndarray
        X values (1D, length n)
    numerators : np.ndarray
        Numerator counts (1D for binomial: length n, or 2D for multinomial: n × K)
    denominators : np.ndarray
        Denominator totals (1D, length n)
    k : int
        Number of nearest neighbors

    Returns
    -------
    x_sorted : np.ndarray
        Sorted x values (length n)
    aggregated_num : np.ndarray
        Aggregated numerators (1D or 2D matching input shape)
    aggregated_denom : np.ndarray
        Aggregated denominators (1D, length n)
    """
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(x)
    x_sorted = np.asarray(x)[order].reshape(-1, 1)

    k = max(1, min(k, len(x_sorted)))
    tree = cKDTree(x_sorted)

    # Handle 1D (binomial) vs 2D (multinomial) numerators
    is_2d = numerators.ndim == 2
    if is_2d:
        num_sorted = np.asarray(numerators)[order, :]  # (n, K)
        K = num_sorted.shape[1]
        aggregated_num = np.zeros((len(x_sorted), K), dtype=float)
    else:
        num_sorted = np.asarray(numerators)[order]  # (n,)
        aggregated_num = np.zeros(len(x_sorted), dtype=float)

    denom_sorted = np.asarray(denominators)[order]
    aggregated_denom = np.zeros(len(x_sorted), dtype=float)

    for i in range(len(x_sorted)):
        _, idx = tree.query(x_sorted[i], k=k)

        if is_2d:
            aggregated_num[i, :] = np.sum(num_sorted[idx, :], axis=0)
        else:
            aggregated_num[i] = np.sum(num_sorted[idx])

        aggregated_denom[i] = np.sum(denom_sorted[idx])

    return x_sorted.ravel(), aggregated_num, aggregated_denom


def _to_2d(a):
    """
    Convert input to 2D numpy array.

    Handles torch tensors and reshapes 1D arrays.
    """
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype(float)
    except Exception:
        pass
    a = np.asarray(a)
    return a.reshape(1, -1).astype(float) if a.ndim == 1 else a.astype(float)


def reorder_xtrue_by_barcode(
    x_true: Union[np.ndarray, torch.Tensor],
    src_barcodes: np.ndarray,
    dst_barcodes: np.ndarray
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reorder x_true from source barcode order to destination barcode order.

    Parameters
    ----------
    x_true : array or tensor
        Values to reorder
    src_barcodes : np.ndarray
        Source barcode order
    dst_barcodes : np.ndarray
        Destination barcode order

    Returns
    -------
    array or tensor
        Reordered x_true
    """
    is_torch = False
    try:
        import torch
        if isinstance(x_true, torch.Tensor):
            is_torch = True
            x_np = x_true.detach().cpu().numpy()
            torch_dtype, torch_device = x_true.dtype, x_true.device
        else:
            x_np = np.asarray(x_true)
    except Exception:
        x_np = np.asarray(x_true)

    src = np.asarray(src_barcodes).astype(str)
    dst = np.asarray(dst_barcodes).astype(str)

    pos = {b: i for i, b in enumerate(src)}
    perm = np.fromiter((pos[b] for b in dst), dtype=int, count=len(dst))

    x_np = x_np if x_np.ndim == 1 else x_np[:, :]
    x_np = x_np[perm] if x_np.ndim == 1 else x_np[:, perm]

    if is_torch:
        import torch
        return torch.as_tensor(x_np, dtype=torch_dtype, device=torch_device)
    return x_np


# ============================================================================
# Feature Lookup Utilities
# ============================================================================

def _resolve_features(feature_or_gene: str, modality) -> Tuple[List[int], List[str], bool]:
    """
    Resolve input to feature index(es) and name(s).

    Returns
    -------
    feature_indices : List[int]
        Integer positions of features
    feature_names : List[str]
        Feature names
    is_gene : bool
        True if input was a gene name (multiple features), False if single feature

    Raises
    ------
    ValueError
        If feature/gene not found or modality doesn't have gene information
    """
    # First, try as a direct feature match
    feature_idx = _get_feature_index(feature_or_gene, modality)
    if feature_idx is not None:
        # Found as a feature - get the actual feature name
        if hasattr(modality, 'feature_names') and modality.feature_names is not None:
            feature_name = modality.feature_names[feature_idx]
        else:
            feature_name = feature_or_gene
        return [feature_idx], [feature_name], False

    # Not found as feature - try as gene
    # Check if modality has gene information
    gene_cols = ['gene', 'gene_name', 'gene_id']
    available_gene_cols = [col for col in gene_cols if col in modality.feature_meta.columns]

    if not available_gene_cols:
        raise ValueError(
            f"'{feature_or_gene}' not found as a feature, and modality '{modality.name}' "
            f"has no gene information (no 'gene', 'gene_name', or 'gene_id' columns in feature_meta). "
            f"Cannot search by gene name."
        )

    # Search for gene across all gene columns
    matching_features = []
    for gene_col in available_gene_cols:
        mask = modality.feature_meta[gene_col] == feature_or_gene
        if mask.sum() > 0:
            # Found matches
            indices = mask.values.nonzero()[0].tolist()

            # Get feature names for matched indices
            if hasattr(modality, 'feature_names') and modality.feature_names is not None:
                names = [modality.feature_names[i] for i in indices]
            elif modality.feature_meta.index.name:
                names = modality.feature_meta.iloc[indices].index.tolist()
            else:
                names = [str(i) for i in indices]

            return indices, names, True

    # Not found as feature or gene
    raise ValueError(
        f"'{feature_or_gene}' not found as a feature or gene in modality '{modality.name}'. "
        f"Checked: feature names/index and gene columns {available_gene_cols}"
    )


def _get_feature_index(feature: str, modality) -> Optional[int]:
    """
    Robustly find feature index, checking multiple locations.

    Checks feature_meta.index, feature_meta columns, and feature_names attribute.

    IMPORTANT: Does NOT check 'gene', 'gene_name', or 'gene_id' columns -
    those are reserved for gene-level searches in _resolve_features().

    Parameters
    ----------
    feature : str
        Feature name to find
    modality : Modality
        Modality object to search in

    Returns
    -------
    int or None
        Feature index if found, None otherwise
    """
    # Check index first
    if feature in modality.feature_meta.index:
        return modality.feature_meta.index.get_loc(feature)

    # Check common column names (EXCLUDING gene columns)
    for col in ['junction', 'feature', 'name', 'coord.intron', 'SJ']:
        if col in modality.feature_meta.columns:
            mask = modality.feature_meta[col] == feature
            if mask.sum() > 0:
                # Use argmax to get integer position, not index label
                return mask.values.argmax()

    # Check feature_names attribute
    if hasattr(modality, 'feature_names'):
        feature_names = modality.feature_names
        if isinstance(feature_names, (list, np.ndarray)):
            feature_names_list = list(feature_names)
            if feature in feature_names_list:
                return feature_names_list.index(feature)

    return None


def _to_scalar(val):
    """
    Convert value to Python scalar.

    Handles PyTorch tensors, numpy arrays, and Python scalars.

    Parameters
    ----------
    val : tensor, array, or scalar
        Value to convert

    Returns
    -------
    float
        Python scalar
    """
    # PyTorch tensor
    if hasattr(val, 'item'):
        return val.item()
    # Numpy array
    if isinstance(val, np.ndarray):
        return float(val)
    # Already scalar
    return float(val)

def _multinomial_correct_binned_probs(
    props_binned: np.ndarray,         # (n_bins, K) proportions from aggregated counts
    alpha_y_add,                      # [S, C, T, K] or [C, T, K]
    group_code: int,
    feature_idx: int,
    zero_cat_mask: np.ndarray         # (K,) True where category is globally zero
) -> np.ndarray:
    """
    Apply multinomial technical correction to binned proportions.

    For each bin: softmax(log(P) - alpha_s) for each posterior sample s,
    then average across s.

    Parameters
    ----------
    props_binned : np.ndarray
        Proportions computed from aggregated counts in k-NN bins (n_bins, K)
    alpha_y_add : torch.Tensor or np.ndarray
        Technical correction parameters [S, C, T, K] or [C, T, K]
    group_code : int
        Technical group code for this data
    feature_idx : int
        Feature index in alpha array
    zero_cat_mask : np.ndarray
        Boolean mask (K,) indicating globally absent categories

    Returns
    -------
    np.ndarray
        Corrected proportions (n_bins, K)
    """
    epsilon = 1e-6
    # Clip and log
    P = np.clip(props_binned, epsilon, 1 - epsilon)   # (n_bins, K)
    logP = np.log(P)                                  # (n_bins, K)

    # Pull alpha with explicit samples dim
    if alpha_y_add.ndim == 4:        # [S, C, T, K]
        A = alpha_y_add[:, group_code, feature_idx, :]          # (S, K)
    elif alpha_y_add.ndim == 3:      # [C, T, K] -> add S=1
        A = alpha_y_add[None, group_code, feature_idx, :]       # (1, K)
    else:
        raise ValueError(f"Unexpected alpha_y_add shape: {alpha_y_add.shape}")

    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()

    # Broadcast to (S, n_bins, K)
    logP_S = logP[None, :, :]
    A_S    = A[:, None, :]

    logits = logP_S - A_S                                 # (S, n_bins, K)

    # Hard-mask zero categories
    logits[:, :, zero_cat_mask] = -np.inf

    # Stable softmax with mask
    m = np.nanmax(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - m)
    exps[:, :, zero_cat_mask] = 0.0
    Z = exps.sum(axis=-1, keepdims=True)
    Z[Z == 0] = 1.0
    P_corr_S = exps / Z                                    # (S, n_bins, K)

    # Posterior mean in probability space
    return P_corr_S.mean(axis=0)                           # (n_bins, K)



# ============================================================================
# Technical Group Utilities
# ============================================================================

def get_technical_group_labels(model) -> List[str]:
    """
    Get informative labels for technical groups.

    Returns human-readable labels like "CRISPRa", "CRISPRi:lane1" instead of
    generic "technical_group_0", "technical_group_1".

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model

    Returns
    -------
    List[str]
        Informative labels for each technical group code
    """
    if 'technical_group_code' not in model.meta.columns:
        raise ValueError("technical_group_code not set. Run set_technical_groups() first.")

    # Get unique technical groups
    group_codes = sorted(model.meta['technical_group_code'].unique())

    # Determine which covariates were used
    # We need to reverse engineer this from the grouping
    # Try common covariate combinations
    potential_covariates = ['cell_line', 'lane', 'batch', 'replicate']
    available_covariates = [c for c in potential_covariates if c in model.meta.columns]

    if not available_covariates:
        # Fall back to generic labels
        return [f'Group_{i}' for i in group_codes]

    # Find which covariates were used by checking if they distinguish groups
    used_covariates = []
    for cov in available_covariates:
        test_grouping = model.meta.groupby(used_covariates + [cov]).ngroup()
        if test_grouping.nunique() == len(group_codes) and (test_grouping == model.meta['technical_group_code']).all():
            # This covariate set matches
            used_covariates.append(cov)
            break
        elif len(used_covariates) == 0:
            # Try single covariate
            test_grouping = model.meta.groupby([cov]).ngroup()
            if test_grouping.nunique() == len(group_codes) and (test_grouping == model.meta['technical_group_code']).all():
                used_covariates = [cov]
                break

    if not used_covariates:
        # Couldn't determine covariates, use generic labels
        return [f'Group_{i}' for i in group_codes]

    # Create labels
    labels = []
    for code in group_codes:
        mask = model.meta['technical_group_code'] == code
        group_data = model.meta.loc[mask, used_covariates].iloc[0]

        if len(used_covariates) == 1:
            # Single covariate: just use the value
            labels.append(str(group_data[used_covariates[0]]))
        else:
            # Multiple covariates: join with ":"
            parts = [str(group_data[cov]) for cov in used_covariates]
            labels.append(':'.join(parts))

    return labels


def get_default_color_palette(labels: List[str]) -> Dict[str, str]:
    """
    Get default color palette for technical groups.

    Parameters
    ----------
    labels : List[str]
        Technical group labels

    Returns
    -------
    Dict[str, str]
        Mapping from label to color
    """
    # Default colors for common cases
    default_colors = {
        'CRISPRa': 'crimson',
        'CRISPRi': 'dodgerblue',
        'a': 'crimson',
        'i': 'dodgerblue'
    }

    # Check if labels match defaults
    palette = {}
    for label in labels:
        if label in default_colors:
            palette[label] = default_colors[label]
        else:
            # Check if label contains CRISPRa or CRISPRi
            if 'CRISPRa' in label or 'a' in label:
                palette[label] = 'crimson'
            elif 'CRISPRi' in label or 'i' in label:
                palette[label] = 'dodgerblue'
            else:
                # Fall back to matplotlib default colors
                palette[label] = f'C{len(palette)}'

    return palette


def _labels_by_code_for_df(model, df) -> dict[int, str]:
    """
    Return {technical_group_code -> human-readable label} *for codes present in df*.
    Prefers 'cell_line' if available, else falls back to a stable generic label.
    """
    present = np.sort(df['technical_group_code'].unique())
    label_col = 'cell_line' if 'cell_line' in model.meta.columns else None

    mapping = {}
    for code in present:
        if label_col is not None:
            vals = model.meta.loc[model.meta['technical_group_code'] == code, label_col]
            mapping[int(code)] = str(vals.iloc[0]) if len(vals) else f'Group_{int(code)}'
        else:
            mapping[int(code)] = f'Group_{int(code)}'
    return mapping


def _color_for_label(label: str, fallback_idx: int = 0, palette: dict | None = None) -> str:
    """
    Use user's desired palette: CRISPRa=red, CRISPRi=blue; otherwise fall back.
    """
    palette = palette or {}
    if label in palette:
        return palette[label]
    # hard defaults you wanted
    if 'CRISPRa' in label:
        return 'crimson'
    if 'CRISPRi' in label:
        return 'dodgerblue'
    return f'C{fallback_idx}'


def plot_colored_line(
    x: np.ndarray,
    y: np.ndarray,
    color_values: np.ndarray,
    cmap,
    ax: plt.Axes,
    linewidth: float = 2
):
    """
    Plot a line with color gradient based on values.

    Parameters
    ----------
    x, y : np.ndarray
        Line coordinates
    color_values : np.ndarray
        Values for coloring (same length as x, y)
    cmap : colormap
        Matplotlib colormap
    ax : plt.Axes
        Axes to plot on
    linewidth : float
        Line width

    Returns
    -------
    LineCollection
        The created line collection
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = Normalize(vmin=0, vmax=1)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(color_values)
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)

    return lc


# ============================================================================
# Trans Function Prediction Utilities (for overlaying fitted functions)
# ============================================================================

def Hill_based_positive(x, Vmax, A, K, n, epsilon=1e-6):
    """
    Positive Hill function.

    Parameters
    ----------
    x : array
        Input values
    Vmax, A, K, n : float or array
        Hill parameters
    epsilon : float
        Small constant for numerical stability

    Returns
    -------
    array
        Hill function values
    """
    x_safe = x + epsilon
    K_safe = K + epsilon
    x_log = np.log(x_safe)
    K_log = np.log(K_safe)
    x_n = np.exp(n * x_log)
    K_n = np.exp(n * K_log)
    denominator = K_n + x_n
    return Vmax * (x_n / denominator) + A


def predict_trans_function(
    model,
    feature: str,
    x_range: np.ndarray,
    modality_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Predict trans effect function for a feature given x_true values.

    Automatically detects function type (additive_hill, single_hill, polynomial)
    from posterior_samples_trans and computes predictions.

    Parameters
    ----------
    model : bayesDREAM
        Model with fit_trans() completed
    feature : str
        Feature name to predict
    x_range : np.ndarray
        X values to predict at (cis expression levels)
    modality_name : str, optional
        Modality name (default: primary modality)

    Returns
    -------
    np.ndarray or None
        Predicted y values at x_range, or None if:
        - Trans model not fitted
        - Feature not in trans_genes
        - Cannot determine function type

    Notes
    -----
    Works with all function types:
    - additive_hill: A + alpha*Hill_a + beta*Hill_b
    - single_hill: A + Vmax*Hill(x, K, n)
    - polynomial: sum(coeffs[i] * x^i)
    """
    # Determine which modality to use
    if modality_name is None:
        modality_name = model.primary_modality

    # Check if trans model fitted for this modality
    if modality_name == model.primary_modality:
        # Primary modality: check model-level posterior
        if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
            return None
        posterior = model.posterior_samples_trans
    else:
        # Non-primary modality: check modality-level posterior
        modality = model.get_modality(modality_name)
        if not hasattr(modality, 'posterior_samples_trans') or modality.posterior_samples_trans is None:
            return None
        posterior = modality.posterior_samples_trans

    # Get baseline A (present in all function types)
    if 'A' not in posterior:
        return None

    A_samples = posterior['A']

    # Get posterior dimensions
    # For primary modality: A_samples is (S, T) where S=samples, T=trans_genes
    #   After mean(dim=0): (T,)
    # For non-primary modality: A_samples is (S, C, T) where C=cis_genes, T=trans_features
    #   After mean(dim=0): (C, T)
    # We want the LAST dimension (T) in both cases
    if hasattr(A_samples, 'mean'):
        A_mean = A_samples.mean(dim=0)
        # Squeeze out cis gene dimension if present (should be size 1 for non-primary modalities)
        if A_mean.ndim > 1:
            A_mean = A_mean.squeeze(0)
        n_genes_posterior = A_mean.shape[0]
    else:
        A_mean = A_samples.mean(axis=0)
        # Squeeze out cis gene dimension if present
        if A_mean.ndim > 1:
            A_mean = A_mean.squeeze(0)
        n_genes_posterior = A_mean.shape[0]

    # Get feature list from modality (NOT model.trans_genes!)
    # model.trans_genes is only valid for primary modality
    if modality_name == model.primary_modality:
        # Primary modality: use model.trans_genes if available
        feature_list = model.trans_genes if hasattr(model, 'trans_genes') else []
    else:
        # Non-primary modality: get feature names from modality
        modality = model.get_modality(modality_name)
        if modality.feature_meta is not None:
            # Try common identifier columns in order of preference
            # For splicing: prioritize coordinate-based identifiers (coord.intron, junction_id)
            # For others: prioritize feature_id, feature, then fall back to gene names
            feature_list = None
            for col in ['feature_id', 'feature', 'coord.intron', 'junction_id', 'gene_name', 'gene']:
                if col in modality.feature_meta.columns:
                    feature_list = modality.feature_meta[col].tolist()
                    break

            # If no column worked, try the index
            if feature_list is None:
                feature_list = modality.feature_meta.index.tolist()
        else:
            # No feature metadata - try to use feature count from posterior
            # Assume features are indexed 0, 1, 2, ... and cannot be matched by name
            return None

    n_features_list = len(feature_list)

    # Check dimension consistency BEFORE using feature_list for indexing
    if n_genes_posterior != n_features_list:
        # Mismatch - cannot use feature_list for indexing
        # This happens when posterior was fitted on a subset of features
        return None

    # Now safe to check if feature is in feature_list and get its index
    if feature not in feature_list:
        return None

    feature_idx = feature_list.index(feature)
    A = A_mean[feature_idx].item() if hasattr(A_mean, 'item') else A_mean[feature_idx]

    # Helper function to extract parameter value for a specific feature
    # Handles both primary modality (S, T) and non-primary modality (S, C, T) shapes
    def _extract_param(param_samples, feature_idx):
        """Extract mean parameter value for a specific feature, handling dimension squeezing."""
        if hasattr(param_samples, 'mean'):
            param_mean = param_samples.mean(dim=0)
        else:
            param_mean = param_samples.mean(axis=0)

        # Squeeze out cis gene dimension if present
        if param_mean.ndim > 1:
            param_mean = param_mean.squeeze(0)

        # Extract value for this feature
        val = param_mean[feature_idx]
        return val.item() if hasattr(val, 'item') else val

    # Determine function type from available parameters
    if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL (negbinom/normal/studentt) =====
        try:
            # Extract parameters using helper function
            alpha = _extract_param(posterior['alpha'], feature_idx)
            beta = _extract_param(posterior['beta'], feature_idx)
            Vmax_a = _extract_param(posterior['Vmax_a'], feature_idx)
            Vmax_b = _extract_param(posterior['Vmax_b'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            K_b = _extract_param(posterior['K_b'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)
            n_b = _extract_param(posterior['n_b'], feature_idx)

            # Compute Hill functions
            Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a, A=0, K=K_a, n=n_a)
            Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b, A=0, K=K_b, n=n_b)

            # Combined prediction
            y_pred = A + alpha * Hill_a + beta * Hill_b
            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    elif 'upper_limit' in posterior and 'alpha' in posterior and 'beta' in posterior:
        # ===== ADDITIVE HILL (binomial/multinomial with upper_limit) =====
        try:
            # Extract parameters using helper function
            alpha = _extract_param(posterior['alpha'], feature_idx)
            beta = _extract_param(posterior['beta'], feature_idx)
            upper_limit = _extract_param(posterior['upper_limit'], feature_idx)
            K_a = _extract_param(posterior['K_a'], feature_idx)
            K_b = _extract_param(posterior['K_b'], feature_idx)
            n_a = _extract_param(posterior['n_a'], feature_idx)
            n_b = _extract_param(posterior['n_b'], feature_idx)

            # IMPORTANT: Match the actual model computation!
            # Model uses: y = A + Vmax_sum * (alpha * Hill_{Vmax=1}(x) + beta * Hill_{Vmax=1}(x))
            # where Hills with Vmax=1 return values in [0, 1]
            Vmax_sum = upper_limit - A

            # Compute Hill functions with Vmax=1 (normalized to [0, 1])
            Hill_a = Hill_based_positive(x_range, Vmax=1.0, A=0, K=K_a, n=n_a)
            Hill_b = Hill_based_positive(x_range, Vmax=1.0, A=0, K=K_b, n=n_b)

            # Combined Hill can exceed 1 if alpha + beta > 1!
            combined_hill = alpha * Hill_a + beta * Hill_b

            # Combined prediction (can exceed upper_limit if combined_hill > 1)
            y_pred = A + Vmax_sum * combined_hill
            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    elif 'Vmax' in posterior and 'K' in posterior and 'n' in posterior:
        # ===== SINGLE HILL =====
        try:
            Vmax = _extract_param(posterior['Vmax'], feature_idx)
            K = _extract_param(posterior['K'], feature_idx)
            n = _extract_param(posterior['n'], feature_idx)

            # Compute Hill function
            y_pred = Hill_based_positive(x_range, Vmax=Vmax, A=A, K=K, n=n)
            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    elif 'theta' in posterior:
        # ===== POLYNOMIAL OR THETA-BASED =====
        try:
            theta_samples = posterior['theta']
            # theta shape for primary: (samples, features, n_params)
            # theta shape for non-primary: (samples, cis_genes, features, n_params)

            # Average over samples and squeeze cis gene dimension if present
            if hasattr(theta_samples, 'mean'):
                theta_mean = theta_samples.mean(dim=0)
                # Squeeze out cis gene dimension if present
                if theta_mean.ndim > 2:  # (cis_genes, features, n_params)
                    theta_mean = theta_mean.squeeze(0)  # (features, n_params)
                theta_mean = theta_mean[feature_idx, :]  # (n_params,)
                theta_np = theta_mean.cpu().numpy() if hasattr(theta_mean, 'cpu') else np.array(theta_mean)
            else:
                theta_mean = theta_samples.mean(axis=0)
                if theta_mean.ndim > 2:
                    theta_mean = theta_mean.squeeze(0)
                theta_mean = theta_mean[feature_idx, :]
                theta_np = np.array(theta_mean)

            # Polynomial: y = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
            # First coefficient is baseline (like A)
            baseline = theta_np[0]
            poly_coeffs = theta_np[1:]  # Remaining coefficients

            # Compute polynomial
            y_pred = np.full_like(x_range, baseline)
            for i, coeff in enumerate(poly_coeffs, start=1):
                y_pred += coeff * (x_range ** i)

            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    else:
        # Unknown function type
        return None


# ============================================================================
# Distribution-Specific Plotting Functions
# ============================================================================

def plot_negbinom_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_hill_function: bool,
    show_ntc_gradient: bool = False,
    sum_factor_col: str = 'sum_factor',
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot negbinom (gene counts) with optional Hill function overlay.

    Y-axis: log2(expression) where expression = counts / (sum_factor * alpha_y)

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
    sum_factor_col : str
        Column name in model.meta for sum factors (default: 'sum_factor')
        Can be 'sum_factor', 'sum_factor_adj', or any other sum factor column
    """
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts for this feature
    if modality.cells_axis == 1:
        y_obs = modality.counts[feature_idx, :]
    else:
        y_obs = modality.counts[:, feature_idx]

    # Check if sum_factor_col exists
    if sum_factor_col not in model.meta.columns:
        raise ValueError(f"Sum factor column '{sum_factor_col}' not found in model.meta. "
                        f"Available columns: {list(model.meta.columns)}")

    # Align cells between model.meta and modality
    x_true_aligned, y_obs_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, y_obs, subset_mask
    )

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true_aligned,
        'y_obs': y_obs_aligned,
        'technical_group_code': meta_aligned['technical_group_code'].values,
        'target': meta_aligned['target'].values,
        'sum_factor': meta_aligned[sum_factor_col].values
    })

    # Technical correction
    has_technical_fit = modality.alpha_y_prefit is not None

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create axes
    if ax is None:
        if show_correction == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            axes = [ax]
    else:
        # Single axis provided - cannot show both corrected and uncorrected
        if show_correction == 'both':
            warnings.warn("show_correction='both' not supported when ax is provided (multi-panel mode). Showing uncorrected only.")
            show_correction = 'uncorrected'
        axes = [ax]

    # Map technical_group_code → label using only codes present in this df
    code_to_label = _labels_by_code_for_df(model, df)
    group_codes   = np.sort(df['technical_group_code'].unique())
    group_labels  = [code_to_label[int(c)] for c in group_codes]

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormaps for NTC gradient (per technical group)
    # Each group gets white → group_color gradient
    group_cmaps = {}
    if show_ntc_gradient:
        for idx, group_code in enumerate(group_codes):
            group_code  = int(group_code)
            group_label = code_to_label[group_code]
            base_color  = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
            group_cmaps[group_label] = LinearSegmentedColormap.from_list(
                f"white_{group_label}",
                ["white", base_color]
            )

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added

        
        for idx, group_code in enumerate(group_codes):
            group_code = int(group_code)
            group_label = code_to_label[group_code]
            df_group = df[df['technical_group_code'] == group_code].copy()

            if corrected and has_technical_fit:
                # Apply alpha_y correction
                alpha_y_full = modality.alpha_y_prefit  # [S or 1, C, T]
                if alpha_y_full.ndim == 3:
                    alpha_y_val = _to_scalar(alpha_y_full[:, group_code, feature_idx].mean())
                else:
                    alpha_y_val = _to_scalar(alpha_y_full[group_code, feature_idx])
                y_expr = df_group['y_obs'] / (df_group['sum_factor'] * alpha_y_val)
            else:
                y_expr = df_group['y_obs'] / df_group['sum_factor']

            # Filter valid
            valid = (df_group['x_true'] > 0) & np.isfinite(y_expr)
            df_group = df_group[valid].copy()
            y_expr = y_expr[valid]

            if len(df_group) == 0:
                continue

            # Get is_ntc for this group
            is_ntc_group = is_ntc[df_group.index].values

            # Transform to log2(y+1) BEFORE smoothing
            y_log = np.log2(y_expr + 1)

            # k-NN smoothing in log space
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient:
                # Smoothing with NTC tracking
                x_smooth, y_smooth, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_log,
                    k,
                    is_ntc=is_ntc_group
                )

                # Use per-group gradient coloring (white → group color)
                # Color value = 1 - ntc_prop: high NTC → 0 → white, low NTC → 1 → group color
                group_cmap = group_cmaps.get(group_label, plt.cm.gray)
                plot_colored_line(
                    x=np.log2(x_smooth),
                    y=y_smooth,  # Already in log2(y+1) space
                    color_values=1 - ntc_prop,  # Darker (group color) = fewer NTCs
                    cmap=group_cmap,
                    ax=ax_plot,
                    linewidth=2
                )

                # Add dummy invisible line for legend label (using base group color)
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                ax_plot.plot([], [], color=color, linewidth=2, label=group_label)

                # Add colorbar (once per axis) - use grayscale to show NTC gradient
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    cmap_gray = LinearSegmentedColormap.from_list("gray_gradient", ["white", "black"])
                    sm = ScalarMappable(cmap=cmap_gray, norm=Normalize(0, 1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax_plot)
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                # Standard smoothing without NTC tracking in log space
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_log, k)

                # Use standard coloring
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                ax_plot.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        # Show on corrected plot only (trans model was fitted on corrected data)
        if show_hill_function and corrected:
            x_range = np.linspace(x_true.min(), x_true.max(), 100)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=None)

            if y_pred is not None:
                # Transform prediction to log2(y+1) space to match data
                ax_plot.plot(np.log2(x_range), np.log2(y_pred + 1),
                            color='blue', linestyle='--', linewidth=2,
                            label='Fitted Trans Function')

                # Add baseline if available
                if hasattr(model, 'posterior_samples_trans') and 'A' in model.posterior_samples_trans:
                    A_samples = model.posterior_samples_trans['A']
                    trans_genes = model.trans_genes if hasattr(model, 'trans_genes') else []
                    if feature in trans_genes:
                        gene_idx = trans_genes.index(feature)
                        A = A_samples.mean(dim=0)[gene_idx].item() if hasattr(A_samples, 'mean') else A_samples.mean(axis=0)[gene_idx]
                        ax_plot.axhline(np.log2(A + 1), color='red', linestyle=':', linewidth=1, label='log2(A+1) baseline')

        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel('log2(Expression + 1)')
        title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
        ax_plot.set_title(f"{model.cis_gene} → {feature}{title_suffix}")
        ax_plot.legend(frameon=False)

    # Plot
    if show_correction == 'both':
        _plot_one(axes[0], corrected=False)
        _plot_one(axes[1], corrected=True)
    elif show_correction == 'corrected':
        _plot_one(axes[0], corrected=True)
    else:
        _plot_one(axes[0], corrected=False)

    return axes[0] if len(axes) == 1 else axes


def plot_binomial_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    min_counts: int,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,
    show_correction: str = 'both',
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot binomial (PSI - percent spliced in).

    Y-axis: PSI (%) = (counts / denominator) * 100  (percentage scale: 0-100)
    Filter: min_counts on denominator

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
    show_correction : str
        'uncorrected': no technical correction
        'corrected': apply alpha_y_add additive correction (PSI - alpha_y_add)
        'both': show both side-by-side

    Notes
    -----
    Technical correction for binomial uses alpha_y_add (additive correction on LOGIT scale).
    logit(PSI) = log(PSI / (1 - PSI))
    logit_corrected = logit(PSI) - alpha_y_add
    PSI_corrected = 1 / (1 + exp(-logit_corrected))
    """
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts and denominator
    if modality.cells_axis == 1:
        counts = modality.counts[feature_idx, :]
        denom = modality.denominator[feature_idx, :]
    else:
        counts = modality.counts[:, feature_idx]
        denom = modality.denominator[:, feature_idx]

    # Align cells between model.meta and modality
    x_true_aligned, counts_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, counts, subset_mask
    )
    # Also align denominator (same cell alignment)
    _, denom_aligned, _ = _align_cells_to_modality(
        model, modality, x_true, denom, subset_mask
    )

    # Filter by min_counts
    valid_mask = denom_aligned >= min_counts

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true_aligned[valid_mask],
        'counts': counts_aligned[valid_mask],
        'denominator': denom_aligned[valid_mask],
        'technical_group_code': meta_aligned['technical_group_code'].values[valid_mask],
        'target': meta_aligned['target'].values[valid_mask]
    })

    # Keep counts and denominators for binning (don't compute PSI per-cell)
    # Filter valid
    valid = (df['x_true'] > 0) & (df['denominator'] > 0)
    df = df[valid].copy()

    if len(df) == 0:
        raise ValueError(f"No data remaining after filtering (min_counts={min_counts})")

    # Check technical correction availability
    # First check attribute, then fall back to posterior_samples_technical dict
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            # Set the attribute for future use
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from posterior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create axes
    if ax is None:
        if show_correction == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            axes = [ax]
    else:
        # Single axis provided - cannot show both corrected and uncorrected
        if show_correction == 'both':
            warnings.warn("show_correction='both' not supported when ax is provided (multi-panel mode). Showing uncorrected only.")
            show_correction = 'uncorrected'
        axes = [ax]

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)
    group_codes   = np.sort(df['technical_group_code'].unique())

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormaps for NTC gradient (per technical group)
    # Each group gets white → group_color gradient
    group_cmaps = {}
    if show_ntc_gradient:
        for group_label in group_labels:
            base_color = color_palette.get(group_label, f'C0')
            # Create colormap: white (low color value) → group color (high color value)
            group_cmaps[group_label] = LinearSegmentedColormap.from_list(
                f"white_{group_label}",
                ["white", base_color]
            )

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added
        
        for idx, group_code in enumerate(group_codes):
            group_code = int(group_code)
            group_label = code_to_label[group_code]
            color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

            # Get data for this group
            group_mask = df['technical_group_code'] == group_code
            if group_mask.sum() == 0:
                continue

            x_group = df.loc[group_mask, 'x_true'].values
            counts_group = df.loc[group_mask, 'counts'].values       # (n_cells_group,)
            denom_group = df.loc[group_mask, 'denominator'].values   # (n_cells_group,)

            # Bin counts using k-NN
            k = _knn_k(len(x_group), window)
            x_smooth, counts_binned, denom_binned = _smooth_knn_counts(
                x_group, counts_group, denom_group, k
            )  # counts_binned: (n_bins,), denom_binned: (n_bins,)

            # Compute PSI from binned counts (as proportion [0, 1])
            epsilon = 1e-10
            with np.errstate(divide='ignore', invalid='ignore'):
                p_binned = np.where(denom_binned > epsilon,
                                    counts_binned / denom_binned,
                                    0.5)  # Default to 0.5 if no data

            if corrected and has_technical_fit:
                # Apply additive correction on LOGIT scale to binned proportions
                # 1. Clip proportions to avoid log(0) or log(1)
                epsilon_clip = 1e-6
                p_clipped = np.clip(p_binned, epsilon_clip, 1 - epsilon_clip)

                # 2. Calculate logit
                logit_p = np.log(p_clipped / (1 - p_clipped))

                # 3. Apply correction on logit scale
                alpha_y_add = modality.alpha_y_prefit_add
                if alpha_y_add.ndim == 3:
                    correction = _to_scalar(alpha_y_add[:, group_code, feature_idx].mean())
                else:
                    correction = _to_scalar(alpha_y_add[group_code, feature_idx])
                logit_corrected = logit_p - correction

                # 4. Convert back to proportion
                p_corrected = 1.0 / (1.0 + np.exp(-logit_corrected))

                # 5. Convert to percentage
                y_smooth = p_corrected * 100.0
            else:
                # Convert uncorrected proportion to percentage
                y_smooth = p_binned * 100.0

            # Plot (no additional smoothing needed - already binned)
            # Note: We don't currently support NTC gradient for binned data
            ax_plot.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        # Show on corrected plot only (trans model was fitted on corrected data)
        if show_trans_function and corrected:
            x_range = np.linspace(x_true.min(), x_true.max(), 100)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

            if y_pred is not None:
                # For binomial, PSI is in percentage scale [0, 100], so scale and clip predictions
                y_pred_pct = y_pred * 100.0
                y_pred_clipped = np.clip(y_pred_pct, 0, 100)
                ax_plot.plot(np.log2(x_range), y_pred_clipped,
                           color='blue', linestyle='--', linewidth=2,
                           label='Fitted Trans Function')

        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel('PSI (%)')
        title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
        ax_plot.set_title(f"{model.cis_gene} → {feature} (min_counts={min_counts}{title_suffix})")
        ax_plot.legend(frameon=False)

    # Plot
    if show_correction == 'both':
        _plot_one(axes[0], corrected=False)
        _plot_one(axes[1], corrected=True)
    elif show_correction == 'corrected':
        _plot_one(axes[0], corrected=True)
    else:
        _plot_one(axes[0], corrected=False)

    return axes[0] if len(axes) == 1 else axes


def plot_multinomial_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    min_counts: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> Union[plt.Figure, List[plt.Axes]]:
    """
    Plot multinomial (e.g., donor/acceptor usage) - one subplot per category.

    Y-axis: Proportion for each category
    Layout:
    - If show_correction='uncorrected' or 'corrected': K columns (one per category), 1 row
    - If show_correction='both': 2 columns (uncorrected left, corrected right), K rows (one per category)

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        **Note**: Not yet fully implemented for multinomial - will issue warning

    Notes
    -----
    Technical correction for multinomial uses alpha_y_add (additive on LOGIT scale).
    logits_corrected = logits - alpha_y_add
    proportions_corrected = softmax(logits_corrected)
    """
    if show_ntc_gradient:
        warnings.warn("NTC gradient coloring not yet implemented for multinomial distributions - using standard colors")

    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts: shape (cells, K) for this feature
    if modality.counts.ndim == 3:
        counts_3d = modality.counts[feature_idx, :, :]  # (cells, K)
    else:
        raise ValueError(f"Expected 3D counts for multinomial, got {modality.counts.ndim}D")

    K = counts_3d.shape[1]

    # Align cells between model.meta and modality
    # For multinomial, we need to align the first dimension (cells) of the 3D array
    # We'll use the total counts across categories for alignment
    total_counts_all = counts_3d.sum(axis=1)
    x_true_aligned, total_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, total_counts_all, subset_mask
    )

    # Now we need to align the full 3D counts array using the same cell mapping
    # Get the cell mapping
    if 'cell' in model.meta.columns:
        model_cells = model.meta['cell'].values
    else:
        model_cells = model.meta.index.values
    modality_cells = modality.cell_names
    modality_mask = np.isin(model_cells, modality_cells)
    if subset_mask is not None:
        final_mask = modality_mask & subset_mask
    else:
        final_mask = modality_mask
    final_cells = model_cells[final_mask]
    modality_cell_to_idx = {cell: idx for idx, cell in enumerate(modality_cells)}
    y_indices = np.array([modality_cell_to_idx[cell] for cell in final_cells])
    counts_3d = counts_3d[y_indices, :]  # Align 3D array

    # Get category labels if available in feature_meta
    category_labels = None
    if hasattr(modality, 'feature_meta') and modality.feature_meta is not None:
        if 'category_labels' in modality.feature_meta.columns:
            category_labels = modality.feature_meta.loc[feature_idx, 'category_labels']
            if category_labels is not None and len(category_labels) != K:
                warnings.warn(f"category_labels length ({len(category_labels)}) doesn't match K ({K}) - ignoring labels")
                category_labels = None

    # Identify non-zero categories (skip padded zeros and empty labels)
    # A category is non-zero if it has any counts across all cells AND has a non-empty label
    non_zero_cats = []
    for k in range(K):
        has_counts = counts_3d[:, k].sum() > 0
        has_label = True if category_labels is None else (k < len(category_labels) and category_labels[k] != "")
        if has_counts and has_label:
            non_zero_cats.append(k)

    if len(non_zero_cats) == 0:
        raise ValueError(f"No non-zero categories found for feature '{feature}'")

    K_plot = len(non_zero_cats)  # Only plot non-zero categories

    # Filter by min_counts (total across categories)
    total_counts = counts_3d.sum(axis=1)
    valid_mask = total_counts >= min_counts

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true_aligned[valid_mask],
        'technical_group_code': meta_aligned['technical_group_code'].values[valid_mask],
        'target': meta_aligned['target'].values[valid_mask]
    })

    # Store raw counts for each category (needed for binning)
    counts_filtered = counts_3d[valid_mask, :]  # (n_cells_filtered, K)
    total_filtered = total_counts[valid_mask]

    # Filter valid x_true
    valid = df['x_true'] > 0
    df = df[valid].copy()
    counts_filtered = counts_filtered[valid, :]
    total_filtered = total_filtered[valid]

    if len(df) == 0:
        raise ValueError(f"No data remaining after filtering (min_counts={min_counts})")

    # Check technical correction availability
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            # Set the attribute for future use
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from posterior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create figure
    # Layout: if show_correction='both', then K_plot rows × 2 columns, else 1 row × K_plot columns
    if show_correction == 'both':
        if figsize is None:
            figsize = (12, 3 * K_plot)  # 2 columns, K_plot rows
        fig, axes = plt.subplots(K_plot, 2, figsize=figsize, squeeze=False,
                                 constrained_layout=True)
    else:
        if figsize is None:
            figsize = (4 * K_plot, 5)  # K_plot columns, 1 row
        fig, axes_row = plt.subplots(1, K_plot, figsize=figsize, squeeze=False,
                                     constrained_layout=True)
        axes = axes_row  # Shape: (1, K_plot)

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)
    group_codes   = np.sort(df['technical_group_code'].unique())

    # Plot each non-zero category
    for plot_idx, k in enumerate(non_zero_cats):
        # Get label for this category
        if category_labels is not None:
            cat_label = category_labels[k]
        else:
            cat_label = f"Category {k}"

        if show_correction == 'both':
            # Row plot_idx, columns 0 (uncorrected) and 1 (corrected)
            for col_idx, corrected in enumerate([False, True]):
                ax = axes[plot_idx, col_idx]

                for idx, group_code in enumerate(group_codes):
                    group_code = int(group_code)
                    group_label = code_to_label[group_code]
                    color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                    # Get data for this group
                    group_mask = df['technical_group_code'] == group_code
                    if group_mask.sum() == 0:
                        continue

                    x_group = df.loc[group_mask, 'x_true'].values
                    counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                    totals_group = total_filtered[group_mask]      # (n_cells_group,)

                    # Bin counts using k-NN
                    knn = _knn_k(len(x_group), window)
                    x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                        x_group, counts_group, totals_group, knn
                    )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                    # Compute proportions from binned counts
                    epsilon = 1e-10
                    with np.errstate(divide='ignore', invalid='ignore'):
                        props_binned = np.where(totals_binned[:, None] > epsilon,
                                                counts_binned / totals_binned[:, None],
                                                1.0 / K)

                    if corrected and has_technical_fit:
                        # Compute zero-category mask
                        zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                        # Apply correction to binned proportions
                        alpha_y_add = modality.alpha_y_prefit_add
                        props_corrected = _multinomial_correct_binned_probs(
                            props_binned=props_binned,
                            alpha_y_add=alpha_y_add,
                            group_code=group_code,
                            feature_idx=feature_idx,
                            zero_cat_mask=zero_cat_mask
                        )  # (n_bins, K)
                        y_smooth = props_corrected[:, k]
                    else:
                        y_smooth = props_binned[:, k]

                    # Plot (no additional smoothing needed - already binned)
                    ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                           label=group_label if plot_idx == 0 else None)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(f'Proportion')
                title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                ax.set_title(f"{cat_label}{title_suffix}")
                if plot_idx == 0:
                    ax.legend(frameon=False, loc='upper right')
        else:
            # Single row, column plot_idx
            ax = axes[0, plot_idx]
            corrected = (show_correction == 'corrected')

            for idx, group_code in enumerate(group_codes):
                group_code = int(group_code)
                group_label = code_to_label[group_code]
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                # Get data for this group
                group_mask = df['technical_group_code'] == group_code
                if group_mask.sum() == 0:
                    continue

                x_group = df.loc[group_mask, 'x_true'].values
                counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                totals_group = total_filtered[group_mask]      # (n_cells_group,)

                # Bin counts using k-NN
                knn = _knn_k(len(x_group), window)
                x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                    x_group, counts_group, totals_group, knn
                )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                # Compute proportions from binned counts
                epsilon = 1e-10
                with np.errstate(divide='ignore', invalid='ignore'):
                    props_binned = np.where(totals_binned[:, None] > epsilon,
                                            counts_binned / totals_binned[:, None],
                                            1.0 / K)

                if corrected and has_technical_fit:
                    # Compute zero-category mask
                    zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                    # Apply correction to binned proportions
                    alpha_y_add = modality.alpha_y_prefit_add
                    props_corrected = _multinomial_correct_binned_probs(
                        props_binned=props_binned,
                        alpha_y_add=alpha_y_add,
                        group_code=group_code,
                        feature_idx=feature_idx,
                        zero_cat_mask=zero_cat_mask
                    )  # (n_bins, K)
                    y_smooth = props_corrected[:, k]
                else:
                    y_smooth = props_binned[:, k]

                # Plot (no additional smoothing needed - already binned)
                ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                       label=group_label if plot_idx == 0 else None)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'Proportion')
            title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
            ax.set_title(f"{cat_label}{title_suffix}")
            if plot_idx == 0:
                ax.legend(frameon=False, loc='upper right')

    plt.suptitle(f"{model.cis_gene} → {feature} (min_counts={min_counts})")
    return fig


def plot_normal_xy(
    model,
    feature: str,
    modality,
    x_true: np.ndarray,
    window: int,
    show_correction: str,
    color_palette: Dict[str, str],
    show_trans_function: bool,
    show_ntc_gradient: bool = False,
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot normal or studentT distribution (continuous scores like SpliZ).

    Y-axis: Raw value

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
    """
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get values
    if modality.cells_axis == 1:
        y_vals = modality.counts[feature_idx, :]
    else:
        y_vals = modality.counts[:, feature_idx]

    # Align cells between model.meta and modality
    x_true_aligned, y_vals_aligned, meta_aligned = _align_cells_to_modality(
        model, modality, x_true, y_vals, subset_mask
    )

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true_aligned,
        'y_val': y_vals_aligned,
        'technical_group_code': meta_aligned['technical_group_code'].values,
        'target': meta_aligned['target'].values
    })

    # Filter valid
    valid = (df['x_true'] > 0) & np.isfinite(df['y_val'])
    df = df[valid].copy()

    # Check technical correction
    # For normal distribution, check alpha_y_prefit_add (additive correction)
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            # Set the attribute for future use
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from poster ior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create axes
    if ax is None:
        if show_correction == 'both':
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            axes = [ax]
    else:
        # Single axis provided - cannot show both corrected and uncorrected
        if show_correction == 'both':
            warnings.warn("show_correction='both' not supported when ax is provided (multi-panel mode). Showing uncorrected only.")
            show_correction = 'uncorrected'
        axes = [ax]

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)
    group_codes   = np.sort(df['technical_group_code'].unique())

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormaps for NTC gradient (per technical group)
    # Each group gets white → group_color gradient
    group_cmaps = {}
    if show_ntc_gradient:
        for group_label in group_labels:
            base_color = color_palette.get(group_label, f'C0')
            # Create colormap: white (low color value) → group color (high color value)
            group_cmaps[group_label] = LinearSegmentedColormap.from_list(
                f"white_{group_label}",
                ["white", base_color]
            )

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added
        
        for idx, group_code in enumerate(group_codes):
            group_code = int(group_code)
            group_label = code_to_label[group_code]
            color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
            df_group = df[df['technical_group_code'] == group_code].copy()

            if len(df_group) == 0:
                continue

            y_plot = df_group['y_val'].values

            if corrected and has_technical_fit:
                # Apply additive correction (alpha_y_add for normal)
                if hasattr(modality, 'alpha_y_prefit_add'):
                    alpha_y_add = modality.alpha_y_prefit_add
                    if alpha_y_add.ndim == 3:
                        correction = _to_scalar(alpha_y_add[:, group_code, feature_idx].mean())
                    else:
                        correction = _to_scalar(alpha_y_add[group_code, feature_idx])
                    y_plot = y_plot - correction

            # Get is_ntc for this group
            is_ntc_group = is_ntc[df_group.index].values

            # k-NN smoothing
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient and not corrected:
                # Smoothing with NTC tracking
                x_smooth, y_smooth, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_plot,
                    k,
                    is_ntc=is_ntc_group
                )

                # Use per-group gradient coloring (white → group color)
                # Color value = 1 - ntc_prop: high NTC → 0 → white, low NTC → 1 → group color
                group_cmap = group_cmaps.get(group_label, plt.cm.gray)
                plot_colored_line(
                    x=np.log2(x_smooth),
                    y=y_smooth,
                    color_values=1 - ntc_prop,  # Darker (group color) = fewer NTCs
                    cmap=group_cmap,
                    ax=ax_plot,
                    linewidth=2
                )

                # Add dummy invisible line for legend label (using base group color)
                ax_plot.plot([], [], color=color, linewidth=2, label=group_label)

                # Add colorbar (once per axis) - use grayscale to show NTC gradient
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    cmap_gray = LinearSegmentedColormap.from_list("gray_gradient", ["white", "black"])
                    sm = ScalarMappable(cmap=cmap_gray, norm=Normalize(0, 1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax_plot)
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                # Standard smoothing without NTC tracking
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_plot, k)

                # Use standard coloring
                ax_plot.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        # Show on corrected plot only (trans model was fitted on corrected data)
        if show_trans_function and corrected:
            x_range = np.linspace(x_true.min(), x_true.max(), 100)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

            if y_pred is not None:
                ax_plot.plot(np.log2(x_range), y_pred,
                           color='blue', linestyle='--', linewidth=2,
                           label='Fitted Trans Function')

        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel('Value')
        title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
        ax_plot.set_title(f"{model.cis_gene} → {feature}{title_suffix}")
        ax_plot.legend(frameon=False)

    # Plot
    if show_correction == 'both':
        _plot_one(axes[0], corrected=False)
        _plot_one(axes[1], corrected=True)
    elif show_correction == 'corrected':
        _plot_one(axes[0], corrected=True)
    else:
        _plot_one(axes[0], corrected=False)

    return axes[0] if len(axes) == 1 else axes


# ============================================================================
# Multi-feature multinomial plotting
# ============================================================================

def _plot_multinomial_multifeature(
    model,
    feature_indices: List[int],
    feature_names: List[str],
    gene_name: str,
    modality,
    x_true: np.ndarray,
    window: int,
    min_counts: int,
    show_correction: str,
    color_palette: Dict[str, str],
    xlabel: str,
    figsize: Optional[Tuple[int, int]] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot multiple multinomial features (e.g., multiple donor sites for one gene).

    Layout:
    - Single correction: n_features rows × K columns (dimensions as columns, features as rows)
    - Both corrections: (n_features × K) rows × 2 columns (dimensions in rows, corrections as columns)

    Parameters
    ----------
    model : bayesDREAM
        Fitted model
    feature_indices : List[int]
        Integer indices of features to plot
    feature_names : List[str]
        Names of features to plot
    gene_name : str
        Gene name (for title)
    modality : Modality
        Multinomial modality
    x_true : np.ndarray
        Cis expression values
    window : int
        k-NN window size
    min_counts : int
        Minimum total counts filter
    show_correction : str
        'uncorrected', 'corrected', or 'both'
    color_palette : Dict[str, str]
        Technical group colors
    xlabel : str
        X-axis label
    figsize : tuple, optional
        Figure size
    **kwargs
        Additional arguments

    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    n_features = len(feature_indices)

    # Get number of categories (K) for each feature and identify non-zero categories
    # IMPORTANT: Must compute non-zero categories AFTER subsetting to get correct max_non_zero
    Ks = []
    non_zero_cats_per_feature = []

    for feat_idx in feature_indices:
        if modality.counts.ndim == 3:
            counts_3d = modality.counts[feat_idx, :, :]  # (cells, K)

            # Apply subset mask if provided (to get correct max_non_zero for figure sizing)
            if subset_mask is not None:
                counts_3d = counts_3d[subset_mask, :]

            K = counts_3d.shape[1]
            Ks.append(K)

            # Get category labels if available for this feature
            category_labels = None
            if hasattr(modality, 'feature_meta') and modality.feature_meta is not None:
                if 'category_labels' in modality.feature_meta.columns:
                    category_labels = modality.feature_meta.loc[feat_idx, 'category_labels']
                    if category_labels is not None and len(category_labels) != K:
                        category_labels = None

            # Identify non-zero categories AFTER subsetting (skip padded zeros and empty labels)
            non_zero_cats = []
            for k in range(K):
                has_counts = counts_3d[:, k].sum() > 0
                has_label = True if category_labels is None else (k < len(category_labels) and category_labels[k] != "")
                if has_counts and has_label:
                    non_zero_cats.append(k)

            non_zero_cats_per_feature.append(non_zero_cats)
        else:
            raise ValueError(f"Expected 3D counts for multinomial, got {modality.counts.ndim}D")

    # Check technical correction
    has_technical_fit = False
    if hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
        has_technical_fit = True
    elif hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
        if 'alpha_y_add' in modality.posterior_samples_technical:
            has_technical_fit = True
            modality.alpha_y_prefit_add = modality.posterior_samples_technical['alpha_y_add']
            print(f"[INFO] Set alpha_y_prefit_add from posterior_samples_technical for modality '{modality.name}'")

    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create figure layout using GridSpec to allocate rows based on actual categories per feature
    # This avoids blank space when features have different numbers of categories
    if show_correction == 'both':
        # Layout: each feature gets rows for its actual non-zero categories × 2 columns
        total_rows = sum(len(nz) for nz in non_zero_cats_per_feature)
        n_cols = 2
        if figsize is None:
            figsize = (12, 3 * total_rows)  # 3 inches per row
    else:
        # Layout: each feature gets 1 row × columns for its actual non-zero categories
        # Use GridSpec with variable column widths
        max_cats = max(len(nz) for nz in non_zero_cats_per_feature) if non_zero_cats_per_feature else 1
        total_rows = n_features
        n_cols = max_cats
        if figsize is None:
            figsize = (4 * max_cats, 3 * n_features)  # 4 inches per column, 3 inches per row

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(total_rows, n_cols)

    # Create axes dict to store subplot positions
    axes_dict = {}

    # Get technical group labels
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(model.meta['technical_group_code'].unique())

    # Track current row position for GridSpec
    current_row = 0

    # Plot each feature
    for feat_i, (feat_idx, feat_name) in enumerate(zip(feature_indices, feature_names)):
        K = Ks[feat_i]
        non_zero_cats = non_zero_cats_per_feature[feat_i]

        if len(non_zero_cats) == 0:
            continue  # Skip features with no non-zero categories

        # Get counts for this feature
        counts_3d = modality.counts[feat_idx, :, :]  # (cells, K)

        # Apply subset mask if provided
        if subset_mask is not None:
            counts_3d = counts_3d[subset_mask, :]

        # Get category labels if available for this feature
        category_labels = None
        if hasattr(modality, 'feature_meta') and modality.feature_meta is not None:
            if 'category_labels' in modality.feature_meta.columns:
                category_labels = modality.feature_meta.loc[feat_idx, 'category_labels']
                if category_labels is not None and len(category_labels) != K:
                    warnings.warn(f"category_labels length ({len(category_labels)}) doesn't match K ({K}) for feature {feat_name} - ignoring labels")
                    category_labels = None

        # Use pre-computed non_zero_cats (already computed from subsetted data)
        # non_zero_cats was set at line 1897
        if len(non_zero_cats) == 0:
            continue  # Skip if no non-zero categories after subsetting

        # Filter by min_counts
        total_counts = counts_3d.sum(axis=1)
        valid_mask = total_counts >= min_counts

        # Build dataframe
        if subset_mask is not None:
            meta_subset = model.meta[subset_mask]
            df = pd.DataFrame({
                'x_true': x_true[valid_mask],
                'technical_group_code': meta_subset['technical_group_code'].values[valid_mask],
                'target': meta_subset['target'].values[valid_mask]
            })
        else:
            df = pd.DataFrame({
                'x_true': x_true[valid_mask],
                'technical_group_code': model.meta['technical_group_code'].values[valid_mask],
                'target': model.meta['target'].values[valid_mask]
            })

        # Keep raw counts for binning (don't compute proportions per-cell)
        counts_filtered = counts_3d[valid_mask, :]
        total_filtered = total_counts[valid_mask]

        # Filter valid x_true
        valid = df['x_true'] > 0
        df = df[valid].copy()
        counts_filtered = counts_filtered[valid, :]
        total_filtered = total_filtered[valid]

        if len(df) == 0:
            # Skip to next feature if no data after filtering
            # Note: we still need to advance current_row for 'both' mode
            if show_correction == 'both':
                current_row += len(non_zero_cats)
            continue

        # Map technical_group_code → label for THIS feature (after all filters)
        code_to_label = _labels_by_code_for_df(model, df)
        group_codes   = np.sort(df['technical_group_code'].unique())
        group_labels  = [code_to_label[int(c)] for c in group_codes]

        # Plot each non-zero category
        for cat_plot_idx, k in enumerate(non_zero_cats):
            # Get label for this category
            if category_labels is not None:
                cat_label = category_labels[k]
            else:
                cat_label = f"Cat{k}"

            if show_correction == 'both':
                # Create axes for this category row
                row = current_row + cat_plot_idx
                for col_idx, corrected in enumerate([False, True]):
                    ax = fig.add_subplot(gs[row, col_idx])

        
                    for idx, group_code in enumerate(group_codes):
                        group_code = int(group_code)
                        group_label = code_to_label[group_code]
                        color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                        # Get data for this group
                        group_mask = df['technical_group_code'] == group_code
                        if group_mask.sum() == 0:
                            continue

                        x_group = df.loc[group_mask, 'x_true'].values
                        counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                        totals_group = total_filtered[group_mask]      # (n_cells_group,)

                        # Bin counts using k-NN
                        knn = _knn_k(len(x_group), window)
                        x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                            x_group, counts_group, totals_group, knn
                        )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                        # Compute proportions from binned counts
                        epsilon = 1e-10
                        with np.errstate(divide='ignore', invalid='ignore'):
                            props_binned = np.where(totals_binned[:, None] > epsilon,
                                                    counts_binned / totals_binned[:, None],
                                                    1.0 / K)

                        if corrected and has_technical_fit:
                            # Compute zero-category mask
                            zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                            # Apply correction to binned proportions
                            alpha_y_add = modality.alpha_y_prefit_add
                            props_corrected = _multinomial_correct_binned_probs(
                                props_binned=props_binned,
                                alpha_y_add=alpha_y_add,
                                group_code=group_code,
                                feature_idx=feat_idx,
                                zero_cat_mask=zero_cat_mask
                            )  # (n_bins, K)
                            y_smooth = props_corrected[:, k]
                        else:
                            y_smooth = props_binned[:, k]

                        # Plot (no additional smoothing needed - already binned)
                        ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                               label=group_label if cat_plot_idx == 0 and feat_i == 0 else None)

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(f'Proportion')
                    title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                    ax.set_title(f"{feat_name[:20]}... {cat_label}{title_suffix}", fontsize=9)
                    if cat_plot_idx == 0 and feat_i == 0:
                        ax.legend(frameon=False, loc='upper right', fontsize=8)
            else:
                # Create ax for this category column
                ax = fig.add_subplot(gs[current_row, cat_plot_idx])
                corrected = (show_correction == 'corrected')
        
                for idx, group_code in enumerate(group_codes):
                    group_code = int(group_code)
                    group_label = code_to_label[group_code]
                    color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)

                    # Get data for this group
                    group_mask = df['technical_group_code'] == group_code
                    if group_mask.sum() == 0:
                        continue

                    x_group = df.loc[group_mask, 'x_true'].values
                    counts_group = counts_filtered[group_mask, :]  # (n_cells_group, K)
                    totals_group = total_filtered[group_mask]      # (n_cells_group,)

                    # Bin counts using k-NN
                    knn = _knn_k(len(x_group), window)
                    x_smooth, counts_binned, totals_binned = _smooth_knn_counts(
                        x_group, counts_group, totals_group, knn
                    )  # counts_binned: (n_bins, K), totals_binned: (n_bins,)

                    # Compute proportions from binned counts
                    epsilon = 1e-10
                    with np.errstate(divide='ignore', invalid='ignore'):
                        props_binned = np.where(totals_binned[:, None] > epsilon,
                                                counts_binned / totals_binned[:, None],
                                                1.0 / K)

                    if corrected and has_technical_fit:
                        # Compute zero-category mask
                        zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)

                        # Apply correction to binned proportions
                        alpha_y_add = modality.alpha_y_prefit_add
                        props_corrected = _multinomial_correct_binned_probs(
                            props_binned=props_binned,
                            alpha_y_add=alpha_y_add,
                            group_code=group_code,
                            feature_idx=feat_idx,
                            zero_cat_mask=zero_cat_mask
                        )  # (n_bins, K)
                        y_smooth = props_corrected[:, k]
                    else:
                        y_smooth = props_binned[:, k]

                    # Plot (no additional smoothing needed - already binned)
                    ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                           label=group_label if cat_plot_idx == 0 and feat_i == 0 else None)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(f'Proportion')
                title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                ax.set_title(f"{feat_name[:20]}... {cat_label}{title_suffix}", fontsize=9)
                if cat_plot_idx == 0 and feat_i == 0:
                    ax.legend(frameon=False, loc='upper right', fontsize=8)

        # Update current_row for next feature (only for 'both' mode)
        if show_correction == 'both':
            current_row += len(non_zero_cats)
        else:
            current_row += 1

    plt.suptitle(f"{model.cis_gene} → {gene_name} (gene, n={n_features} features, min_counts={min_counts})")
    return fig


# ============================================================================
# Unified plot_xy_data function
# ============================================================================

def plot_xy_data(
    model,
    feature: str,
    modality_name: Optional[str] = None,
    window: int = 100,
    show_correction: str = 'both',
    min_counts: int = 3,
    color_palette: Optional[Dict[str, str]] = None,
    show_hill_function: bool = True,
    show_ntc_gradient: bool = False,
    sum_factor_col: str = 'sum_factor',
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    src_barcodes: Optional[np.ndarray] = None,
    subset_meta: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[plt.Figure, plt.Axes]:
    """
    Plot raw x-y data showing relationship between cis gene expression and modality values.

    Requires x_true to be set (must run fit_cis() first).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model with x_true set
    feature : str
        Feature name (junction, donor, etc.) OR gene name.
        - If a specific feature name (e.g., 'chr1:999788:999865'), plots that feature
        - If a gene name (e.g., 'HES4'), plots all features for that gene in subplots
        - Requires modality to have gene information ('gene', 'gene_name', or 'gene_id' columns)
    modality_name : str, optional
        Modality name (default: primary modality)
    window : int
        k-NN window size for smoothing (default: 100 cells)
    show_correction : str
        'uncorrected': no technical correction
        'corrected': apply alpha_y technical correction
        'both': show both side-by-side (default)
    min_counts : int
        Minimum denominator for binomial (default: 3)
        Minimum total counts for multinomial (default: 3)
    color_palette : dict, optional
        Custom colors for technical groups
        Example: {'CRISPRa': 'crimson', 'CRISPRi': 'dodgerblue'}
    show_hill_function : bool
        Overlay fitted trans function if trans model fitted (all distributions, default: True)
        Works with all function types: additive_hill, single_hill, polynomial
        Automatically detects function type from posterior_samples_trans
    show_ntc_gradient : bool
        Color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
        Only applies to uncorrected plots
        Fully implemented for: negbinom, binomial, normal, studentt
        Not yet implemented for: multinomial (will issue warning)
    sum_factor_col : str
        Column name in model.meta for sum factors (default: 'sum_factor')
        Can be 'sum_factor', 'sum_factor_adj', or any other sum factor column
        Only used for negbinom distribution (gene expression)
    xlabel : str
        X-axis label (default: "log2(x_true)")
    figsize : tuple, optional
        Figure size (auto-sized if None)
    src_barcodes : np.ndarray, optional
        Source barcode order if x_true not in model.meta order
    subset_meta : dict, optional
        Subset cells by metadata columns. Dictionary of {column: value} pairs.
        Example: {'target': 'ntc'} - plot only NTC cells
        Example: {'cell_line': 'CRISPRi'} - plot only CRISPRi cells
        Example: {'lane': 'L1', 'cell_line': 'CRISPRa'} - plot L1 lane CRISPRa cells
        Multiple conditions are combined with AND logic.
    **kwargs
        Additional plotting arguments

    Returns
    -------
    plt.Figure or plt.Axes
        Matplotlib figure or axes object

    Raises
    ------
    ValueError
        If x_true not set (must run fit_cis first)
        If feature not found in modality
        If show_correction='corrected' but fit_technical not run

    Warnings
    --------
    If fit_technical not run for modality and show_correction='corrected',
    warns and plots uncorrected only.

    Examples
    --------
    >>> # Plot single gene with Hill function
    >>> model.plot_xy_data('TET2', window=100, show_hill_function=True)
    >>>
    >>> # Plot specific splice junction with min_counts filter
    >>> model.plot_xy_data('chr1:12345:67890:+', modality_name='splicing_sj',
    ...                     min_counts=5)
    >>>
    >>> # Plot all splice junctions for a gene (creates multi-panel figure)
    >>> model.plot_xy_data('HES4', modality_name='splicing_sj')
    >>>
    >>> # Plot with custom colors
    >>> model.plot_xy_data('GFI1B', color_palette={'CRISPRa': 'red', 'CRISPRi': 'blue'})
    >>>
    >>> # Show both corrected and uncorrected (default)
    >>> model.plot_xy_data('TET2', show_correction='both')
    >>>
    >>> # Plot with NTC gradient coloring
    >>> model.plot_xy_data('TET2', show_ntc_gradient=True, show_correction='uncorrected')
    >>>
    >>> # Plot only NTC cells
    >>> model.plot_xy_data('TET2', subset_meta={'target': 'ntc'})
    >>>
    >>> # Plot only CRISPRi cells
    >>> model.plot_xy_data('GFI1B', subset_meta={'cell_line': 'CRISPRi'})
    """
    # Check x_true is set
    if not hasattr(model, 'x_true') or model.x_true is None:
        raise ValueError(
            "x_true not set. Must run fit_cis() before plotting x-y data.\n"
            "Example: model.fit_cis(sum_factor_col='sum_factor')"
        )

    # Check technical_group_code is set
    if 'technical_group_code' not in model.meta.columns:
        raise ValueError(
            "technical_group_code not set. Must run set_technical_groups() first.\n"
            "Example: model.set_technical_groups(['cell_line'])"
        )

    # Get x_true
    x2d = _to_2d(model.x_true)
    x_true = x2d.mean(axis=0) if x2d.ndim == 2 else x2d.ravel()

    # Reorder if needed
    if src_barcodes is not None:
        cell_barcodes = model.meta['cell'].values if 'cell' in model.meta.columns else model.meta.index.values
        x_true = reorder_xtrue_by_barcode(x_true, src_barcodes, cell_barcodes)

    # Apply metadata subsetting
    subset_mask = None
    if subset_meta is not None:
        # Create mask based on subset_meta conditions (AND logic)
        subset_mask = np.ones(len(model.meta), dtype=bool)
        for col, value in subset_meta.items():
            if col not in model.meta.columns:
                raise ValueError(f"Column '{col}' not found in model.meta. Available columns: {list(model.meta.columns)}")
            subset_mask &= (model.meta[col] == value).values

        n_cells_before = len(model.meta)
        n_cells_after = subset_mask.sum()
        if n_cells_after == 0:
            raise ValueError(f"No cells match subset_meta criteria: {subset_meta}")

        print(f"[SUBSET] Filtering {n_cells_before} → {n_cells_after} cells based on {subset_meta}")

        # NOTE: Don't subset x_true here - let _align_cells_to_modality() handle it
        # (subsetting here causes IndexError when mask is applied again in alignment)

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)

    # Get color palette
    if color_palette is None:
        group_labels = get_technical_group_labels(model)
        color_palette = get_default_color_palette(group_labels)

    # Resolve feature(s) - could be single feature or gene name
    feature_indices, feature_names_resolved, is_gene = _resolve_features(feature, modality)

    # If multiple features (gene input), create multi-panel figure
    if is_gene and len(feature_indices) > 1:
        n_features = len(feature_indices)
        distribution = modality.distribution

        # Special handling for multinomial - needs K subplots per feature
        if distribution == 'multinomial':
            # Subset x_true if needed (multinomial path doesn't use _align_cells_to_modality)
            x_true_plot = x_true[subset_mask] if subset_mask is not None else x_true

            return _plot_multinomial_multifeature(
                model=model,
                feature_indices=feature_indices,
                feature_names=feature_names_resolved,
                gene_name=feature,
                modality=modality,
                x_true=x_true_plot,
                window=window,
                min_counts=min_counts,
                show_correction=show_correction,
                color_palette=color_palette,
                xlabel=xlabel,
                figsize=figsize,
                subset_mask=subset_mask,
                **kwargs
            )

        # Standard multi-feature plotting for 2D distributions
        # Layout: features in rows, uncorrected (left) and corrected (right) columns
        n_rows = n_features
        n_cols = 2  # Left = uncorrected, Right = corrected

        if figsize is None:
            figsize = (12, 3 * n_rows)  # 12 inches wide (6 per subplot), 3 inches per row

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False,
                                 constrained_layout=True)

        # Plot each feature (one row per feature)
        for i, (feat_idx, feat_name) in enumerate(zip(feature_indices, feature_names_resolved)):
            # Plot uncorrected (left column)
            if distribution == 'negbinom':
                plot_negbinom_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='uncorrected',
                    color_palette=color_palette, show_hill_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, sum_factor_col=sum_factor_col,
                    xlabel=xlabel, ax=axes[i, 0], subset_mask=subset_mask, **kwargs
                )
            elif distribution == 'binomial':
                plot_binomial_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='uncorrected',
                    min_counts=min_counts, color_palette=color_palette,
                    show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=axes[i, 0],
                    subset_mask=subset_mask, **kwargs
                )
            elif distribution in ('normal', 'studentt'):
                plot_normal_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='uncorrected',
                    color_palette=color_palette, show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=axes[i, 0],
                    subset_mask=subset_mask, **kwargs
                )
            else:
                axes[i, 0].text(0.5, 0.5, f"Multi-panel not supported for {distribution}",
                               ha='center', va='center', transform=axes[i, 0].transAxes)

            # Plot corrected (right column)
            if distribution == 'negbinom':
                plot_negbinom_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='corrected',
                    color_palette=color_palette, show_hill_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, sum_factor_col=sum_factor_col,
                    xlabel=xlabel, ax=axes[i, 1], subset_mask=subset_mask, **kwargs
                )
            elif distribution == 'binomial':
                plot_binomial_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='corrected',
                    min_counts=min_counts, color_palette=color_palette,
                    show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=axes[i, 1],
                    subset_mask=subset_mask, **kwargs
                )
            elif distribution in ('normal', 'studentt'):
                plot_normal_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='corrected',
                    color_palette=color_palette, show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=axes[i, 1],
                    subset_mask=subset_mask, **kwargs
                )
            else:
                axes[i, 1].text(0.5, 0.5, f"Multi-panel not supported for {distribution}",
                               ha='center', va='center', transform=axes[i, 1].transAxes)

        plt.suptitle(f"{model.cis_gene} → {feature} (gene, n={n_features} features)")
        return fig

    # Single feature - use original code path
    feature = feature_names_resolved[0]  # Use resolved feature name

    # Route to distribution-specific plotting function
    distribution = modality.distribution

    if distribution == 'negbinom':
        return plot_negbinom_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            show_correction=show_correction,
            color_palette=color_palette,
            show_hill_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            sum_factor_col=sum_factor_col,
            xlabel=xlabel,
            subset_mask=subset_mask,
            **kwargs
        )

    elif distribution == 'binomial':
        return plot_binomial_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            min_counts=min_counts,
            color_palette=color_palette,
            show_trans_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            xlabel=xlabel,
            subset_mask=subset_mask,
            **kwargs
        )

    elif distribution == 'multinomial':
        return plot_multinomial_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            min_counts=min_counts,
            show_correction=show_correction,
            color_palette=color_palette,
            show_trans_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            xlabel=xlabel,
            figsize=figsize,
            subset_mask=subset_mask,
            **kwargs
        )

    elif distribution in ('normal', 'studentt'):
        return plot_normal_xy(
            model=model,
            feature=feature,
            modality=modality,
            x_true=x_true,
            window=window,
            show_correction=show_correction,
            color_palette=color_palette,
            show_trans_function=show_hill_function,
            show_ntc_gradient=show_ntc_gradient,
            xlabel=xlabel,
            subset_mask=subset_mask,
            **kwargs
        )

    else:
        raise ValueError(f"Plotting not implemented for distribution '{distribution}'")

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

def _multinomial_correct_mean_probs(
    props_group: np.ndarray,          # (n, K) proportions for this group
    alpha_y_add,                      # [S, C, T, K] or [C, T, K]
    group_code: int,
    feature_idx: int,
    zero_cat_mask: np.ndarray         # (K,) True where category is globally zero
) -> np.ndarray:
    """
    Apply multinomial technical correction the *same way as the model*:
    For each posterior sample s: softmax(log(P) - alpha_s), then average across s.
    Enforces a hard zero mask and renormalizes.

    Returns
    -------
    (n, K) corrected proportions
    """
    epsilon = 1e-6
    # Clip and log
    P = np.clip(props_group, epsilon, 1 - epsilon)   # (n, K)
    logP = np.log(P)                                 # (n, K)

    # Pull alpha with explicit samples dim
    if alpha_y_add.ndim == 4:        # [S, C, T, K]
        A = alpha_y_add[:, group_code, feature_idx, :]          # (S, K)
    elif alpha_y_add.ndim == 3:      # [C, T, K] -> add S=1
        A = alpha_y_add[None, group_code, feature_idx, :]       # (1, K)
    else:
        raise ValueError(f"Unexpected alpha_y_add shape: {alpha_y_add.shape}")

    if hasattr(A, "detach"):
        A = A.detach().cpu().numpy()

    # Broadcast to (S, n, K)
    logP_S = logP[None, :, :]
    A_S    = A[:, None, :]

    logits = logP_S - A_S                                 # (S, n, K)

    # Hard-mask zero categories
    logits[:, :, zero_cat_mask] = -np.inf

    # Stable softmax with mask
    m = np.nanmax(logits, axis=-1, keepdims=True)
    exps = np.exp(logits - m)
    exps[:, :, zero_cat_mask] = 0.0
    Z = exps.sum(axis=-1, keepdims=True)
    Z[Z == 0] = 1.0
    P_corr_S = exps / Z                                    # (S, n, K)

    # Posterior mean in probability space
    return P_corr_S.mean(axis=0)                           # (n, K)



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
    # Check if trans model fitted
    if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
        return None

    posterior = model.posterior_samples_trans

    # Get baseline A (present in all function types)
    if 'A' not in posterior:
        return None

    A_samples = posterior['A']

    # Get posterior dimensions
    if hasattr(A_samples, 'mean'):
        A_mean = A_samples.mean(dim=0)
        n_genes_posterior = A_mean.shape[0]
    else:
        A_mean = A_samples.mean(axis=0)
        n_genes_posterior = A_mean.shape[0]

    # Get trans_genes list (should match posterior dimensions)
    trans_genes = model.trans_genes if hasattr(model, 'trans_genes') else []
    n_genes_list = len(trans_genes)

    # Check dimension consistency BEFORE using trans_genes for indexing
    if n_genes_posterior != n_genes_list:
        # Mismatch - cannot use trans_genes for indexing
        # This happens when posterior was fitted on a subset of genes
        return None

    # Now safe to check if feature is in trans_genes and get its index
    if feature not in trans_genes:
        return None

    gene_idx = trans_genes.index(feature)
    A = A_mean[gene_idx].item() if hasattr(A_mean, 'item') else A_mean[gene_idx]

    # Determine function type from available parameters
    if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
        # ===== ADDITIVE HILL =====
        try:
            # Extract parameters
            alpha = posterior['alpha'].mean(dim=0)[gene_idx].item() if hasattr(posterior['alpha'], 'mean') else posterior['alpha'].mean(axis=0)[gene_idx]
            beta = posterior['beta'].mean(dim=0)[gene_idx].item() if hasattr(posterior['beta'], 'mean') else posterior['beta'].mean(axis=0)[gene_idx]
            Vmax_a = posterior['Vmax_a'].mean(dim=0)[gene_idx].item() if hasattr(posterior['Vmax_a'], 'mean') else posterior['Vmax_a'].mean(axis=0)[gene_idx]
            Vmax_b = posterior['Vmax_b'].mean(dim=0)[gene_idx].item() if hasattr(posterior['Vmax_b'], 'mean') else posterior['Vmax_b'].mean(axis=0)[gene_idx]
            K_a = posterior['K_a'].mean(dim=0)[gene_idx].item() if hasattr(posterior['K_a'], 'mean') else posterior['K_a'].mean(axis=0)[gene_idx]
            K_b = posterior['K_b'].mean(dim=0)[gene_idx].item() if hasattr(posterior['K_b'], 'mean') else posterior['K_b'].mean(axis=0)[gene_idx]
            n_a = posterior['n_a'].mean(dim=0)[gene_idx].item() if hasattr(posterior['n_a'], 'mean') else posterior['n_a'].mean(axis=0)[gene_idx]
            n_b = posterior['n_b'].mean(dim=0)[gene_idx].item() if hasattr(posterior['n_b'], 'mean') else posterior['n_b'].mean(axis=0)[gene_idx]

            # Compute Hill functions
            Hill_a = Hill_based_positive(x_range, Vmax=Vmax_a, A=0, K=K_a, n=n_a)
            Hill_b = Hill_based_positive(x_range, Vmax=Vmax_b, A=0, K=K_b, n=n_b)

            # Combined prediction
            y_pred = A + alpha * Hill_a + beta * Hill_b
            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    elif 'Vmax' in posterior and 'K' in posterior and 'n' in posterior:
        # ===== SINGLE HILL =====
        try:
            Vmax = posterior['Vmax'].mean(dim=0)[gene_idx].item() if hasattr(posterior['Vmax'], 'mean') else posterior['Vmax'].mean(axis=0)[gene_idx]
            K = posterior['K'].mean(dim=0)[gene_idx].item() if hasattr(posterior['K'], 'mean') else posterior['K'].mean(axis=0)[gene_idx]
            n = posterior['n'].mean(dim=0)[gene_idx].item() if hasattr(posterior['n'], 'mean') else posterior['n'].mean(axis=0)[gene_idx]

            # Compute Hill function
            y_pred = Hill_based_positive(x_range, Vmax=Vmax, A=A, K=K, n=n)
            return y_pred

        except (KeyError, IndexError, AttributeError):
            return None

    elif 'theta' in posterior:
        # ===== POLYNOMIAL OR THETA-BASED =====
        try:
            theta_samples = posterior['theta']
            # theta shape: (samples, genes, n_params)

            # Check if it's polynomial by number of parameters
            if hasattr(theta_samples, 'mean'):
                theta_mean = theta_samples.mean(dim=0)[gene_idx, :]  # (n_params,)
                theta_np = theta_mean.cpu().numpy() if hasattr(theta_mean, 'cpu') else np.array(theta_mean)
            else:
                theta_mean = theta_samples.mean(axis=0)[gene_idx, :]
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

    # Apply subset mask if provided
    if subset_mask is not None:
        y_obs = y_obs[subset_mask]

    # Build dataframe
    # Check if sum_factor_col exists
    if sum_factor_col not in model.meta.columns:
        raise ValueError(f"Sum factor column '{sum_factor_col}' not found in model.meta. "
                        f"Available columns: {list(model.meta.columns)}")

    # Get metadata arrays (already subsetted in plot_xy_data if subset_mask provided)
    if subset_mask is not None:
        meta_subset = model.meta[subset_mask]
        df = pd.DataFrame({
            'x_true': x_true,
            'y_obs': y_obs,
            'technical_group_code': meta_subset['technical_group_code'].values,
            'target': meta_subset['target'].values,
            'sum_factor': meta_subset[sum_factor_col].values
        })
    else:
        df = pd.DataFrame({
            'x_true': x_true,
            'y_obs': y_obs,
            'technical_group_code': model.meta['technical_group_code'].values,
            'target': model.meta['target'].values,
            'sum_factor': model.meta[sum_factor_col].values
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

            # k-NN smoothing
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient:
                # Smoothing with NTC tracking
                x_smooth, y_smooth, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_expr.values,
                    k,
                    is_ntc=is_ntc_group
                )

                # Use per-group gradient coloring (white → group color)
                # Color value = 1 - ntc_prop: high NTC → 0 → white, low NTC → 1 → group color
                group_cmap = group_cmaps.get(group_label, plt.cm.gray)
                plot_colored_line(
                    x=np.log2(x_smooth),
                    y=np.log2(y_smooth),
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
                # Standard smoothing without NTC tracking
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_expr.values, k)

                # Use standard coloring
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                ax_plot.plot(np.log2(x_smooth), np.log2(y_smooth), color=color, linewidth=2, label=group_label)

        # Trans function overlay (if trans model fitted)
        if show_hill_function and not corrected:
            x_range = np.linspace(x_true.min(), x_true.max(), 100)
            y_pred = predict_trans_function(model, feature, x_range, modality_name=None)

            if y_pred is not None and np.all(y_pred > 0):
                # Only plot if all predictions are positive (for log2 transform)
                ax_plot.plot(np.log2(x_range), np.log2(y_pred),
                            color='blue', linestyle='--', linewidth=2,
                            label='Fitted Trans Function')

                # Add baseline if available
                if hasattr(model, 'posterior_samples_trans') and 'A' in model.posterior_samples_trans:
                    A_samples = model.posterior_samples_trans['A']
                    trans_genes = model.trans_genes if hasattr(model, 'trans_genes') else []
                    if feature in trans_genes:
                        gene_idx = trans_genes.index(feature)
                        A = A_samples.mean(dim=0)[gene_idx].item() if hasattr(A_samples, 'mean') else A_samples.mean(axis=0)[gene_idx]
                        if A > 0:
                            ax_plot.axhline(np.log2(A), color='red', linestyle=':', linewidth=1, label='log2(A) baseline')

        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel('log2(Expression)')
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

    # Apply subset mask if provided
    if subset_mask is not None:
        counts = counts[subset_mask]
        denom = denom[subset_mask]

    # Filter by min_counts
    valid_mask = denom >= min_counts

    # Build dataframe
    if subset_mask is not None:
        meta_subset = model.meta[subset_mask]
        df = pd.DataFrame({
            'x_true': x_true[valid_mask],
            'counts': counts[valid_mask],
            'denominator': denom[valid_mask],
            'technical_group_code': meta_subset['technical_group_code'].values[valid_mask],
            'target': meta_subset['target'].values[valid_mask]
        })
    else:
        df = pd.DataFrame({
            'x_true': x_true[valid_mask],
            'counts': counts[valid_mask],
            'denominator': denom[valid_mask],
            'technical_group_code': model.meta['technical_group_code'].values[valid_mask],
            'target': model.meta['target'].values[valid_mask]
        })

    # Compute PSI (as percentage: 0-100 scale)
    df['PSI'] = (df['counts'] / df['denominator']) * 100.0

    # Filter valid
    valid = (df['x_true'] > 0) & np.isfinite(df['PSI'])
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
            df_group = df[df['technical_group_code'] == group_code].copy()

            if len(df_group) == 0:
                continue

            y_plot = df_group['PSI'].values

            if corrected and has_technical_fit:
                # Apply additive correction on LOGIT scale
                # 1. Convert PSI (percentage) to proportion [0, 1]
                epsilon = 1e-6  # Small constant to avoid log(0)
                p = np.clip(y_plot / 100.0, epsilon, 1 - epsilon)

                # 2. Calculate logit
                logit_p = np.log(p / (1 - p))

                # 3. Apply correction on logit scale
                alpha_y_add = modality.alpha_y_prefit_add
                if alpha_y_add.ndim == 3:
                    correction = _to_scalar(alpha_y_add[:, group_code, feature_idx].mean())
                else:
                    correction = _to_scalar(alpha_y_add[group_code, feature_idx])
                logit_corrected = logit_p - correction

                # 4. Convert back to proportion
                p_corrected = 1.0 / (1.0 + np.exp(-logit_corrected))

                # 5. Convert back to percentage
                y_plot = p_corrected * 100.0

            # Get is_ntc for this group
            is_ntc_group = is_ntc[df_group.index].values

            # k-NN smoothing
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient:
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
        if show_trans_function and not corrected:
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

    # Apply subset mask if provided
    if subset_mask is not None:
        counts_3d = counts_3d[subset_mask, :]

    K = counts_3d.shape[1]

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

    # Store raw counts for each category (needed for correction)
    counts_filtered = counts_3d[valid_mask, :]  # (n_cells_filtered, K)

    # Compute raw proportions
    total_filtered = total_counts[valid_mask]
    with np.errstate(divide='ignore', invalid='ignore'):
        props_raw = np.where(total_filtered[:, None] > 0,
                             counts_filtered / total_filtered[:, None],
                             1.0 / K)  # Uniform if no counts

    # Add raw proportions for each category
    for k in range(K):
        df[f'cat_{k}'] = props_raw[:, k]

    # Filter valid x_true
    valid = df['x_true'] > 0
    df = df[valid].copy()
    props_raw = props_raw[valid, :]

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
            figsize = (12, 4 * K_plot)  # 2 columns, K_plot rows
        fig, axes = plt.subplots(K_plot, 2, figsize=figsize, squeeze=False)
    else:
        if figsize is None:
            figsize = (4 * K_plot, 5)  # K_plot columns, 1 row
        fig, axes_row = plt.subplots(1, K_plot, figsize=figsize, squeeze=False)
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
                    df_group = df[df['technical_group_code'] == group_code].copy()

                    if len(df_group) == 0:
                        continue

                    # Get proportions for this group and category
                    group_mask = df['technical_group_code'] == group_code
                    props_group = props_raw[group_mask, :]  # Shape: (n_cells_in_group, K)

                    if corrected and has_technical_fit:
                        # ---- Compute zero-category mask from the data used on THIS axis ----
                        # counts_3d was already subset by valid_mask above
                        zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)
                        
                        # ---- Per-sample correction, averaged in probability space ----
                        alpha_y_add = modality.alpha_y_prefit_add
                        # Align indices explicitly to avoid accidental reindexing issues
                        df = df.reset_index(drop=True)
                        props_raw = np.asarray(props_raw)  # (n_cells_filtered, K), same length as df now
                        group_mask = (df['technical_group_code'] == group_code).values
                        props_group = props_raw[group_mask, :]  # (n_group_cells, K)
                        
                        props_corrected = _multinomial_correct_mean_probs(
                            props_group=props_group,
                            alpha_y_add=alpha_y_add,
                            group_code=group_code,
                            feature_idx=feature_idx,
                            zero_cat_mask=zero_cat_mask
                        )  # (n_group_cells, K)
                        
                        y_data = props_corrected[:, k]
                    else:
                        y_data = df_group[f'cat_{k}'].values

                    # k-NN smoothing
                    knn = _knn_k(len(df_group), window)
                    x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_data, knn)

                    # Plot
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
                df_group = df[df['technical_group_code'] == group_code].copy()

                if len(df_group) == 0:
                    continue

                # Get proportions for this group and category
                group_mask = df['technical_group_code'] == group_code
                props_group = props_raw[group_mask, :]  # Shape: (n_cells_in_group, K)

                if corrected and has_technical_fit:
                    # ---- Compute zero-category mask from the data used on THIS axis ----
                    # counts_3d was already subset by valid_mask above
                    zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)
                    
                    # ---- Per-sample correction, averaged in probability space ----
                    alpha_y_add = modality.alpha_y_prefit_add
                    # Align indices explicitly to avoid accidental reindexing issues
                    df = df.reset_index(drop=True)
                    props_raw = np.asarray(props_raw)  # (n_cells_filtered, K), same length as df now
                    group_mask = (df['technical_group_code'] == group_code).values
                    props_group = props_raw[group_mask, :]  # (n_group_cells, K)
                    
                    props_corrected = _multinomial_correct_mean_probs(
                        props_group=props_group,
                        alpha_y_add=alpha_y_add,
                        group_code=group_code,
                        feature_idx=feature_idx,
                        zero_cat_mask=zero_cat_mask
                    )  # (n_group_cells, K)
                    
                    y_data = props_corrected[:, k]
                else:
                    y_data = df_group[f'cat_{k}'].values

                # k-NN smoothing
                knn = _knn_k(len(df_group), window)
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_data, knn)

                # Plot
                ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                       label=group_label if plot_idx == 0 else None)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'Proportion')
            title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
            ax.set_title(f"{cat_label}{title_suffix}")
            if plot_idx == 0:
                ax.legend(frameon=False, loc='upper right')

    plt.suptitle(f"{model.cis_gene} → {feature} (min_counts={min_counts})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Leave space for suptitle

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
    Plot normal distribution (continuous scores like SpliZ).

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

    # Apply subset mask if provided
    if subset_mask is not None:
        y_vals = y_vals[subset_mask]

    # Build dataframe
    if subset_mask is not None:
        meta_subset = model.meta[subset_mask]
        df = pd.DataFrame({
            'x_true': x_true,
            'y_val': y_vals,
            'technical_group_code': meta_subset['technical_group_code'].values,
            'target': meta_subset['target'].values
        })
    else:
        df = pd.DataFrame({
            'x_true': x_true,
            'y_val': y_vals,
            'technical_group_code': model.meta['technical_group_code'].values,
            'target': model.meta['target'].values
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
        if show_trans_function and not corrected:
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


def plot_mvnormal_xy(
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
    figsize: Optional[Tuple[int, int]] = None,
    subset_mask: Optional[np.ndarray] = None,
    **kwargs
) -> Union[plt.Figure, List[plt.Axes]]:
    """
    Plot multivariate normal (e.g., SpliZVD) - one subplot per dimension.

    Y-axis: Value for each dimension
    Layout: One row with D subplots (one per dimension)
    If show_correction='both': Two rows × D subplots

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        **Note**: Not yet fully implemented for mvnormal - will issue warning
    """
    if show_ntc_gradient:
        warnings.warn("NTC gradient coloring not yet implemented for mvnormal distributions - using standard colors")
    # Get data
    feature_idx = _get_feature_index(feature, modality)
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get values: shape (cells, D) for this feature
    if modality.counts.ndim == 3:
        values_3d = modality.counts[feature_idx, :, :]  # (cells, D)
    else:
        raise ValueError(f"Expected 3D counts for mvnormal, got {modality.counts.ndim}D")

    # Apply subset mask if provided
    if subset_mask is not None:
        values_3d = values_3d[subset_mask, :]

    D = values_3d.shape[1]

    # Check technical correction
    has_technical_fit = modality.alpha_y_prefit is not None
    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Create figure
    if figsize is None:
        figsize = (4 * D, 5) if show_correction != 'both' else (4 * D, 10)

    if show_correction == 'both':
        fig, axes = plt.subplots(2, D, figsize=figsize, squeeze=False)
    else:
        fig, axes = plt.subplots(1, D, figsize=figsize, squeeze=False)

    # Build dataframe
    if subset_mask is not None:
        meta_subset = model.meta[subset_mask]
        df = pd.DataFrame({
            'x_true': x_true,
            'technical_group_code': meta_subset['technical_group_code'].values,
            'target': meta_subset['target'].values
        })
    else:
        df = pd.DataFrame({
            'x_true': x_true,
            'technical_group_code': model.meta['technical_group_code'].values,
            'target': model.meta['target'].values
        })

    for d in range(D):
        df[f'dim_{d}'] = values_3d[:, d]

    # Filter valid
    valid = df['x_true'] > 0
    for d in range(D):
        valid &= np.isfinite(df[f'dim_{d}'])
    df = df[valid].copy()

    # Get technical group labels
    code_to_label = _labels_by_code_for_df(model, df)
    group_codes   = np.sort(df['technical_group_code'].unique())

    # Plot each dimension
    for d in range(D):
        row_idx = 0 if show_correction != 'both' else (0, 1)
        rows = [row_idx] if isinstance(row_idx, int) else row_idx

        for row, corrected in zip(rows, [False, True] if len(rows) == 2 else [show_correction == 'corrected']):
            ax = axes[row, d]

            for idx, group_code in enumerate(group_codes):
                group_code = int(group_code)
                group_label = code_to_label[group_code]
                color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                df_group = df[df['technical_group_code'] == group_code].copy()

                if len(df_group) == 0:
                    continue

                y_plot = df_group[f'dim_{d}'].values

                if corrected and has_technical_fit:
                    # Apply additive correction
                    if hasattr(modality, 'alpha_y_prefit_add'):
                        alpha_y_add = modality.alpha_y_prefit_add
                        # alpha_y_add shape: [S or 1, C, T, D] or [C, T, D]
                        # Need to extract [group_code, feature_idx, d]
                        if alpha_y_add.ndim == 4:
                            correction = _to_scalar(alpha_y_add[:, group_code, feature_idx, d].mean())
                        elif alpha_y_add.ndim == 3:
                            correction = _to_scalar(alpha_y_add[group_code, feature_idx, d])
                        else:
                            correction = 0  # Fallback
                        y_plot = y_plot - correction

                # k-NN smoothing
                k = _knn_k(len(df_group), window)
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_plot, k)

                # Plot
                ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label if d == 0 else None)

            # Trans function overlay (if trans model fitted)
            # Note: mvnormal has multiple dimensions, so we overlay on first dimension only
            if show_trans_function and not corrected and d == 0:
                x_range = np.linspace(x_true.min(), x_true.max(), 100)
                y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

                if y_pred is not None:
                    ax.plot(np.log2(x_range), y_pred,
                           color='blue', linestyle='--', linewidth=2,
                           label='Fitted Trans Function')
                    warnings.warn("Trans function overlay on mvnormal shows prediction on first dimension only")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'Dimension {d}')
            title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
            ax.set_title(f"Dim {d}{title_suffix}")
            if d == 0 and row == 0:
                ax.legend(frameon=False, loc='upper right')

    plt.suptitle(f"{model.cis_gene} → {feature}")
    plt.tight_layout()

    return fig


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

    # Use max number of non-zero categories (computed from subsetted data)
    max_non_zero = max(len(nz) for nz in non_zero_cats_per_feature) if non_zero_cats_per_feature else 1

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

    # Create figure layout based on non-zero categories only
    if show_correction == 'both':
        # Layout: (n_features × max_non_zero) rows × 2 columns
        n_rows = n_features * max_non_zero
        n_cols = 2
        if figsize is None:
            figsize = (12, 3 * n_rows)  # 3 inches per row
    else:
        # Layout: n_features rows × max_non_zero columns
        n_rows = n_features
        n_cols = max_non_zero
        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)  # 4 inches per column, 3 inches per row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Get technical group labels
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(model.meta['technical_group_code'].unique())

    # Plot each feature
    for feat_i, (feat_idx, feat_name) in enumerate(zip(feature_indices, feature_names)):
        K = Ks[feat_i]
        non_zero_cats = non_zero_cats_per_feature[feat_i]

        if len(non_zero_cats) == 0:
            # Hide axes for this feature (all its rows/columns)
            if show_correction == 'both':
                # Hide all rows for this feature: feat_i * max_non_zero to (feat_i + 1) * max_non_zero
                for row_idx in range(feat_i * max_non_zero, (feat_i + 1) * max_non_zero):
                    for col_idx in range(2):
                        axes[row_idx, col_idx].axis('off')
            else:
                # Hide all columns for this feature row
                for col_idx in range(max_non_zero):
                    axes[feat_i, col_idx].axis('off')
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

        counts_filtered = counts_3d[valid_mask, :]
        total_filtered = total_counts[valid_mask]

        # Compute raw proportions
        with np.errstate(divide='ignore', invalid='ignore'):
            props_raw = np.where(total_filtered[:, None] > 0,
                                 counts_filtered / total_filtered[:, None],
                                 1.0 / K)

        for k in range(K):
            df[f'cat_{k}'] = props_raw[:, k]

        # Filter valid x_true
        valid = df['x_true'] > 0
        df = df[valid].copy()
        props_raw = props_raw[valid, :]

        if len(df) == 0:
            # Hide axes for this feature (no data after filtering)
            if show_correction == 'both':
                for row_idx in range(feat_i * max_non_zero, (feat_i + 1) * max_non_zero):
                    for col_idx in range(2):
                        axes[row_idx, col_idx].axis('off')
            else:
                for col_idx in range(max_non_zero):
                    axes[feat_i, col_idx].axis('off')
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
                # Row: feat_i * max_non_zero + cat_plot_idx, Columns: 0 (uncorrected), 1 (corrected)
                row = feat_i * max_non_zero + cat_plot_idx
                for col_idx, corrected in enumerate([False, True]):
                    ax = axes[row, col_idx]

        
                    for idx, group_code in enumerate(group_codes):
                        group_code = int(group_code)
                        group_label = code_to_label[group_code]
                        color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                        df_group = df[df['technical_group_code'] == group_code].copy()

                        if len(df_group) == 0:
                            continue

                        # Get proportions for this group
                        group_mask = df['technical_group_code'] == group_code
                        props_group = props_raw[group_mask, :]

                        if corrected and has_technical_fit:
                            # ---- Compute zero-category mask from the data used on THIS axis ----
                            zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)
                            
                            # ---- Per-sample correction, averaged in probability space ----
                            alpha_y_add = modality.alpha_y_prefit_add
                            # Align indices explicitly to avoid accidental reindexing issues
                            df = df.reset_index(drop=True)
                            props_raw = np.asarray(props_raw)  # (n_cells_filtered, K), same length as df now
                            group_mask = (df['technical_group_code'] == group_code).values
                            props_group = props_raw[group_mask, :]  # (n_group_cells, K)
                            
                            props_corrected = _multinomial_correct_mean_probs(
                                props_group=props_group,
                                alpha_y_add=alpha_y_add,
                                group_code=group_code,
                                feature_idx=feat_idx,
                                zero_cat_mask=zero_cat_mask
                            )  # (n_group_cells, K)
                            
                            y_data = props_corrected[:, k]
                        else:
                            y_data = df_group[f'cat_{k}'].values

                        # k-NN smoothing
                        knn = _knn_k(len(df_group), window)
                        x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_data, knn)

                        # Plot
                        ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                               label=group_label if cat_plot_idx == 0 and feat_i == 0 else None)

                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(f'Proportion')
                    title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                    ax.set_title(f"{feat_name[:20]}... {cat_label}{title_suffix}", fontsize=9)
                    if cat_plot_idx == 0 and feat_i == 0:
                        ax.legend(frameon=False, loc='upper right', fontsize=8)
            else:
                # Row: feat_i, Column: cat_plot_idx
                ax = axes[feat_i, cat_plot_idx]
                corrected = (show_correction == 'corrected')
        
                for idx, group_code in enumerate(group_codes):
                    group_code = int(group_code)
                    group_label = code_to_label[group_code]
                    color = _color_for_label(group_label, fallback_idx=idx, palette=color_palette)
                    df_group = df[df['technical_group_code'] == group_code].copy()

                    if len(df_group) == 0:
                        continue

                    # Get proportions for this group
                    group_mask = df['technical_group_code'] == group_code
                    props_group = props_raw[group_mask, :]

                    if corrected and has_technical_fit:
                        # ---- Compute zero-category mask from the data used on THIS axis ----
                        zero_cat_mask = (counts_filtered.sum(axis=0) == 0)  # (K,)
                        
                        # ---- Per-sample correction, averaged in probability space ----
                        alpha_y_add = modality.alpha_y_prefit_add
                        # Align indices explicitly to avoid accidental reindexing issues
                        df = df.reset_index(drop=True)
                        props_raw = np.asarray(props_raw)  # (n_cells_filtered, K), same length as df now
                        group_mask = (df['technical_group_code'] == group_code).values
                        props_group = props_raw[group_mask, :]  # (n_group_cells, K)
                        
                        props_corrected = _multinomial_correct_mean_probs(
                            props_group=props_group,
                            alpha_y_add=alpha_y_add,
                            group_code=group_code,
                            feature_idx=feat_idx,
                            zero_cat_mask=zero_cat_mask
                        )  # (n_group_cells, K)
                        
                        y_data = props_corrected[:, k]
                    else:
                        y_data = df_group[f'cat_{k}'].values

                    # k-NN smoothing
                    knn = _knn_k(len(df_group), window)
                    x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_data, knn)

                    # Plot
                    ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2,
                           label=group_label if cat_plot_idx == 0 and feat_i == 0 else None)

                ax.set_xlabel(xlabel)
                ax.set_ylabel(f'Proportion')
                title_suffix = ' (corrected)' if corrected else ' (uncorrected)'
                ax.set_title(f"{feat_name[:20]}... {cat_label}{title_suffix}", fontsize=9)
                if cat_plot_idx == 0 and feat_i == 0:
                    ax.legend(frameon=False, loc='upper right', fontsize=8)

        # Hide unused subplots (if this feature has fewer non-zero cats than max_non_zero)
        n_cats_this_feature = len(non_zero_cats)
        if show_correction == 'both':
            # Hide unused rows for this feature
            for cat_idx in range(n_cats_this_feature, max_non_zero):
                row_idx = feat_i * max_non_zero + cat_idx
                for col_idx in range(2):
                    axes[row_idx, col_idx].axis('off')
        else:
            # Hide unused columns for this feature
            for cat_idx in range(n_cats_this_feature, max_non_zero):
                axes[feat_i, cat_idx].axis('off')

    plt.suptitle(f"{model.cis_gene} → {gene_name} (gene, n={n_features} features, min_counts={min_counts})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])  # Leave space for suptitle

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
        Fully implemented for: negbinom, binomial, normal
        Not yet implemented for: multinomial, mvnormal (will issue warning)
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

        # Apply mask to x_true
        x_true = x_true[subset_mask]

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
            return _plot_multinomial_multifeature(
                model=model,
                feature_indices=feature_indices,
                feature_names=feature_names_resolved,
                gene_name=feature,
                modality=modality,
                x_true=x_true,
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
            figsize = (12, 4 * n_rows)  # 12 inches wide (6 per subplot), 4 inches per row

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

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
            elif distribution == 'normal':
                plot_normal_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='uncorrected',
                    color_palette=color_palette, show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=axes[i, 0],
                    subset_mask=subset_mask, **kwargs
                )
            else:
                # mvnormal returns its own figure
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
            elif distribution == 'normal':
                plot_normal_xy(
                    model=model, feature=feat_name, modality=modality,
                    x_true=x_true, window=window, show_correction='corrected',
                    color_palette=color_palette, show_trans_function=show_hill_function,
                    show_ntc_gradient=show_ntc_gradient, xlabel=xlabel, ax=axes[i, 1],
                    subset_mask=subset_mask, **kwargs
                )
            else:
                # mvnormal returns its own figure
                axes[i, 1].text(0.5, 0.5, f"Multi-panel not supported for {distribution}",
                               ha='center', va='center', transform=axes[i, 1].transAxes)

        plt.suptitle(f"{model.cis_gene} → {feature} (gene, n={n_features} features)")
        plt.tight_layout()
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

    elif distribution == 'normal':
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

    elif distribution == 'mvnormal':
        return plot_mvnormal_xy(
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
            figsize=figsize,
            subset_mask=subset_mask,
            **kwargs
        )

    else:
        raise ValueError(f"Plotting not implemented for distribution '{distribution}'")

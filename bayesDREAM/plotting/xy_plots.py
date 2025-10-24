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
from typing import Optional, Dict, List, Tuple, Union
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


def _smooth_knn(x: np.ndarray, y: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    x_sorted : np.ndarray
        Sorted x values
    y_hat : np.ndarray
        Smoothed y values
    """
    if len(x) == 0:
        return np.array([]), np.array([])

    order = np.argsort(x)
    x_sorted = np.asarray(x)[order].reshape(-1, 1)
    y_sorted = np.asarray(y)[order]

    k = max(1, min(k, len(x_sorted)))
    tree = cKDTree(x_sorted)

    y_hat = np.empty_like(y_sorted, dtype=float)
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
# Hill Function Utilities (for negbinom overlay)
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


# ============================================================================
# Main plotting functions (to be continued in next edit)
# ============================================================================


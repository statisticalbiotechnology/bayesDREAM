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

    # Check if feature in trans genes
    trans_genes = model.trans_genes if hasattr(model, 'trans_genes') else []
    if feature not in trans_genes:
        return None

    gene_idx = trans_genes.index(feature)

    # Get baseline A (present in all function types)
    if 'A' not in posterior:
        return None

    A_samples = posterior['A']
    A = A_samples.mean(dim=0)[gene_idx].item() if hasattr(A_samples, 'mean') else A_samples.mean(axis=0)[gene_idx]

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
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
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
    """
    # Get data
    feature_idx = modality.feature_meta.index.get_loc(feature) if feature in modality.feature_meta.index else None
    if feature_idx is None:
        # Try by gene name
        if 'gene' in modality.feature_meta.columns:
            mask = modality.feature_meta['gene'] == feature
            if mask.sum() > 0:
                feature_idx = mask.idxmax()
        if feature_idx is None:
            raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts for this feature
    if modality.cells_axis == 1:
        y_obs = modality.counts[feature_idx, :]
    else:
        y_obs = modality.counts[:, feature_idx]

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true,
        'y_obs': y_obs,
        'technical_group_code': model.meta['technical_group_code'].values,
        'target': model.meta['target'].values,
        'sum_factor': model.meta['sum_factor'].values if 'sum_factor' in model.meta.columns else 1.0
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
        axes = [ax]

    # Get technical group labels
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(df['technical_group_code'].unique())

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormap for gradient if needed
    if show_ntc_gradient:
        cmap = plt.cm.viridis_r  # Reversed: darker = fewer NTCs

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added

        for group_code, group_label in zip(group_codes, group_labels):
            df_group = df[df['technical_group_code'] == group_code].copy()

            if corrected and has_technical_fit:
                # Apply alpha_y correction
                alpha_y_full = modality.alpha_y_prefit  # [S or 1, C, T]
                if alpha_y_full.ndim == 3:
                    alpha_y_val = alpha_y_full[:, group_code, feature_idx].mean()
                else:
                    alpha_y_val = alpha_y_full[group_code, feature_idx]
                y_expr = df_group['y_obs'] / (df_group['sum_factor'] * alpha_y_val)
            else:
                y_expr = df_group['y_obs'] / df_group['sum_factor']

            # Filter valid
            valid = (df_group['x_true'] > 0) & np.isfinite(y_expr) & (y_expr > 0)
            df_group = df_group[valid].copy()
            y_expr = y_expr[valid]

            if len(df_group) == 0:
                continue

            # Get is_ntc for this group
            is_ntc_group = is_ntc[df_group.index].values

            # k-NN smoothing
            k = _knn_k(len(df_group), window)
            if show_ntc_gradient and not corrected:
                # Smoothing with NTC tracking
                x_smooth, y_smooth, ntc_prop = _smooth_knn(
                    df_group['x_true'].values,
                    y_expr.values,
                    k,
                    is_ntc=is_ntc_group
                )

                # Use gradient coloring
                plot_colored_line(
                    x=np.log2(x_smooth),
                    y=np.log2(y_smooth),
                    color_values=1 - ntc_prop,  # Darker = fewer NTCs
                    cmap=cmap,
                    ax=ax_plot,
                    linewidth=2
                )

                # Add colorbar (once per axis)
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax_plot)
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                # Standard smoothing without NTC tracking
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_expr.values, k)

                # Use standard coloring
                color = color_palette.get(group_label, f'C{group_code}')
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
    xlabel: str = "log2(x_true)",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot binomial (PSI - percent spliced in).

    Y-axis: PSI = counts / denominator
    Filter: min_counts on denominator

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        Lighter colors = more NTC cells, Darker colors = fewer NTC cells
    """
    # Get data
    feature_idx = modality.feature_meta.index.get_loc(feature) if feature in modality.feature_meta.index else None
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts and denominator
    if modality.cells_axis == 1:
        counts = modality.counts[feature_idx, :]
        denom = modality.denominator[feature_idx, :]
    else:
        counts = modality.counts[:, feature_idx]
        denom = modality.denominator[:, feature_idx]

    # Filter by min_counts
    valid_mask = denom >= min_counts

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true[valid_mask],
        'counts': counts[valid_mask],
        'denominator': denom[valid_mask],
        'technical_group_code': model.meta['technical_group_code'].values[valid_mask],
        'target': model.meta['target'].values[valid_mask]
    })

    # Compute PSI
    df['PSI'] = df['counts'] / df['denominator']

    # Filter valid
    valid = (df['x_true'] > 0) & np.isfinite(df['PSI'])
    df = df[valid].copy()

    if len(df) == 0:
        raise ValueError(f"No data remaining after filtering (min_counts={min_counts})")

    # Create axes
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Get technical group labels
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(df['technical_group_code'].unique())

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormap for gradient if needed
    if show_ntc_gradient:
        cmap = plt.cm.viridis_r  # Reversed: darker = fewer NTCs

    # Plot
    colorbar_added = False  # Track if colorbar added

    for group_code, group_label in zip(group_codes, group_labels):
        df_group = df[df['technical_group_code'] == group_code].copy()

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
                df_group['PSI'].values,
                k,
                is_ntc=is_ntc_group
            )

            # Use gradient coloring
            plot_colored_line(
                x=np.log2(x_smooth),
                y=y_smooth,
                color_values=1 - ntc_prop,  # Darker = fewer NTCs
                cmap=cmap,
                ax=ax,
                linewidth=2
            )

            # Add colorbar (once per axis)
            if not colorbar_added:
                fig = ax.get_figure()
                sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                colorbar_added = True
        else:
            # Standard smoothing without NTC tracking
            x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, df_group['PSI'].values, k)

            # Use standard coloring
            color = color_palette.get(group_label, f'C{group_code}')
            ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label)

    # Trans function overlay (if trans model fitted)
    if show_trans_function:
        x_range = np.linspace(x_true.min(), x_true.max(), 100)
        y_pred = predict_trans_function(model, feature, x_range, modality_name=modality.name)

        if y_pred is not None:
            # For binomial, PSI is in [0, 1], so clip predictions
            y_pred_clipped = np.clip(y_pred, 0, 1)
            ax.plot(np.log2(x_range), y_pred_clipped,
                   color='blue', linestyle='--', linewidth=2,
                   label='Fitted Trans Function')

    ax.axhline(0.0, linestyle='--', linewidth=1, alpha=0.6, color='gray')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('PSI (Percent Spliced In)')
    ax.set_title(f"{model.cis_gene} → {feature} (min_counts={min_counts})")
    ax.legend(frameon=False)

    return ax


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
    **kwargs
) -> Union[plt.Figure, List[plt.Axes]]:
    """
    Plot multinomial (e.g., donor/acceptor usage) - one subplot per category.

    Y-axis: Proportion for each category
    Layout: One row with K subplots (one per category)
    If show_correction='both': Two rows × K subplots

    Parameters
    ----------
    show_ntc_gradient : bool
        If True, color lines by NTC proportion in k-NN window (default: False)
        **Note**: Not yet fully implemented for multinomial - will issue warning
    """
    if show_ntc_gradient:
        warnings.warn("NTC gradient coloring not yet implemented for multinomial distributions - using standard colors")
    # Get data
    feature_idx = modality.feature_meta.index.get_loc(feature) if feature in modality.feature_meta.index else None
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get counts: shape (cells, K) for this feature
    if modality.counts.ndim == 3:
        counts_3d = modality.counts[feature_idx, :, :]  # (cells, K)
    else:
        raise ValueError(f"Expected 3D counts for multinomial, got {modality.counts.ndim}D")

    K = counts_3d.shape[1]

    # Filter by min_counts (total across categories)
    total_counts = counts_3d.sum(axis=1)
    valid_mask = total_counts >= min_counts

    # Build dataframe
    df = pd.DataFrame({
        'x_true': x_true[valid_mask],
        'technical_group_code': model.meta['technical_group_code'].values[valid_mask],
        'target': model.meta['target'].values[valid_mask]
    })

    # Add proportions for each category
    for k in range(K):
        df[f'cat_{k}'] = counts_3d[valid_mask, k] / total_counts[valid_mask]

    # Filter valid x_true
    valid = df['x_true'] > 0
    df = df[valid].copy()

    if len(df) == 0:
        raise ValueError(f"No data remaining after filtering (min_counts={min_counts})")

    # Check technical correction availability
    has_technical_fit = modality.alpha_y_prefit is not None
    if show_correction == 'corrected' and not has_technical_fit:
        warnings.warn(f"Technical fit not available for modality '{modality.name}' - showing uncorrected only")
        show_correction = 'uncorrected'

    # Note: Multinomial technical correction is complex and may not be implemented yet
    if show_correction != 'uncorrected' and has_technical_fit:
        warnings.warn("Technical correction for multinomial not yet fully implemented - showing uncorrected")
        show_correction = 'uncorrected'

    # Create figure
    if figsize is None:
        figsize = (4 * K, 5) if show_correction != 'both' else (4 * K, 10)

    if show_correction == 'both':
        fig, axes = plt.subplots(2, K, figsize=figsize, squeeze=False)
    else:
        fig, axes = plt.subplots(1, K, figsize=figsize, squeeze=False)

    # Get technical group labels
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(df['technical_group_code'].unique())

    # Plot each category
    for k in range(K):
        row_idx = 0 if show_correction != 'both' else (0, 1)
        rows = [row_idx] if isinstance(row_idx, int) else row_idx

        for row in rows:
            ax = axes[row, k]

            for group_code, group_label in zip(group_codes, group_labels):
                df_group = df[df['technical_group_code'] == group_code].copy()

                if len(df_group) == 0:
                    continue

                # k-NN smoothing
                knn = _knn_k(len(df_group), window)
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, df_group[f'cat_{k}'].values, knn)

                # Plot
                color = color_palette.get(group_label, f'C{group_code}')
                ax.plot(np.log2(x_smooth), y_smooth, color=color, linewidth=2, label=group_label if k == 0 else None)

            # Trans function overlay (note: not fully supported for multinomial)
            if show_trans_function and k == 0 and row == 0:
                warnings.warn("Trans function overlay for multinomial not yet fully implemented - "
                             "prediction would require modeling all K proportions simultaneously")

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'Proportion (Category {k})')
            ax.set_title(f"Category {k}")
            if k == 0 and row == 0:
                ax.legend(frameon=False, loc='upper right')

    plt.suptitle(f"{model.cis_gene} → {feature} (min_counts={min_counts})")
    plt.tight_layout()

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
    feature_idx = modality.feature_meta.index.get_loc(feature) if feature in modality.feature_meta.index else None
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get values
    if modality.cells_axis == 1:
        y_vals = modality.counts[feature_idx, :]
    else:
        y_vals = modality.counts[:, feature_idx]

    # Build dataframe
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
        axes = [ax]

    # Get technical group labels
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(df['technical_group_code'].unique())

    # Detect NTC cells for gradient coloring
    is_ntc = df['target'].str.lower() == 'ntc'

    # Create colormap for gradient if needed
    if show_ntc_gradient:
        cmap = plt.cm.viridis_r  # Reversed: darker = fewer NTCs

    # Plot function
    def _plot_one(ax_plot, corrected):
        colorbar_added = False  # Track if colorbar added

        for group_code, group_label in zip(group_codes, group_labels):
            df_group = df[df['technical_group_code'] == group_code].copy()

            if len(df_group) == 0:
                continue

            y_plot = df_group['y_val'].values

            if corrected and has_technical_fit:
                # Apply additive correction (alpha_y_add for normal)
                if hasattr(modality, 'alpha_y_prefit_add'):
                    alpha_y_add = modality.alpha_y_prefit_add
                    if alpha_y_add.ndim == 3:
                        correction = alpha_y_add[:, group_code, feature_idx].mean()
                    else:
                        correction = alpha_y_add[group_code, feature_idx]
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

                # Use gradient coloring
                plot_colored_line(
                    x=np.log2(x_smooth),
                    y=y_smooth,
                    color_values=1 - ntc_prop,  # Darker = fewer NTCs
                    cmap=cmap,
                    ax=ax_plot,
                    linewidth=2
                )

                # Add colorbar (once per axis)
                if not colorbar_added:
                    fig = ax_plot.get_figure()
                    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax_plot)
                    cbar.set_label('1 - Proportion NTC (darker = fewer NTCs)')
                    colorbar_added = True
            else:
                # Standard smoothing without NTC tracking
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_plot, k)

                # Use standard coloring
                color = color_palette.get(group_label, f'C{group_code}')
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
    feature_idx = modality.feature_meta.index.get_loc(feature) if feature in modality.feature_meta.index else None
    if feature_idx is None:
        raise ValueError(f"Feature '{feature}' not found in modality")

    # Get values: shape (cells, D) for this feature
    if modality.counts.ndim == 3:
        values_3d = modality.counts[feature_idx, :, :]  # (cells, D)
    else:
        raise ValueError(f"Expected 3D counts for mvnormal, got {modality.counts.ndim}D")

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
    group_labels = get_technical_group_labels(model)
    group_codes = sorted(df['technical_group_code'].unique())

    # Plot each dimension
    for d in range(D):
        row_idx = 0 if show_correction != 'both' else (0, 1)
        rows = [row_idx] if isinstance(row_idx, int) else row_idx

        for row, corrected in zip(rows, [False, True] if len(rows) == 2 else [show_correction == 'corrected']):
            ax = axes[row, d]

            for group_code, group_label in zip(group_codes, group_labels):
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
                            correction = alpha_y_add[:, group_code, feature_idx, d].mean()
                        elif alpha_y_add.ndim == 3:
                            correction = alpha_y_add[group_code, feature_idx, d]
                        else:
                            correction = 0  # Fallback
                        y_plot = y_plot - correction

                # k-NN smoothing
                k = _knn_k(len(df_group), window)
                x_smooth, y_smooth = _smooth_knn(df_group['x_true'].values, y_plot, k)

                # Plot
                color = color_palette.get(group_label, f'C{group_code}')
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
# Unified plot_xy_data function
# ============================================================================

def plot_xy_data(
    model,
    feature: str,
    modality_name: Optional[str] = None,
    window: int = 100,
    show_correction: str = 'corrected',
    min_counts: int = 3,
    color_palette: Optional[Dict[str, str]] = None,
    show_hill_function: bool = True,
    show_ntc_gradient: bool = False,
    xlabel: str = "log2(x_true)",
    figsize: Optional[Tuple[int, int]] = None,
    src_barcodes: Optional[np.ndarray] = None,
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
        Feature name (gene, junction, donor, etc.)
    modality_name : str, optional
        Modality name (default: primary modality)
    window : int
        k-NN window size for smoothing (default: 100 cells)
    show_correction : str
        'uncorrected': no technical correction
        'corrected': apply alpha_y technical correction (default)
        'both': show both side-by-side
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
    xlabel : str
        X-axis label (default: "log2(x_true)")
    figsize : tuple, optional
        Figure size (auto-sized if None)
    src_barcodes : np.ndarray, optional
        Source barcode order if x_true not in model.meta order
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
    >>> # Plot gene counts with Hill function
    >>> model.plot_xy_data('TET2', window=100, show_hill_function=True)
    >>>
    >>> # Plot splice junction with min_counts filter
    >>> model.plot_xy_data('chr1:12345:67890:+', modality_name='splicing_sj',
    ...                     min_counts=5)
    >>>
    >>> # Plot with custom colors
    >>> model.plot_xy_data('GFI1B', color_palette={'CRISPRa': 'red', 'CRISPRi': 'blue'})
    >>>
    >>> # Show both corrected and uncorrected
    >>> model.plot_xy_data('TET2', show_correction='both')
    >>>
    >>> # Plot with NTC gradient coloring
    >>> model.plot_xy_data('TET2', show_ntc_gradient=True)
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

    # Get modality
    if modality_name is None:
        modality_name = model.primary_modality
    modality = model.get_modality(modality_name)

    # Get color palette
    if color_palette is None:
        group_labels = get_technical_group_labels(model)
        color_palette = get_default_color_palette(group_labels)

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
            xlabel=xlabel,
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
            **kwargs
        )

    else:
        raise ValueError(f"Plotting not implemented for distribution '{distribution}'")

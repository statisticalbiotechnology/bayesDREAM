"""
Basic plotting functions for x_true distributions.

Provides scatter, violin, and density plots colored by guide.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from .helpers import to_np
from .colors import ColorScheme


def scatter_by_guide(model, cis_gene, log2=False, color_scheme=None, ax=None, show=True):
    """
    Scatter plot of per-cell mean vs std of x_true, colored by guide.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str
        Cis gene name
    log2 : bool, default False
        Whether to use log2 scale
    color_scheme : ColorScheme, optional
        Custom color scheme. If None, uses default.
    ax : matplotlib axes, optional
        Axes to plot on. If None, creates new figure.
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    df = model[cis_gene].meta.copy()
    X = to_np(model[cis_gene].x_true)

    if log2:
        # filter strictly positive before log
        mask_pos = (X > 0).all(axis=0)
        X = X[:, mask_pos]
        df = df.loc[mask_pos].reset_index(drop=True)
        X = np.log2(X)

    x_mean, x_std = X.mean(axis=0), X.std(axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for guide, subidx in df.groupby('guide').groups.items():
        color = color_scheme.get_guide_color(guide, 'black')
        ax.scatter(x_mean[subidx], x_std[subidx], s=14, alpha=0.8,
                  color=color, label=guide)

    ax.set_xlabel('mean x_true' + (' (log2)' if log2 else ''))
    ax.set_ylabel('std x_true' + (' (log2)' if log2 else ''))
    ax.set_title(f'{cis_gene}: mean vs std of x_true' + (' (log2)' if log2 else ''))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(title='guide', fontsize=8, markerscale=1.2, frameon=False)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def scatter_ci95_by_guide(model, cis_gene, log2=False, full_width=False,
                          color_scheme=None, ax=None, show=True):
    """
    Scatter of per-cell mean vs 95% CI width (or half-width) of x_true samples.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str
        Cis gene name
    log2 : bool, default False
        Whether to use log2 scale
    full_width : bool, default False
        If True, plot full width. If False, plot half-width.
    color_scheme : ColorScheme, optional
        Custom color scheme
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    df = model[cis_gene].meta.copy()
    X = to_np(model[cis_gene].x_true)  # shape [S, N] (samples x cells)

    if log2:
        mask_pos = (X > 0).all(axis=0)
        X = X[:, mask_pos]
        df = df.loc[mask_pos].reset_index(drop=True)
        X = np.log2(X)

    x_mean = X.mean(axis=0)
    q_lo  = np.percentile(X, 2.5, axis=0)
    q_hi  = np.percentile(X, 97.5, axis=0)
    y_val = (q_hi - q_lo) if full_width else 0.5 * (q_hi - q_lo)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    for guide, idx in df.groupby('guide').groups.items():
        color = color_scheme.get_guide_color(guide, 'black')
        ax.scatter(x_mean[idx], y_val[idx], s=14, alpha=0.85,
                  color=color, label=guide)

    ax.set_xlabel('mean x_true' + (' (log2)' if log2 else ''))
    ylabel = '95% CI ' + ('width' if full_width else 'half-width')
    ylabel += ' of x_true' + (' (log2)' if log2 else '')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{cis_gene}: mean vs 95% CI of x_true' + (' (log2)' if log2 else ''))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(title='guide', fontsize=8, markerscale=1.2, frameon=False)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def violin_by_guide_log2(model, cis_gene, color_scheme=None, ax=None, show=True):
    """
    Violin plot of x_true (log2) grouped by guide, colored by target.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str
        Cis gene name
    color_scheme : ColorScheme, optional
        Custom color scheme
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    df = model[cis_gene].meta.copy()
    X = to_np(model[cis_gene].x_true)

    pos_mask = (X > 0).all(axis=0)
    X = X[:, pos_mask]
    df = df.loc[pos_mask].reset_index(drop=True)

    Xlog = np.log2(X)
    x_cell_mean = Xlog.mean(axis=0)
    df = df.assign(x_true_mean_log2=x_cell_mean)

    guide_order = sorted(df['guide'].astype(str).unique(),
                        key=lambda g: (g.split('_')[0], int(g.split('_')[1])
                                     if '_' in g and g.split('_')[1].isdigit() else 0))
    data = [df.loc[df['guide'] == g, 'x_true_mean_log2'].values for g in guide_order]

    colors = []
    for g in guide_order:
        target = g.split('_')[0] if '_' in g else g
        colors.append(color_scheme.get_target_color(target, 'gray'))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    parts = ax.violinplot(data, positions=np.arange(1, len(guide_order)+1),
                         showmeans=True, showextrema=True, widths=0.7)
    for body, c in zip(parts['bodies'], colors):
        body.set_facecolor(c)
        body.set_edgecolor('black')
        body.set_alpha(0.85)

    for pc_key in ['cmeans', 'cmaxes', 'cmins', 'cbars']:
        if pc_key in parts:
            parts[pc_key].set_edgecolor('black')
            parts[pc_key].set_linewidth(1.2)

    ax.set_xticks(np.arange(1, len(guide_order)+1))
    ax.set_xticklabels(guide_order, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('x_true mean (log₂)', fontsize=11)
    ax.set_title(f'{cis_gene}: x_true distribution by guide', fontsize=12)
    ax.grid(axis='y', linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def filled_density_by_guide_log2(model, cis_gene, bw=None, color_scheme=None,
                                 ax=None, show=True):
    """
    Filled KDE density plot of x_true (log2), colored by guide.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str
        Cis gene name
    bw : float, optional
        KDE bandwidth. If None, uses scott's rule.
    color_scheme : ColorScheme, optional
        Custom color scheme
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool, default True
        Whether to display the plot

    Returns
    -------
    ax : matplotlib axes
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    df = model[cis_gene].meta.copy()
    X = to_np(model[cis_gene].x_true)

    pos_mask = (X > 0).all(axis=0)
    X = X[:, pos_mask]
    df = df.loc[pos_mask].reset_index(drop=True)

    Xlog = np.log2(X)
    x_cell_mean = Xlog.mean(axis=0)
    df = df.assign(x_true_mean_log2=x_cell_mean)

    xmin, xmax = np.percentile(x_cell_mean, [0.5, 99.5])
    x_grid = np.linspace(xmin, xmax, 500)

    guide_order = sorted(df['guide'].astype(str).unique(),
                        key=lambda g: (g.split('_')[0], int(g.split('_')[1])
                                     if '_' in g and g.split('_')[1].isdigit() else 0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    for g in guide_order:
        sub = df.loc[df['guide'] == g, 'x_true_mean_log2'].values
        if len(sub) < 2:
            continue

        kde = gaussian_kde(sub, bw_method=bw)
        density = kde(x_grid)

        color = color_scheme.get_guide_color(g, 'black')
        ax.fill_between(x_grid, density, alpha=0.4, color=color, label=g)
        ax.plot(x_grid, density, color=color, linewidth=1.5)

    ax.set_xlabel('x_true mean (log₂)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{cis_gene}: x_true density by guide', fontsize=12)
    ax.legend(title='guide', fontsize=8, frameon=False)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def scatter_param_mean_vs_ci(
    param_samps,
    param_name='parameter',
    subset_mask=None,
    color_by=None,
    color_label='color metric',
    cmap='Blues_r',
    vmin=None,
    vmax=None,
    log2=False,
    ax=None,
    show=True,
    title=None,
    figsize=(7, 5),
):
    """
    Scatter plot of parameter mean vs 95% CI width, with optional color coding.

    This is useful for visualizing parameter uncertainty vs magnitude, with
    optional coloring by dependency masks, NaN fractions, or other metrics.

    Parameters
    ----------
    param_samps : np.ndarray
        Parameter samples, shape (n_samples, n_features)
    param_name : str
        Parameter name for axis labels (default: 'parameter')
    subset_mask : np.ndarray, optional
        Boolean mask to subset features. If provided, plots two groups:
        masked (colored) and unmasked (grey).
    color_by : np.ndarray, optional
        Values to color points by (length n_features). Requires subset_mask.
        Common uses:
        - NaN fraction: color by how many samples are NaN
        - Dependency metric: color by strength of effect
    color_label : str
        Label for colorbar (default: 'color metric')
    cmap : str
        Colormap name (default: 'Blues_r' for darker = fewer NaNs)
    vmin, vmax : float, optional
        Color scale limits. If None, uses data range.
    log2 : bool
        Whether param_samps are on log2 scale (affects axis label only)
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool
        Whether to display the plot (default: True)
    title : str, optional
        Plot title. If None, auto-generates from param_name.
    figsize : tuple
        Figure size (default: (7, 5))

    Returns
    -------
    ax : matplotlib axes

    Examples
    --------
    >>> # Example 1: Inflection point with NaN fraction coloring
    >>> xinf_samps = hill_xinf_samples(K_samps, n_samps, tol_n=0.2)
    >>> dep_mask = dependency_mask_from_n(n_samps)
    >>> frac_nan = np.mean(np.isnan(xinf_samps), axis=0)
    >>> fig = scatter_param_mean_vs_ci(
    ...     xinf_samps,
    ...     param_name='x_infl',
    ...     subset_mask=dep_mask,
    ...     color_by=frac_nan,
    ...     color_label='fraction NaN (lighter = more NaN)',
    ...     cmap='Blues_r',
    ...     log2=True
    ... )

    >>> # Example 2: Hill coefficient n with dependency coloring
    >>> n_samps = model['GFI1B'].posterior_samples_trans['n_a'][:, 0, :].detach().cpu().numpy()
    >>> dep_mask = dependency_mask_from_n(n_samps)
    >>> fig = scatter_param_mean_vs_ci(
    ...     n_samps,
    ...     param_name='n (Hill coefficient)',
    ...     subset_mask=dep_mask,
    ...     log2=False
    ... )
    """
    param_samps = np.asarray(param_samps)

    # Compute mean and CI width
    param_mean = np.nanmean(param_samps, axis=0)
    param_lo = np.nanpercentile(param_samps, 2.5, axis=0)
    param_hi = np.nanpercentile(param_samps, 97.5, axis=0)
    param_ci_width = param_hi - param_lo

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Case 1: No subsetting - plot all points in one color
    if subset_mask is None:
        ax.scatter(param_mean, param_ci_width, s=8, alpha=0.6, color='blue')

    # Case 2: Subsetting without color coding
    elif color_by is None:
        # Plot non-masked points in grey
        if not np.all(subset_mask):
            ax.scatter(
                param_mean[~subset_mask],
                param_ci_width[~subset_mask],
                s=5, alpha=0.3, color='grey', label='not selected'
            )

        # Plot masked points in blue
        ax.scatter(
            param_mean[subset_mask],
            param_ci_width[subset_mask],
            s=5, alpha=0.2, color='blue', label='selected'
        )
        ax.legend(frameon=False, loc='best')

    # Case 3: Subsetting with color coding
    else:
        color_by = np.asarray(color_by)
        if len(color_by) != len(subset_mask):
            raise ValueError(f"color_by length ({len(color_by)}) must match subset_mask length ({len(subset_mask)})")

        # Plot non-masked points in grey
        if not np.all(subset_mask):
            ax.scatter(
                param_mean[~subset_mask],
                param_ci_width[~subset_mask],
                s=5, alpha=0.3, color='grey'
            )

        # Plot masked points with color coding
        valid_mask = subset_mask & np.isfinite(param_mean) & np.isfinite(param_ci_width)

        if vmin is None:
            vmin = np.nanmin(color_by[valid_mask])
        if vmax is None:
            vmax = np.nanmax(color_by[valid_mask])

        sc = ax.scatter(
            param_mean[valid_mask],
            param_ci_width[valid_mask],
            c=color_by[valid_mask],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=8,
            alpha=0.9
        )

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(color_label)

    # Labels and formatting
    xlabel = f'Mean {param_name}' + (' (log₂)' if log2 else '')
    ylabel = f'95% CI width of {param_name}' + (' (log₂)' if log2 else '')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f'{param_name}: mean vs uncertainty'
    ax.set_title(title)

    ax.axhline(0, color='black', linestyle=':', linewidth=1)
    ax.grid(True, linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax

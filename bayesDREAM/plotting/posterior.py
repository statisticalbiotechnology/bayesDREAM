"""
Posterior density visualization functions.

Provides vertical density line plots for parameter posteriors and x_true distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from .helpers import to_np
from .colors import ColorScheme


def plot_posterior_density_lines(
    samples,
    title="Posterior density lines",
    sort_by="median",
    subset_mask=None,
    cmap="viridis",
    alpha_overall=0.5,
    density_gamma=0.7,
    norm_global=True,
    y_quantiles=(0.5, 99.5),
    grid_points=350,
    linewidth=0.8,
    add_median_lines=True,
    y_label=r"$\theta$",
    ax=None,
    show=True,
    y_range=None,
):
    """
    Plot per-feature posterior densities as vertical color lines.

    Parameters
    ----------
    samples : array-like, shape (n_samples, n_features)
        Posterior samples
    title : str
        Plot title
    sort_by : {'median', 'mean', None}
        How to sort features
    subset_mask : array-like, optional
        Boolean mask to subset features
    cmap : str or Colormap
        Colormap for density visualization
    alpha_overall : float
        Overall alpha for density colors
    density_gamma : float
        Gamma correction for density intensity
    norm_global : bool
        If True, normalize density across all features
    y_quantiles : tuple
        Quantiles for y-axis range
    grid_points : int
        Number of grid points for KDE
    linewidth : float
        Width of median lines
    add_median_lines : bool
        Whether to add horizontal median lines
    y_label : str
        Y-axis label
    ax : matplotlib axes, optional
        Axes to plot on
    show : bool
        Whether to display the plot
    y_range : tuple, optional
        Explicit (y_min, y_max) range

    Returns
    -------
    ax : matplotlib axes
    """
    samples = np.asarray(samples)

    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim > 2:
        samples = samples.reshape(samples.shape[0], -1)

    S, T = samples.shape

    if subset_mask is not None:
        subset_mask = np.asarray(subset_mask, dtype=bool)
        samples = samples[:, subset_mask]
        S, T = samples.shape

    if sort_by == "median":
        order = np.argsort(np.nanmedian(samples, axis=0))
    elif sort_by == "mean":
        order = np.argsort(np.nanmean(samples, axis=0))
    else:
        order = np.arange(T)
    samples_sorted = samples[:, order]

    # --- y-range: either from samples, or overridden explicitly ---
    if y_range is None:
        y_min, y_max = np.nanpercentile(samples_sorted, y_quantiles)
    else:
        y_min, y_max = y_range

    y_grid = np.linspace(y_min, y_max, grid_points)

    # KDE per feature
    dens_list = []
    for t in range(T):
        vals = samples_sorted[:, t]
        vals = vals[~np.isnan(vals)]
        if vals.size < 2:
            dens = np.zeros_like(y_grid)
        else:
            kde = gaussian_kde(vals)
            dens = kde(y_grid)
        dens_list.append(dens)
    dens_mat = np.stack(dens_list, axis=0)  # (T, grid_points)

    # Normalize density
    if norm_global:
        dens_max = dens_mat.max()
    else:
        dens_max = dens_mat.max(axis=1, keepdims=True)
    dens_max = np.maximum(dens_max, 1e-12)
    dens_norm = dens_mat / dens_max

    # Apply gamma correction
    dens_norm = dens_norm ** density_gamma

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Get colormap
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # Draw vertical density lines
    for t in range(T):
        x_pos = t + 1
        for i in range(grid_points):
            alpha = dens_norm[t, i] * alpha_overall
            if alpha > 0.01:  # Skip very faint lines
                color = cmap_obj(dens_norm[t, i])
                ax.plot([x_pos, x_pos], [y_grid[i], y_grid[i]],
                       color=color, alpha=alpha, linewidth=linewidth)

    # Add median lines
    if add_median_lines:
        medians = np.nanmedian(samples_sorted, axis=0)
        for t in range(T):
            ax.plot([t+0.7, t+1.3], [medians[t], medians[t]],
                   color='red', linewidth=1.2, alpha=0.8)

    ax.set_xlim(0.5, T + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Feature index (sorted)', fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(True, axis='y', linewidth=0.5, alpha=0.3)
    plt.tight_layout()

    if show:
        plt.show()

    return ax


def plot_xtrue_density_by_guide(
    model,
    cis_gene,
    log2=False,
    cmap="viridis",
    alpha_overall=0.5,
    density_gamma=0.7,
    norm_global=True,
    y_quantiles=(0.5, 99.5),
    grid_points=350,
    linewidth=0.8,
    group_by_guide=True,
    color_scheme=None,
    show=True,
):
    """
    One vertical density line per cell for x_true, with guide annotations.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_gene : str
        Cis gene name
    log2 : bool
        Whether to use log2 scale
    cmap : str or Colormap
        Colormap for density visualization
    alpha_overall : float
        Overall alpha for density colors
    density_gamma : float
        Gamma correction for density intensity
    norm_global : bool
        If True, normalize density across all cells
    y_quantiles : tuple
        Quantiles for y-axis range
    grid_points : int
        Number of grid points for KDE
    linewidth : float
        Width of lines
    group_by_guide : bool
        If True, group cells by guide. If False, order by median only.
    color_scheme : ColorScheme, optional
        Custom color scheme for guide annotations
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib figure
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    df = model[cis_gene].meta.copy()
    X = to_np(model[cis_gene].x_true)  # [S, N_cells]

    # log2 transform without dropping guides
    if log2:
        eps = 1e-6
        X = np.log2(np.maximum(X, eps))

    samples = np.asarray(X)       # [S, N]
    S, N = samples.shape
    guides = df['guide'].astype(str).to_numpy()   # length N

    # ---------- choose ordering ----------
    med_per_cell = np.nanmedian(samples, axis=0)

    if group_by_guide:
        # order cells: by guide, then median within guide
        def guide_sort_key(g):
            parts = g.split('_')
            root = parts[0]
            idx  = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            return (root, idx)

        unique_guides = sorted(np.unique(guides), key=guide_sort_key)
        guide_block_rank = {g: i for i, g in enumerate(unique_guides)}
        guide_ranks = np.array([guide_block_rank[g] for g in guides])

        order = np.lexsort((med_per_cell, guide_ranks))  # (N,)
    else:
        unique_guides = sorted(np.unique(guides))
        order = np.argsort(med_per_cell)

    samples_sorted = samples[:, order]
    guides_sorted  = guides[order]

    # ---------- draw density background using generic function ----------
    ylabel = "x_true" + (" (log₂)" if log2 else "")

    # Create figure with space for guide color bar
    fig = plt.figure(figsize=(12, 6))
    ax = plt.subplot2grid((20, 1), (2, 0), rowspan=18)

    # Draw density lines (no title, we'll add it to figure)
    plot_posterior_density_lines(
        samples_sorted,
        title="",
        sort_by=None,
        subset_mask=None,
        cmap=cmap,
        alpha_overall=alpha_overall,
        density_gamma=density_gamma,
        norm_global=norm_global,
        y_quantiles=y_quantiles,
        grid_points=grid_points,
        linewidth=linewidth,
        add_median_lines=False,
        y_label=ylabel,
        ax=ax,
        show=False,
    )

    # ---------- add colored median ticks per cell ----------
    medians_sorted = np.nanmedian(samples_sorted, axis=0)
    for i, g in enumerate(guides_sorted):
        color = color_scheme.get_guide_color(g, 'black')
        ax.plot([i+0.7, i+1.3], [medians_sorted[i], medians_sorted[i]],
               color=color, linewidth=1.5, alpha=0.9)

    # ---------- add guide color bar between title and axes ----------
    ax_bar = plt.subplot2grid((20, 1), (0, 0), rowspan=1)
    ax_bar.set_xlim(0.5, N + 0.5)
    ax_bar.set_ylim(0, 1)
    ax_bar.axis('off')

    for i, g in enumerate(guides_sorted):
        color = color_scheme.get_guide_color(g, 'black')
        rect = Rectangle((i+0.5, 0), 1, 1, facecolor=color, edgecolor='none')
        ax_bar.add_patch(rect)

    # ---------- add legend ----------
    legend_handles = []
    for g in unique_guides:
        color = color_scheme.get_guide_color(g, 'black')
        legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=color, markersize=8, label=g))

    ax.legend(handles=legend_handles, title='guide', fontsize=9,
             loc='upper left', frameon=True, framealpha=0.9)

    # Figure title
    fig.suptitle(f'{cis_gene}: x_true posterior density by cell (colored by guide)',
                fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if show:
        plt.show()

    return fig

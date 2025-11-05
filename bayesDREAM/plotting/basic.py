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

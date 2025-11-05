"""
Diagnostic plotting functions.

Provides quality control and diagnostic plots for bayesDREAM fits.
"""

import numpy as np
import matplotlib.pyplot as plt

from .colors import ColorScheme


def plot_sum_factor_comparison(model, cis_genes, sf_col1='clustered.sum.factor',
                               sf_col2='sum_factor_adj', color_scheme=None, show=True):
    """
    Plot pairwise comparison of sum factors (e.g., original vs adjusted).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    cis_genes : list of str
        Cis gene names to plot
    sf_col1 : str
        First sum factor column name
    sf_col2 : str
        Second sum factor column name
    color_scheme : ColorScheme, optional
        Custom color scheme
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : matplotlib figure
    """
    if color_scheme is None:
        color_scheme = ColorScheme()

    n_genes = len(cis_genes)
    fig, axes = plt.subplots(1, n_genes, figsize=(5*n_genes, 4),
                            sharex=False, sharey=False)
    if n_genes == 1:
        axes = [axes]

    for ax, cg in zip(axes, cis_genes):
        if cg not in model:
            print(f"[{cg}] not found in model, skipping.")
            continue

        df = model[cg].meta.copy()

        # Filter to positive values
        if sf_col1 not in df.columns or sf_col2 not in df.columns:
            print(f"[{cg}] missing sum factor columns, skipping.")
            continue

        df = df[(df[sf_col1] > 0) & (df[sf_col2] > 0)]

        # Plot by guide
        for guide, sub in df.groupby('guide'):
            color = color_scheme.get_guide_color(guide, 'black')
            ax.scatter(
                sub[sf_col1],
                sub[sf_col2],
                s=12,
                alpha=0.8,
                color=color,
                label=guide
            )

        # Identity line
        all_sf = np.concatenate([df[sf_col1].values, df[sf_col2].values])
        sf_min, sf_max = all_sf.min(), all_sf.max()
        ax.plot([sf_min, sf_max], [sf_min, sf_max], 'k--', linewidth=1, alpha=0.6)

        ax.set_xlabel(sf_col1, fontsize=10)
        ax.set_ylabel(sf_col2, fontsize=10)
        ax.set_title(f'{cg}: sum factor comparison', fontsize=11)
        ax.legend(fontsize=8, markerscale=1.2, frameon=False)
        ax.grid(True, linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig

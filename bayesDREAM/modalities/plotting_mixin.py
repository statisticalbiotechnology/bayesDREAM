"""
Plotting mixin for bayesDREAM.

Provides convenient plotting methods that wrap the plotting module functions.
"""

from typing import Optional, List, Union
import matplotlib.pyplot as plt

from ..plotting import (
    ColorScheme,
    scatter_by_guide,
    scatter_ci95_by_guide,
    violin_by_guide_log2,
    filled_density_by_guide_log2,
    plot_posterior_density_lines,
    plot_xtrue_density_by_guide,
    plot_sum_factor_comparison,
    plot_edger_vs_bayes_full_range,
    plot_edger_vs_bayes_observed_range
)


class PlottingMixin:
    """Mixin providing plotting methods for bayesDREAM."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._color_scheme = None

    def set_color_scheme(self, color_scheme: ColorScheme):
        """
        Set a custom color scheme for all plots.

        Parameters
        ----------
        color_scheme : ColorScheme
            Color scheme object
        """
        self._color_scheme = color_scheme

    def get_color_scheme(self) -> ColorScheme:
        """Get the current color scheme (creates default if none set)."""
        if self._color_scheme is None:
            self._color_scheme = ColorScheme()
        return self._color_scheme

    # ========================================================================
    # Basic x_true plots
    # ========================================================================

    def plot_xtrue_scatter(self, cis_gene: str, log2: bool = False,
                          ax: Optional[plt.Axes] = None, show: bool = True):
        """
        Plot x_true mean vs std scatter, colored by guide.

        Parameters
        ----------
        cis_gene : str
            Cis gene name
        log2 : bool
            Whether to use log2 scale
        ax : matplotlib axes, optional
            Axes to plot on
        show : bool
            Whether to display the plot

        Returns
        -------
        ax : matplotlib axes
        """
        return scatter_by_guide(
            self, cis_gene, log2=log2,
            color_scheme=self.get_color_scheme(),
            ax=ax, show=show
        )

    def plot_xtrue_ci(self, cis_gene: str, log2: bool = False,
                     full_width: bool = False,
                     ax: Optional[plt.Axes] = None, show: bool = True):
        """
        Plot x_true mean vs 95% CI width, colored by guide.

        Parameters
        ----------
        cis_gene : str
            Cis gene name
        log2 : bool
            Whether to use log2 scale
        full_width : bool
            If True, plot full width. If False, plot half-width.
        ax : matplotlib axes, optional
            Axes to plot on
        show : bool
            Whether to display the plot

        Returns
        -------
        ax : matplotlib axes
        """
        return scatter_ci95_by_guide(
            self, cis_gene, log2=log2, full_width=full_width,
            color_scheme=self.get_color_scheme(),
            ax=ax, show=show
        )

    def plot_xtrue_violin(self, cis_gene: str,
                         ax: Optional[plt.Axes] = None, show: bool = True):
        """
        Plot x_true violin plot (log2), grouped by guide, colored by target.

        Parameters
        ----------
        cis_gene : str
            Cis gene name
        ax : matplotlib axes, optional
            Axes to plot on
        show : bool
            Whether to display the plot

        Returns
        -------
        ax : matplotlib axes
        """
        return violin_by_guide_log2(
            self, cis_gene,
            color_scheme=self.get_color_scheme(),
            ax=ax, show=show
        )

    def plot_xtrue_density(self, cis_gene: str, bw: Optional[float] = None,
                          ax: Optional[plt.Axes] = None, show: bool = True):
        """
        Plot filled KDE density of x_true (log2), colored by guide.

        Parameters
        ----------
        cis_gene : str
            Cis gene name
        bw : float, optional
            KDE bandwidth
        ax : matplotlib axes, optional
            Axes to plot on
        show : bool
            Whether to display the plot

        Returns
        -------
        ax : matplotlib axes
        """
        return filled_density_by_guide_log2(
            self, cis_gene, bw=bw,
            color_scheme=self.get_color_scheme(),
            ax=ax, show=show
        )

    # ========================================================================
    # Posterior density plots
    # ========================================================================

    def plot_posterior_densities(self, samples, title: str = "Posterior density lines",
                                 sort_by: str = "median", **kwargs):
        """
        Plot posterior densities as vertical color lines.

        Parameters
        ----------
        samples : array-like, shape (n_samples, n_features)
            Posterior samples
        title : str
            Plot title
        sort_by : {'median', 'mean', None}
            How to sort features
        **kwargs
            Additional arguments passed to plot_posterior_density_lines

        Returns
        -------
        ax : matplotlib axes
        """
        return plot_posterior_density_lines(
            samples, title=title, sort_by=sort_by, **kwargs
        )

    def plot_xtrue_density_by_guide(self, cis_gene: str, log2: bool = False,
                                   group_by_guide: bool = True,
                                   show: bool = True, **kwargs):
        """
        Plot x_true posterior density with guide annotations.

        Parameters
        ----------
        cis_gene : str
            Cis gene name
        log2 : bool
            Whether to use log2 scale
        group_by_guide : bool
            If True, group cells by guide
        show : bool
            Whether to display the plot
        **kwargs
            Additional arguments passed to plot_xtrue_density_by_guide

        Returns
        -------
        fig : matplotlib figure
        """
        return plot_xtrue_density_by_guide(
            self, cis_gene, log2=log2,
            group_by_guide=group_by_guide,
            color_scheme=self.get_color_scheme(),
            show=show,
            **kwargs
        )

    # ========================================================================
    # Diagnostic plots
    # ========================================================================

    def plot_sum_factors(self, cis_genes: Union[str, List[str]],
                        sf_col1: str = 'clustered.sum.factor',
                        sf_col2: str = 'sum_factor_adj',
                        show: bool = True):
        """
        Plot sum factor comparison.

        Parameters
        ----------
        cis_genes : str or list of str
            Cis gene name(s)
        sf_col1 : str
            First sum factor column name
        sf_col2 : str
            Second sum factor column name
        show : bool
            Whether to display the plot

        Returns
        -------
        fig : matplotlib figure
        """
        if isinstance(cis_genes, str):
            cis_genes = [cis_genes]

        return plot_sum_factor_comparison(
            self, cis_genes, sf_col1=sf_col1, sf_col2=sf_col2,
            color_scheme=self.get_color_scheme(),
            show=show
        )

    # ========================================================================
    # DE comparison plots
    # ========================================================================

    def compare_with_edger_full(self, cis_genes: Union[str, List[str]],
                               de_df, fc_thresh: float = 0.5,
                               flip_edger_x: bool = True):
        """
        Compare bayesDREAM full-range log2FC with edgeR results.

        Parameters
        ----------
        cis_genes : str or list of str
            Cis gene name(s)
        de_df : pd.DataFrame
            edgeR results with columns: gene, logFC, FDR, guide
        fc_thresh : float
            Fold-change threshold for classification
        flip_edger_x : bool
            If True, flip sign of edgeR logFC

        Returns
        -------
        None (displays plots)
        """
        if isinstance(cis_genes, str):
            cis_genes = [cis_genes]

        plot_edger_vs_bayes_full_range(
            cis_genes, self, de_df,
            fc_thresh=fc_thresh,
            flip_edger_x=flip_edger_x,
            color_scheme=self.get_color_scheme()
        )

    def compare_with_edger_observed(self, cis_genes: Union[str, List[str]],
                                   de_df, fc_thresh: float = 0.5,
                                   flip_edger_x: bool = True,
                                   aggregate_by_guide: bool = True):
        """
        Compare bayesDREAM observed-range log2FC with edgeR results.

        Parameters
        ----------
        cis_genes : str or list of str
            Cis gene name(s)
        de_df : pd.DataFrame
            edgeR results with columns: gene, logFC, FDR, guide
        fc_thresh : float
            Fold-change threshold for classification
        flip_edger_x : bool
            If True, flip sign of edgeR logFC
        aggregate_by_guide : bool
            If True, aggregate x_true by guide before computing min/max

        Returns
        -------
        None (displays plots)
        """
        if isinstance(cis_genes, str):
            cis_genes = [cis_genes]

        plot_edger_vs_bayes_observed_range(
            cis_genes, self, de_df,
            fc_thresh=fc_thresh,
            flip_edger_x=flip_edger_x,
            aggregate_by_guide=aggregate_by_guide,
            color_scheme=self.get_color_scheme()
        )

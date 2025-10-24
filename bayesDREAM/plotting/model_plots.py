"""
Convenience plotting methods for bayesDREAM model parameters.

These are added as methods to the bayesDREAM model class for easy access.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import warnings

from .prior_posterior import (
    plot_scalar_parameter,
    plot_1d_parameter,
    plot_2d_parameter
)
from .prior_sampling import get_prior_samples


class PlottingMixin:
    """Mixin class providing plotting methods for bayesDREAM models."""

    def plot_technical_fit(
        self,
        param: str = 'alpha_y',
        modality_name: Optional[str] = None,
        cell_line_index: Optional[int] = None,
        order_by: str = 'mean',
        subset_features: Optional[List[str]] = None,
        plot_type: str = 'auto',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for technical fit parameters.

        Parameters
        ----------
        param : str
            Parameter to plot: 'beta_o', 'alpha_x', 'alpha_y', 'mu_ntc', 'o_y'
        modality_name : str, optional
            Modality name (default: primary modality)
        cell_line_index : int, optional
            Cell line index for alpha_y (if None, plots all cell lines)
        order_by : str
            Feature ordering: 'mean', 'difference', 'alphabetical', 'input'
        subset_features : List[str], optional
            Subset to specific features
        plot_type : str
            'auto', 'violin', or 'scatter'
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot beta_o (scalar parameter)
        >>> fig = model.plot_technical_fit('beta_o')
        >>>
        >>> # Plot alpha_y for first cell line
        >>> fig = model.plot_technical_fit('alpha_y', cell_line_index=0)
        >>>
        >>> # Plot alpha_y for all cell lines (2D parameter)
        >>> fig = model.plot_technical_fit('alpha_y')
        >>>
        >>> # Plot alpha_y for specific genes
        >>> fig = model.plot_technical_fit('alpha_y', subset_features=['GFI1B', 'TET2'])
        """
        if not hasattr(self, 'posterior_samples_technical'):
            raise ValueError("No technical fit found. Run fit_technical() first.")

        if modality_name is None:
            modality_name = self.primary_modality

        modality = self.get_modality(modality_name)
        posterior = self.posterior_samples_technical

        # Get feature names
        if 'gene' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene'].tolist()
        elif 'gene_name' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene_name'].tolist()
        else:
            feature_names = modality.feature_meta.index.tolist()

        # Sample priors
        prior_dict = get_prior_samples(
            self,
            fit_type='technical',
            modality_name=modality_name,
            nsamples=posterior[list(posterior.keys())[0]].shape[0],  # Match posterior sample count
            distribution=modality.distribution
        )

        # Extract parameter and prior
        if param == 'beta_o':
            # Scalar parameter
            if 'beta_o' not in posterior:
                raise ValueError("beta_o not found in posterior_samples_technical")

            post_samples = posterior['beta_o'].numpy() if hasattr(posterior['beta_o'], 'numpy') else posterior['beta_o']
            prior_samples = prior_dict['beta_o'].numpy() if hasattr(prior_dict['beta_o'], 'numpy') else prior_dict['beta_o']

            return plot_scalar_parameter(prior_samples, post_samples, 'beta_o', **kwargs)

        elif param == 'alpha_x':
            # Scalar or 1D parameter (per cell line)
            if not hasattr(self, 'alpha_x_prefit'):
                raise ValueError("alpha_x not found. Check if cis gene was included in technical fit.")

            post_samples = self.alpha_x_prefit
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # For alpha_x, we need cis priors (will implement below)
            # For now use technical priors as approximation
            if 'alpha_x' in prior_dict:
                prior_samples = prior_dict['alpha_x'].numpy() if hasattr(prior_dict['alpha_x'], 'numpy') else prior_dict['alpha_x']
            else:
                # Fall back to alpha_y structure if alpha_x not in prior_dict
                warnings.warn("alpha_x not in prior_dict - using alpha_y structure as approximation")
                prior_samples = prior_dict['alpha_y'].numpy() if hasattr(prior_dict['alpha_y'], 'numpy') else prior_dict['alpha_y']
                if prior_samples.ndim == 3:
                    prior_samples = prior_samples[:, :, 0]  # Take first gene dimension

            if post_samples.ndim == 1:
                # Scalar (one cell line)
                return plot_scalar_parameter(prior_samples, post_samples, 'alpha_x', **kwargs)
            else:
                # Multiple cell lines
                return plot_scalar_parameter(prior_samples, post_samples, 'alpha_x', **kwargs)

        elif param == 'alpha_y':
            # 1D or 2D parameter: (samples, genes) or (samples, cell_lines, genes)
            if 'alpha_y' not in posterior:
                raise ValueError("alpha_y not found in posterior_samples_technical")

            post_samples = posterior['alpha_y']
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # Use sampled priors
            prior_samples = prior_dict['alpha_y'].numpy() if hasattr(prior_dict['alpha_y'], 'numpy') else prior_dict['alpha_y']

            if post_samples.ndim == 2:
                # (samples, genes)
                return plot_1d_parameter(
                    prior_samples, post_samples, feature_names, 'alpha_y',
                    order_by, subset_features=subset_features, plot_type=plot_type,
                    **kwargs
                )
            elif post_samples.ndim == 3:
                # (samples, cell_lines, genes)
                if cell_line_index is not None:
                    # Plot single cell line
                    prior_cl = prior_samples[:, cell_line_index, :]
                    post_cl = post_samples[:, cell_line_index, :]
                    return plot_1d_parameter(
                        prior_cl, post_cl, feature_names, f'alpha_y (cell_line {cell_line_index})',
                        order_by, subset_features=subset_features, plot_type=plot_type,
                        **kwargs
                    )
                else:
                    # Plot all cell lines (2D plot)
                    n_cell_lines = post_samples.shape[1]
                    cell_line_names = [f'CL_{i}' for i in range(n_cell_lines)]
                    return plot_2d_parameter(
                        prior_samples, post_samples, feature_names, cell_line_names, 'alpha_y',
                        order_by=order_by, subset_features=subset_features,
                        plot_type=plot_type, **kwargs
                    )

        else:
            raise ValueError(f"Unknown parameter: {param}. Must be one of: "
                           "'beta_o', 'alpha_x', 'alpha_y', 'mu_ntc', 'o_y'")

    def plot_cis_fit(
        self,
        param: str = 'x_true',
        order_by: str = 'mean',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for cis fit parameters.

        Parameters
        ----------
        param : str
            Parameter to plot: 'x_true', 'mu_x', 'log2_x_eff'
        order_by : str
            Guide ordering: 'mean', 'difference', 'alphabetical', 'input'
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot x_true (cis gene expression per guide)
        >>> fig = model.plot_cis_fit('x_true')
        """
        if not hasattr(self, 'posterior_samples_cis'):
            raise ValueError("No cis fit found. Run fit_cis() first.")

        posterior = self.posterior_samples_cis

        # Sample priors
        prior_dict = get_prior_samples(
            self,
            fit_type='cis',
            nsamples=posterior[list(posterior.keys())[0]].shape[0]
        )

        # Get guide names
        guide_names = self.meta_guides['guide'].tolist() if hasattr(self, 'meta_guides') else None
        if guide_names is None:
            n_guides = posterior['x_true'].shape[1] if 'x_true' in posterior else posterior['mu_x'].shape[1]
            guide_names = [f'Guide_{i}' for i in range(n_guides)]

        if param == 'x_true':
            if 'x_true' not in posterior:
                raise ValueError("x_true not found in posterior_samples_cis")

            post_samples = posterior['x_true']
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # Use sampled priors
            prior_samples = prior_dict['x_true'].numpy() if hasattr(prior_dict['x_true'], 'numpy') else prior_dict['x_true']

            return plot_1d_parameter(
                prior_samples, post_samples, guide_names, 'x_true',
                order_by, plot_type='violin', **kwargs
            )

        else:
            raise ValueError(f"Unknown parameter: {param}. Must be 'x_true'")

    def plot_trans_fit(
        self,
        param: str = 'theta',
        modality_name: Optional[str] = None,
        subset_features: Optional[List[str]] = None,
        order_by: str = 'mean',
        plot_type: str = 'auto',
        function_type: str = 'additive_hill',
        **kwargs
    ) -> plt.Figure:
        """
        Plot prior vs posterior for trans fit parameters.

        Parameters
        ----------
        param : str
            Parameter to plot: 'theta', 'gamma', 'mu_y'
        modality_name : str, optional
            Modality name (default: primary modality)
        subset_features : List[str], optional
            Subset to specific features
        order_by : str
            Feature ordering
        plot_type : str
            'auto', 'violin', or 'scatter'
        function_type : str
            Function type used in fit: 'additive_hill', 'single_hill', 'polynomial'
        **kwargs
            Additional plotting arguments

        Returns
        -------
        plt.Figure
            Matplotlib figure

        Examples
        --------
        >>> # Plot theta (Hill function parameters)
        >>> fig = model.plot_trans_fit('theta', function_type='additive_hill')
        >>>
        >>> # Plot for specific genes
        >>> fig = model.plot_trans_fit('theta', subset_features=['GFI1B', 'TET2'])
        """
        if not hasattr(self, 'posterior_samples_trans'):
            raise ValueError("No trans fit found. Run fit_trans() first.")

        if modality_name is None:
            modality_name = self.primary_modality

        modality = self.get_modality(modality_name)
        posterior = self.posterior_samples_trans

        # Sample priors
        prior_dict = get_prior_samples(
            self,
            fit_type='trans',
            modality_name=modality_name,
            nsamples=posterior[list(posterior.keys())[0]].shape[0],
            function_type=function_type,
            distribution=modality.distribution
        )

        # Get feature names
        if 'gene' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene'].tolist()
        elif 'gene_name' in modality.feature_meta.columns:
            feature_names = modality.feature_meta['gene_name'].tolist()
        else:
            feature_names = modality.feature_meta.index.tolist()

        if param == 'theta':
            if 'theta' not in posterior:
                raise ValueError("theta not found in posterior_samples_trans")

            post_samples = posterior['theta']
            if hasattr(post_samples, 'numpy'):
                post_samples = post_samples.numpy()

            # Use sampled priors
            prior_samples = prior_dict['theta'].numpy() if hasattr(prior_dict['theta'], 'numpy') else prior_dict['theta']

            # theta is typically (samples, genes, n_params)
            if post_samples.ndim == 3:
                # Plot as 2D parameter with dimensions = parameter indices
                param_names = [f'param_{i}' for i in range(post_samples.shape[2])]
                return plot_2d_parameter(
                    prior_samples, post_samples, feature_names, param_names, 'theta',
                    order_by=order_by, subset_features=subset_features,
                    plot_type=plot_type, **kwargs
                )
            elif post_samples.ndim == 2:
                # Single parameter per gene
                return plot_1d_parameter(
                    prior_samples, post_samples, feature_names, 'theta',
                    order_by, subset_features=subset_features, plot_type=plot_type,
                    **kwargs
                )

        else:
            raise ValueError(f"Unknown parameter: {param}. Must be 'theta', 'gamma', or 'mu_y'")

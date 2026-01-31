"""
Summary export for bayesDREAM results.

Exports model results as R-friendly CSV files with:
- Mean and 95% credible intervals for all parameters
- Cell-wise and feature-wise summaries
- Observed log2FC, predicted log2FC, inflection points for trans fits
- Derivative roots and function classification for additive Hill
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any, Tuple, List
from scipy.optimize import brentq


class ModelSummarizer:
    """
    Export bayesDREAM results as CSV files for plotting.

    Creates summary tables with:
    - Mean and 95% credible intervals for parameters
    - Cell-wise and feature-wise data
    - Compatible with R for downstream plotting
    """

    def __init__(self, model):
        """
        Initialize summarizer with bayesDREAM model.

        Parameters
        ----------
        model : bayesDREAM
            Fitted bayesDREAM model instance
        """
        self.model = model

    # ========================================================================
    # Technical Fit Summary
    # ========================================================================

    def save_technical_summary(
        self,
        output_dir: Optional[str] = None,
        modality_name: Optional[str] = None
    ):
        """
        Save technical fit parameters as feature-wise CSV.

        Creates: technical_feature_summary_{modality}.csv

        Columns:
        - feature: Feature name
        - modality: Modality name
        - distribution: Distribution type
        - group_{i}_alpha_y_mean: Mean alpha_y for group i
        - group_{i}_alpha_y_lower: 2.5% quantile for group i
        - group_{i}_alpha_y_upper: 97.5% quantile for group i

        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: model.output_dir)
        modality_name : str, optional
            Modality to export (default: primary modality)
        """
        if output_dir is None:
            output_dir = self.model.output_dir

        if modality_name is None:
            modality_name = self.model.primary_modality

        # Get modality
        modality = self.model.get_modality(modality_name)

        # Check if technical fit has been run for this modality
        if not hasattr(modality, 'alpha_y_prefit') or modality.alpha_y_prefit is None:
            raise ValueError(f"Technical fit not found for modality '{modality_name}'. Run fit_technical() first.")

        # Get alpha_y for this modality (prefer distribution-specific versions)
        if hasattr(modality, 'alpha_y_prefit_mult') and modality.alpha_y_prefit_mult is not None:
            alpha_y = modality.alpha_y_prefit_mult  # [n_samples, n_groups, n_features]
        elif hasattr(modality, 'alpha_y_prefit_add') and modality.alpha_y_prefit_add is not None:
            alpha_y = modality.alpha_y_prefit_add  # [n_samples, n_groups, n_features]
        else:
            alpha_y = modality.alpha_y_prefit  # [n_samples, n_groups, n_features]

        # Convert to numpy if tensor
        if isinstance(alpha_y, torch.Tensor):
            alpha_y = alpha_y.cpu().numpy()

        # Get feature names
        feature_names = modality.feature_meta.index.tolist()
        n_features = len(feature_names)
        n_groups = alpha_y.shape[1]

        # Compute mean and CI
        alpha_mean = alpha_y.mean(axis=0)  # [n_groups, n_features]
        alpha_lower = np.quantile(alpha_y, 0.025, axis=0)
        alpha_upper = np.quantile(alpha_y, 0.975, axis=0)

        # Build DataFrame
        data = {
            'feature': feature_names,
            'modality': modality_name,
            'distribution': modality.distribution
        }

        # Add columns for each group
        for g in range(n_groups):
            data[f'group_{g}_alpha_y_mean'] = alpha_mean[g, :]
            data[f'group_{g}_alpha_y_lower'] = alpha_lower[g, :]
            data[f'group_{g}_alpha_y_upper'] = alpha_upper[g, :]

        df = pd.DataFrame(data)

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'technical_feature_summary_{modality_name}.csv')
        df.to_csv(output_file, index=False)

        print(f"[SAVE] Technical summary saved to {output_file}")
        print(f"       {n_features} features × {n_groups} groups")

        return df

    # ========================================================================
    # Cis Fit Summary
    # ========================================================================

    def save_cis_summary(
        self,
        output_dir: Optional[str] = None,
        include_cell_level: bool = True
    ):
        """
        Save cis fit parameters as guide-wise and cell-wise CSVs.

        Creates:
        - cis_guide_summary.csv: Guide-level x_true with CI
        - cis_cell_summary.csv: Cell-level data (if include_cell_level=True)

        Guide-level columns:
        - guide: Guide name
        - target: Target gene
        - n_cells: Number of cells
        - x_true_mean: Mean x_true
        - x_true_lower: 2.5% quantile
        - x_true_upper: 97.5% quantile
        - raw_counts_mean: Average raw counts

        Cell-level columns:
        - cell: Cell barcode
        - guide: Guide name
        - target: Target gene
        - cell_line: Cell line (if available)
        - x_true_mean: Mean x_true for this guide
        - x_true_lower: 2.5% quantile
        - x_true_upper: 97.5% quantile
        - raw_counts: Raw counts for this cell

        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: model.output_dir)
        include_cell_level : bool
            Whether to save cell-level summary (default: True)
        """
        if output_dir is None:
            output_dir = self.model.output_dir

        # Check if cis fit has been run
        if not hasattr(self.model, 'x_true') or self.model.x_true is None:
            raise ValueError("Cis fit not found. Run fit_cis() first.")

        # Get x_true samples
        if hasattr(self.model, 'posterior_samples_cis') and 'x_true' in self.model.posterior_samples_cis:
            x_true_samples = self.model.posterior_samples_cis['x_true']  # [n_samples, n_guides]
            if isinstance(x_true_samples, torch.Tensor):
                x_true_samples = x_true_samples.cpu().numpy()
        else:
            # Use point estimate
            x_true = self.model.x_true
            if isinstance(x_true, torch.Tensor):
                x_true = x_true.cpu().numpy()
            x_true_samples = x_true[np.newaxis, :]  # [1, n_guides]

        # Compute mean and CI per guide
        x_true_mean = x_true_samples.mean(axis=0)
        x_true_lower = np.quantile(x_true_samples, 0.025, axis=0)
        x_true_upper = np.quantile(x_true_samples, 0.975, axis=0)

        # Get guide-level metadata
        guide_meta = self.model.meta.groupby('guide').agg({
            'target': 'first',
            'cell': 'count'
        }).rename(columns={'cell': 'n_cells'})

        guides = guide_meta.index.tolist()

        # Get raw counts from cis modality
        cis_mod = self.model.get_modality('cis')
        cis_counts = cis_mod.counts[0, :]  # [n_cells]

        # Map cells to guides
        cell_to_guide = dict(zip(self.model.meta['cell'], self.model.meta['guide']))
        guide_to_cells = self.model.meta.groupby('guide')['cell'].apply(list).to_dict()

        # Compute average raw counts per guide
        raw_counts_mean = []
        for guide in guides:
            guide_cells = guide_to_cells.get(guide, [])
            guide_cell_indices = [i for i, cell in enumerate(self.model.meta['cell']) if cell in guide_cells]
            if len(guide_cell_indices) > 0:
                raw_counts_mean.append(cis_counts[guide_cell_indices].mean())
            else:
                raw_counts_mean.append(np.nan)

        # Build guide-level DataFrame
        guide_df = pd.DataFrame({
            'guide': guides,
            'target': guide_meta['target'].values,
            'n_cells': guide_meta['n_cells'].values,
            'x_true_mean': x_true_mean,
            'x_true_lower': x_true_lower,
            'x_true_upper': x_true_upper,
            'raw_counts_mean': raw_counts_mean
        })

        # Save guide-level
        os.makedirs(output_dir, exist_ok=True)
        guide_file = os.path.join(output_dir, 'cis_guide_summary.csv')
        guide_df.to_csv(guide_file, index=False)

        print(f"[SAVE] Cis guide summary saved to {guide_file}")
        print(f"       {len(guides)} guides")

        # Build cell-level DataFrame if requested
        if include_cell_level:
            # Map guide-level x_true to cells
            guide_to_idx = {g: i for i, g in enumerate(guides)}

            cell_data = {
                'cell': self.model.meta['cell'].values,
                'guide': self.model.meta['guide'].values,
                'target': self.model.meta['target'].values,
            }

            # Add cell_line if available
            if 'cell_line' in self.model.meta.columns:
                cell_data['cell_line'] = self.model.meta['cell_line'].values

            # Add x_true (same for all cells in a guide)
            cell_x_true_mean = np.array([x_true_mean[guide_to_idx[g]] for g in self.model.meta['guide']])
            cell_x_true_lower = np.array([x_true_lower[guide_to_idx[g]] for g in self.model.meta['guide']])
            cell_x_true_upper = np.array([x_true_upper[guide_to_idx[g]] for g in self.model.meta['guide']])

            cell_data['x_true_mean'] = cell_x_true_mean
            cell_data['x_true_lower'] = cell_x_true_lower
            cell_data['x_true_upper'] = cell_x_true_upper
            cell_data['raw_counts'] = cis_counts

            cell_df = pd.DataFrame(cell_data)

            # Save cell-level
            cell_file = os.path.join(output_dir, 'cis_cell_summary.csv')
            cell_df.to_csv(cell_file, index=False)

            print(f"[SAVE] Cis cell summary saved to {cell_file}")
            print(f"       {len(cell_df)} cells")

            return guide_df, cell_df

        return guide_df

    # ========================================================================
    # Trans Fit Summary
    # ========================================================================

    def save_trans_summary(
        self,
        output_dir: Optional[str] = None,
        modality_name: Optional[str] = None,
        compute_inflection: bool = True,
        compute_full_log2fc: bool = True,
        compute_derivative_roots: bool = True,
        use_posterior_samples: bool = False,
        compute_log2fc_params: bool = True
    ):
        """
        Save trans fit parameters as feature-wise CSV.

        Creates: trans_feature_summary_{modality}.csv

        Columns (depend on function_type and distribution):
        - feature: Feature name
        - modality: Modality name
        - distribution: Distribution type
        - function_type: Function type (additive_hill, single_hill, polynomial)
        - x_obs_min, x_obs_max: Observed x range (min/max of guide-level x_true)
        - observed_log2fc: log2(y_max / y_min) over observed x range from fitted function
        - full_log2fc_mean/lower/upper: log2(y_max / y_min) over theoretical x range (0 to ∞)

        For additive_hill (all parameters needed to recreate: y = A + alpha*Vmax_a*Hill(x;K_a,n_a) + beta*Vmax_b*Hill(x;K_b,n_b)):
        - A_mean, A_lower, A_upper: Baseline (intercept)
        - Vmax_a_mean, Vmax_a_lower, Vmax_a_upper: Component A magnitude
        - K_a_mean, K_a_lower, K_a_upper: Component A half-max (EC50)
        - EC50_a_mean, EC50_a_lower, EC50_a_upper: Same as K_a (alias)
        - n_a_mean, n_a_lower, n_a_upper: Component A Hill coefficient (cooperativity)
        - alpha_mean, alpha_lower, alpha_upper: Component A weight
        - Vmax_b_mean, Vmax_b_lower, Vmax_b_upper: Component B magnitude
        - K_b_mean, K_b_lower, K_b_upper: Component B half-max (EC50)
        - EC50_b_mean, EC50_b_lower, EC50_b_upper: Same as K_b (alias)
        - n_b_mean, n_b_lower, n_b_upper: Component B Hill coefficient (cooperativity)
        - beta_mean, beta_lower, beta_upper: Component B weight
        - pi_y_mean, pi_y_lower, pi_y_upper: Sparsity weight (optional)
        - inflection_a_mean, inflection_a_lower, inflection_a_upper: Component A inflection x
        - inflection_b_mean, inflection_b_lower, inflection_b_upper: Component B inflection x
        - classification: Function shape classification based on dependency masks:
          - 'single_positive': Only one Hill component active, with positive n (increasing)
          - 'single_negative': Only one Hill component active, with negative n (decreasing)
          - 'additive_positive': Both components active, both with positive n
          - 'additive_negative': Both components active, both with negative n
          - 'non_monotonic_min': Both active, opposite signs, concave up at x=x_ntc (local min)
          - 'non_monotonic_max': Both active, opposite signs, concave down at x=x_ntc (local max)
          - 'flat': No active components
        - first_deriv_roots_mean: Roots of first derivative (x values where dy/dx=0)
        - second_deriv_roots_mean: Roots of second derivative (x values where d²y/dx²=0, inflections)
        - third_deriv_roots_mean: Roots of third derivative (x values where d³y/dx³=0)
        - n_first_deriv_roots: Number of first derivative roots (0 for monotonic)
        - n_second_deriv_roots: Number of second derivative roots (inflection points)
        - n_third_deriv_roots: Number of third derivative roots

        If compute_log2fc_params=True (log2FC relative to NTC):
        - x_ntc: NTC mean for cis gene (same for all trans genes)
        - y_ntc: NTC mean for each trans gene
        - log2fc_at_u0: log2FC value at u=0 (x = x_ntc), i.e., g(0) = log2(y(x_ntc)) - log2(y_ntc)
        - dg_du_at_u0: First derivative of log2FC at u=0 (dg/du at x = x_ntc)
        - d2g_du2_at_u0: Second derivative of log2FC at u=0 (d²g/du² at x = x_ntc)
        - d3g_du3_at_u0: Third derivative of log2FC at u=0 (d³g/du³ at x = x_ntc)
        - EC50_a_log2fc, EC50_b_log2fc: EC50 in log2FC x-space (log2(K) - log2(x_ntc))
        - inflection_a_log2fc, inflection_b_log2fc: Inflection points in log2FC x-space
        - first_deriv_roots_log2fc, second_deriv_roots_log2fc, third_deriv_roots_log2fc: Roots in log2FC x-space
        - A_log2fc: Baseline in log2FC y-space (log2(A) - log2(y_ntc))

        For single_hill:
        - B_mean, B_lower, B_upper: Hill magnitude
        - K_mean, K_lower, K_upper: Hill coefficient
        - xc_mean, xc_lower, xc_upper: Half-max point (EC50 or IC50)
        - inflection_mean, inflection_lower, inflection_upper: Inflection x
        - full_log2fc_mean, full_log2fc_lower, full_log2fc_upper: Full dynamic range

        For polynomial:
        - coef_{i}_mean, coef_{i}_lower, coef_{i}_upper: Coefficient i
        - full_log2fc_mean, full_log2fc_lower, full_log2fc_upper: Full dynamic range

        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: model.output_dir)
        modality_name : str, optional
            Modality to export (default: primary modality)
        compute_inflection : bool
            Whether to compute inflection points for Hill functions (default: True)
        compute_full_log2fc : bool
            Whether to compute full log2FC range (default: True)
        compute_derivative_roots : bool
            Whether to compute roots of first and second derivatives (default: True).
            Roots are found empirically over the observed x_range.
        use_posterior_samples : bool
            If True, compute derivative roots for each posterior sample and report
            summary statistics. If False (default), compute only for mean parameters.
        compute_log2fc_params : bool
            If True (default), compute parameters in log2FC space relative to NTC:
            - x-axis: log2(x) - log2(x_ntc) where x_ntc is cis gene NTC mean
            - y-axis: log2(y) - log2(y_ntc) where y_ntc is trans gene NTC mean
            Requires posterior_samples_technical to be available.
        """
        if output_dir is None:
            output_dir = self.model.output_dir

        if modality_name is None:
            modality_name = self.model.primary_modality

        # Check if trans fit has been run
        if not hasattr(self.model, 'posterior_samples_trans') or self.model.posterior_samples_trans is None:
            raise ValueError("Trans fit not found. Run fit_trans() first.")

        posterior = self.model.posterior_samples_trans

        # Get modality
        modality = self.model.get_modality(modality_name)

        # Get feature names - prefer named columns over index if index is integer
        feature_meta = modality.feature_meta
        if feature_meta is not None and len(feature_meta) > 0:
            # Check if index is integer-based (RangeIndex or integer dtype)
            index_is_integer = (
                isinstance(feature_meta.index, pd.RangeIndex) or
                feature_meta.index.dtype in ['int64', 'int32', 'int']
            )

            if index_is_integer:
                # Prefer named columns for feature identifiers
                if 'gene' in feature_meta.columns:
                    feature_names = feature_meta['gene'].tolist()
                elif 'gene_name' in feature_meta.columns:
                    feature_names = feature_meta['gene_name'].tolist()
                elif 'feature' in feature_meta.columns:
                    feature_names = feature_meta['feature'].tolist()
                else:
                    # Fall back to index
                    feature_names = feature_meta.index.tolist()
            else:
                # Index is named, use it
                feature_names = feature_meta.index.tolist()
        else:
            # No feature_meta, use range
            feature_names = list(range(modality.counts.shape[0]))

        n_features = len(feature_names)

        # Determine function type from posterior keys
        if 'Vmax_a' in posterior and 'Vmax_b' in posterior:
            function_type = 'additive_hill'
        elif 'params' in posterior:
            function_type = 'single_hill'
        elif 'poly_coefs' in posterior:
            function_type = 'polynomial'
        else:
            raise ValueError(
                f"Cannot determine function_type from posterior_samples_trans keys. "
                f"Found: {list(posterior.keys())}"
            )

        # Initialize DataFrame with basic info
        data = {
            'feature': feature_names,
            'modality': modality_name,
            'distribution': modality.distribution,
            'function_type': function_type
        }

        # Get x_range from model (guide-level x_true values)
        # This defines the "observed" x range as min to max of x_eff_g (guide-level x_true)
        x_range = None
        x_obs_min = None
        x_obs_max = None
        if hasattr(self.model, 'x_true') and self.model.x_true is not None:
            x_true = self.model.x_true
            if isinstance(x_true, torch.Tensor):
                x_true = x_true.cpu().numpy()
            if x_true.ndim > 1:
                x_true = x_true.mean(axis=0)  # Average over posterior samples
            # Store observed x range (min to max of guide-level x_true)
            x_obs_min = max(x_true.min(), 1e-6)
            x_obs_max = x_true.max()
            # Create evenly spaced points in log2 space for derivative root finding
            log2_min = np.log2(x_obs_min)
            log2_max = np.log2(x_obs_max)
            x_range = 2 ** np.linspace(log2_min, log2_max, 2000)
            # Store in data dict
            data['x_obs_min'] = x_obs_min
            data['x_obs_max'] = x_obs_max

        # Get x_ntc and y_ntc for log2FC parameter computation
        x_ntc = None
        y_ntc = None
        if compute_log2fc_params:
            # Get x_ntc from cis modality's technical fit
            cis_mod = self.model.get_modality('cis')
            if hasattr(cis_mod, 'posterior_samples_technical') and cis_mod.posterior_samples_technical is not None:
                if 'mu_ntc' in cis_mod.posterior_samples_technical:
                    mu_ntc_cis = cis_mod.posterior_samples_technical['mu_ntc']
                    if isinstance(mu_ntc_cis, torch.Tensor):
                        mu_ntc_cis = mu_ntc_cis.cpu().numpy()
                    # mu_ntc is [n_samples, n_groups, 1] for cis (single gene)
                    # Average over samples and groups to get scalar x_ntc
                    x_ntc = mu_ntc_cis.mean()
                    data['x_ntc'] = x_ntc

            # Get y_ntc from trans modality's technical fit (per-feature)
            if hasattr(modality, 'posterior_samples_technical') and modality.posterior_samples_technical is not None:
                if 'mu_ntc' in modality.posterior_samples_technical:
                    mu_ntc_trans = modality.posterior_samples_technical['mu_ntc']
                    if isinstance(mu_ntc_trans, torch.Tensor):
                        mu_ntc_trans = mu_ntc_trans.cpu().numpy()
                    # mu_ntc is [n_samples, n_groups, n_features]
                    # Average over samples and groups to get [n_features] y_ntc
                    y_ntc = mu_ntc_trans.mean(axis=(0, 1))
                    data['y_ntc'] = y_ntc

            if x_ntc is None:
                print("[WARNING] Cannot compute log2FC params: x_ntc not available from cis modality")
                compute_log2fc_params = False
            if y_ntc is None:
                print("[WARNING] Cannot compute log2FC params: y_ntc not available from trans modality")
                compute_log2fc_params = False

        # Add function-specific parameters
        if function_type == 'additive_hill':
            data = self._add_additive_hill_params(
                data, posterior, n_features,
                compute_inflection, compute_full_log2fc,
                compute_derivative_roots, use_posterior_samples, x_range,
                compute_log2fc_params, x_ntc, y_ntc,
                x_obs_min, x_obs_max
            )
        elif function_type == 'single_hill':
            data = self._add_single_hill_params(
                data, posterior, n_features,
                compute_inflection, compute_full_log2fc,
                compute_log2fc_params, x_ntc, y_ntc
            )
        elif function_type == 'polynomial':
            data = self._add_polynomial_params(
                data, posterior, n_features,
                compute_full_log2fc
            )

        df = pd.DataFrame(data)

        # Merge with feature_meta from modality
        if modality.feature_meta is not None and len(modality.feature_meta) > 0:
            # Reset index to get feature names as a column for merging
            feature_meta_df = modality.feature_meta.reset_index()
            # Rename index column to avoid conflicts
            if 'index' in feature_meta_df.columns:
                feature_meta_df = feature_meta_df.rename(columns={'index': 'feature_meta_idx'})

            # Ensure feature column is string type for consistent merging
            df['feature'] = df['feature'].astype(str)

            # Merge on feature names - try to match by index or common columns
            if 'feature' in feature_meta_df.columns:
                feature_meta_df['feature'] = feature_meta_df['feature'].astype(str)
                df = df.merge(feature_meta_df, on='feature', how='left')
            elif 'gene' in feature_meta_df.columns:
                feature_meta_df['gene'] = feature_meta_df['gene'].astype(str)
                df = df.merge(feature_meta_df, left_on='feature', right_on='gene', how='left')
            elif 'gene_name' in feature_meta_df.columns:
                feature_meta_df['gene_name'] = feature_meta_df['gene_name'].astype(str)
                df = df.merge(feature_meta_df, left_on='feature', right_on='gene_name', how='left')
            else:
                # Try direct concatenation if feature order matches
                if len(feature_meta_df) == len(df):
                    for col in feature_meta_df.columns:
                        if col not in df.columns:
                            df[col] = feature_meta_df[col].values

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'trans_feature_summary_{modality_name}.csv')
        df.to_csv(output_file, index=False)

        print(f"[SAVE] Trans summary saved to {output_file}")
        print(f"       {n_features} features, function_type={function_type}")

        return df

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _compute_empirical_log2fc(self, modality):
        """Compute empirical log2FC from raw counts (perturbed vs NTC means)."""
        counts = modality.counts  # [n_features, n_cells]

        # Get NTC and perturbed masks
        ntc_mask = (self.model.meta['target'] == 'ntc').values
        pert_mask = (self.model.meta['target'] == self.model.cis_gene).values

        if ntc_mask.sum() == 0 or pert_mask.sum() == 0:
            # Return NaN if no NTC or perturbed cells
            return np.full(counts.shape[0], np.nan), np.full(counts.shape[0], np.nan)

        # Compute mean expression
        ntc_mean = counts[:, ntc_mask].mean(axis=1)
        pert_mean = counts[:, pert_mask].mean(axis=1)

        # Log2 fold change
        log2fc = np.log2((pert_mean + 1) / (ntc_mean + 1))

        # Compute standard error (approximate)
        ntc_std = counts[:, ntc_mask].std(axis=1)
        pert_std = counts[:, pert_mask].std(axis=1)

        n_ntc = ntc_mask.sum()
        n_pert = pert_mask.sum()

        # SE on log scale (delta method approximation)
        se_ntc = ntc_std / (np.sqrt(n_ntc) * (ntc_mean + 1))
        se_pert = pert_std / (np.sqrt(n_pert) * (pert_mean + 1))
        log2fc_se = np.sqrt(se_ntc**2 + se_pert**2) / np.log(2)

        return log2fc, log2fc_se

    def _add_additive_hill_params(self, data, posterior, n_features, compute_inflection, compute_full_log2fc,
                                    compute_derivative_roots=True, use_posterior_samples=False, x_range=None,
                                    compute_log2fc_params=False, x_ntc=None, y_ntc=None,
                                    x_obs_min=None, x_obs_max=None):
        """
        Add additive Hill parameters to data dict.

        Uses individual parameter architecture: Vmax_a, K_a (EC50), n_a, Vmax_b, K_b, n_b.

        For binomial/multinomial:
        - Vmax_a = Vmax_sum * alpha (effective magnitude for component A)
        - Vmax_b = Vmax_sum * beta (effective magnitude for component B)
        - Where Vmax_sum comes from Beta/Dirichlet prior
        - Model: y = A + Vmax_sum * (alpha * Hill_a + beta * Hill_b)

        For negbinom/normal/studentt:
        - Vmax_a and Vmax_b are independent magnitudes sampled from log-normal priors
        """
        # All additive Hill models use individual parameters (Vmax_a, K_a, n_a, etc.)
        if 'Vmax_a' not in posterior:
            raise ValueError(
                f"Cannot find additive Hill parameters in posterior. Expected 'Vmax_a', 'K_a', 'n_a', etc. "
                f"Found keys: {list(posterior.keys())}"
            )

        return self._add_additive_hill_params_individual(
            data, posterior, n_features, compute_inflection, compute_full_log2fc,
            compute_derivative_roots, use_posterior_samples, x_range,
            compute_log2fc_params, x_ntc, y_ntc, x_obs_min, x_obs_max
        )

    def _add_additive_hill_params_individual(self, data, posterior, n_features, compute_inflection, compute_full_log2fc,
                                              compute_derivative_roots=True, use_posterior_samples=False, x_range=None,
                                              compute_log2fc_params=False, x_ntc=None, y_ntc=None,
                                              x_obs_min=None, x_obs_max=None):
        """
        Add additive Hill parameters from individual parameter architecture (Vmax_a, K_a, n_a, etc.).

        Includes:
        - Basic parameters with mean and 95% CI
        - Function classification based on dependency masks
        - Derivative roots (where dy/dx=0 and d²y/dx²=0)
        - full_log2fc: log2(y_max / y_min) over theoretical range (in log2 space)
        - observed_log2fc: log2(y_max / y_min) over observed x range (in log2 space)
        - Log2FC versions of parameters relative to NTC (if compute_log2fc_params=True)
        """

        def extract_param(name):
            """Helper to extract parameter with consistent handling of samples vs point estimates."""
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()

            # Handle different dimensionalities
            # Common shapes:
            # - [n_samples, n_features]: standard 2D
            # - [n_samples, 1, n_features]: has cis gene dimension (size 1) to squeeze
            # - [n_samples, n_features, n_categories]: multinomial, average over categories
            if param.ndim >= 2:
                if param.ndim == 3:
                    # Check if middle dimension is size 1 (cis gene dim to squeeze)
                    if param.shape[1] == 1:
                        param = param.squeeze(1)  # [n_samples, 1, n_features] -> [n_samples, n_features]
                    else:
                        # Otherwise it's [n_samples, n_features, n_categories] - average over categories
                        param = param.mean(axis=2)
                elif param.ndim > 3:
                    # Higher dimensions: squeeze size-1 dims, then average remaining extras
                    # First try to squeeze the cis gene dimension (axis 1) if size 1
                    if param.shape[1] == 1:
                        param = param.squeeze(1)
                    # Average over any remaining dimensions beyond 2
                    if param.ndim > 2:
                        param = param.mean(axis=tuple(range(2, param.ndim)))

                mean_val = param.mean(axis=0)
                lower_val = np.quantile(param, 0.025, axis=0)
                upper_val = np.quantile(param, 0.975, axis=0)
            else:
                # Point estimate: [n_features]
                mean_val = param
                lower_val = param
                upper_val = param

            return mean_val, lower_val, upper_val

        def extract_param_full(name):
            """Extract full posterior samples for per-sample computations."""
            param = posterior[name]
            if isinstance(param, torch.Tensor):
                param = param.cpu().numpy()
            # Handle [n_samples, 1, n_features] -> [n_samples, n_features]
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            elif param.ndim > 2:
                # For other 3D+ cases, average over extra dimensions
                if param.shape[1] == 1:
                    param = param.squeeze(1)
                if param.ndim > 2:
                    param = param.mean(axis=tuple(range(2, param.ndim)))
            return param  # [n_samples, n_features] or [n_features]

        # Component A (first Hill function)
        Vmax_a_mean, Vmax_a_lower, Vmax_a_upper = extract_param('Vmax_a')
        K_a_mean, K_a_lower, K_a_upper = extract_param('K_a')  # EC50
        n_a_mean, n_a_lower, n_a_upper = extract_param('n_a')  # Hill coefficient

        data['Vmax_a_mean'] = Vmax_a_mean
        data['Vmax_a_lower'] = Vmax_a_lower
        data['Vmax_a_upper'] = Vmax_a_upper
        data['EC50_a_mean'] = K_a_mean
        data['EC50_a_lower'] = K_a_lower
        data['EC50_a_upper'] = K_a_upper
        data['n_a_mean'] = n_a_mean
        data['n_a_lower'] = n_a_lower
        data['n_a_upper'] = n_a_upper

        # Component B (second Hill function)
        Vmax_b_mean, Vmax_b_lower, Vmax_b_upper = extract_param('Vmax_b')
        K_b_mean, K_b_lower, K_b_upper = extract_param('K_b')  # EC50
        n_b_mean, n_b_lower, n_b_upper = extract_param('n_b')  # Hill coefficient

        data['Vmax_b_mean'] = Vmax_b_mean
        data['Vmax_b_lower'] = Vmax_b_lower
        data['Vmax_b_upper'] = Vmax_b_upper
        data['EC50_b_mean'] = K_b_mean
        data['EC50_b_lower'] = K_b_lower
        data['EC50_b_upper'] = K_b_upper
        data['n_b_mean'] = n_b_mean
        data['n_b_lower'] = n_b_lower
        data['n_b_upper'] = n_b_upper

        # Pi_y (sparsity weight) - optional
        if 'pi_y' in posterior:
            pi_y_mean, pi_y_lower, pi_y_upper = extract_param('pi_y')
            data['pi_y_mean'] = pi_y_mean
            data['pi_y_lower'] = pi_y_lower
            data['pi_y_upper'] = pi_y_upper

        # Alpha and beta (component weights)
        # Default to 1.0 if not present (no sparsity)
        # Note: alpha/beta may be scalars (shared across features) or per-feature
        def _broadcast_to_features(arr, n_features):
            """Broadcast scalar or size-1 array to n_features length."""
            arr = np.atleast_1d(arr)
            if arr.size == 1:
                return np.full(n_features, arr.flat[0])
            return arr

        if 'alpha' in posterior:
            alpha_mean, alpha_lower, alpha_upper = extract_param('alpha')
            alpha_full = extract_param_full('alpha')
            # Broadcast if scalar
            alpha_mean = _broadcast_to_features(alpha_mean, n_features)
            alpha_lower = _broadcast_to_features(alpha_lower, n_features)
            alpha_upper = _broadcast_to_features(alpha_upper, n_features)
            if alpha_full.ndim == 1 or alpha_full.shape[-1] == 1:
                alpha_full = np.broadcast_to(alpha_full.reshape(-1, 1), (alpha_full.shape[0] if alpha_full.ndim > 1 else 1, n_features)).copy()
        else:
            alpha_mean = np.ones(n_features)
            alpha_lower = np.ones(n_features)
            alpha_upper = np.ones(n_features)
            alpha_full = np.ones((1, n_features))

        if 'beta' in posterior:
            beta_mean, beta_lower, beta_upper = extract_param('beta')
            beta_full = extract_param_full('beta')
            # Broadcast if scalar
            beta_mean = _broadcast_to_features(beta_mean, n_features)
            beta_lower = _broadcast_to_features(beta_lower, n_features)
            beta_upper = _broadcast_to_features(beta_upper, n_features)
            if beta_full.ndim == 1 or beta_full.shape[-1] == 1:
                beta_full = np.broadcast_to(beta_full.reshape(-1, 1), (beta_full.shape[0] if beta_full.ndim > 1 else 1, n_features)).copy()
        else:
            beta_mean = np.ones(n_features)
            beta_lower = np.ones(n_features)
            beta_upper = np.ones(n_features)
            beta_full = np.ones((1, n_features))

        data['alpha_mean'] = alpha_mean
        data['alpha_lower'] = alpha_lower
        data['alpha_upper'] = alpha_upper
        data['beta_mean'] = beta_mean
        data['beta_lower'] = beta_lower
        data['beta_upper'] = beta_upper

        # Also save K_a and K_b explicitly (same as EC50 but clearer naming for function recreation)
        data['K_a_mean'] = K_a_mean
        data['K_a_lower'] = K_a_lower
        data['K_a_upper'] = K_a_upper
        data['K_b_mean'] = K_b_mean
        data['K_b_lower'] = K_b_lower
        data['K_b_upper'] = K_b_upper

        # Get A (baseline) - required to recreate the fitted function
        if 'A' in posterior:
            A_mean, A_lower, A_upper = extract_param('A')
            A_full = extract_param_full('A')
            # Broadcast if scalar
            A_mean = _broadcast_to_features(A_mean, n_features)
            A_lower = _broadcast_to_features(A_lower, n_features)
            A_upper = _broadcast_to_features(A_upper, n_features)
            if A_full.ndim == 1 or A_full.shape[-1] == 1:
                A_full = np.broadcast_to(A_full.reshape(-1, 1), (A_full.shape[0] if A_full.ndim > 1 else 1, n_features)).copy()
        else:
            A_mean = np.zeros(n_features)
            A_lower = np.zeros(n_features)
            A_upper = np.zeros(n_features)
            A_full = np.zeros((1, n_features))

        # Save A (baseline) to data dict
        data['A_mean'] = A_mean
        data['A_lower'] = A_lower
        data['A_upper'] = A_upper

        # Compute inflection points for individual Hill components
        if compute_inflection:
            # Inflection point for Hill function: x_inflection = EC50 * ((n - 1) / (n + 1)) ^ (1/n)
            inflection_a_mean = self._compute_hill_inflection(n_a_mean, K_a_mean)
            inflection_a_lower = self._compute_hill_inflection(n_a_lower, K_a_lower)
            inflection_a_upper = self._compute_hill_inflection(n_a_upper, K_a_upper)

            inflection_b_mean = self._compute_hill_inflection(n_b_mean, K_b_mean)
            inflection_b_lower = self._compute_hill_inflection(n_b_lower, K_b_lower)
            inflection_b_upper = self._compute_hill_inflection(n_b_upper, K_b_upper)

            data['inflection_a_mean'] = inflection_a_mean
            data['inflection_a_lower'] = inflection_a_lower
            data['inflection_a_upper'] = inflection_a_upper
            data['inflection_b_mean'] = inflection_b_mean
            data['inflection_b_lower'] = inflection_b_lower
            data['inflection_b_upper'] = inflection_b_upper

        # Get full posterior samples for derivative computation
        Vmax_a_full = extract_param_full('Vmax_a')
        K_a_full = extract_param_full('K_a')
        n_a_full = extract_param_full('n_a')
        Vmax_b_full = extract_param_full('Vmax_b')
        K_b_full = extract_param_full('K_b')
        n_b_full = extract_param_full('n_b')

        # Initialize arrays for per-feature results
        classifications = []
        first_deriv_roots_mean_list = []
        second_deriv_roots_mean_list = []
        third_deriv_roots_mean_list = []
        n_first_deriv_roots = []
        n_second_deriv_roots = []
        n_third_deriv_roots = []
        full_log2fc_mean_list = []
        full_log2fc_lower_list = []
        full_log2fc_upper_list = []
        observed_log2fc_list = []  # log2(y_max / y_min) over observed x range

        # Per-sample results (if requested)
        if use_posterior_samples:
            first_deriv_roots_samples_list = []
            second_deriv_roots_samples_list = []
            third_deriv_roots_samples_list = []

        # Process each feature
        for i in range(n_features):
            # Get mean parameters for this feature
            alpha_i = alpha_mean[i]
            beta_i = beta_mean[i]
            Vmax_a_i = Vmax_a_mean[i]
            Vmax_b_i = Vmax_b_mean[i]
            K_a_i = K_a_mean[i]
            K_b_i = K_b_mean[i]
            n_a_i = n_a_mean[i]
            n_b_i = n_b_mean[i]
            A_i = A_mean[i]

            # Find derivative roots for mean parameters
            first_roots_mean = []
            second_roots_mean = []
            third_roots_mean = []

            if compute_derivative_roots and x_range is not None:
                # First derivative roots (where dy/dx = 0)
                def first_deriv_func(x):
                    return self._additive_hill_first_derivative(
                        x, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )
                first_roots_mean = self._find_roots_empirical(first_deriv_func, x_range)

                # Second derivative roots (inflection points of combined function)
                def second_deriv_func(x):
                    return self._additive_hill_second_derivative(
                        x, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )
                second_roots_mean = self._find_roots_empirical(second_deriv_func, x_range)

                # Third derivative roots (where d³y/dx³ = 0)
                def third_deriv_func(x):
                    return self._additive_hill_third_derivative(
                        x, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )
                third_roots_mean = self._find_roots_empirical(third_deriv_func, x_range)

            first_deriv_roots_mean_list.append(first_roots_mean)
            second_deriv_roots_mean_list.append(second_roots_mean)
            third_deriv_roots_mean_list.append(third_roots_mean)
            n_first_deriv_roots.append(len(first_roots_mean))
            n_second_deriv_roots.append(len(second_roots_mean))
            n_third_deriv_roots.append(len(third_roots_mean))

            # Classify function
            classification = self._classify_additive_hill(
                alpha_i, alpha_lower[i], alpha_upper[i],
                beta_i, beta_lower[i], beta_upper[i],
                n_a_i, n_a_lower[i], n_a_upper[i],
                n_b_i, n_b_lower[i], n_b_upper[i],
                first_roots_mean, second_roots_mean,
                x_range if x_range is not None else np.array([1.0]),
                Vmax_a_i, K_a_i, Vmax_b_i, K_b_i,
                x_ntc=x_ntc  # Pass x_ntc for non-monotonic classification
            )
            classifications.append(classification)

            # Compute full_log2fc with CORRECTED formula based on classification
            # Accounts for sign of n (positive n = activator, negative n = inhibitor)
            # Always use the comprehensive function which handles all cases
            full_log2fc_i = self._compute_full_log2fc_additive(
                A_i, alpha_i, Vmax_a_i, beta_i, Vmax_b_i,
                x_range if x_range is not None else np.linspace(0.1, 100, 100),
                K_a_i, n_a_i, K_b_i, n_b_i, classification,
                first_deriv_roots=first_roots_mean
            )

            full_log2fc_mean_list.append(full_log2fc_i)

            # Compute observed_log2fc over observed x range (min to max x_eff_g)
            if x_obs_min is not None and x_obs_max is not None:
                obs_log2fc_i = self._compute_observed_log2fc_fitted(
                    A_i, alpha_i, Vmax_a_i, beta_i, Vmax_b_i,
                    K_a_i, n_a_i, K_b_i, n_b_i,
                    x_obs_min, x_obs_max
                )
                observed_log2fc_list.append(obs_log2fc_i)
            else:
                observed_log2fc_list.append(np.nan)

            # Compute CI for full_log2fc from posterior samples (in log2 space)
            if Vmax_a_full.ndim > 1:
                # Have posterior samples
                n_samples = Vmax_a_full.shape[0]
                sample_log2fcs = []
                epsilon = 1e-10
                for s in range(n_samples):
                    alpha_s = alpha_full[s, i] if alpha_full.ndim > 1 else alpha_full[i]
                    beta_s = beta_full[s, i] if beta_full.ndim > 1 else beta_full[i]
                    A_s = A_full[s, i] if A_full.ndim > 1 else A_full[i]
                    Vmax_a_s = Vmax_a_full[s, i]
                    Vmax_b_s = Vmax_b_full[s, i]
                    n_a_s = n_a_full[s, i]
                    n_b_s = n_b_full[s, i]

                    # Compute boundary values based on signs of n
                    hill_a_at_0 = Vmax_a_s if n_a_s < 0 else 0
                    hill_a_at_inf = 0 if n_a_s < 0 else Vmax_a_s
                    hill_b_at_0 = Vmax_b_s if n_b_s < 0 else 0
                    hill_b_at_inf = 0 if n_b_s < 0 else Vmax_b_s

                    y_at_0 = A_s + alpha_s * hill_a_at_0 + beta_s * hill_b_at_0
                    y_at_inf = A_s + alpha_s * hill_a_at_inf + beta_s * hill_b_at_inf

                    # Compute in log2 space: log2(y_max / y_min)
                    y_max = max(y_at_0, y_at_inf, epsilon)
                    y_min = max(min(y_at_0, y_at_inf), epsilon)
                    sample_log2fcs.append(np.log2(y_max / y_min))

                full_log2fc_lower_list.append(np.quantile(sample_log2fcs, 0.025))
                full_log2fc_upper_list.append(np.quantile(sample_log2fcs, 0.975))
            else:
                # Point estimate only
                full_log2fc_lower_list.append(full_log2fc_i)
                full_log2fc_upper_list.append(full_log2fc_i)

            # Per-sample derivative root computation (only if BOTH use_posterior_samples AND compute_derivative_roots)
            if use_posterior_samples and compute_derivative_roots and x_range is not None and Vmax_a_full.ndim > 1:
                n_samples = min(Vmax_a_full.shape[0], 1000)  # Limit to 1000 samples
                sample_first_roots = []
                sample_second_roots = []
                sample_third_roots = []

                for s in range(n_samples):
                    alpha_s = alpha_full[s, i] if alpha_full.ndim > 1 else alpha_full[i]
                    beta_s = beta_full[s, i] if beta_full.ndim > 1 else beta_full[i]
                    Vmax_a_s = Vmax_a_full[s, i]
                    Vmax_b_s = Vmax_b_full[s, i]
                    K_a_s = K_a_full[s, i]
                    K_b_s = K_b_full[s, i]
                    n_a_s = n_a_full[s, i]
                    n_b_s = n_b_full[s, i]

                    def first_deriv_s(x):
                        return self._additive_hill_first_derivative(
                            x, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                            beta_s, Vmax_b_s, K_b_s, n_b_s
                        )

                    def second_deriv_s(x):
                        return self._additive_hill_second_derivative(
                            x, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                            beta_s, Vmax_b_s, K_b_s, n_b_s
                        )

                    def third_deriv_s(x):
                        return self._additive_hill_third_derivative(
                            x, alpha_s, Vmax_a_s, K_a_s, n_a_s,
                            beta_s, Vmax_b_s, K_b_s, n_b_s
                        )

                    sample_first_roots.append(self._find_roots_empirical(first_deriv_s, x_range))
                    sample_second_roots.append(self._find_roots_empirical(second_deriv_s, x_range))
                    sample_third_roots.append(self._find_roots_empirical(third_deriv_s, x_range))

                first_deriv_roots_samples_list.append(sample_first_roots)
                second_deriv_roots_samples_list.append(sample_second_roots)
                third_deriv_roots_samples_list.append(sample_third_roots)

        # Add computed values to data dict
        data['classification'] = classifications
        data['n_first_deriv_roots'] = n_first_deriv_roots
        data['n_second_deriv_roots'] = n_second_deriv_roots
        data['n_third_deriv_roots'] = n_third_deriv_roots

        # Store derivative roots as strings (list of x values)
        # For mean parameters
        data['first_deriv_roots_mean'] = [
            ';'.join([f'{r:.4f}' for r in roots]) if roots else ''
            for roots in first_deriv_roots_mean_list
        ]
        data['second_deriv_roots_mean'] = [
            ';'.join([f'{r:.4f}' for r in roots]) if roots else ''
            for roots in second_deriv_roots_mean_list
        ]
        data['third_deriv_roots_mean'] = [
            ';'.join([f'{r:.4f}' for r in roots]) if roots else ''
            for roots in third_deriv_roots_mean_list
        ]

        # Compute summary of derivative roots across posterior samples
        if use_posterior_samples and compute_derivative_roots and x_range is not None:
            # For first derivative roots: report median and CI of first root (if any)
            first_root_medians = []
            first_root_lowers = []
            first_root_uppers = []
            second_root_medians = []
            second_root_lowers = []
            second_root_uppers = []
            third_root_medians = []
            third_root_lowers = []
            third_root_uppers = []

            for i in range(n_features):
                if i < len(first_deriv_roots_samples_list):
                    # Collect first root from each sample (if exists)
                    first_roots_flat = [roots[0] for roots in first_deriv_roots_samples_list[i] if len(roots) > 0]
                    if first_roots_flat:
                        first_root_medians.append(np.median(first_roots_flat))
                        first_root_lowers.append(np.quantile(first_roots_flat, 0.025))
                        first_root_uppers.append(np.quantile(first_roots_flat, 0.975))
                    else:
                        first_root_medians.append(np.nan)
                        first_root_lowers.append(np.nan)
                        first_root_uppers.append(np.nan)

                    # Same for second derivative
                    second_roots_flat = [roots[0] for roots in second_deriv_roots_samples_list[i] if len(roots) > 0]
                    if second_roots_flat:
                        second_root_medians.append(np.median(second_roots_flat))
                        second_root_lowers.append(np.quantile(second_roots_flat, 0.025))
                        second_root_uppers.append(np.quantile(second_roots_flat, 0.975))
                    else:
                        second_root_medians.append(np.nan)
                        second_root_lowers.append(np.nan)
                        second_root_uppers.append(np.nan)

                    # Same for third derivative
                    if i < len(third_deriv_roots_samples_list):
                        third_roots_flat = [roots[0] for roots in third_deriv_roots_samples_list[i] if len(roots) > 0]
                        if third_roots_flat:
                            third_root_medians.append(np.median(third_roots_flat))
                            third_root_lowers.append(np.quantile(third_roots_flat, 0.025))
                            third_root_uppers.append(np.quantile(third_roots_flat, 0.975))
                        else:
                            third_root_medians.append(np.nan)
                            third_root_lowers.append(np.nan)
                            third_root_uppers.append(np.nan)
                    else:
                        third_root_medians.append(np.nan)
                        third_root_lowers.append(np.nan)
                        third_root_uppers.append(np.nan)
                else:
                    first_root_medians.append(np.nan)
                    first_root_lowers.append(np.nan)
                    first_root_uppers.append(np.nan)
                    second_root_medians.append(np.nan)
                    second_root_lowers.append(np.nan)
                    second_root_uppers.append(np.nan)
                    third_root_medians.append(np.nan)
                    third_root_lowers.append(np.nan)
                    third_root_uppers.append(np.nan)

            data['first_deriv_root1_median'] = first_root_medians
            data['first_deriv_root1_lower'] = first_root_lowers
            data['first_deriv_root1_upper'] = first_root_uppers
            data['second_deriv_root1_median'] = second_root_medians
            data['second_deriv_root1_lower'] = second_root_lowers
            data['second_deriv_root1_upper'] = second_root_uppers
            data['third_deriv_root1_median'] = third_root_medians
            data['third_deriv_root1_lower'] = third_root_lowers
            data['third_deriv_root1_upper'] = third_root_uppers

        # Compute full log2FC (theoretical range) and observed log2FC (observed x range)
        # Both are in log2 space: log2(y_max / y_min)
        if compute_full_log2fc:
            data['full_log2fc_mean'] = full_log2fc_mean_list
            data['full_log2fc_lower'] = full_log2fc_lower_list
            data['full_log2fc_upper'] = full_log2fc_upper_list

        # Observed log2FC over the observed x range (min to max x_eff_g)
        data['observed_log2fc'] = observed_log2fc_list

        # Compute log2FC versions of parameters relative to NTC
        if compute_log2fc_params and x_ntc is not None and y_ntc is not None:
            epsilon = 1e-10  # Avoid log(0)
            ln2 = np.log(2)

            # Log2FC x-transformation: u = log2(x) - log2(x_ntc)
            log2_x_ntc = np.log2(max(x_ntc, epsilon))

            # =====================================================================
            # Compute log2FC (g), dg/du, d²g/du² at u=0 (x = x_ntc)
            # =====================================================================
            # At u=0, x = x_ntc
            # g(u) = log2(y(x)) - log2(y_ntc)
            #
            # Chain rule derivation:
            # u = log2(x) - log2(x_ntc), so x = x_ntc * 2^u, dx/du = x * ln(2)
            # g = log2(y) - const = ln(y)/ln(2) - const
            # dg/du = (1/ln(2)) * (1/y) * dy/dx * dx/du
            #       = (1/ln(2)) * (1/y) * S'(x) * x * ln(2)
            #       = x * S'(x) / y
            #
            # d²g/du² = d/du [x * S'/S] = dx/du * d/dx [x * S'/S]
            #         = x*ln(2) * [S'/S + x*S''/S - x*(S'/S)²]
            #         = ln(2) * [x*S'/S + x²*S''/S - x²*(S'/S)²]
            #
            # d³g/du³ = ln(2) * (x²*S'''/S - 3x²*S'*S''/S² + 3x*S''/S
            #                    + 2x²*S'³/S³ - 3x*S'²/S² + S'/S)

            log2fc_at_0 = np.full(n_features, np.nan)
            dg_du_at_0 = np.full(n_features, np.nan)
            d2g_du2_at_0 = np.full(n_features, np.nan)
            d3g_du3_at_0 = np.full(n_features, np.nan)

            for i in range(n_features):
                # Get mean parameters for this feature
                alpha_i = alpha_mean[i]
                beta_i = beta_mean[i]
                Vmax_a_i = Vmax_a_mean[i]
                Vmax_b_i = Vmax_b_mean[i]
                K_a_i = K_a_mean[i]
                K_b_i = K_b_mean[i]
                n_a_i = n_a_mean[i]
                n_b_i = n_b_mean[i]
                A_i = A_mean[i]
                y_ntc_i = y_ntc[i] if y_ntc is not None else 1.0

                # Compute y(x_ntc) using Hill function
                def _hill_value(x, Vmax, K, n):
                    x_safe = max(x, epsilon)
                    K_safe = max(K, epsilon)
                    x_n = x_safe ** n
                    K_n = K_safe ** n
                    return Vmax * x_n / (K_n + x_n + epsilon)

                H_a_at_ntc = _hill_value(x_ntc, Vmax_a_i, K_a_i, n_a_i)
                H_b_at_ntc = _hill_value(x_ntc, Vmax_b_i, K_b_i, n_b_i)
                y_at_ntc = A_i + alpha_i * H_a_at_ntc + beta_i * H_b_at_ntc

                # log2FC at u=0: g(0) = log2(y(x_ntc)) - log2(y_ntc)
                if y_at_ntc > epsilon and y_ntc_i > epsilon:
                    log2fc_at_0[i] = np.log2(y_at_ntc) - np.log2(y_ntc_i)

                    # First derivative S'(x_ntc)
                    S_prime = self._additive_hill_first_derivative(
                        x_ntc, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )

                    # Second derivative S''(x_ntc)
                    S_double_prime = self._additive_hill_second_derivative(
                        x_ntc, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )

                    # dg/du = x * S'(x) / S(x) at x = x_ntc
                    # Note: S(x) = y(x), so we use y_at_ntc
                    dg_du_at_0[i] = x_ntc * S_prime / y_at_ntc

                    # d²g/du² = ln(2) * [x*S'/S + x²*S''/S - x²*(S'/S)²]
                    term1 = x_ntc * S_prime / y_at_ntc
                    term2 = (x_ntc ** 2) * S_double_prime / y_at_ntc
                    term3 = (x_ntc ** 2) * (S_prime / y_at_ntc) ** 2
                    d2g_du2_at_0[i] = ln2 * (term1 + term2 - term3)

                    # Third derivative S'''(x_ntc)
                    S_triple_prime = self._additive_hill_third_derivative(
                        x_ntc, alpha_i, Vmax_a_i, K_a_i, n_a_i,
                        beta_i, Vmax_b_i, K_b_i, n_b_i
                    )

                    # d³g/du³ = ln(2) * (x²*S'''/S - 3x²*S'*S''/S² + 3x*S''/S
                    #                    + 2x²*S'³/S³ - 3x*S'²/S² + S'/S)
                    S_ratio = S_prime / y_at_ntc  # S'/S
                    S2_ratio = S_double_prime / y_at_ntc  # S''/S
                    S3_ratio = S_triple_prime / y_at_ntc  # S'''/S
                    x2 = x_ntc ** 2

                    t1 = x2 * S3_ratio                   # x²*S'''/S
                    t2 = -3 * x2 * S_ratio * S2_ratio    # -3x²*S'*S''/S²
                    t3 = 3 * x_ntc * S2_ratio            # 3x*S''/S
                    t4 = 2 * x2 * S_ratio ** 3           # 2x²*S'³/S³
                    t5 = -3 * x_ntc * S_ratio ** 2       # -3x*S'²/S²
                    t6 = S_ratio                         # S'/S

                    d3g_du3_at_0[i] = ln2 * (t1 + t2 + t3 + t4 + t5 + t6)

            data['log2fc_at_u0'] = log2fc_at_0
            data['dg_du_at_u0'] = dg_du_at_0
            data['d2g_du2_at_u0'] = d2g_du2_at_0
            data['d3g_du3_at_u0'] = d3g_du3_at_0

            # EC50 in log2FC space: log2(K) - log2(x_ntc)
            data['EC50_a_log2fc'] = np.log2(np.maximum(K_a_mean, epsilon)) - log2_x_ntc
            data['EC50_b_log2fc'] = np.log2(np.maximum(K_b_mean, epsilon)) - log2_x_ntc

            # Inflection points in log2FC space
            if compute_inflection:
                inflection_a_mean = data.get('inflection_a_mean', np.full(n_features, np.nan))
                inflection_b_mean = data.get('inflection_b_mean', np.full(n_features, np.nan))

                # Handle NaN values properly for log transformation
                inflection_a_log2fc = np.full(n_features, np.nan)
                inflection_b_log2fc = np.full(n_features, np.nan)
                valid_a = ~np.isnan(inflection_a_mean) & (inflection_a_mean > epsilon)
                valid_b = ~np.isnan(inflection_b_mean) & (inflection_b_mean > epsilon)

                if np.any(valid_a):
                    inflection_a_log2fc[valid_a] = np.log2(inflection_a_mean[valid_a]) - log2_x_ntc
                if np.any(valid_b):
                    inflection_b_log2fc[valid_b] = np.log2(inflection_b_mean[valid_b]) - log2_x_ntc

                data['inflection_a_log2fc'] = inflection_a_log2fc
                data['inflection_b_log2fc'] = inflection_b_log2fc

            # Derivative roots in log2FC space
            if compute_derivative_roots:
                first_deriv_roots_log2fc = []
                second_deriv_roots_log2fc = []
                third_deriv_roots_log2fc = []

                for i in range(n_features):
                    # First derivative roots
                    roots_1 = first_deriv_roots_mean_list[i]
                    if roots_1:
                        roots_1_log2fc = [np.log2(max(r, epsilon)) - log2_x_ntc for r in roots_1]
                        first_deriv_roots_log2fc.append(';'.join([f'{r:.4f}' for r in roots_1_log2fc]))
                    else:
                        first_deriv_roots_log2fc.append('')

                    # Second derivative roots
                    roots_2 = second_deriv_roots_mean_list[i]
                    if roots_2:
                        roots_2_log2fc = [np.log2(max(r, epsilon)) - log2_x_ntc for r in roots_2]
                        second_deriv_roots_log2fc.append(';'.join([f'{r:.4f}' for r in roots_2_log2fc]))
                    else:
                        second_deriv_roots_log2fc.append('')

                    # Third derivative roots
                    roots_3 = third_deriv_roots_mean_list[i]
                    if roots_3:
                        roots_3_log2fc = [np.log2(max(r, epsilon)) - log2_x_ntc for r in roots_3]
                        third_deriv_roots_log2fc.append(';'.join([f'{r:.4f}' for r in roots_3_log2fc]))
                    else:
                        third_deriv_roots_log2fc.append('')

                data['first_deriv_roots_log2fc'] = first_deriv_roots_log2fc
                data['second_deriv_roots_log2fc'] = second_deriv_roots_log2fc
                data['third_deriv_roots_log2fc'] = third_deriv_roots_log2fc

            # A (baseline) in log2FC y-space: log2(A) - log2(y_ntc)
            # y_ntc is per-feature, A_mean is per-feature
            A_log2fc = np.full(n_features, np.nan)
            valid_A = (A_mean > epsilon) & (y_ntc > epsilon)
            if np.any(valid_A):
                A_log2fc[valid_A] = np.log2(A_mean[valid_A]) - np.log2(y_ntc[valid_A])
            data['A_log2fc'] = A_log2fc

        return data

    def _add_single_hill_params(self, data, posterior, n_features, compute_inflection, compute_full_log2fc,
                                  compute_log2fc_params=False, x_ntc=None, y_ntc=None):
        """Add single Hill parameters to data dict."""
        params = posterior['params']  # [n_samples, n_features, 4] or [n_features, 4]
        if isinstance(params, torch.Tensor):
            params = params.cpu().numpy()

        if params.ndim == 3:
            B_mean = params[:, :, 0].mean(axis=0)
            B_lower = np.quantile(params[:, :, 0], 0.025, axis=0)
            B_upper = np.quantile(params[:, :, 0], 0.975, axis=0)

            K_mean = params[:, :, 1].mean(axis=0)
            K_lower = np.quantile(params[:, :, 1], 0.025, axis=0)
            K_upper = np.quantile(params[:, :, 1], 0.975, axis=0)

            xc_mean = params[:, :, 2].mean(axis=0)
            xc_lower = np.quantile(params[:, :, 2], 0.025, axis=0)
            xc_upper = np.quantile(params[:, :, 2], 0.975, axis=0)

            # Get n for inflection calculation (if available)
            n_mean = params[:, :, 3].mean(axis=0) if params.shape[-1] > 3 else np.ones(n_features)
        else:
            B_mean = params[:, 0]
            B_lower = B_mean
            B_upper = B_mean

            K_mean = params[:, 1]
            K_lower = K_mean
            K_upper = K_mean

            xc_mean = params[:, 2]
            xc_lower = xc_mean
            xc_upper = xc_mean

            n_mean = params[:, 3] if params.shape[-1] > 3 else np.ones(n_features)

        data['B_mean'] = B_mean
        data['B_lower'] = B_lower
        data['B_upper'] = B_upper
        data['K_mean'] = K_mean
        data['K_lower'] = K_lower
        data['K_upper'] = K_upper
        data['xc_mean'] = xc_mean
        data['xc_lower'] = xc_lower
        data['xc_upper'] = xc_upper

        if compute_inflection:
            inflection_mean = self._compute_hill_inflection(K_mean, xc_mean)
            inflection_lower = self._compute_hill_inflection(K_lower, xc_lower)
            inflection_upper = self._compute_hill_inflection(K_upper, xc_upper)

            data['inflection_mean'] = inflection_mean
            data['inflection_lower'] = inflection_lower
            data['inflection_upper'] = inflection_upper

        if compute_full_log2fc:
            data['full_log2fc_mean'] = np.abs(B_mean)
            data['full_log2fc_lower'] = np.abs(B_lower)
            data['full_log2fc_upper'] = np.abs(B_upper)

        # Compute log2FC versions of parameters relative to NTC
        if compute_log2fc_params and x_ntc is not None and y_ntc is not None:
            epsilon = 1e-10  # Avoid log(0)

            # Log2FC x-transformation: u = log2(x) - log2(x_ntc)
            log2_x_ntc = np.log2(max(x_ntc, epsilon))

            # EC50 (xc) in log2FC space: log2(xc) - log2(x_ntc)
            data['EC50_log2fc'] = np.log2(np.maximum(xc_mean, epsilon)) - log2_x_ntc

            # Inflection point in log2FC space
            if compute_inflection:
                inflection_log2fc = np.full(n_features, np.nan)
                valid = ~np.isnan(inflection_mean) & (inflection_mean > epsilon)
                if np.any(valid):
                    inflection_log2fc[valid] = np.log2(inflection_mean[valid]) - log2_x_ntc
                data['inflection_log2fc'] = inflection_log2fc

            # For single Hill, baseline is often implicit (A=0)
            # But the function value at x=0 depends on sign of n
            # For n>0: y(0) = 0, for n<0: y(0) = B
            # We can compute this as the offset relative to y_ntc
            # This is more complex - for now, just provide the basic transforms

        return data

    def _add_polynomial_params(self, data, posterior, n_features, compute_full_log2fc):
        """Add polynomial parameters to data dict."""
        coefs = posterior['poly_coefs']  # [n_samples, n_features, degree+1] or [n_features, degree+1]
        if isinstance(coefs, torch.Tensor):
            coefs = coefs.cpu().numpy()

        degree = coefs.shape[-1] - 1

        if coefs.ndim == 3:
            for i in range(degree + 1):
                coef_mean = coefs[:, :, i].mean(axis=0)
                coef_lower = np.quantile(coefs[:, :, i], 0.025, axis=0)
                coef_upper = np.quantile(coefs[:, :, i], 0.975, axis=0)

                data[f'coef_{i}_mean'] = coef_mean
                data[f'coef_{i}_lower'] = coef_lower
                data[f'coef_{i}_upper'] = coef_upper
        else:
            for i in range(degree + 1):
                data[f'coef_{i}_mean'] = coefs[:, i]
                data[f'coef_{i}_lower'] = coefs[:, i]
                data[f'coef_{i}_upper'] = coefs[:, i]

        if compute_full_log2fc:
            # Estimate full log2FC by evaluating polynomial at x_true range
            if hasattr(self.model, 'x_true'):
                x_true = self.model.x_true
                if isinstance(x_true, torch.Tensor):
                    x_true = x_true.cpu().numpy()

                x_min = x_true.min()
                x_max = x_true.max()

                # Evaluate polynomial at endpoints
                if coefs.ndim == 3:
                    # [n_samples, n_features, degree+1]
                    y_min = np.sum([coefs[:, :, i] * (x_min ** i) for i in range(degree + 1)], axis=0)
                    y_max = np.sum([coefs[:, :, i] * (x_max ** i) for i in range(degree + 1)], axis=0)

                    full_log2fc = np.abs(y_max - y_min)
                    full_log2fc_mean = full_log2fc.mean(axis=0)
                    full_log2fc_lower = np.quantile(full_log2fc, 0.025, axis=0)
                    full_log2fc_upper = np.quantile(full_log2fc, 0.975, axis=0)
                else:
                    y_min = np.sum([coefs[:, i] * (x_min ** i) for i in range(degree + 1)], axis=0)
                    y_max = np.sum([coefs[:, i] * (x_max ** i) for i in range(degree + 1)], axis=0)

                    full_log2fc_mean = np.abs(y_max - y_min)
                    full_log2fc_lower = full_log2fc_mean
                    full_log2fc_upper = full_log2fc_mean

                data['full_log2fc_mean'] = full_log2fc_mean
                data['full_log2fc_lower'] = full_log2fc_lower
                data['full_log2fc_upper'] = full_log2fc_upper

        return data

    def _compute_hill_inflection(self, K, xc):
        """
        Compute inflection point of Hill function.

        For Hill equation y = B * x^K / (xc^K + x^K), the inflection point is at:
        x_inflection = xc * ((K - 1) / (K + 1)) ^ (1/K)

        Only defined for K > 1.
        """
        K = np.asarray(K)
        xc = np.asarray(xc)

        # Only compute for K > 1
        inflection = np.full_like(K, np.nan, dtype=float)
        valid = K > 1

        if np.any(valid):
            ratio = (K[valid] - 1) / (K[valid] + 1)
            inflection[valid] = xc[valid] * (ratio ** (1.0 / K[valid]))

        return inflection

    # ========================================================================
    # Derivative and Root Finding Methods (for Additive Hill)
    # ========================================================================

    def _hill_first_derivative(self, x, Vmax, K, n, epsilon=1e-6):
        """
        First derivative of Hill function: d/dx [V * x^n / (K^n + x^n)]

        Formula: (K^n * V * n * x^(n-1)) / (K^n + x^n)^2

        Note: n can be negative (for inhibitor Hill functions).
        """
        x = np.maximum(x, epsilon)
        K = np.maximum(K, epsilon)
        # Do NOT constrain n - it can be negative for inhibitor functions

        # Use safe computation to avoid overflow
        try:
            K_n = K ** n
            x_n = x ** n
            x_nm1 = x ** (n - 1)
            denom = (K_n + x_n) ** 2

            result = (K_n * Vmax * n * x_nm1) / (denom + epsilon)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except (OverflowError, FloatingPointError):
            return 0.0

    def _hill_second_derivative(self, x, Vmax, K, n, epsilon=1e-6):
        """
        Second derivative of Hill function: d²/dx² [V * x^n / (K^n + x^n)]

        Formula: -(K^n * V * n * x^(n-2) * ((n+1)*x^n - K^n*(n-1))) / (K^n + x^n)^3

        Note: n can be negative (for inhibitor Hill functions).
        """
        x = np.maximum(x, epsilon)
        K = np.maximum(K, epsilon)
        # Do NOT constrain n - it can be negative for inhibitor functions

        # Use safe computation to avoid overflow
        try:
            K_n = K ** n
            x_n = x ** n
            x_nm2 = x ** (n - 2)
            denom = (K_n + x_n) ** 3

            inner = (n + 1) * x_n - K_n * (n - 1)

            result = -(K_n * Vmax * n * x_nm2 * inner) / (denom + epsilon)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except (OverflowError, FloatingPointError):
            return 0.0

    def _additive_hill_first_derivative(self, x, alpha, Vmax_a, K_a, n_a,
                                         beta, Vmax_b, K_b, n_b, epsilon=1e-6):
        """
        First derivative of additive Hill: dy/dx = alpha * dHill_a/dx + beta * dHill_b/dx
        """
        d_a = self._hill_first_derivative(x, Vmax_a, K_a, n_a, epsilon)
        d_b = self._hill_first_derivative(x, Vmax_b, K_b, n_b, epsilon)
        return alpha * d_a + beta * d_b

    def _additive_hill_second_derivative(self, x, alpha, Vmax_a, K_a, n_a,
                                          beta, Vmax_b, K_b, n_b, epsilon=1e-6):
        """
        Second derivative of additive Hill: d²y/dx² = alpha * d²Hill_a/dx² + beta * d²Hill_b/dx²
        """
        d2_a = self._hill_second_derivative(x, Vmax_a, K_a, n_a, epsilon)
        d2_b = self._hill_second_derivative(x, Vmax_b, K_b, n_b, epsilon)
        return alpha * d2_a + beta * d2_b

    def _hill_third_derivative(self, x, Vmax, K, n, epsilon=1e-6):
        """
        Third derivative of Hill function: d³/dx³ [V * x^n / (K^n + x^n)]

        Formula: (K^n * V * n * x^(n-3) * ((n²+3n+2)*x^(2n) + (4K^n - 4K^n*n²)*x^n
                 + K^(2n)*n² - 3K^(2n)*n + 2K^(2n))) / (K^n + x^n)^4

        Note: n can be negative (for inhibitor Hill functions).
        """
        x = np.maximum(x, epsilon)
        K = np.maximum(K, epsilon)
        # Do NOT constrain n - it can be negative for inhibitor functions

        # Use safe computation to avoid overflow
        try:
            x_n = x ** n
            x_2n = x ** (2 * n)
            K_n = K ** n
            K_2n = K ** (2 * n)
            x_nm3 = x ** (n - 3)
            denom = (K_n + x_n) ** 4

            # Numerator terms
            # (n² + 3n + 2) * x^(2n)
            term1 = (n**2 + 3*n + 2) * x_2n
            # (4K^n - 4K^n*n²) * x^n = 4K^n * (1 - n²) * x^n
            term2 = 4 * K_n * (1 - n**2) * x_n
            # K^(2n) * n² - 3K^(2n) * n + 2K^(2n) = K^(2n) * (n² - 3n + 2)
            term3 = K_2n * (n**2 - 3*n + 2)

            numer = K_n * Vmax * n * x_nm3 * (term1 + term2 + term3)

            result = numer / (denom + epsilon)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except (OverflowError, FloatingPointError):
            return 0.0

    def _additive_hill_third_derivative(self, x, alpha, Vmax_a, K_a, n_a,
                                         beta, Vmax_b, K_b, n_b, epsilon=1e-6):
        """
        Third derivative of additive Hill: d³y/dx³ = alpha * d³Hill_a/dx³ + beta * d³Hill_b/dx³
        """
        d3_a = self._hill_third_derivative(x, Vmax_a, K_a, n_a, epsilon)
        d3_b = self._hill_third_derivative(x, Vmax_b, K_b, n_b, epsilon)
        return alpha * d3_a + beta * d3_b

    def _find_roots_empirical(self, func, x_range: np.ndarray) -> List[float]:
        """
        Find roots of a function empirically by detecting sign changes.

        Parameters
        ----------
        func : callable
            Function to find roots of (takes x, returns y)
        x_range : np.ndarray
            X values to search over

        Returns
        -------
        List[float]
            List of approximate root locations (interpolated between sign changes)
        """
        y_vals = func(x_range)
        roots = []

        for i in range(len(y_vals) - 1):
            if np.isnan(y_vals[i]) or np.isnan(y_vals[i+1]):
                continue
            # Check for sign change
            if y_vals[i] * y_vals[i+1] < 0:
                # Interpolate to find approximate root
                try:
                    root = brentq(func, x_range[i], x_range[i+1])
                    roots.append(root)
                except (ValueError, RuntimeError):
                    # If brentq fails, use linear interpolation
                    t = abs(y_vals[i]) / (abs(y_vals[i]) + abs(y_vals[i+1]))
                    root = x_range[i] + t * (x_range[i+1] - x_range[i])
                    roots.append(root)

        return roots

    def _classify_additive_hill(self, alpha, alpha_lower, alpha_upper,
                                 beta, beta_lower, beta_upper,
                                 n_a, n_a_lower, n_a_upper,
                                 n_b, n_b_lower, n_b_upper,
                                 first_deriv_roots: List[float],
                                 second_deriv_roots: List[float],
                                 x_range: np.ndarray,
                                 Vmax_a, K_a, Vmax_b, K_b,
                                 x_ntc: float = None) -> str:
        """
        Classify additive Hill function into categories based on dependency masks.

        Classification Logic:
        - dep_mask_a: Component A is active (0 NOT in CI of alpha AND 0 NOT in CI of n_a)
        - dep_mask_b: Component B is active (0 NOT in CI of beta AND 0 NOT in CI of n_b)
        - dep_mask: Either component is active

        Categories:
        - 'single_positive': Only one component active, with positive n (increasing)
            Formula: (~(dep_mask_a & dep_mask_b) & dep_mask) & (n >= 0 for active component)
        - 'single_negative': Only one component active, with negative n (decreasing)
            Formula: (~(dep_mask_a & dep_mask_b) & dep_mask) & (n <= 0 for active component)
        - 'additive_positive': Both components active, both with positive n
            Formula: (dep_mask_a & dep_mask_b) & (n_a >= 0) & (n_b >= 0)
        - 'additive_negative': Both components active, both with negative n
            Formula: (dep_mask_a & dep_mask_b) & (n_a <= 0) & (n_b <= 0)
        - 'non_monotonic_min': Both active, opposite signs, at minimum at x=x_ntc (S''(x_ntc) > 0)
        - 'non_monotonic_max': Both active, opposite signs, at maximum at x=x_ntc (S''(x_ntc) < 0)
        """
        # Define dependency masks
        # Component is "active" if its weight is non-zero AND its Hill coefficient is non-zero
        # Use CI to determine if 0 is in the credible interval
        alpha_active = not (alpha_lower <= 0 <= alpha_upper)  # 0 NOT in CI
        n_a_active = not (n_a_lower <= 0 <= n_a_upper)  # n_a is not ~0

        beta_active = not (beta_lower <= 0 <= beta_upper)  # 0 NOT in CI
        n_b_active = not (n_b_lower <= 0 <= n_b_upper)  # n_b is not ~0

        # Dependency masks
        dep_mask_a = alpha_active and n_a_active
        dep_mask_b = beta_active and n_b_active
        dep_mask = dep_mask_a or dep_mask_b  # At least one component active

        # Check if both components are active
        both_active = dep_mask_a and dep_mask_b
        single_active = dep_mask and not both_active

        # Determine signs of Hill coefficients for active components
        # Use mean values for classification
        if single_active:
            # Only one component is active
            if dep_mask_a:
                # Only component A is active
                if n_a >= 0:
                    return 'single_positive'
                else:
                    return 'single_negative'
            else:
                # Only component B is active
                if n_b >= 0:
                    return 'single_positive'
                else:
                    return 'single_negative'

        if both_active:
            # Both components are active - check if same or opposite signs
            same_sign = (n_a * n_b) >= 0  # Both positive or both negative

            if same_sign:
                # Additive: both increasing or both decreasing
                if n_a >= 0 and n_b >= 0:
                    return 'additive_positive'
                else:
                    return 'additive_negative'
            else:
                # Non-monotonic: opposite signs of n
                # Determine min vs max based on second derivative at x_ntc (curvature test)
                # S''(x_ntc) > 0 → concave up → local minimum
                # S''(x_ntc) < 0 → concave down → local maximum

                if x_ntc is not None:
                    # Compute second derivative at x_ntc
                    S_double_prime = self._additive_hill_second_derivative(
                        x_ntc, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b
                    )
                    if S_double_prime > 0:
                        return 'non_monotonic_min'  # Concave up at x_ntc → local minimum
                    else:
                        return 'non_monotonic_max'  # Concave down at x_ntc → local maximum
                else:
                    # Fallback: use boundary analysis
                    if len(first_deriv_roots) > 0:
                        # Has interior extremum - check curvature at the extremum
                        x_extremum = first_deriv_roots[0]
                        d2_extremum = self._additive_hill_second_derivative(
                            x_extremum, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b
                        )
                        if d2_extremum > 0:
                            return 'non_monotonic_min'  # Concave up at extremum → local min
                        else:
                            return 'non_monotonic_max'  # Concave down at extremum → local max
                    return 'non_monotonic'  # Generic non-monotonic fallback

        # No active components or indeterminate
        return 'flat'

    def _compute_full_log2fc_additive(self, A, alpha, Vmax_a, beta, Vmax_b,
                                       x_range: np.ndarray,
                                       K_a, n_a, K_b, n_b,
                                       classification: str,
                                       first_deriv_roots: List[float] = None) -> float:
        """
        Compute full log2FC for additive Hill based on classification.

        Returns log2(y_max / y_min) - the dynamic range in log2 space.

        Hill function limits depend on sign of n:
        - n > 0: Hill(0) = 0, Hill(∞) = Vmax
        - n < 0: Hill(0) = Vmax, Hill(∞) = 0
        """
        epsilon = 1e-10

        # Compute boundary values based on signs of n
        # At x→0: Hill = Vmax if n<0, else 0
        # At x→∞: Hill = 0 if n<0, else Vmax
        hill_a_at_0 = Vmax_a if n_a < 0 else 0
        hill_a_at_inf = 0 if n_a < 0 else Vmax_a
        hill_b_at_0 = Vmax_b if n_b < 0 else 0
        hill_b_at_inf = 0 if n_b < 0 else Vmax_b

        y_at_0 = A + alpha * hill_a_at_0 + beta * hill_b_at_0
        y_at_inf = A + alpha * hill_a_at_inf + beta * hill_b_at_inf

        # Ensure positive values for log
        y_at_0 = max(y_at_0, epsilon)
        y_at_inf = max(y_at_inf, epsilon)

        if classification in ['monotonic_positive', 'monotonic_negative',
                               'single_positive', 'single_negative',
                               'additive_positive', 'additive_negative',
                               'single_hill_a', 'single_hill_b']:
            # For monotonic, log2FC is log2(y_max / y_min)
            y_max = max(y_at_0, y_at_inf)
            y_min = min(y_at_0, y_at_inf)
            return np.log2(max(y_max, epsilon) / max(y_min, epsilon))

        # For non-monotonic, find interior extrema
        if first_deriv_roots is None:
            def first_deriv(x):
                return self._additive_hill_first_derivative(
                    x, alpha, Vmax_a, K_a, n_a, beta, Vmax_b, K_b, n_b
                )
            first_deriv_roots = self._find_roots_empirical(first_deriv, x_range)

        if len(first_deriv_roots) == 0:
            # No interior extremum, use boundary difference
            y_max = max(y_at_0, y_at_inf)
            y_min = min(y_at_0, y_at_inf)
            return np.log2(max(y_max, epsilon) / max(y_min, epsilon))

        # Evaluate y at the extrema
        def hill_func(x):
            x_safe = max(x, epsilon)
            K_a_safe = max(K_a, epsilon)
            K_b_safe = max(K_b, epsilon)

            # Handle potential numerical issues with negative n
            if n_a >= 0:
                H_a = Vmax_a * (x_safe ** n_a) / (K_a_safe ** n_a + x_safe ** n_a + epsilon)
            else:
                # For n < 0: x^n = 1/x^|n|, need careful handling
                x_n_a = x_safe ** n_a
                K_n_a = K_a_safe ** n_a
                H_a = Vmax_a * x_n_a / (K_n_a + x_n_a + epsilon)

            if n_b >= 0:
                H_b = Vmax_b * (x_safe ** n_b) / (K_b_safe ** n_b + x_safe ** n_b + epsilon)
            else:
                x_n_b = x_safe ** n_b
                K_n_b = K_b_safe ** n_b
                H_b = Vmax_b * x_n_b / (K_n_b + x_n_b + epsilon)

            return A + alpha * H_a + beta * H_b

        y_extrema = [hill_func(r) for r in first_deriv_roots]

        # Find the largest range (total dynamic range) in log2 space
        all_y = [y_at_0, y_at_inf] + y_extrema
        y_max = max(all_y)
        y_min = min(all_y)
        return np.log2(max(y_max, epsilon) / max(y_min, epsilon))

    def _compute_observed_log2fc_fitted(self, A, alpha, Vmax_a, beta, Vmax_b,
                                         K_a, n_a, K_b, n_b,
                                         x_obs_min: float, x_obs_max: float) -> float:
        """
        Compute observed log2FC over the observed x range using the fitted Hill function.

        Returns log2(y_max / y_min) where y_max and y_min are evaluated over
        the observed x range [x_obs_min, x_obs_max].

        Parameters
        ----------
        x_obs_min : float
            Minimum observed x value (min of guide-level x_true)
        x_obs_max : float
            Maximum observed x value (max of guide-level x_true)
        """
        epsilon = 1e-10

        # Evaluate at dense grid points (vectorized for speed)
        n_points = 200
        x_eval = np.linspace(max(x_obs_min, epsilon), x_obs_max, n_points)

        # Vectorized Hill function evaluation
        K_a_safe = max(K_a, epsilon)
        K_b_safe = max(K_b, epsilon)

        # Compute Hill A
        x_n_a = x_eval ** n_a
        K_n_a = K_a_safe ** n_a
        H_a = Vmax_a * x_n_a / (K_n_a + x_n_a + epsilon)

        # Compute Hill B
        x_n_b = x_eval ** n_b
        K_n_b = K_b_safe ** n_b
        H_b = Vmax_b * x_n_b / (K_n_b + x_n_b + epsilon)

        # Combined function
        y_vals = A + alpha * H_a + beta * H_b

        # Handle any NaN/inf values
        y_vals = np.nan_to_num(y_vals, nan=A, posinf=A, neginf=A)

        # Compute log2FC as log2(y_max / y_min)
        y_max = max(y_vals.max(), epsilon)
        y_min = max(y_vals.min(), epsilon)
        return np.log2(y_max / y_min)

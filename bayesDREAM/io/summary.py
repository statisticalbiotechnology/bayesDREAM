"""
Summary export for bayesDREAM results.

Exports model results as R-friendly CSV files with:
- Mean and 95% credible intervals for all parameters
- Cell-wise and feature-wise summaries
- Observed log2FC, predicted log2FC, inflection points for trans fits
"""

import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Any


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

        # Check if technical fit has been run
        if not hasattr(self.model, 'alpha_y_prefit') or self.model.alpha_y_prefit is None:
            raise ValueError("Technical fit not found. Run fit_technical() first.")

        # Get modality
        modality = self.model.get_modality(modality_name)

        # Get alpha_y for this modality
        if modality_name == self.model.primary_modality:
            alpha_y = self.model.alpha_y_prefit  # [n_samples, n_groups, n_features]
        elif hasattr(modality, 'alpha_y_prefit_mult'):
            alpha_y = modality.alpha_y_prefit_mult  # [n_samples, n_groups, n_features]
        else:
            raise ValueError(f"No technical fit parameters found for modality '{modality_name}'")

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
        compute_full_log2fc: bool = True
    ):
        """
        Save trans fit parameters as feature-wise CSV.

        Creates: trans_feature_summary_{modality}.csv

        Columns (depend on function_type and distribution):
        - feature: Feature name
        - modality: Modality name
        - distribution: Distribution type
        - function_type: Function type (additive_hill, single_hill, polynomial)
        - observed_log2fc: Observed log2FC (perturbed vs NTC)
        - observed_log2fc_se: Standard error of observed log2FC

        For additive_hill:
        - B_pos_mean, B_pos_lower, B_pos_upper: Positive Hill magnitude
        - K_pos_mean, K_pos_lower, K_pos_upper: Positive Hill coefficient
        - EC50_pos_mean, EC50_pos_lower, EC50_pos_upper: Positive Hill EC50
        - B_neg_mean, B_neg_lower, B_neg_upper: Negative Hill magnitude
        - K_neg_mean, K_neg_lower, K_neg_upper: Negative Hill coefficient
        - IC50_neg_mean, IC50_neg_lower, IC50_neg_upper: Negative Hill IC50
        - pi_y_mean, pi_y_lower, pi_y_upper: Sparsity weight
        - inflection_pos_mean, inflection_pos_lower, inflection_pos_upper: Positive inflection x
        - inflection_neg_mean, inflection_neg_lower, inflection_neg_upper: Negative inflection x
        - full_log2fc_mean, full_log2fc_lower, full_log2fc_upper: Full dynamic range

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
        feature_names = modality.feature_meta.index.tolist()
        n_features = len(feature_names)

        # Determine function type from posterior keys
        if 'params_pos' in posterior and 'params_neg' in posterior:
            function_type = 'additive_hill'
        elif 'params' in posterior:
            function_type = 'single_hill'
        elif 'poly_coefs' in posterior:
            function_type = 'polynomial'
        else:
            raise ValueError("Cannot determine function_type from posterior_samples_trans keys")

        # Initialize DataFrame with basic info
        data = {
            'feature': feature_names,
            'modality': modality_name,
            'distribution': modality.distribution,
            'function_type': function_type
        }

        # Compute observed log2FC
        obs_log2fc, obs_log2fc_se = self._compute_observed_log2fc(modality)
        data['observed_log2fc'] = obs_log2fc
        data['observed_log2fc_se'] = obs_log2fc_se

        # Add function-specific parameters
        if function_type == 'additive_hill':
            data = self._add_additive_hill_params(
                data, posterior, n_features,
                compute_inflection, compute_full_log2fc
            )
        elif function_type == 'single_hill':
            data = self._add_single_hill_params(
                data, posterior, n_features,
                compute_inflection, compute_full_log2fc
            )
        elif function_type == 'polynomial':
            data = self._add_polynomial_params(
                data, posterior, n_features,
                compute_full_log2fc
            )

        df = pd.DataFrame(data)

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

    def _compute_observed_log2fc(self, modality):
        """Compute observed log2FC (perturbed vs NTC) for a modality."""
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

    def _add_additive_hill_params(self, data, posterior, n_features, compute_inflection, compute_full_log2fc):
        """Add additive Hill parameters to data dict."""
        # Positive Hill
        params_pos = posterior['params_pos']  # [n_samples, n_features, 3] or [n_features, 3]
        if isinstance(params_pos, torch.Tensor):
            params_pos = params_pos.cpu().numpy()

        if params_pos.ndim == 3:
            # Posterior samples
            B_pos_mean = params_pos[:, :, 0].mean(axis=0)
            B_pos_lower = np.quantile(params_pos[:, :, 0], 0.025, axis=0)
            B_pos_upper = np.quantile(params_pos[:, :, 0], 0.975, axis=0)

            K_pos_mean = params_pos[:, :, 1].mean(axis=0)
            K_pos_lower = np.quantile(params_pos[:, :, 1], 0.025, axis=0)
            K_pos_upper = np.quantile(params_pos[:, :, 1], 0.975, axis=0)

            EC50_pos_mean = params_pos[:, :, 2].mean(axis=0)
            EC50_pos_lower = np.quantile(params_pos[:, :, 2], 0.025, axis=0)
            EC50_pos_upper = np.quantile(params_pos[:, :, 2], 0.975, axis=0)
        else:
            # Point estimate
            B_pos_mean = params_pos[:, 0]
            B_pos_lower = B_pos_mean
            B_pos_upper = B_pos_mean

            K_pos_mean = params_pos[:, 1]
            K_pos_lower = K_pos_mean
            K_pos_upper = K_pos_mean

            EC50_pos_mean = params_pos[:, 2]
            EC50_pos_lower = EC50_pos_mean
            EC50_pos_upper = EC50_pos_mean

        data['B_pos_mean'] = B_pos_mean
        data['B_pos_lower'] = B_pos_lower
        data['B_pos_upper'] = B_pos_upper
        data['K_pos_mean'] = K_pos_mean
        data['K_pos_lower'] = K_pos_lower
        data['K_pos_upper'] = K_pos_upper
        data['EC50_pos_mean'] = EC50_pos_mean
        data['EC50_pos_lower'] = EC50_pos_lower
        data['EC50_pos_upper'] = EC50_pos_upper

        # Negative Hill
        params_neg = posterior['params_neg']
        if isinstance(params_neg, torch.Tensor):
            params_neg = params_neg.cpu().numpy()

        if params_neg.ndim == 3:
            B_neg_mean = params_neg[:, :, 0].mean(axis=0)
            B_neg_lower = np.quantile(params_neg[:, :, 0], 0.025, axis=0)
            B_neg_upper = np.quantile(params_neg[:, :, 0], 0.975, axis=0)

            K_neg_mean = params_neg[:, :, 1].mean(axis=0)
            K_neg_lower = np.quantile(params_neg[:, :, 1], 0.025, axis=0)
            K_neg_upper = np.quantile(params_neg[:, :, 1], 0.975, axis=0)

            IC50_neg_mean = params_neg[:, :, 2].mean(axis=0)
            IC50_neg_lower = np.quantile(params_neg[:, :, 2], 0.025, axis=0)
            IC50_neg_upper = np.quantile(params_neg[:, :, 2], 0.975, axis=0)
        else:
            B_neg_mean = params_neg[:, 0]
            B_neg_lower = B_neg_mean
            B_neg_upper = B_neg_mean

            K_neg_mean = params_neg[:, 1]
            K_neg_lower = K_neg_mean
            K_neg_upper = K_neg_mean

            IC50_neg_mean = params_neg[:, 2]
            IC50_neg_lower = IC50_neg_mean
            IC50_neg_upper = IC50_neg_mean

        data['B_neg_mean'] = B_neg_mean
        data['B_neg_lower'] = B_neg_lower
        data['B_neg_upper'] = B_neg_upper
        data['K_neg_mean'] = K_neg_mean
        data['K_neg_lower'] = K_neg_lower
        data['K_neg_upper'] = K_neg_upper
        data['IC50_neg_mean'] = IC50_neg_mean
        data['IC50_neg_lower'] = IC50_neg_lower
        data['IC50_neg_upper'] = IC50_neg_upper

        # Pi_y (sparsity weight)
        if 'pi_y' in posterior:
            pi_y = posterior['pi_y']
            if isinstance(pi_y, torch.Tensor):
                pi_y = pi_y.cpu().numpy()

            if pi_y.ndim == 2:
                pi_y_mean = pi_y.mean(axis=0)
                pi_y_lower = np.quantile(pi_y, 0.025, axis=0)
                pi_y_upper = np.quantile(pi_y, 0.975, axis=0)
            else:
                pi_y_mean = pi_y
                pi_y_lower = pi_y
                pi_y_upper = pi_y

            data['pi_y_mean'] = pi_y_mean
            data['pi_y_lower'] = pi_y_lower
            data['pi_y_upper'] = pi_y_upper

        # Compute inflection points
        if compute_inflection:
            inflection_pos_mean = self._compute_hill_inflection(K_pos_mean, EC50_pos_mean)
            inflection_pos_lower = self._compute_hill_inflection(K_pos_lower, EC50_pos_lower)
            inflection_pos_upper = self._compute_hill_inflection(K_pos_upper, EC50_pos_upper)

            inflection_neg_mean = self._compute_hill_inflection(K_neg_mean, IC50_neg_mean)
            inflection_neg_lower = self._compute_hill_inflection(K_neg_lower, IC50_neg_lower)
            inflection_neg_upper = self._compute_hill_inflection(K_neg_upper, IC50_neg_upper)

            data['inflection_pos_mean'] = inflection_pos_mean
            data['inflection_pos_lower'] = inflection_pos_lower
            data['inflection_pos_upper'] = inflection_pos_upper
            data['inflection_neg_mean'] = inflection_neg_mean
            data['inflection_neg_lower'] = inflection_neg_lower
            data['inflection_neg_upper'] = inflection_neg_upper

        # Compute full log2FC
        if compute_full_log2fc:
            full_log2fc_mean = B_pos_mean + B_neg_mean  # Total effect magnitude
            full_log2fc_lower = B_pos_lower + B_neg_lower
            full_log2fc_upper = B_pos_upper + B_neg_upper

            data['full_log2fc_mean'] = full_log2fc_mean
            data['full_log2fc_lower'] = full_log2fc_lower
            data['full_log2fc_upper'] = full_log2fc_upper

        return data

    def _add_single_hill_params(self, data, posterior, n_features, compute_inflection, compute_full_log2fc):
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

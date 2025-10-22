"""
Technical variation fitting for bayesDREAM.

This module contains the technical model and fitting logic.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer



class TechnicalFitter:
    """Handles technical variation fitting on NTC cells."""

    def __init__(self, model):
        """
        Initialize technical fitter.

        Parameters
        ----------
        model : _BayesDREAMCore
            The parent model instance
        """
        self.model = model

    def _model_technical(
        self,
        N,
        T,
        C,
        groups_ntc_tensor,
        y_obs_ntc_tensor,
        sum_factor_ntc_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        epsilon_tensor,
        distribution='negbinom',
        denominator_ntc_tensor=None,
        K=None,
        D=None,
    ):
    
        # Global parameters (no plate needed here)
    
        # Define the c_plate ONCE and store it
        c_plate = pyro.plate("c_plate", C - 1, dim=-2)  # <-- Store it as a variableit
        cfull_plate = pyro.plate("cfull_plate", C, dim=-2)  # <-- Store it as a variable
    
        # Sample alpha_alpha and alpha_mu using the stored c_plate
        #with cfull_plate:
        #    beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
    
        # Sample alpha_alpha and alpha_mu using the stored c_plate
        #with c_plate:
        #    #alpha_log2mu = pyro.sample("alpha_log2mu", dist.Normal(0, 1)) 
        #    alpha_log2mu = pyro.sample("alpha_log2mu", dist.StudentT(df=1, loc=0., scale=5.)) 
        #    #alpha_sigma  = pyro.sample("alpha_sigma", dist.HalfNormal(4.0)) 
        #    alpha_sigma  = pyro.sample("alpha_sigma", dist.HalfCauchy(10.)) 
    
        # One single trans_plate for all T-dependent variables
        with pyro.plate("trans_plate", T, dim=-1):
    
            # Reuse the SAME c_plate inside trans_plate
            with c_plate:  # <-- This ensures the same plate is used
                #alpha_sigma  = pyro.sample("alpha_sigma", dist.HalfNormal(4.0)) 
                #log2_alpha_y = pyro.sample("log2_alpha_y", dist.Normal(
                log2_alpha_y = pyro.sample("log2_alpha_y", dist.StudentT(
                    df=3,
                    #loc=alpha_log2mu,
                    loc=0.0,
                    #scale=alpha_sigma
                    scale=20.0
                ))
                alpha_y = pyro.deterministic("alpha_y", 2 ** log2_alpha_y)
                #print("alpha_y shape:", alpha_y.shape)
            
            o_y = pyro.sample("o_y", dist.Exponential(beta_o))
            phi_y = 1 / (o_y**2)

        # Add baseline cell line of all ones
        alpha_y = alpha_y.to(self.model.device)
        if alpha_y.ndim == 2:
            alpha_y_full = torch.cat([torch.ones(1, T, device=self.model.device), alpha_y], dim=0)  # shape => (C, T)
        elif alpha_y.ndim == 3:
            alpha_y_full = torch.cat([torch.ones(alpha_y.size(0), 1, T, device=self.model.device), alpha_y], dim=1)  # shape => (num_samples, C, T)

        alpha_y_used = alpha_y_full[..., groups_ntc_tensor, :]  # [N, T]
        #phi_y_used = phi_y[..., groups_ntc_tensor, :] # shape => [N, T]
        phi_y_used = phi_y.unsqueeze(-2)
    
        # Sample mu_ntc (baseline expression for each feature)
        # Use Normal prior for distributions with potentially negative values
        with pyro.plate("feature_plate_technical", T, dim=-1):
            if distribution in ['normal', 'mvnormal']:
                mu_ntc = pyro.sample(
                    "mu_ntc",
                    dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor)
                )  # [T]
            else:
                # Use Gamma prior for count-based distributions
                mu_ntc = pyro.sample(
                    "mu_ntc",
                    dist.Gamma((mu_x_mean_tensor**2) / (mu_x_sd_tensor**2), mu_x_mean_tensor / (mu_x_sd_tensor**2))
                )  # [T]

        # Compute mu_y from mu_ntc and alpha_y
        mu_y = alpha_y_used * mu_ntc  # [N, T]

        # Call distribution-specific observation sampler
        from ..distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')

        if distribution == 'negbinom':
            # For negbinom, we pass mu_y and sum_factor separately
            # Note: alpha_y already applied in mu_y, so we pass alpha_y_full=None
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y,
                phi_y_used=phi_y_used,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                sum_factor_tensor=sum_factor_ntc_tensor,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'normal':
            sigma_y = 1.0 / torch.sqrt(phi_y)
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                denominator_tensor=denominator_ntc_tensor,
                mu_y=mu_y,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'multinomial':
            # For multinomial, need to sample category probabilities for each feature
            # y_obs_ntc_tensor is [N, T, K]
            # Sample baseline category probabilities for each feature
            # Use Dirichlet distribution to sample probability vectors
            with pyro.plate("feature_plate_technical", T, dim=-1):
                # Compute empirical category proportions as concentration parameters
                # Sum over observations to get total counts per feature per category
                total_counts_per_feature = y_obs_ntc_tensor.sum(dim=0)  # [T, K]
                # Add pseudocount to avoid zeros
                concentration = total_counts_per_feature + 1.0  # [T, K]

                # Sample category probabilities
                probs_baseline = pyro.sample(
                    "probs_baseline",
                    dist.Dirichlet(concentration)
                )  # [T, K]

            # Expand to [N, T, K] by broadcasting
            mu_y_multi = probs_baseline.unsqueeze(0).expand(N, T, K)  # [N, T, K]

            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y_multi,
                N=N,
                T=T,
                K=K
            )
        elif distribution == 'mvnormal':
            # For mvnormal, need covariance matrix
            # For now, use diagonal covariance from phi_y
            # phi_y is [T], we need [T, D, D] covariance matrices
            # Create diagonal covariance: each feature has same variance for all dimensions
            variance_per_feature = 1.0 / phi_y  # [T]
            # Expand to [T, D] then create diagonal matrices [T, D, D]
            cov_y = torch.diag_embed(variance_per_feature.unsqueeze(-1).expand(T, D))  # [T, D, D]

            # Also need to expand mu_y from [N, T] to [N, T, D]
            # For technical fit, assume same mean across all dimensions
            mu_y_mv = mu_y.unsqueeze(-1).expand(N, T, D)  # [N, T, D]

            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y_mv,
                cov_y=cov_y,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                N=N,
                T=T,
                D=D,
                C=C
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")


    def set_technical_groups(self, covariates: list[str]):
        """
        Set technical_group_code based on covariates.

        This should be called before fit_technical() to define technical groups.
        The technical_group_code will be used for:
        - fit_technical(): modeling technical variation across groups
        - fit_cis(): optional correction for group effects
        - fit_trans(): optional correction for group effects

        Parameters
        ----------
        covariates : list[str]
            Column names in self.model.meta to group by (e.g., ['cell_line'])

        Examples
        --------
        >>> model.set_technical_groups(['cell_line'])
        >>> model.fit_technical(sum_factor_col='sum_factor')
        """
        if not covariates:
            raise ValueError("covariates must not be empty")

        missing_cols = [c for c in covariates if c not in self.model.meta.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in meta: {missing_cols}")

        self.model.meta["technical_group_code"] = self.model.meta.groupby(covariates).ngroup()
        print(f"[INFO] Set technical_group_code with {self.model.meta['technical_group_code'].nunique()} groups based on {covariates}")

    ########################################################
    # Step 1: Optional Prefit for alpha_y (NTC only)
    ########################################################
    def fit_technical(
        self,
        sum_factor_col: str = 'sum_factor',
        lr: float = 1e-3,
        niters: int = 50000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        epsilon: float = 1e-6,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        **kwargs
    ):
        """
        Prefits gene-level "technical" variables (alpha_y) using only NTC samples,
        grouped by technical groups.

        If you skip this method, you'd provide alpha_y_fixed manually in the next step.

        Parameters
        ----------
        modality_name : str, optional
            Name of modality to fit. If None, uses primary modality.
        distribution : str, optional
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
            If None, auto-detected from modality.
        sum_factor_col : str, optional
            Column name for size factors (required for negbinom, ignored otherwise)
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (shape: [n_features, n_cells])
            If None, auto-detected from modality.

        Notes
        -----
        Each modality stores its own fitting results.
        Primary modality results are also stored at model level for backward compatibility.

        Examples
        --------
        >>> model.set_technical_groups(['cell_line'])
        >>> model.fit_technical(sum_factor_col='sum_factor')
        """

        # Determine which modality to use
        if modality_name is None:
            modality_name = self.model.primary_modality
        modality = self.model.get_modality(modality_name)

        # Auto-detect distribution from modality
        if distribution is None:
            distribution = modality.distribution

        # Auto-detect denominator from modality (for binomial)
        if denominator is None and modality.denominator is not None:
            denominator = modality.denominator

        # Get counts from modality
        counts_to_fit = modality.counts

        # Get cell names from modality
        if modality.cell_names is not None:
            modality_cells = modality.cell_names
        else:
            # Modality doesn't have cell names - assume same order as model.meta['cell']
            # This is safe because add_modality() subsets to common cells in same order
            modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]

        print(f"[INFO] Fitting technical model for modality '{modality_name}' (distribution: {distribution})")

        # Validate distribution-specific requirements
        from ..distributions import requires_sum_factor, requires_denominator

        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")
        
        # Check technical_group_code exists
        if "technical_group_code" not in self.model.meta.columns:
            raise ValueError(
                "technical_group_code not set. Call set_technical_groups(covariates) before fit_technical().\n"
                "Example: model.set_technical_groups(['cell_line'])"
            )

        print("Running prefit_cellline...")

        # Subset model.meta to cells present in this modality
        # Create a mapping of cell -> index in modality cells
        modality_cell_set = set(modality_cells)

        # Subset meta to cells in this modality (work with copy)
        meta_subset = self.model.meta[self.model.meta['cell'].isin(modality_cell_set)].copy()

        # Further subset to NTC cells
        meta_ntc = meta_subset[meta_subset["target"] == "ntc"].copy()

        # Get indices of NTC cells in the modality's cell list
        ntc_cell_list = meta_ntc["cell"].tolist()
        ntc_indices = [i for i, c in enumerate(modality_cells) if c in ntc_cell_list]

        # Subset counts to NTC cells
        # counts_to_fit is already the modality's counts (numpy array)
        # Handle both 2D and 3D arrays
        if counts_to_fit.ndim == 2:
            if modality.cells_axis == 1:
                counts_ntc_array = counts_to_fit[:, ntc_indices]
            else:
                counts_ntc_array = counts_to_fit[ntc_indices, :]
        elif counts_to_fit.ndim == 3:
            # 3D data: (features, cells, categories) or (features, cells, dimensions)
            # For multinomial/mvnormal, cells are always on axis 1
            counts_ntc_array = counts_to_fit[:, ntc_indices, :]
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_to_fit.ndim}")

        print(f"[INFO] Modality '{modality_name}': {len(modality_cells)} total cells, {len(ntc_indices)} NTC cells")

        # Filter features based on NTC data quality

        # Step 1: Identify features with zero counts in NTC
        # Handle both 2D and 3D data
        if counts_ntc_array.ndim == 2:
            # For 2D: sum across cells
            if modality.cells_axis == 0:
                feature_sums_ntc = counts_ntc_array.sum(axis=0)  # cells are rows
            else:
                feature_sums_ntc = counts_ntc_array.sum(axis=1)  # features are rows
        elif counts_ntc_array.ndim == 3:
            # For 3D: sum across cells and categories/dimensions
            # Shape is (features, cells, categories/dimensions)
            feature_sums_ntc = counts_ntc_array.sum(axis=(1, 2))
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_ntc_array.ndim}")

        zero_count_mask = feature_sums_ntc == 0
        num_zero_count = zero_count_mask.sum()

        # Step 2: Identify features with zero standard deviation in NTC
        # Different approaches for different distributions
        zero_std_mask = np.zeros(len(feature_sums_ntc), dtype=bool)

        if distribution == 'multinomial':
            # For multinomial: check if ALL ratios (k/total) have std=0
            # counts_ntc_array shape: (features, cells, categories)
            for f_idx in range(counts_ntc_array.shape[0]):
                feature_counts = counts_ntc_array[f_idx, :, :]  # (cells, categories)
                # Compute total counts per cell
                totals = feature_counts.sum(axis=1, keepdims=True)  # (cells, 1)

                # Compute ratios, avoiding division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(totals > 0, feature_counts / totals, 0)  # (cells, categories)

                # Check if ALL ratios have zero std across cells
                ratio_stds = ratios.std(axis=0)  # std across cells for each category
                if np.all(ratio_stds == 0):
                    zero_std_mask[f_idx] = True

        elif distribution == 'binomial':
            # For binomial: check if (numerator/denominator) ratio has std=0
            # (excluding where denominator is 0)
            # counts_ntc_array: (features, cells) - numerator
            # denominator_ntc_array: (features, cells) - denominator
            if denominator is None:
                raise ValueError("Binomial distribution requires denominator for variance check")

            # Get denominator for NTC cells
            if denominator.ndim == 2:
                if modality.cells_axis == 1:
                    denom_ntc = denominator[:, ntc_indices]
                else:
                    denom_ntc = denominator[ntc_indices, :]
            else:
                raise ValueError(f"Unexpected denominator dimensions: {denominator.ndim}")

            for f_idx in range(counts_ntc_array.shape[0]):
                if modality.cells_axis == 0:
                    numer = counts_ntc_array[:, f_idx]  # cells are rows
                    denom = denom_ntc[:, f_idx]
                else:
                    numer = counts_ntc_array[f_idx, :]  # features are rows
                    denom = denom_ntc[f_idx, :]

                # Compute ratios, excluding cells where denominator is 0
                valid_mask = denom > 0
                if valid_mask.sum() == 0:
                    # All denominators are zero - can't compute ratio
                    zero_std_mask[f_idx] = True
                else:
                    ratios = numer[valid_mask] / denom[valid_mask]
                    if ratios.std() == 0:
                        zero_std_mask[f_idx] = True

        elif distribution == 'mvnormal':
            # For mvnormal: check if std=0 in ALL dimensions
            # counts_ntc_array shape: (features, cells, dimensions)
            for f_idx in range(counts_ntc_array.shape[0]):
                feature_data = counts_ntc_array[f_idx, :, :]  # (cells, dimensions)
                # Compute std across cells for each dimension
                dim_stds = feature_data.std(axis=0)  # std for each dimension
                if np.all(dim_stds == 0):
                    zero_std_mask[f_idx] = True

        else:
            # For negbinom, normal: check if std=0 in raw counts
            if counts_ntc_array.ndim == 2:
                if modality.cells_axis == 0:
                    feature_std_ntc = counts_ntc_array.std(axis=0)  # cells are rows
                else:
                    feature_std_ntc = counts_ntc_array.std(axis=1)  # features are rows
                zero_std_mask = feature_std_ntc == 0
            else:
                raise ValueError(f"Unexpected dimensions for distribution '{distribution}': {counts_ntc_array.ndim}")

        num_zero_std = zero_std_mask.sum()

        # Step 3: For multinomial (donor/acceptor), filter features with only one category
        only_one_category_mask = np.zeros(len(feature_sums_ntc), dtype=bool)
        if distribution == 'multinomial':
            # Check how many categories have non-zero counts for each feature
            # counts_ntc_array shape: (features, cells, categories)
            for f_idx in range(counts_ntc_array.shape[0]):
                # Sum across cells to get total counts per category
                category_totals = counts_ntc_array[f_idx, :, :].sum(axis=0)
                num_nonzero_categories = (category_totals > 0).sum()
                if num_nonzero_categories <= 1:
                    only_one_category_mask[f_idx] = True
        num_single_category = only_one_category_mask.sum()

        # Step 4: Combine filtering criteria
        # Features to REMOVE from fitting (but not from data)
        needs_filtering_mask = zero_std_mask | only_one_category_mask

        # Features with zero counts will NOT be removed, but will be handled post-hoc
        # They still need to be excluded from fitting
        needs_exclusion_mask = zero_count_mask | needs_filtering_mask

        # Step 5: Warnings and checks
        if num_zero_count > 0:
            warnings.warn(
                f"[WARNING] {num_zero_count} feature(s) have zero counts in NTC. "
                f"These will not be corrected for technical factors (alpha_y will be set to 1).",
                UserWarning
            )

        if num_zero_std > 0:
            warnings.warn(
                f"[WARNING] {num_zero_std} feature(s) have zero standard deviation in NTC and will be excluded from fitting.",
                UserWarning
            )

        if num_single_category > 0:
            warnings.warn(
                f"[WARNING] {num_single_category} donor/acceptor feature(s) have only one category and will be excluded from fitting.",
                UserWarning
            )

        # Check if cis gene has zero counts (for gene modality only)
        if modality_name == 'gene' and self.model.cis_gene is not None:
            # Find cis gene index in feature_meta
            cis_gene_features = modality.feature_meta.index.tolist()
            if self.model.cis_gene in cis_gene_features:
                cis_gene_idx = cis_gene_features.index(self.model.cis_gene)
                if zero_count_mask[cis_gene_idx]:
                    warnings.warn(
                        f"[WARNING] The cis gene '{self.model.cis_gene}' has zero counts in NTC! "
                        f"This may affect trans modeling.",
                        UserWarning
                    )

        # Step 6: Subset data for fitting (exclude features that can't be fit)
        fit_mask = ~needs_exclusion_mask
        num_features_to_fit = fit_mask.sum()

        if num_features_to_fit == 0:
            raise ValueError("No features left to fit after filtering! All features have zero counts, zero variance, or single categories.")

        # Subset counts_ntc_array and denominator for fitting
        if counts_ntc_array.ndim == 2:
            if modality.cells_axis == 0:
                counts_ntc_for_fit = counts_ntc_array[:, fit_mask]
            else:
                counts_ntc_for_fit = counts_ntc_array[fit_mask, :]
        elif counts_ntc_array.ndim == 3:
            counts_ntc_for_fit = counts_ntc_array[fit_mask, :, :]

        # Also subset denominator if present
        denominator_ntc_array = None
        if denominator is not None:
            # Get denominator for NTC cells
            if denominator.ndim == 2:
                if modality.cells_axis == 1:
                    denominator_ntc_array = denominator[:, ntc_indices]
                else:
                    denominator_ntc_array = denominator[ntc_indices, :]
            elif denominator.ndim == 3:
                denominator_ntc_array = denominator[:, ntc_indices, :]

            # Subset denominator for fitting
            if denominator_ntc_array.ndim == 2:
                if modality.cells_axis == 0:
                    denominator_ntc_for_fit = denominator_ntc_array[:, fit_mask]
                else:
                    denominator_ntc_for_fit = denominator_ntc_array[fit_mask, :]
            elif denominator_ntc_array.ndim == 3:
                denominator_ntc_for_fit = denominator_ntc_array[fit_mask, :, :]
        else:
            denominator_ntc_for_fit = None

        # Replace counts_ntc_array with filtered version for fitting
        counts_ntc_array = counts_ntc_for_fit
        if denominator is not None:
            denominator_ntc_array = denominator_ntc_for_fit

        # Prepare inputs to pyro
        N = meta_ntc.shape[0]

        # Transpose to [N, T, ...] format expected by model
        if counts_ntc_array.ndim == 2:
            if modality.cells_axis == 1:
                y_obs_ntc = counts_ntc_array.T  # [T, N] -> [N, T]
                T = counts_ntc_array.shape[0]  # Number of features
            else:
                y_obs_ntc = counts_ntc_array  # Already [N, T]
                T = counts_ntc_array.shape[1]  # Number of features
        elif counts_ntc_array.ndim == 3:
            # 3D: (features, cells, categories/dimensions)
            # Transpose to (cells, features, categories/dimensions)
            y_obs_ntc = counts_ntc_array.transpose(1, 0, 2)  # [T, N, K] -> [N, T, K]
            T = counts_ntc_array.shape[0]  # Number of features
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_ntc_array.ndim}")

        C = meta_ntc['technical_group_code'].nunique()
        groups_ntc = meta_ntc['technical_group_code'].values
        guides = meta_ntc['guide_code'].values

        # Calculate priors
        # For 3D distributions, handle specially based on distribution type
        if y_obs_ntc.ndim == 3:
            if distribution == 'multinomial':
                # For multinomial, sum over categories to get total counts per feature per cell
                y_obs_ntc_for_priors = y_obs_ntc.sum(axis=2)  # [N, T, K] -> [N, T]
            elif distribution == 'mvnormal':
                # For mvnormal, DON'T sum - compute priors from mean across dimensions
                # This gives us a better estimate of central tendency
                y_obs_ntc_for_priors = y_obs_ntc.mean(axis=2)  # [N, T, D] -> [N, T]
            else:
                y_obs_ntc_for_priors = y_obs_ntc.sum(axis=2)  # Default: sum
        else:
            y_obs_ntc_for_priors = y_obs_ntc

        # Handle sum factors (only for distributions that need them)
        if sum_factor_col is not None:
            y_obs_ntc_factored = y_obs_ntc_for_priors / meta_ntc[sum_factor_col].values.reshape(-1, 1)
        else:
            y_obs_ntc_factored = y_obs_ntc_for_priors

        baseline_mask = (groups_ntc == 0)
        mu_x_mean = np.mean(y_obs_ntc_factored[baseline_mask, :],axis=0)
        guide_means = np.array([np.mean(y_obs_ntc_factored[guides == g], axis=0) for g in np.unique(guides)])
        mu_x_sd = np.std(guide_means, axis=0) + epsilon

        # For distributions with potentially negative values, don't add epsilon to mean
        # Only add epsilon to avoid division by zero in sd
        if distribution not in ['normal', 'mvnormal']:
            mu_x_mean = mu_x_mean + epsilon

        # Convert to tensors
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.model.device)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.model.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.model.device)

        # Handle sum factors
        if sum_factor_col is not None:
            sum_factor_ntc_tensor = torch.tensor(meta_ntc[sum_factor_col].values, dtype=torch.float32, device=self.model.device)
        else:
            sum_factor_ntc_tensor = torch.ones(N, dtype=torch.float32, device=self.model.device)

        groups_ntc_tensor = torch.tensor(groups_ntc, dtype=torch.long, device=self.model.device)
        y_obs_ntc_tensor = torch.tensor(y_obs_ntc, dtype=torch.float32, device=self.model.device)
        mu_x_mean_tensor = torch.tensor(mu_x_mean, dtype=torch.float32, device=self.model.device)
        mu_x_sd_tensor = torch.tensor(mu_x_sd, dtype=torch.float32, device=self.model.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.model.device)

        # Handle denominator (for binomial)
        denominator_ntc_tensor = None
        if denominator is not None:
            # Subset denominator using same ntc_indices
            if modality.cells_axis == 1:
                denominator_ntc = denominator[:, ntc_indices].T  # [T, N_ntc] -> [N_ntc, T]
            else:
                denominator_ntc = denominator[ntc_indices, :]  # [N_ntc, T]
            denominator_ntc_tensor = torch.tensor(denominator_ntc, dtype=torch.float32, device=self.model.device)

        # Detect data dimensions (for multinomial and mvnormal)
        from ..distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs_ntc.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs_ntc.shape[2]
                elif distribution == 'mvnormal':
                    D = y_obs_ntc.shape[2]

        def init_loc_fn(site):
            if site["name"] == "log2_alpha_y":
                # Compute per-group log2 ratios vs. baseline group 0
                group_codes = meta_ntc["technical_group_code"].values
                group_labels = np.unique(group_codes)
                group_labels = group_labels[group_labels != 0]  # C-1 groups vs. baseline        
                init_values = []
                for g in group_labels:
                    group_mean = np.mean(y_obs_ntc_factored[group_codes == g], axis=0)
                    baseline_mean = np.mean(y_obs_ntc_factored[group_codes == 0], axis=0)
                    log2_ratio = np.log2(group_mean / baseline_mean)
                    init_values.append(log2_ratio)
        
                init_tensor = torch.tensor(np.stack(init_values), dtype=torch.float32, device=self.model.device)  # shape: [C-1, T]
                return init_tensor
            else:
                return pyro.infer.autoguide.initialization.init_to_median(site)
        
        # Use in guide
        guide_cellline = pyro.infer.autoguide.AutoIAFNormal(self._model_technical, init_loc_fn=init_loc_fn)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(self._model_technical, guide_cellline, optimizer, loss=pyro.infer.Trace_ELBO())
        
        losses = []
        smoothed_loss = None
        for step in range(niters):
            loss = svi.step(
                N,
                T,
                C,
                groups_ntc_tensor,
                y_obs_ntc_tensor,
                sum_factor_ntc_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                epsilon_tensor,
                distribution,
                denominator_ntc_tensor,
                K,
                D,
            )
            losses.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {mu_x_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss


        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = self.model.device.type != "cpu"

        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_cellline.to("cpu")
            self.model.device = torch.device("cpu")

            # Move inputs to CPU
            model_inputs = {
                "N": N,
                "T": T,
                "C": C,
                "groups_ntc_tensor": groups_ntc_tensor.cpu(),
                "y_obs_ntc_tensor": y_obs_ntc_tensor.cpu(),
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor.cpu(),
                "beta_o_alpha_tensor": beta_o_alpha_tensor.cpu(),
                "beta_o_beta_tensor": beta_o_beta_tensor.cpu(),
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor.cpu(),
                "mu_x_mean_tensor": mu_x_mean_tensor.cpu(),
                "mu_x_sd_tensor": mu_x_sd_tensor.cpu(),
                "epsilon_tensor": epsilon_tensor.cpu(),
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor.cpu() if denominator_ntc_tensor is not None else None,
                "K": K,
                "D": D,
            }

        else:
            model_inputs = {
                "N": N,
                "T": T,
                "C": C,
                "groups_ntc_tensor": groups_ntc_tensor,
                "y_obs_ntc_tensor": y_obs_ntc_tensor,
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "epsilon_tensor": epsilon_tensor,
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor,
                "K": K,
                "D": D,
            }

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Mini-batching logic (optional)
        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "y_obs_ntc")

        if minibatch_size is not None:
            from collections import defaultdict

            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_technical = pyro.infer.Predictive(
                self._model_technical,
                guide=guide_cellline,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_technical(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(v.cpu())
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            posterior_samples = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}

        else:
            predictive_technical = pyro.infer.Predictive(
                self._model_technical,
                guide=guide_cellline,
                num_samples=nsamples#,
                #parallel=True
            )
            with torch.no_grad():
                posterior_samples = predictive_technical(**model_inputs)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        # Restore self.model.device if it was changed
        if run_on_cpu:
            self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.model.device to:", self.model.device)

        for k, v in posterior_samples.items():
            posterior_samples[k] = v.cpu()

        # Reconstruct full alpha_y array with 1s for excluded features
        # posterior_samples["alpha_y"] has shape [num_samples, C-1, T_fitted] or [C, T_fitted]
        # We need to expand to [..., C, T_total] where T_total is the original number of features
        alpha_y_fitted = posterior_samples["alpha_y"]
        original_shape = alpha_y_fitted.shape

        # Get total number of features before filtering
        T_total = len(fit_mask)  # Original number of features

        # Determine if we have samples dimension
        if alpha_y_fitted.ndim == 3:
            # Shape: [num_samples, C-1, T_fitted]
            num_samples = original_shape[0]
            C_groups = original_shape[1]
            T_fitted = original_shape[2]

            # Create full alpha_y array initialized with 1s
            alpha_y_full = torch.ones((num_samples, C_groups, T_total),
                                     dtype=alpha_y_fitted.dtype, device=alpha_y_fitted.device)

            # Fill in fitted values at positions indicated by fit_mask
            fit_indices = np.where(fit_mask)[0]
            alpha_y_full[:, :, fit_indices] = alpha_y_fitted

        elif alpha_y_fitted.ndim == 2:
            # Shape: [C, T_fitted]
            C_groups = original_shape[0]
            T_fitted = original_shape[1]

            # Create full alpha_y array initialized with 1s
            alpha_y_full = torch.ones((C_groups, T_total),
                                     dtype=alpha_y_fitted.dtype, device=alpha_y_fitted.device)

            # Fill in fitted values at positions indicated by fit_mask
            fit_indices = np.where(fit_mask)[0]
            alpha_y_full[:, fit_indices] = alpha_y_fitted
        else:
            raise ValueError(f"Unexpected alpha_y shape: {original_shape}")

        # Replace alpha_y in posterior_samples with the full version
        posterior_samples["alpha_y"] = alpha_y_full

        # Also update feature metadata to record which features were excluded
        modality.feature_meta['ntc_zero_count'] = zero_count_mask
        modality.feature_meta['ntc_zero_std'] = zero_std_mask
        modality.feature_meta['ntc_single_category'] = only_one_category_mask
        modality.feature_meta['ntc_excluded_from_fit'] = needs_exclusion_mask
        modality.feature_meta['technical_correction_applied'] = ~needs_exclusion_mask

        # Store results in modality
        modality.alpha_y_prefit = posterior_samples["alpha_y"]
        modality.posterior_samples_technical = posterior_samples

        # Mark exon skipping aggregation as locked (if applicable)
        if modality.is_exon_skipping():
            modality.mark_technical_fit_complete()

        # If primary modality, also store at model level (backward compatibility)
        if modality_name == self.model.primary_modality:
            self.model.loss_technical = losses
            self.model.posterior_samples_technical = posterior_samples

            # IMPORTANT: The primary modality may contain the cis_gene
            # We need to extract it and store separately as alpha_x_prefit
            if self.model.cis_gene is not None:
                # Check if cis_gene is in the ORIGINAL counts (before modality creation)
                # The modality already excludes it, but we need to find its alpha from fit results

                # Get the full alpha_y (which now includes positions for all features)
                full_alpha_y = posterior_samples["alpha_y"]  # Shape: (S, C, T_total)

                # Check if cis_gene is in BOTH:
                # 1) The original counts (self.model.counts) - for backward compatibility
                # 2) The current modality's feature list - to get the correct index
                #
                # NOTE: In the new multi-modal architecture, cis_gene is excluded from
                # the gene modality's features, so we need to check if it's in self.model.counts
                # but NOT try to find it in the fitted alpha_y (which doesn't include it)

                if hasattr(self, 'counts') and self.model.cis_gene in self.model.counts.index:
                    # Check if cis_gene is actually in the modality's features
                    # (In multi-modal, it's excluded from gene modality)
                    modality_feature_ids = [str(fid) for fid in modality.feature_meta['gene'].values]

                    if self.model.cis_gene in modality_feature_ids:
                        # Find the index of cis_gene in the modality's features
                        cis_idx_in_modality = modality_feature_ids.index(self.model.cis_gene)

                        # Extract alpha for cis gene
                        # Shape: (S, C, 1) -> squeeze to (S, C)
                        self.model.alpha_x_prefit = full_alpha_y[:, :, cis_idx_in_modality]
                        self.model.alpha_x_type = 'posterior'

                        # Remove cis gene from alpha_y_prefit
                        # Create mask for all indices except cis_gene
                        all_indices = list(range(full_alpha_y.shape[2]))
                        trans_indices = [i for i in all_indices if i != cis_idx_in_modality]
                        self.model.alpha_y_prefit = full_alpha_y[:, :, trans_indices]
                        self.model.alpha_y_type = 'posterior'

                        print(f"[INFO] Extracted alpha_x_prefit for cis gene '{self.model.cis_gene}' from primary modality")
                        print(f"[INFO] alpha_y_prefit excludes cis gene ({len(trans_indices)} trans genes)")
                    else:
                        # Cis gene in self.model.counts but not in this modality
                        # This is the expected case for multi-modal architecture
                        self.model.alpha_y_prefit = full_alpha_y
                        self.model.alpha_y_type = 'posterior'
                        # alpha_x_prefit will be set in fit_cis from self.model.counts
                        print(f"[INFO] Cis gene '{self.model.cis_gene}' not in modality '{modality_name}' - alpha_x will be fitted in fit_cis")
                else:
                    # Cis gene not in original counts at all
                    self.model.alpha_y_prefit = full_alpha_y
                    self.model.alpha_y_type = 'posterior'
                    print(f"[INFO] Cis gene '{self.model.cis_gene}' not in original counts - alpha_x will be fitted in fit_cis")
            else:
                # No cis gene specified
                self.model.alpha_y_prefit = posterior_samples["alpha_y"]
                self.model.alpha_y_type = 'posterior'

            print(f"[INFO] Stored results in modality '{modality_name}' and at model level (primary modality)")
        else:
            print(f"[INFO] Stored results in modality '{modality_name}'")

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        pyro.clear_param_store()
        import gc
        gc.collect()

        print("Finished fit_technical.")


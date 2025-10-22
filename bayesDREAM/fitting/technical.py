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
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        epsilon_tensor,
        distribution='negbinom',
        denominator_ntc_tensor=None,
        K=None,
        D=None,
    ):
        """
        Technical model used for NTC-only prefit of cell-line effects.
        - negbinom: multiplicative effects on mu (alpha_full_mul), NB dispersion phi_y
        - normal/binomial: additive or logit-scale effects (alpha_full_add)
        - mvnormal: additive effects per (feature, dim), diagonal covariance
        - multinomial: Dirichlet probabilities per (feature, category); no cell-line effects yet
        """
    
        # ----------------------------
        # PLATES for groups and feats
        # ----------------------------
        c_plate  = pyro.plate("c_plate", C - 1, dim=-2)      # groups except baseline
        f_plate  = pyro.plate("feature_plate_technical", T, dim=-1)
    
        # ----------------------------
        # Cell-line effects (group × T)
        # We sample a single latent matrix log2_alpha_y and
        # expose two parameterizations:
        #   multiplicative: alpha_full_mul  (baseline=1)
        #   additive/logit: alpha_full_add  (baseline=0)
        # ----------------------------
        with pyro.plate("trans_plate", T, dim=-1):
            with c_plate:  # [C-1, T]
                log2_alpha_y = pyro.sample(
                    "log2_alpha_y",
                    dist.StudentT(df=3, loc=0.0, scale=20.0)
                )
                alpha_y_mul = pyro.deterministic("alpha_y_mul", 2.0 ** log2_alpha_y)   # multiplicative
                delta_y_add = pyro.deterministic("delta_y_add", log2_alpha_y)          # additive/logit
    
        # Build full [C, T] by adding the baseline row
        if alpha_y_mul.ndim == 2:  # [C-1, T]
            alpha_full_mul = torch.cat(
                [torch.ones(1, T, device=self.model.device), alpha_y_mul.to(self.model.device)],
                dim=0
            )  # [C, T]
            alpha_full_add = torch.cat(
                [torch.zeros(1, T, device=self.model.device), delta_y_add.to(self.model.device)],
                dim=0
            )  # [C, T]
        elif alpha_y_mul.ndim == 3:  # [S, C-1, T]
            S = alpha_y_mul.size(0)
            alpha_full_mul = torch.cat(
                [torch.ones(S, 1, T, device=self.model.device), alpha_y_mul.to(self.model.device)],
                dim=1
            )  # [S, C, T]
            alpha_full_add = torch.cat(
                [torch.zeros(S, 1, T, device=self.model.device), delta_y_add.to(self.model.device)],
                dim=1
            )  # [S, C, T]
        else:
            raise ValueError(f"Unexpected alpha/log2 shapes: {alpha_y_mul.shape}, {delta_y_add.shape}")
    
        # --------------------------------
        # Dispersion / variance priors
        # --------------------------------
        phi_y       = None  # NB overdispersion (as total_count)
        sigma_y     = None  # Normal per-feature std
        sigma_y_mv  = None  # MVN per-(feature,dim) std
    
        if distribution == 'negbinom':
            beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
            with f_plate:
                o_y = pyro.sample("o_y", dist.Exponential(beta_o))     # [T]
            phi_y = 1.0 / (o_y ** 2)                                   # [T]
    
        elif distribution == 'normal':
            with f_plate:
                sigma_y = pyro.sample("sigma_y", dist.HalfCauchy(10.0))  # [T]
    
        elif distribution == 'mvnormal':
            assert D is not None, "mvnormal requires D"
            with f_plate:
                with pyro.plate("dim_plate", D, dim=-2):
                    sigma_y_mv = pyro.sample("sigma_y_mv", dist.HalfCauchy(10.0))  # [T, D]
    
        # --------------------------------
        # Baseline means per distribution
        # --------------------------------
        # NOTE: we DO NOT pre-apply cell-line effects here; samplers will.
        mu_y = None
    
        if distribution in ('negbinom', 'normal'):
            # mu_ntc shape: [T]
            with f_plate:
                if distribution == 'normal':
                    mu_ntc = pyro.sample("mu_ntc", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
                else:  # negbinom
                    mu_ntc = pyro.sample(
                        "mu_ntc",
                        dist.Gamma(
                            (mu_x_mean_tensor**2) / (mu_x_sd_tensor**2),
                            mu_x_mean_tensor / (mu_x_sd_tensor**2)
                        )
                    )
            mu_y = mu_ntc  # [T]
    
        elif distribution == 'mvnormal':
            # per-dimension means: mu_ntc_mv shape [T, D]
            with f_plate:
                # mu_x_mean_tensor and mu_x_sd_tensor are [T, D]
                mu_ntc_mv = pyro.sample("mu_ntc_mv", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
            mu_y = mu_ntc_mv  # [T, D]
    
        elif distribution == 'binomial':
            # Beta prior from empirical NTC successes/totals (data-driven EB)
            y_sum  = y_obs_ntc_tensor.sum(dim=0)                 # [T]
            den_sum = denominator_ntc_tensor.sum(dim=0) + 1e-6   # [T]
            a = y_sum + 1.0
            b = (den_sum - y_sum) + 1.0
            with f_plate:
                mu_ntc = pyro.sample("mu_ntc", dist.Beta(a, b))  # [T]
            mu_y = mu_ntc  # [T] in [0,1]
    
        elif distribution == 'multinomial':
            with f_plate:
                total_counts_per_feature = y_obs_ntc_tensor.sum(dim=0)  # [T, K]
                concentration = total_counts_per_feature + 1.0
                probs_baseline = pyro.sample("probs_baseline", dist.Dirichlet(concentration))  # [T, K]
            mu_y = probs_baseline  # [T, K]
    
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
        # --------------------------------
        # Call distribution-specific sampler
        # --------------------------------
        from ..distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')
    
        if distribution == 'negbinom':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                mu_y=mu_y,                               # [T]
                phi_y_used=phi_y.unsqueeze(-2),          # [1, T]
                alpha_y_full=alpha_full_mul,             # [C, T]
                groups_tensor=groups_ntc_tensor,
                sum_factor_tensor=sum_factor_ntc_tensor,
                N=N, T=T, C=C
            )
    
        elif distribution == 'normal':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                mu_y=mu_y,                               # [T]
                sigma_y=sigma_y,                         # [T]
                alpha_y_full=alpha_full_add,             # [C, T]
                groups_tensor=groups_ntc_tensor,
                N=N, T=T, C=C
            )
    
        elif distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,           # [N, T]
                denominator_tensor=denominator_ntc_tensor,
                mu_y=mu_y,                               # [T] in [0,1]
                alpha_y_full=alpha_full_add,             # [C, T] (add on logit)
                groups_tensor=groups_ntc_tensor,
                N=N, T=T, C=C
            )
    
        elif distribution == 'multinomial':
            mu_y_multi = mu_y.unsqueeze(0).expand(N, T, K)  # [N, T, K]
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y_multi,
                N=N, T=T, K=K
            )
    
        elif distribution == 'mvnormal':
            # diagonal covariance per (T, D)
            cov_y = torch.diag_embed((sigma_y_mv ** 2))          # [T, D, D]
            mu_y_mv = mu_y.unsqueeze(0).expand(N, T, D)          # [N, T, D]
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y_mv,
                cov_y=cov_y,
                alpha_y_full=alpha_full_add,   # [C, T] (sampler broadcasts to [C, T, D])
                groups_tensor=groups_ntc_tensor,
                N=N, T=T, D=D, C=C
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
        niters: int = 50_000,
        nsamples: int = 1_000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4,   # recommended to keep based on cell2location
        beta_o_beta: float = 3,    # recommended to keep based on cell2location
        beta_o_alpha: float = 9,   # recommended to keep based on cell2location
        epsilon: float = 1e-6,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        **kwargs
    ):
        """
        Prefit cell-line technical effects (NTC-only) for a given modality.
        Stores both multiplicative ('alpha_y_mult') and additive/logit ('alpha_y_add') effects.
        """
    
        # ---------------------------
        # Resolve modality/distribution
        # ---------------------------
        if modality_name is None:
            modality_name = self.model.primary_modality
        modality = self.model.get_modality(modality_name)

        if distribution is None:
            distribution = modality.distribution

        if denominator is None and modality.denominator is not None:
            denominator = modality.denominator

        # For primary modality, use original counts (includes cis gene)
        # For other modalities, use modality counts
        if modality_name == self.model.primary_modality and hasattr(self.model, 'counts'):
            # Use original counts which include the cis gene
            if isinstance(self.model.counts, pd.DataFrame):
                counts_to_fit = self.model.counts.values
                # Cell names from the DataFrame columns
                modality_cells = self.model.counts.columns.tolist()
            else:
                counts_to_fit = self.model.counts
                # Fall back to meta cells
                modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]
            print(f"[INFO] Using original counts for primary modality '{modality_name}' (includes cis gene if present)")
        else:
            counts_to_fit = modality.counts
            if modality.cell_names is not None:
                modality_cells = modality.cell_names
            else:
                modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]
    
        print(f"[INFO] Fitting technical model for modality '{modality_name}' (distribution: {distribution})")
    
        # ---------------------------
        # Validate requirements
        # ---------------------------
        from ..distributions import requires_sum_factor, requires_denominator, is_3d_distribution
    
        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")
    
        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")
    
        if "technical_group_code" not in self.model.meta.columns:
            raise ValueError(
                "technical_group_code not set. Call set_technical_groups(covariates) before fit_technical().\n"
                "Example: model.set_technical_groups(['cell_line'])"
            )
    
        print("Running prefit_cellline...")
    
        # ---------------------------
        # Subset to NTC cells
        # ---------------------------
        modality_cell_set = set(modality_cells)
        meta_subset = self.model.meta[self.model.meta['cell'].isin(modality_cell_set)].copy()
        meta_ntc = meta_subset[meta_subset["target"] == "ntc"].copy()
    
        ntc_cell_list = meta_ntc["cell"].tolist()
        ntc_indices = [i for i, c in enumerate(modality_cells) if c in ntc_cell_list]
    
        # Subset counts -> NTC
        if counts_to_fit.ndim == 2:
            counts_ntc_array = counts_to_fit[:, ntc_indices] if modality.cells_axis == 1 else counts_to_fit[ntc_indices, :]
        elif counts_to_fit.ndim == 3:
            counts_ntc_array = counts_to_fit[:, ntc_indices, :]  # cells are axis 1 for 3D
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_to_fit.ndim}")
    
        print(f"[INFO] Modality '{modality_name}': {len(modality_cells)} total cells, {len(ntc_indices)} NTC cells")
    
        # ---------------------------
        # Quality filters per feature
        # ---------------------------
        if counts_ntc_array.ndim == 2:
            feature_sums_ntc = counts_ntc_array.sum(axis=1 if modality.cells_axis == 1 else 0)
        elif counts_ntc_array.ndim == 3:
            feature_sums_ntc = counts_ntc_array.sum(axis=(1, 2))
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_ntc_array.ndim}")
    
        zero_count_mask = feature_sums_ntc == 0
        zero_std_mask = np.zeros(len(feature_sums_ntc), dtype=bool)
    
        if distribution == 'multinomial':
            for f_idx in range(counts_ntc_array.shape[0]):
                feature_counts = counts_ntc_array[f_idx, :, :]  # (cells, K)
                totals = feature_counts.sum(axis=1, keepdims=True)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(totals > 0, feature_counts / totals, 0)
                ratio_stds = ratios.std(axis=0)
                if np.all(ratio_stds == 0):
                    zero_std_mask[f_idx] = True
    
        elif distribution == 'binomial':
            if denominator is None:
                raise ValueError("Binomial distribution requires denominator for variance check")
            denom_ntc = denominator[:, ntc_indices] if denominator.ndim == 2 and modality.cells_axis == 1 else \
                        denominator[ntc_indices, :] if denominator.ndim == 2 else \
                        denominator[:, ntc_indices, :]
            for f_idx in range(counts_ntc_array.shape[0]):
                if modality.cells_axis == 0:
                    numer = counts_ntc_array[:, f_idx]
                    denom = denom_ntc[:, f_idx]
                else:
                    numer = counts_ntc_array[f_idx, :]
                    denom = denom_ntc[f_idx, :]
                valid = denom > 0
                if valid.sum() == 0:
                    zero_std_mask[f_idx] = True
                else:
                    ratios = numer[valid] / denom[valid]
                    if ratios.std() == 0:
                        zero_std_mask[f_idx] = True
    
        elif distribution == 'mvnormal':
            for f_idx in range(counts_ntc_array.shape[0]):
                feature_data = counts_ntc_array[f_idx, :, :]  # (cells, D)
                if np.all(feature_data.std(axis=0) == 0):
                    zero_std_mask[f_idx] = True
    
        else:
            if counts_ntc_array.ndim == 2:
                feature_std_ntc = counts_ntc_array.std(axis=1 if modality.cells_axis == 1 else 0)
                zero_std_mask = feature_std_ntc == 0
            else:
                raise ValueError(f"Unexpected dims for distribution '{distribution}': {counts_ntc_array.ndim}")
    
        only_one_category_mask = np.zeros(len(feature_sums_ntc), dtype=bool)
        if distribution == 'multinomial':
            for f_idx in range(counts_ntc_array.shape[0]):
                category_totals = counts_ntc_array[f_idx, :, :].sum(axis=0)
                if (category_totals > 0).sum() <= 1:
                    only_one_category_mask[f_idx] = True
    
        needs_filtering_mask = zero_std_mask | only_one_category_mask
        needs_exclusion_mask = zero_count_mask | needs_filtering_mask
    
        num_zero_count = zero_count_mask.sum()
        num_zero_std = zero_std_mask.sum()
        num_single_category = only_one_category_mask.sum()
    
        if num_zero_count > 0:
            warnings.warn(f"[WARNING] {num_zero_count} feature(s) have zero counts in NTC; alpha set to baseline for them.", UserWarning)
        if num_zero_std > 0:
            warnings.warn(f"[WARNING] {num_zero_std} feature(s) have zero std in NTC; excluded from fitting.", UserWarning)
        if num_single_category > 0:
            warnings.warn(f"[WARNING] {num_single_category} multinomial feature(s) have 1 category; excluded from fitting.", UserWarning)
    
        # ---------------------------
        # Subset to features that can be fit
        # ---------------------------
        fit_mask = ~needs_exclusion_mask
        if fit_mask.sum() == 0:
            raise ValueError("No features left to fit after filtering!")
    
        if counts_ntc_array.ndim == 2:
            counts_ntc_for_fit = counts_ntc_array[fit_mask, :] if modality.cells_axis == 1 else counts_ntc_array[:, fit_mask]
        else:
            counts_ntc_for_fit = counts_ntc_array[fit_mask, :, :]
    
        denominator_ntc_for_fit = None
        if denominator is not None:
            if denominator.ndim == 2:
                denom_ntc = denominator[:, ntc_indices] if modality.cells_axis == 1 else denominator[ntc_indices, :]
                denominator_ntc_for_fit = denom_ntc[fit_mask, :] if modality.cells_axis == 1 else denom_ntc[:, fit_mask]
            elif denominator.ndim == 3:
                denom_ntc = denominator[:, ntc_indices, :]
                denominator_ntc_for_fit = denom_ntc[fit_mask, :, :]
    
        # ---------------------------
        # Build tensors for model
        # ---------------------------
        N = meta_ntc.shape[0]
    
        if counts_ntc_for_fit.ndim == 2:
            if modality.cells_axis == 1:
                y_obs_ntc = counts_ntc_for_fit.T   # [T, N] -> [N, T]
                T_fit = counts_ntc_for_fit.shape[0]
            else:
                y_obs_ntc = counts_ntc_for_fit     # [N, T]
                T_fit = counts_ntc_for_fit.shape[1]
        else:
            y_obs_ntc = counts_ntc_for_fit.transpose(1, 0, 2)  # [T, N, K/D] -> [N, T, K/D]
            T_fit = counts_ntc_for_fit.shape[0]
    
        C = meta_ntc['technical_group_code'].nunique()
        groups_ntc = meta_ntc['technical_group_code'].values
        guides = meta_ntc['guide_code'].values
    
        # Detect K/D
        K = None
        D = None
        if is_3d_distribution(distribution):
            if distribution == 'multinomial':
                K = y_obs_ntc.shape[2]
            elif distribution == 'mvnormal':
                D = y_obs_ntc.shape[2]
    
        # ---------------------------
        # Build data for priors per distribution
        # ---------------------------
        # y_obs_ntc_for_priors: [N, T] (or mean over last axis when 3D for mvnormal)
        if y_obs_ntc.ndim == 3:
            if distribution == 'multinomial':
                y_obs_ntc_for_priors = y_obs_ntc.sum(axis=2)   # [N, T]
            elif distribution == 'mvnormal':
                y_obs_ntc_for_priors = y_obs_ntc.mean(axis=2)  # [N, T]
            else:
                y_obs_ntc_for_priors = y_obs_ntc.sum(axis=2)
        else:
            y_obs_ntc_for_priors = y_obs_ntc
    
        # Size-factor use only for NB; otherwise ignore
        if distribution == 'negbinom' and sum_factor_col is not None:
            y_obs_ntc_factored = y_obs_ntc_for_priors / meta_ntc[sum_factor_col].values.reshape(-1, 1)
        else:
            y_obs_ntc_factored = y_obs_ntc_for_priors
    
        # ---- mu_x priors per distribution ----
        if distribution == 'negbinom':
            baseline_mask = (groups_ntc == 0)
            mu_x_mean = np.mean(y_obs_ntc_factored[baseline_mask, :], axis=0)                 # [T_fit]
            guide_means = np.array([np.mean(y_obs_ntc_factored[guides == g], axis=0) for g in np.unique(guides)])
            mu_x_sd = np.std(guide_means, axis=0) + epsilon
            mu_x_mean = mu_x_mean + epsilon  # strictly positive for Gamma
    
        elif distribution == 'normal':
            baseline_mask = (groups_ntc == 0)
            mu_x_mean = np.mean(y_obs_ntc_factored[baseline_mask, :], axis=0)                 # [T_fit]
            guide_means = np.array([np.mean(y_obs_ntc_factored[guides == g], axis=0) for g in np.unique(guides)])
            mu_x_sd = np.std(guide_means, axis=0) + epsilon                                   # [T_fit]
    
        elif distribution == 'mvnormal':
            # compute per-dimension means/sds: shapes [T_fit, D]
            mu_x_mean = np.mean(y_obs_ntc, axis=0)                                            # [T_fit, D]
            gids = np.unique(guides)
            guide_means = np.stack([np.mean(y_obs_ntc[guides == g], axis=0) for g in gids], axis=0)  # [G, T_fit, D]
            mu_x_sd = np.std(guide_means, axis=0) + epsilon                                   # [T_fit, D]
    
        else:
            # binomial & multinomial: handled with Beta/Dirichlet in the model
            mu_x_mean = np.zeros((T_fit,), dtype=float)
            mu_x_sd   = np.ones((T_fit,), dtype=float)
    
        # Tensors
        beta_o_beta_tensor  = torch.tensor(beta_o_beta,  dtype=torch.float32, device=self.model.device)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.model.device)
    
        if distribution == 'mvnormal':
            mu_x_mean_tensor = torch.tensor(mu_x_mean, dtype=torch.float32, device=self.model.device)  # [T_fit, D]
            mu_x_sd_tensor   = torch.tensor(mu_x_sd,   dtype=torch.float32, device=self.model.device)  # [T_fit, D]
        else:
            mu_x_mean_tensor = torch.tensor(mu_x_mean, dtype=torch.float32, device=self.model.device)  # [T_fit]
            mu_x_sd_tensor   = torch.tensor(mu_x_sd,   dtype=torch.float32, device=self.model.device)  # [T_fit]
    
        if sum_factor_col is not None and distribution == 'negbinom':
            sum_factor_ntc_tensor = torch.tensor(meta_ntc[sum_factor_col].values, dtype=torch.float32, device=self.model.device)
        else:
            sum_factor_ntc_tensor = torch.ones(N, dtype=torch.float32, device=self.model.device)
    
        groups_ntc_tensor   = torch.tensor(groups_ntc, dtype=torch.long, device=self.model.device)
        y_obs_ntc_tensor    = torch.tensor(y_obs_ntc, dtype=torch.float32, device=self.model.device)
        epsilon_tensor      = torch.tensor(epsilon, dtype=torch.float32, device=self.model.device)
    
        denominator_ntc_tensor = None
        if denominator is not None:
            if modality.cells_axis == 1:
                denom_ntc = denominator[:, ntc_indices].T   # [T, N] -> [N, T]
            else:
                denom_ntc = denominator[ntc_indices, :]     # [N, T]
            denominator_ntc_tensor = torch.tensor(denom_ntc, dtype=torch.float32, device=self.model.device)
    
        # ---------------------------
        # Guide (init) for log2_alpha_y
        # ---------------------------
        def init_loc_fn(site):
            if site["name"] == "log2_alpha_y":
                # Sensible init:
                if distribution == 'negbinom':
                    group_codes = meta_ntc["technical_group_code"].values
                    group_labels = np.array(sorted(set(group_codes) - {0}))
                    init_values = []
                    baseline_mean = np.mean(y_obs_ntc_factored[group_codes == 0], axis=0) + 1e-8
                    for g in group_labels:
                        group_mean = np.mean(y_obs_ntc_factored[group_codes == g], axis=0) + 1e-8
                        init_values.append(np.log2(group_mean / baseline_mean))
                    init_arr = np.stack(init_values) if len(init_values) else np.zeros((0, T_fit), dtype=np.float32)
                    return torch.tensor(init_arr, dtype=torch.float32, device=self.model.device)
                else:
                    # neutral init for additive/logit models
                    return torch.zeros((C - 1, T_fit), dtype=torch.float32, device=self.model.device)
            else:
                return pyro.infer.autoguide.initialization.init_to_median(site)
    
        guide_cellline = pyro.infer.autoguide.AutoIAFNormal(self._model_technical, init_loc_fn=init_loc_fn)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(self._model_technical, guide_cellline, optimizer, loss=pyro.infer.Trace_ELBO())
    
        # ---------------------------
        # Optimize
        # ---------------------------
        losses = []
        smoothed_loss = None
        for step in range(niters):
            loss = svi.step(
                N, T_fit, C,
                groups_ntc_tensor,
                y_obs_ntc_tensor,
                sum_factor_ntc_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                epsilon_tensor,
                distribution,
                denominator_ntc_tensor,
                K, D,
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
    
        # ---------------------------
        # Predictive (optionally on CPU)
        # ---------------------------
        run_on_cpu = self.model.device.type != "cpu"
        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_cellline.to("cpu")
            self.model.device = torch.device("cpu")
    
            model_inputs = {
                "N": N, "T": T_fit, "C": C,
                "groups_ntc_tensor": groups_ntc_tensor.cpu(),
                "y_obs_ntc_tensor": y_obs_ntc_tensor.cpu(),
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor.cpu(),
                "beta_o_alpha_tensor": beta_o_alpha_tensor.cpu(),
                "beta_o_beta_tensor": beta_o_beta_tensor.cpu(),
                "mu_x_mean_tensor": mu_x_mean_tensor.cpu(),
                "mu_x_sd_tensor": mu_x_sd_tensor.cpu(),
                "epsilon_tensor": epsilon_tensor.cpu(),
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor.cpu() if denominator_ntc_tensor is not None else None,
                "K": K, "D": D,
            }
        else:
            model_inputs = {
                "N": N, "T": T_fit, "C": C,
                "groups_ntc_tensor": groups_ntc_tensor,
                "y_obs_ntc_tensor": y_obs_ntc_tensor,
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "epsilon_tensor": epsilon_tensor,
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor,
                "K": K, "D": D,
            }
    
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc; gc.collect()
    
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "y_obs_ntc")
    
        if minibatch_size is not None:
            from collections import defaultdict
            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_technical = pyro.infer.Predictive(
                self._model_technical, guide=guide_cellline, num_samples=minibatch_size, parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, nsamples, minibatch_size):
                    samples = predictive_technical(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(v.cpu())
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
            posterior_samples = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}
        else:
            predictive_technical = pyro.infer.Predictive(
                self._model_technical, guide=guide_cellline, num_samples=nsamples
            )
            with torch.no_grad():
                posterior_samples = predictive_technical(**model_inputs)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
    
        if run_on_cpu:
            self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.model.device to:", self.model.device)
    
        for k, v in posterior_samples.items():
            posterior_samples[k] = v.cpu()
    
        # ----------------------------------------
        # Reconstruct full α (all original T)
        # multiplicative baseline=1; additive baseline=0
        # ----------------------------------------
        # Back-compat: some guides name the mult version "alpha_y_mul"
        if "alpha_y" not in posterior_samples and "alpha_y_mul" in posterior_samples:
            posterior_samples["alpha_y"] = posterior_samples["alpha_y_mul"]
    
        def _reconstruct_full(alpha_fit, baseline_value, fit_mask_bool):
            fit_idx = np.where(fit_mask_bool)[0]
            if alpha_fit.dim() == 3:
                S, Cminus1, T_fit_local = alpha_fit.shape
                full = torch.full((S, Cminus1 + 1, len(fit_mask_bool)),
                                  baseline_value, dtype=alpha_fit.dtype, device=alpha_fit.device)
                full[:, 1:, fit_idx] = alpha_fit
            elif alpha_fit.dim() == 2:
                Cminus1, T_fit_local = alpha_fit.shape
                full = torch.full((Cminus1 + 1, len(fit_mask_bool)),
                                  baseline_value, dtype=alpha_fit.dtype, device=alpha_fit.device)
                full[1:, fit_idx] = alpha_fit
            else:
                raise ValueError(f"Unexpected alpha shape: {alpha_fit.shape}")
            return full
    
        # multiplicative
        alpha_y_mult_fit = posterior_samples["alpha_y"]                 # [S?, C-1, T_fit]
        alpha_y_mult_full = _reconstruct_full(alpha_y_mult_fit, baseline_value=1.0, fit_mask_bool=fit_mask)
    
        # additive/logit (log2_alpha_y)
        log2_alpha_fit = posterior_samples["log2_alpha_y"]              # [S?, C-1, T_fit]
        alpha_y_add_full = _reconstruct_full(log2_alpha_fit, baseline_value=0.0, fit_mask_bool=fit_mask)
    
        # Store in posterior dict
        posterior_samples["alpha_y"]       = alpha_y_mult_full   # back-compat
        posterior_samples["alpha_y_mult"]  = alpha_y_mult_full
        posterior_samples["alpha_y_add"]   = alpha_y_add_full
    
        # ----------------------------------------
        # Feature metadata flags
        # ----------------------------------------
        # For primary modality fitted with original counts, store metadata in base class
        if modality_name == self.model.primary_modality and hasattr(self.model, 'counts') and isinstance(self.model.counts, pd.DataFrame):
            # Create a metadata DataFrame for all features in original counts
            if not hasattr(self.model, 'counts_meta'):
                self.model.counts_meta = pd.DataFrame(index=self.model.counts.index)
            self.model.counts_meta['ntc_zero_count']           = zero_count_mask
            self.model.counts_meta['ntc_zero_std']             = zero_std_mask
            self.model.counts_meta['ntc_single_category']      = only_one_category_mask
            self.model.counts_meta['ntc_excluded_from_fit']    = needs_exclusion_mask
            self.model.counts_meta['technical_correction_applied'] = ~needs_exclusion_mask
        else:
            # Store in modality metadata
            modality.feature_meta['ntc_zero_count']           = zero_count_mask
            modality.feature_meta['ntc_zero_std']             = zero_std_mask
            modality.feature_meta['ntc_single_category']      = only_one_category_mask
            modality.feature_meta['ntc_excluded_from_fit']    = needs_exclusion_mask
            modality.feature_meta['technical_correction_applied'] = ~needs_exclusion_mask
    
        # ----------------------------------------
        # Persist results
        # ----------------------------------------
        modality.alpha_y_prefit       = posterior_samples["alpha_y"]        # multiplicative (back-compat)
        modality.alpha_y_prefit_mult  = posterior_samples["alpha_y_mult"]
        modality.alpha_y_prefit_add   = posterior_samples["alpha_y_add"]
        modality.posterior_samples_technical = posterior_samples
    
        if modality.is_exon_skipping():
            modality.mark_technical_fit_complete()
    
        if modality_name == self.model.primary_modality:
            self.model.loss_technical = losses
            self.model.posterior_samples_technical = posterior_samples

            # Extract cis gene alpha if it exists in the fitted features
            if self.model.cis_gene is not None and 'cis' in self.model.modalities:
                # Check if cis gene was included in the fit (it should be when fitting with original counts)
                # The original counts (passed to base class) include the cis gene
                if hasattr(self.model, 'counts') and self.model.cis_gene in self.model.counts.index:
                    # Get the position of cis gene in the ORIGINAL counts (before filtering)
                    all_genes_orig = self.model.counts.index.tolist()
                    if self.model.cis_gene in all_genes_orig:
                        cis_idx_orig = all_genes_orig.index(self.model.cis_gene)

                        # Check if this index is within the fitted features (not filtered out)
                        full_alpha_y = posterior_samples["alpha_y"]  # (S?, C, T_all)
                        if cis_idx_orig < full_alpha_y.shape[-1]:
                            # Cis gene was included in fit
                            self.model.alpha_x_prefit = full_alpha_y[..., cis_idx_orig]  # (S?, C)
                            self.model.alpha_x_type = 'posterior'

                            # For alpha_y_prefit, exclude cis gene
                            all_idx = list(range(full_alpha_y.shape[-1]))
                            trans_idx = [i for i in all_idx if i != cis_idx_orig]
                            self.model.alpha_y_prefit = full_alpha_y[..., trans_idx]
                            self.model.alpha_y_type = 'posterior'
                            print(f"[INFO] Extracted alpha_x_prefit for cis '{self.model.cis_gene}' (index {cis_idx_orig}) and excluded it from alpha_y_prefit")
                        else:
                            # Cis gene was filtered out during fit
                            self.model.alpha_y_prefit = full_alpha_y
                            self.model.alpha_y_type = 'posterior'
                            print(f"[INFO] Cis '{self.model.cis_gene}' was filtered out during fit - will be fit separately in fit_cis")
                    else:
                        self.model.alpha_y_prefit = posterior_samples["alpha_y"]
                        self.model.alpha_y_type = 'posterior'
                        print(f"[INFO] Cis '{self.model.cis_gene}' not in original counts - will be fit in fit_cis")
                else:
                    self.model.alpha_y_prefit = posterior_samples["alpha_y"]
                    self.model.alpha_y_type = 'posterior'
                    print(f"[INFO] No original counts available - will fit cis in fit_cis later")
            else:
                self.model.alpha_y_prefit = posterior_samples["alpha_y"]
                self.model.alpha_y_type = 'posterior'

            print(f"[INFO] Stored results in modality '{modality_name}' and at model level (primary modality)")
        else:
            print(f"[INFO] Stored results in modality '{modality_name}'")
    
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        pyro.clear_param_store()
        import gc; gc.collect()
    
        print("Finished fit_technical.")


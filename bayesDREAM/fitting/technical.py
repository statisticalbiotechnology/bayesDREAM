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
        - multinomial: Dirichlet probabilities per (feature, category); additive or logit-scale effects (alpha_full_add)
        """
    
        # ----------------------------
        # PLATES for groups and feats
        # ----------------------------
        c_plate  = pyro.plate("c_plate", C - 1, dim=-2)      # groups except baseline
        f_plate  = pyro.plate("feature_plate_technical", T, dim=-1)
    
        # ----------------------------
        # Cell-line effects (group × T [× K])
        # We sample a single latent matrix log2_alpha_y and
        # expose two parameterizations:
        #   multiplicative: alpha_full_mul  (baseline=1)
        #   additive/logit: alpha_full_add  (baseline=0)
        # ----------------------------
        if distribution == "multinomial":
            assert K is not None, "multinomial requires K"
        
            # ---- Cell-line logits α for NON-baseline groups (baseline=0 implicit) ----
            # Avoid plate dims entirely to prevent shape drift in SVI/guide.
            # Sample a fixed-shape tensor as a single event of size (C-1, T, K).
            alpha_logits_y = pyro.sample(
                "alpha_logits_y",  # renamed from *_raw to be the actual latent
                dist.StudentT(df=3, loc=0.0, scale=20.0)
                    .expand([C - 1, T, K])
                    .to_event(3)  # treat all dims as event -> guide won't add plate dims
            )  # [C-1, T, K]
        
            # Center across categories K to keep logits identifiable
            alpha_logits_y = pyro.deterministic(
                "alpha_logits_y_centered",
                alpha_logits_y - alpha_logits_y.mean(dim=-1, keepdim=True)
            )  # [C-1, T, K]
        
            # Build full [C, T, K] (or [S, C, T, K]) with baseline=0
            if alpha_logits_y.dim() == 3:
                # [C-1, T, K]  ->  [C, T, K]
                alpha_full_add_logits = torch.cat(
                    [
                        torch.zeros(1, T, K, device=self.model.device, dtype=alpha_logits_y.dtype),
                        alpha_logits_y.to(self.model.device),
                    ],
                    dim=0,
                )
            elif alpha_logits_y.dim() == 4:
                # [S, C-1, T, K]  ->  [S, C, T, K]
                S = alpha_logits_y.size(0)
                alpha_full_add_logits = torch.cat(
                    [
                        torch.zeros(S, 1, T, K, device=self.model.device, dtype=alpha_logits_y.dtype),
                        alpha_logits_y.to(self.model.device),
                    ],
                    dim=1,
                )
            else:
                raise ValueError(f"Unexpected alpha_logits_y shape: {tuple(alpha_logits_y.shape)}")
            
            alpha_full_mul = None
            alpha_full_add = alpha_full_add_logits

            #print(f'[DEBUG], alpha_logits_y shape = {alpha_logits_y.shape}, expect [C-1={C-1}, T={T}, K={K}')
            #print(f'[DEBUG], alpha_full_add shape = {alpha_full_add.shape}')
        else:
            # reuse f_plate instead of a new trans_plate
            with f_plate:
                with c_plate:  # [C-1, T]
                    log2_alpha_y = pyro.sample(
                        "log2_alpha_y",
                        dist.StudentT(df=3, loc=0.0, scale=20.0)
                    )
                    alpha_y_mul = pyro.deterministic("alpha_y_mul", 2.0 ** log2_alpha_y)
                    delta_y_add = pyro.deterministic("delta_y_add", log2_alpha_y)

            if alpha_y_mul.ndim == 2:
                alpha_full_mul = torch.cat(
                    [torch.ones(1, T, device=self.model.device), alpha_y_mul.to(self.model.device)],
                    dim=0
                )
                alpha_full_add = torch.cat(
                    [torch.zeros(1, T, device=self.model.device), delta_y_add.to(self.model.device)],
                    dim=0
                )
            elif alpha_y_mul.ndim == 3:
                S = alpha_y_mul.size(0)
                alpha_full_mul = torch.cat(
                    [torch.ones(S, 1, T, device=self.model.device), alpha_y_mul.to(self.model.device)],
                    dim=1
                )
                alpha_full_add = torch.cat(
                    [torch.zeros(S, 1, T, device=self.model.device), delta_y_add.to(self.model.device)],
                    dim=1
                )
            else:
                raise ValueError(f"Unexpected alpha/log2 shapes: {alpha_y_mul.shape}, {delta_y_add.shape}")

        #print(f'[DEBUG], alpha_full_add shape = {alpha_full_add.shape}')
    
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
            # Empirical-Bayes Beta with bounded concentration (stable when denom >> counts)
            y_sum  = y_obs_ntc_tensor.sum(dim=0).float()    # [T]
            den_sum = denominator_ntc_tensor.sum(dim=0).float()  # [T]

            # Smooth p-hat to keep it off 0/1 even if den_sum==0
            p_hat = (y_sum + 0.5) / (den_sum + 1.0)         # [T] in (0,1)
            p_hat = torch.clamp(p_hat, 1e-6, 1 - 1e-6)

            # Cap effective sample size: informative but not razor-sharp
            # tune these if needed (e.g., 20..100)
            kappa = torch.clamp(den_sum, min=20.0, max=200.0)

            # Tiny floor to avoid exactly 0 concentration parameters
            a = p_hat * kappa + 1e-3
            b = (1.0 - p_hat) * kappa + 1e-3

            with f_plate:
                mu_ntc = pyro.sample("mu_ntc", dist.Beta(a, b))  # [T]
            mu_y = mu_ntc
    
        elif distribution == 'multinomial':
            total_counts_per_feature = y_obs_ntc_tensor.sum(dim=0)  # [T, K]
            concentration = total_counts_per_feature + 1.0          # [T, K]
            # Each feature (T) has its own Dirichlet over K categories.
            with f_plate:  # <- add this
                probs_baseline = pyro.sample("probs_baseline", dist.Dirichlet(concentration))
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
            # mu_y is [T, K] baseline; expand to [N, T, K]
            mu_y_multi = mu_y.unsqueeze(0).expand(N, T, K)
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y_multi,
                alpha_y_full=alpha_full_add,     # <— ADD THIS
                groups_tensor=groups_ntc_tensor, # <— ADD THIS
                N=N, T=T, K=K, C=C
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
        if not covariates:
            raise ValueError("covariates must not be empty")
    
        missing_cols = [c for c in covariates if c not in self.model.meta.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in meta: {missing_cols}")
    
        self.model.meta["technical_group_code"] = self.model.meta.groupby(covariates).ngroup()
        print(f"[INFO] Set technical_group_code with {self.model.meta['technical_group_code'].nunique()} groups based on {covariates}")
    
        # ---- SAFEGUARD: ensure every technical group is represented among NTCs ----
        if "target" in self.model.meta.columns:
            all_groups  = set(self.model.meta["technical_group_code"].unique().tolist())
            ntc_groups  = set(self.model.meta.loc[self.model.meta["target"] == "ntc", "technical_group_code"].unique().tolist())
            missing_ntc = sorted(all_groups - ntc_groups)
            if missing_ntc:
                n_drop = int(self.model.meta["technical_group_code"].isin(missing_ntc).sum())
                warnings.warn(
                    "[WARNING] Dropping cells from technical groups with no NTC representation. "
                    f"Groups: {missing_ntc} | Cells dropped: {n_drop}",
                    UserWarning
                )
                self.model.meta = self.model.meta.loc[~self.model.meta["technical_group_code"].isin(missing_ntc)].copy()
                # Optional: re-announce counts after drop
                print(f"[INFO] After dropping non-NTC groups, {self.model.meta['technical_group_code'].nunique()} technical group(s) remain.")


    ########################################################
    # Step 1: Optional Prefit for alpha_y (NTC only)
    ########################################################
    def fit_technical(
        self,
        sum_factor_col: str = 'sum_factor',
        lr: float = 1e-3,
        niters: int = None,
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

        # ---------------------------
        # Set conditional default for niters
        # ---------------------------
        if niters is None:
            # Default: 50,000 unless multivariate (multinomial or mvnormal), then 100,000
            if distribution in ('multinomial', 'mvnormal'):
                niters = 100_000
                print(f"[INFO] Using default niters=100,000 for multivariate distribution '{distribution}'")
            else:
                niters = 50_000
                print(f"[INFO] Using default niters=50,000 for distribution '{distribution}'")

        if denominator is None and modality.denominator is not None:
            denominator = modality.denominator

        # ========================================================================
        # CRITICAL VALIDATION: Cis extraction requires negative binomial distribution
        # ========================================================================
        if (
            modality_name == self.model.primary_modality
            and hasattr(self.model, 'counts')
            and self.model.cis_gene is not None
            and 'cis' in self.model.modalities
        ):
            if distribution != 'negbinom':
                raise ValueError(
                    f"Primary modality has distribution '{distribution}', but cis effect extraction "
                    f"requires 'negbinom' distribution. "
                    f"The cis gene/feature must represent count data following a negative binomial distribution. "
                    f"Cannot perform cis modeling with '{distribution}' distribution."
                )

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

        # Count features that are excluded ONLY for each reason (no overlap)
        num_zero_count_only = (zero_count_mask & ~zero_std_mask & ~only_one_category_mask).sum()
        num_zero_std_only = (zero_std_mask & ~only_one_category_mask).sum()
        num_single_category = only_one_category_mask.sum()
        num_excluded = needs_exclusion_mask.sum()

        if num_excluded > 0:
            warnings.warn(
                f"[WARNING] {num_excluded} feature(s) excluded from fitting: "
                f"{num_zero_count_only} zero-count-only, {num_zero_std_only} zero-std, "
                f"{num_single_category} single-category. Alpha set to baseline for excluded features.",
                UserWarning
            )
    
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
        if denominator_ntc_for_fit is not None:
            if modality.cells_axis == 1:
                denom_for_tensor = denominator_ntc_for_fit.T   # [T_fit, N] -> [N, T_fit]
            else:
                denom_for_tensor = denominator_ntc_for_fit     # [N, T_fit]
            denominator_ntc_tensor = torch.tensor(denom_for_tensor, dtype=torch.float32, device=self.model.device)
    
        # ---------------------------
        # Guide (init) for log2_alpha_y
        # ---------------------------
        def init_loc_fn(site):
            name = site["name"]
            if name == "log2_alpha_y":
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
                    return torch.zeros((C - 1, T_fit), dtype=torch.float32, device=self.model.device)

            if name == "alpha_logits_y_raw":
                # start at zero logits (i.e., no shift)
                return torch.zeros((C - 1, T_fit, K), dtype=torch.float32, device=self.model.device)
            if name == "alpha_logits_y":
                return torch.zeros(C - 1, T_fit, K, dtype=torch.float32, device=self.model.device)
            
            return pyro.infer.autoguide.initialization.init_to_median(site)

        # Guide choice: calmer for binomial & multinomial, IAF for others
        if distribution in ('binomial', 'multinomial'):
            guide_cellline = pyro.infer.autoguide.AutoNormal(self._model_technical, init_loc_fn=init_loc_fn)
        else:
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
    
        def _reconstruct_full_2d(alpha_fit, baseline_value, fit_mask_bool):
            fit_idx = np.where(fit_mask_bool)[0]
            if alpha_fit.dim() == 3:           # [S, C-1, T_fit]
                S, Cminus1, _ = alpha_fit.shape
                full = torch.full((S, Cminus1 + 1, len(fit_mask_bool)),
                                  baseline_value, dtype=alpha_fit.dtype, device=alpha_fit.device)
                full[:, 1:, fit_idx] = alpha_fit
            elif alpha_fit.dim() == 2:         # [C-1, T_fit]
                Cminus1, _ = alpha_fit.shape
                full = torch.full((Cminus1 + 1, len(fit_mask_bool)),
                                  baseline_value, dtype=alpha_fit.dtype, device=alpha_fit.device)
                full[1:, fit_idx] = alpha_fit
            else:
                raise ValueError(f"Unexpected alpha shape: {alpha_fit.shape}")
            return full

        def _reconstruct_full_3d(alpha_fit, baseline_value, fit_mask_bool):
            """
            Normalize alpha_fit to [S, C-1, T_fit, K] (S may be 1),
            then expand to full [S, C, T_all, K] inserting the baseline row
            and placing T_fit back into original T_all via fit_mask_bool.
            """
            # last two dims must be (T_fit, K)
            Tfit = alpha_fit.size(-2)
            K_   = alpha_fit.size(-1)
            if Tfit != T_fit or K_ != K:
                raise ValueError(f"alpha_fit trailing dims mismatch: got (T={Tfit},K={K_}) expected (T={T_fit},K={K})")
        
            # ----- Step 1: ensure a leading sample dim S -----
            t = alpha_fit
            if t.dim() == 3:
                # [C-1, T_fit, K] -> [1, C-1, T_fit, K]
                t = t.unsqueeze(0)
            elif t.dim() == 4:
                # [S, C-1, T_fit, K] -> OK
                pass
            elif t.dim() >= 5:
                # e.g. [S, 1, 1, T_fit, K] or even [S, a, b, T_fit, K]
                # Collapse all dims between S and the trailing (T_fit,K) into a single (C-1) axis.
                S = t.size(0)
                # product over dims 1..(dim-3)
                pre_shape = t.shape[1:-2]
                Cminus1 = 1
                for d in pre_shape:
                    Cminus1 *= int(d)
                t = t.reshape(S, Cminus1, Tfit, K_)
            else:
                raise ValueError(f"Unexpected 3D alpha shape: {tuple(alpha_fit.shape)}")
        
            # t is now [S, C-1, T_fit, K]
            S, Cminus1, Tfit, K_ = t.shape
        
            # ----- Step 2: build full [S, C, T_all, K] and scatter T_fit back -----
            fit_idx = torch.as_tensor(np.where(fit_mask_bool)[0], device=t.device)
            full = torch.full(
                (S, Cminus1 + 1, len(fit_mask_bool), K_),
                baseline_value,
                dtype=t.dtype,
                device=t.device,
            )
            full[:, 1:, fit_idx, :] = t  # insert non-baseline rows into fit positions
            return full

    
        if distribution == "multinomial":
            # additive logits only
            alpha_logits_fit = posterior_samples["alpha_logits_y"]       # [S?, C-1, T_fit, K]
            alpha_add_full   = _reconstruct_full_3d(alpha_logits_fit, baseline_value=0.0, fit_mask_bool=fit_mask)

            # store under consistent keys
            posterior_samples["alpha_y_add"]   = alpha_add_full          # [S?, C, T_all, K]
            posterior_samples["alpha_y_mult"]  = None
            posterior_samples["alpha_y"]       = alpha_add_full          # canonical

        else:
            # existing path for 2D α
            alpha_y_mult_fit = posterior_samples["alpha_y"]
            alpha_y_mult_full = _reconstruct_full_2d(alpha_y_mult_fit, baseline_value=1.0, fit_mask_bool=fit_mask)
            log2_alpha_fit = posterior_samples["log2_alpha_y"]
            alpha_y_add_full = _reconstruct_full_2d(log2_alpha_fit, baseline_value=0.0, fit_mask_bool=fit_mask)

            posterior_samples["alpha_y"]       = alpha_y_mult_full
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
        # Persist results at modality level
        # ----------------------------------------
        # For primary modality fitted with original counts (includes cis gene),
        # we need to extract alpha_x for the cis gene and exclude it from modality alpha_y.
        if modality_name == self.model.primary_modality and hasattr(self.model, 'counts') and \
           self.model.cis_gene is not None and 'cis' in self.model.modalities:

            # Enforce: cis effect must NOT be multinomial
            if distribution == 'multinomial':
                raise ValueError(
                    "Primary modality uses a multinomial distribution, "
                    "but cis effect extraction is undefined for multinomial. "
                    "Ensure the primary modality is not multinomial when cis is present."
                )

            # Check if cis gene is in the original counts
            if isinstance(self.model.counts, pd.DataFrame) and self.model.cis_gene in self.model.counts.index:
                all_genes_orig = self.model.counts.index.tolist()
                cis_idx_orig = all_genes_orig.index(self.model.cis_gene)

                # Expect alpha_y_mult and alpha_y_add to be [S?, C, T_all] for non-multinomial primary
                full_alpha_y_mult = posterior_samples["alpha_y_mult"]
                full_alpha_y_add  = posterior_samples["alpha_y_add"]

                if full_alpha_y_mult is None:
                    raise RuntimeError(
                        "alpha_y_mult is None while attempting cis extraction. "
                        "This can happen if the distribution is multinomial; "
                        "check distribution handling earlier."
                    )

                if cis_idx_orig < full_alpha_y_mult.shape[-1]:
                    # Extract cis gene alpha
                    self.model.alpha_x_prefit = full_alpha_y_mult[..., cis_idx_orig]
                    self.model.alpha_x_type = 'posterior'

                    # Exclude cis from modality alpha_y
                    all_idx = list(range(full_alpha_y_mult.shape[-1]))
                    trans_idx = [i for i in all_idx if i != cis_idx_orig]

                    modality.alpha_y_prefit      = full_alpha_y_mult[..., trans_idx]
                    modality.alpha_y_prefit_mult = full_alpha_y_mult[..., trans_idx]
                    modality.alpha_y_prefit_add  = full_alpha_y_add[..., trans_idx]

                    # ========================================
                    # Extract cis gene posterior samples and store in cis modality
                    # ========================================
                    cis_modality = self.model.get_modality('cis')
                    cis_posterior = {}

                    # Extract raw sampled parameters for cis gene
                    if 'log2_alpha_y' in posterior_samples:
                        cis_posterior['log2_alpha_x'] = posterior_samples['log2_alpha_y'][..., cis_idx_orig:cis_idx_orig+1]

                    if 'alpha_y_mul' in posterior_samples:
                        cis_posterior['alpha_x_mul'] = posterior_samples['alpha_y_mul'][..., cis_idx_orig:cis_idx_orig+1]

                    if 'delta_y_add' in posterior_samples:
                        cis_posterior['delta_x_add'] = posterior_samples['delta_y_add'][..., cis_idx_orig:cis_idx_orig+1]

                    if 'o_y' in posterior_samples:
                        cis_posterior['o_x'] = posterior_samples['o_y'][..., cis_idx_orig:cis_idx_orig+1]

                    if 'mu_ntc' in posterior_samples:
                        cis_posterior['mu_ntc'] = posterior_samples['mu_ntc'][..., cis_idx_orig:cis_idx_orig+1]

                    # Create reconstructed alpha with baseline (matching what we do for alpha_y)
                    # These already have shape [S, C-1, 1] from the extraction above
                    if 'alpha_x_mul' in cis_posterior:
                        alpha_x_mul_raw = cis_posterior['alpha_x_mul']  # [S, C-1, 1]
                        # Add baseline (C=1) dimension
                        if alpha_x_mul_raw.dim() == 3:
                            S, Cminus1, _ = alpha_x_mul_raw.shape
                            cis_posterior['alpha_x_mult'] = torch.cat(
                                [torch.ones(S, 1, 1, device=alpha_x_mul_raw.device), alpha_x_mul_raw],
                                dim=1
                            )  # [S, C, 1]
                        cis_posterior['alpha_x'] = cis_posterior['alpha_x_mult']  # alias

                    if 'delta_x_add' in cis_posterior:
                        delta_x_add_raw = cis_posterior['delta_x_add']  # [S, C-1, 1]
                        # Add baseline (C=0) dimension
                        if delta_x_add_raw.dim() == 3:
                            S, Cminus1, _ = delta_x_add_raw.shape
                            cis_posterior['alpha_x_add'] = torch.cat(
                                [torch.zeros(S, 1, 1, device=delta_x_add_raw.device), delta_x_add_raw],
                                dim=1
                            )  # [S, C, 1]

                    # Store in cis modality
                    cis_modality.posterior_samples_technical = cis_posterior
                    print(f"[INFO] Stored cis gene posterior samples in 'cis' modality: {list(cis_posterior.keys())}")

                    # ========================================
                    # Exclude cis gene from ALL primary modality posterior samples
                    # ========================================
                    posterior_samples["alpha_y"]      = modality.alpha_y_prefit
                    posterior_samples["alpha_y_mult"] = modality.alpha_y_prefit_mult
                    posterior_samples["alpha_y_add"]  = modality.alpha_y_prefit_add

                    # CRITICAL: Also exclude cis gene from ALL raw posterior samples
                    # These raw samples are used for analysis/diagnostics, so they must match gene modality
                    for key in ['log2_alpha_y', 'alpha_y_mul', 'delta_y_add', 'o_y', 'mu_ntc']:
                        if key in posterior_samples:
                            raw_sample = posterior_samples[key]
                            # These are typically [S, C-1, T] or [S, 1, T] depending on the parameter
                            if raw_sample is not None and raw_sample.shape[-1] == len(all_genes_orig):
                                posterior_samples[key] = raw_sample[..., trans_idx]
                                print(f"[INFO] Excluded cis gene from '{key}': {raw_sample.shape} -> {posterior_samples[key].shape}")

                    print(f"[INFO] Extracted alpha_x for cis '{self.model.cis_gene}' and excluded it from ALL primary modality posterior samples")
                else:
                    # Cis gene not present after filtering; store as-is
                    modality.alpha_y_prefit      = posterior_samples["alpha_y"]
                    modality.alpha_y_prefit_mult = posterior_samples["alpha_y_mult"]
                    modality.alpha_y_prefit_add  = posterior_samples["alpha_y_add"]
            else:
                # Cis gene not in counts; store as-is
                modality.alpha_y_prefit      = posterior_samples["alpha_y"]
                modality.alpha_y_prefit_mult = posterior_samples["alpha_y_mult"]
                modality.alpha_y_prefit_add  = posterior_samples["alpha_y_add"]
        else:
            # Not primary modality or no cis gene, store as-is
            modality.alpha_y_prefit      = posterior_samples["alpha_y"]
            modality.alpha_y_prefit_mult = posterior_samples["alpha_y_mult"]
            modality.alpha_y_prefit_add  = posterior_samples["alpha_y_add"]

        modality.posterior_samples_technical = posterior_samples

        if modality.is_exon_skipping():
            modality.mark_technical_fit_complete()

        # Store losses for primary modality only
        if modality_name == self.model.primary_modality:
            self.model.loss_technical = losses

        print(f"[INFO] Stored technical fit results in modality '{modality_name}'")
    
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        pyro.clear_param_store()
        import gc; gc.collect()
    
        print("Finished fit_technical.")


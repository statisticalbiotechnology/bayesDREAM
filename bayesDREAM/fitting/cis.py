"""
Cis gene expression fitting for bayesDREAM.

This module contains the cis model and fitting logic.
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



class CisFitter:
    """Handles cis gene expression fitting."""

    def __init__(self, model):
        """
        Initialize cis fitter.

        Parameters
        ----------
        model : _BayesDREAMCore
            The parent model instance
        """
        self.model = model

    def _model_x(
        self,
        N,
        G,
        alpha_dirichlet_tensor,
        guides_tensor,
        x_obs_tensor,
        sum_factor_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        sigma_eff_mean_tensor,
        sigma_eff_sd_tensor,
        epsilon_tensor,
        C=None,
        groups_tensor=None,
        alpha_x_sample=None,
        o_x_sample=None,
        target_per_guide_tensor=None,
        independent_mu_sigma=False,
):

    if groups_tensor is not None:
        cfull_plate = pyro.plate("cfull_plate", C, dim=-2)  # <-- Store it as a variable
        
    ###################################
    ## Technical covariate modelling ##
    ###################################
    if alpha_x_sample is not None:
        alpha_x = alpha_x_sample
    elif groups_tensor is not None:
        with pyro.plate("c_plate", C - 1):
            alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(1 / alpha_alpha_mu_tensor)) # shape = [C-1]
            alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))
            alpha_x = pyro.sample("alpha_x", dist.Gamma(alpha_alpha, alpha_alpha/alpha_mu)) # shape = [C-1]
    else:
        alpha_x = None
    
    ####################
    ## Overdispersion ##
    ####################
    if (o_x_sample is None) and (groups_tensor is not None):
        #with cfull_plate:
        #    beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        #    o_x = pyro.sample("o_x", dist.Exponential(beta_o))
        #    phi_x = 1 / (o_x**2)
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        o_x = pyro.sample("o_x", dist.Exponential(beta_o))
        phi_x = 1 / (o_x**2)
        #phi_x_used = phi_x[..., groups_tensor] # shape => [N]
        phi_x_used = phi_x
    elif (o_x_sample is not None) and (groups_tensor is not None):
        phi_x = 1/(o_x_sample ** 2)
        phi_x_used = phi_x[..., groups_tensor]
    else:
        ####################
        ## Overdispersion ##
        ####################
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        o_x = pyro.sample("o_x", dist.Exponential(beta_o))
        phi_x_used = (1 / (o_x**2))

    ###############################
    ## Mixture model for x_eff_g ##
    ###############################
    if independent_mu_sigma:
        unique_targets = torch.unique(target_per_guide_tensor)
        mu_targets = {}
        sigma_targets = {}
        for t in unique_targets:
            mu_targets[int(t.item())] = pyro.sample(f"mu_target_{int(t.item())}", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
            sigma_targets[int(t.item())] = pyro.sample(f"sigma_target_{int(t.item())}", dist.HalfCauchy(scale=torch.tensor(2.0, device=self.device)))
        # gather mu and sigma for each guide
        mu_target_tensor = torch.stack([mu_targets[int(t.item())] for t in unique_targets], dim=0)
        sigma_target_tensor = torch.stack([sigma_targets[int(t.item())] for t in unique_targets], dim=0)
        mu = mu_target_tensor[target_per_guide_tensor]
        sigma = sigma_target_tensor[target_per_guide_tensor]
    else:
        mu = pyro.sample("mu", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
        sigma = pyro.sample("sigma", dist.HalfCauchy(scale=torch.tensor(2.0, device=self.device)))
        mu = mu.expand(G)
        sigma = sigma.expand(G)

    # Non-centered parameterization
    if (sigma_eff_mean_tensor >= 0.01) and (sigma_eff_sd_tensor >= 0.01):
        rate_alpha = (sigma_eff_sd_tensor ** 2) / (sigma_eff_mean_tensor ** 2)
        rate_beta = (sigma_eff_sd_tensor ** 2) / sigma_eff_mean_tensor
        sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Exponential(rate_alpha))
        sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Exponential(rate_beta))
    else:
        sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Gamma(1.0, 0.01))
        sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Gamma(1.0, 0.01))
    with pyro.plate("guides_plate", G):
        eps_x_eff_g = pyro.sample("eps_x_eff_g", dist.StudentT(df=3.0, loc=0.0, scale=1.0))
        log2_x_eff_g = mu + sigma * eps_x_eff_g
        x_eff_g = pyro.deterministic("x_eff_g", torch.tensor(2.0, device=self.device) ** log2_x_eff_g)
    
        sigma_eff = pyro.sample("sigma_eff", dist.Gamma(sigma_eff_alpha, sigma_eff_beta))
            
    ##########################
    ## Cell-level variables ##
    ##########################
    if alpha_x is not None:
        ones_ = torch.ones(alpha_x.shape[:-1] + (1,), device=self.device)
        alpha_x_full = torch.cat([ones_, alpha_x], dim=-1)  # shape = [C] or [S,C]
        alpha_x_used = alpha_x_full[..., groups_tensor]     # shape = [N] or [S,N]
    else:
        alpha_x_used = torch.ones_like(sum_factor_tensor)  # shape = [N] or [S,N]
    
    x_mean = x_eff_g[..., guides_tensor]  # no alpha_x_used here anymore
    
    ######################
    ## Cell-level plate ##
    ######################
    with pyro.plate("data_plate", N):
        log_x_true = pyro.sample( # use log2 of xtrue to allow small values of xtrue
            "log_x_true",
            dist.Normal(torch.log2(x_mean), sigma_eff[..., guides_tensor])
        )
        x_true = pyro.deterministic("x_true", torch.tensor(2.0, device=self.device) ** log_x_true)
        mu_obs = alpha_x_used * x_true * sum_factor_tensor
        pyro.sample(
            "x_obs",
            dist.NegativeBinomial(total_count=phi_x_used,
                                  logits=torch.log(mu_obs) - torch.log(phi_x_used),
                                  validate_args=False
                                 ), # important to use logits to allow small x_true
            obs=x_obs_tensor
        )


    def fit_cis(
        self,
        technical_covariates: list[str] = None,
        sum_factor_col: str = 'sum_factor',
        modality_name: str = None,
        cis_feature: str = None,
        manual_guide_effects: pd.DataFrame = None,
        prior_strength: float = 1.0,
        lr: float = 1e-3,
        niters: int = 50000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        epsilon: float = 1e-6,
        alpha_dirichlet: float = 0.1,
        minibatch_size: int = None,
        independent_mu_sigma: bool = False,   # <--- NEW
        **kwargs
):
    """
    Fits the cis effects (model_x) for your gene_of_interest.
    This step can be repeated multiple times with different priors
    or hyperparameters.

    Parameters
    ----------
    technical_covariates : list, optional
        Technical covariates for correction
    sum_factor_col : str
        Column name for size factors
    modality_name : str, optional
        DEPRECATED: fit_cis always uses the primary modality.
        This parameter is ignored.
    cis_feature : str, optional
        Feature ID to use as cis proxy from the primary modality.
        If None, uses self.cis_gene (must exist in primary modality).
        For ATAC: region ID (e.g., 'chr9:132283881-132284881')
        For genes: gene name
    manual_guide_effects : pd.DataFrame, optional
        Manual guide effect estimates as priors. DataFrame with columns:
        - guide: guide identifier (matches meta['guide'])
        - log2FC: expected log2 fold-change vs NTC
    prior_strength : float
        Weight for manual guide effects (default: 1.0)
        0 = ignore manual effects, higher = trust more
    lr : float
        Learning rate for Adam
    niters : int
        Number of SVI iterations
    nsamples : int
        Number of posterior samples
    alpha_ewma : float
        Exponential weight for smoothing the ELBO
    tolerance : float
        Convergence tolerance
    independent_mu_sigma : bool
        Whether to use independent mu/sigma per target type
    kwargs :
        Additional arguments controlling priors, etc.
    """
    print("Running fit_cis...")

    if self.cis_gene is None:
        raise ValueError("self.cis_gene must be set.")

    # IMPORTANT: fit_cis ALWAYS uses the PRIMARY modality
    # This ensures cis_gene is always in the modality being modeled
    if modality_name is not None:
        warnings.warn(
            "modality_name parameter is deprecated. fit_cis always uses the primary modality. "
            f"Ignoring modality_name='{modality_name}' and using primary_modality='{self.primary_modality}'",
            DeprecationWarning
        )

    # Get primary modality
    modality = self.get_modality(self.primary_modality)

    # Determine which feature to use as cis proxy
    if cis_feature is None:
        # Try to find cis feature from cis_feature_map (set by add_atac_modality)
        if hasattr(self, 'cis_feature_map') and self.primary_modality in self.cis_feature_map:
            cis_feature = self.cis_feature_map[self.primary_modality]
            print(f"[INFO] Using stored cis feature: {cis_feature}")
        else:
            # Use self.cis_gene (must exist in primary modality)
            cis_feature = self.cis_gene
            if cis_feature not in modality.feature_meta.index:
                raise ValueError(
                    f"cis_gene '{self.cis_gene}' not found in primary modality '{self.primary_modality}'.\n"
                    f"The cis_gene must exist in the primary modality.\n"
                    f"Available features: {modality.feature_meta.index[:10].tolist()}...\n"
                    f"Either:\n"
                    f"  1. Specify cis_feature parameter with a valid feature ID from the primary modality, OR\n"
                    f"  2. Use add_atac_modality(..., cis_region='...') to set the cis feature automatically"
                )
            print(f"[INFO] Using cis_gene '{self.cis_gene}' from primary modality '{self.primary_modality}'")
    else:
        # User provided explicit cis_feature - validate it exists in primary modality
        if cis_feature not in modality.feature_meta.index:
            raise ValueError(
                f"cis_feature '{cis_feature}' not found in primary modality '{self.primary_modality}'.\n"
                f"Available features: {modality.feature_meta.index[:10].tolist()}..."
            )
        print(f"[INFO] Using cis_feature '{cis_feature}' from primary modality '{self.primary_modality}'")

    # Get counts for this feature from the primary modality
    if isinstance(modality.counts, pd.DataFrame):
        cis_counts = modality.counts.loc[cis_feature].values
    else:
        # numpy array - need to find index
        feature_idx = modality.feature_meta.index.get_loc(cis_feature)
        if modality.cells_axis == 1:
            cis_counts = modality.counts[feature_idx, :]
        else:
            cis_counts = modality.counts[:, feature_idx]

    # convert to gpu for fitting
    if self.alpha_x_prefit is not None and self.alpha_x_prefit.device != self.device:
        self.alpha_x_prefit = self.alpha_x_prefit.to(self.device)

    if technical_covariates:
        if "technical_group_code" in self.meta.columns:
            warnings.warn("technical_group already set. Overwriting.")
            if self.alpha_x_prefit is not None:
                warnings.warn("Overwriting alpha_x prefit, and refitting.")
                self.alpha_x_prefit = None

        self.meta["technical_group_code"] = self.meta.groupby(technical_covariates).ngroup()
        C = self.meta['technical_group_code'].nunique()
        groups_tensor = torch.tensor(self.meta['technical_group_code'].values, dtype=torch.long, device=self.device)

        # Check if alpha_x_prefit should exist but doesn't
        if self.alpha_x_prefit is None:
            warnings.warn(
                f"Technical covariates provided but alpha_x_prefit not set. "
                f"You should run fit_technical() on the primary modality ('{self.primary_modality}') first "
                f"to estimate technical effects for the cis gene. "
                f"Proceeding without technical correction for cis gene (alpha_x will be fitted fresh)."
            )

    elif self.alpha_x_prefit is None:
        C = None
        groups_tensor = None
        warnings.warn("no alpha_x_prefit and no technical_covariates provided, assuming no confounding effect.")
    else:
        # alpha_x_prefit exists but no new technical_covariates specified
        # Use existing technical groups
        C = self.meta['technical_group_code'].nunique()
        groups_tensor = torch.tensor(self.meta['technical_group_code'].values, dtype=torch.long, device=self.device)
    
    N = self.meta.shape[0]
    G = self.meta['guide_code'].nunique()

    # Use cis_counts from modality-specific lookup (or traditional self.counts)
    x_obs_tensor = torch.tensor(cis_counts, dtype=torch.float32, device=self.device)
    guides_tensor = torch.tensor(self.meta['guide_code'].values, dtype=torch.long, device=self.device)

    # ========================================================================
    # MANUAL GUIDE EFFECTS INFRASTRUCTURE
    # ========================================================================
    # If user provides manual guide effects (log2FC estimates), prepare them
    # as tensors that can be used as priors in the Pyro model.
    manual_guide_log2fc_tensor = None
    manual_guide_mask_tensor = None

    if manual_guide_effects is not None:
        # Validate format
        required_cols = ['guide', 'log2FC']
        missing_cols = set(required_cols) - set(manual_guide_effects.columns)
        if missing_cols:
            raise ValueError(f"manual_guide_effects must have columns {required_cols}, missing: {missing_cols}")

        # Create mapping from guide name → log2FC
        manual_dict = dict(zip(manual_guide_effects['guide'], manual_guide_effects['log2FC']))

        # Map to guide_code indices
        # self.meta has 'guide' (name) and 'guide_code' (integer)
        guide_code_to_name = self.meta[['guide_code', 'guide']].drop_duplicates().set_index('guide_code')['guide'].to_dict()

        # Create tensor of shape (G,) with log2FC for each guide_code
        # Use NaN for guides without manual effects
        manual_log2fc_list = []
        manual_mask_list = []
        for g in range(G):
            guide_name = guide_code_to_name.get(g, None)
            if guide_name in manual_dict:
                manual_log2fc_list.append(manual_dict[guide_name])
                manual_mask_list.append(1.0)  # This guide has a manual prior
            else:
                manual_log2fc_list.append(0.0)  # Placeholder
                manual_mask_list.append(0.0)  # No prior for this guide

        manual_guide_log2fc_tensor = torch.tensor(manual_log2fc_list, dtype=torch.float32, device=self.device)
        manual_guide_mask_tensor = torch.tensor(manual_mask_list, dtype=torch.float32, device=self.device)

        print(f"[INFO] Manual guide effects provided for {int(manual_guide_mask_tensor.sum())} / {G} guides")
        print(f"[INFO] Prior strength: {prior_strength}")

        # PSEUDOCODE: How these tensors would be used in _model_x:
        # ------------------------------------------------------------------
        # In _model_x, when sampling mu_x (guide-level mean log2 expression):
        #
        # for g in range(G):
        #     if manual_guide_mask_tensor[g] == 1.0:
        #         # This guide has a manual prior
        #         # Sample from a Gaussian centered on the manual log2FC
        #         # with width controlled by prior_strength
        #         prior_mean = manual_guide_log2fc_tensor[g]
        #         prior_sd = 1.0 / prior_strength  # Higher strength → narrower prior
        #         mu_x[g] = pyro.sample(f"mu_x_{g}",
        #                               dist.Normal(prior_mean, prior_sd))
        #     else:
        #         # No manual prior - use standard hierarchical prior
        #         mu_x[g] = pyro.sample(f"mu_x_{g}",
        #                               dist.Normal(mu, sigma))
        #
        # DECISIONS TO MAKE:
        # 1. Should prior_sd = 1.0 / prior_strength, or some other function?
        # 2. Should manual effects override hierarchical priors completely,
        #    or combine them (e.g., weighted average)?
        # 3. Should NTC guide always have log2FC=0 enforced, or learned?
        # 4. How to handle cell-line-specific effects with manual priors?
        # ------------------------------------------------------------------
    # ========================================================================
    if independent_mu_sigma:
        if ('target' not in self.meta.columns):
            raise ValueError("independent_mu_sigma is True, self.meta['target'] column not found.")
        elif self.meta['target'].nunique() < 2:
            raise ValueError("independent_mu_sigma is True, but only 1 target type found in self.meta['target'] column.")
        self.meta['target_code'] = pd.factorize(self.meta['target'])[0]
        target_codes_tensor = torch.tensor(self.meta['target_code'].values, dtype=torch.long, device=self.device)
    
        ### BUILD target_per_guide_tensor [G] based on guide → target
        target_per_guide_tensor = torch.empty(G, dtype=torch.long, device=self.device)
        for g in range(G):
            idx = (guides_tensor == g)
            guide_targets = torch.unique(target_codes_tensor[idx])
            if guide_targets.shape[0] != 1:
                raise ValueError(f"Guide {g} maps to multiple targets: {guide_targets}")
            target_per_guide_tensor[g] = guide_targets[0]
    else:
        target_per_guide_tensor = None
    sum_factor_tensor = torch.tensor(self.meta[sum_factor_col].values, dtype=torch.float32, device=self.device)

    # Compute guide means (adjusting for alpha_x if provided)
    x_obs_factored = x_obs_tensor / sum_factor_tensor
    
    if self.alpha_x_prefit is not None:
        if self.alpha_x_type == 'posterior':
            # Take mean over posterior samples (S, C-1) -> (C-1)
            alpha_x_prefit_mean = self.alpha_x_prefit.mean(dim=0)
        else:
            # If it's a point estimate, ensure correct shape (C-1)
            alpha_x_prefit_mean = self.alpha_x_prefit.reshape((C-1,))
    
        # Ensure correct shape for alpha_x_full: (C, 1)
        alpha_x_full = torch.cat([torch.ones((1,), device=self.device), alpha_x_prefit_mean], dim=0)  # Shape: (2,1)
    
        # Select the correct alpha_x for each observation (expand for broadcasting)
        alpha_x_used = alpha_x_full[groups_tensor]  # groups_tensor indexes into (2,1)
        
        # Adjust x_obs_factored by alpha_x_used
        x_obs_factored /= alpha_x_used  # Ensure correct shape for division
    
    # Continue with the rest of the setup
    beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.device)
    beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.device)
    alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.device)
    alpha_dirichlet_tensor = torch.tensor(alpha_dirichlet, dtype=torch.float32, device=self.device)
    epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

    unique_guides = torch.unique(guides_tensor)
    guide_means = torch.tensor([
        torch.log2(torch.mean(x_obs_factored[guides_tensor == g]))
        for g in unique_guides
        if torch.sum(x_obs_factored[guides_tensor == g]) > 0
    ], dtype=torch.float32, device=self.device)
    print(f"[DEBUG] guide_means min={guide_means.min()} median={guide_means.median()} mean={guide_means.mean()} max=={guide_means.max()}")
    mu_x_mean_tensor = torch.mean(guide_means)#.to(self.device)
    print(f"[DEBUG] mu_x_mean_tensor: {mu_x_mean_tensor}")
    mu_x_sd_tensor = torch.std(guide_means)#.to(self.device)
    # Compute guide-level MAD (robust estimate of log2 spread)
    guide_mads_tensor = torch.tensor([
        torch.median(torch.abs(torch.log2((x_obs_factored)[guides_tensor == g] + epsilon) - 
                               torch.median(torch.log2((x_obs_factored)[guides_tensor == g] + epsilon))))
        for g in unique_guides
    ], dtype=torch.float32, device=self.device)
    guide_mads_tensor = guide_mads_tensor * 1.4826  # Gaussian-equivalent spread
    sigma_eff_mean_tensor = torch.mean(guide_mads_tensor)#.to(self.device)
    sigma_eff_sd_tensor = torch.std(guide_mads_tensor)#.to(self.device)        
    mu_x_alpha_tensor = mu_x_mean_tensor ** 2 / (mu_x_sd_tensor ** 2)
    mu_x_beta_tensor = mu_x_mean_tensor / (mu_x_sd_tensor ** 2)

    def init_loc_fn(site):
        name = site["name"]
    
        if name == "mu":
            return mu_x_mean_tensor
        elif name == "sigma":
            return mu_x_sd_tensor.clamp(min=1e-3)
    
        elif name == "sigma_eff_alpha":
            # Corresponds to (mean^2 / var)
            sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
            sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
            return ((sigma_eff_mean_tensor_tmp ** 2) / (sigma_eff_sd_tensor_tmp ** 2))
        elif name == "sigma_eff_beta":
            # Corresponds to (mean / var)
            sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
            sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
            return (sigma_eff_mean_tensor_tmp / (sigma_eff_sd_tensor_tmp ** 2))
    
        elif name == "sigma_eff":
            # Assume you’re sampling per-guide (e.g. G = 37)
            return sigma_eff_mean_tensor.clamp(min=1e-2).expand(G)
    
        return pyro.infer.autoguide.initialization.init_to_median(site)
    
    guide_x = pyro.infer.autoguide.AutoNormalMessenger(self._model_x, init_loc_fn=init_loc_fn)
    optimizer = pyro.optim.Adam({"lr": lr})
    svi = pyro.infer.SVI(self._model_x, guide_x, optimizer, 
                         loss=pyro.infer.Trace_ELBO())

    losses = []
    smoothed_loss = None
    for step in range(niters):
        
        if self.alpha_x_prefit is not None:
            samp = torch.randint(high=self.alpha_x_prefit.shape[0], size=(1,)).item()
        alpha_x_sample = (
            self.alpha_x_prefit[samp] if self.alpha_x_type == "posterior"
            else self.alpha_x_prefit
        ) if self.alpha_x_prefit is not None else None
        o_x_sample = None
        
        loss = svi.step(
            N,
            G,
            alpha_dirichlet_tensor,
            guides_tensor,
            x_obs_tensor,
            sum_factor_tensor,
            beta_o_alpha_tensor,
            beta_o_beta_tensor,
            alpha_alpha_mu_tensor,
            mu_x_mean_tensor,
            mu_x_sd_tensor,
            sigma_eff_mean_tensor.clamp(min=1e-2),
            sigma_eff_sd_tensor.clamp(min=1e-2),
            epsilon_tensor,
            C = C,
            groups_tensor=groups_tensor,
            alpha_x_sample=alpha_x_sample,
            o_x_sample=o_x_sample,
            target_per_guide_tensor=target_per_guide_tensor,
            independent_mu_sigma=independent_mu_sigma,
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
    run_on_cpu = self.device.type != "cpu"

    if run_on_cpu:
        print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
        guide_x.to("cpu")
        self.device = torch.device("cpu")

        model_inputs = {
            "N": N,
            "G": G,
            "alpha_dirichlet_tensor": alpha_dirichlet_tensor.cpu(),
            "guides_tensor": guides_tensor.cpu(),
            "x_obs_tensor": x_obs_tensor.cpu(),
            "sum_factor_tensor": sum_factor_tensor.cpu(),
            "beta_o_alpha_tensor": beta_o_alpha_tensor.cpu(),
            "beta_o_beta_tensor": beta_o_beta_tensor.cpu(),
            "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor.cpu(),
            "mu_x_mean_tensor": mu_x_mean_tensor.cpu(),
            "mu_x_sd_tensor": mu_x_sd_tensor.cpu(),
            "sigma_eff_mean_tensor": sigma_eff_mean_tensor.clamp(min=1e-2).cpu(),
            "sigma_eff_sd_tensor": sigma_eff_sd_tensor.clamp(min=1e-2).cpu(),
            "epsilon_tensor": epsilon_tensor.cpu(),
            "C": C,
            "groups_tensor": groups_tensor.cpu() if groups_tensor is not None else None,
            "alpha_x_sample": self.alpha_x_prefit.mean(dim=0).cpu() if self.alpha_x_type == "posterior" else (self.alpha_x_prefit.cpu() if self.alpha_x_prefit is not None else None),
            "o_x_sample": None,
            "target_per_guide_tensor": target_per_guide_tensor.cpu() if target_per_guide_tensor is not None else None,
            "independent_mu_sigma": independent_mu_sigma,
        }
    else:
        model_inputs = {
            "N": N,
            "G": G,
            "alpha_dirichlet_tensor": alpha_dirichlet_tensor,
            "guides_tensor": guides_tensor,
            "x_obs_tensor": x_obs_tensor,
            "sum_factor_tensor": sum_factor_tensor,
            "beta_o_alpha_tensor": beta_o_alpha_tensor,
            "beta_o_beta_tensor": beta_o_beta_tensor,
            "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
            "mu_x_mean_tensor": mu_x_mean_tensor,
            "mu_x_sd_tensor": mu_x_sd_tensor,
            "sigma_eff_mean_tensor": sigma_eff_mean_tensor.clamp(min=1e-2),
            "sigma_eff_sd_tensor": sigma_eff_sd_tensor.clamp(min=1e-2),
            "epsilon_tensor": epsilon_tensor,
            "C": C,
            "groups_tensor": groups_tensor,
            "alpha_x_sample": self.alpha_x_prefit.mean(dim=0) if self.alpha_x_type == "posterior" else (self.alpha_x_prefit if self.alpha_x_prefit is not None else None),
            "o_x_sample": None,
            "target_per_guide_tensor": target_per_guide_tensor if target_per_guide_tensor is not None else None,
            "independent_mu_sigma": independent_mu_sigma,
        }

    if self.device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()

    max_samples = nsamples
    keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "x_obs")

    if minibatch_size is not None:
        from collections import defaultdict
        print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
        predictive_x = pyro.infer.Predictive(
            self._model_x,
            guide=guide_x,
            num_samples=minibatch_size,
            parallel=True
        )
        all_samples = defaultdict(list)
        with torch.no_grad():
            for i in range(0, max_samples, minibatch_size):
                samples = predictive_x(**model_inputs)
                for k, v in samples.items():
                    if keep_sites(k, {"value": v}):
                        all_samples[k].append(v.cpu())
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
        posterior_samples_x = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}
    else:
        predictive_x = pyro.infer.Predictive(
            self._model_x,
            guide=guide_x,
            num_samples=nsamples#,
            #parallel=True
        )
        with torch.no_grad():
            posterior_samples_x = predictive_x(**model_inputs)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    if run_on_cpu:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[INFO] Reset self.device to:", self.device)

    for k, v in posterior_samples_x.items():
        posterior_samples_x[k] = v.cpu()
    
    self.loss_x = losses
    self.posterior_samples_cis = posterior_samples_x
    self.x_true = posterior_samples_x['x_true']
    self.x_true_type = 'posterior'
    self.log2_x_true = posterior_samples_x['log_x_true']
    self.log2_x_true_type = 'posterior'
    if self.alpha_x_prefit is None and technical_covariates:
        self.alpha_x_prefit = posterior_samples_x["alpha_x"]
        self.alpha_x_type = 'posterior'

    if self.device.type == "cuda":
        torch.cuda.empty_cache()
    import gc
    gc.collect()
    pyro.clear_param_store()

    print("Finished fit_cis.")


    def refit_sumfactor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_refit: str = "sum_factor_new",
        covariates: list[str] = None, # ["lane", "cell_line"] or could be empty
        n_knots: int = 5,
        degree: int = 3,
        alpha: float = 0.1
):
    """
    Step 2 of sum factor adjustment: Remove cis gene contribution.

    Use AFTER fit_cis() and BEFORE fit_trans().
    Fits a spline regression: (sum_factor - baseline_ntc) ~ f(x_true)
    Then removes the predicted x_true contribution from sum factors.

    This ensures trans modeling isn't confounded by cis expression levels.

    Typical workflow:
        1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
        2. fit_cis(sum_factor_col='sum_factor_adj')
        3. refit_sumfactor() -> creates 'sum_factor_refit'  <-- This step
        4. fit_trans(sum_factor_col='sum_factor_refit')

    Parameters
    ----------
    sum_factor_col_old : str
        Name of existing sum factor column (typically from adjust_ntc_sum_factor)
    sum_factor_col_refit : str
        Name for refitted sum factor column to create (default: 'sum_factor_new')
    covariates : list of str, optional
        Columns to group by for baseline NTC calculation (e.g., ['cell_line', 'lane'])
    n_knots : int
        Number of spline knots for regression (default: 5)
    degree : int
        Polynomial degree of spline pieces (default: 3)
    alpha : float
        Ridge regression regularization parameter (default: 0.1)

    Returns
    -------
    None
        Creates new column sum_factor_col_refit in self.meta

    Notes
    -----
    Algorithm:
    1. Compute baseline NTC sum factor for each covariate group
    2. Compute leftover = sum_factor - baseline_ntc
    3. Fit spline model: leftover ~ f(x_true)
    4. Predict x_true contribution: y_pred = model(x_true)
    5. Adjusted sum factor = max(0, leftover - y_pred + baseline_ntc)

    Requires:
    - self.x_true must be set (from fit_cis())
    - self.x_true_type must be 'posterior' or 'point'
    """
    sum_factor_data = self.meta[sum_factor_col_old].values  # shape (N,)
    
    if covariates is None:
        covariates = []

    if covariates:
        # Create a single group identifier by concatenating covariate values
        tech_group = self.meta[covariates].astype(str).agg('_'.join, axis=1)
        groups, group_id = np.unique(tech_group, return_inverse=True)
        n_groups = len(groups)
    else:
        # No grouping, treat all samples as a single group
        groups, group_id = np.array(["all"]), np.zeros(len(self.meta), dtype=int)
        n_groups = 1
    
    baseline_ntc_of_group = np.zeros(n_groups)
    for grp, grp_name in enumerate(groups):
        mask_grp = (tech_group == grp_name)
        # Among that group, pick rows with gene == 'ntc'
        mask_ntc = (self.meta['target'] == 'ntc') & mask_grp
    
        # If a group has no NTC, you must decide on a fallback
        # For example, use the overall mean or 1.0, or skip that group
        if not np.any(mask_ntc):
            baseline_ntc_of_group[grp] = 1.0  # fallback
        else:
            # The mean of sum_factor_data among the NTC rows
            baseline_ntc_of_group[grp] = np.mean(sum_factor_data[mask_ntc])
    
    # leftover_data[i] = sum_factor[i] - baseline_ntc_of_group[group_id[i]]
    leftover_data = sum_factor_data - baseline_ntc_of_group[group_id]

    # ------------------------------------------------------------------------------
    # Build a pipeline with a Spline transformer + Linear regression
    # ------------------------------------------------------------------------------
    # 'n_knots' is how many spline knots to use;
    # 'degree' is the polynomial degree of each spline piece.
    model_spline_ridge = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=degree),
        Ridge(alpha=alpha)  # alpha > 0 adds penalty
    )
    
    # ------------------------------------------------------------------------------
    # Fit the model on training data
    # ------------------------------------------------------------------------------
    if self.x_true_type == 'posterior':
        X_true = self.x_true.mean(dim=0)
    else:
        X_true = self.x_true
    model_spline_ridge.fit(X_true.reshape(-1, 1), leftover_data)
    
    # ------------------------------------------------------------------------------
    # Predict on train & test
    # ------------------------------------------------------------------------------
    y_pred = model_spline_ridge.predict(X_true.reshape(-1, 1))
    self.meta[sum_factor_col_refit] = np.max(np.vstack((leftover_data - y_pred + baseline_ntc_of_group[group_id], np.zeros_like(leftover_data))), axis=0)
    print(f"[INFO] Created '{sum_factor_col_refit}' in meta with xtrue-based adjustment.")

#########################################
## Step 3: Fit trans effects (model_y) ##


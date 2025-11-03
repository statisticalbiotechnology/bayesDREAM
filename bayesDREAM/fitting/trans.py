"""
Trans effects fitting for bayesDREAM.

This module contains the trans model and fitting logic.
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

from ..utils import find_beta, Hill_based_positive, Polynomial_function, check_tensor



class TransFitter:
    """Handles trans effects fitting."""

    def __init__(self, model):
        """
        Initialize trans fitter.

        Parameters
        ----------
        model : _BayesDREAMCore
            The parent model instance
        """
        self.model = model

    def _t(self, x, dtype=torch.float32):
        return torch.as_tensor(x, dtype=dtype, device=self.model.device)

    def _to_cpu(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return x

    #########################################
    ## Step 3: Fit trans effects (model_y) ##
    #########################################
    def _model_y(
        self,
        N,
        T,
        y_obs_tensor,
        sum_factor_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        K_max_tensor,
        K_alpha_tensor,
        Vmax_mean_tensor,
        Vmax_alpha_tensor,
        n_mu_tensor,
        n_sigma_base_tensor,
        Amean_tensor,
        p_n_tensor,
        threshold,
        slope,
        epsilon_tensor,
        x_true_sample,
        log2_x_true_sample,
        nmin,
        nmax,
        alpha_y_sample=None,
        C=None,
        groups_tensor=None,
        predictive_mode=False,
        temperature=1.0,
        use_straight_through=False,
        function_type='single_hill',
        polynomial_degree=6,
        use_alpha=True,
        distribution='negbinom',
        denominator_tensor=None,
        K=None,
        D=None,
    ):

        if use_alpha:
            alpha_dist = dist.RelaxedBernoulliStraightThrough if use_straight_through else dist.RelaxedBernoulli

        if groups_tensor is not None:
            cfull_plate = pyro.plate("cfull_plate", C, dim=-2)  # <-- Store it as a variable
        trans_plate = pyro.plate("trans_plate", T, dim=-1)
        
        ##########
        ## x_true (Now Fully Observed) ##
        ##########
        #x_true = pyro.deterministic("x_true", x_true_sample)
        x_true = x_true_sample
    
        ####################
        ## alpha_y ##
        ####################
        alpha_y = None
        if alpha_y_sample is not None:
            alpha_y = alpha_y_sample
        elif groups_tensor is not None:
            with pyro.plate("cell_lines_plate", C-1, dim=-2):  # **Now correctly uses C-1**
                with trans_plate:
                    alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(1 / alpha_alpha_mu_tensor))  # shape = [C-1, T]
                    alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))  # shape = [C-1, T]
                    alpha_y = pyro.sample("alpha_y", dist.Gamma(alpha_alpha, alpha_alpha / alpha_mu))  # shape = [C-1, T]
    
        ####################
        ## Overdispersion ##
        ####################
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        with trans_plate:
            o_y = pyro.sample("o_y", dist.Exponential(beta_o))
            phi_y = 1 / (o_y**2)
        phi_y_used = phi_y.unsqueeze(-2)
    
        #################
        ## Hill-based: ##
        #################
        if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
            # ----------------------------------------------------
            # 1) Define global hyperparameters for n_a
            # ----------------------------------------------------
            sigma_n_a = pyro.sample("sigma_n_a", dist.Exponential(self._t(1/5))) #   -> controls how variable n_a can be across genes
            if function_type in ['additive_hill', 'nested_hill']:
                sigma_n_b = pyro.sample("sigma_n_b", dist.Exponential(self._t(1/5))) #   -> controls how variable n_a can be across genes
        if function_type in ['polynomial']:
            #sigma_coeff = pyro.sample("sigma_coeff", dist.Exponential(100)) #   -> controls how variable n_a can be across genes
            sigma_coeff = pyro.sample("sigma_coeff", dist.HalfCauchy(scale=self._t(1.0)))
        
        # Now enter the trans_plate (T dimension)
        with trans_plate:
            
            weight = o_y / (o_y + (beta_o_beta_tensor / beta_o_alpha_tensor)).clamp_min(epsilon_tensor)
            Amean_adjusted = ((1 - weight) * Amean_tensor) + (weight * Vmax_mean_tensor) + epsilon_tensor
            A = pyro.sample("A", dist.Exponential(1 / Amean_adjusted))

            if use_alpha:
                # Relaxed Bernoulli: alpha ~ (0,1), becomes more discrete as temperature -> 0
                alpha = pyro.sample("alpha", alpha_dist(temperature=temperature, probs=p_n_tensor))
            else:
                alpha = torch.ones((T,), device=self.model.device)
            
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                
                #####################################
                ## function priors (depend on o_y) ##
                #####################################
                # Gamma and delta depend on T dimension
                # Reduce over group dimension if necessary
                K_sigma = (K_max_tensor / (self._t(2) * torch.sqrt(K_alpha_tensor))) + epsilon_tensor
                Vmax_sigma = (Vmax_mean_tensor / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor
                # Replace Laplace with SoftLaplace
                # Instead of directly n_a, we do n_a_raw
                n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_tensor, sigma_n_a))
                #n_a = pyro.sample("n_a", dist.Normal(n_mu_tensor, sigma_n))
                # Make sure nmin/nmax are scalar tensors; then get python floats for clamp
                BOX_LOW  = self._t(-20.0)
                BOX_HIGH = self._t( 20.0)
                low  = torch.maximum(nmin, BOX_LOW)
                high = torch.minimum(nmax, BOX_HIGH)
                
                n_a = pyro.deterministic(
                    "n_a",
                    (alpha * n_a_raw).clamp(min=low.item(), max=high.item())
                )
                
                # Scale for Vmax, K is multiplied by alpha
                #eff_Vmax_sigma = alpha * Vmaxa_sigma + epsilon_tensor
                #eff_Ka_sigma   = alpha * Ka_sigma    + epsilon_tensor
        
                Vmax_a = pyro.sample("Vmax_a", dist.Gamma((Vmax_mean_tensor ** 2) / (Vmax_sigma ** 2), Vmax_mean_tensor / (Vmax_sigma ** 2)))
                K_a = pyro.sample("K_a", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))
                # Sample all required parameters (same for all hill types)            
                if function_type in ['additive_hill', 'nested_hill']:
                    beta = pyro.sample("beta", alpha_dist(temperature=temperature, probs=p_n_tensor))
                    n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_tensor, sigma_n_b))
                    n_b = pyro.deterministic("n_b", (beta * n_b_raw).clamp(min=-20, max=20)) # clamp for numerical stability
                    Vmax_b = pyro.sample("Vmax_b", dist.Gamma((Vmax_mean_tensor ** 2) / (Vmax_sigma ** 2), Vmax_mean_tensor / (Vmax_sigma ** 2)))
                    K_b = pyro.sample("K_b", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))
                
                # Compute Hill function(s)
                if function_type == 'single_hill':
                    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                    y_true_mu = A + (alpha * Hilla)
                elif function_type == 'additive_hill':
                    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                    Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                    y_true_mu = A + (alpha * Hilla) + (beta * Hillb)
                elif function_type == 'nested_hill':
                    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                    Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                    y_true_mu = A + (alpha * Hillb)
                log_y_true_mu = torch.log(y_true_mu)

            elif function_type == 'polynomial':
                assert polynomial_degree is not None and polynomial_degree >= 1, \
                    "polynomial_degree must be ≥ 1 (no intercept, A is handled separately)"
            
                # 1) take log2 of inputs
                log2_x_true = torch.log2(x_true_sample)  # shape [N]
                
                # Stack polynomial coefficients
                coeffs = []
                for d in range(1, polynomial_degree + 1):  # start at degree 1, no intercept
                    coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))
                    #check_tensor(f"coeff_{d}", coeff)
                    coeffs.append(coeff)
                coeffs = torch.stack(coeffs, dim=-2)  # [degree, T]
                if (coeffs.shape[1] == 1) & (coeffs.ndim == 4):
                    coeffs = coeffs.squeeze(1)        # [S, D, T]
                #check_tensor(f"coeffs", coeffs)
                
                # 4) compute polynomial value exactly as before
                poly_val = Polynomial_function(log2_x_true, coeffs)  # [N, T]
                log2_y_true_mu = torch.log2(A) + alpha * poly_val       # [N, T]
                #log2_y_true_mu = torch.log2(A) + poly_val       # [N, T]
                log_y_true_mu  = log2_y_true_mu * torch.log(torch.tensor(2.0, device=self.model.device))  # Convert log2 to ln
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
            
            
            ##########################
            ## Cell-level variables ##
            ##########################
            if alpha_y is not None:   
                # If in sampling mode, alpha_y will be [S, C-1, T]
                if alpha_y.dim() == 3:  # Predictive case: shape is (S, C-1, T)
                    ones_shape = (alpha_y.shape[0], 1, T)  # Match batch dimension S
                    # Ensure ones tensor has correct batch shape
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.model.device), alpha_y], dim=1)
                elif alpha_y.dim() == 2:  # Training case: shape is (C-1, T)
                    ones_shape = (1, T)
                    # Ensure ones tensor has correct batch shape
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.model.device), alpha_y], dim=0)
                else:  # Unexpected extra dimension
                    raise ValueError(f"[ERROR] Unexpected alpha_y shape: {alpha_y.shape}")
                alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
                log_y_true = log_y_true_mu + torch.log(alpha_y_used) #+ 1e-10
            else:
                log_y_true = log_y_true_mu #+ 1e-10
    
            # Now we sample y_obs which is NxT (or NxTxK for multinomial, NxTxD for mvnormal)
            # Use data_plate for N dimension

            # Compute mu_y from log_y_true (this is the dose-response function output)
            mu_y = torch.exp(log_y_true)  # [N, T]

            # Debug checks (keep for troubleshooting)
            if torch.isnan(log_y_true).any() or torch.isinf(log_y_true).any():
                check_tensor("log_y_true", log_y_true)
                check_tensor("y_true_mu", y_true_mu)
                check_tensor("sum_factor_tensor", sum_factor_tensor)
                check_tensor("phi_y_used", phi_y_used)
                check_tensor("A", A)
                if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                    check_tensor("n_a", n_a)
                    print(f"used nmin={low}, used nmax={high}")
                    check_tensor("Vmax_a", Vmax_a)
                    check_tensor("K_a", K_a)
                if function_type in ['additive_hill', 'nested_hill']:
                    check_tensor("n_b", n_b)
                    check_tensor("Vmax_b", Vmax_b)
                    check_tensor("K_b", K_b)
                if function_type == 'polynomial':
                    check_tensor("coeffs", coeffs)

            # Call distribution-specific observation sampler
            from ..distributions import get_observation_sampler
            observation_sampler = get_observation_sampler(distribution, 'trans')

            # Prepare alpha_y_full (full C cell lines, including reference)
            if alpha_y is not None and groups_tensor is not None:
                if alpha_y.dim() == 3:  # Predictive: (S, C-1, T)
                    ones_shape = (alpha_y.shape[0], 1, T)
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.model.device), alpha_y], dim=1)
                elif alpha_y.dim() == 2:  # Training: (C-1, T)
                    ones_shape = (1, T)
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.model.device), alpha_y], dim=0)
                else:
                    raise ValueError(f"Unexpected alpha_y shape: {alpha_y.shape}")
            else:
                alpha_y_full = None

            # Call the appropriate sampler based on distribution
            if distribution == 'negbinom':
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,
                    phi_y_used=phi_y_used,
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    sum_factor_tensor=sum_factor_tensor,
                    N=N,
                    T=T,
                    C=C
                )
            elif distribution == 'multinomial':
                # For multinomial, mu_y should be probabilities [N, T, K]
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,  # Should be [N, T, K] probabilities
                    N=N,
                    T=T,
                    K=K
                )
            elif distribution == 'binomial':
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    denominator_tensor=denominator_tensor,
                    mu_y=mu_y,  # Should be probabilities [N, T]
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    N=N,
                    T=T,
                    C=C
                )
            elif distribution == 'normal':
                # For normal, we need sigma_y (standard deviation)
                sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,
                    sigma_y=sigma_y,
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    N=N,
                    T=T,
                    C=C
                )
            elif distribution == 'mvnormal':
                # For mvnormal, mu_y should be [N, T, D] and we need covariance
                # Use phi_y to construct covariance (for now, diagonal)
                sigma_y = 1.0 / torch.sqrt(phi_y)  # [T]
                cov_y = torch.diag_embed(sigma_y.unsqueeze(-1).expand(T, D))  # [T, D, D]
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,  # Should be [N, T, D]
                    cov_y=cov_y,
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    N=N,
                    T=T,
                    D=D,
                    C=C
                )
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

    ########################################################
    # Step 3: Fit trans effects (model_y)
    ########################################################
    def fit_trans(
        self,
        sum_factor_col: str = None,
        function_type: str = 'single_hill',  # or 'additive', 'nested'
        polynomial_degree: int = 6,
        lr: float = 1e-3,
        niters: int = None,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        K_alpha: float = 2,
        Vmax_alpha: float = 2,
        n_mu: float = 0,
        n_sigma_base: float = 5,
        p_n: float = 1e-6,
        epsilon: float = 1e-6,
        threshold: float = 0.1,
        slope: float = 50,
        init_temp: float = 1.0,
        #final_temp: float = 1e-8,
        final_temp: float = 0.1,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        **kwargs
    ):
        """
        Fit trans effects using distribution-specific likelihood.

        Parameters
        ----------
        modality_name : str, optional
            Name of modality to fit. If None, uses primary modality.
        distribution : str, optional
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
            If None, auto-detected from modality.
        sum_factor_col : str, optional
            Sum factor column name. Required for negbinom, ignored for others.
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (e.g., total counts for PSI)
            If None, auto-detected from modality.
        function_type : str
            Dose-response function: 'single_hill', 'additive_hill', 'polynomial'
        **kwargs
            Additional parameters for specific distributions

        Notes
        -----
        Each modality stores its own fitting results.
        Primary modality results are also stored at model level for backward compatibility.
        Trans fitting requires that technical fit has been performed for the modality.

        If technical_group_code is set (via set_technical_groups()), it will be used for
        correction. Otherwise, no group correction is applied.

        Examples
        --------
        >>> model.set_technical_groups(['cell_line'])  # Optional, for correction
        >>> model.fit_trans(sum_factor_col='sum_factor', function_type='additive_hill')
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

        # ---------------------------
        # Set conditional default for niters
        # ---------------------------
        if niters is None:
            # Default: 100,000 unless multivariate (multinomial or mvnormal) OR polynomial function, then 200,000
            if distribution in ('multinomial', 'mvnormal'):
                niters = 200_000
                print(f"[INFO] Using default niters=200,000 for multivariate distribution '{distribution}'")
            elif function_type == 'polynomial':
                niters = 200_000
                print(f"[INFO] Using default niters=200,000 for polynomial function")
            else:
                niters = 100_000
                print(f"[INFO] Using default niters=100,000 for distribution '{distribution}' and function_type '{function_type}'")

        # Check that technical fit has been done for this modality
        if modality.alpha_y_prefit is None and 'technical_group_code' in self.model.meta.columns:
            raise ValueError(
                f"Modality '{modality_name}' has not been fit with fit_technical(). "
                f"Please run fit_technical(modality_name='{modality_name}') first."
            )

        # Get counts from modality
        counts_to_fit = modality.counts

        # Get cell names from modality
        if modality.cell_names is not None:
            modality_cells = modality.cell_names
        else:
            # Modality doesn't have cell names - assume same order as model.meta['cell']
            modality_cells = self.model.meta['cell'].values[:counts_to_fit.shape[modality.cells_axis]]

        # Get technical fit results from modality (NOT self.model.alpha_y_prefit!)
        alpha_y_prefit = modality.alpha_y_prefit
        # alpha_y_type is always 'posterior' since it came from fit_technical
        alpha_y_type = 'posterior' if alpha_y_prefit is not None else None

        print(f"[INFO] Fitting trans model for modality '{modality_name}' (distribution: {distribution})")

        # Validate distribution-specific requirements
        from ..distributions import requires_sum_factor, requires_denominator

        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")

        # convert to gpu for fitting if applicable
        if self.model.x_true is not None and self.model.x_true.device != self.model.device:
            self.model.x_true = self.model.x_true.to(self.model.device)
        if alpha_y_prefit is not None and alpha_y_prefit.device != self.model.device:
            alpha_y_prefit = alpha_y_prefit.to(self.model.device)

        if not hasattr(self, "log2_x_true") or self.log2_x_true is None:
            if self.model.x_true is not None:
                self.log2_x_true = torch.log2(self.model.x_true)
                self.log2_x_true_type = self.model.x_true_type

        # Handle cell subsetting
        # Subset meta to cells in this modality (work with copy)
        modality_cell_set = set(modality_cells)
        meta_subset = self.model.meta[self.model.meta['cell'].isin(modality_cell_set)].copy()

        # Check if technical_group_code exists (for correction)
        if "technical_group_code" in meta_subset.columns:
            C = meta_subset['technical_group_code'].nunique()
            groups_tensor = torch.tensor(meta_subset['technical_group_code'].values, dtype=torch.long, device=self.model.device)
            print(f"[INFO] Using technical_group_code with {C} groups for correction")
        else:
            C = None
            groups_tensor = None
            if alpha_y_prefit is None:
                warnings.warn("no alpha_y_prefit and no technical_group_code, assuming no confounding effect.")

        # Get cell indices for this modality
        cell_indices = [i for i, c in enumerate(modality_cells) if c in modality_cell_set]
        N = len(cell_indices)

        # Subset counts to modality cells
        # Handle both 2D and 3D arrays
        if counts_to_fit.ndim == 2:
            if modality.cells_axis == 1:
                y_obs = counts_to_fit[:, cell_indices].T  # [T, N] -> [N, T]
            else:
                y_obs = counts_to_fit[cell_indices, :]  # Already [N, T]
            T = y_obs.shape[1]
        elif counts_to_fit.ndim == 3:
            # 3D data: (features, cells, categories/dimensions)
            # Subset and transpose to (cells, features, categories/dimensions)
            counts_subset = counts_to_fit[:, cell_indices, :]  # [T, N, K]
            y_obs = counts_subset.transpose(1, 0, 2)  # [T, N, K] -> [N, T, K]
            T = counts_subset.shape[0]
        else:
            raise ValueError(f"Unexpected number of dimensions: {counts_to_fit.ndim}")

        # Handle sum factors for modality cells
        if sum_factor_col is not None:
            sum_factor_tensor = torch.tensor(meta_subset[sum_factor_col].values, dtype=torch.float32, device=self.model.device)
        else:
            sum_factor_tensor = torch.ones(N, dtype=torch.float32, device=self.model.device)

        # Handle denominator for modality cells
        denominator_tensor = None
        if denominator is not None:
            if denominator.ndim == 2:
                if modality.cells_axis == 1:
                    denominator_subset = denominator[:, cell_indices].T  # [T, N] -> [N, T]
                else:
                    denominator_subset = denominator[cell_indices, :]  # [N, T]
                denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32, device=self.model.device)
            elif denominator.ndim == 3:
                # 3D denominator (shouldn't happen for current distributions, but handle it)
                denominator_subset = denominator[:, cell_indices, :].transpose(1, 0, 2)
                denominator_tensor = torch.tensor(denominator_subset, dtype=torch.float32, device=self.model.device)

        # Detect data dimensions (for multinomial and mvnormal)
        from ..distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs.shape[2]  # Number of categories
                elif distribution == 'mvnormal':
                    D = y_obs.shape[2]  # Dimensionality
            else:
                raise ValueError(f"Distribution '{distribution}' requires 3D data but got shape {y_obs.shape}")
        if self.model.x_true_type == 'point':
            x_true_mean = self.model.x_true
        elif self.model.x_true_type == 'posterior':
            x_true_mean = self.model.x_true.mean(dim=0)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.model.device)
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.model.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.model.device)
        K_alpha_tensor = torch.tensor(K_alpha, dtype=torch.float32, device=self.model.device)
        Vmax_alpha_tensor = torch.tensor(Vmax_alpha, dtype=torch.float32, device=self.model.device)
        n_mu_tensor = torch.tensor(n_mu, dtype=torch.float32, device=self.model.device)
        n_sigma_base_tensor = torch.tensor(n_sigma_base, dtype=torch.float32, device=self.model.device)
        y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32, device=self.model.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.model.device)
        p_n_tensor = torch.tensor(p_n, dtype=torch.float32, device=self.model.device)

        # --- robust, finite bounds for n to avoid overflow in x**n ---
        # use the same x_true sample type you use elsewhere
        x_for_bounds = self.model.x_true if self.model.x_true_type == "posterior" else self.model.x_true
        x_min = torch.clamp(x_for_bounds.min(), min=1e-12)  # strictly > 0 to avoid log(0)
        x_max = x_for_bounds.max()

        log_fmax = torch.log(torch.tensor(torch.finfo(torch.float32).max, device=self.model.device))

        # candidates can be inf if denominator ~ 0; we cap them later
        nmin_cand = (-log_fmax / torch.abs(torch.log(x_min))) if (x_min < 1) else torch.tensor(float('-inf'), device=self.model.device)
        nmax_cand = ( log_fmax / torch.abs(torch.log(x_max))) if (x_max > 1) else torch.tensor(float('inf'),  device=self.model.device)
        print(f'nmin_cand={nmin_cand}, nmax_cand={nmax_cand}')

        BOX_LOW  = torch.tensor(-20.0, device=self.model.device)
        BOX_HIGH = torch.tensor( 20.0, device=self.model.device)

        nmin = torch.where(torch.isfinite(nmin_cand), torch.maximum(nmin_cand, BOX_LOW),  BOX_LOW)
        nmax = torch.where(torch.isfinite(nmax_cand), torch.minimum(nmax_cand, BOX_HIGH), BOX_HIGH)
        # ensure proper ordering just in case
        nmin = torch.minimum(nmin, nmax)
        print(f'nmin={nmin}, nmax={nmax}')


        guides_tensor = torch.tensor(self.model.meta['guide_code'].values, dtype=torch.long, device=self.model.device)
        K_max_tensor = torch.max(torch.stack([torch.mean(x_true_mean[guides_tensor == g]) for g in torch.unique(guides_tensor)]))

        # For negbinom, normalize by sum factors; for other distributions, use raw values
        if sum_factor_col is not None:
            y_obs_factored = y_obs_tensor / sum_factor_tensor.view(-1, 1)
        else:
            y_obs_factored = y_obs_tensor

        Vmax_mean_tensor = torch.max(torch.stack([torch.mean(y_obs_factored[guides_tensor == g, :], dim=0) for g in torch.unique(guides_tensor)]), dim=0)[0]
        Amean_tensor = torch.min(torch.stack([torch.mean(y_obs_factored[guides_tensor == g, :], dim=0) for g in torch.unique(guides_tensor)]), dim=0)[0]

        Amean_tensor = torch.where(
            torch.isfinite(Amean_tensor),
            Amean_tensor.clamp_min(epsilon_tensor),
            torch.full_like(Amean_tensor, 1.0)
        )
        Vmax_mean_tensor = torch.where(
            torch.isfinite(Vmax_mean_tensor),
            Vmax_mean_tensor.clamp_min(epsilon_tensor),
            torch.full_like(Vmax_mean_tensor, 1.0)
        )

        assert self.model.x_true.device == self.model.device
        if alpha_y_prefit is not None:
            assert alpha_y_prefit.device == self.model.device
        assert y_obs_tensor.device == self.model.device
        assert sum_factor_tensor.device == self.model.device

        def init_loc_fn(site):
            name = site["name"]
        
            if "poly_coeff" in name:
                return torch.zeros(T)
        
            return pyro.infer.autoguide.initialization.init_to_median(site)
        
        if function_type == "polynomial":
            from torch.optim.lr_scheduler import OneCycleLR
            guide_y = pyro.infer.autoguide.AutoMultivariateNormal(self._model_y, init_loc_fn=init_loc_fn)

            guide_y(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                n_sigma_base_tensor,
                Amean_tensor,
                p_n_tensor,
                threshold,
                slope,
                epsilon_tensor,
                x_true_sample = self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true,
                log2_x_true_sample = self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit,
                C = C,
                groups_tensor=groups_tensor,
                predictive_mode=False,
                temperature=torch.tensor(init_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=False,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
            )

            # 1) Create the base torch optimizer (with gradient clipping built in)
            base_optimizer = torch.optim.Adam(
                guide_y.parameters(),
                lr=1e-3,                   # initial lr (this will be overridden by the scheduler)
                betas=(0.9, 0.999),
            )
            
            # 2) Wrap it in PyroLRScheduler by passing the constructor + kwargs
            optimizer = pyro.optim.PyroLRScheduler(
                # 1) Which scheduler class?
                scheduler_constructor=OneCycleLR,
            
                # 2) Build optim_args with the optimizer *class* + its init args,
                #    then your scheduler’s own kwargs.
                optim_args={
                    # -- for your torch optimizer --
                    "optimizer": torch.optim.Adam,
                    "optim_args": {
                        "lr":     1e-3,          # placeholder, overridden by scheduler
                        "betas": (0.9, 0.999),
                    },
                    # -- for OneCycleLR itself --
                    "max_lr":          1e-2,
                    "total_steps":     niters,
                    "pct_start":       0.1,
                    "div_factor":      25.0,
                    "final_div_factor":1e4,
                },
            
                # 3) (Optional) still want gradient clipping?
                clip_args={"clip_norm": 5}
            )

            #svi   = pyro.infer.SVI(self._model_y, guide_y, optimizer, pyro.infer.Trace_ELBO(num_particles=5, vectorize_particles=True))
            svi   = pyro.infer.SVI(self._model_y, guide_y, optimizer, pyro.infer.Trace_ELBO())
        else:
            guide_y = pyro.infer.autoguide.AutoNormalMessenger(self._model_y)
            optimizer = pyro.optim.Adam({"lr": lr})
            svi = pyro.infer.SVI(self._model_y, guide_y, optimizer, 
                                 loss=pyro.infer.Trace_ELBO())
        
        for name, value in pyro.get_param_store().items():
            if "poly_coeff" in name and "loc" in name:
                print(name, value.shape, value.min().item(), value.max().item())

        self.losses_trans = []
        smoothed_loss = None
        for step in range(niters):
            # a simple linear schedule from init_temp down to final_temp:
            fraction_done = step / float(niters)
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                current_temp = init_temp + (final_temp - init_temp) * fraction_done
            elif function_type == 'polynomial':
                current_temp = init_temp + (final_temp - init_temp) * (2*fraction_done-1)
                #current_temp = init_temp + (final_temp - init_temp) * fraction_done
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
                
            #if step < 0.7 * niters:
            #    # First 70% of training: linearly decrease from 1.0 to 0.1
            #    current_temp = 1.0 - (0.9 * (step / (0.7 * niters)))
            #else:
            #    # Last 30% of training: exponentially cool down to 0.0005
            #    current_temp = 0.1 * (final_temp/0.1) ** ((step - 0.7 * niters) / (0.3 * niters))


            # Sample from posterior - use alpha_y_prefit shape if available, otherwise x_true
            if alpha_y_prefit is not None and alpha_y_type == "posterior":
                samp = torch.randint(high=alpha_y_prefit.shape[0], size=(1,)).item()
            elif self.model.x_true_type == "posterior":
                samp = torch.randint(high=self.model.x_true.shape[0], size=(1,)).item()
            else:
                samp = 0  # No sampling needed if both are point estimates

            # Sample from posterior
            x_true_sample = (
                self.model.x_true[samp] if samp < self.model.x_true.shape[0] else self.model.x_true.mean(dim=0)
                if self.model.x_true_type == "posterior" else self.model.x_true
            )
            log2_x_true_sample = (
                self.log2_x_true[samp] if samp < self.log2_x_true.shape[0] else self.log2_x_true.mean(dim=0)
                if self.log2_x_true_type == "posterior" else self.log2_x_true
            )
            alpha_y_sample = (
                alpha_y_prefit[samp] if samp < alpha_y_prefit.shape[0] else alpha_y_prefit.mean(dim=0)
                if alpha_y_type == "posterior" else alpha_y_prefit
            ) if alpha_y_prefit is not None else None

            #use_straight_through = step >= int(0.7 * niters)
            use_straight_through = False
            
            loss = svi.step(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                n_sigma_base_tensor,
                Amean_tensor,
                p_n_tensor,
                threshold,
                slope,
                epsilon_tensor,
                x_true_sample = x_true_sample,
                log2_x_true_sample = log2_x_true_sample,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_sample,
                C = C,
                groups_tensor=groups_tensor,
                predictive_mode=False,
                temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=use_straight_through,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True if function_type != 'polynomial' else True if fraction_done>=0.5 else False,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
            )
            
            self.losses_trans.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {Vmax_mean_tensor.device}")
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
            guide_y.to("cpu")
            self.model.device = torch.device("cpu")
        
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": self._to_cpu(y_obs_tensor),
                "sum_factor_tensor": self._to_cpu(sum_factor_tensor),
                "beta_o_alpha_tensor": self._to_cpu(beta_o_alpha_tensor),
                "beta_o_beta_tensor": self._to_cpu(beta_o_beta_tensor),
                "alpha_alpha_mu_tensor": self._to_cpu(alpha_alpha_mu_tensor),
                "K_max_tensor": self._to_cpu(K_max_tensor),
                "K_alpha_tensor": self._to_cpu(K_alpha_tensor),
                "Vmax_mean_tensor": self._to_cpu(Vmax_mean_tensor),
                "Vmax_alpha_tensor": self._to_cpu(Vmax_alpha_tensor),
                "n_mu_tensor": self._to_cpu(n_mu_tensor),
                "n_sigma_base_tensor": self._to_cpu(n_sigma_base_tensor),
                "Amean_tensor": self._to_cpu(Amean_tensor),
                "p_n_tensor": self._to_cpu(p_n_tensor),
                "threshold": threshold,     # plain Python ok
                "slope": slope,
                "epsilon_tensor": self._to_cpu(epsilon_tensor),
                "x_true_sample": self._to_cpu(self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true),
                "log2_x_true_sample": self._to_cpu(self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true),
                "nmin": self._to_cpu(nmin),
                "nmax": self._to_cpu(nmax),
                # Only move if not None:
                "alpha_y_sample": self._to_cpu(alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit) if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": self._to_cpu(groups_tensor) if groups_tensor is not None else None,
                "predictive_mode": True,
                # create on CPU explicitly since we just set self.model.device="cpu"
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=torch.device("cpu")),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": self._to_cpu(denominator_tensor) if denominator_tensor is not None else None,
                "K": K,
                "D": D,
            }
        else:
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": y_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "K_max_tensor": K_max_tensor,
                "K_alpha_tensor": K_alpha_tensor,
                "Vmax_mean_tensor": Vmax_mean_tensor,
                "Vmax_alpha_tensor": Vmax_alpha_tensor,
                "n_mu_tensor": n_mu_tensor,
                "n_sigma_base_tensor": n_sigma_base_tensor,
                "Amean_tensor": Amean_tensor,
                "p_n_tensor": p_n_tensor,
                "threshold": threshold,
                "slope": slope,
                "epsilon_tensor": epsilon_tensor,
                "x_true_sample": self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true,
                "log2_x_true_sample": self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true,
                "nmin": nmin,
                "nmax": nmax,
                "alpha_y_sample": alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": groups_tensor if groups_tensor is not None else None,
                "predictive_mode": True,
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=self.model.device),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": denominator_tensor if denominator_tensor is not None else None,
                "K": K,
                "D": D,
            }
        
        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "y_obs")

        if minibatch_size is not None:
            from collections import defaultdict

            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_y(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(self._to_cpu(v))
                    if self.model.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            posterior_samples_y = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}

        else:
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=nsamples#,
                #parallel=True
            )
            with torch.no_grad():
                posterior_samples_y = predictive_y(**model_inputs)
                if self.model.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        if run_on_cpu:
            self.model.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.model.device to:", self.model.device)

        for k, v in posterior_samples_y.items():
            posterior_samples_y[k] = self._to_cpu(v)

        # Store results
        # Store in modality
        modality.posterior_samples_trans = posterior_samples_y

        # Update alpha_y_prefit in modality if it was None
        if modality.alpha_y_prefit is None and groups_tensor is not None:
            modality.alpha_y_prefit = posterior_samples_y["alpha_y"].mean(dim=0)

        # If primary modality, also store at model level (backward compatibility)
        if modality_name == self.model.primary_modality:
            self.model.posterior_samples_trans = posterior_samples_y
            if self.model.alpha_y_prefit is None and groups_tensor is not None:
                self.model.alpha_y_prefit = posterior_samples_y["alpha_y"].mean(dim=0)
            print(f"[INFO] Stored results in modality '{modality_name}' and at model level (primary modality)")
        else:
            print(f"[INFO] Stored results in modality '{modality_name}'")

        if self.model.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        pyro.clear_param_store()

        print("Finished fit_trans.")



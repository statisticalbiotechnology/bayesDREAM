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
        Amean_tensor,
        p_n_tensor,
        epsilon_tensor,
        x_true_sample,
        log2_x_true_sample,
        nmin,
        nmax,
        alpha_y_sample=None,
        C=None,
        groups_tensor=None,
        temperature=1.0,
        use_straight_through=False,
        function_type='single_hill',
        polynomial_degree=6,
        use_alpha=True,
        distribution='negbinom',
        denominator_tensor=None,
        K=None,
        D=None,
        mean_within_guide_var=None,
        x_true_CV=None,
        use_data_driven_priors=True,
    ):

        if use_alpha:
            alpha_dist = dist.RelaxedBernoulliStraightThrough if use_straight_through else dist.RelaxedBernoulli

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
            with pyro.plate("technical_groups_plate", C-1, dim=-2):  # **Now correctly uses C-1**
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

        # Degrees of freedom for Student-t distribution (nu_y)
        # Only needed if using studentt distribution
        nu_y = None
        if distribution == 'studentt':
            # Two options (must match fit_technical choice):
            # Option 1: Fixed value (simpler, faster) - COMMENTED OUT FOR NOW
            # nu_y = self._t(3.0)
            # Option 2: Sample per-feature (more flexible, slower) - ACTIVE
            with trans_plate:
                nu_y = pyro.sample("nu_y", dist.Gamma(self._t(10.0), self._t(2.0)))  # mean~5, ensures df>2
    
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

            # For multinomial, Amean_tensor and Vmax_mean_tensor are already [T, K] shaped
            # For binomial, they are [T] shaped
            # We keep them in their native shape for per-category priors in multinomial

            # Baseline parameter A depends on distribution:
            # - normal/studentt: can be negative (natural value space)
            # - negbinom: positive count
            # - binomial/multinomial: probability in [0,1], using NEW reparameterization
            if distribution in ['normal', 'studentt']:
                # For continuous distributions: average of min and max
                Amean_for_A = Amean_tensor  # [T]
                Vmax_for_A = Vmax_mean_tensor  # [T]
                Amean_adjusted = ((1 - weight) * Amean_for_A) + (weight * Vmax_for_A) + epsilon_tensor
                # Use Normal distribution to allow negative baseline values
                A = pyro.sample("A", dist.Normal(Amean_adjusted, Amean_adjusted.abs()))

            elif distribution in ['binomial', 'multinomial']:
                # SIMPLIFIED Beta/Dirichlet priors with weak regularization
                # For binomial:
                #   A ~ Beta with α=1 (pushes toward 0, mean = A_mean)
                #   upper_limit ~ Beta with β=1 (pushes toward 1, mean = Vmax_mean)
                # For multinomial:
                #   A ~ Dirichlet with concentration = mean * K (weak prior)
                #   upper_limit ~ Dirichlet with concentration = mean * K

                # Amean_tensor: [T] for binomial, [T, K] for multinomial
                # Vmax_mean_tensor: [T] for binomial, [T, K] for multinomial

                # Sample A and upper_limit
                if distribution == 'multinomial' and Amean_tensor.ndim > 1:
                    # For multinomial: Use Dirichlet with weak concentration
                    # concentration = mean_normalized * K (where K = number of categories)
                    # This gives each category concentration ≈ 1 on average (weak prior)
                    K_dim = Amean_tensor.shape[-1]

                    # For multinomial: use Dirichlet priors over K categories
                    if use_data_driven_priors:
                        # Data-driven Dirichlet: concentration proportional to mean probabilities
                        # Normalize means to sum to 1
                        A_mean_clamped = Amean_tensor.clamp(min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        A_mean_normalized = A_mean_clamped / A_mean_clamped.sum(dim=-1, keepdim=True)  # [T, K]

                        Vmax_clamped = Vmax_mean_tensor.clamp(min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        upper_mean_normalized = Vmax_clamped / Vmax_clamped.sum(dim=-1, keepdim=True)  # [T, K]

                        # Weak concentration: mean_normalized * K gives ~1 per category
                        concentration_A = A_mean_normalized * K_dim  # [T, K]
                        concentration_upper = upper_mean_normalized * K_dim  # [T, K]

                        #print(f"[INFO] Dirichlet data-driven concentration: A={concentration_A.mean().item():.2f}, upper={concentration_upper.mean().item():.2f}")
                    else:
                        # Uniform Dirichlet: all categories have equal concentration=1
                        concentration_A = self._t(1.0).expand([T_dim, K_dim])  # [T, K] with all 1s
                        concentration_upper = self._t(1.0).expand([T_dim, K_dim])  # [T, K] with all 1s
                        print(f"[INFO] Using uniform Dirichlet(1, ..., 1) priors for A and upper_limit ({K_dim} categories)")

                    # Sample K-dimensional probability vectors from Dirichlet
                    # Each row sums to 1
                    A = pyro.sample("A", dist.Dirichlet(concentration_A))  # [T, K]
                    upper_limit = pyro.sample("upper_limit", dist.Dirichlet(concentration_upper))  # [T, K]

                else:
                    # For binomial: Beta priors
                    if use_data_driven_priors:
                        # Data-driven Beta priors with α=1 or β=1
                        # A ~ Beta(α=1, β) with mean = A_mean
                        #   mean = α/(α+β) = 1/(1+β) = A_mean
                        #   β = (1-A_mean)/A_mean
                        beta_A = (1.0 - Amean_tensor) / Amean_tensor  # [T]
                        alpha_A = self._t(1.0)  # [scalar] or could be torch.ones_like(beta_A)

                        # upper_limit ~ Beta(α, β=1) with mean = Vmax_mean
                        #   mean = α/(α+1) = Vmax_mean
                        #   α = Vmax_mean/(1-Vmax_mean)
                        alpha_upper = Vmax_mean_tensor / (1.0 - Vmax_mean_tensor)  # [T]
                        beta_upper = self._t(1.0)  # [scalar]

                        #print(f"[INFO] Beta data-driven priors: A (α=1, β̄={beta_A.mean().item():.2f}), upper (ᾱ={alpha_upper.mean().item():.2f}, β=1)")
                    else:
                        # Uniform priors: Beta(1, 1) for both A and upper_limit
                        alpha_A = self._t(1.0)
                        beta_A = self._t(1.0)
                        alpha_upper = self._t(1.0)
                        beta_upper = self._t(1.0)
                        print(f"[INFO] Using uniform Beta(1, 1) priors for A and upper_limit")

                    # Sample per-feature [T]
                    # Note: When using uniform priors, these become scalars broadcast to [T]
                    if use_data_driven_priors:
                        A = pyro.sample("A", dist.Beta(alpha_A, beta_A))  # [T]
                        upper_limit = pyro.sample("upper_limit", dist.Beta(alpha_upper, beta_upper))  # [T]
                    else:
                        # Uniform priors - need to expand to match shape [T]
                        A = pyro.sample("A", dist.Beta(alpha_A, beta_A).expand([T_dim]))  # [T]
                        upper_limit = pyro.sample("upper_limit", dist.Beta(alpha_upper, beta_upper).expand([T_dim]))  # [T]

                # Compute Vmax_sum (total amplitude available for Hills)
                # For multinomial: Both A and upper_limit are K-dimensional from Dirichlet (sum to 1)
                # Vmax_sum = upper_limit - A gives amplitude for each category
                # Just clamp to ensure non-negative, don't enforce ordering with min/max
                Vmax_sum = (upper_limit - A).clamp_min(epsilon_tensor)

                # Store Vmax_sum for use in Hill computation
                Vmax_sum = pyro.deterministic("Vmax_sum", Vmax_sum)  # [T] or [T, K]

            else:
                # For negbinom: A must be positive count
                Amean_for_A = Amean_tensor  # [T]
                Vmax_for_A = Vmax_mean_tensor  # [T]
                Amean_adjusted = ((1 - weight) * Amean_for_A) + (weight * Vmax_for_A) + epsilon_tensor
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

                # For multinomial, use reduced Vmax_for_A (without category dimension) for priors
                # For other distributions, use Vmax_mean_tensor directly
                if distribution == 'multinomial' and Vmax_mean_tensor.ndim > 1:
                    Vmax_prior_mean = Vmax_for_A  # [T] - already reduced
                else:
                    Vmax_prior_mean = Vmax_mean_tensor  # [T]

                Vmax_sigma = (Vmax_prior_mean / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor

                # For multinomial, we need per-category parameters (K-1 categories, Kth is residual)
                if distribution == 'multinomial' and K is not None:
                    K_minus_1 = K - 1
                    # Sample parameters for K-1 categories
                    # Each category gets its own Hill function parameters
                    with pyro.plate("category_plate", K_minus_1, dim=-2):
                        n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_tensor, sigma_n_a))  # [K-1, T]
                        BOX_LOW  = self._t(-20.0)
                        BOX_HIGH = self._t( 20.0)
                        low  = torch.maximum(nmin, BOX_LOW)
                        high = torch.minimum(nmax, BOX_HIGH)
                        n_a = pyro.deterministic(
                            "n_a",
                            (alpha.unsqueeze(-2) * n_a_raw).clamp(min=low.item(), max=high.item())
                        )  # [K-1, T] * [1, T] -> [K-1, T]
                else:
                    # For non-multinomial: single set of parameters per feature
                    n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_tensor, sigma_n_a))
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

                # UNIFIED Vmax and K priors (Log-Normal for all distributions)
                # K uses CV-based std (scale-invariant, works without guides)
                # Vmax uses raw variance (data-driven) for negbinom/normal/studentt

                # K parameterization (UNIFIED for all distributions)
                K_mean_prior = (K_max_tensor / 2.0).clamp_min(epsilon_tensor)  # Half of max cis expression
                if x_true_CV is not None:
                    K_std_prior = K_mean_prior * x_true_CV  # Scale by coefficient of variation
                else:
                    # Fallback to fixed alpha if CV not provided
                    K_std_prior = K_max_tensor / (self._t(2.0) * torch.sqrt(K_alpha_tensor))

                # Log-Normal parameterization for K
                ratio_K = (K_std_prior / K_mean_prior).clamp_min(self._t(1e-6))
                K_log_sigma = torch.sqrt(torch.log1p(ratio_K ** 2))
                K_log_mu = torch.log(K_mean_prior) - 0.5 * K_log_sigma ** 2

                if distribution in ['binomial', 'multinomial']:
                    # For binomial/multinomial: Vmax_sum from Beta/Dirichlet reparameterization
                    # (Vmax_sum already computed from upper_limit - A, no need to register again)
                    Vmax_a = Vmax_sum  # [T] or [T, K]

                    # K_a: unified Log-Normal
                    if distribution == 'multinomial' and K is not None:
                        # For multinomial: K-1 parameters (Kth category is residual)
                        K_minus_1 = K - 1
                        with pyro.plate("category_plate_K_a", K_minus_1, dim=-1):
                            log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))  # [T, K-1]
                            K_a = pyro.deterministic("K_a", torch.exp(log_K_a))  # [T, K-1]
                    else:
                        # For binomial: K parameters are per-feature [T]
                        log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))
                        K_a = pyro.deterministic("K_a", torch.exp(log_K_a))

                else:
                    # For negbinom/normal/studentt: Learn Vmax using Log-Normal with data-driven variance
                    Vmax_mean_prior = Vmax_prior_mean.clamp_min(epsilon_tensor)  # [T]

                    # Use raw variance (not CV) for Vmax
                    if mean_within_guide_var is not None:
                        if mean_within_guide_var.ndim > 1:
                            # Multinomial case: already handled by Vmax_prior_mean reduction
                            Vmax_var_prior = mean_within_guide_var.mean(dim=-1)  # [T]
                        else:
                            Vmax_var_prior = mean_within_guide_var  # [T]
                    else:
                        # Fallback if variance not available
                        Vmax_var_prior = (Vmax_mean_prior ** 2) / Vmax_alpha_tensor

                    Vmax_std_prior = torch.sqrt(Vmax_var_prior.clamp_min(epsilon_tensor))

                    # Log-Normal parameterization for Vmax
                    ratio_Vmax = (Vmax_std_prior / Vmax_mean_prior).clamp_min(self._t(1e-6))
                    Vmax_log_sigma = torch.sqrt(torch.log1p(ratio_Vmax ** 2))
                    Vmax_log_mu = torch.log(Vmax_mean_prior) - 0.5 * Vmax_log_sigma ** 2

                    log_Vmax_a = pyro.sample("log_Vmax_a", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
                    Vmax_a = pyro.deterministic("Vmax_a", torch.exp(log_Vmax_a))

                    # K_a: unified Log-Normal (same as binomial/multinomial)
                    log_K_a = pyro.sample("log_K_a", dist.Normal(K_log_mu, K_log_sigma))
                    K_a = pyro.deterministic("K_a", torch.exp(log_K_a))

                # Sample all required parameters (additive_hill and nested_hill need second set)
                if function_type in ['additive_hill', 'nested_hill']:
                    beta = pyro.sample("beta", alpha_dist(temperature=temperature, probs=p_n_tensor))

                    # n_b: per-category for multinomial, single for others
                    if distribution == 'multinomial' and K is not None:
                        K_minus_1 = K - 1
                        with pyro.plate("category_plate_b", K_minus_1, dim=-2):
                            n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_tensor, sigma_n_b))  # [K-1, T]
                            BOX_LOW  = self._t(-20.0)
                            BOX_HIGH = self._t( 20.0)
                            low  = torch.maximum(nmin, BOX_LOW)
                            high = torch.minimum(nmax, BOX_HIGH)
                            n_b = pyro.deterministic(
                                "n_b",
                                (beta.unsqueeze(-2) * n_b_raw).clamp(min=low.item(), max=high.item())
                            )  # [K-1, T]
                    else:
                        n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_tensor, sigma_n_b))
                        BOX_LOW  = self._t(-20.0)
                        BOX_HIGH = self._t( 20.0)
                        low  = torch.maximum(nmin, BOX_LOW)
                        high = torch.minimum(nmax, BOX_HIGH)
                        n_b = pyro.deterministic(
                            "n_b",
                            (beta * n_b_raw).clamp(min=low.item(), max=high.item())
                        )

                    # Vmax_b and K_b: UNIFIED priors (same as Vmax_a and K_a)
                    if distribution in ['binomial', 'multinomial']:
                        # For binomial/multinomial: SAME Vmax_sum for both Hills
                        # y = A + Vmax_sum * (alpha * Hill_a + beta * Hill_b)
                        Vmax_b = Vmax_sum  # Same as Vmax_a

                        # K_b: unified Log-Normal (same parameterization as K_a)
                        if distribution == 'multinomial' and K is not None:
                            # For multinomial: K-1 parameters (Kth category is residual)
                            K_minus_1 = K - 1
                            with pyro.plate("category_plate_K_b", K_minus_1, dim=-1):
                                log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))  # [T, K-1]
                                K_b = pyro.deterministic("K_b", torch.exp(log_K_b))  # [T, K-1]
                        else:
                            # For binomial: K parameters are per-feature [T]
                            log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))
                            K_b = pyro.deterministic("K_b", torch.exp(log_K_b))

                    else:
                        # For negbinom/normal/studentt: Vmax_b and K_b use Log-Normal (same as Vmax_a and K_a)
                        log_Vmax_b = pyro.sample("log_Vmax_b", dist.Normal(Vmax_log_mu, Vmax_log_sigma))
                        Vmax_b = pyro.deterministic("Vmax_b", torch.exp(log_Vmax_b))

                        log_K_b = pyro.sample("log_K_b", dist.Normal(K_log_mu, K_log_sigma))
                        K_b = pyro.deterministic("K_b", torch.exp(log_K_b))
                
                # Compute Hill function(s)
                # Hill_based_positive returns values in [0, Vmax]
                # We compute: y = A + alpha * Hill + beta * Hill (for additive)

                if distribution == 'multinomial' and K is not None:
                    # NEW FORMULATION for multinomial:
                    # A and Vmax_sum are K-dimensional (from Dirichlet, sum to 1)
                    # Fit K-1 independent Hill functions
                    # For each category k in K-1:
                    #   y_k = A_k + Vmax_sum_k * (alpha * Hill_a_k + beta * Hill_b_k)
                    # Then: y_K = 1 - sum(y_1, ..., y_{K-1})
                    # This ensures probabilities sum to 1, and Kth category gets whatever is left

                    K_dim = K
                    K_minus_1 = K - 1
                    x_expanded = x_true.unsqueeze(-1)  # [N, 1]

                    # A and Vmax_sum are [T, K] from Dirichlet
                    # Extract K-1 for fitting Hills (Kth doesn't get a Hill function)
                    A_kminus1 = A[..., :K_minus_1]  # [T, K-1]
                    Vmax_sum_kminus1 = Vmax_sum[..., :K_minus_1]  # [T, K-1]

                    # Expand for broadcasting
                    A_kminus1_expanded = A_kminus1.unsqueeze(0)  # [1, T, K-1]
                    Vmax_sum_kminus1_expanded = Vmax_sum_kminus1.unsqueeze(0)  # [1, T, K-1]

                    if function_type == 'single_hill':
                        # Compute Hills for K-1 categories
                        Hilla_list = []
                        for k in range(K_minus_1):
                            hill_k = Hill_based_positive(x_expanded, Vmax=self._t(1.0), A=0,
                                                        K=K_a[:, k], n=n_a[:, k],  # [T]
                                                        epsilon=epsilon_tensor)  # [N, T]
                            Hilla_list.append(hill_k.unsqueeze(-1))  # [N, T, 1]
                        Hilla_kminus1 = torch.cat(Hilla_list, dim=-1)  # [N, T, K-1]

                        # Combine: y = A + Vmax_sum * alpha * Hill
                        combined_hill = alpha.unsqueeze(0).unsqueeze(-1) * Hilla_kminus1  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + Vmax_sum_kminus1_expanded * combined_hill  # [N, T, K-1]

                    elif function_type == 'additive_hill':
                        # Compute Hills for K-1 categories
                        Hilla_list = []
                        Hillb_list = []
                        for k in range(K_minus_1):
                            hill_a_k = Hill_based_positive(x_expanded, Vmax=self._t(1.0), A=0,
                                                          K=K_a[:, k], n=n_a[:, k],  # [T]
                                                          epsilon=epsilon_tensor)  # [N, T]
                            hill_b_k = Hill_based_positive(x_expanded, Vmax=self._t(1.0), A=0,
                                                          K=K_b[:, k], n=n_b[:, k],  # [T]
                                                          epsilon=epsilon_tensor)  # [N, T]
                            Hilla_list.append(hill_a_k.unsqueeze(-1))  # [N, T, 1]
                            Hillb_list.append(hill_b_k.unsqueeze(-1))  # [N, T, 1]
                        Hilla_kminus1 = torch.cat(Hilla_list, dim=-1)  # [N, T, K-1]
                        Hillb_kminus1 = torch.cat(Hillb_list, dim=-1)  # [N, T, K-1]

                        # Combine
                        combined_hill = (alpha.unsqueeze(0).unsqueeze(-1) * Hilla_kminus1 +
                                       beta.unsqueeze(0).unsqueeze(-1) * Hillb_kminus1)  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + Vmax_sum_kminus1_expanded * combined_hill  # [N, T, K-1]

                    elif function_type == 'nested_hill':
                        # Compute nested Hills for K-1 categories
                        Hillb_list = []
                        for k in range(K_minus_1):
                            hill_a_k = Hill_based_positive(x_expanded, Vmax=self._t(1.0), A=0,
                                                          K=K_a[:, k], n=n_a[:, k],
                                                          epsilon=epsilon_tensor)  # [N, T]
                            hill_b_k = Hill_based_positive(hill_a_k.unsqueeze(-1), Vmax=self._t(1.0), A=0,
                                                          K=K_b[:, k], n=n_b[:, k],
                                                          epsilon=epsilon_tensor)  # [N, T]
                            Hillb_list.append(hill_b_k.unsqueeze(-1))  # [N, T, 1]
                        Hillb_kminus1 = torch.cat(Hillb_list, dim=-1)  # [N, T, K-1]

                        combined_hill = alpha.unsqueeze(0).unsqueeze(-1) * Hillb_kminus1  # [N, T, K-1]
                        y_kminus1 = A_kminus1_expanded + Vmax_sum_kminus1_expanded * combined_hill  # [N, T, K-1]

                    # Clamp K-1 probabilities to [epsilon, 1-epsilon] to ensure valid residual
                    y_kminus1 = torch.clamp(y_kminus1, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Ensure sum of K-1 doesn't exceed 1
                    sum_kminus1 = y_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    # If sum > (1-epsilon), rescale proportionally
                    y_kminus1 = torch.where(
                        sum_kminus1 > (1.0 - epsilon_tensor),
                        y_kminus1 * (1.0 - epsilon_tensor) / sum_kminus1,
                        y_kminus1
                    )

                    # Compute Kth category as residual: y_K = 1 - sum(y_1, ..., y_{K-1})
                    y_K = 1.0 - y_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    y_K = torch.clamp(y_K, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Concatenate to get all K probabilities
                    y_dose_response = torch.cat([y_kminus1, y_K], dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: standard Hill computation
                    if distribution in ['binomial', 'multinomial']:
                        # NEW FORMULATION for binomial:
                        # Compute Hills with Vmax=1 (output [0, 1])
                        # Then scale by Vmax_sum: y = A + Vmax_sum * (alpha * Hill_a + beta * Hill_b)
                        # Hills output [0,1], alpha and beta are [0,1] from RelaxedBernoulli
                        # So combined_hill naturally stays in valid range without clamps

                        if function_type == 'single_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=self._t(1.0), A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            combined_hill = alpha * Hilla
                            y_dose_response = A + Vmax_sum * combined_hill

                        elif function_type == 'additive_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=self._t(1.0), A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=self._t(1.0), A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            combined_hill = alpha * Hilla + beta * Hillb
                            y_dose_response = A + Vmax_sum * combined_hill

                        elif function_type == 'nested_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=self._t(1.0), A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            Hillb = Hill_based_positive(Hilla, Vmax=self._t(1.0), A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            combined_hill = alpha * Hillb
                            y_dose_response = A + Vmax_sum * combined_hill

                        # NO clamp on combined_hill or y_dose_response

                    else:
                        # For negbinom/normal/studentt: use standard formulation with learned Vmax_a/Vmax_b
                        if function_type == 'single_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hilla)
                        elif function_type == 'additive_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hilla) + (beta * Hillb)
                        elif function_type == 'nested_hill':
                            Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                            Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                            y_dose_response = A + (alpha * Hillb)

            elif function_type == 'polynomial':
                assert polynomial_degree is not None and polynomial_degree >= 1, \
                    "polynomial_degree must be ≥ 1 (no intercept, A is handled separately)"

                if distribution == 'multinomial' and K is not None:
                    # For multinomial: fit K independent polynomials (one per category) in logit space
                    # Then apply softmax to get K probabilities that sum to 1
                    # This is more standard than K-1 with residual for unbounded logit space

                    # Sample per-category polynomial coefficients
                    coeffs_per_category = []
                    for d in range(1, polynomial_degree + 1):
                        with pyro.plate(f"poly_category_plate_deg{d}", K, dim=-2):
                            coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))  # [K, T]
                            coeffs_per_category.append(coeff)
                    coeffs = torch.stack(coeffs_per_category, dim=-3)  # [degree, K, T]

                    # Compute polynomial for each category
                    # coeffs: [degree, K, T]
                    # Need to permute to [degree, T, K] for Polynomial_function to work correctly
                    coeffs_permuted = coeffs.permute(0, 2, 1)  # [degree, T, K]
                    poly_val = Polynomial_function(x_true_sample, coeffs_permuted)  # [N, T, K]

                    # A is baseline logit for each category (need K logits)
                    # Sample K baseline logits (unbounded)
                    with pyro.plate("category_plate_A", K, dim=-2):
                        A_clamped = torch.clamp(A.unsqueeze(-2), min=epsilon_tensor, max=1.0 - epsilon_tensor)  # [1, T]
                        logit_A_transpose = torch.log(A_clamped) - torch.log(1 - A_clamped)  # [K, T] logits
                    logit_A = logit_A_transpose.transpose(-1, -2)  # [T, K]

                    # Compute logits for each category
                    logits_K = logit_A.unsqueeze(0) + alpha.unsqueeze(0).unsqueeze(-1) * poly_val  # [N, T, K]

                    # Apply softmax to get probabilities that sum to 1
                    y_dose_response = torch.softmax(logits_K, dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: single set of coefficients per feature
                    # Stack polynomial coefficients
                    coeffs = []
                    for d in range(1, polynomial_degree + 1):  # start at degree 1, no intercept
                        coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))
                        coeffs.append(coeff)
                    coeffs = torch.stack(coeffs, dim=-2)  # [degree, T]
                    if (coeffs.shape[1] == 1) & (coeffs.ndim == 4):
                        coeffs = coeffs.squeeze(1)        # [S, D, T]

                    # Distribution-specific polynomial computation:
                    if distribution in ['normal', 'studentt']:
                        # For continuous distributions: work in natural space (no logs!)
                        # Polynomial is applied to x_true directly: y = A + alpha * poly(x)
                        poly_val = Polynomial_function(x_true_sample, coeffs)  # [N, T]
                        y_dose_response = A + alpha * poly_val  # [N, T] - can be negative!

                    elif distribution in ['negbinom']:
                        # For negbinom: work in log space
                        # Polynomial is applied to log2(x): log2(y) = log2(A) + alpha * poly(log2(x))
                        log2_x_true = torch.log2(x_true_sample)  # [N]
                        poly_val = Polynomial_function(log2_x_true, coeffs)  # [N, T]
                        log2_y_dose_response = torch.log2(A) + alpha * poly_val  # [N, T]
                        y_dose_response = 2 ** log2_y_dose_response  # Convert back to count space

                    elif distribution == 'binomial':
                        # For binomial: work in LOGIT space (unbounded: -inf to +inf)
                        # A is in [0,1] from Beta prior, convert to logit
                        # Polynomial is applied to x_true: logit(p) = logit(A) + alpha * poly(x)
                        A_clamped = torch.clamp(A, min=epsilon_tensor, max=1.0 - epsilon_tensor)
                        logit_A = torch.log(A_clamped) - torch.log(1 - A_clamped)  # logit(A)
                        poly_val = Polynomial_function(x_true_sample, coeffs)  # [N, T]
                        logit_p = logit_A + alpha * poly_val  # [N, T] - logit space (unbounded)
                        # Convert back to probability space for sampler
                        y_dose_response = torch.sigmoid(logit_p)  # [N, T] in [0, 1]

                    else:
                        raise ValueError(f"Unknown distribution for polynomial: {distribution}")
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
            

        ##########################
        ## Cell-level variables ##
        ##########################
        # At this point, y_dose_response contains the dose-response function output.
        # We need to transform it to the format expected by samplers.
        #
        # IMPORTANT: Samplers in distributions.py handle technical group effects themselves!
        # Do NOT apply alpha_y here - pass it to the sampler via alpha_y_full.

        # Prepare alpha_y_full (full C technical groups, including reference)
        # This will be passed to samplers, which apply technical groups themselves
        # Note: Multinomial sampler doesn't currently support technical groups, skip for now
        if alpha_y is not None and groups_tensor is not None and distribution != 'multinomial':
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

        # Transform dose-response to sampler-expected format
        # (samplers will handle technical groups and normalization)
        if distribution == 'negbinom':
            # Sampler expects: mu_y = dose-response in count space (NO technical groups, NO sum factors)
            # Sampler will apply: mu_final = mu_y * alpha_y * sum_factor
            mu_y = y_dose_response  # [N, T] - just the dose-response

        elif distribution == 'binomial':
            # Sampler expects: mu_y = probability in [0, 1]
            # y_dose_response should already be a probability from Hill/polynomial
            # Sampler will apply technical groups on logit scale
            mu_y = y_dose_response  # [N, T] - already probability

        elif distribution == 'multinomial':
            # Sampler expects: mu_y = baseline probabilities [N, T, K]
            # y_dose_response is already [N, T, K] probabilities that sum to 1
            # Sampler will apply technical groups on logit scale per category
            mu_y = y_dose_response  # [N, T, K] - probabilities per category

        elif distribution in ['normal', 'studentt']:
            # Sampler expects: mu_y = natural value space
            # Sampler will apply technical groups additively
            mu_y = y_dose_response  # [N, T] - can be negative!

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Debug checks (keep for troubleshooting)
        if torch.isnan(mu_y).any() or torch.isinf(mu_y).any():
            check_tensor("mu_y", mu_y)
            check_tensor("y_dose_response", y_dose_response)
            check_tensor("sum_factor_tensor", sum_factor_tensor)
            check_tensor("phi_y_used", phi_y_used)
            check_tensor("A", A)
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                check_tensor("n_a", n_a)
                check_tensor("Vmax_a", Vmax_a)
                check_tensor("K_a", K_a)
            if function_type in ['additive_hill', 'nested_hill']:
                check_tensor("n_b", n_b)
                check_tensor("Vmax_b", Vmax_b)
                check_tensor("K_b", K_b)
            if function_type == 'polynomial':
                check_tensor("coeffs", coeffs)

        # Call distribution-specific observation sampler
        from .distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')

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
                alpha_y_full=alpha_y_full,  # Currently None for multinomial (not yet implemented)
                groups_tensor=groups_tensor,
                N=N,
                T=T,
                K=K,
                C=C
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
        elif distribution == 'studentt':
            # For studentt, we need sigma_y (standard deviation) and nu_y (degrees of freedom)
            sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
            observation_sampler(
                y_obs_tensor=y_obs_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                nu_y=nu_y,
                alpha_y_full=alpha_y_full,
                groups_tensor=groups_tensor,
                N=N,
                T=T,
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
        lr: float = None,
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
        p_n: float = 1e-6,
        epsilon: float = 1e-6,
        init_temp: float = 1.0,
        #final_temp: float = 1e-8,
        final_temp: float = 0.1,
        minibatch_size: int = None,
        distribution: str = None,
        denominator: np.ndarray = None,
        modality_name: str = None,
        min_denominator: int = None,
        use_data_driven_priors: bool = True,
        **kwargs
    ):
        """
        Fit trans effects using distribution-specific likelihood.

        Parameters
        ----------
        modality_name : str, optional
            Name of modality to fit. If None, uses primary modality.
        distribution : str, optional
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal'
            If None, auto-detected from modality.
        sum_factor_col : str, optional
            Sum factor column name. Required for negbinom, ignored for others.
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (e.g., total counts for PSI)
            If None, auto-detected from modality.
        min_denominator : int, optional
            Minimum denominator value for binomial observations. Observations where
            denominator < min_denominator are masked (excluded from fitting).
            Useful for filtering low-coverage splicing junctions. Default: None (no filtering).
        use_data_driven_priors : bool, optional
            If True (default), use Beta priors for A and upper_limit based on data percentiles.
            If False, use uniform priors (Beta(1, 1)). Useful for testing if data-driven
            priors are too strong or causing issues. Default: True.
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

        >>> # Test without data-driven priors
        >>> model.fit_trans(sum_factor_col='sum_factor', function_type='additive_hill',
        ...                 use_data_driven_priors=False)
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
            # Default: 100,000 unless multinomial OR polynomial function, then 200,000
            if distribution == 'multinomial':
                niters = 200_000
                print(f"[INFO] Using default niters=200,000 for multivariate distribution '{distribution}'")
            elif function_type == 'polynomial':
                niters = 200_000
                print(f"[INFO] Using default niters=200,000 for polynomial function")
            else:
                niters = 100_000
                print(f"[INFO] Using default niters=100,000 for distribution '{distribution}' and function_type '{function_type}'")
        
        #if lr is None:
        #    # Default: 100,000 unless multinomial OR polynomial function, then 200,000
        #    if distribution in ['binomial', 'multinomial']:
        #        lr = 1e-4
        #        print(f"[INFO] Using default niters=200,000 for distribution '{distribution}'")
        #    else:
        #        lr = 1e-3
        #        print(f"[INFO] Using default lr=1e-3 for distribution '{distribution}'")

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

        # Validate min_denominator is specified for binomial/multinomial
        if distribution in ['binomial', 'multinomial'] and min_denominator is None:
            raise ValueError(
                f"min_denominator is required for distribution='{distribution}'. "
                f"Please specify min_denominator (e.g., min_denominator=0 for no filtering, "
                f"or min_denominator=3 for standard quality filtering)."
            )

        # Validate distribution-specific requirements
        from .distributions import requires_sum_factor, requires_denominator

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

            # Apply min_denominator filter if specified
            if min_denominator is not None and min_denominator > 0:
                # Create mask for observations where denominator < threshold
                low_coverage_mask = denominator_tensor < min_denominator
                n_masked = low_coverage_mask.sum().item()
                n_total = denominator_tensor.numel()
                pct_masked = 100 * n_masked / n_total if n_total > 0 else 0

                print(f"[INFO] Filtering observations with denominator < {min_denominator}")
                print(f"[INFO] Masked {n_masked}/{n_total} observations ({pct_masked:.1f}%)")

                # For binomial distributions, we'll pass the mask to the model
                # The sampler will need to handle it (for now, set those observations to special value)
                # We'll use a very negative value that the sampler can detect
                # Actually, better approach: modify y_obs to have NaN for masked observations
                # But binomial doesn't support NaN observations...
                # Best approach: set denominator=0 for masked observations, and sampler handles it
                denominator_tensor = torch.where(low_coverage_mask,
                                                 torch.zeros_like(denominator_tensor),
                                                 denominator_tensor)

        # Detect data dimensions (for multinomial)
        from .distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs.shape[2]  # Number of categories
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

        # Distribution-specific normalization for data-driven priors
        # For building priors later:
        y_obs_for_prior = None

        # Note: min_denominator is now required for binomial/multinomial (validated earlier)
        # For negbinom/normal/studentt, it's not used so None is fine

        if distribution == 'binomial' and denominator_tensor is not None:
            # Full probabilities for the likelihood
            y_obs_factored = y_obs_tensor / denominator_tensor.clamp_min(epsilon_tensor)

            # For priors: only use entries where denominator >= min_denominator
            valid_mask = (denominator_tensor >= min_denominator)
            # Mark invalid entries as NaN so we can ignore them in means
            y_obs_for_prior = torch.where(
                valid_mask,
                y_obs_factored,
                torch.full_like(y_obs_factored, float('nan'))
            )
            print(f"[INFO] Binomial: using {valid_mask.float().mean().item()*100:.1f}% of entries with denominator >= {min_denominator} for priors")

        elif distribution == 'multinomial' and y_obs_tensor.ndim == 3:
            total_counts = y_obs_tensor.sum(dim=-1, keepdim=True).clamp_min(epsilon_tensor)  # [N, T, 1]
            y_obs_factored = y_obs_tensor / total_counts  # [N, T, K]

            # For priors: only use entries where total_counts >= min_denominator
            valid_mask = (total_counts >= min_denominator)  # [N, T, 1]
            y_obs_for_prior = torch.where(
                valid_mask,
                y_obs_factored,
                torch.full_like(y_obs_factored, float('nan'))
            )
            print(f"[INFO] Multinomial: using {valid_mask.float().mean().item()*100:.1f}% of entries with total counts >= {min_denominator} for priors")

        elif sum_factor_col is not None:
            y_obs_factored = y_obs_tensor / sum_factor_tensor.view(-1, 1)
            y_obs_for_prior = y_obs_factored
        else:
            y_obs_factored = y_obs_tensor
            y_obs_for_prior = y_obs_factored

        # ===================================================================
        # CORRECT FOR TECHNICAL EFFECTS BEFORE COMPUTING PRIORS
        # ===================================================================
        # The observed data includes technical batch effects. To compute unbiased
        # priors for A and Vmax (baseline parameters), we need to remove these effects
        # using the inverse transformation.
        if alpha_y_prefit is not None and groups_tensor is not None:
            print(f"[INFO] Correcting for technical effects before computing priors (distribution: {distribution})")

            if distribution == 'negbinom':
                # Technical effect: multiplicative (mu_corrected = mu * alpha_y_mult)
                # Inverse: divide by alpha_y_mult to get baseline
                # alpha_y_prefit for negbinom is multiplicative (from fit_technical)
                alpha_y_mult_expanded = alpha_y_prefit[groups_tensor, :]  # [N, T]
                y_obs_for_prior = y_obs_for_prior / alpha_y_mult_expanded.clamp_min(epsilon_tensor)
                print(f"[INFO] negbinom: Applied inverse multiplicative correction (divide by alpha_y_mult)")

            elif distribution in ['normal', 'studentt']:
                # Technical effect: additive (mu_corrected = mu + alpha_y_add)
                # Inverse: subtract alpha_y_add to get baseline
                # alpha_y_prefit for normal/studentt is additive (from fit_technical)
                alpha_y_add_expanded = alpha_y_prefit[groups_tensor, :]  # [N, T]
                y_obs_for_prior = y_obs_for_prior - alpha_y_add_expanded
                print(f"[INFO] {distribution}: Applied inverse additive correction (subtract alpha_y_add)")

            elif distribution == 'binomial':
                # Technical effect: logit scale (logit(p_corrected) = logit(p) + alpha_y_add)
                # Inverse: logit(p_baseline) = logit(p_observed) - alpha_y_add
                # Then: p_baseline = sigmoid(logit(p_baseline))
                alpha_y_add_expanded = alpha_y_prefit[groups_tensor, :]  # [N, T]

                # Convert observed proportions to logit scale
                p_obs_clamped = torch.clamp(y_obs_for_prior, min=epsilon_tensor, max=1.0 - epsilon_tensor)
                logit_obs = torch.log(p_obs_clamped) - torch.log(1.0 - p_obs_clamped)

                # Apply inverse correction on logit scale
                logit_baseline = logit_obs - alpha_y_add_expanded

                # Convert back to probability scale
                y_obs_for_prior = torch.sigmoid(logit_baseline)
                print(f"[INFO] binomial: Applied inverse logit correction (subtract alpha_y_add on logit scale)")

            elif distribution == 'multinomial':
                # Technical effect: log scale (log(probs_corrected) = log(probs) + alpha_y_add)
                # Inverse: log(probs_baseline) = log(probs_observed) - alpha_y_add
                # Then: probs_baseline = exp(log_probs_baseline) / sum(exp(...))
                alpha_y_add_expanded = alpha_y_prefit[groups_tensor, :, :]  # [N, T, K]

                # Convert observed proportions to log scale
                p_obs_clamped = torch.clamp(y_obs_for_prior, min=epsilon_tensor)
                log_probs_obs = torch.log(p_obs_clamped)

                # Apply inverse correction on log scale
                log_probs_baseline = log_probs_obs - alpha_y_add_expanded

                # Normalize (softmax) to get valid probabilities
                y_obs_for_prior = torch.softmax(log_probs_baseline, dim=-1)
                print(f"[INFO] multinomial: Applied inverse log correction (subtract alpha_y_add on log scale)")

            # Handle any NaN or invalid values that may result from correction
            # (e.g., if correction pushes values outside valid range)
            if not torch.isfinite(y_obs_for_prior).all():
                n_invalid = (~torch.isfinite(y_obs_for_prior)).sum().item()
                n_total = y_obs_for_prior.numel()
                print(f"[WARNING] Technical correction produced {n_invalid}/{n_total} "
                      f"({100*n_invalid/n_total:.2f}%) non-finite values. "
                      f"These will be excluded from prior computation.")
                # Mark invalid values as NaN so they're excluded by nanmean/nanvar
                y_obs_for_prior = torch.where(
                    torch.isfinite(y_obs_for_prior),
                    y_obs_for_prior,
                    torch.full_like(y_obs_for_prior, float('nan'))
                )
        else:
            if alpha_y_prefit is not None:
                print(f"[INFO] alpha_y_prefit provided but groups_tensor is None - skipping technical correction")

        unique_guides = torch.unique(guides_tensor)

        # nanmean and nanvar helpers (in case torch.nanmean/nanvar isn't available)
        if hasattr(torch, "nanmean"):
            def nanmean(x, dim):
                return torch.nanmean(x, dim=dim)
        else:
            def nanmean(x, dim):
                mask = ~torch.isnan(x)
                num = torch.where(mask, x, torch.zeros_like(x)).sum(dim=dim)
                den = mask.sum(dim=dim).clamp_min(1)
                return num / den

        if hasattr(torch, "nanvar"):
            def nanvar(x, dim):
                return torch.var(x, dim=dim)  # Note: torch.var ignores NaN in older versions
        else:
            def nanvar(x, dim):
                mask = ~torch.isnan(x)
                n = mask.sum(dim=dim).clamp_min(2)  # Need at least 2 values for variance
                x_mean = nanmean(x, dim=dim)
                # Expand mean to match x shape for broadcasting
                if dim == 0:
                    x_mean_expanded = x_mean.unsqueeze(0)
                else:
                    x_mean_expanded = x_mean
                sq_diff = torch.where(mask, (x - x_mean_expanded) ** 2, torch.zeros_like(x))
                return sq_diff.sum(dim=dim) / (n - 1)  # Unbiased variance

        # nanquantile helper
        def nanquantile(x, q, dim):
            """Compute quantile ignoring NaN values."""
            if x.ndim == 2:  # [G, T]
                result = []
                for t in range(x.shape[1]):
                    vals = x[:, t]
                    valid = vals[~torch.isnan(vals)]
                    if valid.numel() > 0:
                        result.append(torch.quantile(valid, q))
                    else:
                        result.append(torch.tensor(float('nan'), device=x.device))
                return torch.stack(result)  # [T]
            elif x.ndim == 3:  # [G, T, K]
                result = []
                for t in range(x.shape[1]):
                    result_k = []
                    for k in range(x.shape[2]):
                        vals = x[:, t, k]
                        valid = vals[~torch.isnan(vals)]
                        if valid.numel() > 0:
                            result_k.append(torch.quantile(valid, q))
                        else:
                            result_k.append(torch.tensor(float('nan'), device=x.device))
                    result.append(torch.stack(result_k))
                return torch.stack(result)  # [T, K]
            else:
                raise ValueError(f"Unsupported ndim for nanquantile: {x.ndim}")

        # Compute priors: use percentiles from guide means (works with or without guides)
        if 'guide_code' in self.model.meta.columns and len(unique_guides) > 1:
            # WITH GUIDES: Compute guide-level statistics
            guide_means = []
            guide_vars = []
            for g in unique_guides:
                vals_g = y_obs_for_prior[guides_tensor == g, ...]  # [Ng, T] or [Ng, T, K]
                guide_means.append(nanmean(vals_g, dim=0))         # [T] or [T, K]
                guide_vars.append(nanvar(vals_g, dim=0))           # [T] or [T, K]

            guide_means = torch.stack(guide_means, dim=0)  # [G, T] or [G, T, K]
            guide_vars = torch.stack(guide_vars, dim=0)    # [G, T] or [G, T, K]

            # For A and Vmax: use 5th and 95th percentiles of guide means
            Amean_tensor = nanquantile(guide_means, 0.05, dim=0)  # [T] or [T, K]
            Vmax_mean_tensor = nanquantile(guide_means, 0.95, dim=0)  # [T] or [T, K]

            # Ensure A_mean >= 1e-3
            Amean_tensor = Amean_tensor.clamp_min(self._t(1e-3))

            # For variances: use mean variance across guides (average within-guide variability)
            # This captures how much variability exists around each guide's mean
            mean_within_guide_var = nanmean(guide_vars, dim=0)  # [T] or [T, K]

            print(f"[INFO] Using guide-based priors: {len(unique_guides)} guides, 5th/95th percentiles")

        else:
            # WITHOUT GUIDES: Use overall percentiles
            print("[INFO] No guides found or only 1 guide. Using overall data percentiles for priors.")

            # Compute 5th and 95th percentiles across all cells for each feature
            if y_obs_for_prior.ndim == 2:  # [N, T]
                Amean_tensor = []
                Vmax_mean_tensor = []
                overall_vars = []
                for t in range(y_obs_for_prior.shape[1]):
                    vals_t = y_obs_for_prior[:, t]
                    valid_t = vals_t[~torch.isnan(vals_t)]
                    if valid_t.numel() > 0:
                        Amean_tensor.append(torch.quantile(valid_t, 0.05))
                        Vmax_mean_tensor.append(torch.quantile(valid_t, 0.95))
                        overall_vars.append(torch.var(valid_t))
                    else:
                        Amean_tensor.append(torch.tensor(float('nan'), device=self.model.device))
                        Vmax_mean_tensor.append(torch.tensor(float('nan'), device=self.model.device))
                        overall_vars.append(torch.tensor(float('nan'), device=self.model.device))

                Amean_tensor = torch.stack(Amean_tensor)  # [T]
                Vmax_mean_tensor = torch.stack(Vmax_mean_tensor)  # [T]
                mean_within_guide_var = torch.stack(overall_vars)  # [T]

            elif y_obs_for_prior.ndim == 3:  # [N, T, K]
                Amean_tensor = []
                Vmax_mean_tensor = []
                overall_vars = []
                for t in range(y_obs_for_prior.shape[1]):
                    Amean_k = []
                    Vmax_k = []
                    var_k = []
                    for k in range(y_obs_for_prior.shape[2]):
                        vals_tk = y_obs_for_prior[:, t, k]
                        valid_tk = vals_tk[~torch.isnan(vals_tk)]
                        if valid_tk.numel() > 0:
                            Amean_k.append(torch.quantile(valid_tk, 0.05))
                            Vmax_k.append(torch.quantile(valid_tk, 0.95))
                            var_k.append(torch.var(valid_tk))
                        else:
                            Amean_k.append(torch.tensor(float('nan'), device=self.model.device))
                            Vmax_k.append(torch.tensor(float('nan'), device=self.model.device))
                            var_k.append(torch.tensor(float('nan'), device=self.model.device))
                    Amean_tensor.append(torch.stack(Amean_k))
                    Vmax_mean_tensor.append(torch.stack(Vmax_k))
                    overall_vars.append(torch.stack(var_k))

                Amean_tensor = torch.stack(Amean_tensor)  # [T, K]
                Vmax_mean_tensor = torch.stack(Vmax_mean_tensor)  # [T, K]
                mean_within_guide_var = torch.stack(overall_vars)  # [T, K]

            # Ensure A_mean >= 1e-3
            Amean_tensor = Amean_tensor.clamp_min(self._t(1e-3))

        # For binomial/multinomial: clamp Vmax_mean to valid Beta range
        if distribution in ['binomial', 'multinomial']:
            Vmax_mean_tensor = Vmax_mean_tensor.clamp(min=self._t(1e-3), max=self._t(1.0 - 1e-6))
            Amean_tensor = Amean_tensor.clamp(min=self._t(1e-3), max=self._t(1.0 - 1e-6))

        # Handle NaN values (features where ALL observations were filtered out)
        # For binomial/multinomial with denominator filtering, some features may have no valid observations
        nan_mask = torch.isnan(Amean_tensor) | torch.isnan(Vmax_mean_tensor)
        if nan_mask.any():
            n_nan = nan_mask.sum().item() if nan_mask.ndim == 1 else nan_mask.any(dim=-1).sum().item()
            print(f"[WARNING] {n_nan} features have all observations filtered (denominator < {min_denominator}). Using fallback values.")

            # Fallback: use global mean/var across all valid features
            if distribution in ['binomial', 'multinomial']:
                # Compute fallback from valid (non-NaN) entries only
                valid_means = guide_means[~torch.isnan(guide_means)]
                valid_vars = guide_vars[~torch.isnan(guide_vars)]

                if valid_means.numel() > 0:
                    # Use median for robustness
                    valid_mean = torch.median(valid_means)
                    valid_var = torch.median(valid_vars) if valid_vars.numel() > 0 else self._t(0.1)
                else:
                    # Last resort: use generic defaults for PSI data
                    print("[WARNING] No valid observations found! Using generic defaults: A=0.1, Vmax=0.5, var=0.1")
                    valid_mean = self._t(0.3)  # Midpoint for PSI
                    valid_var = self._t(0.1)  # Reasonable variance for PSI

                print(f"[INFO] Using fallback: mean={valid_mean.item():.3f}, var={valid_var.item():.3f}")

                # Replace NaN with fallback
                fallback_A = valid_mean * 0.5  # A = 50% of mean (minimum)
                fallback_Vmax = valid_mean * 1.5  # Vmax = 150% of mean (maximum)
                fallback_A = torch.clamp(fallback_A, min=0.01, max=0.99)  # Keep in valid Beta range
                fallback_Vmax = torch.clamp(fallback_Vmax, min=0.01, max=0.99)

                Amean_tensor = torch.where(torch.isnan(Amean_tensor),
                                          torch.full_like(Amean_tensor, fallback_A),
                                          Amean_tensor)
                Vmax_mean_tensor = torch.where(torch.isnan(Vmax_mean_tensor),
                                              torch.full_like(Vmax_mean_tensor, fallback_Vmax),
                                              Vmax_mean_tensor)
                mean_within_guide_var = torch.where(torch.isnan(mean_within_guide_var),
                                                   torch.full_like(mean_within_guide_var, valid_var),
                                                   mean_within_guide_var)

            else:  # negbinom, normal, studentt
                # Compute fallback from valid (non-NaN) entries only
                valid_means = guide_means[~torch.isnan(guide_means)]
                valid_vars = guide_vars[~torch.isnan(guide_vars)]

                if valid_means.numel() > 0:
                    # Use median for robustness
                    valid_mean = torch.median(valid_means)
                    valid_var = torch.median(valid_vars) if valid_vars.numel() > 0 else valid_mean * 0.5
                else:
                    # Last resort: use generic defaults for count/continuous data
                    print("[WARNING] No valid observations found! Using generic defaults: A=1.0, Vmax=10.0, var=5.0")
                    valid_mean = self._t(5.0)  # Midpoint for log-space counts
                    valid_var = self._t(5.0)  # Reasonable variance

                print(f"[INFO] Using fallback: mean={valid_mean.item():.3f}, var={valid_var.item():.3f}")

                # Replace NaN with fallback
                fallback_A = valid_mean * 0.5  # A = 50% of mean (minimum)
                fallback_Vmax = valid_mean * 1.5  # Vmax = 150% of mean (maximum)
                fallback_A = torch.clamp(fallback_A, min=1e-3)  # Ensure positive
                fallback_Vmax = torch.clamp(fallback_Vmax, min=1e-3)

                Amean_tensor = torch.where(torch.isnan(Amean_tensor),
                                          torch.full_like(Amean_tensor, fallback_A),
                                          Amean_tensor)
                Vmax_mean_tensor = torch.where(torch.isnan(Vmax_mean_tensor),
                                              torch.full_like(Vmax_mean_tensor, fallback_Vmax),
                                              Vmax_mean_tensor)
                mean_within_guide_var = torch.where(torch.isnan(mean_within_guide_var),
                                                   torch.full_like(mean_within_guide_var, valid_var),
                                                   mean_within_guide_var)

        # For K: use CV (coefficient of variation) of x_true (works with or without guides)
        # CV = std(x_true) / mean(x_true) - scale-invariant measure of variability
        x_true_mean_global = x_true_mean.mean()
        x_true_std_global = x_true_mean.std()
        x_true_CV = x_true_std_global / x_true_mean_global.clamp_min(epsilon_tensor)

        # K_max: max of x_true (or max of guide means if guides exist)
        if 'guide_code' in self.model.meta.columns and len(unique_guides) > 1:
            guide_x_means = torch.stack([x_true_mean[guides_tensor == g].mean() for g in torch.unique(guides_tensor)])
            K_max_tensor = guide_x_means.max()
        else:
            K_max_tensor = x_true_mean.max()

        print("[DEBUG] Amean:", Amean_tensor.min().item(), Amean_tensor.max().item())
        print("[DEBUG] Vmax_mean:", Vmax_mean_tensor.min().item(), Vmax_mean_tensor.max().item())
        print("[DEBUG] Mean within-guide variance:", mean_within_guide_var.min().item(), mean_within_guide_var.max().item())
        print("[DEBUG] x_true CV:", x_true_CV.item(), "K_max:", K_max_tensor.item())

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
        
        from torch.optim.lr_scheduler import OneCycleLR
        if function_type == "polynomial":
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
                Amean_tensor,
                p_n_tensor,
                epsilon_tensor,
                x_true_sample = self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true,
                log2_x_true_sample = self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit,
                C = C,
                groups_tensor=groups_tensor,
                temperature=torch.tensor(init_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=False,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
                mean_within_guide_var=mean_within_guide_var,
                x_true_CV=x_true_CV,
                use_data_driven_priors=use_data_driven_priors,
            )
        else:
            guide_y = pyro.infer.autoguide.AutoNormalMessenger(self._model_y)

        # -------------------------------
        # Shared OneCycleLR scheduler for ALL function types
        # -------------------------------
        # Use lr if the user passed it, otherwise default to 1e-3
        base_lr = 1e-3 if lr is None else lr
        
        optimizer = pyro.optim.PyroLRScheduler(
            scheduler_constructor=OneCycleLR,
            optim_args={
                # underlying torch optimizer
                "optimizer": torch.optim.Adam,
                "optim_args": {
                    "lr": base_lr,      # initial lr (OneCycle will move it)
                    "betas": (0.9, 0.999),
                },
                # OneCycleLR hyperparameters
                "max_lr":          base_lr * 10,  # was 1e-2 when base_lr=1e-3
                "total_steps":     niters,
                "pct_start":       0.1,
                "div_factor":      25.0,
                "final_div_factor": 1e4,
            },
            clip_args={"clip_norm": 5},  # same clipping everywhere
        )
        
        svi = pyro.infer.SVI(
            self._model_y,
            guide_y,
            optimizer,
            loss=pyro.infer.Trace_ELBO()
        )

        
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
                Amean_tensor,
                p_n_tensor,
                epsilon_tensor,
                x_true_sample = x_true_sample,
                log2_x_true_sample = log2_x_true_sample,
                nmin = nmin,
                nmax = nmax,
                alpha_y_sample = alpha_y_sample,
                C = C,
                groups_tensor=groups_tensor,
                temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.model.device),
                use_straight_through=use_straight_through,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True if function_type != 'polynomial' else True if fraction_done>=0.5 else False,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
                mean_within_guide_var=mean_within_guide_var,
                x_true_CV=x_true_CV,
                use_data_driven_priors=use_data_driven_priors,
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
                "Amean_tensor": self._to_cpu(Amean_tensor),
                "p_n_tensor": self._to_cpu(p_n_tensor),
                "epsilon_tensor": self._to_cpu(epsilon_tensor),
                "x_true_sample": self._to_cpu(self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true),
                "log2_x_true_sample": self._to_cpu(self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true),
                "nmin": self._to_cpu(nmin),
                "nmax": self._to_cpu(nmax),
                # Only move if not None:
                "alpha_y_sample": self._to_cpu(alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit) if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": self._to_cpu(groups_tensor) if groups_tensor is not None else None,
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
                "mean_within_guide_var": self._to_cpu(mean_within_guide_var) if mean_within_guide_var is not None else None,
                "x_true_CV": self._to_cpu(x_true_CV) if x_true_CV is not None else None,
                "use_data_driven_priors": use_data_driven_priors,
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
                "Amean_tensor": Amean_tensor,
                "p_n_tensor": p_n_tensor,
                "epsilon_tensor": epsilon_tensor,
                "x_true_sample": self.model.x_true.mean(dim=0) if self.model.x_true_type == "posterior" else self.model.x_true,
                "log2_x_true_sample": self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true,
                "nmin": nmin,
                "nmax": nmax,
                "alpha_y_sample": alpha_y_prefit.mean(dim=0) if alpha_y_type == "posterior" else alpha_y_prefit if alpha_y_prefit is not None else None,
                "C": C,
                "groups_tensor": groups_tensor if groups_tensor is not None else None,
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=self.model.device),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": denominator_tensor if denominator_tensor is not None else None,
                "K": K,
                "D": D,
                "mean_within_guide_var": mean_within_guide_var,
                "x_true_CV": x_true_CV,
                "use_data_driven_priors": use_data_driven_priors,
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

        # ----------------------------------------
        # Store nmin/nmax and check boundaries
        # ----------------------------------------
        posterior_samples_y['nmin'] = self._to_cpu(nmin)
        posterior_samples_y['nmax'] = self._to_cpu(nmax)

        # Warn if fitted n_a parameters are close to boundaries
        if 'n_a' in posterior_samples_y and function_type in ['single_hill', 'additive_hill', 'nested_hill']:
            n_a_samples = posterior_samples_y['n_a']  # Shape: [S, T]
            nmin_val = nmin.item()
            nmax_val = nmax.item()

            # Define "close to boundary" threshold (e.g., within 10% of range)
            boundary_threshold = 0.1 * (nmax_val - nmin_val)

            # Check how many samples are close to boundaries
            close_to_min = (n_a_samples < (nmin_val + boundary_threshold)).float().mean().item()
            close_to_max = (n_a_samples > (nmax_val - boundary_threshold)).float().mean().item()

            # Warn if >10% of samples are at boundaries
            if close_to_min > 0.1:
                warnings.warn(
                    f"[WARNING] {close_to_min*100:.1f}% of n_a samples are close to lower boundary (nmin={nmin_val:.2f}). "
                    f"Consider: (1) checking if x_true range is appropriate, or (2) relaxing nmin constraint.",
                    UserWarning
                )
            if close_to_max > 0.1:
                warnings.warn(
                    f"[WARNING] {close_to_max*100:.1f}% of n_a samples are close to upper boundary (nmax={nmax_val:.2f}). "
                    f"Consider: (1) checking if x_true range is appropriate, or (2) relaxing nmax constraint.",
                    UserWarning
                )

            # Summary statistics
            print(f"[INFO] n_a boundary check: nmin={nmin_val:.2f}, nmax={nmax_val:.2f}")
            print(f"[INFO]   {close_to_min*100:.1f}% of samples near lower bound, {close_to_max*100:.1f}% near upper bound")

        # Store results
        # Store in modality
        modality.posterior_samples_trans = posterior_samples_y

        # Update alpha_y_prefit in modality if it was None and alpha_y was sampled
        if modality.alpha_y_prefit is None and groups_tensor is not None and "alpha_y" in posterior_samples_y:
            modality.alpha_y_prefit = posterior_samples_y["alpha_y"].mean(dim=0)

        # If primary modality, also store at model level (backward compatibility)
        if modality_name == self.model.primary_modality:
            self.model.posterior_samples_trans = posterior_samples_y
            if self.model.alpha_y_prefit is None and groups_tensor is not None and "alpha_y" in posterior_samples_y:
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



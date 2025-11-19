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

            # For multinomial, reduce category dimension to get per-feature baseline
            if distribution == 'multinomial' and Amean_tensor.ndim > 1:
                Amean_for_A = Amean_tensor.mean(dim=-1)  # [T, K] -> [T]
                Vmax_for_A = Vmax_mean_tensor.mean(dim=-1)  # [T, K] -> [T]
            else:
                Amean_for_A = Amean_tensor
                Vmax_for_A = Vmax_mean_tensor

            Amean_adjusted = ((1 - weight) * Amean_for_A) + (weight * Vmax_for_A) + epsilon_tensor

            # Baseline parameter A depends on distribution:
            # - normal/studentt: can be negative (natural value space)
            # - negbinom: positive count
            # - binomial/multinomial: probability in [0,1]
            if distribution in ['normal', 'studentt']:
                # Use Normal distribution to allow negative baseline values
                A = pyro.sample("A", dist.Normal(Amean_adjusted, Amean_adjusted.abs()))
            elif distribution in ['binomial', 'multinomial']:
                # For probability distributions, A should be in [0, 1]
                # Use Beta distribution to constrain to [0,1]
                # Transform Amean to Beta parameters
                Amean_clamped = Amean_adjusted.clamp(min=0.01, max=0.99)
                # Use concentration parameters that give mean = Amean_clamped with moderate variance
                # alpha = mean * concentration, beta = (1-mean) * concentration
                concentration = self._t(10.0)  # Higher = tighter around mean
                alpha_beta = Amean_clamped * concentration
                beta_beta = (1 - Amean_clamped) * concentration
                A = pyro.sample("A", dist.Beta(alpha_beta, beta_beta))
            else:
                # For negbinom: A must be positive count
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

                # Vmax_a and K_a: Distribution-specific priors
                if distribution == 'multinomial' and K is not None:
                    # For multinomial, each of K-1 categories gets its own Vmax and K
                    K_minus_1 = K - 1
                    with pyro.plate("category_plate_vmax", K_minus_1, dim=-2):
                        # Vmax for probability distributions should be in [0, 1]
                        # Use Vmax_prior_mean which is [T] (averaged across categories)
                        Vmax_mean_clamped = Vmax_prior_mean.clamp(min=0.01, max=0.99)
                        concentration_vmax = self._t(10.0)
                        alpha_vmax = Vmax_mean_clamped.unsqueeze(-2) * concentration_vmax  # [1, T]
                        beta_vmax = (1 - Vmax_mean_clamped.unsqueeze(-2)) * concentration_vmax  # [1, T]
                        Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax, beta_vmax))  # [K-1, T]

                        # K_max_tensor and K_sigma are scalars - broadcast automatically
                        K_a = pyro.sample("K_a", dist.Gamma(
                            ((K_max_tensor/2) ** 2) / (K_sigma ** 2),
                            (K_max_tensor/2) / (K_sigma ** 2)
                        ))  # [K-1, T]

                elif distribution == 'binomial':
                    # For binomial, Vmax should be in [0, 1]
                    Vmax_mean_clamped = Vmax_mean_tensor.clamp(min=0.01, max=0.99)
                    concentration_vmax = self._t(10.0)
                    alpha_vmax = Vmax_mean_clamped * concentration_vmax
                    beta_vmax = (1 - Vmax_mean_clamped) * concentration_vmax
                    Vmax_a = pyro.sample("Vmax_a", dist.Beta(alpha_vmax, beta_vmax))
                    K_a = pyro.sample("K_a", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))

                else:
                    # For count/continuous distributions, use Gamma prior
                    Vmax_a = pyro.sample("Vmax_a", dist.Gamma((Vmax_mean_tensor ** 2) / (Vmax_sigma ** 2), Vmax_mean_tensor / (Vmax_sigma ** 2)))
                    K_a = pyro.sample("K_a", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))

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

                    # Vmax_b and K_b: Distribution-specific priors
                    if distribution == 'multinomial' and K is not None:
                        K_minus_1 = K - 1
                        with pyro.plate("category_plate_vmax_b", K_minus_1, dim=-2):
                            # Reuse alpha_vmax and beta_vmax from Vmax_a (same prior mean)
                            Vmax_b = pyro.sample("Vmax_b", dist.Beta(alpha_vmax, beta_vmax))  # [K-1, T]
                            # K_max_tensor and K_sigma are scalars - broadcast automatically
                            K_b = pyro.sample("K_b", dist.Gamma(
                                ((K_max_tensor/2) ** 2) / (K_sigma ** 2),
                                (K_max_tensor/2) / (K_sigma ** 2)
                            ))  # [K-1, T]

                    elif distribution == 'binomial':
                        Vmax_b = pyro.sample("Vmax_b", dist.Beta(alpha_vmax, beta_vmax))
                        K_b = pyro.sample("K_b", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))

                    else:
                        Vmax_b = pyro.sample("Vmax_b", dist.Gamma((Vmax_mean_tensor ** 2) / (Vmax_sigma ** 2), Vmax_mean_tensor / (Vmax_sigma ** 2)))
                        K_b = pyro.sample("K_b", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))
                
                # Compute Hill function(s)
                # Hill_based_positive returns values in [0, Vmax]
                # We compute: y = A + alpha * Hill + beta * Hill (for additive)

                if distribution == 'multinomial' and K is not None:
                    # For multinomial: compute K-1 independent Hill functions
                    # Each category gets its own function with its own parameters
                    # Parameters have shape [K-1, T], need output [N, K-1, T] -> transpose to [N, T, K-1]
                    K_minus_1 = K - 1
                    x_expanded = x_true.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]

                    if function_type == 'single_hill':
                        # Compute Hill for each category
                        # Vmax_a, K_a, n_a have shape [K-1, T]
                        # Add batch dimension: [1, K-1, T]
                        Hilla = Hill_based_positive(x_expanded, Vmax=Vmax_a.unsqueeze(0), A=0,
                                                   K=K_a.unsqueeze(0), n=n_a.unsqueeze(0),
                                                   epsilon=epsilon_tensor)  # [N, K-1, T]
                        # A is [T], alpha is [T] -> broadcast to [1, 1, T] and [1, 1, T]
                        y_dose_response_kminus1_transposed = A.unsqueeze(0).unsqueeze(0) + (alpha.unsqueeze(0).unsqueeze(0) * Hilla)
                        # Transpose to [N, T, K-1]
                        y_dose_response_kminus1 = y_dose_response_kminus1_transposed.transpose(-1, -2)

                    elif function_type == 'additive_hill':
                        Hilla = Hill_based_positive(x_expanded, Vmax=Vmax_a.unsqueeze(0), A=0,
                                                   K=K_a.unsqueeze(0), n=n_a.unsqueeze(0),
                                                   epsilon=epsilon_tensor)  # [N, K-1, T]
                        Hillb = Hill_based_positive(x_expanded, Vmax=Vmax_b.unsqueeze(0), A=0,
                                                   K=K_b.unsqueeze(0), n=n_b.unsqueeze(0),
                                                   epsilon=epsilon_tensor)  # [N, K-1, T]
                        y_dose_response_kminus1_transposed = (A.unsqueeze(0).unsqueeze(0) +
                                                              (alpha.unsqueeze(0).unsqueeze(0) * Hilla) +
                                                              (beta.unsqueeze(0).unsqueeze(0) * Hillb))
                        y_dose_response_kminus1 = y_dose_response_kminus1_transposed.transpose(-1, -2)

                    elif function_type == 'nested_hill':
                        Hilla = Hill_based_positive(x_expanded, Vmax=Vmax_a.unsqueeze(0), A=0,
                                                   K=K_a.unsqueeze(0), n=n_a.unsqueeze(0),
                                                   epsilon=epsilon_tensor)  # [N, K-1, T]
                        Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b.unsqueeze(0), A=0,
                                                   K=K_b.unsqueeze(0), n=n_b.unsqueeze(0),
                                                   epsilon=epsilon_tensor)  # [N, K-1, T]
                        y_dose_response_kminus1_transposed = A.unsqueeze(0).unsqueeze(0) + (alpha.unsqueeze(0).unsqueeze(0) * Hillb)
                        y_dose_response_kminus1 = y_dose_response_kminus1_transposed.transpose(-1, -2)

                    # Clamp K-1 probabilities to valid range
                    y_dose_response_kminus1 = torch.clamp(y_dose_response_kminus1,
                                                          min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Ensure sum of K-1 probabilities doesn't exceed 1
                    sum_kminus1 = y_dose_response_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    # If sum > 1, rescale proportionally
                    y_dose_response_kminus1 = torch.where(
                        sum_kminus1 > (1.0 - epsilon_tensor),
                        y_dose_response_kminus1 * (1.0 - epsilon_tensor) / sum_kminus1,
                        y_dose_response_kminus1
                    )

                    # Compute Kth category as residual: p_K = 1 - sum(p_1, ..., p_{K-1})
                    p_K = 1.0 - y_dose_response_kminus1.sum(dim=-1, keepdim=True)  # [N, T, 1]
                    p_K = torch.clamp(p_K, min=epsilon_tensor, max=1.0 - epsilon_tensor)

                    # Concatenate to get all K probabilities
                    y_dose_response = torch.cat([y_dose_response_kminus1, p_K], dim=-1)  # [N, T, K]

                else:
                    # For non-multinomial: standard Hill computation
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

                    # For binomial, clamp output to [0, 1] to ensure valid probabilities
                    if distribution == 'binomial':
                        y_dose_response = torch.clamp(y_dose_response, min=epsilon_tensor, max=1.0 - epsilon_tensor)

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
        p_n: float = 1e-6,
        epsilon: float = 1e-6,
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
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal'
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
        K_max_tensor = torch.max(torch.stack([torch.mean(x_true_mean[guides_tensor == g]) for g in torch.unique(guides_tensor)]))

        # Distribution-specific normalization for data-driven priors
        if distribution == 'binomial' and denominator_tensor is not None:
            # For binomial: normalize by denominator to get probabilities [0,1]
            y_obs_factored = y_obs_tensor / denominator_tensor.clamp_min(epsilon_tensor)
        elif distribution == 'multinomial' and y_obs_tensor.ndim == 3:
            # For multinomial: normalize by total across categories to get proportions [0,1]
            # y_obs_tensor is [N, T, K], sum across K to get [N, T, 1]
            total_counts = y_obs_tensor.sum(dim=-1, keepdim=True).clamp_min(epsilon_tensor)
            y_obs_factored = y_obs_tensor / total_counts  # [N, T, K] with proportions in [0,1]
        elif sum_factor_col is not None:
            # For negbinom: normalize by sum factors to get expression per size factor
            y_obs_factored = y_obs_tensor / sum_factor_tensor.view(-1, 1)
        else:
            # For other distributions: use raw values
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



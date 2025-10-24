"""
Prior sampling utilities for bayesDREAM plotting.

This module implements prior sampling from the Pyro model distributions
to enable true prior/posterior comparison plots.
"""

import numpy as np
import torch
import pyro
import pyro.distributions as dist


def sample_technical_priors(
    model,
    modality_name=None,
    nsamples=1000,
    distribution=None
):
    """
    Sample from the prior distributions defined in _model_technical.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    modality_name : str, optional
        Modality name (default: primary modality)
    nsamples : int
        Number of prior samples to draw
    distribution : str, optional
        Distribution type (default: from modality)

    Returns
    -------
    dict
        Dictionary of prior samples matching posterior_samples_technical structure
    """
    if modality_name is None:
        modality_name = model.primary_modality

    modality = model.get_modality(modality_name)

    if distribution is None:
        distribution = modality.distribution

    # Get model dimensions
    if modality_name == model.primary_modality and hasattr(model, 'counts'):
        T = model.counts.shape[0]  # All features including cis gene
    else:
        if modality.counts.ndim == 2:
            T = modality.counts.shape[0 if modality.cells_axis == 1 else 1]
        else:
            T = modality.counts.shape[0]

    C = model.meta['technical_group_code'].nunique()

    # Sample priors based on distribution
    prior_samples = {}

    # Cell-line effects: log2_alpha_y ~ StudentT(df=3, loc=0, scale=20)
    log2_alpha_y = dist.StudentT(df=3, loc=0.0, scale=20.0).sample((nsamples, C - 1, T))
    prior_samples['log2_alpha_y'] = log2_alpha_y
    prior_samples['alpha_y_mul'] = 2.0 ** log2_alpha_y  # multiplicative
    prior_samples['delta_y_add'] = log2_alpha_y  # additive/logit

    # Add baseline row
    alpha_full_mul = torch.cat([torch.ones(nsamples, 1, T), prior_samples['alpha_y_mul']], dim=1)
    alpha_full_add = torch.cat([torch.zeros(nsamples, 1, T), prior_samples['delta_y_add']], dim=1)
    prior_samples['alpha_y'] = alpha_full_mul  # back-compat
    prior_samples['alpha_y_mult'] = alpha_full_mul
    prior_samples['alpha_y_add'] = alpha_full_add

    # Distribution-specific parameters
    if distribution == 'negbinom':
        # Default prior parameters from fit_technical
        beta_o_alpha = 9.0
        beta_o_beta = 3.0

        # beta_o ~ Gamma(alpha, beta)
        beta_o = dist.Gamma(beta_o_alpha, beta_o_beta).sample((nsamples,))
        prior_samples['beta_o'] = beta_o

        # o_y ~ Exponential(beta_o) for each feature
        o_y = dist.Exponential(beta_o.unsqueeze(-1)).sample((T,)).squeeze()  # (nsamples, T)
        prior_samples['o_y'] = o_y

        # mu_ntc: Would need empirical estimates from data
        # For now, sample from broad Gamma (will be replaced with actual data-driven priors)
        mu_ntc = dist.Gamma(2.0, 0.1).sample((nsamples, T))
        prior_samples['mu_ntc'] = mu_ntc

    elif distribution == 'normal':
        # sigma_y ~ HalfCauchy(10.0)
        sigma_y = dist.HalfCauchy(10.0).sample((nsamples, T))
        prior_samples['sigma_y'] = sigma_y

        # mu_ntc ~ Normal(mu, sd) - would need data-driven priors
        mu_ntc = dist.Normal(0.0, 10.0).sample((nsamples, T))
        prior_samples['mu_ntc'] = mu_ntc

    elif distribution == 'binomial':
        # mu_ntc ~ Beta(a, b) - would need empirical Beta parameters from data
        # For now, use uniform Beta(1, 1)
        mu_ntc = dist.Beta(1.0, 1.0).sample((nsamples, T))
        prior_samples['mu_ntc'] = mu_ntc

    elif distribution == 'multinomial':
        # Get K from modality
        K = modality.counts.shape[2] if modality.counts.ndim == 3 else None
        if K is None:
            raise ValueError("Cannot determine K for multinomial distribution")

        # probs_baseline ~ Dirichlet(concentration)
        # Use uniform Dirichlet(1, ..., 1)
        concentration = torch.ones(T, K)
        probs_baseline = dist.Dirichlet(concentration).sample((nsamples,))  # (nsamples, T, K)
        prior_samples['probs_baseline'] = probs_baseline

    elif distribution == 'mvnormal':
        # Get D from modality
        D = modality.counts.shape[2] if modality.counts.ndim == 3 else None
        if D is None:
            raise ValueError("Cannot determine D for mvnormal distribution")

        # sigma_y_mv ~ HalfCauchy(10.0)
        sigma_y_mv = dist.HalfCauchy(10.0).sample((nsamples, T, D))
        prior_samples['sigma_y_mv'] = sigma_y_mv

        # mu_ntc_mv ~ Normal(0, 10)
        mu_ntc_mv = dist.Normal(0.0, 10.0).sample((nsamples, T, D))
        prior_samples['mu_ntc_mv'] = mu_ntc_mv

    return prior_samples


def sample_cis_priors(
    model,
    nsamples=1000
):
    """
    Sample from the prior distributions defined in _model_x (cis model).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    nsamples : int
        Number of prior samples to draw

    Returns
    -------
    dict
        Dictionary of prior samples matching posterior_samples_cis structure
    """
    # Get dimensions
    N = model.meta.shape[0]  # number of cells
    n_guides = model.meta['guide_code'].nunique()

    prior_samples = {}

    # x_true ~ LogNormal(mu, sigma) per guide
    # mu ~ Normal(data-driven mean, sd)
    # sigma ~ HalfNormal(1.0) or similar

    # For now, sample from broad priors
    # In practice, these would be data-driven like in fit_cis
    mu = dist.Normal(0.0, 2.0).sample((nsamples, n_guides))
    sigma = dist.HalfNormal(1.0).sample((nsamples, n_guides))

    # x_true ~ LogNormal(mu[guide], sigma[guide])
    # Expand to all cells based on guide assignment
    guide_codes = torch.tensor(model.meta['guide_code'].values, dtype=torch.long)

    x_true_samples = []
    for s in range(nsamples):
        x_true_per_cell = []
        for i, g in enumerate(guide_codes):
            x_true_per_cell.append(
                dist.LogNormal(mu[s, g], sigma[s, g]).sample()
            )
        x_true_samples.append(torch.stack(x_true_per_cell))

    prior_samples['x_true'] = torch.stack(x_true_samples)  # (nsamples, N)

    # alpha_x: if cell-line specific, would be similar to alpha_y
    # For now, use scalar or simple prior
    if hasattr(model, 'alpha_x_prefit') and model.alpha_x_prefit is not None:
        if model.alpha_x_prefit.ndim >= 2:
            # Cell-line specific
            C = model.meta['technical_group_code'].nunique()
            log2_alpha_x = dist.StudentT(df=3, loc=0.0, scale=20.0).sample((nsamples, C - 1, 1))
            alpha_x_mul = 2.0 ** log2_alpha_x
            alpha_full_mul = torch.cat([torch.ones(nsamples, 1, 1), alpha_x_mul], dim=1)
            prior_samples['alpha_x'] = alpha_full_mul.squeeze(-1)  # (nsamples, C)
        else:
            # Scalar
            log2_alpha_x = dist.StudentT(df=3, loc=0.0, scale=20.0).sample((nsamples,))
            prior_samples['alpha_x'] = 2.0 ** log2_alpha_x

    # beta_o, o_x for negbinom (same as technical model)
    beta_o_alpha = 9.0
    beta_o_beta = 3.0
    beta_o = dist.Gamma(beta_o_alpha, beta_o_beta).sample((nsamples,))
    prior_samples['beta_o'] = beta_o

    o_x = dist.Exponential(beta_o).sample()
    prior_samples['o_x'] = o_x

    # mu, sigma for per-guide variation
    prior_samples['mu'] = mu
    prior_samples['sigma'] = sigma

    return prior_samples


def sample_trans_priors(
    model,
    modality_name=None,
    function_type='additive_hill',
    nsamples=1000,
    distribution=None
):
    """
    Sample from the prior distributions defined in _model_y (trans model).

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    modality_name : str, optional
        Modality name (default: primary modality)
    function_type : str
        Function type used in trans fit ('additive_hill', 'single_hill', 'polynomial')
    nsamples : int
        Number of prior samples to draw
    distribution : str, optional
        Distribution type (default: from modality)

    Returns
    -------
    dict
        Dictionary of prior samples matching posterior_samples_trans structure
    """
    if modality_name is None:
        modality_name = model.primary_modality

    modality = model.get_modality(modality_name)

    if distribution is None:
        distribution = modality.distribution

    # Get dimensions
    T = len(model.trans_genes)

    prior_samples = {}

    # Baseline A ~ Normal or LogNormal depending on distribution
    if distribution == 'negbinom':
        A = dist.Gamma(2.0, 0.1).sample((nsamples, T))
    elif distribution == 'normal':
        A = dist.Normal(0.0, 10.0).sample((nsamples, T))
    elif distribution == 'binomial':
        A = dist.Beta(1.0, 1.0).sample((nsamples, T))
    else:
        A = dist.Normal(0.0, 10.0).sample((nsamples, T))

    prior_samples['A'] = A

    # Function-specific parameters
    if function_type == 'additive_hill':
        # Gamma priors for sparsity on effect sizes
        lambda_a = dist.Gamma(1.0, 0.1).sample((nsamples,))
        lambda_b = dist.Gamma(1.0, 0.1).sample((nsamples,))
        prior_samples['lambda_a'] = lambda_a
        prior_samples['lambda_b'] = lambda_b

        # Effect sizes with sparsity priors
        alpha = dist.Gamma(1.0, lambda_a.unsqueeze(-1)).sample((T,)).squeeze()
        beta = dist.Gamma(1.0, lambda_b.unsqueeze(-1)).sample((T,)).squeeze()
        prior_samples['alpha'] = alpha
        prior_samples['beta'] = beta

        # Hill parameters
        Vmax_a = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        Vmax_b = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        K_a = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        K_b = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        n_a = dist.LogNormal(0.0, 1.0).sample((nsamples, T))
        n_b = dist.LogNormal(0.0, 1.0).sample((nsamples, T))

        prior_samples['Vmax_a'] = Vmax_a
        prior_samples['Vmax_b'] = Vmax_b
        prior_samples['K_a'] = K_a
        prior_samples['K_b'] = K_b
        prior_samples['n_a'] = n_a
        prior_samples['n_b'] = n_b

        # Theta is composite
        theta = torch.stack([A, alpha, beta, Vmax_a, Vmax_b, K_a, K_b, n_a, n_b], dim=-1)
        prior_samples['theta'] = theta

    elif function_type == 'single_hill':
        # Single Hill function priors
        Vmax = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        K = dist.LogNormal(0.0, 2.0).sample((nsamples, T))
        n = dist.LogNormal(0.0, 1.0).sample((nsamples, T))

        prior_samples['Vmax'] = Vmax
        prior_samples['K'] = K
        prior_samples['n'] = n

        theta = torch.stack([A, Vmax, K, n], dim=-1)
        prior_samples['theta'] = theta

    elif function_type == 'polynomial':
        # Polynomial coefficients
        degree = 6  # default
        coef_names = [f'coef_{i}' for i in range(degree + 1)]

        coefs = []
        for i in range(degree + 1):
            # Decreasing variance for higher order terms
            scale = 10.0 / (i + 1)
            coef = dist.Normal(0.0, scale).sample((nsamples, T))
            prior_samples[f'coef_{i}'] = coef
            coefs.append(coef)

        theta = torch.stack(coefs, dim=-1)
        prior_samples['theta'] = theta

    return prior_samples


def get_prior_samples(
    model,
    fit_type='technical',
    modality_name=None,
    nsamples=1000,
    **kwargs
):
    """
    Unified interface for sampling priors from any fit type.

    Parameters
    ----------
    model : bayesDREAM
        Fitted bayesDREAM model
    fit_type : str
        Type of fit: 'technical', 'cis', or 'trans'
    modality_name : str, optional
        Modality name (default: primary modality)
    nsamples : int
        Number of prior samples to draw
    **kwargs
        Additional parameters passed to specific sampling functions

    Returns
    -------
    dict
        Dictionary of prior samples
    """
    if fit_type == 'technical':
        return sample_technical_priors(model, modality_name, nsamples, **kwargs)
    elif fit_type == 'cis':
        return sample_cis_priors(model, nsamples, **kwargs)
    elif fit_type == 'trans':
        return sample_trans_priors(model, modality_name, nsamples=nsamples, **kwargs)
    else:
        raise ValueError(f"Unknown fit_type: {fit_type}")

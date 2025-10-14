"""
Distribution-specific observation samplers for multi-modal bayesDREAM.

This module provides observation likelihoods for different distributions.
These are called by the main Pyro models (_model_y, _model_technical) after
computing the dose-response function parameters.

Supported distributions:
- negbinom: Negative binomial (gene counts, transcript counts)
- multinomial: Categorical/proportional data (isoform usage, donor/acceptor usage)
- binomial: Binary outcomes with denominator (exon skipping PSI, raw SJ counts)
- normal: Continuous measurements (SpliZ scores)
- mvnormal: Multivariate normal (SpliZVD with z0, z1, z2)

Cell-line covariate handling:
- negbinom: Multiplicative effects on mu via alpha_y
- normal/mvnormal: Additive effects on mu
- binomial: Effects on logit scale
- multinomial: Not supported yet (complex - need to maintain probability simplex)
"""

import torch
import pyro
import pyro.distributions as dist


#######################################
# NEGATIVE BINOMIAL
#######################################

def sample_negbinom_trans(
    y_obs_tensor,
    mu_y,
    phi_y_used,
    alpha_y_full,
    groups_tensor,
    sum_factor_tensor,
    N,
    T,
    C=None
):
    """
    Sample observations from negative binomial likelihood for trans model.

    Cell-line effects: Multiplicative on mu (via alpha_y parameter).

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed counts, shape [N, T]
    mu_y : torch.Tensor
        Expected expression from f(x), shape [N, T]
    phi_y_used : torch.Tensor
        Overdispersion parameter, shape [1, T] or [T]
    alpha_y_full : torch.Tensor or None
        Cell-line effects, shape [C, T] if provided
    groups_tensor : torch.Tensor or None
        Cell-line group codes, shape [N] if provided
    sum_factor_tensor : torch.Tensor
        Size factors, shape [N]
    N : int
        Number of guides/observations
    T : int
        Number of features (trans genes)
    C : int or None
        Number of cell-line groups

    Notes
    -----
    mu_final = mu_y * alpha_y[group] * sum_factor
    """
    # Apply cell-line effects if present (multiplicative)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_adjusted = mu_y * alpha_y_used
    else:
        mu_adjusted = mu_y

    # Apply sum factors
    mu_final = mu_adjusted * sum_factor_tensor.unsqueeze(-1)  # [N, T]

    # Sample observations
    with pyro.plate("obs_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.NegativeBinomial(
                total_count=phi_y_used,
                logits=torch.log(mu_final + 1e-8) - torch.log(phi_y_used + 1e-8)
            ),
            obs=y_obs_tensor
        )


#######################################
# MULTINOMIAL
#######################################

def sample_multinomial_trans(
    y_obs_tensor,
    mu_y,
    N,
    T,
    K
):
    """
    Sample observations from multinomial likelihood for trans model.

    Cell-line effects: NOT SUPPORTED YET (too complex).

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed counts, shape [N, T, K] where K is number of categories
    mu_y : torch.Tensor
        Expected probabilities from f(x), shape [N, T, K]
        These should sum to 1 over K dimension
    N : int
        Number of guides/observations
    T : int
        Number of features
    K : int
        Number of categories per feature

    Notes
    -----
    For multinomial data, mu_y should already be normalized probabilities.
    Cell-line effects are not yet supported due to complexity of maintaining
    probability simplex constraints.
    """
    # Compute total counts per feature per observation
    total_counts = y_obs_tensor.sum(dim=-1)  # [N, T]

    # Ensure probabilities sum to 1
    probs = mu_y / (mu_y.sum(dim=-1, keepdim=True) + 1e-8)  # [N, T, K]

    # Sample observations
    with pyro.plate("feature_plate", T, dim=-1):
        with pyro.plate("obs_plate", N, dim=-2):
            pyro.sample(
                "y_obs",
                dist.Multinomial(total_count=total_counts, probs=probs),
                obs=y_obs_tensor
            )


#######################################
# BINOMIAL
#######################################

def sample_binomial_trans(
    y_obs_tensor,
    denominator_tensor,
    mu_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    C=None
):
    """
    Sample observations from binomial likelihood for trans model.

    Cell-line effects: Applied on LOGIT scale to maintain p ∈ [0, 1].

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed success counts, shape [N, T]
    denominator_tensor : torch.Tensor
        Total counts (denominator), shape [N, T]
    mu_y : torch.Tensor
        Expected probability from f(x), shape [N, T]
        Should be in [0, 1] but we'll apply sigmoid anyway
    alpha_y_full : torch.Tensor or None
        Cell-line effects (additive on logit scale), shape [C, T] if provided
    groups_tensor : torch.Tensor or None
        Cell-line group codes, shape [N] if provided
    N : int
        Number of guides/observations
    T : int
        Number of features
    C : int or None
        Number of cell-line groups

    Notes
    -----
    logit(p) = logit(mu_y) + alpha_y[group]
    p = sigmoid(logit(p))
    """
    # Convert mu_y to logit scale
    mu_y_clamped = torch.clamp(mu_y, min=1e-6, max=1-1e-6)
    logit_mu = torch.log(mu_y_clamped) - torch.log(1 - mu_y_clamped)

    # Apply cell-line effects on logit scale (additive)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        logit_final = logit_mu + alpha_y_used
    else:
        logit_final = logit_mu

    # Convert back to probability
    probs = torch.sigmoid(logit_final)

    # Sample observations
    with pyro.plate("obs_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.Binomial(total_count=denominator_tensor, probs=probs),
            obs=y_obs_tensor
        )


#######################################
# NORMAL
#######################################

def sample_normal_trans(
    y_obs_tensor,
    mu_y,
    sigma_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    C=None
):
    """
    Sample observations from normal likelihood for trans model.

    Cell-line effects: ADDITIVE on mu (not multiplicative).

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed values, shape [N, T]
    mu_y : torch.Tensor
        Expected mean from f(x), shape [N, T]
    sigma_y : torch.Tensor
        Standard deviation, shape [1, T] or [T]
    alpha_y_full : torch.Tensor or None
        Cell-line effects (additive), shape [C, T] if provided
    groups_tensor : torch.Tensor or None
        Cell-line group codes, shape [N] if provided
    N : int
        Number of guides/observations
    T : int
        Number of features
    C : int or None
        Number of cell-line groups

    Notes
    -----
    mu_final = mu_y + alpha_y[group]
    For normal distributions, additive cell-line shifts make more sense
    than multiplicative effects.
    """
    # Apply cell-line effects if present (additive)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_adjusted = mu_y + alpha_y_used
    else:
        mu_adjusted = mu_y

    # Sample observations
    with pyro.plate("obs_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.Normal(mu_adjusted, sigma_y.unsqueeze(0) if sigma_y.dim() == 1 else sigma_y),
            obs=y_obs_tensor
        )


#######################################
# MULTIVARIATE NORMAL
#######################################

def sample_mvnormal_trans(
    y_obs_tensor,
    mu_y,
    cov_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    D,
    C=None
):
    """
    Sample observations from multivariate normal likelihood for trans model.

    Cell-line effects: ADDITIVE on mu (not multiplicative).

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed values, shape [N, T, D] where D is dimensionality
    mu_y : torch.Tensor
        Expected mean from f(x), shape [N, T, D]
    cov_y : torch.Tensor
        Covariance matrix, shape [T, D, D] or [1, T, D, D]
    alpha_y_full : torch.Tensor or None
        Cell-line effects (additive), shape [C, T, D] if provided
    groups_tensor : torch.Tensor or None
        Cell-line group codes, shape [N] if provided
    N : int
        Number of guides/observations
    T : int
        Number of features
    D : int
        Dimensionality (e.g., 3 for SpliZVD)
    C : int or None
        Number of cell-line groups

    Notes
    -----
    mu_final = mu_y + alpha_y[group]
    """
    # Apply cell-line effects if present (additive)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :, :]  # [N, T, D]
        mu_adjusted = mu_y + alpha_y_used
    else:
        mu_adjusted = mu_y

    # Sample observations
    with pyro.plate("feature_plate", T, dim=-2):
        with pyro.plate("obs_plate", N, dim=-3):
            pyro.sample(
                "y_obs",
                dist.MultivariateNormal(mu_adjusted, cov_y),
                obs=y_obs_tensor
            )


#######################################
# DISTRIBUTION REGISTRY
#######################################

DISTRIBUTION_REGISTRY = {
    'negbinom': {
        'trans': sample_negbinom_trans,
        'requires_denominator': False,
        'requires_sum_factor': True,
        'is_3d': False,
        'supports_cell_line': True,
        'cell_line_type': 'multiplicative'
    },
    'multinomial': {
        'trans': sample_multinomial_trans,
        'requires_denominator': False,
        'requires_sum_factor': False,
        'is_3d': True,
        'supports_cell_line': False,  # Not yet implemented
        'cell_line_type': None
    },
    'binomial': {
        'trans': sample_binomial_trans,
        'requires_denominator': True,
        'requires_sum_factor': False,
        'is_3d': False,
        'supports_cell_line': True,
        'cell_line_type': 'logit'
    },
    'normal': {
        'trans': sample_normal_trans,
        'requires_denominator': False,
        'requires_sum_factor': False,
        'is_3d': False,
        'supports_cell_line': True,
        'cell_line_type': 'additive'
    },
    'mvnormal': {
        'trans': sample_mvnormal_trans,
        'requires_denominator': False,
        'requires_sum_factor': False,
        'is_3d': True,
        'supports_cell_line': True,
        'cell_line_type': 'additive'
    }
}


def get_observation_sampler(distribution, model_type='trans'):
    """
    Get the appropriate observation sampler for a distribution.

    Parameters
    ----------
    distribution : str
        Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
    model_type : str
        Model type: currently only 'trans' is fully supported

    Returns
    -------
    callable
        Observation sampler function

    Raises
    ------
    ValueError
        If distribution is not recognized
    """
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(
            f"Unknown distribution: {distribution}. "
            f"Must be one of: {list(DISTRIBUTION_REGISTRY.keys())}"
        )

    if model_type != 'trans':
        raise NotImplementedError(
            f"Model type '{model_type}' not yet implemented. "
            f"Currently only 'trans' model is supported."
        )

    return DISTRIBUTION_REGISTRY[distribution][model_type]


def requires_denominator(distribution):
    """Check if a distribution requires a denominator array (e.g., binomial)."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['requires_denominator']


def requires_sum_factor(distribution):
    """Check if a distribution requires sum factors (only negbinom)."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['requires_sum_factor']


def is_3d_distribution(distribution):
    """Check if a distribution uses 3D data structure (multinomial, mvnormal)."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['is_3d']


def supports_cell_line_effects(distribution):
    """Check if cell-line covariate effects are supported for this distribution."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['supports_cell_line']


def get_cell_line_effect_type(distribution):
    """
    Get the type of cell-line effects for this distribution.

    Returns
    -------
    str or None
        'multiplicative', 'additive', 'logit', or None if not supported
    """
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['cell_line_type']

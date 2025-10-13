"""
Distribution-specific model implementations for multi-modal bayesDREAM.

This module provides observation likelihoods for different distributions:
- negbinom: Negative binomial (gene counts, transcript counts)
- multinomial: Categorical/proportional data (isoform usage, donor/acceptor usage)
- binomial: Binary outcomes (exon skipping PSI, raw SJ counts)
- normal: Continuous measurements (SpliZ scores)
- mvnormal: Multivariate normal (SpliZVD)

Each distribution has:
1. Technical model (_model_technical_<dist>): Estimates overdispersion from NTC data
2. Trans model (_model_y_<dist>): Models trans effects with distribution-specific likelihood
"""

import torch
import pyro
import pyro.distributions as dist
import warnings


#######################################
# NEGATIVE BINOMIAL (current implementation)
#######################################

def sample_negbinom_technical(
    y_obs_ntc_tensor,
    sum_factor_ntc_tensor,
    alpha_y_used,
    phi_y_used,
    mu_ntc,
    N,
    T
):
    """
    Sample observation from negative binomial likelihood for technical model.

    Parameters
    ----------
    y_obs_ntc_tensor : torch.Tensor
        Observed counts [N, T]
    sum_factor_ntc_tensor : torch.Tensor
        Sum factors [N]
    alpha_y_used : torch.Tensor
        Cell line effects [N, T]
    phi_y_used : torch.Tensor
        Overdispersion parameter [T]
    mu_ntc : torch.Tensor
        Baseline expression [T]
    N : int
        Number of cells
    T : int
        Number of features

    Returns
    -------
    None (samples into Pyro trace)
    """
    with pyro.plate("data_plate", N, dim=-2):
        pyro.sample(
            "y_obs_ntc",
            dist.NegativeBinomial(
                total_count=phi_y_used,
                logits=torch.log(alpha_y_used * mu_ntc * sum_factor_ntc_tensor.unsqueeze(-1)) - torch.log(phi_y_used)
            ),
            obs=y_obs_ntc_tensor
        )


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
    Sample observation from negative binomial likelihood for trans model.

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed counts [N, T]
    mu_y : torch.Tensor
        Expected expression (from f(x)) [N, T]
    phi_y_used : torch.Tensor
        Overdispersion parameter [T]
    alpha_y_full : torch.Tensor
        Cell line effects [C, T] or None
    groups_tensor : torch.Tensor
        Cell line group codes [N] or None
    sum_factor_tensor : torch.Tensor
        Sum factors [N]
    N : int
        Number of cells
    T : int
        Number of features
    C : int or None
        Number of cell line groups

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Apply cell line effects if present
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_y_adjusted = mu_y * alpha_y_used
    else:
        mu_y_adjusted = mu_y

    # Sample observations
    with pyro.plate("data_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.NegativeBinomial(
                total_count=phi_y_used,
                logits=torch.log(mu_y_adjusted * sum_factor_tensor.unsqueeze(-1)) - torch.log(phi_y_used)
            ),
            obs=y_obs_tensor
        )


#######################################
# MULTINOMIAL (isoform usage, donor/acceptor usage)
#######################################

def sample_multinomial_technical(
    y_obs_ntc_tensor,
    alpha_y_used,
    mu_ntc,
    N,
    T,
    K
):
    """
    Sample observation from multinomial likelihood for technical model.

    For multinomial data (e.g., isoform usage), we model the probabilities
    of each category using a Dirichlet prior and multinomial likelihood.

    Parameters
    ----------
    y_obs_ntc_tensor : torch.Tensor
        Observed counts [N, T, K] where K is number of categories
    alpha_y_used : torch.Tensor
        Cell line effects [N, T, K]
    mu_ntc : torch.Tensor
        Baseline probabilities [T, K]
    N : int
        Number of cells
    T : int
        Number of features
    K : int
        Number of categories (e.g., number of isoforms per gene)

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Compute total counts per feature per cell
    total_counts = y_obs_ntc_tensor.sum(dim=-1)  # [N, T]

    # Compute expected probabilities (must sum to 1 over K dimension)
    probs = alpha_y_used * mu_ntc.unsqueeze(0)  # [N, T, K]
    probs = probs / probs.sum(dim=-1, keepdim=True)  # Normalize

    # Sample observations
    with pyro.plate("feature_plate", T, dim=-1):
        with pyro.plate("data_plate", N, dim=-2):
            pyro.sample(
                "y_obs_ntc",
                dist.Multinomial(total_count=total_counts, probs=probs),
                obs=y_obs_ntc_tensor
            )


def sample_multinomial_trans(
    y_obs_tensor,
    mu_y,
    alpha_y_full,
    groups_tensor,
    N,
    T,
    K,
    C=None
):
    """
    Sample observation from multinomial likelihood for trans model.

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed counts [N, T, K]
    mu_y : torch.Tensor
        Expected probabilities (from f(x)) [N, T, K]
    alpha_y_full : torch.Tensor
        Cell line effects [C, T, K] or None
    groups_tensor : torch.Tensor
        Cell line group codes [N] or None
    N : int
        Number of cells
    T : int
        Number of features
    K : int
        Number of categories
    C : int or None
        Number of cell line groups

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Compute total counts per feature per cell
    total_counts = y_obs_tensor.sum(dim=-1)  # [N, T]

    # Apply cell line effects if present
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :, :]  # [N, T, K]
        probs = mu_y * alpha_y_used
    else:
        probs = mu_y

    # Normalize probabilities
    probs = probs / probs.sum(dim=-1, keepdim=True)

    # Sample observations
    with pyro.plate("feature_plate", T, dim=-1):
        with pyro.plate("data_plate", N, dim=-2):
            pyro.sample(
                "y_obs",
                dist.Multinomial(total_count=total_counts, probs=probs),
                obs=y_obs_tensor
            )


#######################################
# BINOMIAL (exon skipping PSI, raw SJ counts)
#######################################

def sample_binomial_technical(
    y_obs_ntc_tensor,
    denominator_ntc_tensor,
    alpha_y_used,
    mu_ntc,
    N,
    T
):
    """
    Sample observation from binomial likelihood for technical model.

    For binomial data (e.g., exon inclusion counts), we model the probability
    of "success" (e.g., inclusion) given total counts (denominator).

    Parameters
    ----------
    y_obs_ntc_tensor : torch.Tensor
        Observed success counts [N, T]
    denominator_ntc_tensor : torch.Tensor
        Total counts (denominator) [N, T]
    alpha_y_used : torch.Tensor
        Cell line effects [N, T]
    mu_ntc : torch.Tensor
        Baseline probability [T]
    N : int
        Number of cells
    T : int
        Number of features

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Compute probability (must be in [0, 1])
    logits = torch.log(alpha_y_used * mu_ntc.unsqueeze(0) + 1e-6) - torch.log(1 - alpha_y_used * mu_ntc.unsqueeze(0) + 1e-6)
    probs = torch.sigmoid(logits)

    # Sample observations
    with pyro.plate("data_plate", N, dim=-2):
        pyro.sample(
            "y_obs_ntc",
            dist.Binomial(total_count=denominator_ntc_tensor, probs=probs),
            obs=y_obs_ntc_tensor
        )


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
    Sample observation from binomial likelihood for trans model.

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed success counts [N, T]
    denominator_tensor : torch.Tensor
        Total counts (denominator) [N, T]
    mu_y : torch.Tensor
        Expected probability (from f(x)) [N, T]
    alpha_y_full : torch.Tensor
        Cell line effects [C, T] or None
    groups_tensor : torch.Tensor
        Cell line group codes [N] or None
    N : int
        Number of cells
    T : int
        Number of features
    C : int or None
        Number of cell line groups

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Apply cell line effects if present
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_y_adjusted = mu_y * alpha_y_used
    else:
        mu_y_adjusted = mu_y

    # Convert to probability
    probs = torch.sigmoid(mu_y_adjusted)

    # Sample observations
    with pyro.plate("data_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.Binomial(total_count=denominator_tensor, probs=probs),
            obs=y_obs_tensor
        )


#######################################
# NORMAL (SpliZ scores)
#######################################

def sample_normal_technical(
    y_obs_ntc_tensor,
    alpha_y_used,
    sigma_y,
    mu_ntc,
    N,
    T
):
    """
    Sample observation from normal likelihood for technical model.

    For continuous measurements (e.g., SpliZ scores), we model with normal distribution.

    Parameters
    ----------
    y_obs_ntc_tensor : torch.Tensor
        Observed values [N, T]
    alpha_y_used : torch.Tensor
        Cell line effects [N, T]
    sigma_y : torch.Tensor
        Standard deviation [T]
    mu_ntc : torch.Tensor
        Baseline mean [T]
    N : int
        Number of cells
    T : int
        Number of features

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Compute expected mean
    mu = alpha_y_used * mu_ntc.unsqueeze(0)

    # Sample observations
    with pyro.plate("data_plate", N, dim=-2):
        pyro.sample(
            "y_obs_ntc",
            dist.Normal(mu, sigma_y.unsqueeze(0)),
            obs=y_obs_ntc_tensor
        )


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
    Sample observation from normal likelihood for trans model.

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed values [N, T]
    mu_y : torch.Tensor
        Expected mean (from f(x)) [N, T]
    sigma_y : torch.Tensor
        Standard deviation [T]
    alpha_y_full : torch.Tensor
        Cell line effects [C, T] or None
    groups_tensor : torch.Tensor
        Cell line group codes [N] or None
    N : int
        Number of cells
    T : int
        Number of features
    C : int or None
        Number of cell line groups

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Apply cell line effects if present
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_y_adjusted = mu_y * alpha_y_used
    else:
        mu_y_adjusted = mu_y

    # Sample observations
    with pyro.plate("data_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.Normal(mu_y_adjusted, sigma_y.unsqueeze(0)),
            obs=y_obs_tensor
        )


#######################################
# MULTIVARIATE NORMAL (SpliZVD)
#######################################

def sample_mvnormal_technical(
    y_obs_ntc_tensor,
    alpha_y_used,
    cov_y,
    mu_ntc,
    N,
    T,
    D
):
    """
    Sample observation from multivariate normal likelihood for technical model.

    For multivariate continuous measurements (e.g., SpliZVD with z0, z1, z2),
    we model with multivariate normal distribution.

    Parameters
    ----------
    y_obs_ntc_tensor : torch.Tensor
        Observed values [N, T, D] where D is dimensionality (e.g., 3 for SpliZVD)
    alpha_y_used : torch.Tensor
        Cell line effects [N, T, D]
    cov_y : torch.Tensor
        Covariance matrix [T, D, D]
    mu_ntc : torch.Tensor
        Baseline mean [T, D]
    N : int
        Number of cells
    T : int
        Number of features
    D : int
        Dimensionality

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Compute expected mean
    mu = alpha_y_used * mu_ntc.unsqueeze(0)  # [N, T, D]

    # Sample observations
    with pyro.plate("feature_plate", T, dim=-1):
        with pyro.plate("data_plate", N, dim=-2):
            pyro.sample(
                "y_obs_ntc",
                dist.MultivariateNormal(mu, cov_y.unsqueeze(0)),
                obs=y_obs_ntc_tensor
            )


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
    Sample observation from multivariate normal likelihood for trans model.

    Parameters
    ----------
    y_obs_tensor : torch.Tensor
        Observed values [N, T, D]
    mu_y : torch.Tensor
        Expected mean (from f(x)) [N, T, D]
    cov_y : torch.Tensor
        Covariance matrix [T, D, D]
    alpha_y_full : torch.Tensor
        Cell line effects [C, T, D] or None
    groups_tensor : torch.Tensor
        Cell line group codes [N] or None
    N : int
        Number of cells
    T : int
        Number of features
    D : int
        Dimensionality
    C : int or None
        Number of cell line groups

    Returns
    -------
    None (samples into Pyro trace)
    """
    # Apply cell line effects if present
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :, :]  # [N, T, D]
        mu_y_adjusted = mu_y * alpha_y_used
    else:
        mu_y_adjusted = mu_y

    # Sample observations
    with pyro.plate("feature_plate", T, dim=-1):
        with pyro.plate("data_plate", N, dim=-2):
            pyro.sample(
                "y_obs",
                dist.MultivariateNormal(mu_y_adjusted, cov_y.unsqueeze(0)),
                obs=y_obs_tensor
            )


#######################################
# DISTRIBUTION REGISTRY
#######################################

DISTRIBUTION_REGISTRY = {
    'negbinom': {
        'technical': sample_negbinom_technical,
        'trans': sample_negbinom_trans,
        'requires_denominator': False,
        'is_3d': False
    },
    'multinomial': {
        'technical': sample_multinomial_technical,
        'trans': sample_multinomial_trans,
        'requires_denominator': False,
        'is_3d': True
    },
    'binomial': {
        'technical': sample_binomial_technical,
        'trans': sample_binomial_trans,
        'requires_denominator': True,
        'is_3d': False
    },
    'normal': {
        'technical': sample_normal_technical,
        'trans': sample_normal_trans,
        'requires_denominator': False,
        'is_3d': False
    },
    'mvnormal': {
        'technical': sample_mvnormal_technical,
        'trans': sample_mvnormal_trans,
        'requires_denominator': False,
        'is_3d': True
    }
}


def get_observation_sampler(distribution, model_type='trans'):
    """
    Get the appropriate observation sampler for a distribution and model type.

    Parameters
    ----------
    distribution : str
        Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
    model_type : str
        Model type: 'technical' or 'trans'

    Returns
    -------
    callable
        Observation sampler function

    Raises
    ------
    ValueError
        If distribution or model_type is not recognized
    """
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}. "
                        f"Must be one of: {list(DISTRIBUTION_REGISTRY.keys())}")

    if model_type not in ['technical', 'trans']:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'technical' or 'trans'")

    return DISTRIBUTION_REGISTRY[distribution][model_type]


def requires_denominator(distribution):
    """Check if a distribution requires a denominator array."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['requires_denominator']


def is_3d_distribution(distribution):
    """Check if a distribution uses 3D data structure."""
    if distribution not in DISTRIBUTION_REGISTRY:
        raise ValueError(f"Unknown distribution: {distribution}")
    return DISTRIBUTION_REGISTRY[distribution]['is_3d']

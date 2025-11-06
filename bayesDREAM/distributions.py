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

#######################################
# MULTINOMIAL (robust masked softmax)
#######################################

def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    logits: [..., K] real (unmasked positions can be any finite value)
    mask:   [..., K] bool, True = masked/disallowed category
    Returns probs with:
      - probs[..., k] = 0 where mask is True
      - rows with at least one unmasked get a proper softmax over unmasked
      - rows with all entries masked fall back to uniform (no NaNs)
    """
    # Set masked to -inf, unmasked unchanged
    neg_inf = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(mask, neg_inf, logits)

    # Numerically stable shift by max over (potentially -inf) entries
    max_logits = masked_logits.max(dim=dim, keepdim=True).values
    # If row is all -inf, max is -inf; replace by 0 to avoid NaNs in exp
    max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))

    exps = torch.exp(masked_logits - max_logits)
    # Zero out masked entries explicitly (exp(-inf)=0 already; this is just defensive)
    exps = torch.where(mask, torch.zeros_like(exps), exps)

    sums = exps.sum(dim=dim, keepdim=True)
    # Normal case: divide by sum
    probs = torch.where(sums > 0, exps / sums, exps)  # if sums==0 we’ll fix below

    # Handle fully-masked rows (sums==0): return uniform over K (can’t infer anything)
    all_masked = (sums == 0)
    if all_masked.any():
        K = logits.size(dim)
        uniform = torch.full_like(probs, 1.0 / K)
        probs = torch.where(all_masked, uniform, probs)

    return probs


def sample_multinomial_trans(
    y_obs_tensor,         # [N, T, K] integer counts
    mu_y,                 # [T, K] (baseline probs) or [N, T, K]; must be 0 at masked K
    alpha_y_full,         # [C, T, K] or [S, C, T, K] additive logits
    groups_tensor,        # [N] int in [0..C-1]
    N, T, K, C=None
):
    # Basic sanity checks (fail fast instead of NaNs downstream)
    if not torch.isfinite(y_obs_tensor).all():
        raise ValueError("y_obs_tensor contains non-finite values")
    if (y_obs_tensor < 0).any():
        raise ValueError("y_obs_tensor contains negative counts")

    # Totals per (obs, feature)
    total_counts = y_obs_tensor.sum(dim=-1)  # [N, T]

    # Ensure mu_y has shape [N, T, K]
    if mu_y.dim() == 2:  # [T, K] -> [N, T, K]
        mu_y = mu_y.unsqueeze(0).expand(N, -1, -1)

    # Zero-category mask from mu_y (assumed exact zeros at masked K)
    zero_mask = (mu_y == 0)  # [N, T, K] (rows identical if mu_y was [T,K])

    # Base logits: log(mu_y) where unmasked; will never be taken at masked due to masked softmax
    safe_log_mu = torch.where(zero_mask, torch.zeros_like(mu_y), mu_y).log()  # avoid log(0)

    def _probs_from_logits(base_log_mu, alpha):
        logits = base_log_mu + alpha
        # Use masked softmax so rows with all masked don't produce NaN
        return _masked_softmax(logits, mask=zero_mask, dim=-1)

    # Add α and compute probabilities
    if alpha_y_full is not None and groups_tensor is not None:
        if alpha_y_full.dim() == 3:
            # [C, T, K] -> index by group per observation -> [N, T, K]
            alpha_used = alpha_y_full[groups_tensor, :, :]
            probs = _probs_from_logits(safe_log_mu, alpha_used)        # [N, T, K]
        elif alpha_y_full.dim() == 4:
            # [S, C, T, K] -> index groups along dim=1 -> [S, N, T, K]
            S = alpha_y_full.size(0)
            alpha_used = alpha_y_full[:, groups_tensor, :, :]          # [S, N, T, K]
            base_logits_S = safe_log_mu.unsqueeze(0).expand(S, -1, -1, -1)
            probs = _probs_from_logits(base_logits_S, alpha_used)      # [S, N, T, K]
        else:
            raise ValueError(f"Unexpected alpha_y_full shape: {tuple(alpha_y_full.shape)}")
    else:
        # No α: just take masked softmax of baseline logits (respects zeros)
        probs = _masked_softmax(safe_log_mu, mask=zero_mask, dim=-1)   # [N, T, K] or [S, N, T, K]

    # Log-likelihood (stable)
    if probs.dim() == 3:
        log_probs = torch.log(probs.clamp_min(1e-12))
        ll = (
            torch.lgamma(total_counts + 1.0)
            - torch.lgamma(y_obs_tensor + 1.0).sum(dim=-1)
            + (y_obs_tensor * log_probs).sum(dim=-1)
        )  # [N, T]
        with pyro.plate("obs_plate", N * T):
            pyro.factor("y_obs", ll.reshape(-1))
    elif probs.dim() == 4:
        S = probs.size(0)
        y = y_obs_tensor.unsqueeze(0).expand(S, -1, -1, -1)            # [S, N, T, K]
        total_counts_S = total_counts.unsqueeze(0).expand(S, -1, -1)   # [S, N, T]
        log_probs = torch.log(probs.clamp_min(1e-12))
        ll = (
            torch.lgamma(total_counts_S + 1.0)
            - torch.lgamma(y + 1.0).sum(dim=-1)
            + (y * log_probs).sum(dim=-1)
        )  # [S, N, T]
        with pyro.plate("obs_plate", S * N * T):
            pyro.factor("y_obs", ll.reshape(-1))
    else:
        raise ValueError(f"Unexpected probs shape: {tuple(probs.shape)}")


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

    # Sample observations
    with pyro.plate("obs_plate", N, dim=-2):
        pyro.sample(
            "y_obs",
            dist.Binomial(total_count=denominator_tensor, logits=logit_final),
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
    Normal likelihood with *masked* observations.

    - NaNs / inf in y_obs_tensor are treated as missing
      and contribute 0 to the log-likelihood (factor of 1 in prob).
    - Cell-line effects are additive on the mean only; sigma_y is unchanged.
    """

    # Apply cell-line effects if present (additive on the mean)
    if alpha_y_full is not None and groups_tensor is not None:
        alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
        mu_adjusted = mu_y + alpha_y_used              # [N, T]
    else:
        mu_adjusted = mu_y                             # [N, T]

    # Broadcast sigma_y to [N, T]
    if sigma_y.dim() == 1:
        # [T] -> [N, T]
        sigma_b = sigma_y.unsqueeze(0).expand(N, -1)
    else:
        # Assume already broadcastable to [N, T]
        sigma_b = sigma_y

    # Mask of *valid* entries: finite = observed, non-finite = missing
    mask = torch.isfinite(y_obs_tensor)  # [N, T]
    if not mask.any():
        # Everything missing -> no likelihood contribution
        return

    # Replace missing y's by mu_adjusted so log_prob is defined;
    # they will get zeroed out later via the mask.
    y_clean = torch.where(mask, y_obs_tensor, mu_adjusted)

    # Log-prob under Normal
    dist_normal = dist.Normal(mu_adjusted, sigma_b)
    log_prob = dist_normal.log_prob(y_clean)          # [N, T]

    # Zero-out missing entries so they don't affect the joint density
    log_prob = torch.where(mask, log_prob,
                           torch.zeros_like(log_prob))

    # Flatten and attach as a factor
    with pyro.plate("obs_plate", N * T):
        pyro.factor("y_obs", log_prob.reshape(-1))


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
    # If additive alpha is [C, T], expand to [C, T, D] so it can add to mu
    if alpha_y_full is not None and alpha_y_full.dim() == 2:
        alpha_y_full = alpha_y_full.unsqueeze(-1).expand(-1, -1, D)  # [C, T, D]
    
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
        'supports_cell_line': True,     # <- change to True
        'cell_line_type': 'logit'
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

"""
Utility functions for plotting prior/posterior distributions.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import warnings


def compute_distribution_overlap(prior_samples: np.ndarray, posterior_samples: np.ndarray) -> float:
    """
    Compute percentage overlap between two distributions using KDE.

    This is a symmetric metric that measures the intersection area between distributions.
    Low overlap indicates the data was informative and concentrated the posterior.

    Parameters
    ----------
    prior_samples : np.ndarray
        Samples from prior distribution
    posterior_samples : np.ndarray
        Samples from posterior distribution

    Returns
    -------
    float
        Percentage overlap between 0 and 100, or 0.0 if KDE fails
    """
    from scipy.stats import gaussian_kde

    try:
        # Combine samples to get common evaluation points
        all_samples = np.concatenate([prior_samples, posterior_samples])
        x_eval = np.linspace(all_samples.min(), all_samples.max(), 200)

        # Compute KDEs
        kde_prior = gaussian_kde(prior_samples)
        kde_post = gaussian_kde(posterior_samples)

        # Evaluate densities
        prior_density = kde_prior(x_eval)
        post_density = kde_post(x_eval)

        # Compute overlap as integral of minimum
        overlap = np.trapz(np.minimum(prior_density, post_density), x_eval)

        # Normalize to percentage
        return overlap * 100

    except (np.linalg.LinAlgError, ValueError) as e:
        # Return 0 overlap if KDE fails (e.g., singular covariance, constant data)
        warnings.warn(f"KDE failed for overlap computation ({str(e)}), returning 0% overlap")
        return 0.0


def compute_kl_divergence(posterior_samples: np.ndarray, prior_samples: np.ndarray) -> float:
    """
    Compute KL divergence KL(posterior || prior) using KDE estimates.

    Measures information gained from prior to posterior. Higher values indicate
    more information gained from the data.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Samples from posterior distribution
    prior_samples : np.ndarray
        Samples from prior distribution

    Returns
    -------
    float
        KL divergence in nats, or NaN if computation fails
    """
    from scipy.stats import gaussian_kde

    try:
        # Combine samples to get common evaluation points
        all_samples = np.concatenate([prior_samples, posterior_samples])
        x_eval = np.linspace(all_samples.min(), all_samples.max(), 500)

        # Compute KDEs
        kde_prior = gaussian_kde(prior_samples)
        kde_post = gaussian_kde(posterior_samples)

        # Evaluate densities
        prior_density = kde_prior(x_eval)
        post_density = kde_post(x_eval)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        prior_density = np.maximum(prior_density, eps)
        post_density = np.maximum(post_density, eps)

        # KL(P||Q) = ∫ P(x) log(P(x)/Q(x)) dx
        kl = np.trapz(post_density * np.log(post_density / prior_density), x_eval)

        return max(0.0, kl)  # KL should be non-negative

    except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
        warnings.warn(f"KL divergence computation failed ({str(e)}), returning NaN")
        return np.nan


def compute_posterior_coverage(posterior_samples: np.ndarray, prior_samples: np.ndarray,
                               ci_level: float = 0.95) -> float:
    """
    Compute what percentage of posterior mass is covered by the prior's credible interval.

    This answers: "What fraction of the posterior is within the prior's plausible range?"
    High coverage (>90%) indicates the prior was reasonable and covered the posterior support.
    Low coverage (<50%) suggests prior-data conflict or that the prior was too narrow.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Samples from posterior distribution
    prior_samples : np.ndarray
        Samples from prior distribution
    ci_level : float
        Credible interval level for prior (default: 0.95)

    Returns
    -------
    float
        Percentage of posterior samples within prior CI, between 0 and 100
    """
    try:
        # Compute prior credible interval
        alpha = (1 - ci_level) / 2
        prior_lower = np.percentile(prior_samples, alpha * 100)
        prior_upper = np.percentile(prior_samples, (1 - alpha) * 100)

        # Check what fraction of posterior is within prior CI
        within_prior_ci = np.sum((posterior_samples >= prior_lower) &
                                 (posterior_samples <= prior_upper))
        coverage = (within_prior_ci / len(posterior_samples)) * 100

        return coverage

    except (ValueError, RuntimeWarning) as e:
        warnings.warn(f"Posterior coverage computation failed ({str(e)}), returning NaN")
        return np.nan


def compute_all_metrics(prior_samples: np.ndarray, posterior_samples: np.ndarray,
                       ci_level: float = 0.95) -> dict:
    """
    Compute all available prior/posterior comparison metrics.

    Parameters
    ----------
    prior_samples : np.ndarray
        Samples from prior distribution
    posterior_samples : np.ndarray
        Samples from posterior distribution
    ci_level : float
        Credible interval level for coverage metric (default: 0.95)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'overlap': Symmetric intersection percentage (0-100)
        - 'kl_divergence': KL(posterior || prior) in nats
        - 'posterior_coverage': % of posterior within prior CI (0-100)
    """
    metrics = {
        'overlap': compute_distribution_overlap(prior_samples, posterior_samples),
        'kl_divergence': compute_kl_divergence(posterior_samples, prior_samples),
        'posterior_coverage': compute_posterior_coverage(posterior_samples, prior_samples, ci_level)
    }
    return metrics


def get_feature_ordering(
    values: np.ndarray,
    feature_names: List[str],
    order_by: str = 'mean',
    custom_order: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Get ordering indices for features based on specified criterion.

    Parameters
    ----------
    values : np.ndarray
        Parameter values, shape (n_samples, n_features) or (n_features,)
    feature_names : List[str]
        Feature names
    order_by : str
        One of: 'mean', 'difference', 'alphabetical', 'custom', 'input'
    custom_order : List[str], optional
        Custom ordering of feature names (only used if order_by='custom')

    Returns
    -------
    indices : np.ndarray
        Sorting indices
    ordered_names : List[str]
        Feature names in sorted order
    """
    n_features = len(feature_names)

    if order_by == 'input':
        # Keep input order
        indices = np.arange(n_features)

    elif order_by == 'alphabetical':
        indices = np.argsort(feature_names)

    elif order_by == 'custom':
        if custom_order is None:
            raise ValueError("custom_order must be provided when order_by='custom'")
        # Map feature names to indices
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        indices = np.array([name_to_idx[name] for name in custom_order if name in name_to_idx])
        if len(indices) < len(custom_order):
            missing = set(custom_order) - set(feature_names)
            warnings.warn(f"Some features in custom_order not found: {missing}")

    elif order_by == 'mean':
        # Order by mean value
        if values.ndim == 2:
            means = values.mean(axis=0)
        else:
            means = values
        indices = np.argsort(means)

    elif order_by == 'difference':
        # Requires prior and posterior - will be handled by caller
        raise ValueError("order_by='difference' requires both prior and posterior values")

    else:
        raise ValueError(f"Unknown order_by: {order_by}. Must be one of: "
                        "'mean', 'difference', 'alphabetical', 'custom', 'input'")

    ordered_names = [feature_names[i] for i in indices]
    return indices, ordered_names


def subset_features_by_mismatch(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    n_features: int = 100
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Subset to features with largest prior/posterior mismatch.

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples, n_features)
    posterior_samples : np.ndarray
        Posterior samples, shape (n_samples, n_features)
    feature_names : List[str]
        Feature names
    n_features : int
        Number of features to keep

    Returns
    -------
    prior_subset : np.ndarray
        Prior samples for selected features
    posterior_subset : np.ndarray
        Posterior samples for selected features
    selected_names : List[str]
        Names of selected features
    """
    n_total = prior_samples.shape[1]

    # Compute overlap for each feature
    overlaps = []
    for i in range(n_total):
        overlap = compute_distribution_overlap(prior_samples[:, i], posterior_samples[:, i])
        overlaps.append(overlap)

    overlaps = np.array(overlaps)

    # Select features with smallest overlap (largest mismatch)
    indices = np.argsort(overlaps)[:n_features]

    prior_subset = prior_samples[:, indices]
    posterior_subset = posterior_samples[:, indices]
    selected_names = [feature_names[i] for i in indices]

    return prior_subset, posterior_subset, selected_names


def prepare_violin_data(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    order_by: str = 'mean',
    custom_order: Optional[List[str]] = None,
    subset_features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Prepare data for violin plots with prior/posterior split.

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples, n_features)
    posterior_samples : np.ndarray
        Posterior samples, shape (n_samples, n_features)
    feature_names : List[str]
        Feature names
    order_by : str
        How to order features on x-axis
    custom_order : List[str], optional
        Custom feature ordering
    subset_features : List[str], optional
        Subset to these features

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: feature, value, distribution
    """
    # Subset if requested
    if subset_features is not None:
        if len(subset_features) > 100:
            warnings.warn(f"Plotting {len(subset_features)} features - plot may be crowded")

        # Get indices
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        indices = [name_to_idx[name] for name in subset_features if name in name_to_idx]

        prior_samples = prior_samples[:, indices]
        posterior_samples = posterior_samples[:, indices]
        feature_names = [feature_names[i] for i in indices]

    # Get ordering
    if order_by == 'difference':
        # Order by difference between posterior and prior means
        prior_means = prior_samples.mean(axis=0)
        post_means = posterior_samples.mean(axis=0)
        diffs = np.abs(post_means - prior_means)
        indices = np.argsort(diffs)[::-1]  # Descending
        ordered_names = [feature_names[i] for i in indices]
        prior_samples = prior_samples[:, indices]
        posterior_samples = posterior_samples[:, indices]
    else:
        indices, ordered_names = get_feature_ordering(
            posterior_samples, feature_names, order_by, custom_order
        )
        prior_samples = prior_samples[:, indices]
        posterior_samples = posterior_samples[:, indices]

    # Create long-format dataframe
    n_samples, n_features = prior_samples.shape

    data = []
    for i, name in enumerate(ordered_names):
        # Prior
        for val in prior_samples[:, i]:
            data.append({'feature': name, 'value': val, 'distribution': 'Prior'})
        # Posterior
        for val in posterior_samples[:, i]:
            data.append({'feature': name, 'value': val, 'distribution': 'Posterior'})

    df = pd.DataFrame(data)

    # Set categorical order to maintain sort
    df['feature'] = pd.Categorical(df['feature'], categories=ordered_names, ordered=True)

    return df


def compute_log2fc_vs_overlap(
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    feature_names: List[str],
    metric: str = 'posterior_coverage',
    is_additive: bool = False
) -> pd.DataFrame:
    """
    Compute mean shift (log2FC or difference) vs prior/posterior comparison metric.

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples, n_features)
    posterior_samples : np.ndarray
        Posterior samples, shape (n_samples, n_features)
    feature_names : List[str]
        Feature names
    metric : str
        Comparison metric: 'overlap', 'kl_divergence', or 'posterior_coverage' (default)
    is_additive : bool
        If True, compute difference (posterior - prior) for additive parameters.
        If False, compute log2 fold change for multiplicative parameters. (default: False)

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: feature, shift (log2fc or difference), <metric_name>
    """
    n_features = prior_samples.shape[1]

    # Determine metric column name
    if metric == 'overlap':
        metric_col = 'overlap'
    elif metric == 'kl_divergence':
        metric_col = 'kl_divergence'
    elif metric == 'posterior_coverage':
        metric_col = 'posterior_coverage'
    else:
        raise ValueError(f"Unknown metric: {metric}. Must be 'overlap', 'kl_divergence', or 'posterior_coverage'")

    results = []
    for i in range(n_features):
        prior_mean = prior_samples[:, i].mean()
        post_mean = posterior_samples[:, i].mean()

        # Compute shift based on scale type
        if is_additive:
            # Additive scale: compute difference
            shift = post_mean - prior_mean
            shift_col = 'difference'
        else:
            # Multiplicative scale: compute log2FC (add small epsilon to avoid log(0))
            shift = np.log2((post_mean + 1e-8) / (prior_mean + 1e-8))
            shift_col = 'log2fc'

        # Compute chosen metric
        if metric == 'overlap':
            metric_value = compute_distribution_overlap(prior_samples[:, i], posterior_samples[:, i])
        elif metric == 'kl_divergence':
            metric_value = compute_kl_divergence(posterior_samples[:, i], prior_samples[:, i])
        elif metric == 'posterior_coverage':
            metric_value = compute_posterior_coverage(posterior_samples[:, i], prior_samples[:, i])

        results.append({
            'feature': feature_names[i],
            shift_col: shift,
            metric_col: metric_value
        })

    return pd.DataFrame(results)

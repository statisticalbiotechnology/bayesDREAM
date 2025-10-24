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

    Parameters
    ----------
    prior_samples : np.ndarray
        Samples from prior distribution
    posterior_samples : np.ndarray
        Samples from posterior distribution

    Returns
    -------
    float
        Percentage overlap between 0 and 100
    """
    from scipy.stats import gaussian_kde

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
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Compute log2 fold change in means vs distribution overlap.

    Parameters
    ----------
    prior_samples : np.ndarray
        Prior samples, shape (n_samples, n_features)
    posterior_samples : np.ndarray
        Posterior samples, shape (n_samples, n_features)
    feature_names : List[str]
        Feature names

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: feature, log2fc, overlap
    """
    n_features = prior_samples.shape[1]

    results = []
    for i in range(n_features):
        prior_mean = prior_samples[:, i].mean()
        post_mean = posterior_samples[:, i].mean()

        # Compute log2FC (add small epsilon to avoid log(0))
        log2fc = np.log2((post_mean + 1e-8) / (prior_mean + 1e-8))

        # Compute overlap
        overlap = compute_distribution_overlap(prior_samples[:, i], posterior_samples[:, i])

        results.append({
            'feature': feature_names[i],
            'log2fc': log2fc,
            'overlap': overlap
        })

    return pd.DataFrame(results)

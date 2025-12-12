#!/usr/bin/env python
"""
Plot prior vs posterior distributions for a specific splice junction
to diagnose trans fitting issues.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_sj_prior_vs_posterior(model, sj_id, modality_name='splicing_sj', save_path=None):
    """
    Plot prior vs posterior distributions for a specific splice junction.

    Parameters
    ----------
    model : bayesDREAM
        Fitted model with trans results
    sj_id : str
        Splice junction ID (e.g., 'chr1:12345:67890:+')
    modality_name : str
        Name of the modality containing the SJ
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """

    # Get modality
    modality = model.get_modality(modality_name)

    # Find SJ index
    if 'coord.intron' in modality.feature_meta.columns:
        sj_idx = modality.feature_meta[modality.feature_meta['coord.intron'] == sj_id].index
    else:
        # Try using index directly
        sj_idx = modality.feature_meta.index[modality.feature_meta.index == sj_id]

    if len(sj_idx) == 0:
        raise ValueError(f"SJ '{sj_id}' not found in modality '{modality_name}'")

    sj_idx = sj_idx[0]

    # Get feature position in counts array
    sj_position = list(modality.feature_meta.index).index(sj_idx)

    print(f"Found SJ '{sj_id}' at position {sj_position}")

    # Get posterior samples
    if not hasattr(modality, 'posterior_samples_trans'):
        raise ValueError(f"No trans fitting results found for modality '{modality_name}'")

    posterior = modality.posterior_samples_trans

    # Extract posterior samples for this SJ
    A_post = posterior['A'][:, sj_position].cpu().numpy()  # [S]
    Vmax_sum_post = posterior['Vmax_sum'][:, sj_position].cpu().numpy()  # [S]

    # Get upper_limit if available
    if 'upper_limit' in posterior:
        upper_post = posterior['upper_limit'][:, sj_position].cpu().numpy()  # [S]
    else:
        upper_post = A_post + Vmax_sum_post

    print(f"Posterior samples: {len(A_post)} samples")
    print(f"A: [{A_post.min():.4f}, {A_post.max():.4f}], mean={A_post.mean():.4f}")
    print(f"Vmax_sum: [{Vmax_sum_post.min():.6f}, {Vmax_sum_post.max():.6f}], mean={Vmax_sum_post.mean():.6f}")
    print(f"upper_limit: [{upper_post.min():.4f}, {upper_post.max():.4f}], mean={upper_post.mean():.4f}")

    # Try to reconstruct priors from the data
    # This requires accessing the data that was used to compute priors

    # Get the actual observations for this SJ
    counts = modality.counts[sj_position, :].cpu().numpy()  # [N_cells]
    if modality.denominator is not None:
        denom = modality.denominator[sj_position, :].cpu().numpy()  # [N_cells]
        psi = counts / np.maximum(denom, 1)
    else:
        psi = counts

    # Get guide information
    guides = model.meta['guide'].values
    unique_guides = np.unique(guides)

    # Compute guide-level means (5th and 95th percentiles)
    guide_means = []
    for g in unique_guides:
        guide_mask = guides == g
        guide_psi = psi[guide_mask]
        # Filter out invalid values
        guide_psi = guide_psi[np.isfinite(guide_psi)]
        if len(guide_psi) > 0:
            guide_means.append(np.mean(guide_psi))

    guide_means = np.array(guide_means)
    guide_means = guide_means[np.isfinite(guide_means)]

    if len(guide_means) > 0:
        Amean_est = np.percentile(guide_means, 5)
        Vmax_mean_est = np.percentile(guide_means, 95)
        print(f"\nEstimated prior parameters (from data):")
        print(f"  Amean (5th percentile): {Amean_est:.4f}")
        print(f"  Vmax_mean (95th percentile): {Vmax_mean_est:.4f}")
    else:
        Amean_est = 0.1
        Vmax_mean_est = 0.5
        print(f"\nUsing fallback prior parameters")

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: A (baseline)
    ax = axes[0, 0]
    ax.hist(A_post, bins=50, density=True, alpha=0.7, label='Posterior', color='steelblue')

    # Prior for A: Beta(alpha=1, beta=(1-Amean)/Amean)
    # This gives mean = Amean
    if Amean_est > 0 and Amean_est < 1:
        beta_A = (1 - Amean_est) / Amean_est
        alpha_A = 1.0
        x = np.linspace(0.001, 0.999, 1000)
        prior_A = stats.beta.pdf(x, alpha_A, beta_A)
        ax.plot(x, prior_A, 'r-', linewidth=2, label=f'Prior Beta({alpha_A:.1f}, {beta_A:.2f})')

    ax.axvline(A_post.mean(), color='steelblue', linestyle='--', linewidth=2, label='Posterior mean')
    ax.axvline(Amean_est, color='red', linestyle='--', linewidth=2, label='Prior mean')
    ax.set_xlabel('A (baseline)')
    ax.set_ylabel('Density')
    ax.set_title(f'A (baseline) for {sj_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: upper_limit
    ax = axes[0, 1]
    ax.hist(upper_post, bins=50, density=True, alpha=0.7, label='Posterior', color='darkgreen')

    # Prior for upper_limit: Beta(alpha=Vmax/(1-Vmax), beta=1)
    if Vmax_mean_est > 0 and Vmax_mean_est < 1:
        alpha_upper = Vmax_mean_est / (1 - Vmax_mean_est)
        beta_upper = 1.0
        x = np.linspace(0.001, 0.999, 1000)
        prior_upper = stats.beta.pdf(x, alpha_upper, beta_upper)
        ax.plot(x, prior_upper, 'r-', linewidth=2, label=f'Prior Beta({alpha_upper:.2f}, {beta_upper:.1f})')

    ax.axvline(upper_post.mean(), color='darkgreen', linestyle='--', linewidth=2, label='Posterior mean')
    ax.axvline(Vmax_mean_est, color='red', linestyle='--', linewidth=2, label='Prior mean')
    ax.set_xlabel('upper_limit')
    ax.set_ylabel('Density')
    ax.set_title(f'upper_limit for {sj_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Vmax_sum (amplitude)
    ax = axes[1, 0]
    ax.hist(Vmax_sum_post, bins=50, density=True, alpha=0.7, label='Posterior', color='purple')
    ax.axvline(Vmax_sum_post.mean(), color='purple', linestyle='--', linewidth=2, label='Posterior mean')
    ax.set_xlabel('Vmax_sum (amplitude)')
    ax.set_ylabel('Density')
    ax.set_title(f'Vmax_sum for {sj_id}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Check if stuck at floor
    if Vmax_sum_post.mean() < 1e-4:
        ax.text(0.5, 0.9, 'WARNING: Stuck at floor!',
                transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                fontsize=12, fontweight='bold')

    # Plot 4: Data overview
    ax = axes[1, 1]

    # Plot guide-level means
    if len(guide_means) > 0:
        ax.hist(guide_means, bins=30, alpha=0.7, label='Guide means', color='orange', edgecolor='black')
        ax.axvline(Amean_est, color='red', linestyle='--', linewidth=2, label=f'5th %ile = {Amean_est:.3f}')
        ax.axvline(Vmax_mean_est, color='darkred', linestyle='--', linewidth=2, label=f'95th %ile = {Vmax_mean_est:.3f}')
        ax.axvline(A_post.mean(), color='steelblue', linestyle=':', linewidth=2, label=f'Fit A = {A_post.mean():.3f}')
        ax.axvline(upper_post.mean(), color='darkgreen', linestyle=':', linewidth=2, label=f'Fit upper = {upper_post.mean():.3f}')

    ax.set_xlabel('PSI value')
    ax.set_ylabel('Count')
    ax.set_title(f'Guide-level PSI distribution\n({len(unique_guides)} guides)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage:
    # from bayesDREAM import bayesDREAM
    # model = bayesDREAM.load('path/to/saved/model.h5')
    # fig = plot_sj_prior_vs_posterior(model, 'chr1:12345:67890:+', save_path='sj_diagnostic.png')
    # plt.show()

    print("Import this module and use plot_sj_prior_vs_posterior(model, sj_id)")

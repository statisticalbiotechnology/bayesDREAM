"""
Plot prior vs posterior distributions for binomial additive Hill model.

This shows the priors for A, Vmax_a, Vmax_b, K_a, K_b for a specific splice junction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def plot_additive_hill_priors(model, sj_id, modality_name='splicing_sj', save_path=None):
    """
    Plot prior vs posterior distributions for additive Hill parameters.

    Parameters
    ----------
    model : bayesDREAM
        Fitted model
    sj_id : str
        Splice junction ID (coord.intron value)
    modality_name : str
        Name of modality (default: 'splicing_sj')
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
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
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    A_post = to_numpy(posterior['A'][:, sj_position])
    Vmax_a_post = to_numpy(posterior['Vmax_a'][:, sj_position])
    Vmax_b_post = to_numpy(posterior['Vmax_b'][:, sj_position])
    K_a_post = to_numpy(posterior['K_a'][:, sj_position])
    K_b_post = to_numpy(posterior['K_b'][:, sj_position])

    # Get beta if available
    if 'beta' in posterior:
        beta_post = to_numpy(posterior['beta'][:, sj_position]) if posterior['beta'].ndim > 1 else to_numpy(posterior['beta'][sj_position])
    else:
        beta_post = None

    print(f"\nPosterior samples: {len(A_post)} samples")
    print(f"A: [{A_post.min():.4f}, {A_post.max():.4f}], mean={A_post.mean():.4f}")
    print(f"Vmax_a: [{Vmax_a_post.min():.4f}, {Vmax_a_post.max():.4f}], mean={Vmax_a_post.mean():.4f}")
    print(f"Vmax_b: [{Vmax_b_post.min():.4f}, {Vmax_b_post.max():.4f}], mean={Vmax_b_post.mean():.4f}")
    print(f"K_a: [{K_a_post.min():.2f}, {K_a_post.max():.2f}], mean={K_a_post.mean():.2f}")
    print(f"K_b: [{K_b_post.min():.2f}, {K_b_post.max():.2f}], mean={K_b_post.mean():.2f}")
    if beta_post is not None:
        print(f"beta: [{beta_post.min():.4f}, {beta_post.max():.4f}], mean={beta_post.mean():.4f}")

    # Get the actual observations for this SJ
    counts = modality.counts[sj_position, :]
    if isinstance(counts, torch.Tensor):
        counts = counts.cpu().numpy()

    if modality.denominator is not None:
        denom = modality.denominator[sj_position, :]
        if isinstance(denom, torch.Tensor):
            denom = denom.cpu().numpy()
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

    # Get x_true for K priors
    if hasattr(model, 'x_true'):
        if model.x_true_type == 'posterior':
            x_true_mean = to_numpy(model.x_true.mean(dim=0))
        else:
            x_true_mean = to_numpy(model.x_true)
        K_max_est = x_true_mean.max()
        print(f"  K_max (from x_true): {K_max_est:.2f}")
    else:
        K_max_est = 10.0
        print(f"  K_max (fallback): {K_max_est:.2f}")

    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ==========================================
    # Plot 1: A (baseline)
    # ==========================================
    ax = axes[0, 0]
    ax.hist(A_post, bins=50, density=True, alpha=0.7, label='Posterior', color='steelblue')

    # Prior for A: Beta(alpha=1, beta=(1-Amean)/Amean)
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
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==========================================
    # Plot 2: Vmax_a
    # ==========================================
    ax = axes[0, 1]
    ax.hist(Vmax_a_post, bins=50, density=True, alpha=0.7, label='Posterior', color='darkgreen')

    # Prior for Vmax_a: Beta(alpha=Vmax/(1-Vmax), beta=1)
    if Vmax_mean_est > 0 and Vmax_mean_est < 1:
        alpha_Vmax = Vmax_mean_est / (1 - Vmax_mean_est)
        beta_Vmax = 1.0
        x = np.linspace(0.001, 0.999, 1000)
        prior_Vmax = stats.beta.pdf(x, alpha_Vmax, beta_Vmax)
        ax.plot(x, prior_Vmax, 'r-', linewidth=2, label=f'Prior Beta({alpha_Vmax:.2f}, {beta_Vmax:.1f})')

    ax.axvline(Vmax_a_post.mean(), color='darkgreen', linestyle='--', linewidth=2, label='Posterior mean')
    ax.axvline(Vmax_mean_est, color='red', linestyle='--', linewidth=2, label='Prior mean')
    ax.set_xlabel('Vmax_a')
    ax.set_ylabel('Density')
    ax.set_title(f'Vmax_a for {sj_id}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Check if stuck at floor
    if Vmax_a_post.mean() < 1e-4:
        ax.text(0.5, 0.9, 'WARNING: Stuck at floor!',
                transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                fontsize=10, fontweight='bold')

    # ==========================================
    # Plot 3: Vmax_b
    # ==========================================
    ax = axes[0, 2]
    ax.hist(Vmax_b_post, bins=50, density=True, alpha=0.7, label='Posterior', color='purple')

    # Prior for Vmax_b: same as Vmax_a
    if Vmax_mean_est > 0 and Vmax_mean_est < 1:
        x = np.linspace(0.001, 0.999, 1000)
        prior_Vmax = stats.beta.pdf(x, alpha_Vmax, beta_Vmax)
        ax.plot(x, prior_Vmax, 'r-', linewidth=2, label=f'Prior Beta({alpha_Vmax:.2f}, {beta_Vmax:.1f})')

    ax.axvline(Vmax_b_post.mean(), color='purple', linestyle='--', linewidth=2, label='Posterior mean')
    ax.axvline(Vmax_mean_est, color='red', linestyle='--', linewidth=2, label='Prior mean')
    ax.set_xlabel('Vmax_b')
    ax.set_ylabel('Density')
    ax.set_title(f'Vmax_b for {sj_id}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Check if stuck at floor
    if Vmax_b_post.mean() < 1e-4:
        ax.text(0.5, 0.9, 'WARNING: Stuck at floor!',
                transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                fontsize=10, fontweight='bold')

    # ==========================================
    # Plot 4: K_a (EC50 for component a)
    # ==========================================
    ax = axes[1, 0]
    ax.hist(K_a_post, bins=50, density=True, alpha=0.7, label='Posterior', color='orange')

    # Prior for K: Gamma(K_alpha, K_alpha/K_max)
    # Default K_alpha = 2.0
    K_alpha = 2.0
    K_rate = K_alpha / K_max_est
    x = np.linspace(0.01, K_max_est * 3, 1000)
    prior_K = stats.gamma.pdf(x, K_alpha, scale=1/K_rate)
    ax.plot(x, prior_K, 'r-', linewidth=2, label=f'Prior Gamma({K_alpha:.1f}, rate={K_rate:.3f})')

    ax.axvline(K_a_post.mean(), color='orange', linestyle='--', linewidth=2, label='Posterior mean')
    ax.axvline(K_max_est, color='red', linestyle='--', linewidth=2, label=f'K_max={K_max_est:.1f}')
    ax.set_xlabel('K_a (EC50)')
    ax.set_ylabel('Density')
    ax.set_title(f'K_a for {sj_id}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==========================================
    # Plot 5: K_b (EC50 for component b)
    # ==========================================
    ax = axes[1, 1]
    ax.hist(K_b_post, bins=50, density=True, alpha=0.7, label='Posterior', color='brown')

    # Same prior as K_a
    ax.plot(x, prior_K, 'r-', linewidth=2, label=f'Prior Gamma({K_alpha:.1f}, rate={K_rate:.3f})')

    ax.axvline(K_b_post.mean(), color='brown', linestyle='--', linewidth=2, label='Posterior mean')
    ax.axvline(K_max_est, color='red', linestyle='--', linewidth=2, label=f'K_max={K_max_est:.1f}')
    ax.set_xlabel('K_b (EC50)')
    ax.set_ylabel('Density')
    ax.set_title(f'K_b for {sj_id}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ==========================================
    # Plot 6: Data overview
    # ==========================================
    ax = axes[1, 2]

    # Plot guide-level means
    if len(guide_means) > 0:
        ax.hist(guide_means, bins=30, alpha=0.7, label='Guide means', color='lightblue', edgecolor='black')
        ax.axvline(Amean_est, color='steelblue', linestyle='--', linewidth=2,
                   label=f'5th %ile = {Amean_est:.3f}')
        ax.axvline(Vmax_mean_est, color='darkred', linestyle='--', linewidth=2,
                   label=f'95th %ile = {Vmax_mean_est:.3f}')
        ax.axvline(A_post.mean(), color='blue', linestyle=':', linewidth=2,
                   label=f'Fit A = {A_post.mean():.3f}')

        # Show Vmax_a + Vmax_b
        if beta_post is not None:
            vmax_sum = Vmax_a_post.mean() + beta_post.mean() * Vmax_b_post.mean()
        else:
            vmax_sum = Vmax_a_post.mean() + Vmax_b_post.mean()
        ax.axvline(A_post.mean() + vmax_sum, color='purple', linestyle=':', linewidth=2,
                   label=f'Fit A+Vmax_sum = {A_post.mean() + vmax_sum:.3f}')

    ax.set_xlabel('PSI value')
    ax.set_ylabel('Count')
    ax.set_title(f'Guide-level PSI distribution\n({len(unique_guides)} guides)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")

    return fig


def compare_one_vs_two_groups(model_1grp, model_2grp, sj_id, modality_name='splicing_sj', save_path=None):
    """
    Compare posteriors between 1-group and 2-group fits.

    Parameters
    ----------
    model_1grp : bayesDREAM
        Model fitted with 1 technical group
    model_2grp : bayesDREAM
        Model fitted with 2 technical groups
    sj_id : str
        Splice junction ID
    modality_name : str
        Modality name
    save_path : str, optional
        Save path for figure

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Get modalities
    mod_1grp = model_1grp.get_modality(modality_name)
    mod_2grp = model_2grp.get_modality(modality_name)

    # Find SJ index
    if 'coord.intron' in mod_1grp.feature_meta.columns:
        sj_idx = mod_1grp.feature_meta[mod_1grp.feature_meta['coord.intron'] == sj_id].index[0]
    else:
        sj_idx = sj_id

    sj_position = list(mod_1grp.feature_meta.index).index(sj_idx)

    print(f"Comparing {sj_id} (position {sj_position})")

    # Extract posteriors
    def extract_params(modality, pos):
        post = modality.posterior_samples_trans
        def to_np(x):
            return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        return {
            'A': to_np(post['A'][:, pos]),
            'Vmax_a': to_np(post['Vmax_a'][:, pos]),
            'Vmax_b': to_np(post['Vmax_b'][:, pos]),
            'K_a': to_np(post['K_a'][:, pos]),
            'K_b': to_np(post['K_b'][:, pos]),
            'n_a': to_np(post['n_a'][:, pos]),
            'n_b': to_np(post['n_b'][:, pos]),
        }

    params_1grp = extract_params(mod_1grp, sj_position)
    params_2grp = extract_params(mod_2grp, sj_position)

    # Create comparison plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    param_names = ['A', 'Vmax_a', 'Vmax_b', 'K_a', 'K_b', 'n_a', 'n_b']
    colors = ['steelblue', 'darkgreen', 'purple', 'orange', 'brown', 'red', 'pink']

    for i, param in enumerate(param_names):
        ax = axes[i // 4, i % 4]

        # Plot histograms
        ax.hist(params_1grp[param], bins=30, density=True, alpha=0.5,
                label='1 group', color=colors[i], edgecolor='black')
        ax.hist(params_2grp[param], bins=30, density=True, alpha=0.5,
                label='2 groups', color='gray', edgecolor='black')

        # Add means
        ax.axvline(params_1grp[param].mean(), color=colors[i], linestyle='--', linewidth=2)
        ax.axvline(params_2grp[param].mean(), color='gray', linestyle='--', linewidth=2)

        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.set_title(f'{param}\n1grp: {params_1grp[param].mean():.3f}, 2grp: {params_2grp[param].mean():.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Last plot: summary statistics
    ax = axes[1, 3]
    ax.axis('off')

    summary_text = f"Summary for {sj_id}\n\n"
    for param in param_names:
        mean_1 = params_1grp[param].mean()
        mean_2 = params_2grp[param].mean()
        std_1 = params_1grp[param].std()
        std_2 = params_2grp[param].std()
        summary_text += f"{param}:\n"
        summary_text += f"  1grp: {mean_1:.3f} ± {std_1:.3f}\n"
        summary_text += f"  2grp: {mean_2:.3f} ± {std_2:.3f}\n\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison to {save_path}")

    return fig

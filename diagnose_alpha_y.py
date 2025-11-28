"""
Diagnostic script to check if alpha_y_add is causing inverted priors.

Run this after fit_technical() and before fit_trans() to check if technical
correction parameters are reasonable.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_alpha_y(model, modality_name='splicing_sj'):
    """
    Check if alpha_y parameters could cause inverted priors.

    Parameters
    ----------
    model : bayesDREAM
        Model after fit_technical()
    modality_name : str
        Modality to check
    """
    modality = model.get_modality(modality_name)

    # Get alpha_y_prefit
    if not hasattr(modality, 'alpha_y_prefit_add'):
        print("[ERROR] No alpha_y_prefit_add found. Run fit_technical() first.")
        return

    alpha_y = modality.alpha_y_prefit_add

    # Convert to numpy for analysis
    if isinstance(alpha_y, torch.Tensor):
        if alpha_y.dim() == 3:  # Posterior samples
            alpha_y_mean = alpha_y.mean(dim=0).cpu().numpy()  # [C, T]
        else:
            alpha_y_mean = alpha_y.cpu().numpy()  # [C, T]
    else:
        alpha_y_mean = np.array(alpha_y)

    C, T = alpha_y_mean.shape
    print(f"\n{'='*60}")
    print(f"Alpha_y Diagnostic for {modality_name}")
    print(f"{'='*60}")
    print(f"Number of technical groups: {C}")
    print(f"Number of features: {T}")

    # Check magnitude
    print(f"\nAlpha_y statistics (logit scale):")
    for c in range(C):
        alpha_c = alpha_y_mean[c, :]
        print(f"  Group {c}: mean={alpha_c.mean():.3f}, std={alpha_c.std():.3f}, "
              f"range=[{alpha_c.min():.3f}, {alpha_c.max():.3f}]")

    # Check for extreme values
    alpha_abs = np.abs(alpha_y_mean)
    n_extreme = (alpha_abs > 5).sum()
    pct_extreme = 100 * n_extreme / alpha_abs.size
    print(f"\nExtreme values (|alpha| > 5): {n_extreme}/{alpha_abs.size} ({pct_extreme:.1f}%)")

    if n_extreme > 0:
        print("[WARNING] Large alpha values may cause over-correction and inverted priors!")

    # Simulate prior computation to check for inversions
    print(f"\nSimulating prior computation...")

    # Get observed data
    if modality.cells_axis == 1:
        counts = modality.counts.T  # [cells, features]
    else:
        counts = modality.counts    # [cells, features]

    if hasattr(modality, 'denominator') and modality.denominator is not None:
        if modality.cells_axis == 1:
            denom = modality.denominator.T
        else:
            denom = modality.denominator

        # Compute proportions
        p_obs = counts / (denom + 1e-6)
        p_obs = np.clip(p_obs, 1e-6, 1-1e-6)
    else:
        print("[ERROR] No denominator found for binomial distribution")
        return

    # Get group assignments
    if not hasattr(model, 'technical_group_code'):
        print("[ERROR] No technical_group_code found. Run set_technical_groups() first.")
        return

    groups = model.meta['technical_group_code'].values
    if len(groups) != p_obs.shape[0]:
        print(f"[ERROR] Mismatch: {len(groups)} groups but {p_obs.shape[0]} cells")
        return

    # Apply inverse correction
    logit_obs = np.log(p_obs) - np.log(1 - p_obs)

    # Expand alpha_y to match observations
    alpha_expanded = alpha_y_mean[groups, :]  # [N_cells, T]
    logit_baseline = logit_obs - alpha_expanded
    p_baseline = 1 / (1 + np.exp(-logit_baseline))

    # Group by guide and compute means (as trans fitting does)
    guide_means = pd.DataFrame(p_baseline).groupby(model.meta['guide'].values).mean()

    # Compute priors per feature
    priors_A = guide_means.quantile(0.05, axis=0).values
    priors_Vmax = guide_means.quantile(0.95, axis=0).values

    # Check for inversions
    n_inverted = (priors_A > priors_Vmax).sum()
    pct_inverted = 100 * n_inverted / T

    print(f"\nPrior inversion check:")
    print(f"  Features with A > Vmax: {n_inverted}/{T} ({pct_inverted:.1f}%)")

    if n_inverted > 0:
        print(f"  [WARNING] {n_inverted} features have inverted priors!")
        print(f"  This will cause negative Vmax_sum and backwards Hill functions!")

        # Show examples
        inverted_idx = np.where(priors_A > priors_Vmax)[0]
        print(f"\n  Example inverted features (showing first 5):")
        for i in inverted_idx[:5]:
            print(f"    Feature {i}: A={priors_A[i]:.3f}, Vmax={priors_Vmax[i]:.3f}, "
                  f"alpha_y group1={alpha_y_mean[1, i]:.3f}")

    # Plot distribution of Vmax - A
    vmax_minus_a = priors_Vmax - priors_A

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(alpha_y_mean.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('alpha_y_add (logit scale)')
    plt.ylabel('Count')
    plt.title('Distribution of alpha_y values')
    plt.axvline(0, color='red', linestyle='--', label='Zero')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.hist(vmax_minus_a, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Vmax - A')
    plt.ylabel('Count')
    plt.title('Distribution of Vmax_sum\n(should all be > 0)')
    plt.axvline(0, color='red', linestyle='--', label='Zero')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(alpha_abs.flatten(), np.repeat(vmax_minus_a, C),
                alpha=0.3, s=1)
    plt.xlabel('|alpha_y_add|')
    plt.ylabel('Vmax - A')
    plt.title('Vmax_sum vs alpha magnitude')
    plt.axhline(0, color='red', linestyle='--')
    plt.axvline(5, color='orange', linestyle='--', label='|alpha|=5')
    plt.legend()

    plt.tight_layout()
    plt.savefig('alpha_y_diagnostic.png', dpi=150)
    print(f"\n  Saved diagnostic plot to: alpha_y_diagnostic.png")
    plt.show()

    return {
        'n_inverted': n_inverted,
        'pct_inverted': pct_inverted,
        'n_extreme_alpha': n_extreme,
        'priors_A': priors_A,
        'priors_Vmax': priors_Vmax,
        'alpha_y_mean': alpha_y_mean
    }


# Example usage:
# results = diagnose_alpha_y(model, modality_name='splicing_sj')

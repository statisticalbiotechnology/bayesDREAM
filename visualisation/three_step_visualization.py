"""
Three-step bayesDREAM visualization.

Creates a 3-panel figure showing:
1. Step 1 (Fit technical): Cis vs trans with technical group coloring
2. Step 2 (Fit cis): Observed cis vs x_true with guide coloring + density
3. Step 3 (Fit trans): x_true vs trans with dose-response curve

Requires a fully fitted bayesDREAM model (technical, cis, and trans fitted).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from typing import Optional, Dict, Tuple
import warnings


def knn_smooth(x_values, y_values, k=30):
    """
    Apply k-NN smoothing to y values based on x coordinates.

    Parameters
    ----------
    x_values : np.ndarray
        X coordinates (1D)
    y_values : np.ndarray
        Y values to smooth (1D)
    k : int
        Number of neighbors (default: 30)

    Returns
    -------
    y_smoothed : np.ndarray
        Smoothed y values
    """
    # Build KDTree
    tree = cKDTree(x_values.reshape(-1, 1))

    # Query k nearest neighbors for each point
    distances, indices = tree.query(x_values.reshape(-1, 1), k=k)

    # Average y values of k nearest neighbors
    y_smoothed = np.array([y_values[idx].mean() for idx in indices])

    return y_smoothed


def get_purple_palette(n_groups, reference_idx=None):
    """
    Generate purple color palette for technical groups.

    Parameters
    ----------
    n_groups : int
        Number of technical groups
    reference_idx : int, optional
        Index of reference group (will be grey)

    Returns
    -------
    colors : list
        List of color strings
    """
    if n_groups == 1:
        return ['grey']

    # Purple shades
    purples = plt.cm.Purples(np.linspace(0.4, 0.9, n_groups))
    colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
              for r, g, b, a in purples]

    # Set reference group to grey if specified
    if reference_idx is not None:
        colors[reference_idx] = 'grey'

    return colors


def get_green_grey_palette(guides, ntc_guides):
    """
    Generate green/grey color palette for guides.

    Parameters
    ----------
    guides : list
        All guide names
    ntc_guides : list
        NTC guide names

    Returns
    -------
    color_map : dict
        Dictionary mapping guide names to colors
    """
    # Separate targeted and NTC guides
    targeted = [g for g in guides if g not in ntc_guides]

    color_map = {}

    # NTC guides: grey shades
    if len(ntc_guides) > 0:
        greys = plt.cm.Greys(np.linspace(0.3, 0.7, len(ntc_guides)))
        for i, guide in enumerate(ntc_guides):
            r, g, b, a = greys[i]
            color_map[guide] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    # Targeted guides: green shades
    if len(targeted) > 0:
        greens = plt.cm.Greens(np.linspace(0.4, 0.9, len(targeted)))
        for i, guide in enumerate(targeted):
            r, g, b, a = greens[i]
            color_map[guide] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    return color_map


def plot_three_steps(
    model,
    trans_feature: str,
    sum_factor_col: str = 'sum_factor',
    k_smooth: int = 30,
    show_correction_vector: bool = True,
    reference_group: Optional[str] = None,
    technical_subsample_frac: float = 0.1,
    n_guides_cis: int = 10,
    figsize: Tuple[int, int] = (18, 5),
    alpha: float = 0.5,
    s: float = 5,
    save_path: Optional[str] = None
):
    """
    Create 3-panel visualization of bayesDREAM fitting steps.

    Parameters
    ----------
    model : bayesDREAM
        Fully fitted model (technical, cis, and trans)
    trans_feature : str
        Name of trans feature to visualize
    sum_factor_col : str
        Sum factor column in model.meta (default: 'sum_factor')
    k_smooth : int
        k for k-NN smoothing in step 3 (default: 30)
    show_correction_vector : bool
        Show correction vector in step 1 (default: True)
    reference_group : str, optional
        Name of reference technical group (will be grey)
    technical_subsample_frac : float
        Fraction of cells to randomly sample per technical group in step 1 (default: 0.1)
    n_guides_cis : int
        Number of guides to show in step 2 cis plot (default: 10)
        Randomly samples from all available guides
    figsize : tuple
        Figure size (default: (18, 5))
    alpha : float
        Point transparency (default: 0.5)
    s : float
        Point size (default: 5)
    save_path : str, optional
        Path to save figure

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure

    Raises
    ------
    ValueError
        If required fitting steps not completed
        If trans_feature not found
    """
    # Validate that all steps are fitted
    if not hasattr(model, 'x_true') or model.x_true is None:
        raise ValueError("x_true not set. Must run fit_cis() before visualization.")

    if not hasattr(model, 'posterior_samples_trans') or model.posterior_samples_trans is None:
        raise ValueError("Trans model not fitted. Must run fit_trans() before visualization.")

    if 'technical_group_code' not in model.meta.columns:
        raise ValueError("Technical groups not set. Must run set_technical_groups() before visualization.")

    # Get modality
    if 'cis' not in model.modalities:
        raise ValueError("'cis' modality not found.")

    # Get cis counts
    cis_counts = model.get_modality('cis').counts.flatten()
    sum_factors = model.meta[sum_factor_col].values

    # Get trans feature counts
    # Try to find in primary modality
    primary_mod = model.get_modality(model.primary_modality)

    # Find trans feature index
    if trans_feature in primary_mod.feature_meta.index:
        trans_idx = list(primary_mod.feature_meta.index).index(trans_feature)
    else:
        # Try gene columns
        found = False
        for col in ['gene', 'gene_name', 'gene_id']:
            if col in primary_mod.feature_meta.columns:
                mask = primary_mod.feature_meta[col] == trans_feature
                if mask.sum() > 0:
                    trans_idx = mask.values.nonzero()[0][0]
                    found = True
                    break
        if not found:
            raise ValueError(f"Trans feature '{trans_feature}' not found in primary modality")

    trans_counts = primary_mod.counts[trans_idx, :]

    # Normalize and log transform
    cis_norm = np.log2(cis_counts / sum_factors + 1)
    trans_norm = np.log2(trans_counts / sum_factors + 1)

    # Get x_true (mean if posterior, otherwise use directly)
    # Handle both tensor and numpy array
    import torch

    if isinstance(model.x_true, torch.Tensor):
        x_true_arr = model.x_true.cpu().numpy()
    else:
        x_true_arr = np.array(model.x_true)

    # If 2D (posterior samples), take mean over samples
    if x_true_arr.ndim == 2:
        x_true = x_true_arr.mean(axis=0)
    else:
        x_true = x_true_arr.ravel()  # Ensure 1D

    log2_x_true = np.log2(x_true)

    # Get technical groups
    tech_groups = model.meta['technical_group_code'].values
    unique_groups = sorted(model.meta['technical_group_code'].unique())

    # Get labels - check if technical_group_label column exists
    if 'technical_group_label' in model.meta.columns:
        tech_labels = model.meta['technical_group_label'].values
        unique_labels = [model.meta[model.meta['technical_group_code'] == g]['technical_group_label'].iloc[0]
                         for g in unique_groups]
    else:
        # Use the code directly as label (or try to get from cell_line/group columns)
        if 'cell_line' in model.meta.columns:
            # Map code to cell_line (assumes they're aligned)
            code_to_label = {}
            for code in unique_groups:
                mask = model.meta['technical_group_code'] == code
                # Get most common cell_line for this code
                cell_line = model.meta[mask]['cell_line'].mode()[0] if mask.sum() > 0 else f"Group_{code}"
                code_to_label[code] = cell_line
            unique_labels = [code_to_label[g] for g in unique_groups]
            tech_labels = np.array([code_to_label[code] for code in tech_groups])
        else:
            # Fallback: use codes as labels
            unique_labels = [f"Group_{g}" for g in unique_groups]
            tech_labels = np.array([f"Group_{code}" for code in tech_groups])

    # Get guide information
    if 'guide' not in model.meta.columns:
        raise ValueError("'guide' column not found in model.meta")

    guides = model.meta['guide'].values
    unique_guides = sorted(model.meta['guide'].unique())

    # Identify NTC guides
    if 'target' in model.meta.columns:
        ntc_guides = model.meta[model.meta['target'] == 'ntc']['guide'].unique().tolist()
    else:
        # Heuristic: guides with 'ntc' or 'NTC' in name
        ntc_guides = [g for g in unique_guides if 'ntc' in str(g).lower()]

    # Get color palettes
    reference_idx = None
    if reference_group is not None:
        try:
            reference_idx = unique_labels.index(reference_group)
        except ValueError:
            warnings.warn(f"Reference group '{reference_group}' not found. Using default colors.")

    tech_colors = get_purple_palette(len(unique_groups), reference_idx=reference_idx)
    guide_colors = get_green_grey_palette(unique_guides, ntc_guides)

    # Create figure with gridspec
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.8, 0.2, 1], wspace=0.4)

    ax1 = fig.add_subplot(gs[0])  # Step 1
    ax2 = fig.add_subplot(gs[1])  # Step 2 (scatter)
    ax3 = fig.add_subplot(gs[2])  # Step 2 (density)
    ax4 = fig.add_subplot(gs[3])  # Step 3

    # ========================================================================
    # Step 1: Fit technical
    # ========================================================================
    ax1.set_title("Step 1: Fit technical", fontsize=12, fontweight='bold')

    # Subsample cells per technical group
    np.random.seed(42)  # For reproducibility
    sampled_indices = []
    for group_code in unique_groups:
        group_mask = tech_groups == group_code
        group_indices = np.where(group_mask)[0]
        n_sample = max(1, int(len(group_indices) * technical_subsample_frac))
        sampled_idx = np.random.choice(group_indices, size=n_sample, replace=False)
        sampled_indices.extend(sampled_idx)

    sampled_indices = np.array(sampled_indices)

    # Plot each technical group (subsampled)
    for i, (group_code, label) in enumerate(zip(unique_groups, unique_labels)):
        mask = tech_groups[sampled_indices] == group_code
        # Lighter alpha for better visibility of correction vector
        ax1.scatter(cis_norm[sampled_indices][mask], trans_norm[sampled_indices][mask],
                   c=tech_colors[i], alpha=alpha * 0.7, s=s, label=label)

    # Show correction vector if requested
    correction_handle = None
    if show_correction_vector and len(unique_groups) > 1:
        # Pick two example points from different groups (from subsampled data)
        group0_mask = tech_groups[sampled_indices] == unique_groups[0]
        group1_mask = tech_groups[sampled_indices] == unique_groups[1]
        group0_pts = sampled_indices[group0_mask]
        group1_pts = sampled_indices[group1_mask]

        if len(group0_pts) > 0 and len(group1_pts) > 0:
            # Find closest pair
            pt0 = group0_pts[len(group0_pts)//2]  # Middle point
            pt1_candidates = group1_pts[np.argsort(np.abs(cis_norm[group1_pts] - cis_norm[pt0]))]
            pt1 = pt1_candidates[0]

            # Draw correction vector (more visible - red color)
            ax1.annotate('', xy=(cis_norm[pt0], trans_norm[pt0]),
                        xytext=(cis_norm[pt1], trans_norm[pt1]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.9))

            # Add to legend by creating invisible line
            correction_handle = plt.Line2D([0], [0], color='red', lw=2.5, label='Correction vector')

    ax1.set_xlabel(f"log2({model.cis_gene} + 1)", fontsize=10)
    ax1.set_ylabel(f"log2({trans_feature} + 1)", fontsize=10)

    # Add legend with correction vector if shown
    handles, labels = ax1.get_legend_handles_labels()
    if correction_handle is not None:
        handles.append(correction_handle)
        labels.append('Correction vector')
    ax1.legend(handles, labels, fontsize=8, frameon=False, title='Technical groups')
    ax1.grid(alpha=0.3)

    # ========================================================================
    # Step 2: Fit cis (Scatter + Density)
    # ========================================================================
    ax2.set_title("Step 2: Fit cis", fontsize=12, fontweight='bold')

    # Subsample guides
    n_guides_to_plot = min(n_guides_cis, len(unique_guides))
    if len(unique_guides) > n_guides_to_plot:
        # Randomly select guides
        np.random.seed(42)
        selected_guides = np.random.choice(unique_guides, size=n_guides_to_plot, replace=False)
    else:
        selected_guides = unique_guides

    # Plot scatter by guide (subsampled)
    for guide in selected_guides:
        mask = guides == guide
        ax2.scatter(cis_norm[mask], log2_x_true[mask],
                   c=guide_colors[guide], alpha=alpha, s=s, label=guide)

    ax2.set_xlabel(f"Observed log2({model.cis_gene} + 1)", fontsize=10)
    ax2.set_ylabel(f"log2(x_true)", fontsize=10)
    # Too many guides for legend, skip it
    ax2.grid(alpha=0.3)

    # Density plots (rotated) - one plot per guide, all on same axes
    ax3.set_title("", fontsize=10)
    ax3.set_ylabel(f"log2(x_true)", fontsize=10)
    ax3.set_xlabel("Density", fontsize=8)

    # Plot density for each selected guide
    y_min, y_max = log2_x_true.min(), log2_x_true.max()
    y_grid = np.linspace(y_min, y_max, 200)

    max_density = 0  # Track max for x-axis scaling
    for guide in selected_guides:
        mask = guides == guide
        guide_x_true = log2_x_true[mask]

        if len(guide_x_true) > 5:  # Need enough points for KDE
            kde = gaussian_kde(guide_x_true, bw_method='scott')
            density = kde(y_grid)

            # Plot density curve (all start from x=0, no offset)
            ax3.plot(density, y_grid, c=guide_colors[guide], lw=1.5, alpha=0.7)
            ax3.fill_betweenx(y_grid, 0, density,
                             color=guide_colors[guide], alpha=0.2)

            max_density = max(max_density, density.max())

    ax3.set_ylim(ax2.get_ylim())
    ax3.set_xlim(0, max_density * 1.1)
    ax3.set_xticks([])
    ax3.grid(alpha=0.3, axis='y')

    # ========================================================================
    # Step 3: Fit trans
    # ========================================================================
    ax4.set_title("Step 3: Fit trans", fontsize=12, fontweight='bold')

    # k-NN smooth trans counts
    trans_smooth = knn_smooth(log2_x_true, trans_norm, k=k_smooth)

    # Plot smoothed data by technical group
    for i, (group_code, label) in enumerate(zip(unique_groups, unique_labels)):
        mask = tech_groups == group_code
        ax4.scatter(log2_x_true[mask], trans_smooth[mask],
                   c=tech_colors[i], alpha=alpha, s=s, label=label)

    # Overlay Hill function if available
    # Check both model-level and modality-level posteriors
    posterior = None
    if hasattr(model, 'posterior_samples_trans') and model.posterior_samples_trans is not None:
        posterior = model.posterior_samples_trans
    elif hasattr(primary_mod, 'posterior_samples_trans') and primary_mod.posterior_samples_trans is not None:
        posterior = primary_mod.posterior_samples_trans

    if posterior is not None and 'A' in posterior:
        try:
            # Get posterior means
            A_samps = posterior['A']

            # For modality-level posteriors, shape is (samples, cis_genes, trans_features)
            # For model-level, shape is (samples, trans_features)
            if hasattr(A_samps, 'mean'):
                A_mean = A_samps.mean(dim=0)
            else:
                A_mean = A_samps.mean(axis=0)

            # If 2D (modality-level), squeeze out cis dimension
            if A_mean.ndim == 2:
                A_mean = A_mean.squeeze(0)  # Assume single cis gene

            A = A_mean[trans_idx].item() if hasattr(A_mean, 'item') else A_mean[trans_idx]

            # Check if additive Hill
            if 'n_a' in posterior and 'n_b' in posterior:
                def extract_mean(param_samps, idx):
                    """Extract mean for a parameter, handling 2D and 3D shapes."""
                    if hasattr(param_samps, 'mean'):
                        param_mean = param_samps.mean(dim=0)
                    else:
                        param_mean = param_samps.mean(axis=0)
                    # Squeeze if 2D (modality-level)
                    if param_mean.ndim == 2:
                        param_mean = param_mean.squeeze(0)
                    return param_mean[idx].item() if hasattr(param_mean, 'item') else param_mean[idx]

                # Extract all Hill parameters
                Vmax_a = extract_mean(posterior['Vmax_a'], trans_idx)
                Vmax_b = extract_mean(posterior['Vmax_b'], trans_idx)
                K_a = extract_mean(posterior['K_a'], trans_idx)
                K_b = extract_mean(posterior['K_b'], trans_idx)
                n_a = extract_mean(posterior['n_a'], trans_idx)
                n_b = extract_mean(posterior['n_b'], trans_idx)
                alpha_val = extract_mean(posterior['alpha'], trans_idx)
                beta_val = extract_mean(posterior['beta'], trans_idx)

                # Compute Hill function
                x_range = np.linspace(log2_x_true.min(), log2_x_true.max(), 200)
                x_range_lin = 2**x_range  # Convert back to linear scale

                # Hill function (in linear space)
                Hill_a = Vmax_a * (x_range_lin**n_a) / (K_a**n_a + x_range_lin**n_a + 1e-10)
                Hill_b = Vmax_b * (x_range_lin**n_b) / (K_b**n_b + x_range_lin**n_b + 1e-10)
                y_pred = A + alpha_val * Hill_a + beta_val * Hill_b

                # Convert to log2 space for plotting
                y_pred_log = np.log2(y_pred + 1)

                ax4.plot(x_range, y_pred_log, 'k-', lw=2.5, alpha=0.8, label='Fitted Hill function')
        except Exception as e:
            warnings.warn(f"Could not plot Hill function: {e}")
            import traceback
            traceback.print_exc()

    ax4.set_xlabel(f"log2(x_true)", fontsize=10)
    ax4.set_ylabel(f"log2({trans_feature} + 1) [k={k_smooth} smoothed]", fontsize=10)
    ax4.legend(fontsize=8, frameon=False, title='Technical groups')
    ax4.grid(alpha=0.3)

    # Save if requested
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


# Example usage
if __name__ == "__main__":
    # This would be run after fitting
    # fig = plot_three_steps(model, 'GAPDH', k_smooth=30, save_path='three_steps.png')
    # plt.show()
    pass

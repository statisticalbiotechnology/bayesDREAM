"""
Simulate negative binomial count data from fitted trans summary parameters.

Given a trans summary CSV (from save_trans_summary), cell metadata, and per-cell
x_true values, this module reconstructs the additive Hill dose-response and
samples NegBin observations to produce a synthetic count matrix suitable for
re-fitting with bayesDREAM.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


def _hill(x, Vmax, K, n, eps=1e-12):
    """Evaluate Hill function: Vmax * x^n / (K^n + x^n)."""
    x_safe = np.maximum(x, eps)
    K_safe = np.maximum(K, eps)
    x_n = np.power(x_safe, n)
    K_n = np.power(K_safe, n)
    return Vmax * x_n / (K_n + x_n + eps)


def simulate_from_trans_summary(
    trans_summary_df: pd.DataFrame,
    meta: pd.DataFrame,
    x_true: Union[np.ndarray, pd.Series],
    x_counts: Union[np.ndarray, pd.Series],
    cis_gene: str,
    genes: Optional[list] = None,
    sum_factor_col: str = 'sum_factor',
    group_col: str = 'technical_group_code',
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate NegBin count data from fitted trans summary parameters.

    Reconstructs the additive Hill dose-response for each gene, applies
    technical group effects and sum factors, then draws NegBin samples.

    The generative model is:
        y_pred = A + alpha * Hill(x; Vmax_a, K_a, n_a) + beta * Hill(x; Vmax_b, K_b, n_b)
        mu_final = y_pred * alpha_y[group] * sum_factor
        y_obs ~ NegBin(total_count=phi_y, logits=log(mu_final) - log(phi_y))

    Parameters
    ----------
    trans_summary_df : pd.DataFrame
        Output of save_trans_summary, must contain Hill parameters
        (A_mean, Vmax_a_mean, K_a_mean, n_a_mean, alpha_mean,
        Vmax_b_mean, K_b_mean, n_b_mean, beta_mean) and phi_y_mean.
        If technical groups are used, must contain group_{g}_alpha_y_mean columns.
    meta : pd.DataFrame
        Cell metadata with columns: cell, guide, target, and sum_factor_col.
        If technical groups are used, must contain group_col.
    x_true : array-like, shape (n_cells,)
        Per-cell x_true values (cis gene expression), aligned to meta rows.
    x_counts : array-like, shape (n_cells,)
        Per-cell raw counts for the cis gene, used as-is in the output matrix.
    cis_gene : str
        Name of the cis gene (added as a row in the output count matrix).
    genes : list of str, optional
        Subset of genes to simulate. Default: all genes in trans_summary_df.
    sum_factor_col : str
        Column in meta for sum factors (default: 'sum_factor').
    group_col : str
        Column in meta for technical group codes (default: 'technical_group_code').
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    counts_df : pd.DataFrame
        Simulated counts with genes as rows and cells as columns.
        Includes both trans genes and the cis gene row.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    x_true = np.asarray(x_true, dtype=float)
    x_counts = np.asarray(x_counts, dtype=float)
    n_cells = len(x_true)

    # Validate meta alignment
    if len(meta) != n_cells:
        raise ValueError(
            f"meta has {len(meta)} rows but x_true has {n_cells} elements"
        )

    sum_factors = meta[sum_factor_col].values.astype(float)

    # Determine which genes to simulate
    df = trans_summary_df.copy()
    if genes is not None:
        df = df[df['feature'].isin(genes)].reset_index(drop=True)
    gene_names = df['feature'].values
    n_genes = len(gene_names)

    if n_genes == 0:
        raise ValueError("No genes to simulate after filtering.")

    # Detect function type
    function_type = df['function_type'].iloc[0]

    # Check for required columns
    required_cols = ['A_mean', 'phi_y_mean']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Extract Hill parameters (per gene)
    A = df['A_mean'].values  # (n_genes,)
    phi_y = df['phi_y_mean'].values  # (n_genes,)

    # Compute y_pred for each cell and gene
    # y_pred shape: (n_genes, n_cells)
    if function_type == 'additive_hill':
        for col in ['Vmax_a_mean', 'K_a_mean', 'n_a_mean', 'alpha_mean',
                     'Vmax_b_mean', 'K_b_mean', 'n_b_mean', 'beta_mean']:
            if col not in df.columns:
                raise ValueError(f"Missing required column for additive_hill: {col}")

        Vmax_a = df['Vmax_a_mean'].values
        K_a = df['K_a_mean'].values
        n_a = df['n_a_mean'].values
        alpha = df['alpha_mean'].values
        Vmax_b = df['Vmax_b_mean'].values
        K_b = df['K_b_mean'].values
        n_b = df['n_b_mean'].values
        beta = df['beta_mean'].values

        # x_true is (n_cells,), broadcast with gene params (n_genes,)
        # Expand: x -> (1, n_cells), params -> (n_genes, 1)
        x_exp = x_true[np.newaxis, :]  # (1, n_cells)

        hill_a = _hill(x_exp, Vmax_a[:, np.newaxis], K_a[:, np.newaxis], n_a[:, np.newaxis])
        hill_b = _hill(x_exp, Vmax_b[:, np.newaxis], K_b[:, np.newaxis], n_b[:, np.newaxis])

        y_pred = (A[:, np.newaxis]
                  + alpha[:, np.newaxis] * hill_a
                  + beta[:, np.newaxis] * hill_b)  # (n_genes, n_cells)

    elif function_type == 'single_hill':
        for col in ['Vmax_a_mean', 'K_a_mean', 'n_a_mean', 'alpha_mean']:
            if col not in df.columns:
                raise ValueError(f"Missing required column for single_hill: {col}")

        Vmax_a = df['Vmax_a_mean'].values
        K_a = df['K_a_mean'].values
        n_a = df['n_a_mean'].values
        alpha = df['alpha_mean'].values

        x_exp = x_true[np.newaxis, :]
        hill_a = _hill(x_exp, Vmax_a[:, np.newaxis], K_a[:, np.newaxis], n_a[:, np.newaxis])

        y_pred = A[:, np.newaxis] + alpha[:, np.newaxis] * hill_a

    elif function_type == 'polynomial':
        # Find coefficient columns
        coef_cols = sorted(
            [c for c in df.columns if c.startswith('coef_') and c.endswith('_mean')],
            key=lambda c: int(c.split('_')[1])
        )
        if not coef_cols:
            raise ValueError("No polynomial coefficient columns found.")

        x_exp = x_true[np.newaxis, :]  # (1, n_cells)
        y_pred = np.zeros((n_genes, n_cells))
        for i, col in enumerate(coef_cols):
            y_pred += df[col].values[:, np.newaxis] * np.power(x_exp, i)

    else:
        raise ValueError(f"Unsupported function_type: {function_type}")

    # Ensure y_pred is positive for NegBin
    y_pred = np.maximum(y_pred, 1e-6)

    # Apply technical group effects (alpha_y)
    alpha_y_cols = sorted(
        [c for c in df.columns if c.startswith('group_') and c.endswith('_alpha_y_mean')],
        key=lambda c: int(c.split('_')[1])
    )

    if alpha_y_cols and group_col in meta.columns:
        n_groups = len(alpha_y_cols)
        groups = meta[group_col].values.astype(int)

        # Build alpha_y matrix: (n_genes, n_groups)
        alpha_y_matrix = np.column_stack([df[col].values for col in alpha_y_cols])

        # Map each cell to its group's alpha_y: (n_genes, n_cells)
        alpha_y_per_cell = alpha_y_matrix[:, groups]  # (n_genes, n_cells)
        mu_final = y_pred * alpha_y_per_cell * sum_factors[np.newaxis, :]
    else:
        # No technical groups: just apply sum factors
        mu_final = y_pred * sum_factors[np.newaxis, :]

    mu_final = np.maximum(mu_final, 1e-10)

    # Sample from NegBin: y ~ NegBin(total_count=phi_y, prob=phi_y/(phi_y+mu))
    # PyTorch/Pyro NegBin parameterization: total_count=phi_y, logits=log(mu)-log(phi_y)
    # numpy parameterization: n=phi_y, p=phi_y/(phi_y+mu)
    phi_y_exp = phi_y[:, np.newaxis]  # (n_genes, 1)
    prob = phi_y_exp / (phi_y_exp + mu_final)  # (n_genes, n_cells)

    # Clip probabilities to valid range
    prob = np.clip(prob, 1e-10, 1 - 1e-10)

    y_obs = rng.negative_binomial(n=phi_y_exp, p=prob)  # (n_genes, n_cells)

    # Build output DataFrame
    cells = meta['cell'].values if 'cell' in meta.columns else [f'cell_{i}' for i in range(n_cells)]

    counts_dict = {}
    for i, gene in enumerate(gene_names):
        counts_dict[gene] = y_obs[i, :]

    # Add cis gene row from raw counts
    counts_dict[cis_gene] = x_counts.astype(int)

    counts_df = pd.DataFrame(counts_dict, index=cells).T
    counts_df.index.name = None

    return counts_df

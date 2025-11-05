"""
Helper utility functions for bayesDREAM plotting.
"""

import numpy as np


def to_np(a):
    """
    Safely convert torch/array-like to numpy.

    Parameters
    ----------
    a : torch.Tensor or array-like
        Input to convert

    Returns
    -------
    np.ndarray
        Numpy array
    """
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)


def per_cell_mean_std(x):
    """
    Compute per-cell mean and std along axis 0 (samples x cells).

    Parameters
    ----------
    x : array-like, shape (n_samples, n_cells)
        Posterior samples

    Returns
    -------
    mean : np.ndarray, shape (n_cells,)
        Per-cell mean across samples
    std : np.ndarray, shape (n_cells,)
        Per-cell std across samples
    """
    x_np = to_np(x)
    return x_np.mean(axis=0), x_np.std(axis=0)

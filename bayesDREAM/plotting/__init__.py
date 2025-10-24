"""
Plotting functions for bayesDREAM goodness-of-fit visualizations.

This module provides functions to visualize prior/posterior distributions for model parameters.
"""

from .prior_posterior import (
    plot_scalar_parameter,
    plot_1d_parameter,
    plot_2d_parameter
)

__all__ = [
    'plot_scalar_parameter',
    'plot_1d_parameter',
    'plot_2d_parameter'
]

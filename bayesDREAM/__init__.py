"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A Bayesian framework for modeling perturbation effects across multiple
molecular modalities using PyTorch and Pyro.
"""

from .model import bayesDREAM
from .modality import Modality

__version__ = "1.0.0"
__all__ = ["bayesDREAM", "Modality"]

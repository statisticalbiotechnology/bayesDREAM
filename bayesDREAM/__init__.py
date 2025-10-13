"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A framework for modeling CRISPR perturbation effects across multiple molecular modalities.
"""

from .model import bayesDREAM
from .modality import Modality
from .multimodal import MultiModalBayesDREAM
from .splicing import (
    create_splicing_modality,
    process_donor_usage,
    process_acceptor_usage,
    process_exon_skipping
)
from .distributions import (
    get_observation_sampler,
    requires_denominator,
    is_3d_distribution,
    DISTRIBUTION_REGISTRY
)

__version__ = "0.2.0"

__all__ = [
    'bayesDREAM',
    'MultiModalBayesDREAM',
    'Modality',
    'create_splicing_modality',
    'process_donor_usage',
    'process_acceptor_usage',
    'process_exon_skipping',
    'get_observation_sampler',
    'requires_denominator',
    'is_3d_distribution',
    'DISTRIBUTION_REGISTRY',
]

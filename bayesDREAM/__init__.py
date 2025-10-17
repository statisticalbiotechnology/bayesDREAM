"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A Bayesian framework for modeling perturbation effects across multiple molecular modalities.
"""

from .model import bayesDREAM
from .modality import Modality
from .splicing import (
    create_splicing_modality,
    process_donor_usage,
    process_acceptor_usage,
    process_exon_skipping
)
from .distributions import (
    get_observation_sampler,
    requires_denominator,
    requires_sum_factor,
    is_3d_distribution,
    supports_cell_line_effects,
    get_cell_line_effect_type,
    DISTRIBUTION_REGISTRY
)
from .utils import (
    set_max_threads,
    Hill_based_positive,
    Hill_based_negative,
    Hill_based_piecewise,
    Polynomial_function
)

__version__ = "1.0.0"

__all__ = [
    'bayesDREAM',
    'Modality',
    'create_splicing_modality',
    'process_donor_usage',
    'process_acceptor_usage',
    'process_exon_skipping',
    'get_observation_sampler',
    'requires_denominator',
    'requires_sum_factor',
    'is_3d_distribution',
    'supports_cell_line_effects',
    'get_cell_line_effect_type',
    'DISTRIBUTION_REGISTRY',
    'set_max_threads',
    'Hill_based_positive',
    'Hill_based_negative',
    'Hill_based_piecewise',
    'Polynomial_function',
]

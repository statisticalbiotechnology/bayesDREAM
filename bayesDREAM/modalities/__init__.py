"""
Modality management for bayesDREAM.

This module contains mixins for adding different data modalities.
"""

from .transcript import TranscriptModalityMixin
from .splicing_modality import SplicingModalityMixin
from .atac import ATACModalityMixin
from .custom import CustomModalityMixin
from .plotting_mixin import PlottingMixin

__all__ = [
    'TranscriptModalityMixin',
    'SplicingModalityMixin',
    'ATACModalityMixin',
    'CustomModalityMixin',
    'PlottingMixin'
]

"""
Fitting methods for bayesDREAM.

This module contains the model fitting logic separated by stage:
- technical: Technical variation fitting
- cis: Cis gene expression fitting
- trans: Trans effects fitting
"""

from .technical import TechnicalFitter
from .cis import CisFitter
from .trans import TransFitter

__all__ = ['TechnicalFitter', 'CisFitter', 'TransFitter']

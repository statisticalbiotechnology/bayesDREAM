"""
Color scheme management for bayesDREAM plots.

Provides consistent coloring across all visualization functions for targets, guides,
and technical groups.
"""

import numpy as np
import matplotlib.colors as mcolors
from matplotlib import cm


class ColorScheme:
    """
    Manages consistent color schemes for bayesDREAM visualizations.

    Attributes
    ----------
    palette : dict
        Target-specific color palettes (colormaps with linspace gradients)
    guide_colors : dict
        Guide-specific colors built from palette
    target_colors : dict
        Target-level representative colors
    """

    def __init__(self, palette=None):
        """
        Initialize ColorScheme.

        Parameters
        ----------
        palette : dict, optional
            Target -> list of colors mapping. If None, uses default palette.
            Example: {'GFI1B': [cm.Greens(i) for i in np.linspace(0.4, 0.9, 3)]}
        """
        if palette is None:
            # Default palette based on 10x.ipynb
            self.palette = {
                'GFI1B': [cm.Greens(i) for i in np.linspace(0.4, 0.9, 3)],
                'NTC':   [cm.Greys(i)  for i in np.linspace(0.4, 0.8, 5)],
                'GEMIN5': [cm.Blues(i)  for i in np.linspace(0.4, 0.8, 2)],
                'DDX6':  [cm.Reds(i)   for i in np.linspace(0.4, 0.8, 3)],
            }
        else:
            self.palette = palette

        # Build guide and target colors
        self.guide_colors = build_guide_colors(self.palette)
        self.target_colors = self._build_target_colors()

    def _build_target_colors(self):
        """Build target-level representative colors from palette."""
        target_colors = {}
        for gene, colors in self.palette.items():
            # Use middle color as representative
            target_colors[gene] = colors[len(colors) // 2]
            # Also handle lowercase 'ntc'
            if gene.upper() == 'NTC':
                target_colors['ntc'] = colors[len(colors) // 2]
        return target_colors

    def get_guide_color(self, guide, default='black'):
        """Get color for a specific guide."""
        return self.guide_colors.get(str(guide), default)

    def get_target_color(self, target, default='gray'):
        """Get color for a specific target."""
        return self.target_colors.get(str(target), default)

    @classmethod
    def from_targets(cls, targets, colormaps=None, n_guides_per_target=None):
        """
        Create ColorScheme from target list.

        Parameters
        ----------
        targets : list of str
            Target names
        colormaps : list of str or colormaps, optional
            Colormap for each target. If None, uses default rotation.
        n_guides_per_target : dict or int, optional
            Number of guides per target. If int, uses same for all.
            If None, defaults to 3.

        Returns
        -------
        ColorScheme
        """
        if colormaps is None:
            # Default colormap rotation
            default_cmaps = [cm.Greens, cm.Blues, cm.Reds, cm.Purples,
                           cm.Oranges, cm.YlOrBr, cm.PuBu]
            colormaps = [default_cmaps[i % len(default_cmaps)]
                        for i in range(len(targets))]

        if n_guides_per_target is None:
            n_guides_per_target = 3

        if isinstance(n_guides_per_target, int):
            n_guides = {t: n_guides_per_target for t in targets}
        else:
            n_guides = n_guides_per_target

        palette = {}
        for target, cmap in zip(targets, colormaps):
            n = n_guides.get(target, 3)
            # Special handling for NTC (use greys)
            if target.upper() == 'NTC':
                palette[target] = [cm.Greys(i) for i in np.linspace(0.4, 0.8, n)]
            else:
                palette[target] = [cmap(i) for i in np.linspace(0.4, 0.9, n)]

        return cls(palette=palette)


def build_guide_colors(palette_dict):
    """
    Build guide-level colors from target palette.

    Parameters
    ----------
    palette_dict : dict
        Target -> list of colors mapping

    Returns
    -------
    dict
        Guide -> color mapping (e.g., 'GFI1B_1' -> color)
    """
    guide_colors = {}
    for gene, colors in palette_dict.items():
        for i, color in enumerate(colors, start=1):
            guide_colors[f"{gene}_{i}"] = color
    return guide_colors


def lighten(color, amount=0.3):
    """
    Lighten an RGBA/RGB color by mixing with white.

    Parameters
    ----------
    color : color spec
        Any matplotlib color specification
    amount : float
        Amount to lighten (0=no change, 1=white)

    Returns
    -------
    tuple
        RGBA color tuple
    """
    c = np.array(mcolors.to_rgba(color))
    white = np.array([1, 1, 1, 1])
    return tuple((1 - amount) * c + amount * white)


def darken(color, amount=0.3):
    """
    Darken an RGBA/RGB color by mixing with black.

    Parameters
    ----------
    color : color spec
        Any matplotlib color specification
    amount : float
        Amount to darken (0=no change, 1=black)

    Returns
    -------
    tuple
        RGBA color tuple
    """
    c = np.array(mcolors.to_rgba(color))
    black = np.array([0, 0, 0, 1])
    return tuple((1 - amount) * c + amount * black)

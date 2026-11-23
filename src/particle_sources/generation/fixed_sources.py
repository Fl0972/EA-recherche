"""
Sources déterministes, toujours présentes, indépendantes du tirage aléatoire.
Elles ne comptent ni dans N ni dans la somme des intensités tirées, mais
contribuent aux mesures.
"""

import numpy as np

from ..model import Box, Sources


def central_source(box: Box, intensity: float, position=None) -> Sources:
    """Une source d'intensité `intensity`, au centre de `box` par défaut."""
    position = box.center if position is None else np.asarray(position, dtype=float)
    return Sources(position[None, :], [intensity])


def bright_and_faint(positions, ratio: float, total_intensity: float) -> Sources:
    """
    Deux sources : la première `ratio` fois plus intense que la seconde,
    de somme `total_intensity`.  positions : [[x_forte, y_forte], [x_faible, y_faible]].
    """
    intensities = total_intensity * np.array([ratio, 1.0]) / (ratio + 1.0)
    return Sources(np.asarray(positions, dtype=float), intensities)

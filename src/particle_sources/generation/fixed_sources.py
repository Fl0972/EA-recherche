"""
Sources déterministes, toujours présentes, indépendantes du tirage aléatoire.
Elles ne comptent ni dans N ni dans la somme des intensités tirées, mais
contribuent aux mesures.
"""

import numpy as np

from ..model import Box, Sources


def central_source(box: Box, intensity: float, position=None) -> Sources:
    """
    Une source d'intensité `intensity`, au centre de `box` par défaut.

    Paramètres
    ----------
    box : Box
        Domaine, utilisé pour le centre par défaut.
    intensity : float
        Intensité de la source (dans la config : intensity_factor * total_intensity).
    position : array_like (d,) ou None
        Position imposée ; None = centre du domaine.

    Renvoie
    -------
    Sources
        Une seule source.
    """
    position = box.center if position is None else np.asarray(position, dtype=float)
    return Sources(position[None, :], [intensity])

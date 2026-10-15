"""Tirage d'une configuration aléatoire de sources."""

import numpy as np

from ..model import Box, Sources
from .count_laws import CountLaw
from .intensity_laws import IntensityLaw, uniform_simplex
from .position_laws import PositionLaw, uniform_in_box


def sample_sources(
    box: Box,
    count_law: CountLaw,
    total_intensity: float,
    rng: np.random.Generator,
    intensity_law: IntensityLaw = uniform_simplex,
    position_law: PositionLaw = uniform_in_box,
) -> Sources:
    """
    1. N ~ count_law
    2. positions ~ position_law dans `box`
    3. intensités ~ intensity_law, de somme total_intensity
    """
    n = count_law(rng)
    positions = position_law(n, box, rng)
    intensities = intensity_law(n, total_intensity, rng)
    return Sources(positions.reshape(n, box.dim), intensities)

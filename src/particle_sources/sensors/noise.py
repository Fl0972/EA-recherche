"""
Modèles de bruit de mesure.

Un modèle de bruit est une fonction  (y_exact, rng) -> y_bruité.
"""

from typing import Callable

import numpy as np

NoiseModel = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def no_noise(y, rng):
    return y


def gaussian_relative(sigma: float) -> NoiseModel:
    """y * (1 + sigma * N(0, 1))."""
    return lambda y, rng: y * (1 + sigma * rng.standard_normal(y.shape))


def poisson_counts(exposure: float) -> NoiseModel:
    """
    Comptage de particules pendant un temps `exposure` :
    nombre de coups ~ Poisson(exposure * y), renvoyé en taux (coups / exposure).
    """
    return lambda y, rng: rng.poisson(exposure * y) / exposure

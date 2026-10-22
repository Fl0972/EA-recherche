"""
Modèles de bruit de mesure.

Un modèle de bruit est une fonction  (y_exact, rng) -> y_bruité.
"""

from typing import Callable

import numpy as np

NoiseModel = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def no_noise(y, rng):
    """Mesures exactes : renvoie y tel quel."""
    return y


def gaussian_relative(sigma: float) -> NoiseModel:
    """
    Bruit multiplicatif gaussien : y * (1 + sigma * N(0, 1)).
    sigma est l'erreur relative (0.01 = 1 %), indépendante d'un capteur à l'autre.
    """
    return lambda y, rng: y * (1 + sigma * rng.standard_normal(y.shape))


def poisson_counts(exposure: float) -> NoiseModel:
    """
    Comptage de particules pendant un temps `exposure` :
    nombre de coups ~ Poisson(exposure * y), renvoyé en taux (coups / exposure).
    Erreur relative ~ 1 / sqrt(exposure * y) : plus forte sur les capteurs peu éclairés.
    """
    return lambda y, rng: rng.poisson(exposure * y) / exposure

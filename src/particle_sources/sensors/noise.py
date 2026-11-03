"""
Modèles de bruit de mesure.

Un modèle de bruit est un objet appelable  (y_exact, rng) -> y_bruité, qui
expose aussi :
    std(mu)          écart-type de la mesure quand la valeur exacte est mu
                     (None si pas de bruit)
    fisher(mu, J)    information de Fisher sur les paramètres, J = d mu / d theta
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class NoiseModel(Protocol):
    def __call__(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
    def std(self, mu: np.ndarray) -> np.ndarray | None: ...
    def fisher(self, mu: np.ndarray, J: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class NoNoise:
    def __call__(self, y, rng):
        return y

    def std(self, mu):
        return None

    def fisher(self, mu, J):
        return np.full((J.shape[1], J.shape[1]), np.inf)


@dataclass(frozen=True)
class GaussianRelative:
    """y * (1 + sigma * N(0, 1))."""
    sigma: float

    def __call__(self, y, rng):
        return y * (1 + self.sigma * rng.standard_normal(np.shape(y)))

    def std(self, mu):
        return self.sigma * np.abs(mu)

    def fisher(self, mu, J):
        # Gaussienne de variance (sigma mu)^2 dépendant des paramètres :
        # F = J^T diag(1/(sigma mu)^2) J + 2 (J/mu)^T (J/mu)
        Jr = J / mu[:, None]
        return Jr.T @ Jr * (1 / self.sigma**2 + 2)


@dataclass(frozen=True)
class PoissonCounts:
    """
    Comptage de particules pendant un temps `exposure` :
    nombre de coups ~ Poisson(exposure * y), renvoyé en taux (coups / exposure).
    """
    exposure: float

    def __call__(self, y, rng):
        return rng.poisson(self.exposure * y) / self.exposure

    def std(self, mu):
        return np.sqrt(np.maximum(mu, 1e-300) / self.exposure)

    def fisher(self, mu, J):
        return self.exposure * (J / mu[:, None]).T @ J


# Noms utilisés dans la config (fabriques)
no_noise = NoNoise()
gaussian_relative = GaussianRelative
poisson_counts = PoissonCounts

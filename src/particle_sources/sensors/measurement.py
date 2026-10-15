"""Simulation des mesures des capteurs."""

import numpy as np

from ..model import Sources, intensity_at
from .noise import NoiseModel, no_noise


def measure(
    sources: Sources,
    sensors,
    rng: np.random.Generator | None = None,
    noise: NoiseModel = no_noise,
    eps: float = 0.0,
) -> np.ndarray:
    """Intensité mesurée par chaque capteur, de forme (K,)."""
    y = intensity_at(sources, sensors.positions, eps)
    return noise(y, rng if rng is not None else np.random.default_rng())

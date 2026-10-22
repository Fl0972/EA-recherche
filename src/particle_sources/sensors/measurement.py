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
    """
    Intensité mesurée par chaque capteur, de forme (K,).

    Calcule Phi exact à la position de chaque capteur, puis applique le modèle
    de bruit.

    Paramètres
    ----------
    sources : Sources
        Toutes les sources vues par les capteurs (tirées + fixes).
    sensors : objet exposant `.positions` (K, d)
        Réseau de capteurs, par ex. `CircleSensors`.
    rng : np.random.Generator ou None
        Générateur pour le bruit ; None = nouveau générateur non reproductible.
    noise : NoiseModel
        Modèle de bruit (voir `sensors.noise`). Par défaut aucun.
    eps : float
        Régularisation de 1/r^2 (voir `model.field`).
    """
    y = intensity_at(sources, sensors.positions, eps)
    return noise(y, rng if rng is not None else np.random.default_rng())

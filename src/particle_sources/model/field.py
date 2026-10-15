"""
Modèle direct : intensité totale mesurée en un point.

    Phi(x) = sum_i I_i / (||x - x_i||^2 + eps^2)

eps = 0 donne la loi exacte en 1/r^2 ; eps > 0 régularise la singularité
sur les sources (utile pour l'affichage).
"""

import numpy as np

from .sources import Sources


def intensity_at(sources: Sources, x, eps: float = 0.0) -> np.ndarray:
    """
    x : tableau de forme (..., d). Renvoie Phi(x), de forme (...).
    """
    x = np.asarray(x, dtype=float)
    if sources.n == 0:
        return np.zeros(x.shape[:-1])
    diff = x[..., None, :] - sources.positions       # (..., N, d)
    r2 = np.sum(diff**2, axis=-1) + eps**2           # (..., N)
    with np.errstate(divide="ignore"):
        return np.sum(sources.intensities / r2, axis=-1)

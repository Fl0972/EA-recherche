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


def intensity_jacobian(sources: Sources, x, eps: float = 0.0) -> np.ndarray:
    """
    Dérivées de Phi(x_k) par rapport aux paramètres des sources, rangés en
    theta = [positions.ravel(), intensities]  (N*d + N paramètres).
    x : (K, d). Renvoie un tableau (K, N*d + N).
    """
    x = np.asarray(x, dtype=float)
    diff = x[:, None, :] - sources.positions                 # (K, N, d)
    r2 = np.sum(diff**2, axis=-1) + eps**2                   # (K, N)
    d_pos = 2 * sources.intensities[None, :, None] * diff / r2[..., None] ** 2
    return np.concatenate([d_pos.reshape(len(x), -1), 1 / r2], axis=1)

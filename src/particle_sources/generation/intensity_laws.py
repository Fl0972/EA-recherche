"""
Lois des intensités (I_1, ..., I_N), de somme fixée I_tot.

Une loi d'intensités est une fonction  (n, total, rng) -> tableau (n,)
de somme `total`.
"""

from typing import Callable

import numpy as np

IntensityLaw = Callable[[int, float, np.random.Generator], np.ndarray]


def dirichlet(alpha: float) -> IntensityLaw:
    """I = total * Dirichlet(alpha, ..., alpha).
    alpha < 1 : intensités très inégales ; alpha > 1 : plus homogènes."""
    def sample(n, total, rng):
        if n == 0:
            return np.empty(0)
        return total * rng.dirichlet(np.full(n, alpha))
    return sample


# Tirage uniforme sur le simplexe {I_i >= 0, sum I_i = total}
uniform_simplex: IntensityLaw = dirichlet(1.0)


def cluster_boost(
    cluster_ratio: float = 0.8,
    cluster_share: float = 0.9,
    alpha: float = 1.0,
) -> IntensityLaw:
    """
    Intensités réparties en allouant une part dominante (`cluster_share`)
    aux sources du cluster (les `cluster_ratio * n` premières sources),
    et le reste aux sources d'arrière-plan.
    """
    if not (0.0 <= cluster_ratio <= 1.0):
        raise ValueError(f"cluster_ratio doit être compris entre 0 et 1, reçu {cluster_ratio}")
    if not (0.0 <= cluster_share <= 1.0):
        raise ValueError(f"cluster_share doit être compris entre 0 et 1, reçu {cluster_share}")
    if alpha <= 0:
        raise ValueError(f"alpha doit être strictement positif, reçu {alpha}")

    def sample(n: int, total: float, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.empty(0)
        n_cluster = int(np.round(n * cluster_ratio))
        n_cluster = max(0, min(n, n_cluster))
        n_bg = n - n_cluster

        if n_cluster > 0 and n_bg > 0:
            I_cluster = total * cluster_share
            I_bg = total * (1.0 - cluster_share)
            c_draw = I_cluster * rng.dirichlet(np.full(n_cluster, alpha))
            bg_draw = I_bg * rng.dirichlet(np.full(n_bg, alpha))
            return np.concatenate([c_draw, bg_draw])
        else:
            return total * rng.dirichlet(np.full(n, alpha))

    return sample


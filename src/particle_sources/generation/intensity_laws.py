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

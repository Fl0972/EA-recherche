"""
Géométrie des capteurs.

Tout réseau de capteurs expose `.positions` (K, d). Pour ajouter une
géométrie (grille, plusieurs cercles, ...), créer une classe avec cet attribut.
"""

from dataclasses import dataclass

import numpy as np

from ..model import Box


@dataclass(frozen=True)
class CircleSensors:
    """
    K capteurs équirépartis sur un cercle :
        s_k = center + radius * (cos t_k, sin t_k),  t_k = t0 + 2 pi k / K
    (t = 0 à l'est, sens trigonométrique).
    """
    center: np.ndarray
    radius: float
    n: int
    t0: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))

    @classmethod
    def circumscribed(cls, box: Box, n: int, t0: float = 0.0) -> "CircleSensors":
        """Capteurs sur le cercle dans lequel `box` (2D) est inscrit."""
        if box.dim != 2:
            raise ValueError("CircleSensors n'a de sens qu'en 2D")
        return cls(box.center, box.circumradius, n, t0)

    @property
    def angles(self) -> np.ndarray:
        return self.t0 + 2 * np.pi * np.arange(self.n) / self.n

    @property
    def positions(self) -> np.ndarray:
        t = self.angles
        return self.center + self.radius * np.stack([np.cos(t), np.sin(t)], axis=-1)

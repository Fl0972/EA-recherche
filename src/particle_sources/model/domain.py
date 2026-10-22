"""Domaine rectangulaire (pavé de dimension d) contenant les sources."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """
    Pavé [lo_1, hi_1] x ... x [lo_d, hi_d] dans lequel sont tirées les sources.

    Immuable ; `lo` et `hi` sont convertis en tableaux numpy de flottants à la
    construction. Sert aussi à placer les capteurs (cercle circonscrit) et à
    centrer la source fixe.

    Attributs
    ---------
    lo : np.ndarray, forme (d,)
        Bornes inférieures.
    hi : np.ndarray, forme (d,)
        Bornes supérieures (strictement plus grandes que `lo`).
    """
    lo: np.ndarray  # (d,) bornes inférieures
    hi: np.ndarray  # (d,) bornes supérieures

    def __post_init__(self):
        """Convertit les bornes en float et vérifie leur cohérence."""
        lo = np.asarray(self.lo, dtype=float)
        hi = np.asarray(self.hi, dtype=float)
        if lo.shape != hi.shape or lo.ndim != 1:
            raise ValueError("lo et hi doivent être deux vecteurs de même taille")
        if np.any(hi <= lo):
            raise ValueError("il faut hi > lo dans chaque dimension")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def from_bounds(cls, bounds) -> "Box":
        """
        Construit le pavé à partir de la liste de ses intervalles, au format
        utilisé dans `simulation.yaml`.

        bounds : [[xmin, xmax], [ymin, ymax], ...]
        """
        bounds = np.asarray(bounds, dtype=float)
        return cls(bounds[:, 0], bounds[:, 1])

    @classmethod
    def around_ball(cls, center, radius: float) -> "Box":
        """Plus petit pavé contenant la boule (centre, rayon)."""
        center = np.asarray(center, dtype=float)
        return cls(center - radius, center + radius)

    @property
    def dim(self) -> int:
        """Dimension d de l'espace."""
        return len(self.lo)

    @property
    def size(self) -> np.ndarray:
        """Longueurs des côtés, hi - lo, forme (d,)."""
        return self.hi - self.lo

    @property
    def center(self) -> np.ndarray:
        """Centre du pavé, forme (d,)."""
        return 0.5 * (self.lo + self.hi)

    @property
    def circumradius(self) -> float:
        """Rayon de la sphère (cercle en 2D) dans laquelle le pavé est inscrit."""
        return 0.5 * float(np.linalg.norm(self.size))

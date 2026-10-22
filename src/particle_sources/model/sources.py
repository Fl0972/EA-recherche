"""Configuration de sources ponctuelles : positions et intensités."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sources:
    """
    Ensemble de N sources ponctuelles.

    C'est l'objet qui circule dans tout le code : produit par la génération,
    lu par le modèle direct et renvoyé par les méthodes d'inversion. Immuable.

    Attributs
    ---------
    positions : np.ndarray, forme (N, d)
        Position de chaque source.
    intensities : np.ndarray, forme (N,)
        Intensité I_i de chaque source.
    """
    positions: np.ndarray    # (N, d)
    intensities: np.ndarray  # (N,)

    def __post_init__(self):
        """Convertit en tableaux float et vérifie que les formes concordent."""
        positions = np.asarray(self.positions, dtype=float)
        intensities = np.asarray(self.intensities, dtype=float)
        if positions.ndim != 2 or intensities.shape != (len(positions),):
            raise ValueError("positions doit être (N, d) et intensities (N,)")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "intensities", intensities)

    @classmethod
    def empty(cls, dim: int = 2) -> "Sources":
        """Ensemble vide (N = 0) en dimension `dim`."""
        return cls(np.empty((0, dim)), np.empty(0))

    def __add__(self, other: "Sources") -> "Sources":
        """Réunion de deux ensembles de sources."""
        return Sources(np.concatenate([self.positions, other.positions]),
                       np.concatenate([self.intensities, other.intensities]))

    @property
    def n(self) -> int:
        """Nombre de sources N."""
        return len(self.intensities)

    @property
    def dim(self) -> int:
        """Dimension d de l'espace."""
        return self.positions.shape[1]

    @property
    def total_intensity(self) -> float:
        """Somme des intensités."""
        return float(self.intensities.sum())

    def __str__(self) -> str:
        """Résumé lisible : nombre de sources puis une ligne par source."""
        lines = [f"{self.n} sources (intensité totale = {self.total_intensity:.4g})"]
        for x, I in zip(self.positions, self.intensities):
            lines.append(f"  x = {np.round(x, 3)}   I = {I:.4g}")
        return "\n".join(lines)

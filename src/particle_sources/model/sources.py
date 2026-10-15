"""Configuration de sources ponctuelles : positions et intensités."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Sources:
    positions: np.ndarray    # (N, d)
    intensities: np.ndarray  # (N,)

    def __post_init__(self):
        positions = np.asarray(self.positions, dtype=float)
        intensities = np.asarray(self.intensities, dtype=float)
        if positions.ndim != 2 or intensities.shape != (len(positions),):
            raise ValueError("positions doit être (N, d) et intensities (N,)")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "intensities", intensities)

    @classmethod
    def empty(cls, dim: int = 2) -> "Sources":
        return cls(np.empty((0, dim)), np.empty(0))

    @property
    def n(self) -> int:
        return len(self.intensities)

    @property
    def dim(self) -> int:
        return self.positions.shape[1]

    @property
    def total_intensity(self) -> float:
        return float(self.intensities.sum())

    def __add__(self, other: "Sources") -> "Sources":
        if self.n == 0:
            return other
        if other.n == 0:
            return self
        if self.dim != other.dim:
            raise ValueError(f"dimensions incompatibles : {self.dim} vs {other.dim}")
        return Sources(
            np.vstack([self.positions, other.positions]),
            np.concatenate([self.intensities, other.intensities]),
        )

    def __str__(self) -> str:
        lines = [f"{self.n} sources (intensité totale = {self.total_intensity:.4g})"]
        for x, I in zip(self.positions, self.intensities):
            lines.append(f"  x = {np.round(x, 3)}   I = {I:.4g}")
        return "\n".join(lines)

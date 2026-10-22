"""
Lois des positions des sources.

Une loi de positions est une fonction  (n, box, rng) -> tableau (n, d).
"""

from typing import Callable

import numpy as np

from ..model import Box

PositionLaw = Callable[[int, Box, np.random.Generator], np.ndarray]


def uniform_in_box(n: int, box: Box, rng: np.random.Generator) -> np.ndarray:
    """n positions indépendantes, uniformes dans `box`. Forme (n, d)."""
    return rng.uniform(box.lo, box.hi, size=(n, box.dim))

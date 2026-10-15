"""Prise en compte de sources connues avant l'inversion."""

import numpy as np

from ..model import Sources, intensity_at


def subtract_known_sources(y: np.ndarray, known: Sources, sensors,
                           eps: float = 0.0) -> np.ndarray:
    """
    Retire des mesures la contribution exacte des sources connues : l'inversion
    ne voit alors plus que les sources inconnues (plus le bruit, qui lui reste).
    """
    return y - intensity_at(known, sensors.positions, eps)

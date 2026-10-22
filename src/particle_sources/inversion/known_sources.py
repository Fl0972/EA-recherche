"""Prise en compte de sources connues avant l'inversion."""

import numpy as np

from ..model import Sources, intensity_at


def subtract_known_sources(y: np.ndarray, known: Sources, sensors,
                           eps: float = 0.0) -> np.ndarray:
    """
    Retire des mesures la contribution exacte des sources connues : l'inversion
    ne voit alors plus que les sources inconnues (plus le bruit, qui lui reste).

    Paramètres
    ----------
    y : np.ndarray, forme (K,)
        Mesures (éventuellement bruitées).
    known : Sources
        Sources dont la position et l'intensité sont connues (source centrale).
    sensors : objet exposant `.positions` (K, d)
        Capteurs ayant produit `y`.
    eps : float
        Même régularisation que celle utilisée pour simuler les mesures.

    Renvoie
    -------
    np.ndarray, forme (K,)
        y - Phi_known(capteurs).

    Remarque : avec un bruit relatif, le bruit dû à la source centrale (très
    intense) reste dans le résidu et peut dominer le signal des petites sources.
    """
    return y - intensity_at(known, sensors.positions, eps)

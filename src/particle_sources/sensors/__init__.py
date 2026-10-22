"""
Capteurs : géométrie, bruit et simulation des mesures.

Chaîne de mesure : positions des capteurs (layouts) -> Phi exact en chaque
capteur (model.intensity_at) -> bruit (noise) ; le tout via `measure`.
"""

from . import noise
from .layouts import CircleSensors
from .measurement import measure

__all__ = ["CircleSensors", "measure", "noise"]

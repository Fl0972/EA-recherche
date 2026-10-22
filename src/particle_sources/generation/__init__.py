"""
Génération de configurations de sources synthétiques.

Une configuration aléatoire est obtenue en combinant trois lois
interchangeables (nombre, intensités, positions) via `sample_sources`.
Les sources fixes (`central_source`) sont déterministes et ajoutées à part.
"""

from . import count_laws, intensity_laws, position_laws
from .fixed_sources import central_source
from .sampler import sample_sources

__all__ = ["central_source", "count_laws", "intensity_laws", "position_laws",
           "sample_sources"]

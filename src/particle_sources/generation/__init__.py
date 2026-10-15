"""Génération de configurations de sources synthétiques."""

from . import count_laws, intensity_laws, position_laws
from .fixed_sources import central_source
from .sampler import sample_sources

__all__ = ["central_source", "count_laws", "intensity_laws", "position_laws",
           "sample_sources"]

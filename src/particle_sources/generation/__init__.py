"""Génération de configurations de sources synthétiques."""

from . import count_laws, intensity_laws, position_laws
from .sampler import sample_sources

__all__ = ["count_laws", "intensity_laws", "position_laws", "sample_sources"]

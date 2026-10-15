"""Objets physiques du problème : domaine, sources et modèle direct."""

from .domain import Box
from .field import intensity_at
from .sources import Sources

__all__ = ["Box", "Sources", "intensity_at"]

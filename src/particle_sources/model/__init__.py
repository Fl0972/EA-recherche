"""
Objets physiques du problème : domaine, sources et modèle direct.

Ce sous-paquet ne dépend d'aucun autre : il ne contient ni aléatoire ni
affichage, seulement la description du système et le calcul de Phi.
"""

from .domain import Box
from .field import intensity_at
from .sources import Sources

__all__ = ["Box", "Sources", "intensity_at"]

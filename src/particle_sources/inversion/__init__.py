"""
Problème inverse : retrouver les sources à partir des mesures.

Une méthode d'inversion est une fonction  (y, sensors, **params) -> Sources.
`subtract_known_sources` est un prétraitement facultatif qui retire la
contribution des sources connues (source centrale) avant l'inversion.
"""

from .known_sources import subtract_known_sources
from .prony import prony_circle

__all__ = ["prony_circle", "subtract_known_sources"]

"""
Problème inverse : retrouver les sources à partir des mesures.

Une méthode d'inversion est une fonction  (y, problem: InverseProblem) -> Sources.
"""

from .known_sources import subtract_known_sources
from .likelihood import prony_mle, refine
from .problem import InverseProblem
from .prony import prony, prony_circle

__all__ = ["InverseProblem", "prony", "prony_circle", "prony_mle", "refine",
           "subtract_known_sources"]

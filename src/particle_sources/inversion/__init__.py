"""
Problème inverse : retrouver les sources à partir des mesures.

Une méthode d'inversion est une fonction  (y, problem: InverseProblem) -> Sources.
"""

from .gradient_descent import (DescentResult, curvature_spectrum, gradient_descent,
                               gradient_descent_method, random_init)
from .known_sources import subtract_known_sources
from .likelihood import prony_mle, refine
from .problem import InverseProblem
from .prony import prony, prony_circle

__all__ = ["DescentResult", "InverseProblem", "curvature_spectrum", "gradient_descent",
           "gradient_descent_method",
           "prony", "prony_circle", "prony_mle", "random_init", "refine",
           "subtract_known_sources"]

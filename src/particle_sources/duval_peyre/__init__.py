"""
Vérification numérique du théorème de Duval-Peyré (stabilité du BLASSO
positif au bruit) pour le modèle direct du projet, et validation sur données
bruitées.

    kernel       noyau G = 1 / (r^2 + eps^2) et ses dérivées analytiques
    certificate  Gamma, pré-certificat eta_V, hypothèses H1, H2, H3
    blasso       LASSO positif sur grille, Frank-Wolfe glissant, débiaisage

Point d'entrée : scripts/duval_peyre.py.
"""

from .certificate import Diagnostic, diagnose, eta, gamma_matrix, precertificate
from .kernel import GaussianKernel, InverseSquareKernel, check_derivatives

__all__ = ["Diagnostic", "GaussianKernel", "InverseSquareKernel", "check_derivatives",
           "diagnose", "eta", "gamma_matrix", "precertificate"]

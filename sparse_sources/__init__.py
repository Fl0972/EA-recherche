"""
sparse_sources : localisation de sources ponctuelles positives par
parcimonie (LASSO positif sur grille, BLASSO par Frank-Wolfe glissant) et
vérification de la condition de source non dégénérée de Duval-Peyré.

    kernel      Setup (capteurs, hauteur h, domaine D), φ, ∇φ, ∇²φ
    common      forward, certificat, débiaisage, diagnostics
    lasso       LASSO positif sur grille (FISTA / sklearn), regroupement, raffinement
    blasso      Frank-Wolfe glissant
    hypotheses  check_hypotheses (H1, H2, H3)

Le noyau est celui du projet : G(x, z) = 1 / (||x - z||² + h²).
"""

from .kernel import Setup, get_default_setup, make_setup, set_default_setup
from .common import (Grid, certificate, debias, diagnostics, forward, make_grid,
                     match_sources, max_certificate)

__all__ = ["Grid", "Setup", "certificate", "debias", "diagnostics", "forward",
           "get_default_setup", "make_grid", "make_setup", "match_sources",
           "max_certificate", "set_default_setup"]

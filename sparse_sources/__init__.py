"""
sparse_sources : localisation de sources ponctuelles positives par
parcimonie (LASSO positif sur grille, BLASSO par Frank-Wolfe glissant) et
vérification de la condition de source non dégénérée de Duval-Peyré.

    kernel      Setup (capteurs, hauteur h, domaine D), φ, ∇φ, ∇²φ
    common      forward, certificat, débiaisage, diagnostics
    lasso       LASSO positif sur grille (FISTA / sklearn), regroupement, raffinement
    blasso      Frank-Wolfe glissant
    hypotheses  check_hypotheses (H1, H2, H3)
    ml          maximum de vraisemblance glouton (choix de K par BIC), Cramér-Rao
    plotting    style des figures

Le noyau est celui du projet : G(x, z) = 1 / (||x - z||² + h²).
"""

from .kernel import Setup, get_default_setup, make_setup, set_default_setup
from .common import (Grid, certificate, debias, diagnostics, forward, make_grid,
                     match_sources, max_certificate)
from .lasso import build_grid, lasso_pipeline, merge_peaks, refine, solve_lasso
from .blasso import sliding_frank_wolfe
from .hypotheses import check_hypotheses, hypotheses_after_fit
from .ml import cramer_rao, greedy_ml, relative_noise_std

__all__ = ["Grid", "Setup", "build_grid", "certificate", "check_hypotheses", "cramer_rao", "debias",
           "diagnostics", "forward", "get_default_setup", "greedy_ml", "hypotheses_after_fit",
           "lasso_pipeline", "make_grid", "make_setup", "match_sources", "max_certificate",
           "merge_peaks", "refine", "relative_noise_std", "set_default_setup", "sliding_frank_wolfe", "solve_lasso"]

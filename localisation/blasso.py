"""
BLASSO positif sans grille par Frank–Wolfe glissant
(Denoyelle, Duval, Peyré & Soubies, Inverse Problems, 2019).

    min_{s ≥ 0}  ½ ||m − A(s)||² + λ ||s||_TV,    s = Σ_k I_k δ_{z_k}.

À chaque itération : un Dirac est ajouté au maximum du certificat η = Φ*(m − A(s)) / λ,
puis intensités ET positions sont réoptimisées (« glissement »). Arrêt quand
max_D η ≤ 1 + tol : c'est la condition d'optimalité du BLASSO positif.

Réglages du projet (``estimer``) : capteurs blanchis (w_i = 1 / σ_i), noyau normalisé,
λ = 1,5 √M (M = nombre de capteurs), au plus 40 itérations, puis débiaisage.
"""

from __future__ import annotations

import time

import numpy as np

from . import probleme as P
from .ml_bic import setup_et_grille
from .noyau import Grid, Setup, make_grid
from .optimisation import (debias, joint_descent, max_certificate, objective, positive_lasso,
                           prune_and_merge)

LAMBDA_FACTEUR = 1.5     # λ = LAMBDA_FACTEUR √M
MAX_ITER = 40


def sliding_frank_wolfe(m, lam: float, setup: Setup, z_init=None, I_init=None,
                        tol: float = 1e-6, max_iter: int = 50,
                        grid: Grid | float | None = None, merge_dist: float | None = None,
                        n_starts: int = 5, verbose: bool = False) -> dict:
    """
    Frank–Wolfe glissant pour le BLASSO positif.

    À chaque itération :
      a) résidu r = m − A(μ) ;  b) z* = argmax_D η ;
      c) si η(z*) ≤ 1 + tol : arrêt (solution certifiée) ;
      d) ajout d'un Dirac en z* ;  e) intensités à positions fixées (LASSO positif exact) ;
      f) glissement conjoint positions + intensités ;  g) suppression / fusion des Diracs.
    Puis débiaisage (moindres carrés positifs à positions fixées).
    Renvoie un dict : z, I, z_debiased, I_debiased, history, converged, eta_max, n_iter, time.
    """
    t0 = time.perf_counter()
    m = np.asarray(m, float)
    if not isinstance(grid, Grid):
        grid = make_grid(setup.domain, grid, setup)
    if merge_dist is None:
        merge_dist = 1e-3 * float(np.linalg.norm(setup.domain.size))
    if z_init is not None and len(np.atleast_1d(I_init)):
        z = np.asarray(z_init, float).reshape(-1, 2).copy()
        I = np.asarray(I_init, float).reshape(-1).copy()
    else:
        z, I = np.zeros((0, 2)), np.zeros(0)

    history = []
    converged = False
    eta_max = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        r = m - (I @ setup.phi(z) if len(I) else 0.0)
        z_star, eta_max = max_certificate(r, lam, grid, setup, n_starts)
        history.append({"iter": it, "K": len(I), "eta_max": eta_max,
                        "objective": objective(z, I, m, lam, setup)})
        if verbose:
            print(f"  it {it:2d} : K = {len(I):2d}, max η = {eta_max:.6f}, "
                  f"F = {history[-1]['objective']:.6e}")
        if eta_max <= 1 + tol:
            converged = True
            break
        z = np.vstack([z, z_star])
        I = positive_lasso(setup.phi(z).T, m, lam)
        z, I, _ = joint_descent(z, I, m, lam, setup)
        z, I = prune_and_merge(z, I, merge_dist)

    if not converged and it == max_iter:
        r = m - (I @ setup.phi(z) if len(I) else 0.0)
        _, eta_max = max_certificate(r, lam, grid, setup, n_starts)
        converged = eta_max <= 1 + tol

    z_deb, I_deb = debias(z, I, m, setup)
    return {"z": z, "I": I, "z_debiased": z_deb, "I_debiased": I_deb,
            "history": history, "converged": converged, "eta_max": eta_max,
            "n_iter": it, "time": time.perf_counter() - t0}


def estimer(sit: P.Situation) -> dict:
    """BLASSO (Frank–Wolfe glissant) blanchi et normalisé, sur tous les capteurs."""
    S0, g = setup_et_grille(sit.capteurs)
    S = S0.with_weights(1.0 / sit.s)
    gw = Grid(g.xs, g.ys, g.points, g.step, S.phi(g.flat).T)
    lam = LAMBDA_FACTEUR * np.sqrt(len(sit.m))
    fw = sliding_frank_wolfe(S.whiten(sit.m), lam, setup=S, grid=gw, tol=1e-5,
                             max_iter=MAX_ITER)
    z = np.atleast_2d(fw["z_debiased"]).reshape(-1, 2)
    I = S.physical_intensities(z, fw["I_debiased"]) if len(z) else np.zeros(0)
    return {"z": z, "I": np.asarray(I, float), "iterations": fw["n_iter"]}

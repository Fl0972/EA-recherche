"""
BLASSO positif sans grille par Frank-Wolfe glissant
(Denoyelle, Duval, Peyré & Soubies, « The sliding Frank-Wolfe algorithm and
its application to super-resolution microscopy », Inverse Problems, 2019).

Problème :
    min_{s ≥ 0}  ½ ||m - A(s)||² + λ ||s||_TV,    s = Σ_k I_k δ_{z_k}.

Frank-Wolfe classique sur l'espace des mesures : la direction de descente est
un Dirac placé au maximum du certificat η = Φ^*(m - A(s)) / λ. La version
« glissante » ajoute, après chaque ajout, une optimisation locale conjointe
des intensités ET des positions ; en pratique l'algorithme termine en un
nombre fini d'itérations (≈ K) sous la condition de source non dégénérée.

Critère d'arrêt : max_D η ≤ 1 + tol, qui est exactement la condition
d'optimalité du BLASSO positif (la solution est alors certifiée).
"""

from __future__ import annotations

import time

import numpy as np

from .common import (Grid, debias, diagnostics, joint_descent, make_grid, max_certificate,
                     objective, prune_and_merge)
from .kernel import Setup, resolve
from .lasso import active_set_positive_lasso


def sliding_frank_wolfe(m, lam: float, z_init=None, I_init=None, tol: float = 1e-6,
                        max_iter: int = 50, setup: Setup | None = None,
                        grid: Grid | float | None = None, merge_dist: float | None = None,
                        n_starts: int = 5, sigma=None, verbose: bool = False) -> dict:
    """
    Frank-Wolfe glissant pour le BLASSO positif.

    Départ : μ = 0, ou μ = Σ I_init,k δ_{z_init,k} (par ex. la sortie du LASSO).
    À chaque itération :
      a) résidu r = m - A(μ) ;
      b) z* = argmax_D η, η(z) = φ(z)·r / λ (``max_certificate`` : grille
         fine puis L-BFGS-B borné) ;
      c) si η(z*) ≤ 1 + tol : arrêt, μ est solution (certificat) ;
      d) ajout d'un Dirac en z* d'intensité 0 ;
      e) réajustement des intensités à positions fixées : LASSO positif exact
         sur les K colonnes φ(z_k) (ensemble actif) ;
      f) glissement : min_{I ≥ 0, z ∈ D} ½ ||m - Σ I_k φ(z_k)||² + λ Σ I_k
         (L-BFGS-B, gradient analytique, ``common.joint_descent``) ;
      g) suppression des Diracs d'intensité nulle, fusion de ceux à distance
         < merge_dist (défaut 1e-3 × diagonale de D).
    Puis débiaisage (moindres carrés positifs à positions fixées) et
    diagnostics (si sigma est donné).

    Paramètres
    ----------
    m : (M,) mesures ; lam : λ > 0
    z_init, I_init : initialisation facultative ((K, 2), (K,))
    tol : tolérance sur max η - 1
    max_iter : nombre max d'itérations (ajouts de Diracs)
    grid : Grid, pas, ou None : grille fine de ``max_certificate``
    n_starts : nombre de maxima locaux de la grille affinés à l'étape b)
    sigma : niveau de bruit pour ``diagnostics`` (facultatif)

    Renvoie un dict :
        z, I                 solution du BLASSO (pénalisée)
        z_debiased, I_debiased
        history              liste de {iter, K, eta_max, objective} (une entrée
                             par itération, état AVANT l'ajout de l'itération)
        converged            max η ≤ 1 + tol atteint
        eta_max, n_iter, time, diagnostics (si sigma)
    """
    setup = resolve(setup)
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
        # a) b) résidu et maximum du certificat
        r = m - (I @ setup.phi(z) if len(I) else 0.0)
        z_star, eta_max = max_certificate(r, lam, grid, setup, n_starts)
        history.append({"iter": it, "K": len(I), "eta_max": eta_max,
                        "objective": objective(z, I, m, lam, setup)})
        if verbose:
            print(f"  it {it:2d} : K = {len(I):2d}, max η = {eta_max:.6f}, "
                  f"F = {history[-1]['objective']:.6e}")
        # c) arrêt : certificat ≤ 1
        if eta_max <= 1 + tol:
            converged = True
            break
        # d) nouveau Dirac
        z = np.vstack([z, z_star])
        # e) intensités à positions fixées (LASSO positif sur K variables)
        I, _ = active_set_positive_lasso(setup.phi(z).T, m, lam)
        # f) glissement
        z, I, _ = joint_descent(z, I, m, lam, setup)
        # g) nettoyage
        z, I = prune_and_merge(z, I, merge_dist)

    if not converged and it == max_iter:
        r = m - (I @ setup.phi(z) if len(I) else 0.0)
        _, eta_max = max_certificate(r, lam, grid, setup, n_starts)
        converged = eta_max <= 1 + tol

    z_deb, I_deb = debias(z, I, m, setup)
    out = {"z": z, "I": I, "z_debiased": z_deb, "I_debiased": I_deb,
           "history": history, "converged": converged, "eta_max": eta_max,
           "n_iter": it, "time": time.perf_counter() - t0}
    if sigma is not None:
        out["diagnostics"] = diagnostics(z_deb, I_deb, m, sigma, setup)
    if verbose:
        print(f"Frank-Wolfe glissant : {len(I)} sources, max η = {eta_max:.6f} "
              f"({'certifié' if converged else 'NON certifié'}), {it} itérations, "
              f"{out['time']:.2f} s")
    return out

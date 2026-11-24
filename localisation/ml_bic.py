"""
Maximum de vraisemblance glouton avec choix du nombre de sources par le BIC.

Bruit gaussien de variance s_i² connue (bruit relatif du projet : s_i = σ y_i^brut) :
    log-vraisemblance ∝ −½ Σ_i (m_i − A(s)_i)² / s_i² = −½ ||m_w − A_w(s)||²
avec m_w = m / s et A_w l'opérateur blanchi.
Pour K = 1, 2, ... :
    a) nouveau Dirac au maximum de la corrélation ⟨ψ_w(z), r_w⟩ (noyau blanchi normalisé) ;
    b) intensités par moindres carrés positifs, puis glissement SANS pénalité de toutes
       les positions et intensités : maximum de vraisemblance à K fixé ;
    c) critère C(K) = ||r_w||² + κ · 3K  (κ = log M : BIC).
Arrêt quand C n'a pas diminué depuis ``patience`` ajouts ; on garde le meilleur K, puis
on essaie de retirer chaque source (étape arrière) tant que C diminue.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from scipy.optimize import nnls

from . import probleme as P
from .noyau import Grid, Setup, grille_recherche, make_grid
from .optimisation import grid_local_maxima, joint_descent, max_certificate, prune_and_merge

K_MAX = 20               # nombre maximal de sources


def _penalty(criterion, M: int) -> float:
    """κ du critère : 'bic' → log M, 'aic' → 2, nombre → lui-même."""
    if criterion == "bic":
        return float(np.log(M))
    if criterion == "aic":
        return 2.0
    return float(criterion)


def greedy_ml(m, noise_std, setup: Setup, grid: Grid | float | None = None,
              k_max: int = 10, criterion="bic", patience: int = 2, backward: bool = True,
              z_init=None, I_init=None, merge_dist: float | None = None,
              n_starts: int = 5, n_candidates: int = 1, verbose: bool = False) -> dict:
    """
    Maximum de vraisemblance glouton avec choix de K par critère d'information.

    m : (M,) mesures (source centrale retirée) ; noise_std : (M,) écart type par capteur ;
    setup : géométrie (noyau normalisé conseillé pour choisir les Diracs) ;
    grid : grille de recherche du maximum de corrélation ; k_max : nombre maximal de sources.
    Renvoie un dict : z, I (physiques), K, criterion, chi2, history, time.
    """
    t0 = time.perf_counter()
    base = setup
    m = np.asarray(m, float)
    s = np.broadcast_to(np.asarray(noise_std, float), m.shape)
    S = base.with_weights(1.0 / s)
    mw = S.whiten(m)
    M = len(m)
    kappa = _penalty(criterion, M)
    if isinstance(grid, Grid):       # mêmes nœuds, colonnes recalculées pour le noyau blanchi
        grid = Grid(grid.xs, grid.ys, grid.points, grid.step, S.phi(grid.flat).T)
    else:
        grid = make_grid(S.domain, grid, S)
    if merge_dist is None:
        merge_dist = 1e-3 * float(np.linalg.norm(S.domain.size))

    def chi2(z, I):
        r = mw - (I @ S.phi(z) if len(I) else 0.0)
        return float(r @ r)

    def crit(z, I):
        return chi2(z, I) + kappa * 3 * len(I)

    def slide(z, I):
        if len(I) == 0:
            return z, I
        I = np.maximum(I, 1e-3 * max(float(I.max()), 1e-12))
        z, I, _ = joint_descent(z, I, mw, 0.0, S)
        return prune_and_merge(z, I, merge_dist)

    z, I = np.zeros((0, 2)), np.zeros(0)
    if z_init is not None and len(np.atleast_1d(I_init)):
        z0 = np.asarray(z_init, float).reshape(-1, 2)
        I0 = np.asarray(I_init, float).reshape(-1)
        ok = np.all((z0 >= S.domain.lo) & (z0 <= S.domain.hi), axis=1) & (I0 > 0)
        if ok.any():
            z, I = slide(z0[ok], S.normalized_intensities(z0[ok], I0[ok]))

    best = (crit(z, I), z, I)
    history = [{"K": len(I), "chi2": chi2(z, I), "criterion": best[0]}]
    fails = 0
    while len(I) < k_max:
        r = mw - (I @ S.phi(z) if len(I) else 0.0)
        starts = [max_certificate(r, 1.0, grid, S, n_starts)[0]]
        if n_candidates > 1:
            img = (r @ grid.G).reshape(grid.shape)
            peaks = grid_local_maxima(img, n_candidates + 1)
            starts += [grid.points[tuple(q)] for q in peaks[1:n_candidates]]
        trials = []
        for z_new in starts:
            zt = np.vstack([z, z_new])
            It, _ = nnls(S.phi(zt).T, mw, maxiter=50 * len(zt) + 100)
            zt, It = slide(zt, It)
            trials.append((chi2(zt, It), zt, It))
        _, z, I = min(trials, key=lambda t: t[0])
        c = crit(z, I)
        history.append({"K": len(I), "chi2": chi2(z, I), "criterion": c})
        if verbose:
            print(f"  K = {len(I):2d} : χ² = {history[-1]['chi2']:.1f}, critère = {c:.1f}")
        if c < best[0] - 1e-9:
            best, fails = (c, z, I), 0
        else:
            fails += 1
            if fails >= patience:
                break

    c, z, I = best
    improved = backward
    while improved and len(I) > 1:
        improved = False
        for k in range(len(I)):
            zk, Ik = slide(np.delete(z, k, 0), np.delete(I, k))
            ck = crit(zk, Ik)
            if ck < c - 1e-9:
                c, z, I, improved = ck, zk, Ik, True
                if verbose:
                    print(f"  retrait d'une source : K = {len(I)}, critère = {c:.1f}")
                break

    return {"z": z, "I": S.physical_intensities(z, I), "K": len(I), "criterion": c,
            "chi2": chi2(z, I), "history": history, "time": time.perf_counter() - t0}


@lru_cache(maxsize=None)
def setup_et_grille(capteurs: str):
    """Setup (noyau normalisé) et grille de recherche (pas 0,25) d'une disposition de capteurs."""
    S = Setup(P.capteurs(capteurs), P.DOMAINE, 0.0, True, None)
    return S, grille_recherche(S)


def estimer(sit: P.Situation) -> dict:
    """ML + BIC sur tous les capteurs de la situation, au plus K_MAX sources."""
    S, g = setup_et_grille(sit.capteurs)
    r = greedy_ml(sit.m, sit.s, setup=S, grid=g, k_max=K_MAX)
    return {"z": np.asarray(r["z"]).reshape(-1, 2), "I": np.asarray(r["I"]),
            "chi2": r["chi2"]}

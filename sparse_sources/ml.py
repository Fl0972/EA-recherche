"""
Maximum de vraisemblance glouton (« OMP continu glissant ») avec choix du
nombre de sources par critère d'information, et borne de Cramér-Rao.

Motivation (scénario du projet) : la grosse source centrale (intensité 5000,
au centre du cercle de capteurs) est connue et retirée des mesures, mais son
bruit relatif reste : l'écart type du bruit du capteur i vaut
s_i = σ y_i, avec y_i ≈ 40 + (signal utile)_i. Le BLASSO a deux défauts dans
ce régime : (1) la pénalité λ Σ I_k biaise positions et intensités (biais
O(λ), et λ doit être ~ ||w|| pour éliminer les fausses sources), (2) le
nombre de sources est réglé indirectement par λ.

Méthode (bruit gaussien de variance s_i² connue) :
    log-vraisemblance  ∝ -½ Σ_i (m_i - A(s)_i)² / s_i² = -½ ||m_w - A_w(s)||²
avec m_w = m / s et A_w l'opérateur blanchi (``Setup.with_weights(1/s)``).
Pour K = 1, 2, ... :
    a) nouveau Dirac au maximum de la corrélation ⟨ψ_w(z), r_w⟩ (noyau
       blanchi normalisé, comme une étape de Frank-Wolfe) ;
    b) intensités par moindres carrés positifs, puis glissement SANS pénalité
       (λ = 0) de toutes les positions et intensités : c'est l'estimateur du
       maximum de vraisemblance à K fixé (minimum local) ;
    c) critère  C(K) = ||r_w||² + κ · 3K  (κ = log M : BIC ; κ = 2 : AIC).
On s'arrête quand C n'a pas diminué depuis ``patience`` ajouts ; on garde le
meilleur K, puis on essaie de retirer chaque source (étape arrière) tant que
C diminue.

La borne de Cramér-Rao (``cramer_rao``) donne l'écart type minimal d'un
estimateur sans biais des positions : c'est la limite physique du problème,
quel que soit l'algorithme.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import nnls

from .common import (Grid, grid_local_maxima, joint_descent, make_grid, max_certificate,
                     prune_and_merge)
from .kernel import Setup, resolve


def relative_noise_std(y, sigma: float, floor: float = 1e-12) -> np.ndarray:
    """
    Écart type du bruit relatif du projet, y (1 + σ ε) : s_i = σ |y_i|.
    ``y`` = mesures BRUTES (source centrale comprise), pas les mesures dont
    on a retiré la source centrale : c'est elle qui fixe le niveau de bruit.
    """
    y = np.abs(np.asarray(y, float))
    return np.maximum(sigma * y, floor * max(float(y.max()), 1.0))


def _penalty(criterion, M: int) -> float:
    """κ du critère : 'bic' → log M, 'aic' → 2, nombre → lui-même."""
    if criterion == "bic":
        return float(np.log(M))
    if criterion == "aic":
        return 2.0
    return float(criterion)


def greedy_ml(m, noise_std, setup: Setup | None = None, grid: Grid | float | None = None,
              k_max: int = 10, criterion="bic", patience: int = 2, backward: bool = True,
              z_init=None, I_init=None, merge_dist: float | None = None,
              n_starts: int = 5, n_candidates: int = 1, verbose: bool = False) -> dict:
    """
    Maximum de vraisemblance glouton avec choix de K par critère d'information.

    Paramètres
    ----------
    m : (M,) mesures (source centrale retirée)
    noise_std : (M,) ou scalaire, écart type du bruit par capteur
        (``relative_noise_std(y_brut, σ)`` pour le bruit du projet)
    setup : géométrie ; la pondération ||φ||^α du setup sert seulement à
        choisir les nouveaux Diracs (α = 1 conseillé avec le noyau 1/r²) ;
        la vraisemblance, elle, ne dépend pas de α
    grid : grille de recherche du maximum de corrélation
    k_max : nombre maximal de sources
    criterion : 'bic' (défaut), 'aic' ou κ numérique
    patience : nombre d'ajouts sans amélioration du critère avant arrêt
    backward : étape arrière (retrait de sources)
    z_init, I_init : départ facultatif (positions, intensités PHYSIQUES),
        par ex. la sortie de Prony ; les sources hors de D sont ignorées
    n_candidates : à chaque ajout, nombre de positions candidates essayées
        (le maximum de corrélation, puis les maxima locaux suivants de la
        grille) ; on garde celle qui donne le plus petit χ² après glissement

    Renvoie un dict :
        z, I          positions et intensités physiques
        K, criterion  nombre de sources et valeur du critère
        chi2          ||r_w||² (≈ M si le modèle explique les données)
        history       [{K, chi2, criterion}] pour chaque ajout
        time
    """
    t0 = time.perf_counter()
    base = resolve(setup)
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


def cramer_rao(z, I, noise_std, setup: Setup | None = None) -> dict:
    """
    Borne de Cramér-Rao pour (I_k, z_k), bruit gaussien d'écart type
    ``noise_std`` (par capteur). Noyau physique (pondération et poids du
    setup ignorés). Matrice de Fisher :

        F = Dᵀ diag(1/s²) D,   D = [φ(z_1) … φ(z_K) | I_1 ∇φ(z_1) … I_K ∇φ(z_K)]

    Renvoie un dict :
        pos_std  (K,)  sqrt(trace de la covariance de z_k) : erreur de
                       position minimale (rms) d'un estimateur sans biais
        int_std  (K,)  écart type minimal de I_k
        cov            inverse (pseudo-inverse) de F, ordre (I_1..I_K, z_1x, z_1y, ...)
    """
    base = resolve(setup).with_power(0.0).with_weights(None)
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    K = len(I)
    P = base.phi(z)                                       # K, M
    J = base.grad(z)                                      # K, M, 2
    M = P.shape[1]
    s = np.broadcast_to(np.asarray(noise_std, float), (M,))
    D = np.concatenate([P.T, (I[:, None, None] * J).transpose(1, 0, 2).reshape(M, 2 * K)], 1)
    Dw = D / s[:, None]
    cov = np.linalg.pinv(Dw.T @ Dw)
    d = np.diag(cov)
    return {"pos_std": np.sqrt(d[K:].reshape(K, 2).sum(1)), "int_std": np.sqrt(d[:K]),
            "cov": cov}

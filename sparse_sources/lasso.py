"""
LASSO positif sur grille.

On restreint les positions aux nœuds y_1, ..., y_P d'une grille de D :
s = Σ_j c_j δ_{y_j}, et le BLASSO devient le LASSO positif

    min_{c ≥ 0}  ½ ||m - G c||² + λ Σ_j c_j,      G_ij = G(x_i, y_j)   (M × P).

Une source hors grille est approchée par plusieurs nœuds voisins (en général
deux ou quatre) : ``merge_peaks`` les regroupe, puis ``refine`` libère les
positions (optimisation locale sans grille, K fixé).

Pipeline complet : ``lasso_pipeline``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .common import (Grid, debias, diagnostics, joint_descent, make_grid,
                     max_certificate, prune_and_merge)
from .kernel import Setup, resolve


def build_grid(D=None, step: float | None = None, setup: Setup | None = None) -> Grid:
    """
    Grille de D de pas ≤ step (bords compris) et matrice G (M × P),
    G_ij = G(x_i, y_j). ``D`` est un ``Box`` (défaut : domaine du setup).
    Voir ``common.make_grid``.
    """
    return make_grid(D, step, setup)


# ---- Solveurs ----------------------------------------------------------------------------
def fista_positive_lasso(G: np.ndarray, m: np.ndarray, lam: float, tol: float = 1e-4,
                         max_iter: int = 20_000, normalize: bool = False,
                         c0: np.ndarray | None = None):
    """
    FISTA (Beck & Teboulle 2009) avec redémarrage adaptatif (O'Donoghue &
    Candès 2015) pour

        min_{c ≥ 0}  f(c) + λ Σ c_j,     f(c) = ½ ||m - G c||².

    Itération (L = ||G||₂², constante de Lipschitz de ∇f = G^T (G c - m)) :

        c_{k+1} = prox(y_k - ∇f(y_k)/L),   prox(v) = max(v - λ/L, 0)
        t_{k+1} = (1 + sqrt(1 + 4 t_k²)) / 2
        y_{k+1} = c_{k+1} + (t_k - 1)/t_{k+1} (c_{k+1} - c_k)

    et t ← 1 dès que (y_k - c_{k+1})·(c_{k+1} - c_k) > 0 (le moment repart
    dans la mauvaise direction).

    Préconditionnement facultatif (``normalize=True``) : changement de
    variable c = D u avec D = diag(1/||G_j||). Le problème en u,
        min_{u ≥ 0} ½ ||m - (G D) u||² + λ Σ_j u_j / ||G_j||,
    a exactement le même minimiseur (c = D u) ; seul le terme linéaire devient
    pondéré (prox : max(v - λ w_j / L, 0)). En pratique il n'accélère pas la
    convergence ici (testé), d'où la valeur par défaut False.

    Remarque : les colonnes de nœuds voisins sont presque colinéaires (noyau
    lisse, M = 64), le problème est très mal conditionné et FISTA converge
    lentement (≈ 1e-5 d'écart sur l'objectif après 1e4 itérations). D'où la
    tolérance par défaut 1e-4 ; ``method="active_set"`` donne la solution
    exacte pour comparaison.

    Arrêt : conditions KKT, avec g = G^T (G c - m) :
        c_j > 0 : g_j + λ = 0 ;    c_j = 0 : g_j + λ ≥ 0,
    violation max (relative à λ) < tol.

    Renvoie (c (P,), infos {n_iter, kkt, converged}).
    """
    G = np.asarray(G, float)
    m = np.asarray(m, float)
    P = G.shape[1]
    norms = np.linalg.norm(G, axis=0) if normalize else np.ones(P)
    norms[norms == 0] = 1.0
    B = G / norms                                     # G D
    w = lam / norms                                   # poids du terme linéaire en u
    L = float(np.linalg.eigvalsh(B @ B.T)[-1])        # ||B||₂², via la matrice M × M
    u = np.zeros(P) if c0 is None else np.asarray(c0, float) * norms
    y, t = u.copy(), 1.0
    Btm = B.T @ m
    kkt = np.inf
    for k in range(1, max_iter + 1):
        grad = B.T @ (B @ y) - Btm
        u_new = np.maximum(y - (grad + w) / L, 0.0)
        if np.dot(y - u_new, u_new - u) > 0:          # redémarrage adaptatif
            t = 1.0
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = u_new + (t - 1) / t_new * (u_new - u)
        u, t = u_new, t_new
        if k % 50 == 0 or k == max_iter:
            g = G.T @ (G @ (u / norms) - m) + lam     # ∂/∂c de l'objectif
            viol = np.where(u > 0, np.abs(g), np.maximum(-g, 0.0))
            kkt = float(viol.max() / lam)
            if kkt < tol:
                break
    return u / norms, {"n_iter": k, "kkt": kkt, "converged": kkt < tol}


def sklearn_positive_lasso(G: np.ndarray, m: np.ndarray, lam: float, tol: float = 1e-8,
                           max_iter: int = 20_000):
    """
    Même problème avec ``sklearn.linear_model.Lasso(positive=True)``.

    Conversion de λ : sklearn minimise
        (1 / (2 n)) ||y - X w||² + α ||w||₁,      n = nombre d'échantillons = M.
    En multipliant par n : ½ ||y - X w||² + n α ||w||₁, donc α = λ / M.
    Pas d'intercept (``fit_intercept=False``).
    """
    from sklearn.linear_model import Lasso

    M = G.shape[0]
    model = Lasso(alpha=lam / M, positive=True, fit_intercept=False, tol=tol,
                  max_iter=max_iter, selection="cyclic")
    model.fit(G, m)
    return model.coef_.copy(), {"n_iter": int(model.n_iter_)}


def active_set_positive_lasso(G: np.ndarray, m: np.ndarray, lam: float, **kwargs):
    """
    Solution exacte du même problème par l'algorithme à ensemble actif du
    projet (Lawson-Hanson adapté au terme λ Σ c_j,
    ``particle_sources.duval_peyre.blasso._positive_lasso``). Sert de référence
    pour FISTA et sklearn.
    """
    from particle_sources.duval_peyre.blasso import _positive_lasso

    c = _positive_lasso(np.asarray(G, float), np.asarray(m, float), lam, **kwargs)
    return c, {}


def lasso_objective(G, m, lam, c) -> float:
    """½ ||m - G c||² + λ Σ c_j."""
    r = m - G @ c
    return float(0.5 * r @ r + lam * np.sum(c))


def solve_lasso(m, lam: float, grid: Grid | float | None = None, method: str = "fista",
                setup: Setup | None = None, **kwargs):
    """
    min_{c ≥ 0} ½ ||m - G c||² + λ ||c||₁ sur la grille.

    Paramètres
    ----------
    m : (M,) mesures
    lam : float, λ > 0
    grid : Grid, pas (float) ou None (pas par défaut)
    method : "fista" (codé ici), "sklearn" ou "active_set" (exact, référence)
    kwargs : passés au solveur (tol, max_iter, ...)

    Renvoie (c (P,), grid, infos).
    """
    setup = resolve(setup)
    if not isinstance(grid, Grid):
        grid = build_grid(setup.domain, grid, setup)
    if method == "fista":
        c, info = fista_positive_lasso(grid.G, m, lam, **kwargs)
    elif method == "sklearn":
        c, info = sklearn_positive_lasso(grid.G, m, lam, **kwargs)
    elif method == "active_set":
        c, info = active_set_positive_lasso(grid.G, m, lam, **kwargs)
    else:
        raise ValueError(f"méthode inconnue : {method!r} (fista, sklearn ou active_set)")
    info["objective"] = lasso_objective(grid.G, m, lam, c)
    return c, grid, info


# ---- Regroupement et raffinement -------------------------------------------------------------
def merge_peaks(c, grid: Grid, radius: float | None = None, rel_threshold: float = 1e-4):
    """
    Regroupe les nœuds actifs voisins en une source chacun.

    Nœuds actifs : c_j > rel_threshold × max c. Deux nœuds actifs sont reliés
    s'ils sont à distance ≤ radius (défaut 1.5 × pas effectif : 8-voisinage) ;
    chaque composante connexe C donne une source

        I = Σ_{j ∈ C} c_j,     z = Σ_{j ∈ C} c_j y_j / I.

    Renvoie (z (K, 2), I (K,), labels (P,) : numéro de composante, -1 si inactif).
    """
    c = np.asarray(c, float)
    labels = -np.ones(len(c), int)
    if c.max(initial=0.0) <= 0:
        return np.zeros((0, 2)), np.zeros(0), labels
    if radius is None:
        dx = np.diff(grid.xs[:2])[0] if len(grid.xs) > 1 else grid.step
        dy = np.diff(grid.ys[:2])[0] if len(grid.ys) > 1 else grid.step
        radius = 1.5 * max(dx, dy)
    active = np.flatnonzero(c > rel_threshold * c.max())
    pts = grid.flat[active]
    pairs = np.array(sorted(cKDTree(pts).query_pairs(radius)), int).reshape(-1, 2)
    n = len(active)
    adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, lab = connected_components(adj, directed=False)
    labels[active] = lab
    I = np.bincount(lab, weights=c[active], minlength=n_comp)
    z = np.stack([np.bincount(lab, weights=c[active] * pts[:, a], minlength=n_comp)
                  for a in range(2)], axis=1) / I[:, None]
    order = np.argsort(-I)
    return z[order], I[order], labels


def refine(z0, I0, m, lam: float, setup: Setup | None = None, merge_dist: float | None = None):
    """
    K fixé, optimisation locale sans grille :

        min_{I ≥ 0, z_k ∈ D}  ½ ||m - Σ_k I_k φ(z_k)||² + λ Σ_k I_k

    (L-BFGS-B avec bornes, gradient analytique : voir ``common.joint_descent``).
    Les sources mises à 0 sont retirées, celles qui se rejoignent
    (distance < merge_dist, défaut 1e-3 × diagonale de D) fusionnées.

    Renvoie (z, I, F final).
    """
    setup = resolve(setup)
    if merge_dist is None:
        merge_dist = 1e-3 * float(np.linalg.norm(setup.domain.size))
    z, I, _ = joint_descent(z0, I0, m, lam, setup)
    z, I = prune_and_merge(z, I, merge_dist)
    r = m - (I @ setup.phi(z) if len(I) else 0)
    return z, I, float(0.5 * r @ r + lam * I.sum())


# ---- Pipeline ----------------------------------------------------------------------------
def lasso_pipeline(m, lam: float, step: float | None = None, sigma=None,
                   refine_positions: bool = True, method: str = "fista",
                   grid: Grid | None = None, setup: Setup | None = None,
                   rel_threshold: float = 1e-4, verbose: bool = False, **solver_kwargs) -> dict:
    """
    Grille → LASSO → regroupement → raffinement (optionnel) → vérification
    max η ≤ 1 → débiaisage → diagnostics.

    Paramètres
    ----------
    m : (M,) mesures ; lam : λ ; step : pas de grille (ignoré si ``grid`` est donnée)
    sigma : niveau de bruit (scalaire ou (M,)) pour les diagnostics, ou None
    refine_positions : libérer les positions après regroupement (``refine``)
    method : "fista", "sklearn" ou "active_set"

    Renvoie un dict :
        z, I                 sources finales (débiaisées)
        z_merged, I_merged   après regroupement (avant raffinement)
        z_lambda, I_lambda   solution pénalisée (après raffinement si demandé)
        c, grid              coefficients sur la grille
        n_active             nombre de nœuds actifs avant regroupement
        eta_max, eta_argmax  max du certificat sur D pour la solution pénalisée
        certified            eta_max ≤ 1 + 1e-3 (optimalité pour le BLASSO)
        diagnostics, time, solver
    """
    setup = resolve(setup)
    t0 = time.perf_counter()
    if grid is None:
        grid = build_grid(setup.domain, step, setup)
    c, grid, info = solve_lasso(m, lam, grid, method, setup, **solver_kwargs)
    z_merged, I_merged, labels = merge_peaks(c, grid, rel_threshold=rel_threshold)
    z, I = z_merged, I_merged
    if refine_positions and len(I):
        z, I, _ = refine(z, I, m, lam, setup)
    r = m - (I @ setup.phi(z) if len(I) else 0)
    z_star, eta_max = max_certificate(r, lam, grid, setup)
    z_deb, I_deb = debias(z, I, m, setup)
    out = {
        "z": z_deb, "I": I_deb,
        "z_merged": z_merged, "I_merged": I_merged,
        "z_lambda": z, "I_lambda": I,
        "c": c, "grid": grid, "n_active": int(np.sum(labels >= 0)),
        "eta_max": eta_max, "eta_argmax": z_star, "certified": eta_max <= 1 + 1e-3,
        "solver": info, "time": time.perf_counter() - t0,
    }
    if sigma is not None:
        out["diagnostics"] = diagnostics(z_deb, I_deb, m, sigma, setup)
    if verbose:
        print(f"LASSO ({method}) : {out['n_active']} nœuds actifs -> {len(I_merged)} pics "
              f"-> {len(I_deb)} sources ; max η = {eta_max:.4f} "
              f"({'certifié' if out['certified'] else 'NON certifié'}), {out['time']:.2f} s")
    return out

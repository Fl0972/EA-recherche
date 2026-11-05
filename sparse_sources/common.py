"""
Briques communes au LASSO sur grille et au Frank-Wolfe glissant.

Notations
---------
Une mesure positive discrète s = Σ_k I_k δ_{z_k} (z_k ∈ D ⊂ R², I_k ≥ 0) donne
les mesures

    A(s) = Σ_k I_k φ(z_k) ∈ R^M,      φ(z) = (G(x_1, z), ..., G(x_M, z)).

BLASSO positif (problème sur les mesures, sans grille) :

    min_{s ≥ 0}  F(s) = ½ ||m - A(s)||² + λ ||s||_TV,   ||s||_TV = Σ_k I_k.

Certificat dual : pour le résidu r = m - A(s),

    η(z) = (1/λ) Σ_i r_i G(x_i, z) = φ(z)·r / λ.

s est solution si et seulement si η ≤ 1 sur D et η(z_k) = 1 sur le support
(conditions d'optimalité du BLASSO positif). ``max_certificate`` mesure donc
l'écart à l'optimalité, et c'est le critère d'arrêt du Frank-Wolfe glissant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment, minimize, nnls

from .kernel import Setup, resolve

# Options de L-BFGS-B pour les descentes locales (précision élevée : on veut
# des positions à 1e-4 près sans bruit)
LBFGS_OPTIONS = {"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12, "maxcor": 30}


# ---- Grille --------------------------------------------------------------------------
@dataclass
class Grid:
    """
    Grille régulière de D (bords compris) et matrice G associée.

    Attributs
    ---------
    xs, ys : np.ndarray
        Coordonnées des nœuds le long de x (n_x) et de y (n_y).
    points : np.ndarray, forme (n_y, n_x, 2)
        Nœuds y_j, rangés comme une image (ordre de matplotlib).
    step : float
        Pas demandé (le pas effectif est ≤ step dans chaque direction).
    G : np.ndarray, forme (M, P), P = n_y n_x
        G[i, j] = G(x_i, y_j) : la colonne j est φ(y_j).
    """
    xs: np.ndarray
    ys: np.ndarray
    points: np.ndarray
    step: float
    G: np.ndarray

    @property
    def shape(self) -> tuple:
        """(n_y, n_x)."""
        return self.points.shape[:2]

    @property
    def flat(self) -> np.ndarray:
        """Nœuds sous forme de liste, (P, 2), dans l'ordre des colonnes de G."""
        return self.points.reshape(-1, 2)

    @property
    def P(self) -> int:
        """Nombre de nœuds."""
        return self.flat.shape[0]


def make_grid(domain=None, step: float | None = None, setup: Setup | None = None) -> Grid:
    """
    Grille de pas ≤ ``step`` couvrant ``domain`` (défaut : D du setup), et la
    matrice G (M × P) des colonnes φ(y_j).

    Par défaut step = min(taille de D) / 100.
    """
    setup = resolve(setup)
    box = domain if domain is not None else setup.domain
    if step is None:
        step = float(min(box.size)) / 100
    n = np.ceil(box.size / step).astype(int) + 1
    xs = np.linspace(box.lo[0], box.hi[0], n[0])
    ys = np.linspace(box.lo[1], box.hi[1], n[1])
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
    G = setup.phi(points.reshape(-1, 2)).T
    return Grid(xs, ys, points, float(step), G)


def _as_grid(grid, setup: Setup) -> Grid:
    """Accepte une Grid, un pas (float) ou None (pas par défaut)."""
    if isinstance(grid, Grid):
        return grid
    return make_grid(setup.domain, grid, setup)


# ---- Modèle direct et certificat ---------------------------------------------------------
def forward(z, I, setup: Setup | None = None) -> np.ndarray:
    """
    A(s) = Σ_k I_k φ(z_k), forme (M,).

    z : (K, 2), I : (K,). K = 0 est permis (renvoie 0).
    """
    setup = resolve(setup)
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    if len(I) == 0:
        return np.zeros(setup.M)
    return I @ setup.phi(z)


def certificate(residual, lam: float, Z, setup: Setup | None = None) -> np.ndarray:
    """
    η(z) = (1/λ) Σ_i r_i G(x_i, z) évalué en chaque point de Z (forme (..., 2)).

    Renvoie un tableau de forme (...).
    """
    return resolve(setup).phi(Z) @ np.asarray(residual, float) / lam


def certificate_grad(residual, lam: float, Z, setup: Setup | None = None) -> np.ndarray:
    """∇η(z) = (1/λ) Σ_i r_i ∇_z G(x_i, z), forme (..., 2)."""
    return np.einsum("...ia,i->...a", resolve(setup).grad(Z), np.asarray(residual, float)) / lam


def maximize_on_domain(f, grad, box, starts):
    """
    Maximise f sur le rectangle ``box`` par L-BFGS-B (bornes = D) depuis
    chaque point de ``starts``. Renvoie la liste des (valeur, point).
    """
    bounds = list(zip(box.lo, box.hi))
    out = []
    for x0 in starts:
        res = minimize(lambda x: -f(x), np.asarray(x0, float), jac=lambda x: -grad(x),
                       method="L-BFGS-B", bounds=bounds)
        out.append((-float(res.fun), res.x))
    return out


def grid_local_maxima(img: np.ndarray, n: int) -> np.ndarray:
    """Indices (ligne, colonne) des n plus grands maxima locaux (8-voisinage) d'une image."""
    peaks = np.argwhere((img == maximum_filter(img, size=3, mode="nearest")) & np.isfinite(img))
    return peaks[np.argsort(-img[tuple(peaks.T)])][:n]


def max_certificate(residual, lam: float, grid=None, setup: Setup | None = None,
                    n_starts: int = 5):
    """
    max_{z ∈ D} η(z) : grille fine, puis raffinement local.

    1. η est évalué sur tous les nœuds (η_grille = r^T G / λ) ;
    2. L-BFGS-B, borné à D, maximise η depuis les ``n_starts`` plus grands
       maxima locaux de la grille (le bord de D est traité par les bornes).

    Paramètres
    ----------
    residual : (M,)  r = m - A(s)
    lam : float      λ > 0
    grid : Grid, float (pas) ou None
        Grille fine ; passer une Grid déjà construite évite de recalculer G.

    Renvoie
    -------
    (z*, η(z*)) : np.ndarray (2,), float
    """
    setup = resolve(setup)
    grid = _as_grid(grid, setup)
    r = np.asarray(residual, float)
    img = (r @ grid.G / lam).reshape(grid.shape)
    peaks = grid_local_maxima(img, n_starts)
    best_val, best_pt = float(img[tuple(peaks[0])]), grid.points[tuple(peaks[0])]
    f = lambda x: float(certificate(r, lam, x, setup))
    g = lambda x: certificate_grad(r, lam, x, setup)
    for v, x in maximize_on_domain(f, g, setup.domain, [grid.points[tuple(i)] for i in peaks]):
        if v > best_val:
            best_val, best_pt = v, x
    return np.asarray(best_pt, float), float(best_val)


# ---- Objectif du BLASSO pour K Diracs et descente locale ---------------------------------
def objective(z, I, m, lam: float, setup: Setup | None = None) -> float:
    """F(Σ I_k δ_{z_k}) = ½ ||m - A(s)||² + λ Σ_k I_k."""
    r = m - forward(z, I, setup)
    return float(0.5 * r @ r + lam * np.sum(I))


def objective_and_grad(z, I, m, lam: float, setup: Setup | None = None):
    """
    F et son gradient pour K Diracs (positions et intensités variables).

    Avec e = A(s) - m :
        ∂F/∂I_k = φ(z_k)·e + λ
        ∇_{z_k} F = I_k ∇_z φ(z_k)^T e
    Renvoie (F, ∂F/∂I (K,), ∇_z F (K, 2)).
    """
    setup = resolve(setup)
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float)
    Ph = setup.phi(z)                                   # (K, M)
    e = I @ Ph - m                                      # (M,)
    gI = Ph @ e + lam
    gz = I[:, None] * np.einsum("kia,i->ka", setup.grad(z), e)
    return float(0.5 * e @ e + lam * I.sum()), gI, gz


def joint_descent(z0, I0, m, lam: float, setup: Setup | None = None,
                  options: dict | None = None):
    """
    Optimisation locale conjointe, K fixé :

        min_{I ≥ 0, z_k ∈ D}  ½ ||m - Σ_k I_k φ(z_k)||² + λ Σ_k I_k

    par L-BFGS-B (bornes : I_k ≥ 0, z_k dans D), gradient analytique
    (``objective_and_grad``). Les intensités sont mises à l'échelle
    (I = S·u, S = max I0) pour équilibrer les variables.

    Renvoie (z (K, 2), I (K,), F final).
    """
    setup = resolve(setup)
    z0 = np.asarray(z0, float).reshape(-1, 2)
    I0 = np.asarray(I0, float).reshape(-1)
    K = len(I0)
    if K == 0:
        return z0, I0, objective(z0, I0, m, lam, setup)
    S = max(float(I0.max()), 1e-12)

    def fg(x):
        u, z = x[:K], x[K:].reshape(K, 2)
        F, gI, gz = objective_and_grad(z, S * u, m, lam, setup)
        return F, np.concatenate([S * gI, gz.ravel()])

    bounds = [(0, None)] * K + list(zip(setup.domain.lo, setup.domain.hi)) * K
    x0 = np.concatenate([I0 / S, np.clip(z0, setup.domain.lo, setup.domain.hi).ravel()])
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                   options=options or LBFGS_OPTIONS)
    z, I = res.x[K:].reshape(K, 2), S * res.x[:K]
    return z, I, float(res.fun)


def prune_and_merge(z, I, merge_dist: float, rel_zero: float = 1e-10):
    """
    Retire les Diracs d'intensité nulle (≤ rel_zero × max I) et fusionne ceux
    qui sont à distance < merge_dist : intensité = somme, position = moyenne
    pondérée par les intensités (fusion gloutonne, la plus forte d'abord).
    """
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    if len(I) == 0:
        return z, I
    keep = I > rel_zero * max(I.max(), 1e-300)
    z, I = z[keep], I[keep]
    order = np.argsort(-I)
    z, I = z[order], I[order]
    used = np.zeros(len(I), bool)
    zs, Is = [], []
    for k in range(len(I)):
        if used[k]:
            continue
        group = (~used) & (np.linalg.norm(z - z[k], axis=1) < merge_dist)
        used |= group
        w = I[group]
        zs.append(w @ z[group] / w.sum())
        Is.append(w.sum())
    return np.array(zs).reshape(-1, 2), np.array(Is)


# ---- Débiaisage, diagnostics, erreurs ----------------------------------------------------
def debias(z, I_init, m, setup: Setup | None = None):
    """
    Débiaisage : positions fixées, intensités réajustées SANS pénalité,

        min_{I ≥ 0} ||m - Σ_k I_k φ(z_k)||²    (scipy.optimize.nnls).

    Le terme λ Σ I_k du BLASSO tire les intensités vers 0 (biais ~ λ) ; ce
    réajustement le supprime. Seules les sources d'intensité I_init > 0 sont
    gardées ; celles que NNLS met à 0 sont retirées.

    Renvoie (z, I) débiaisés.
    """
    z = np.asarray(z, float).reshape(-1, 2)
    I_init = np.asarray(I_init, float).reshape(-1)
    keep = I_init > 0
    z = z[keep]
    if len(z) == 0:
        return z, np.zeros(0)
    I, _ = nnls(resolve(setup).phi(z).T, np.asarray(m, float), maxiter=50 * len(z) + 100)
    keep = I > 0
    return z[keep], I[keep]


def diagnostics(z, I, m, sigma, setup: Setup | None = None, verbose: bool = False) -> dict:
    """
    Qualité de l'ajustement.

    Pour un bruit gaussien de niveau σ (σ scalaire, ou vecteur (M,) d'écarts
    types par capteur si le bruit est hétéroscédastique), on attend
        ||r||² ≈ Σ_i σ_i² = M σ²       (bruit seul),
    et ≈ (M - 3K) σ² après un ajustement à 3K paramètres (positions + intensités).
    Un rapport ||r||² / (Mσ²) ≫ 1 signale des sources manquantes ; ≪ 1 un
    sur-ajustement (sources en trop qui modélisent le bruit).
    """
    setup = resolve(setup)
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    r = np.asarray(m, float) - forward(z, I, setup)
    s2 = np.broadcast_to(np.asarray(sigma, float) ** 2, r.shape)
    noise = float(s2.sum())
    K = len(I)
    out = {
        "K": K,
        "residual_sq": float(r @ r),
        "M_sigma2": noise,
        "ratio": float(r @ r / noise) if noise > 0 else np.inf,
        "expected_ratio": max(setup.M - 3 * K, 0) / setup.M,
        "chi2_normalized": float(np.sum(r**2 / s2)) / max(setup.M - 3 * K, 1) if noise > 0
        else np.inf,
        "total_intensity": float(I.sum()),
    }
    if verbose:
        print(f"K = {K} sources, ||r||² = {out['residual_sq']:.4g}, "
              f"Mσ² = {noise:.4g} (rapport {out['ratio']:.3g}, "
              f"attendu ≈ {out['expected_ratio']:.3g})")
    return out


def match_sources(z_true, I_true, z, I) -> dict:
    """
    Apparie sources estimées et vraies (algorithme hongrois sur les distances).

    Renvoie : n_true, n_est, exact (n_est == n_true), err_pos (distance max
    entre paires appariées), err_int (erreur relative d'intensité max sur les
    paires), err_int_total (erreur relative sur la somme des intensités).
    """
    z_true = np.asarray(z_true, float).reshape(-1, 2)
    I_true = np.asarray(I_true, float).reshape(-1)
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    out = {"n_true": len(I_true), "n_est": len(I), "exact": len(I) == len(I_true),
           "err_pos": np.nan, "err_int": np.nan,
           "err_int_total": abs(I.sum() - I_true.sum()) / I_true.sum() if I_true.sum() else np.nan}
    if len(I) and len(I_true):
        D = np.linalg.norm(z_true[:, None] - z[None], axis=-1)
        i, j = linear_sum_assignment(D)
        out["err_pos"] = float(D[i, j].max())
        out["err_int"] = float(np.max(np.abs(I[j] - I_true[i]) / I_true[i]))
    return out

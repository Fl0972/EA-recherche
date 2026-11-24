"""
Briques communes à ML + BIC et au BLASSO (Frank–Wolfe glissant).

Une mesure positive discrète s = Σ_k I_k δ_{z_k} donne les mesures
A(s) = Σ_k I_k φ(z_k). Pour le BLASSO positif

    min_{s ≥ 0}  F(s) = ½ ||m − A(s)||² + λ Σ_k I_k,

le certificat dual est η(z) = φ(z)·r / λ (r = m − A(s)) : s est solution si et
seulement si η ≤ 1 sur D et η(z_k) = 1 sur le support. ML + BIC utilise les
mêmes briques avec λ = 0 (maximum de corrélation, descente locale).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize, nnls

from .noyau import Grid, Setup, make_grid

# L-BFGS-B pour les descentes locales (précision élevée)
LBFGS_OPTIONS = {"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12, "maxcor": 30}


def _as_grid(grid, setup: Setup) -> Grid:
    if isinstance(grid, Grid):
        return grid
    return make_grid(setup.domain, grid, setup)


def forward(z, I, setup: Setup) -> np.ndarray:
    """A(s) = Σ_k I_k φ(z_k), forme (M,) ; K = 0 permis."""
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    if len(I) == 0:
        return np.zeros(setup.M)
    return I @ setup.phi(z)


def certificate(residual, lam: float, Z, setup: Setup) -> np.ndarray:
    """η(z) = φ(z)·r / λ en chaque point de Z."""
    return setup.phi(Z) @ np.asarray(residual, float) / lam


def certificate_grad(residual, lam: float, Z, setup: Setup) -> np.ndarray:
    """∇η(z), forme (..., 2)."""
    return np.einsum("...ia,i->...a", setup.grad(Z), np.asarray(residual, float)) / lam


def maximize_on_domain(f, grad, box, starts):
    """Maximise f sur le rectangle ``box`` par L-BFGS-B depuis chaque départ."""
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


def max_certificate(residual, lam: float, grid, setup: Setup, n_starts: int = 5):
    """
    max_{z ∈ D} η(z) : évaluation sur la grille, puis L-BFGS-B depuis les ``n_starts``
    plus grands maxima locaux. Renvoie (z*, η(z*)).
    """
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


def objective(z, I, m, lam: float, setup: Setup) -> float:
    """F = ½ ||m − A(s)||² + λ Σ_k I_k."""
    r = m - forward(z, I, setup)
    return float(0.5 * r @ r + lam * np.sum(I))


def objective_and_grad(z, I, m, lam: float, setup: Setup):
    """F, ∂F/∂I (K,) et ∇_z F (K, 2) ; avec e = A(s) − m : ∂F/∂I_k = φ(z_k)·e + λ."""
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float)
    Ph = setup.phi(z)
    e = I @ Ph - m
    gI = Ph @ e + lam
    gz = I[:, None] * np.einsum("kia,i->ka", setup.grad(z), e)
    return float(0.5 * e @ e + lam * I.sum()), gI, gz


def joint_descent(z0, I0, m, lam: float, setup: Setup, options: dict | None = None):
    """
    Glissement : min_{I ≥ 0, z_k ∈ D} ½ ||m − Σ_k I_k φ(z_k)||² + λ Σ_k I_k, K fixé,
    par L-BFGS-B (gradient analytique, intensités mises à l'échelle). Renvoie (z, I, F).
    """
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
    Retire les Diracs d'intensité nulle et fusionne ceux à distance < merge_dist
    (intensité = somme, position = moyenne pondérée ; la plus forte d'abord).
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


def debias(z, I_init, m, setup: Setup):
    """
    Débiaisage : positions fixées, intensités par moindres carrés positifs (NNLS),
    sans pénalité. Les sources que NNLS met à 0 sont retirées.
    """
    z = np.asarray(z, float).reshape(-1, 2)
    I_init = np.asarray(I_init, float).reshape(-1)
    keep = I_init > 0
    z = z[keep]
    if len(z) == 0:
        return z, np.zeros(0)
    I, _ = nnls(setup.phi(z).T, np.asarray(m, float), maxiter=50 * len(z) + 100)
    keep = I > 0
    return z[keep], I[keep]


def positive_lasso(A, m, lam, max_iter: int = 2000):
    """
    min_{a ≥ 0} ½ ||m − A a||² + λ Σ a, solution exacte par ensemble actif
    (Lawson–Hanson adapté au terme linéaire λ Σ a).
    """
    N = A.shape[1]
    a = np.zeros(N)
    P = np.zeros(N, bool)
    tol = 1e-12 * max(np.abs(A.T @ m).max(), lam, 1e-300)

    def solve(P):
        AP = A[:, P]
        pinv = np.linalg.pinv(AP)
        return pinv @ (m - lam * pinv.T @ np.ones(P.sum()))

    for _ in range(max_iter):
        w = A.T @ (m - A @ a) - lam
        w[P] = -np.inf
        j = int(np.argmax(w))
        if w[j] <= tol:
            break
        P[j] = True
        while True:
            s = np.zeros(N)
            s[P] = solve(P)
            if np.all(s[P] > 0):
                a = s
                break
            neg = P & (s <= 0)
            alpha = np.min(a[neg] / (a[neg] - s[neg]))
            a = a + alpha * (s - a)
            P &= a > 1e-14 * max(a.max(), 1e-300)
            a[~P] = 0.0
            if not P.any():
                break
    return a

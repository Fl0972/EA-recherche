"""
Maximum de vraisemblance glouton + BIC et borne de Cramér-Rao en dimension
quelconque (sources en 2D ou en 3D).

Même méthode que ``ml.greedy_ml``, mais sans la géométrie 2D de ``Setup`` :
on donne directement les capteurs (M, d) et le domaine (``Box`` de dimension
d). Le noyau est celui du projet en dimension d,

    G(x_i, z) = 1 / (||x_i - z||² + eps²)     (``InverseSquareKernel``),

et chaque source a d + 1 paramètres (d coordonnées + l'intensité), d'où le
critère

    C(K) = ||r_w||² + κ (d + 1) K,     κ = log M (BIC) ou 2 (AIC),

avec r_w = (m - A(s)) / s le résidu blanchi.

Algorithme (``greedy_ml_nd``) :
    1. résidu blanchi r_w ;
    2. nouvelle source au maximum de la corrélation normalisée
       ⟨φ_w(z) / ||φ_w(z)||, r_w⟩ : grille de D, puis L-BFGS-B depuis les
       meilleurs nœuds ;
    3. intensités par NNLS, puis glissement conjoint (positions + intensités,
       sans pénalité) par L-BFGS-B avec gradient analytique ;
    4. suppression des intensités nulles, fusion des sources confondues ;
    5. arrêt quand C(K) n'a pas baissé pendant ``patience`` ajouts ; on garde
       le meilleur K, puis étape arrière (retrait d'une source si C baisse).

En 3D, un seul cercle de capteurs ne suffit pas (problème non identifiable,
voir ``particle_sources.duval_peyre.experiments_3d``) : il faut des capteurs
non coplanaires (deux ou trois cercles, sphère).
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize, nnls

from particle_sources.duval_peyre.kernel import InverseSquareKernel

from .ml import _penalty

LBFGS = {"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-11}


def box_grid(box, step: float) -> np.ndarray:
    """Nœuds d'une grille régulière de ``box`` (bords compris), pas ≤ step : (P, d)."""
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    axes = [np.linspace(a, b, max(2, int(np.ceil((b - a) / step)) + 1)) for a, b in zip(lo, hi)]
    return np.stack([g.ravel() for g in np.meshgrid(*axes, indexing="ij")], axis=1)


class _Model:
    """Noyau blanchi φ_w = w ⊙ φ et ses dérivées, domaine et grille."""

    def __init__(self, sensors, box, noise_std, eps, grid_step):
        self.k = InverseSquareKernel(np.atleast_2d(np.asarray(sensors, float)), eps)
        self.d = self.k.sensors.shape[1]
        self.lo, self.hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
        if len(self.lo) != self.d:
            raise ValueError("le domaine et les capteurs n'ont pas la même dimension")
        self.w = 1.0 / np.broadcast_to(np.asarray(noise_std, float), (self.k.m,))
        step = grid_step or float(np.min(self.hi - self.lo)) / (40 if self.d == 2 else 20)
        self.P = box_grid(box, step)
        Gw = self.phi(self.P)
        self.Gn = Gw / np.linalg.norm(Gw, axis=1, keepdims=True)

    def phi(self, z):
        """φ_w(z), forme (K, M)."""
        return self.k.phi(np.asarray(z, float).reshape(-1, self.d)) * self.w

    def grad(self, z):
        """∇φ_w(z), forme (K, M, d)."""
        return self.k.grad(np.asarray(z, float).reshape(-1, self.d)) * self.w[:, None]

    def corr(self, z, r):
        """⟨φ_w(z) / ||φ_w(z)||, r⟩ pour un point z."""
        p = self.phi(z)[0]
        return float(p @ r / np.linalg.norm(p))


def _best_candidate(model, r, n_starts):
    """Maximum de la corrélation normalisée : grille puis L-BFGS-B (n_starts départs)."""
    score = model.Gn @ r
    starts = model.P[np.argsort(score)[::-1][:n_starts]]
    best = (-np.inf, starts[0])
    bounds = list(zip(model.lo, model.hi))
    for z0 in starts:
        res = minimize(lambda z: -model.corr(z, r), z0, method="L-BFGS-B", bounds=bounds)
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    return best[1]


def _slide(model, z, I, mw):
    """min_{z ∈ D, I ≥ 0} ½ ||m_w - Σ I_k φ_w(z_k)||² (L-BFGS-B, gradient analytique)."""
    K, d = len(I), model.d
    if K == 0:
        return z, I
    scale = max(float(I.max()), 1e-12)
    I = np.maximum(I, 1e-3 * scale)

    def fg(x):
        u, zz = x[:K], x[K:].reshape(K, d)
        F = model.phi(zz)
        r = mw - (scale * u) @ F
        J = model.grad(zz)
        g_u = -scale * (F @ r)
        g_z = -(scale * u)[:, None] * np.einsum("kmd,m->kd", J, r)
        return 0.5 * float(r @ r), np.concatenate([g_u, g_z.ravel()])

    bounds = [(0, None)] * K + list(zip(model.lo, model.hi)) * K
    x0 = np.concatenate([I / scale, np.clip(z, model.lo, model.hi).ravel()])
    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=LBFGS)
    return res.x[K:].reshape(K, d), scale * res.x[:K]


def _prune_merge(z, I, merge_dist, rel_zero=1e-3):
    """Retire I ≤ rel_zero × max I, fusionne les sources à moins de merge_dist."""
    if len(I) == 0:
        return z, I
    keep = I > rel_zero * I.max()
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
        zs.append(I[group] @ z[group] / I[group].sum())
        Is.append(I[group].sum())
    return np.array(zs).reshape(-1, z.shape[1]), np.array(Is)


def greedy_ml_nd(m, noise_std, sensors, box, eps: float = 0.0, k_max: int = 10,
                 criterion="bic", patience: int = 2, backward: bool = True,
                 grid_step: float | None = None, n_starts: int = 5,
                 merge_dist: float | None = None, verbose: bool = False) -> dict:
    """
    Maximum de vraisemblance glouton, K choisi par critère d'information.

    Paramètres
    ----------
    m : (M,) mesures (sources connues, par ex. la source centrale, retirées)
    noise_std : (M,) ou scalaire, écart type du bruit de chaque capteur
    sensors : (M, d) positions des capteurs ; box : domaine des sources (dimension d)
    eps : régularisation du noyau (0 = loi exacte en 1/r²)
    k_max : nombre maximal de sources
    criterion : 'bic' (défaut), 'aic' ou κ numérique (0 avec k_max = K : K imposé)
    patience : ajouts sans amélioration avant l'arrêt ; backward : étape arrière
    grid_step : pas de la grille de recherche (défaut : côté min / 20 en 3D, / 40 en 2D)
    n_starts : nombre de nœuds de la grille raffinés à chaque ajout
    merge_dist : distance de fusion (défaut 1e-3 × diagonale de D)

    Renvoie {z (K, d), I (K,), K, criterion, chi2, history, time}.
    """
    t0 = time.perf_counter()
    m = np.asarray(m, float)
    model = _Model(sensors, box, noise_std, eps, grid_step)
    mw = m * model.w
    kappa = _penalty(criterion, len(m))
    n_par = model.d + 1
    if merge_dist is None:
        merge_dist = 1e-3 * float(np.linalg.norm(model.hi - model.lo))

    def chi2(z, I):
        r = mw - (I @ model.phi(z) if len(I) else 0.0)
        return float(r @ r)

    def crit(z, I):
        return chi2(z, I) + kappa * n_par * len(I)

    def refit(z, I):
        z, I = _slide(model, z, I, mw)
        z, I = _prune_merge(z, I, merge_dist)
        return _slide(model, z, I, mw) if len(I) else (z, I)

    z, I = np.zeros((0, model.d)), np.zeros(0)
    best = (crit(z, I), z, I)
    history = [{"K": 0, "chi2": chi2(z, I), "criterion": best[0]}]
    fails = 0
    while len(I) < k_max:
        r = mw - (I @ model.phi(z) if len(I) else 0.0)
        z = np.vstack([z, _best_candidate(model, r, n_starts)])
        I, _ = nnls(model.phi(z).T, mw, maxiter=50 * len(z) + 100)
        z, I = refit(z, I)
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
            zk, Ik = refit(np.delete(z, k, 0), np.delete(I, k))
            ck = crit(zk, Ik)
            if ck < c - 1e-9:
                c, z, I, improved = ck, zk, Ik, True
                break
    return {"z": z, "I": I, "K": len(I), "criterion": c, "chi2": chi2(z, I),
            "history": history, "time": time.perf_counter() - t0}


def cramer_rao_nd(z, I, noise_std, sensors, eps: float = 0.0) -> dict:
    """
    Borne de Cramér-Rao pour K sources connues en nombre (dimension d).

    F = D^T diag(1/s²) D, D = [φ(z_k) | I_k ∇φ(z_k)] ; renvoie pos_std (K,)
    = sqrt(trace du bloc position de F⁻¹) pour chaque source, int_std (K,)
    et la covariance complète (intensités puis positions). Si F est singulière
    (problème non identifiable), les écarts types valent +inf.
    """
    k = InverseSquareKernel(np.atleast_2d(np.asarray(sensors, float)), eps)
    d = k.sensors.shape[1]
    z = np.asarray(z, float).reshape(-1, d)
    I = np.asarray(I, float).reshape(-1)
    K = len(I)
    s = np.broadcast_to(np.asarray(noise_std, float), (k.m,))
    D = np.hstack([k.phi(z).T, (k.grad(z) * I[:, None, None]).transpose(1, 0, 2)
                   .reshape(k.m, K * d)])
    Dw = D / s[:, None]
    F = Dw.T @ Dw
    if np.linalg.cond(F) > 1e14:      # non identifiable (par ex. un seul cercle en 3D)
        inf = np.full(K, np.inf)
        return {"pos_std": inf, "int_std": inf, "cov": np.full_like(F, np.nan)}
    cov = np.linalg.inv(F)
    v = np.diag(cov)
    return {"pos_std": np.sqrt(v[K:].reshape(K, d).sum(1)), "int_std": np.sqrt(v[:K]),
            "cov": cov}


def match_nd(z_true, I_true, z, I) -> dict:
    """Appariement hongrois (dimension quelconque) : n_est, exact, err_pos, err_int."""
    z_true, z = np.atleast_2d(z_true), np.asarray(z, float).reshape(-1, np.shape(z_true)[1])
    out = {"n_true": len(I_true), "n_est": len(I), "exact": len(I) == len(I_true),
           "err_pos": np.nan, "err_int": np.nan}
    if len(I) and len(I_true):
        C = np.linalg.norm(z_true[:, None] - z[None], axis=-1)
        a, b = linear_sum_assignment(C)
        out["err_pos"] = float(C[a, b].max())
        out["err_int"] = float(np.max(np.abs(np.asarray(I)[b] - np.asarray(I_true)[a])
                                      / np.asarray(I_true)[a]))
    return out


def min_separation_nd(z) -> float:
    """Plus petite distance entre deux sources (inf si une seule)."""
    z = np.atleast_2d(z)
    if len(z) < 2:
        return np.inf
    D = np.linalg.norm(z[:, None] - z[None], axis=-1)
    return float(D[np.triu_indices(len(z), 1)].min())


def success_nd(res: dict, z_true) -> bool:
    """Bon K et erreur de position < 0,25 × min(séparation, 4) (même critère qu'en 2D)."""
    return bool(res["exact"] and res["err_pos"] < 0.25 * min(min_separation_nd(z_true), 4.0))

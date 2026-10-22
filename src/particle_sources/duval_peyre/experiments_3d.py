"""
Expériences en 3D : sources dans un cube, capteurs hors d'un même plan.

Le modèle est celui du projet en dimension 3 (G = 1 / ||x - z||^2, eps = 0).
Avec un seul cercle de capteurs le problème n'est pas identifiable (une
famille de sources à un paramètre donne les mêmes mesures, rang(Gamma) = 3K
au lieu de 4K) : il faut des capteurs qui ne sont pas coplanaires.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import linear_sum_assignment

from ..model import Box
from .blasso import Estimate


# ---- Dispositions de capteurs ----------------------------------------------------------
def ring(center, radius, n, height) -> np.ndarray:
    """n capteurs équirépartis sur le cercle horizontal (centre, rayon) à la hauteur `height`."""
    t = 2 * np.pi * np.arange(n) / n
    return np.c_[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t),
                 np.full(n, height)]


def sphere(center, radius, n) -> np.ndarray:
    """n capteurs quasi uniformes sur la sphère (centre, rayon) : spirale de Fibonacci."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5**0.5) * i
    return np.asarray(center) + radius * np.c_[np.cos(theta) * np.sin(phi),
                                               np.sin(theta) * np.sin(phi), np.cos(phi)]


def layout(name: str, box: Box, n: int, radius_factor: float) -> np.ndarray:
    """
    Capteurs autour du cube `box`, rayon = radius_factor * rayon de la sphère
    circonscrite :
        un_cercle       n capteurs sur le cercle horizontal à mi-hauteur (non identifiable)
        deux_cercles    n/2 capteurs aux hauteurs bas et haut du cube
        trois_cercles   n/3 capteurs aux hauteurs bas, milieu et haut
        sphere          n capteurs sur la sphère
    """
    c, R = box.center, radius_factor * box.circumradius
    lo, hi = box.lo[2], box.hi[2]
    if name == "un_cercle":
        return ring(c, R, n, c[2])
    if name == "deux_cercles":
        return np.vstack([ring(c, R, n // 2, lo), ring(c, R, n - n // 2, hi)])
    if name == "trois_cercles":
        k = n // 3
        return np.vstack([ring(c, R, k, lo), ring(c, R, k, c[2]), ring(c, R, n - 2 * k, hi)])
    if name == "sphere":
        return sphere(c, R, n)
    raise ValueError(f"disposition inconnue : {name!r}")


# ---- Regroupement des sources proches ------------------------------------------------------
def merge_close(sources, d_min: float):
    """
    Fusionne les sources à moins de d_min les unes des autres, jusqu'à ce que
    toutes soient séparées d'au moins d_min : on fusionne à chaque étape la
    paire la plus proche en une source au barycentre pondéré par les
    intensités, d'intensité la somme (l'intensité totale est conservée).
    """
    from ..model import Sources
    pos = [p for p in sources.positions]
    I = list(sources.intensities)
    while len(I) > 1:
        P = np.array(pos)
        D = np.linalg.norm(P[:, None] - P[None], axis=-1)
        D[np.diag_indices(len(I))] = np.inf
        i, j = np.unravel_index(np.argmin(D), D.shape)
        if D[i, j] >= d_min:
            break
        w = I[i] + I[j]
        pos[i] = (I[i] * pos[i] + I[j] * pos[j]) / w
        I[i] = w
        del pos[j], I[j]
    return Sources(np.array(pos).reshape(-1, sources.dim), np.array(I))


# ---- Erreurs pour un grand nombre de sources ---------------------------------------------
def match_many(true_pos, true_int, est: Estimate, tol: float) -> dict:
    """
    Appariement (hongrois) vraies / estimées ; une vraie source est
    « retrouvée » si son estimée appariée est à moins de `tol`.

    Renvoie K, n_est, nombre et proportion de sources retrouvées, sources
    parasites (estimées non appariées à moins de tol), erreur de position
    médiane des retrouvées, et erreur relative sur l'intensité totale.
    """
    K = len(true_int)
    out = {"K": K, "n_est": est.n, "retrouvees": 0, "taux_retrouvees": 0.0 if K else np.nan,
           "parasites": est.n, "err_pos_med": np.nan,
           "err_I_totale": abs(est.intensities.sum() - true_int.sum()) / true_int.sum() if K else np.nan}
    if K and est.n:
        D = np.linalg.norm(true_pos[:, None] - est.positions[None], axis=-1)
        i, j = linear_sum_assignment(D)
        good = D[i, j] < tol
        out.update(retrouvees=int(good.sum()), taux_retrouvees=float(good.mean() * len(i) / K),
                   parasites=int(est.n - good.sum()),
                   err_pos_med=float(np.median(D[i, j][good])) if good.any() else np.nan)
    return out


# ---- Figures --------------------------------------------------------------------------------
def plot_scene(sources, sensors, box: Box, estimate: Estimate | None = None, title=""):
    """Vue 3D : cube, capteurs, vraies sources (taille ~ intensité) et sources estimées."""
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*sensors.T, s=6, c="darkorange", label="capteurs")
    smax = max(sources.intensities.max(), 1e-12)
    ax.scatter(*sources.positions.T, s=20 + 200 * sources.intensities / smax, c="tab:blue",
               alpha=0.7, label="vraies sources")
    if estimate is not None and estimate.n:
        ax.scatter(*estimate.positions.T, s=40, marker="x", c="k", label="BLASSO")
    for a in range(3):                                          # arêtes du cube
        for b in (box.lo, box.hi):
            for c in (box.lo, box.hi):
                o = [i for i in range(3) if i != a]
                p0 = np.empty(3)
                p0[o[0]], p0[o[1]] = b[o[0]], c[o[1]]
                xs = np.array([p0, p0])
                xs[0, a], xs[1, a] = box.lo[a], box.hi[a]
                ax.plot(*xs.T, c="0.5", lw=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_eta_slice(kernel, p, z, box: Box, height: float, n: int = 200, title=""):
    """Coupe horizontale de eta_V à la hauteur `height`, niveau 1 en noir, sources de la coupe."""
    xs = np.linspace(box.lo[0], box.hi[0], n)
    ys = np.linspace(box.lo[1], box.hi[1], n)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X, Y, np.full_like(X, height)], axis=-1)
    vals = kernel.phi(pts) @ p
    lo, hi = min(np.percentile(vals, 1), 0.9), max(np.percentile(vals, 99), 1.1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.pcolormesh(xs, ys, np.clip(vals, lo, hi), shading="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=1.0, vmin=lo, vmax=hi))
    ax.contour(xs, ys, vals, levels=[1.0], colors="k", linewidths=1)
    near = np.abs(z[:, 2] - height) < 0.5
    ax.plot(*z[near, :2].T, "*", c="lime", mec="k", ms=13, label="sources (|z - coupe| < 0.5)")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    plt.colorbar(im, ax=ax, label=r"$\eta_V$ (écrêté)")
    fig.tight_layout()
    return fig

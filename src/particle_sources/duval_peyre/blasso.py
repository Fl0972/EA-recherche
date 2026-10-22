"""
BLASSO positif

    min_{mu >= 0}  1/2 ||m - Phi mu||^2 + lambda ||mu||_TV

résolu de deux façons :
  - `lasso_grid` : LASSO positif sur une grille fine (Diracs aux nœuds), puis
    regroupement des nœuds voisins actifs en une source chacun ;
  - `sliding_frank_wolfe` : Frank-Wolfe glissant (Denoyelle et al., 2019),
    sans grille : on ajoute un Dirac au maximum du certificat
    eta = Phi^*(m - Phi mu) / lambda, puis on réajuste intensités et positions.
Puis `debias` (moindres carrés positifs à positions fixées) et `match`
(appariement estimées / vraies pour mesurer les erreurs).

Toutes les fonctions prennent un noyau de `duval_peyre.kernel`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.optimize import linear_sum_assignment, minimize, nnls

from ..model import Box
from .certificate import grid_points, maximize_on_box

_LBFGS = {"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12}


@dataclass
class Estimate:
    """Sources estimées : positions (N, d) et intensités (N,)."""
    positions: np.ndarray
    intensities: np.ndarray

    @property
    def n(self) -> int:
        """Nombre de sources estimées."""
        return len(self.intensities)


def operator_norm(kernel, box: Box, h: float) -> float:
    """
    ||Phi|| = max_{z in D} ||phi(z)|| : norme de Phi des mesures (norme TV)
    vers R^M. Avec lambda = c ||bruit|| ||Phi||, le bruit seul donne un
    certificat |Phi^* bruit| / lambda <= 1 / c.
    """
    _, pts = grid_points(box, h)
    return float(np.max(np.linalg.norm(kernel.phi(pts.reshape(-1, box.dim)), axis=-1)))


def _positive_lasso(A, m, lam, max_iter: int = 2000):
    """
    min_{a >= 0} 1/2 ||m - A a||^2 + lam * sum(a), solution exacte.

    Algorithme à ensemble actif de Lawson-Hanson (celui de NNLS), adapté au
    terme linéaire lam * sum(a) : on active la colonne de plus forte
    corrélation A_j^T r - lam > 0, on résout le problème sans contrainte sur
    l'ensemble actif P (A_P^T A_P s = A_P^T m - lam), et on recule vers la
    solution précédente si une composante devient négative. Au plus ~M
    colonnes actives : rapide même pour une grille de 10^5 points, et robuste
    quand les colonnes sont presque colinéaires (noyau très lisse).
    """
    N = A.shape[1]
    a = np.zeros(N)
    P = np.zeros(N, bool)
    tol = 1e-12 * max(np.abs(A.T @ m).max(), lam, 1e-300)

    def solve(P):
        AP = A[:, P]
        # A_P^T A_P s = A_P^T m - lam 1  <=>  s = pinv(A_P) (m - lam pinv(A_P)^T 1)
        pinv = np.linalg.pinv(AP)
        return pinv @ (m - lam * pinv.T @ np.ones(P.sum()))

    for _ in range(max_iter):
        w = A.T @ (m - A @ a) - lam              # - gradient
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
            # Recul : plus grand pas qui garde a >= 0, puis on sort les nuls
            neg = P & (s <= 0)
            alpha = np.min(a[neg] / (a[neg] - s[neg]))
            a = a + alpha * (s - a)
            P &= a > 1e-14 * max(a.max(), 1e-300)
            a[~P] = 0.0
            if not P.any():
                break
    return a


# ---- LASSO sur grille -------------------------------------------------------------
def lasso_grid(kernel, m, lam, box: Box, h: float, rel_threshold: float = 1e-3):
    """
    LASSO positif sur la grille de pas h, puis regroupement.

    Les nœuds d'amplitude > rel_threshold * max sont regroupés par composantes
    connexes (8-voisinage en 2D, 26 en 3D) ; chaque composante devient un Dirac au barycentre
    pondéré, de masse la somme des amplitudes.

    Renvoie (Estimate, dict de la grille {axes, amplitudes}).
    """
    axes, pts = grid_points(box, h)
    flat = pts.reshape(-1, box.dim)
    a = _positive_lasso(kernel.phi(flat).T, m, lam)
    img = a.reshape(pts.shape[:-1])
    active = img > rel_threshold * img.max() if img.max() > 0 else np.zeros_like(img, bool)
    labels, n = label(active, structure=np.ones((3,) * box.dim))
    pos, amp = [], []
    for j in range(1, n + 1):
        sel = labels == j
        w = img[sel]
        pos.append(w @ pts[sel] / w.sum())
        amp.append(w.sum())
    est = Estimate(np.array(pos).reshape(-1, box.dim), np.array(amp))
    return est, {"axes": axes, "amplitudes": img}


# ---- Frank-Wolfe glissant ------------------------------------------------------------
def _objective(kernel, m, lam):
    """Fonction (a, z) -> (valeur, gradient) du BLASSO pour N Diracs (a_k, z_k)."""
    def fg(x, n):
        a, z = x[:n], x[n:].reshape(n, -1)
        P = kernel.phi(z)                         # (N, M)
        r = a @ P - m                             # (M,)
        ga = P @ r + lam
        gz = a[:, None] * np.einsum("kia,i->ka", kernel.grad(z), r)
        return 0.5 * r @ r + lam * a.sum(), np.concatenate([ga, gz.ravel()])
    return fg


def sliding_frank_wolfe(kernel, m, lam, box: Box, h: float, max_iter: int = 20,
                        tol: float = 1e-4, n_starts: int = 5):
    """
    Frank-Wolfe glissant pour le BLASSO positif.

    À chaque itération :
      1. certificat eta(z) = phi(z) . (m - Phi mu) / lambda ; maximum z* sur D
         (grille de pas h puis L-BFGS-B depuis les n_starts meilleurs pics) ;
      2. si eta(z*) <= 1 + tol : mu est optimal, on s'arrête ;
      3. on ajoute un Dirac en z*, on réajuste les intensités (LASSO positif à
         positions fixées), puis intensités ET positions ensemble (L-BFGS-B,
         bornes : a >= 0, z dans D) ;
      4. on retire les Diracs d'intensité nulle.

    Renvoie (Estimate, nombre d'itérations, max du certificat final).
    """
    _, pts = grid_points(box, h)
    grid = pts.reshape(-1, box.dim)
    Pg = kernel.phi(grid)                         # (Ng, M), réutilisé à chaque itération
    fg = _objective(kernel, m, lam)
    a, z = np.zeros(0), np.zeros((0, box.dim))
    bounds_z = list(zip(box.lo, box.hi))
    eta_max = np.inf
    for it in range(1, max_iter + 1):
        r = m - a @ kernel.phi(z) if len(a) else m.copy()
        vals = Pg @ r / lam
        img = vals.reshape(pts.shape[:-1])
        peaks = np.argwhere(img == maximum_filter(img, size=3, mode="nearest"))
        peaks = peaks[np.argsort(-img[tuple(peaks.T)])][:n_starts]
        cands = maximize_on_box(lambda x: kernel.phi(x) @ r / lam,
                                lambda x: kernel.grad(x).T @ r / lam,
                                box, [pts[tuple(i)] for i in peaks])
        eta_max, z_new = max(cands, key=lambda c: c[0])
        if eta_max <= 1 + tol:
            break
        z = np.vstack([z, z_new])
        a = _positive_lasso(kernel.phi(z).T, m, lam)
        n = len(a)
        res = minimize(fg, np.concatenate([a, z.ravel()]), args=(n,), jac=True,
                       method="L-BFGS-B", bounds=[(0, None)] * n + bounds_z * n,
                       options=_LBFGS)
        a, z = res.x[:n], res.x[n:].reshape(n, box.dim)
        keep = a > 1e-10 * max(a.max(), 1e-300)
        a, z = a[keep], z[keep]
    return Estimate(z, a), it, float(eta_max)


# ---- Débiaisage et erreurs -------------------------------------------------------------
def debias(kernel, m, est: Estimate) -> Estimate:
    """Intensités réajustées par moindres carrés positifs (positions fixées, sans pénalité)."""
    if est.n == 0:
        return est
    a, _ = nnls(kernel.phi(est.positions).T, m)
    keep = a > 0
    return Estimate(est.positions[keep], a[keep])


def match(true_pos, true_int, est: Estimate) -> dict:
    """
    Apparie sources estimées et vraies (algorithme hongrois sur les distances).

    Renvoie n_est, exact (n_est == K), l'erreur de position max et l'erreur
    relative d'intensité max sur les paires appariées (nan si aucune).
    """
    true_pos = np.atleast_2d(true_pos)
    true_int = np.asarray(true_int, float)
    out = {"n_est": est.n, "exact": est.n == len(true_int),
           "err_pos": np.nan, "err_int": np.nan}
    if est.n:
        D = np.linalg.norm(true_pos[:, None] - est.positions[None], axis=-1)
        i, j = linear_sum_assignment(D)
        out["err_pos"] = float(D[i, j].max())
        out["err_int"] = float(np.max(np.abs(est.intensities[j] - true_int[i]) / true_int[i]))
    return out

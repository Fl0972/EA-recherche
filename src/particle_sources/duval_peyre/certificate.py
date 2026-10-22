"""
Condition de source non dégénérée (Duval & Peyré, 2015), cas des mesures
positives, pour une configuration de sources z_1, ..., z_K donnée.

    Gamma = [phi(z_1) .. phi(z_K), d1 phi(z_1) .. d1 phi(z_K), d2 phi(z_1) ..]
            (M x (d+1)K : en 2D 3K colonnes, en 3D 4K)
    p_V   = argmin ||p||  s.c.  Gamma^T p = (1, .., 1, 0, .., 0)
    eta_V = Phi^* p_V,  eta_V(z) = sum_i p_V,i G(x_i, z)

La condition tient si et seulement si
    (H1) Gamma est de rang plein (d+1)K (il faut M >= (d+1)K) ;
    (H2) eta_V(z) < 1 pour z dans D hors des z_k (bords de D compris) ;
    (H3) la hessienne de eta_V en chaque z_k est définie négative.
Sous cette condition, le BLASSO positif retrouve exactement K Diracs pour un
bruit et un lambda assez petits (avec ||bruit|| / lambda borné), et l'erreur
sur les positions et les intensités est O(||bruit||).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

from ..model import Box

TOL = 1e-6  # marge minimale pour déclarer eta_V < 1 « strictement »


# ---- Briques --------------------------------------------------------------------
def gamma_matrix(kernel, z) -> np.ndarray:
    """Gamma = [phi(z_k) | d1 phi(z_k) | ... | dd phi(z_k)], forme (M, (d+1)K)."""
    z = np.atleast_2d(np.asarray(z, float))
    g = kernel.grad(z)                                   # (K, M, d)
    blocks = [kernel.phi(z).T] + [g[:, :, a].T for a in range(z.shape[1])]
    return np.hstack(blocks)


def precertificate(gamma: np.ndarray, k: int) -> np.ndarray:
    """p_V = pinv(Gamma^T) (1_K, 0_dK) : solution de norme minimale, forme (M,)."""
    u = np.zeros(gamma.shape[1])
    u[:k] = 1.0
    return np.linalg.pinv(gamma.T) @ u


def eta(kernel, p, z) -> np.ndarray:
    """eta(z) = sum_i p_i G(x_i, z), z de forme (..., d), par paquets (mémoire)."""
    z = np.asarray(z, float)
    flat = z.reshape(-1, z.shape[-1])
    out = np.concatenate([kernel.phi(c) @ p for c in np.array_split(
        flat, max(1, len(flat) * kernel.m // 2_000_000))])
    return out.reshape(z.shape[:-1])


def eta_grad(kernel, p, z) -> np.ndarray:
    """Gradient de eta en z (..., d)."""
    return np.einsum("i,...ia->...a", p, kernel.grad(z))


def eta_hess(kernel, p, z) -> np.ndarray:
    """Hessienne de eta en z (..., d, d)."""
    return np.einsum("i,...iab->...ab", p, kernel.hess(z))


def grid_points(box: Box, h: float):
    """
    Grille régulière de pas ~h couvrant D, bords compris, en dimension d.

    Renvoie (axes, points) : axes = liste des d vecteurs de coordonnées,
    points de forme (n_y, n_x, 2) en 2D (ordre d'affichage de matplotlib) et
    (n_1, ..., n_d, d) au-delà.
    """
    n = np.ceil(box.size / h).astype(int) + 1
    axes = [np.linspace(lo, hi, k) for lo, hi, k in zip(box.lo, box.hi, n)]
    mesh = np.meshgrid(*axes, indexing="xy" if box.dim == 2 else "ij")
    return axes, np.stack(mesh, axis=-1)


def box_grid(box: Box, h: float):
    """Version 2D de `grid_points` : (xs, ys, points (ny, nx, 2))."""
    (xs, ys), pts = grid_points(box, h)
    return xs, ys, pts


def _dist_to_sources(points, z):
    """Distance de chaque point (..., d) à la source la plus proche."""
    return np.min(np.linalg.norm(points[..., None, :] - z, axis=-1), axis=-1)


def maximize_on_box(f, grad, box: Box, starts):
    """
    Maximise f sur D par L-BFGS-B (bornes = D) à partir de chaque point de
    départ. Renvoie la liste (valeur, point).
    """
    bounds = list(zip(box.lo, box.hi))
    out = []
    for x0 in starts:
        res = minimize(lambda x: -f(x), x0, jac=lambda x: -grad(x),
                       method="L-BFGS-B", bounds=bounds)
        out.append((-float(res.fun), res.x))
    return out


def boundary_max(f, grad, box: Box, z, r_ball: float, h: float, n_starts: int = 3):
    """
    Maximum de f sur le bord de D (hors boules), en dimension d : sur chaque
    face (une coordonnée fixée à sa borne), grille de pas h / 4 (2D, faces =
    segments) ou h / 2 (3D), puis L-BFGS-B restreint à la face depuis les
    n_starts meilleurs points.
    """
    best = (-np.inf, np.full(box.dim, np.nan))
    step = h / 4 if box.dim == 2 else h / 2
    for a in range(box.dim):
        for side in (box.lo[a], box.hi[a]):
            lo, hi = box.lo.copy(), box.hi.copy()
            lo[a] = hi[a] = side
            free = [i for i in range(box.dim) if i != a]
            sub = Box(lo[free], hi[free])
            _, pts_free = grid_points(sub, step) if sub.dim > 1 else                 (None, np.linspace(sub.lo[0], sub.hi[0], int(np.ceil(sub.size[0] / step)) + 1)[:, None])
            pts_free = pts_free.reshape(-1, sub.dim)
            pts = np.empty((len(pts_free), box.dim))
            pts[:, free] = pts_free
            pts[:, a] = side
            vals = f(pts)
            vals[_dist_to_sources(pts, z) < r_ball] = -np.inf
            order = np.argsort(-vals)[:n_starts]
            order = order[np.isfinite(vals[order])]
            if not len(order):
                continue
            cand = [(vals[order[0]], pts[order[0]])]
            bounds = list(zip(lo, hi))
            for x0 in pts[order]:
                res = minimize(lambda x: -f(x), x0, jac=lambda x: -grad(x),
                               method="L-BFGS-B", bounds=bounds)
                if _dist_to_sources(res.x[None], z)[0] >= r_ball:
                    cand.append((-float(res.fun), res.x))
            best = max([best] + cand, key=lambda c: c[0])
    return best


# ---- Diagnostic ------------------------------------------------------------------
@dataclass
class Diagnostic:
    """Résultat de `diagnose` (voir le docstring du module)."""
    z: np.ndarray
    m: int
    # H1
    h1: bool
    rank: int
    cond: float              # conditionnement de Gamma
    cond_normalized: float   # idem, colonnes normalisées (insensible aux unités)
    singular_values: np.ndarray
    # Pré-certificat
    p: np.ndarray
    interp_error: float      # max_k |eta_V(z_k) - 1|
    grad_error: float        # max_k ||grad eta_V(z_k)||, relatif à l'échelle de grad eta
    # H2
    h2: bool
    margin: float            # 1 - max eta_V hors des boules
    argmax: np.ndarray       # point où ce max est atteint
    boundary_value: float    # max de eta_V sur le bord de D (hors boules)
    boundary_argmax: np.ndarray
    ball_radius: float
    ball_excess: float       # max(eta_V - 1) dans les boules (hors z_k) ; > 0 = pb local
    # H3
    h3: bool
    hessians: np.ndarray     # (K, d, d)
    eigenvalues: np.ndarray  # (K, d), croissantes
    # Grille (pour les figures), facultative
    grid: dict | None = field(default=None, repr=False)

    @property
    def k(self) -> int:
        """Nombre de sources K."""
        return len(self.z)

    @property
    def nondegenerate(self) -> bool:
        """Condition de source non dégénérée : H1, H2 et H3."""
        return self.h1 and self.h2 and self.h3

    def as_row(self) -> dict:
        """Ligne de tableau (CSV) résumant le diagnostic."""
        return {"K": self.k, "M": self.m, "H1": self.h1, "H2": self.h2, "H3": self.h3,
                "non_degenere": self.nondegenerate, "rang": self.rank,
                "cond": self.cond, "cond_norm": self.cond_normalized,
                "marge": self.margin, "max_bord": self.boundary_value,
                "exces_boules": self.ball_excess,
                "vp_hess_max": float(self.eigenvalues[:, -1].max()),
                "vp_hess_min": float(self.eigenvalues[:, 0].min()),
                "err_interp": self.interp_error, "err_grad": self.grad_error}

    def __str__(self) -> str:
        """Rapport lisible."""
        ok = {True: "VRAIE", False: "FAUSSE"}
        L = [f"Condition de source non dégénérée -- K = {self.k} sources, M = {self.m} capteurs"]
        n = (self.z.shape[1] + 1) * self.k
        L.append(f"  (H1) {ok[self.h1]} : rang(Gamma) = {self.rank} / {n}, "
                 f"cond = {self.cond:.3e} (colonnes normalisées : {self.cond_normalized:.3e})")
        if self.m < n:
            L.append(f"       ATTENTION : M = {self.m} < (d+1)K = {n}")
        L.append(f"  Pré-certificat : |eta_V(z_k) - 1| <= {self.interp_error:.1e}, "
                 f"|grad eta_V(z_k)| (relatif) <= {self.grad_error:.1e}")
        L.append(f"  (H2) {ok[self.h2]} : marge 1 - max eta_V = {self.margin:.4g} "
                 f"(max en {np.round(self.argmax, 3)}, boules de rayon {self.ball_radius:.3g})")
        L.append(f"       max sur le bord de D : {self.boundary_value:.4g} "
                 f"en {np.round(self.boundary_argmax, 3)}")
        if self.ball_excess > TOL:
            L.append(f"       eta_V dépasse 1 de {self.ball_excess:.3g} tout près des z_k "
                     "(z_k n'est pas un maximum local)")
        L.append(f"  (H3) {ok[self.h3]} : valeurs propres de la hessienne de eta_V")
        for zk, ev in zip(self.z, self.eigenvalues):
            L.append(f"       z = {np.round(zk, 3)} : {np.array2string(ev, precision=4)}")
        L.append("  => condition " + ("VÉRIFIÉE" if self.nondegenerate else "NON vérifiée")
                 + " : le théorème de Duval-Peyré "
                 + ("s'applique." if self.nondegenerate else "ne s'applique pas."))
        return "\n".join(L)


def diagnose(kernel, z, box: Box, h: float | None = None, ball_radius: float | None = None,
             n_refine: int = 10, keep_grid: bool = False) -> Diagnostic:
    """
    Vérifie H1, H2, H3 pour les sources z (K, 2) dans le domaine `box`.

    Paramètres
    ----------
    kernel : InverseSquareKernel (ou tout objet phi / grad / hess)
    z : array_like, forme (K, d)
        Positions des sources (dans D).
    box : Box
        Domaine D.
    h : float ou None
        Pas de la grille de H2 ; défaut min(taille de D) / 250 en 2D, / 40 en 3D.
    ball_radius : float ou None
        Rayon des boules exclues autour des z_k ; défaut
        min(0.25 * séparation minimale, 0.01 * diagonale de D).
    n_refine : int
        Nombre de maxima locaux de la grille affinés par optimisation.
    keep_grid : bool
        Garde eta_V sur la grille dans `Diagnostic.grid` (pour les figures).
    """
    z = np.atleast_2d(np.asarray(z, float))
    K, M = len(z), kernel.m
    diag_len = float(np.linalg.norm(box.size))
    if h is None:
        h = float(min(box.size)) / (250 if box.dim == 2 else 40)
    if ball_radius is None:
        sep = np.inf
        if K > 1:
            dz = np.linalg.norm(z[:, None] - z[None], axis=-1)
            sep = dz[np.triu_indices(K, 1)].min()
        ball_radius = min(0.25 * sep, 0.01 * diag_len)

    # (H1) rang et conditionnement de Gamma
    gamma = gamma_matrix(kernel, z)
    sv = np.linalg.svd(gamma, compute_uv=False)
    rank = int(np.sum(sv > max(gamma.shape) * np.finfo(float).eps * sv[0]))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    gn = gamma / np.linalg.norm(gamma, axis=0)
    svn = np.linalg.svd(gn, compute_uv=False)
    cond_n = float(svn[0] / svn[-1]) if svn[-1] > 0 else np.inf
    n_cols = gamma.shape[1]                              # (d+1) K
    h1 = (M >= n_cols) and rank == n_cols

    # Pré-certificat et contrôle des contraintes
    p = precertificate(gamma, K)
    f = lambda x: eta(kernel, p, x)
    grad = lambda x: eta_grad(kernel, p, x)
    interp_error = float(np.max(np.abs(f(z) - 1)))
    grad_scale = np.max(np.linalg.norm(np.einsum("i,kia->kia", np.abs(p), kernel.grad(z)),
                                       axis=-1).sum(axis=-1))
    grad_error = float(np.max(np.linalg.norm(grad(z), axis=-1)) / grad_scale)

    # (H2) grille fine hors des boules, puis affinement local
    axes, pts = grid_points(box, h)
    vals = f(pts)
    dist = _dist_to_sources(pts, z)
    out = np.where(dist >= ball_radius, vals, -np.inf)
    i = np.unravel_index(np.argmax(out), out.shape)
    best_val, best_pt = float(out[i]), pts[i]
    # Départs : les n_refine meilleurs maxima locaux de la grille
    peaks = (out == maximum_filter(out, size=3, mode="nearest")) & np.isfinite(out)
    idx = np.argwhere(peaks)
    idx = idx[np.argsort(-out[peaks])][:n_refine]
    for v, x in maximize_on_box(f, grad, box, [pts[tuple(i)] for i in idx]):
        if _dist_to_sources(x[None], z)[0] >= ball_radius and v > best_val:
            best_val, best_pt = v, x
    b_val, b_pt = boundary_max(f, grad, box, z, ball_radius, h)
    if b_val > best_val:
        best_val, best_pt = b_val, b_pt
    margin = 1.0 - best_val
    inside = (dist < ball_radius) & (dist > 1e-12)
    ball_excess = float(np.max(vals[inside] - 1)) if inside.any() else -np.inf
    h2 = margin > TOL

    # (H3) hessiennes en chaque z_k : définies négatives <=> toutes les valeurs
    # propres < 0 (en 2D : trace < 0 et déterminant > 0)
    hessians = eta_hess(kernel, p, z)
    eig = np.linalg.eigvalsh(hessians)
    h3 = bool(np.all(eig < 0))
    if h3 and ball_excess > TOL:
        # Hessienne négative mais eta_V > 1 dans une boule : pas de max strict
        # dans toute la boule -> on considère que H2 échoue.
        h2 = False

    grid = None
    if keep_grid:
        grid = {"axes": axes, "eta": vals}
        if box.dim == 2:
            grid.update(xs=axes[0], ys=axes[1])
    return Diagnostic(z=z, m=M, h1=h1, rank=rank, cond=cond, cond_normalized=cond_n,
                      singular_values=sv, p=p, interp_error=interp_error,
                      grad_error=grad_error, h2=h2, margin=margin, argmax=best_pt,
                      boundary_value=b_val, boundary_argmax=b_pt, ball_radius=ball_radius,
                      ball_excess=ball_excess, h3=h3, hessians=hessians, eigenvalues=eig,
                      grid=grid)

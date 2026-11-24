"""
Noyau du modèle direct, géométrie (domaine, capteurs) et grille de recherche.

Modèle : un capteur placé en x_i mesure, pour une source d'intensité I en z,

    I · G(x_i, z),      G(x_i, z) = 1 / ||x_i − z||².

    φ(z)      = (G(x_1, z), ..., G(x_M, z))            ∈ R^M
    ∇_z φ(z)  : ligne i = −2 (z − x_i) / q_i²          q_i = ||z − x_i||²
    ∇²_z φ(z) : bloc i  = −2 Id / q_i² + 8 (z − x_i)(z − x_i)ᵀ / q_i³

Un ``Setup`` regroupe ce qui définit l'opérateur : capteurs, domaine D, et
deux options utilisées par ML + BIC et le BLASSO :
- ``normalize`` : noyau normalisé ψ = φ / ||φ|| (intensités « normalisées »
  Ĩ = I ||φ|| ; ``physical_intensities`` / ``normalized_intensities`` convertissent) ;
- ``weights`` : capteurs blanchis φ_w = w ⊙ φ (w_i = 1 / σ_i), pour que
  l'attache aux données soit la log-vraisemblance gaussienne.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


# ---- domaine ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    """Rectangle [lo_1, hi_1] × [lo_2, hi_2] dans lequel sont les sources."""
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "lo", np.asarray(self.lo, float))
        object.__setattr__(self, "hi", np.asarray(self.hi, float))

    @property
    def dim(self) -> int:
        return len(self.lo)

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def circumradius(self) -> float:
        """Rayon du cercle circonscrit."""
        return 0.5 * float(np.linalg.norm(self.size))


# ---- noyaux ----------------------------------------------------------------------------
@dataclass(frozen=True)
class InverseSquareKernel:
    """φ(z) = (1 / (||x_i − z||² + eps²))_i et ses dérivées, pour M capteurs fixés."""
    sensors: np.ndarray
    eps: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "sensors", np.atleast_2d(np.asarray(self.sensors, float)))

    @property
    def m(self) -> int:
        return len(self.sensors)

    def _dq(self, z):
        z = np.asarray(z, dtype=float)
        d = z[..., None, :] - self.sensors
        return d, np.sum(d**2, axis=-1) + self.eps**2

    def phi(self, z) -> np.ndarray:
        """φ(z), forme (..., M)."""
        _, q = self._dq(z)
        return 1.0 / q

    def grad(self, z) -> np.ndarray:
        """∇φ(z), forme (..., M, 2)."""
        d, q = self._dq(z)
        return -2.0 * d / q[..., None] ** 2

    def hess(self, z) -> np.ndarray:
        """∇²φ(z), forme (..., M, 2, 2)."""
        d, q = self._dq(z)
        eye = np.eye(d.shape[-1])
        return (-2.0 / q[..., None, None] ** 2 * eye
                + 8.0 / q[..., None, None] ** 3 * d[..., :, None] * d[..., None, :])


@dataclass(frozen=True)
class WeightedKernel:
    """
    Noyau pondéré ψ(z) = φ(z) / ||φ(z)||^α (α = 1 : noyau normalisé).

    Dérivées (J = ∇φ, S = ||φ||², n = √S, u = n^{-α}) :
        ∇n = Jᵀφ / n,   ∇u = −α n^{-α-1} ∇n,   ∇ψ_i = u J_i + φ_i ∇u
    """
    base: object
    alpha: float = 1.0

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def sensors(self) -> np.ndarray:
        return self.base.sensors

    def phi(self, z) -> np.ndarray:
        p = self.base.phi(z)
        return p * np.linalg.norm(p, axis=-1, keepdims=True) ** (-self.alpha)

    def _parts(self, z):
        a = self.alpha
        p, J = self.base.phi(z), self.base.grad(z)
        n = np.linalg.norm(p, axis=-1)
        grad_n = np.einsum("...ia,...i->...a", J, p) / n[..., None]
        u = n ** (-a)
        grad_u = -a * (n ** (-a - 1))[..., None] * grad_n
        return p, J, n, u, grad_n, grad_u

    def grad(self, z) -> np.ndarray:
        p, J, _, u, _, grad_u = self._parts(z)
        return u[..., None, None] * J + p[..., None] * grad_u[..., None, :]

    def hess(self, z) -> np.ndarray:
        a = self.alpha
        p, J, n, u, grad_n, grad_u = self._parts(z)
        H = self.base.hess(z)
        n2 = n[..., None, None]
        hess_S = 2 * (np.einsum("...ia,...ib->...ab", J, J)
                      + np.einsum("...i,...iab->...ab", p, H))
        grad_S = 2 * n[..., None] * grad_n
        hess_n = (hess_S / (2 * n2)
                  - grad_S[..., :, None] * grad_S[..., None, :] / (4 * n2**3))
        nn = grad_n[..., :, None] * grad_n[..., None, :]
        hess_u = -a * n2 ** (-a - 1) * hess_n + a * (a + 1) * n2 ** (-a - 2) * nn
        return (u[..., None, None, None] * H
                + J[..., :, None] * grad_u[..., None, None, :]
                + grad_u[..., None, :, None] * J[..., None, :]
                + p[..., None, None] * hess_u[..., None, :, :])


@dataclass(frozen=True, eq=False)
class WhitenedKernel:
    """Noyau blanchi φ_w(z) = (w_i G(x_i, z))_i ; dérivées multipliées ligne par ligne par w."""
    base: object
    w: np.ndarray

    @property
    def m(self) -> int:
        return self.base.m

    @property
    def sensors(self) -> np.ndarray:
        return self.base.sensors

    def phi(self, z) -> np.ndarray:
        return self.base.phi(z) * self.w

    def grad(self, z) -> np.ndarray:
        return self.base.grad(z) * self.w[:, None]

    def hess(self, z) -> np.ndarray:
        return self.base.hess(z) * self.w[:, None, None]


# ---- Setup -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Setup:
    """
    Opérateur direct et domaine.

    sensors   (M, 2) positions des capteurs
    domain    Box, domaine des sources
    h         hauteur des capteurs au-dessus du plan des sources (0 ici)
    normalize noyau normalisé ψ = φ / ||φ|| (α = 1)
    alpha     exposant de la pondération ψ = φ / ||φ||^α (prioritaire sur normalize)
    weights   (M,) poids de blanchiment w_i (mesures à passer : ``whiten(m)``)
    """
    sensors: np.ndarray
    domain: Box
    h: float = 0.0
    normalize: bool = False
    alpha: float | None = None
    weights: np.ndarray | None = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "sensors", np.atleast_2d(np.asarray(self.sensors, float)))
        if self.weights is not None:
            w = np.asarray(self.weights, float).reshape(-1)
            if w.shape != (len(self.sensors),) or np.any(w <= 0):
                raise ValueError("weights : M poids strictement positifs attendus")
            object.__setattr__(self, "weights", w)

    @cached_property
    def base_kernel(self):
        k = InverseSquareKernel(self.sensors, eps=self.h)
        return WhitenedKernel(k, self.weights) if self.weights is not None else k

    @property
    def power(self) -> float:
        if self.alpha is not None:
            return float(self.alpha)
        return 1.0 if self.normalize else 0.0

    @cached_property
    def kernel(self):
        return WeightedKernel(self.base_kernel, self.power) if self.power else self.base_kernel

    def with_weights(self, weights) -> "Setup":
        """Même géométrie et pondération, capteurs blanchis par ``weights``."""
        return Setup(self.sensors, self.domain, self.h, self.normalize, self.alpha, weights)

    def whiten(self, m) -> np.ndarray:
        m = np.asarray(m, float)
        return m * self.weights if self.weights is not None else m

    def _weights(self, z) -> np.ndarray:
        n = np.linalg.norm(self.base_kernel.phi(np.asarray(z, float).reshape(-1, 2)), axis=-1)
        return n ** self.power

    def physical_intensities(self, z, I) -> np.ndarray:
        """I_k = Ĩ_k / ||φ(z_k)||^α."""
        I = np.asarray(I, float)
        if not self.power or len(I) == 0:
            return I
        return I / self._weights(z)

    def normalized_intensities(self, z, I) -> np.ndarray:
        """Ĩ_k = I_k ||φ(z_k)||^α."""
        I = np.asarray(I, float)
        if not self.power or len(I) == 0:
            return I
        return I * self._weights(z)

    @property
    def M(self) -> int:
        return len(self.sensors)

    def phi(self, z) -> np.ndarray:
        return self.kernel.phi(z)

    def grad(self, z) -> np.ndarray:
        return self.kernel.grad(z)

    def hess(self, z) -> np.ndarray:
        return self.kernel.hess(z)


# ---- grille ----------------------------------------------------------------------------
@dataclass
class Grid:
    """
    Grille régulière de D (bords compris) et matrice G (M × P) : la colonne j est φ(y_j).
    points a la forme (n_y, n_x, 2) (ordre d'une image).
    """
    xs: np.ndarray
    ys: np.ndarray
    points: np.ndarray
    step: float
    G: np.ndarray

    @property
    def shape(self) -> tuple:
        return self.points.shape[:2]

    @property
    def flat(self) -> np.ndarray:
        return self.points.reshape(-1, 2)

    @property
    def P(self) -> int:
        return self.flat.shape[0]


def make_grid(domain, step: float | None, setup: Setup) -> Grid:
    """Grille de pas ≤ step (défaut : min(taille de D) / 100) et matrice G."""
    box = domain if domain is not None else setup.domain
    if step is None:
        step = float(min(box.size)) / 100
    n = np.ceil(box.size / step).astype(int) + 1
    xs = np.linspace(box.lo[0], box.hi[0], n[0])
    ys = np.linspace(box.lo[1], box.hi[1], n[1])
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)
    G = setup.phi(points.reshape(-1, 2)).T
    return Grid(xs, ys, points, float(step), G)


def grille_recherche(setup: Setup, step: float = 0.25) -> Grid:
    """
    Grille de recherche de ML + BIC et du BLASSO, de pas ≈ 0,25, dont aucun nœud ne
    tombe sur un capteur (le signal y serait infini) : on réduit un peu le pas si besoin.
    """
    for k in range(20):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = make_grid(None, step * (1 - 0.03 * k), setup)
        pts = g.points.reshape(-1, 2)
        d = np.min(np.linalg.norm(pts[:, None] - setup.sensors[None], axis=-1))
        if setup.h > 0 or d > 1e-6:
            return g
    raise ValueError("impossible de construire une grille qui évite les capteurs")

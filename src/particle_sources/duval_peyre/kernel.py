"""
Noyau G(x_i, z) du modèle direct, vu comme fonction de la position z de la
source, avec ses dérivées analytiques.

Le noyau est celui de `model.field.intensity_at` (rien de nouveau) :

    G(x, z) = 1 / (||x - z||^2 + eps^2)

et phi(z) = (G(x_1, z), ..., G(x_M, z)) est la colonne de l'opérateur pour une
source unité en z. En notant d = z - x et q = ||d||^2 + eps^2 :

    grad_z G = -2 d / q^2
    hess_z G = -2 I / q^2 + 8 d d^T / q^3

En 2D, Laplacien : tr(hess) = 4 (||d||^2 - eps^2) / q^3, positif dès que la
source est à plus de eps du capteur : G(x, .) est sous-harmonique sur D.
"""

from dataclasses import dataclass

import numpy as np

from ..model import Sources, intensity_at


@dataclass(frozen=True)
class InverseSquareKernel:
    """
    phi(z) et ses dérivées pour M capteurs fixés.

    Attributs
    ---------
    sensors : np.ndarray, forme (M, d)
        Positions x_1, ..., x_M des capteurs.
    eps : float
        Régularisation de 1/r^2 (même sens que dans `intensity_at`).
    """
    sensors: np.ndarray
    eps: float = 0.0

    def __post_init__(self):
        """Convertit les positions des capteurs en tableau float (M, d)."""
        object.__setattr__(self, "sensors", np.atleast_2d(np.asarray(self.sensors, float)))

    @property
    def m(self) -> int:
        """Nombre de capteurs M."""
        return len(self.sensors)

    def _dq(self, z):
        """d = z - x_i, forme (..., M, d), et q = ||d||^2 + eps^2, forme (..., M)."""
        z = np.asarray(z, dtype=float)
        d = z[..., None, :] - self.sensors
        return d, np.sum(d**2, axis=-1) + self.eps**2

    def phi(self, z) -> np.ndarray:
        """phi(z), forme (..., M) pour z de forme (..., d)."""
        _, q = self._dq(z)
        return 1.0 / q

    def grad(self, z) -> np.ndarray:
        """Gradient en z de chaque G(x_i, z), forme (..., M, d)."""
        d, q = self._dq(z)
        return -2.0 * d / q[..., None] ** 2

    def hess(self, z) -> np.ndarray:
        """Hessienne en z de chaque G(x_i, z), forme (..., M, d, d)."""
        d, q = self._dq(z)
        eye = np.eye(d.shape[-1])
        return (-2.0 / q[..., None, None] ** 2 * eye
                + 8.0 / q[..., None, None] ** 3 * d[..., :, None] * d[..., None, :])

    def forward(self, positions, intensities) -> np.ndarray:
        """
        Mesures exactes A(s) = sum_k I_k phi(z_k), forme (M,).
        Passe par `intensity_at` pour garantir que c'est bien le modèle du projet.
        """
        return intensity_at(Sources(positions, intensities), self.sensors, self.eps)


@dataclass(frozen=True)
class GaussianKernel:
    """
    Noyau gaussien G(x, z) = exp(-||x - z||^2 / (2 s^2)).

    N'est PAS le modèle du projet : il ne sert qu'aux tests, comme cas témoin
    où la condition de source non dégénérée est connue pour tenir (sources
    séparées, capteurs denses), afin de vérifier que le code sait répondre "oui".
    """
    sensors: np.ndarray
    s: float = 1.0

    def __post_init__(self):
        """Convertit les positions des capteurs en tableau float (M, d)."""
        object.__setattr__(self, "sensors", np.atleast_2d(np.asarray(self.sensors, float)))

    @property
    def m(self) -> int:
        """Nombre de capteurs M."""
        return len(self.sensors)

    def phi(self, z):
        """phi(z), forme (..., M)."""
        d = np.asarray(z, float)[..., None, :] - self.sensors
        return np.exp(-np.sum(d**2, axis=-1) / (2 * self.s**2))

    def grad(self, z):
        """Gradient en z, forme (..., M, d)."""
        d = np.asarray(z, float)[..., None, :] - self.sensors
        return -d / self.s**2 * self.phi(z)[..., None]

    def hess(self, z):
        """Hessienne en z, forme (..., M, d, d)."""
        d = np.asarray(z, float)[..., None, :] - self.sensors
        eye = np.eye(d.shape[-1])
        return self.phi(z)[..., None, None] * (d[..., :, None] * d[..., None, :] / self.s**4
                                               - eye / self.s**2)

    def forward(self, positions, intensities):
        """Mesures exactes sum_k I_k phi(z_k), forme (M,)."""
        return np.asarray(intensities, float) @ self.phi(np.asarray(positions, float))


def check_derivatives(kernel, z, h: float = 1e-5) -> dict:
    """
    Compare grad et hess analytiques à des différences finies centrées en z.

    Renvoie les erreurs relatives max (sans dimension) ; de l'ordre de 1e-8
    ou moins attendu pour h = 1e-5. Sert de garde-fou (tests et diagnostic).
    """
    z = np.asarray(z, float)
    dim = len(z)
    g_fd = np.empty((kernel.m, dim))
    h_fd = np.empty((kernel.m, dim, dim))
    for a in range(dim):
        e = np.zeros(dim)
        e[a] = h
        g_fd[:, a] = (kernel.phi(z + e) - kernel.phi(z - e)) / (2 * h)
        h_fd[:, :, a] = (kernel.grad(z + e) - kernel.grad(z - e)) / (2 * h)
    g, H = kernel.grad(z), kernel.hess(z)
    return {
        "grad": float(np.max(np.abs(g - g_fd)) / np.max(np.abs(g))),
        "hess": float(np.max(np.abs(H - h_fd)) / np.max(np.abs(H))),
    }

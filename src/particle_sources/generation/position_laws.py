"""
Lois des positions des sources.

Une loi de positions est une fonction  (n, box, rng) -> tableau (n, d).
"""

from typing import Callable

import numpy as np

from ..model import Box

PositionLaw = Callable[[int, Box, np.random.Generator], np.ndarray]


def uniform_in_box(n: int, box: Box, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(box.lo, box.hi, size=(n, box.dim))


def disk_cluster(
    radius: float = 1.0,
    center: list[float] | tuple[float, ...] | None = None,
) -> PositionLaw:
    """Tire n sources uniformément dans une boule (disque en 2D) de rayon `radius` centrée sur `center`
    (défaut : centre du domaine)."""
    if radius <= 0:
        raise ValueError(f"radius doit être strictement positif, reçu {radius}")

    def sample(n: int, box: Box, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.empty((0, box.dim))
        c = np.asarray(center if center is not None else box.center, dtype=float)
        if len(c) != box.dim:
            raise ValueError(f"dimension du centre ({len(c)}) != dimension du domaine ({box.dim})")

        if box.dim == 2:
            r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n))
            theta = rng.uniform(0.0, 2 * np.pi, size=n)
            pts = c + np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        else:
            dirs = rng.normal(size=(n, box.dim))
            norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            dirs = dirs / norms
            r = radius * (rng.uniform(0.0, 1.0, size=(n, 1)) ** (1.0 / box.dim))
            pts = c + dirs * r

        return np.clip(pts, box.lo, box.hi)

    return sample


def gaussian_cluster(
    std: float = 1.0,
    center: list[float] | tuple[float, ...] | None = None,
) -> PositionLaw:
    """Tire n sources selon une loi normale centrée sur `center` (défaut : centre du domaine)
    avec écart-type `std`."""
    if std <= 0:
        raise ValueError(f"std doit être strictement positif, reçu {std}")

    def sample(n: int, box: Box, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.empty((0, box.dim))
        c = np.asarray(center if center is not None else box.center, dtype=float)
        if len(c) != box.dim:
            raise ValueError(f"dimension du centre ({len(c)}) != dimension du domaine ({box.dim})")

        pts = rng.normal(loc=c, scale=std, size=(n, box.dim))
        return np.clip(pts, box.lo, box.hi)

    return sample


def clustered_center(
    radius: float = 1.0,
    cluster_ratio: float = 0.8,
    std: float | None = None,
    center: list[float] | tuple[float, ...] | None = None,
) -> PositionLaw:
    """
    Tire une proportion `cluster_ratio` des sources dans un cluster central
    (disque de rayon `radius`, ou gaussienne si `std` est renseigné),
    et le reste uniformément dans tout le domaine.
    """
    if radius <= 0:
        raise ValueError(f"radius doit être strictement positif, reçu {radius}")
    if std is not None and std <= 0:
        raise ValueError(f"std doit être strictement positif, reçu {std}")
    if not (0.0 <= cluster_ratio <= 1.0):
        raise ValueError(f"cluster_ratio doit être compris entre 0 et 1, reçu {cluster_ratio}")

    def sample(n: int, box: Box, rng: np.random.Generator) -> np.ndarray:
        if n == 0:
            return np.empty((0, box.dim))
        c = np.asarray(center if center is not None else box.center, dtype=float)
        if len(c) != box.dim:
            raise ValueError(f"dimension du centre ({len(c)}) != dimension du domaine ({box.dim})")

        n_cluster = int(np.round(n * cluster_ratio))
        n_cluster = max(0, min(n, n_cluster))
        n_bg = n - n_cluster

        parts = []
        if n_cluster > 0:
            if std is not None:
                cluster_pts = rng.normal(loc=c, scale=std, size=(n_cluster, box.dim))
            elif box.dim == 2:
                r = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_cluster))
                theta = rng.uniform(0.0, 2 * np.pi, size=n_cluster)
                cluster_pts = c + np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            else:
                dirs = rng.normal(size=(n_cluster, box.dim))
                norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                dirs = dirs / norms
                r = radius * (rng.uniform(0.0, 1.0, size=(n_cluster, 1)) ** (1.0 / box.dim))
                cluster_pts = c + dirs * r
            parts.append(np.clip(cluster_pts, box.lo, box.hi))

        if n_bg > 0:
            bg_pts = rng.uniform(box.lo, box.hi, size=(n_bg, box.dim))
            parts.append(bg_pts)

        return np.vstack(parts)

    return sample


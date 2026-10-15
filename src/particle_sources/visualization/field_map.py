"""Cartes 2D : intensité Phi, sources, domaine."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from ..model import Box, Sources, intensity_at


def plot_intensity_map(sources: Sources, box: Box, ax, eps: float = 0.05,
                       resolution: int = 400):
    """Heatmap de Phi sur `box` (échelle log). Renvoie le mappable ou None."""
    X, Y = np.meshgrid(np.linspace(box.lo[0], box.hi[0], resolution),
                       np.linspace(box.lo[1], box.hi[1], resolution))
    Z = intensity_at(sources, np.stack([X, Y], axis=-1), eps)
    finite = Z[np.isfinite(Z) & (Z > 0)]
    if not finite.size:
        return None
    norm = LogNorm(vmin=finite.min(), vmax=np.percentile(finite, 99.5))
    im = ax.pcolormesh(X, Y, Z, norm=norm, shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=r"$\Phi(x)$")
    ax.set_aspect("equal")
    return im


def plot_sources(sources: Sources, ax, max_size: float = 200, **style):
    """Sources en points, taille proportionnelle à l'intensité."""
    if sources.n == 0:
        return
    style = {"c": "red", "edgecolors": "white", "label": "sources", **style}
    sizes = max_size * sources.intensities / sources.intensities.max()
    ax.scatter(*sources.positions.T, s=sizes, **style)


def plot_box(box: Box, ax, **style):
    style = {"fill": False, "ec": "white", "ls": "--", "lw": 1, **style}
    ax.add_patch(plt.Rectangle(box.lo, *box.size, **style))

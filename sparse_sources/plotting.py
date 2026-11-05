"""
Style commun des figures (matplotlib, PNG).

Couleurs : palette catégorielle à ordre fixe (bleu, orange, aqua, ...),
palette divergente bleu ↔ rouge avec milieu gris neutre, centrée sur η = 1
pour les cartes de certificat ; axes et grilles discrets.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]
INK = "#0b0b0b"
INK_MUTED = "#6b6a63"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list(
    "bleu_rouge", ["#1c4f8f", "#2a78d6", "#a9c8ee", "#f0efec", "#f2b0a8", "#e34948", "#9e2423"])


def style_axes(ax, grid: bool = True):
    """Axes discrets : bordures grises, grille fine en arrière-plan."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if grid:
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    return ax


def place_legend(ax, ncol: int = 4):
    """Légende sous le tracé (ne masque ni capteurs ni sources)."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=ncol, fontsize=8,
              frameon=False)


def certificate_map(ax, xs, ys, values, sensors, sources, domain, title=None,
                    estimates=None, level: float = 1.0):
    """
    Carte d'un certificat η sur D (palette divergente centrée sur ``level``),
    contour η = level, capteurs, sources (et sources estimées facultatives).
    """
    v = np.asarray(values, float)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    # Échelle symétrique autour de `level`, bornée pour garder la partie
    # < level lisible ; les valeurs au-delà saturent (flèche de la barre).
    span = max(level - lo, 0.05 * abs(level), 1e-9)
    vmin, vmax = level - span, min(max(hi, level + 1e-9), level + span)
    norm = TwoSlopeNorm(vcenter=level, vmin=vmin, vmax=vmax)
    extend = "max" if hi > vmax else "neither"
    im = ax.pcolormesh(xs, ys, v, cmap=DIVERGING, norm=norm, shading="auto", rasterized=True)
    ax.contour(xs, ys, v, levels=[level], colors=INK, linewidths=1.2)
    ax.plot(*np.c_[domain.lo, [domain.hi[0], domain.lo[1]], domain.hi,
                   [domain.lo[0], domain.hi[1]], domain.lo], color=INK_MUTED, lw=1)
    ax.scatter(*np.asarray(sensors).T, marker="^", s=28, color=INK_MUTED, label="capteurs",
               zorder=3)
    ax.scatter(*np.atleast_2d(sources).T, marker="x", s=70, linewidths=2, color=INK,
               label="sources", zorder=4)
    if estimates is not None and len(estimates):
        ax.scatter(*np.atleast_2d(estimates).T, marker="o", s=90, facecolors="none",
                   edgecolors=SERIES[1], linewidths=2, label="estimées", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    style_axes(ax, grid=False)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, extend=extend)
    cb.set_label(f"η (contour noir : η = {level:g})", color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK_MUTED)
    cb.ax.set_ylim(max(lo, vmin), vmax)    # n'affiche que la plage utilisée
    if hi > vmax:
        cb.ax.set_title(f"max {hi:.3g}", fontsize=8, color=INK_MUTED)
    place_legend(ax)
    return im

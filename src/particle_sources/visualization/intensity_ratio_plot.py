"""
Figures du cas test à deux sources (rapport d'intensité x).

Chaque fonction `plot_*` trace un panneau dans un axe donné ;
`plot_two_sources_study` les assemble en une figure de synthèse.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..model import Box, Sources, intensity_at
from .field_map import plot_box, plot_intensity_map
from .sensors_plot import plot_circle_sensors

GD_LABEL = "Descente de gradient"
METHOD_STYLES = {
    GD_LABEL: {"color": "#C0392B", "marker": "o"},
    "Gauss-Newton (même init.)": {"color": "#2E86C1", "marker": "s"},
    "Prony + max. vraisemblance": {"color": "#27AE60", "marker": "^"},
}
STRONG_COLOR, FAINT_COLOR = "white", "orange"


def _by_method(rows, key):
    out = {}
    for r in rows:
        xs, values = out.setdefault(r["method"], ([], []))
        xs.append(r["ratio"])
        values.append(r[key])
    return out


def plot_setup(ax, true: Sources, init: Sources, box: Box, sensors):
    """Schéma du cas test : capteurs, domaine, sources, initialisation."""
    plot_circle_sensors(sensors, ax)
    plot_box(box, ax, ec="gray")
    for k, (name, size, dx) in enumerate([("source forte (x I)", 320, 16),
                                          ("source faible (I)", 90, 10)]):
        ax.scatter(*true.positions[k], s=size, c="red", edgecolors="black", zorder=3)
        ax.annotate(name, true.positions[k], xytext=(dx, -4), textcoords="offset points")
    ax.plot(*init.positions.T, "x", c="black", ms=9, mew=2, label="initialisation")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_descent_path(ax, true: Sources, descent, box: Box, title: str):
    """Carte de Phi (vraies sources), trajectoires de la descente, sources finales."""
    plot_intensity_map(true, box, ax, eps=0.05, resolution=250)
    plot_box(box, ax)
    path = descent.trajectory                      # (n_records, 2, 2)
    for k, color in enumerate([STRONG_COLOR, FAINT_COLOR]):
        ax.plot(*path[:, k].T, color=color, lw=1.2)
        ax.plot(*path[0, k], "x", color=color, ms=8, mew=2)
        ax.plot(*path[-1, k], "o", color=color, ms=7, mfc="none", mew=2)
        ax.plot(*true.positions[k], "*", color="red", ms=14, mec="white",
                label="vraies sources" if k == 0 else None)
    ax.plot([], [], "x", color="gray", label="initialisation")
    ax.plot([], [], "o", color="gray", mfc="none", label="fin de la descente")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)
    ax.set_xlim(box.lo[0], box.hi[0])
    ax.set_ylim(box.lo[1], box.hi[1])


def plot_loss_landscape(ax, true: Sources, box: Box, sensors, resolution: int = 120):
    """
    Perte relative en fonction de la position supposée de la source faible
    (source forte et intensités fixées à leurs vraies valeurs) : plus la vallée
    est plate, moins le gradient guide la source faible.
    """
    X, Y = np.meshgrid(np.linspace(box.lo[0], box.hi[0], resolution),
                       np.linspace(box.lo[1], box.hi[1], 2 * resolution))
    y = intensity_at(true, sensors.positions)
    strong = Sources(true.positions[:1], true.intensities[:1])
    rest = y - intensity_at(strong, sensors.positions)            # part de la source faible
    diff = sensors.positions[None, None] - np.stack([X, Y], -1)[..., None, :]
    faint = true.intensities[1] / np.sum(diff**2, axis=-1)        # (ny, nx, K)
    loss = 0.5 * np.sum((faint - rest) ** 2, axis=-1) / (y @ y)
    im = ax.pcolormesh(X, Y, np.maximum(loss, 1e-12), shading="auto", cmap="magma_r",
                       norm=plt.matplotlib.colors.LogNorm(vmin=1e-9, vmax=1e-2))
    ax.plot(*true.positions[0], "*", c="white", ms=14, mec="black")
    ax.plot(*true.positions[1], "*", c="orange", ms=14, mec="black")
    ax.set_aspect("equal")
    return im


def plot_losses(ax, descents: dict):
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(descents)))
    for color, (x, d) in zip(colors, descents.items()):
        ax.loglog(np.arange(1, len(d.losses) + 1), d.losses, color=color, label=f"x = {x:g}")
    ax.set_xlabel("itération")
    ax.set_ylabel("perte relative")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_faint_error(ax, fixed_rows, success_tol: float):
    for label, (xs, errs) in _by_method(fixed_rows, "err_faint").items():
        errs = np.where(np.isfinite(errs), errs, np.nan)
        ax.loglog(xs, np.maximum(errs, 1e-16), label=label, **METHOD_STYLES.get(label, {}))
    ax.axhline(success_tol, color="gray", ls="--", lw=1, label="seuil de réussite")
    ax.set_xlabel("rapport d'intensité x")
    ax.set_ylabel("erreur de position, source faible")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_success(ax, random_rows):
    for label, (xs, rates) in _by_method(random_rows, "success_rate").items():
        ax.semilogx(xs, 100 * np.array(rates), label=label, **METHOD_STYLES.get(label, {}))
    ax.set_ylim(-5, 105)
    ax.set_xlabel("rapport d'intensité x")
    ax.set_ylabel("réussite (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_conditioning(ax, fixed_rows):
    rows = [r for r in fixed_rows if r["method"] == GD_LABEL]
    xs = np.array([r["ratio"] for r in rows])
    cond = np.array([r["condition"] for r in rows])
    ax.loglog(xs, cond, "o-", color="black", label="conditionnement de la perte")
    ax.loglog(xs, cond[-1] * (xs / xs[-1]) ** 2, ":", color="gray", label=r"pente $x^2$")
    ax.set_xlabel("rapport d'intensité x")
    ax.set_ylabel(r"$\lambda_{max} / \lambda_{min}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def descent_title(fixed_rows, descents, x) -> str:
    success = {r["ratio"]: r["success"] for r in fixed_rows if r["method"] == GD_LABEL}
    status = "sources retrouvées" if success[x] else "source faible manquée"
    return f"x = {x:g} : {status} ({descents[x].n_iter} it.)"


def plot_two_sources_study(fixed_rows, descents, truths, random_rows, box: Box,
                           show_ratios, success_tol: float):
    """
    Haut : trajectoires de la descente pour `show_ratios`, et courbes de perte.
    Bas : erreur sur la source faible (cas fixe), taux de réussite (essais
    aléatoires), conditionnement de la perte.
    """
    n_maps = len(show_ratios)
    fig = plt.figure(figsize=(5 * max(n_maps + 1, 3), 10))
    grid = fig.add_gridspec(2, max(n_maps + 1, 3))

    for k, x in enumerate(show_ratios):
        plot_descent_path(fig.add_subplot(grid[0, k]), truths[x], descents[x], box,
                          "Descente de gradient, " + descent_title(fixed_rows, descents, x))

    ax = fig.add_subplot(grid[0, n_maps])
    plot_losses(ax, descents)
    ax.set_title("Descente de gradient : perte", fontsize=10)

    ax = fig.add_subplot(grid[1, 0])
    plot_faint_error(ax, fixed_rows, success_tol)
    ax.set_title("Cas fixe", fontsize=10)

    ax = fig.add_subplot(grid[1, 1])
    if random_rows:
        plot_success(ax, random_rows)
        ax.set_title(f"Positions et initialisation aléatoires "
                     f"({random_rows[0]['n_trials']} essais par x)", fontsize=10)
    else:
        ax.text(0.5, 0.5, "essais aléatoires non calculés", ha="center", va="center",
                transform=ax.transAxes, color="gray")

    ax = fig.add_subplot(grid[1, 2])
    plot_conditioning(ax, fixed_rows)
    ax.set_title("Pourquoi la descente ralentit (cas fixe)", fontsize=10)

    fig.tight_layout()
    return fig

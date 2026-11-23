"""Figures du cas test à deux sources avec bruit."""

import matplotlib.pyplot as plt
import numpy as np

from .intensity_ratio_plot import GD_LABEL, METHOD_STYLES

PRONY_LABEL = "Prony + max. vraisemblance"
GN_LABEL = "Gauss-Newton (même init.)"


def _rows(rows, **filters):
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]


def plot_faint_error(ax, rows, success_tol: float):
    """Erreur sur la source faible (Prony + MV) et CRB, pour chaque x ; descente."""
    ratios = sorted({r["ratio"] for r in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(ratios)))
    for color, x in zip(colors, ratios):
        sel = _rows(rows, ratio=x, method=PRONY_LABEL)
        s = [r["sigma"] for r in sel]
        ax.loglog(s, [r["err_faint"] for r in sel], "o-", color=color,
                  label=f"Prony + MV, x = {x:g}")
        ax.loglog(s, [r["crb_faint"] for r in sel], "--", color=color, lw=1)
    gd = _rows(rows, method=GD_LABEL)
    if gd:
        ax.loglog([r["sigma"] for r in gd], [r["err_faint"] for r in gd], "x",
                  color=METHOD_STYLES[GD_LABEL]["color"], ms=9, mew=2,
                  label=f"descente de gradient, x = {gd[0]['ratio']:g}")
    ax.plot([], [], "--", color="gray", lw=1, label="borne de Cramér-Rao")
    ax.axhline(success_tol, color="gray", ls=":", lw=1)
    ax.set_xlabel(r"bruit relatif $\sigma$")
    ax.set_ylabel("erreur de position médiane, source faible")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_detection(ax, rows):
    """Détection (N = 2 estimé) et réussite de Prony + MV, pour chaque x."""
    ratios = sorted({r["ratio"] for r in rows})
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(ratios)))
    for color, x in zip(colors, ratios):
        sel = _rows(rows, ratio=x, method=PRONY_LABEL)
        s = [r["sigma"] for r in sel]
        ax.semilogx(s, [100 * r["detection"] for r in sel], "o-", color=color,
                    label=f"détection, x = {x:g}")
        ax.semilogx(s, [100 * r["success"] for r in sel], "s:", color=color,
                    label=f"réussite, x = {x:g}")
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"bruit relatif $\sigma$")
    ax.set_ylabel("%")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, which="both")


def plot_random_success(ax, rows):
    """Géométries aléatoires : réussite des méthodes et part localisable (CRB)."""
    for label in (PRONY_LABEL, GN_LABEL):
        sel = _rows(rows, method=label)
        ax.semilogx([r["sigma"] for r in sel], [100 * r["success"] for r in sel],
                    label=f"réussite, {label}", **METHOD_STYLES[label])
    sel = _rows(rows, method=PRONY_LABEL)
    ax.semilogx([r["sigma"] for r in sel], [100 * r["localizable"] for r in sel], "k--",
                label="CRB source faible < seuil")
    ax.semilogx([r["sigma"] for r in sel], [100 * r["detection"] for r in sel], ":",
                color=METHOD_STYLES[PRONY_LABEL]["color"], label="détection, Prony + MV")
    ax.set_ylim(-5, 105)
    ax.set_xlabel(r"bruit relatif $\sigma$")
    ax.set_ylabel("% des géométries")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_geometry(ax, geo: dict):
    """CRB de la source faible (par unité de bruit) selon sa distance au centre."""
    ax.semilogy(geo["rho_faint"], geo["crb_faint_over_sigma"], "o", ms=4, alpha=0.6,
                color="#34495E")
    ax.set_xlabel(r"distance de la source faible au centre, $\rho / R$")
    ax.set_ylabel(r"CRB source faible / $\sigma$")
    ax.grid(alpha=0.3, which="both")


def plot_noise_study(fixed_rows, random_rows, geo, success_tol: float):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    plot_faint_error(axes[0, 0], fixed_rows, success_tol)
    axes[0, 0].set_title("Géométrie fixe : erreur sur la source faible")
    plot_detection(axes[0, 1], fixed_rows)
    axes[0, 1].set_title("Géométrie fixe : détection et réussite (Prony + MV)")
    if random_rows:
        x = random_rows[0]["ratio"]
        plot_random_success(axes[1, 0], random_rows)
        axes[1, 0].set_title(f"{random_rows[0]['n']} géométries aléatoires, x = {x:g}")
        plot_geometry(axes[1, 1], geo)
        axes[1, 1].set_title(f"Effet de la géométrie, x = {x:g}")
    fig.tight_layout()
    return fig

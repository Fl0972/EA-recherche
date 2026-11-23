"""
Figures de synthèse du compte rendu (à lancer après les expériences).

    python -m sparse_sources.apprentissage.figures
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

from .base import OUT  # noqa: E402


def load(name):
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def curve(rows, meth, sigma):
    return [np.mean([r["reussite"] == "True" for r in rows
                     if r["methode"] == meth and int(r["n"]) == n
                     and abs(float(r["sigma"]) - sigma) < 1e-12]) for n in range(1, 6)]


def fig_methodes():
    """Réussite selon le nombre de sources, pour les principales méthodes."""
    hyb = load("hybride.csv")
    sel = load("selection_k.csv")
    rec = load("recherche.csv") if (OUT / "recherche.csv").exists() else []
    series = [
        (hyb, "greedy_ml (reference)", "ML + BIC (référence)", SERIES[0], "-"),
        (sel, "laplace", "ML + évidence de Laplace", SERIES[1], "-"),
        (sel, "logistique", "ML + K par régression logistique", SERIES[2], "-"),
        (hyb, "reseau+ML+bic", "réseau + ML + BIC", SERIES[3], "-"),
        (hyb, "reseau seul", "réseau seul", SERIES[4], "-"),
    ]
    if rec:
        series.append((rec, "tout", "recherche stochastique + BIC", SERIES[5 % len(SERIES)], "-"))
    series.append((hyb, "ML depart vrai, K vrai", "oracle (départ = vraies sources)", INK_MUTED, "--"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, s in zip(axes, (1e-4, 1e-3, 3e-3)):
        for rows, meth, lab, col, ls in series:
            ax.plot(range(1, 6), curve(rows, meth, s), ls, marker="o", ms=4, lw=2, color=col,
                    label=lab)
        ax.set_title(f"σ = {s:g}", fontsize=10)
        ax.set_xlabel("nombre de sources")
        ax.set_xticks(range(1, 6))
        style_axes(ax)
    axes[0].set_ylabel("taux de réussite")
    axes[0].set_ylim(-0.03, 1.03)
    axes[2].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Scénario du projet : 600 tirages de test (40 par case)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "synthese_methodes.png", dpi=130)
    plt.close(fig)


def fig_geometrie():
    rows = load("geometrie_capteurs_resume.csv")
    rows.sort(key=lambda r: float(r["reussite_moyenne"]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    names = [r["geometrie"] for r in rows]
    y = np.arange(len(rows))
    cols = [SERIES[1] if "projet" in n else SERIES[0] for n in names]
    axes[0].barh(y, [float(r["reussite_moyenne"]) for r in rows], color=cols)
    axes[0].set_yticks(y, names, fontsize=8)
    axes[0].set_xlabel("réussite de ML + BIC (K = 2, 3, 5 ; σ = 1e-4, 1e-3)")
    axes[1].barh(y, [float(r["crb_median_3src"]) for r in rows], color=cols)
    axes[1].set_xscale("log")
    axes[1].set_yticks(y, [""] * len(y))
    axes[1].set_xlabel("borne de Cramér-Rao médiane (3 sources, σ = 1e-3)")
    for ax in axes:
        style_axes(ax)
    fig.suptitle("Position des 64 capteurs", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "synthese_geometrie.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_methodes()
    fig_geometrie()
    print("figures écrites dans", OUT)

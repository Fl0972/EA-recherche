"""
Figures : image d'une reconstruction (vraies sources / sources estimées) et courbes de
comparaison des méthodes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import probleme as P  # noqa: E402
from .methodes import NOMS  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
COULEUR = {"prony": "#2a6fdb", "ml_bic": "#e2583e", "blasso": "#169f78", "descente": "#e0a100",
           "gauss_newton": "#6a4fd6"}
MARQUEUR = {"prony": "+", "ml_bic": "s", "blasso": "^", "descente": "D", "gauss_newton": "v"}
TRAIT = {"cercle": "-", "grille": "--"}
NOM_CAPTEURS = {"cercle": "cercle (64 capteurs)", "grille": "cercle + grille (576 capteurs)"}


def fr(v, chiffres=2) -> str:
    """Nombre en écriture française (virgule décimale)."""
    return f"{v:.{chiffres}g}".replace(".", ",") if np.isfinite(v) else "—"


# ---- reconstruction --------------------------------------------------------------------
def dessiner_reconstruction(a, sit: P.Situation, estimations: dict, titre: str = ""):
    """
    Dessine sur l'axe ``a`` : hangar, cercle et capteurs, source centrale, vraies sources
    (cercles noirs, aire ∝ intensité, intensité écrite à côté) et sources estimées de
    chaque méthode de ``estimations`` = {nom: {"z", "I"}}.
    """
    th = np.linspace(0, 2 * np.pi, 400)
    taille = lambda I: 25 + 9 * np.asarray(I, float)
    a.plot(P.CENTRE[0] + P.RAYON * np.cos(th), P.CENTRE[1] + P.RAYON * np.sin(th),
           color="#bbbbbb", lw=0.8)
    a.add_patch(plt.Rectangle(P.LO, *(P.HI - P.LO), fill=False, color="k", lw=1.2))
    X = sit.X
    a.scatter(*X[:P.N_CERCLE].T, s=7, color="#777777", zorder=2, label="capteurs")
    if len(X) > P.N_CERCLE:
        a.scatter(*X[P.N_CERCLE:].T, s=2, color="#b5b5b5", zorder=1)
    if sit.ic > 0:
        a.scatter(*P.CENTRE, marker="*", s=180, color="k", zorder=3, label="source centrale")
    a.scatter(*sit.z.T, s=taille(sit.I) * 2.2, facecolors="none", edgecolors="k", linewidths=1.6,
              zorder=4, label="vraies sources")
    for nom, est in estimations.items():
        z, I = np.asarray(est["z"], float).reshape(-1, 2), np.asarray(est["I"], float)
        mk = MARQUEUR.get(nom, "o")
        if len(z):
            a.scatter(*z.T, s=taille(np.clip(I, 0, None)) * (1.6 if mk == "+" else 0.9), marker=mk,
                      color=COULEUR.get(nom, "gray"), alpha=0.9, linewidths=1.6 if mk == "+" else 0.5,
                      edgecolors="white" if mk != "+" else None, zorder=5, label=NOMS.get(nom, nom))
        else:
            a.scatter([], [], marker=mk, color=COULEUR.get(nom, "gray"), label=NOMS.get(nom, nom))
    for z, I in zip(sit.z, sit.I):
        a.annotate(f"{I:.0f}", z, xytext=(7, 7), textcoords="offset points", fontsize=7.5,
                   color="#333333")
    a.set(xlim=(P.CENTRE[0] - P.RAYON - 0.5, P.CENTRE[0] + P.RAYON + 0.5),
          ylim=(P.CENTRE[1] - P.RAYON - 0.5, P.CENTRE[1] + P.RAYON + 0.5), title=titre,
          xlabel="x", ylabel="y")
    a.set_aspect("equal")
    a.grid(False)


def figure_simulation(sit: P.Situation, estimations: dict, resumes: dict, chemin):
    """Un panneau par méthode ; sous chaque panneau, le résumé (E, détection, …)."""
    n = len(estimations)
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 6.2), squeeze=False)
    for a, (nom, est) in zip(ax[0], estimations.items()):
        dessiner_reconstruction(a, sit, {nom: est}, NOMS.get(nom, nom))
        a.text(0.5, -0.12, resumes[nom], transform=a.transAxes, ha="center", va="top", fontsize=8)
    h, l = [], []
    for a in ax[0]:
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    x = "" if sit.x is None else f", x = I₁/I₂ = {sit.x:g}"
    fig.suptitle(f"N = {sit.N} sources{x}, {NOM_CAPTEURS.get(sit.capteurs, sit.capteurs)}, "
                 f"σ = {sit.sigma:g}, I_c = {sit.ic / P.S_TOTAL:g} S, tirage {sit.tirage} "
                 "— taille des marqueurs ∝ intensité", fontsize=9, y=0.99)
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, bbox_to_anchor=(0.5, 0.955),
               markerscale=0.7)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(chemin, dpi=160)
    plt.close(fig)


# ---- courbes de comparaison ------------------------------------------------------------
AXES = {"N": "nombre de sources N", "sigma": "bruit relatif σ", "ic": "source centrale I_c / S",
        "x": "rapport d'intensité x = I₁ / I₂"}
PANNEAUX = [("E_mediane", "erreur E (médiane)", "log"),
            ("detection", "taux de détection", "linear"),
            ("precision_d", "précision : distance (médiane)", "log"),
            ("temps_median", "temps par tirage (s, médiane)", "log")]


def figure_comparaison(resume: list[dict], axe: str, chemin, titre: str = ""):
    """
    Une ligne de panneaux par combinaison des paramètres qui ne sont pas sur l'axe ;
    couleur = méthode, trait plein = cercle, pointillés = cercle + grille. Bandes : IC 95 %.
    """
    autres = [k for k in ("N", "sigma", "ic", "x") if k != axe
              and len({r[k] for r in resume}) > 1]
    combos = sorted({tuple(r[k] for k in autres) for r in resume}, key=lambda t: [str(v) for v in t])
    fig, ax = plt.subplots(len(combos), len(PANNEAUX), figsize=(3.2 * len(PANNEAUX), 2.8 * len(combos)),
                           squeeze=False)
    valeurs = sorted({r[axe] for r in resume}, key=lambda v: (v == "", v if v != "" else 0))
    categoriel = axe == "ic"
    pos = {v: (i if categoriel else v) for i, v in enumerate(valeurs)}
    for i, combo in enumerate(combos):
        rows = [r for r in resume if tuple(r[k] for k in autres) == combo]
        for m in dict.fromkeys(r["methode"] for r in rows):
            for c in dict.fromkeys(r["capteurs"] for r in rows):
                v = sorted([r for r in rows if r["methode"] == m and r["capteurs"] == c],
                           key=lambda r: pos[r[axe]])
                if not v:
                    continue
                xs = [pos[r[axe]] for r in v]
                for j, (champ, _, _) in enumerate(PANNEAUX):
                    a = ax[i, j]
                    y = np.array([r[champ] for r in v], float)
                    lo, hi = f"{champ}_ic_bas", f"{champ}_ic_haut"
                    if lo in v[0]:
                        a.fill_between(xs, [r[lo] for r in v], [r[hi] for r in v],
                                       color=COULEUR.get(m, "gray"), alpha=0.12, lw=0)
                    a.plot(xs, y, TRAIT.get(c, "-"), marker="o", color=COULEUR.get(m, "gray"),
                           mfc="white" if c == "grille" else COULEUR.get(m, "gray"),
                           label=f"{NOMS.get(m, m)}" if (i == 0 and j == 0 and c == valeurs_capteurs(rows)[0])
                           else None)
        for j, (_, lab, ech) in enumerate(PANNEAUX):
            a = ax[i, j]
            a.set_yscale(ech)
            if ech == "linear":
                a.set_ylim(-0.03, 1.03)
            if axe in ("sigma", "x"):
                a.set_xscale("log")
            if categoriel:
                a.set_xticks(range(len(valeurs)), [f"{v:g}" for v in valeurs])
            elif axe == "N":
                a.set_xticks(valeurs)
            a.set(xlabel=AXES[axe], ylabel=lab)
            if combo:
                a.set_title(", ".join(f"{k} = {v:g}" if isinstance(v, float) else f"{k} = {v}"
                                      for k, v in zip(autres, combo)), fontsize=8.5)
    for c in valeurs_capteurs(resume):
        ax[0, 0].plot([], [], TRAIT[c], color="gray", label=NOM_CAPTEURS[c])
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 7), frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.03 / len(combos)))
    if titre:
        fig.suptitle(titre, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160)
    plt.close(fig)


def valeurs_capteurs(rows):
    return list(dict.fromkeys(r["capteurs"] for r in rows))

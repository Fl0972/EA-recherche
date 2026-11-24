"""
Figures de synthèse du rapport (rapport/images/), avec la métrique après fusion des sources
(compare.protocol.erreur) : moyenne sur les tirages et intervalle de confiance à 95 % par
bootstrap. Mêmes conventions que l'ancien rapport/figures_synthese.py (couleurs, marqueurs,
cercle en traits pleins, cercle + grille en pointillés), mais lues dans les CSV :

    results/agrege_<balayage>_<config>.csv   (python -m compare.run)
    results/delta_grille.csv
    results/sources/deux_sources.jsonl       (python -m compare.deux_sources)

    python -m compare.figures_synthese [--delta 0.1]

Sorties : rapport/images/fig_nsources, fig_balayages, fig_deux_sources, fig_delta (.png 220 dpi, .pdf)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare import protocol as P  # noqa: E402

RDIR = ROOT / "results"
SOURCES_DS = RDIR / "sources" / "deux_sources.jsonl"
OUT = ROOT / "rapport" / "images"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e4e4", "grid.linewidth": 0.6, "lines.linewidth": 1.8,
                     "lines.markersize": 4.5, "savefig.bbox": "tight", "pdf.fonttype": 42})
C = {"descente": "#e0a100", "gn": "#6a4fd6", "ml_bic": "#e2583e", "fw": "#169f78", "prony": "#2a6fdb"}
M = {"descente": "D", "gn": "v", "ml_bic": "s", "fw": "^", "prony": "o"}
L = {"descente": "Descente de gradient (N connu)", "gn": "Gauss–Newton (N connu, même départ)",
     "ml_bic": "ML + BIC", "fw": "Frank–Wolfe glissant", "prony": "Prony"}
METHODES = ["prony", "ml_bic", "fw"]
NS = [2, 4, 6, 10]
XS = [1, 3, 10, 30, 100, 300]
ORANO = {1: 0.95, 3: 0.85, 10: 0.55, 30: 0.20, 100: 0.05}
ALPHA_BANDE = 0.15


def save(fig, nom):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{nom}.png", dpi=220)
    fig.savefig(OUT / f"{nom}.pdf")
    plt.close(fig)


def lire(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k != "methode":
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def serie(rows, m, N, xs, champ="E_moyenne", xkey="x"):
    """Valeurs de ``champ`` (et IC 95 % si disponible) pour chaque x."""
    y, lo, hi = [], [], []
    base = champ.replace("_moyenne", "")
    for x in xs:
        v = [r for r in rows if r["methode"] == m and r["N"] == N and np.isclose(r[xkey], x)]
        if not v:
            y.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            continue
        y.append(v[0][champ])
        lo.append(v[0].get(f"{base}_ic95_bas", np.nan))
        hi.append(v[0].get(f"{base}_ic95_haut", np.nan))
    return np.array(y, float), np.array(lo, float), np.array(hi, float)


def courbe(a, x, y, lo, hi, m, ls="-", label=None, bande=True):
    if bande and not np.all(np.isnan(lo)):
        a.fill_between(x, lo, hi, color=C[m], alpha=ALPHA_BANDE if ls == "-" else 0.10, lw=0)
    a.plot(x, y, ls, marker=M[m], color=C[m], label=label, mfc="white" if ls != "-" else C[m])


def legende_config(a):
    a.plot([], [], color="gray", label="cercle (64 capteurs)")
    a.plot([], [], color="gray", ls="--", label="cercle + grille (576 capteurs)")


# ---- fig_nsources ------------------------------------------------------------------------
def fig_nsources():
    R = {c: lire(RDIR / f"agrege_nsources_{c}.csv") for c in P.CONFIGS}
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
    for m in METHODES:
        for c, ls in (("cercle", "-"), ("grille", "--")):
            if m == "prony" and c == "grille":
                continue                        # Prony n'utilise que le cercle : courbes identiques
            rows = R[c]
            e = [serie(rows, m, N, [0.0]) for N in NS]
            y, lo, hi = (np.array([v[k][0] for v in e]) for k in range(3))
            courbe(ax[0], NS, y, lo, hi, m, ls, L[m] if c == "cercle" else None)
            p = [serie(rows, m, N, [0.0], "masse_parasite_moyenne") for N in NS]
            y, lo, hi = (np.array([v[k][0] for v in p]) for k in range(3))
            courbe(ax[1], NS, y, lo, hi, m, ls)
            t = [serie(rows, m, N, [0.0], "temps_median")[0][0] for N in NS]
            ax[2].plot(NS, t, ls, marker=M[m], color=C[m], mfc="white" if ls != "-" else C[m])
    ax[0].set(yscale="log", ylabel="erreur E (moyenne, IC 95 %)", title="(a) Erreur après fusion")
    ax[0].axhline(1, color="gray", lw=0.8, ls=":")
    ax[1].set(ylabel="masse parasite (moyenne, IC 95 %)", title="(b) Intensité estimée en trop")
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].set_ylim(bottom=0)
    ax[2].set(yscale="log", ylabel="temps par tirage (s, médiane)", title="(c) Temps de calcul")
    for a in ax:
        a.set(xticks=NS, xlabel="nombre de sources N")
    legende_config(ax[0])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_nsources")


# ---- fig_balayages -----------------------------------------------------------------------
def fig_balayages():
    Rb = {c: lire(RDIR / f"agrege_bruit_{c}.csv") for c in P.CONFIGS}
    Rc = {c: lire(RDIR / f"agrege_centrale_{c}.csv") for c in P.CONFIGS}
    sig = sorted({r["x"] for r in Rb["cercle"]})
    rap = sorted({r["x"] for r in Rc["cercle"]})
    xc = np.arange(len(rap))
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.4), sharey=True)
    for j, N in enumerate([4, 6, 10]):
        for row, (R, xs, X) in enumerate([(Rb, sig, sig), (Rc, rap, xc)]):
            a = ax[row, j]
            for m in METHODES:
                for c, ls in (("cercle", "-"), ("grille", "--")):
                    if m == "prony" and c == "grille":
                        continue
                    y, lo, hi = serie(R[c], m, N, xs)
                    courbe(a, X, y, lo, hi, m, ls, L[m] if (row == 0 and j == 0 and c == "cercle") else None)
            a.set_yscale("log")
            a.axhline(1, color="gray", lw=0.8, ls=":")
            if row == 0:
                a.set(xscale="log", xlabel="bruit relatif σ", title=f"N = {N}")
                a.axvline(1e-3, color="gray", lw=0.8, ls="-.")
            else:
                a.set_xticks(xc, [f"{r:g}" for r in rap])
                a.set(xlabel="I_c / S")
                a.axvline(rap.index(50.0), color="gray", lw=0.8, ls="-.")
    ax[0, 0].set_ylabel("erreur E (moyenne, IC 95 %)\nbalayage en bruit, I_c = 50 S")
    ax[1, 0].set_ylabel("erreur E (moyenne, IC 95 %)\nsource centrale, σ = 10⁻³")
    legende_config(ax[0, 0])
    ax[0, 0].plot([], [], color="gray", ls="-.", lw=0.8, label="cadre par défaut")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig_balayages")


# ---- fig_deux_sources --------------------------------------------------------------------
def lire_deux_sources():
    with open(SOURCES_DS, encoding="utf-8") as f:
        return [json.loads(L) for L in f if L.strip()]


def reussites(rows, delta):
    """Réussite après fusion, pour chaque tirage et chaque méthode (recalculée depuis les sources)."""
    from compare.deux_sources import reussite_fusion
    out = []
    for r in rows:
        o = {k: r[k] for k in ("cadre", "x", "tirage", "config", "kappa")}
        for m in ("descente", "gn", "ml_bic", "fw", "prony", "ref"):
            o[m] = reussite_fusion(np.array(r["z_vrai"]), np.array(r["I_vrai"]), r[f"zest_{m}"],
                                   r[f"Iest_{m}"], delta)[0]
        out.append(o)
    return out


def taux(rows, cadre, config, m):
    return np.array([np.mean([r[m] for r in rows if r["cadre"] == cadre and r["config"] == config
                              and float(r["x"]) == x]) for x in XS])


def fig_deux_sources(delta=P.DELTA_DEF):
    rows = reussites(lire_deux_sources(), delta)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
    a = ax[0]
    for m in ("gn", "ml_bic", "fw", "prony", "descente"):
        a.plot(XS, taux(rows, "orano", "cercle", m), marker=M[m], color=C[m], label=L[m])
    a.scatter(list(ORANO), list(ORANO.values()), marker="x", s=45, color="k", zorder=5,
              label="Résultats d'Orano (descente)")
    a.set(xscale="log", title="(a) Cas test d'Orano : σ = 0, sans source centrale",
          xlabel="rapport d'intensité x = I₁ / I₂", ylabel="réussite (après fusion)", ylim=(-0.04, 1.05))
    a = ax[1]
    for m in ("descente", "ml_bic", "fw"):
        a.plot(XS, taux(rows, "projet", "cercle", m), marker=M[m], color=C[m])
        a.plot(XS, taux(rows, "projet", "grille", m), marker=M[m], color=C[m], ls="--", mfc="white")
    legende_config(a)
    a.legend(loc="center right", frameon=False)
    a.set(xscale="log", title="(b) Cadre du projet : σ = 10⁻³, I_c = 50 S", xlabel="rapport d'intensité x",
          ylim=(-0.04, 1.05))
    a = ax[2]
    ro = [r for r in rows if r["cadre"] == "orano" and r["config"] == "cercle"]
    k = [[r["kappa"] for r in ro if float(r["x"]) == x] for x in XS]
    med = np.array([np.median(v) for v in k])
    a.fill_between(XS, [np.percentile(v, 25) for v in k], [np.percentile(v, 75) for v in k],
                   color=C["descente"], alpha=0.18, lw=0)
    a.plot(XS, med, marker="D", color=C["descente"], label="κ médian (Q1–Q3)")
    xx = np.array([1, 300])
    a.plot(xx, med[2] * (xx / 10) ** 2, "k--", lw=1, label="pente 2 : κ ∝ x²")
    a.set(xscale="log", yscale="log", title="(c) Conditionnement de la perte à la solution",
          xlabel="rapport d'intensité x", ylabel="κ = λ_max / λ_min")
    a.legend(loc="lower right", frameon=False)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, "fig_deux_sources")


# ---- fig_delta ---------------------------------------------------------------------------
def fig_delta():
    R = lire(RDIR / "delta_grille.csv")
    deltas = sorted({r["delta"] for r in R})
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    xp = np.arange(len(deltas))                          # positions régulières (δ = 0 inclus)
    for a, N in zip(ax, (4, 10)):
        for m in METHODES:
            y, lo, hi = serie(R, m, N, deltas, xkey="delta")
            courbe(a, xp, y, lo, hi, m, "-", L[m] if N == 4 else None)
        a.axvline(deltas.index(P.DELTA_DEF), color="gray", lw=0.8, ls="-.")
        a.set(yscale="log", xlabel="rayon de fusion δ", title=f"N = {N}, cercle + grille")
        a.set_xticks(xp, [f"{d:g}".replace(".", ",") for d in deltas])
    ax[0].set_ylabel("erreur E (moyenne, IC 95 %)")
    h, l = ax[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="gray", lw=0.8, ls="-.")); l.append("δ par défaut")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_delta")


def toutes(delta=P.DELTA_DEF):
    if (RDIR / "agrege_nsources_cercle.csv").exists():
        fig_nsources()
        fig_balayages()
    if (RDIR / "delta_grille.csv").exists():
        fig_delta()
    if SOURCES_DS.exists():
        fig_deux_sources(delta)
    print(f"  figures de synthèse dans {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pa = argparse.ArgumentParser()
    pa.add_argument("--delta", type=float, default=P.DELTA_DEF)
    toutes(pa.parse_args().delta)

"""
Figures de synthèse du rapport (rapport/images/), avec la métrique après fusion des sources
(compare.protocol.erreur) : médiane de E sur les tirages, taux de détection moyen (vraie source
appariée à moins de P.SEUIL_DETECTION), précision (médiane de la distance moyenne aux sources
détectées), avec intervalles de confiance à 95 % par bootstrap. Mêmes conventions que l'ancien rapport/figures_synthese.py (couleurs, marqueurs,
cercle en traits pleins, cercle + grille en pointillés), mais lues dans les CSV :

    results/agrege_<balayage>_<config>.csv   (python -m compare.run)
    results/delta_grille.csv
    results/sources/deux_sources.jsonl       (python -m compare.deux_sources)

    python -m compare.figures_synthese [--delta 0.1]

Sorties : rapport/images/fig_nsources, fig_balayages, fig_detection, fig_deux_sources, fig_delta (.png 220 dpi, .pdf)
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


def serie(rows, m, N, xs, champ="E_mediane", xkey="x"):
    """Valeurs de ``champ`` (et IC 95 % si disponible) pour chaque x."""
    y, lo, hi = [], [], []
    base = "E_med" if champ == "E_mediane" else champ.replace("_moyenne", "").replace("_mediane", "")
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
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.6))
    ax = ax.ravel()
    seuil = f"{P.SEUIL_DETECTION:g}".replace(".", ",")
    panneaux = [("E_mediane", "erreur E (médiane, IC 95 %)", "(a) Erreur après fusion", "log"),
                ("detection_moyenne", "part des sources détectées (moyenne)",
                 f"(b) Taux de détection (d < {seuil})", "linear"),
                ("d_det_mediane", "distance (médiane, IC 95 %)", "(c) Précision sur les sources détectées", "log"),
                ("masse_parasite_moyenne", "masse parasite / S (moyenne)", "(d) Intensité estimée en trop",
                 "symlog"),
                ("temps_median", "temps par tirage (s, médiane)", "(e) Temps de calcul", "log")]
    for m in METHODES:
        for c, ls in (("cercle", "-"), ("grille", "--")):
            if m == "prony" and c == "grille":
                continue                        # Prony n'utilise que le cercle : courbes identiques
            for k, (champ, *_r) in enumerate(panneaux):
                e = [serie(R[c], m, N, [0.0], champ) for N in NS]
                y, lo, hi = (np.array([v[i][0] for v in e]) for i in range(3))
                courbe(ax[k], NS, y, lo, hi, m, ls, L[m] if (c == "cercle" and k == 0) else None)
    for a, (_c, yl, ti, sc) in zip(ax, panneaux):
        a.set(ylabel=yl, title=ti, xticks=NS, xlabel="nombre de sources N")
        if sc == "symlog":
            a.set_yscale("symlog", linthresh=1e-3)
            a.set_ylim(bottom=0)
        else:
            a.set_yscale(sc)
    ax[0].axhline(1, color="gray", lw=0.8, ls=":")
    ax[1].set_ylim(-0.03, 1.03)
    ax[5].axis("off")
    legende_config(ax[0])
    h, l = ax[0].get_legend_handles_labels()
    ax[5].legend(h, l, loc="center", frameon=False)
    fig.tight_layout()
    save(fig, "fig_nsources")


# ---- fig_balayages -----------------------------------------------------------------------
def fig_balayages(champ="E_mediane", nom="fig_balayages", ylab="erreur E (médiane, IC 95 %)", log=True):
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
                    y, lo, hi = serie(R[c], m, N, xs, champ)
                    courbe(a, X, y, lo, hi, m, ls, L[m] if (row == 0 and j == 0 and c == "cercle") else None)
            if log:
                a.set_yscale("log")
                a.axhline(1, color="gray", lw=0.8, ls=":")
            else:
                a.set_ylim(-0.03, 1.03)
            if row == 0:
                a.set(xscale="log", xlabel="bruit relatif σ", title=f"N = {N}")
                a.axvline(1e-3, color="gray", lw=0.8, ls="-.")
            else:
                a.set_xticks(xc, [f"{r:g}" for r in rap])
                a.set(xlabel="I_c / S")
                a.axvline(rap.index(50.0), color="gray", lw=0.8, ls="-.")
    ax[0, 0].set_ylabel(f"{ylab}\nbalayage en bruit, I_c = 50 S")
    ax[1, 0].set_ylabel(f"{ylab}\nsource centrale, σ = 10⁻³")
    legende_config(ax[0, 0])
    ax[0, 0].plot([], [], color="gray", ls="-.", lw=0.8, label="cadre par défaut")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, nom)


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
            y, lo, hi = serie(R, m, N, deltas, "E_mediane", xkey="delta")
            courbe(a, xp, y, lo, hi, m, "-", L[m] if N == 4 else None)
        a.axvline(deltas.index(P.DELTA_DEF), color="gray", lw=0.8, ls="-.")
        a.set(yscale="log", xlabel="rayon de fusion δ", title=f"N = {N}, cercle + grille")
        a.set_xticks(xp, [f"{d:g}".replace(".", ",") for d in deltas])
    ax[0].set_ylabel("erreur E (médiane, IC 95 %)")
    h, l = ax[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="gray", lw=0.8, ls="-.")); l.append("δ par défaut")
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig_delta")


# ---- fig_reconstruction : un exemple de sources vraies / estimées --------------------------
def choisir_tirage_reconstruction(N, cas, delta=P.DELTA_DEF):
    """Premier tirage où ML + BIC détecte toutes les sources dans chacun des cas (et sources
    vraies assez espacées pour être lisibles)."""
    from compare import run as RUN
    src = {c: RUN.lire_sources(RUN.chemin_sources(c)) for c in P.CONFIGS}
    for t in range(100):
        ok = True
        for (c, s, ic) in cas:
            r = src[c].get(RUN.cle(c, N, s, ic, t, "ml_bic"))
            if r is None:
                return None
            sit = P.situation(N, t, s, ic, c)
            if min(np.linalg.norm(a - b) for i, a in enumerate(sit.z) for b in sit.z[i + 1:]) < 1.5:
                ok = False
                break
            e = P.erreur(sit.z, sit.I, np.array(r["z_est"] or [], float).reshape(-1, 2),
                         np.array(r["I_est"] or [], float), delta=delta)
            if e["taux_detection"] < 1 or e["n_est"] != N:
                ok = False
                break
        if ok:
            return t
    return None


def panneau_reconstruction(a, src, c, s, ic, N, t, titre, montrer_E=False):
    """Vraies sources (cercles, aire ∝ intensité), sources estimées par les trois méthodes, capteurs."""
    from compare import run as RUN
    th = np.linspace(0, 2 * np.pi, 400)
    taille = lambda I: 25 + 9 * np.asarray(I, float)                     # aire ∝ intensité
    sit = P.situation(N, t, s, ic, c)
    a.plot(P.CENTRE[0] + P.RAYON * np.cos(th), P.CENTRE[1] + P.RAYON * np.sin(th), color="#bbbbbb", lw=0.8)
    a.add_patch(plt.Rectangle(P.LO, *(P.HI - P.LO), fill=False, color="k", lw=1.2))
    X = P.capteurs(c)
    a.scatter(*X[:P.N_CERCLE].T, s=7, color="#777777", zorder=2, label="capteurs")
    if len(X) > P.N_CERCLE:
        a.scatter(*X[P.N_CERCLE:].T, s=2, color="#b5b5b5", zorder=1)
    if ic > 0:
        a.scatter(*P.CENTRE, marker="*", s=180, color="k", zorder=3, label="source centrale")
    a.scatter(*sit.z.T, s=taille(sit.I) * 2.2, facecolors="none", edgecolors="k", linewidths=1.6,
              zorder=4, label="vraies sources")
    lignes_E = []
    for m, mk in (("ml_bic", "s"), ("fw", "^"), ("prony", "+")):
        r = src[c].get(RUN.cle(c, N, s, ic, t, m))
        if r is None:
            continue
        ze = np.array(r["z_est"] or [], float).reshape(-1, 2)
        Ie = np.array(r["I_est"] or [], float).reshape(-1)
        e = P.erreur(sit.z, sit.I, ze, Ie)
        lignes_E.append(f"{L[m]} : E = {e['E']:.2f}, {round(e['taux_detection'] * e['N_fusion'])}/{e['N_fusion']} "
                        f"détectées, {len(Ie)} estimées".replace(".", ","))
        if len(ze):
            a.scatter(*ze.T, s=taille(np.clip(Ie, 0, None)) * (1.6 if mk == "+" else 0.9), marker=mk,
                      color=C[m], alpha=0.9, linewidths=1.6 if mk == "+" else 0.5,
                      edgecolors="white" if mk != "+" else None, zorder=5, label=L[m])
        else:
            a.scatter([], [], marker=mk, color=C[m], label=L[m])
    for z, I in zip(sit.z, sit.I):
        a.annotate(f"{I:.0f}", z, xytext=(7, 7), textcoords="offset points", fontsize=7.5, color="#333333")
    a.set(xlim=(P.CENTRE[0] - P.RAYON - 0.5, P.CENTRE[0] + P.RAYON + 0.5),
          ylim=(P.CENTRE[1] - P.RAYON - 0.5, P.CENTRE[1] + P.RAYON + 0.5), title=titre, xlabel="x", ylabel="y")
    a.set_aspect("equal")
    a.grid(False)
    if montrer_E:
        a.text(0.5, -0.12,"\n".join(lignes_E), transform=a.transAxes, ha="center", va="top", fontsize=7.5)


def fig_reconstruction(N=6):
    from compare import run as RUN
    cas = [("cercle", 1e-3, 0.0, "(a) Cercle seul, σ = 10⁻³, sans source centrale"),
           ("grille", 1e-3, 50.0, "(b) Cercle + grille, cadre du projet (σ = 10⁻³, I_c = 50 S)")]
    t = choisir_tirage_reconstruction(N, [c[:3] for c in cas])
    if t is None:
        print("  fig_reconstruction : aucun tirage adapté")
        return
    src = {c: RUN.lire_sources(RUN.chemin_sources(c)) for c in P.CONFIGS}
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.4))
    for a, (c, s, ic, titre) in zip(ax, cas):
        panneau_reconstruction(a, src, c, s, ic, N, t, titre)
    h, l = ax[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.03), markerscale=0.7)
    fig.suptitle(f"Exemple de reconstruction, N = {N} sources (tirage {t}) ; taille des marqueurs ∝ intensité, "
                 "nombres : intensités vraies (S = 100)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_reconstruction")
    print(f"  fig_reconstruction : tirage {t}")


def fig_reconstruction_E(N=6, cibles=(1.0, 3.0), config="cercle"):
    """Deux tirages du cadre par défaut où l'erreur E de ML + BIC vaut à peu près 1 et 3."""
    from compare import run as RUN
    src = {c: RUN.lire_sources(RUN.chemin_sources(c)) for c in P.CONFIGS}
    s, ic = RUN.SIGMA_DEF, RUN.IC_DEF
    E = {}
    for t in range(100):
        r = src[config].get(RUN.cle(config, N, s, ic, t, "ml_bic"))
        if r is None:
            continue
        sit = P.situation(N, t, s, ic, config)
        E[t] = P.erreur(sit.z, sit.I, np.array(r["z_est"] or [], float).reshape(-1, 2),
                        np.array(r["I_est"] or [], float))["E"]
    choix = [min(E, key=lambda t: abs(E[t] - cible)) for cible in cibles]
    fig, ax = plt.subplots(1, 2, figsize=(10, 6.8))
    cfg = "cercle seul" if config == "cercle" else "cercle + grille"
    for a, t, lettre in zip(ax, choix, "ab"):
        panneau_reconstruction(a, src, config, s, ic, N, t,
                               f"({lettre}) E(ML + BIC) = {E[t]:.2f} — tirage {t}".replace(".", ","), montrer_E=True)
    h, l = ax[0].get_legend_handles_labels()
    fig.suptitle(f"À quoi ressemble une erreur E ≈ 1 et E ≈ 3 : N = {N}, {cfg}, cadre du projet "
                 "(σ = 10⁻³, I_c = 50 S) ; taille ∝ intensité", fontsize=9, y=0.99)
    fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.965), markerscale=0.7)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, "fig_reconstruction_E")
    print(f"  fig_reconstruction_E : tirages {choix} (E = {[round(E[t], 2) for t in choix]})")


# ---- fig_detection_seuil : détection selon le rayon de tolérance ---------------------------
def detection_par_seuil(config, N, seuils, s=None, ic=None):
    """Taux de détection moyen (vraies sources appariées à moins de chaque seuil), par méthode."""
    from compare import run as RUN
    s = RUN.SIGMA_DEF if s is None else s
    ic = RUN.IC_DEF if ic is None else ic
    src = RUN.lire_sources(RUN.chemin_sources(config))
    out = {}
    for m in METHODES:
        dist = []                                              # distances d_i de chaque vraie source
        for t in range(100):
            r = src.get(RUN.cle(config, N, s, ic, t, m))
            if r is None:
                continue
            sit = P.situation(N, t, s, ic, config)
            e = P.erreur(sit.z, sit.I, np.array(r["z_est"] or [], float).reshape(-1, 2),
                         np.array(r["I_est"] or [], float), seuil=np.inf)
            dist.append(np.array(e["d_toutes"]))
        out[m] = np.array([np.mean([np.mean(d < sv) for d in dist]) for sv in seuils])
    return out


def fig_detection_seuil(seuil=0.5):
    seuils = np.logspace(-3, np.log10(3), 60)          # < 5 : une source manquée compte d = 5
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    a = ax[0]
    for c, ls in (("cercle", "-"), ("grille", "--")):
        for m in METHODES:
            if m == "prony" and c == "grille":
                continue
            y01 = [detection_par_seuil(c, N, [P.SEUIL_DETECTION, seuil])[m] for N in NS]
            a.plot(NS, [v[1] for v in y01], ls, marker=M[m], color=C[m], mfc="white" if ls != "-" else C[m],
                   label=L[m] if c == "cercle" else None)
            a.plot(NS, [v[0] for v in y01], ls, color=C[m], lw=0.8, alpha=0.45)
    a.set(xticks=NS, xlabel="nombre de sources N", ylabel="part des sources détectées", ylim=(-0.03, 1.03),
          title=f"(a) Détection à d < {seuil:g}\n(trait fin : d < {P.SEUIL_DETECTION:g})".replace(".", ","))
    for a, N in zip(ax[1:], (4, 10)):
        for c, ls in (("cercle", "-"), ("grille", "--")):
            D = detection_par_seuil(c, N, seuils)
            for m in METHODES:
                if m == "prony" and c == "grille":
                    continue
                a.plot(seuils, D[m], ls, color=C[m])
        for sv in (P.SEUIL_DETECTION, seuil):
            a.axvline(sv, color="gray", lw=0.8, ls=":")
        a.set(xscale="log", xlabel="rayon de tolérance (unités du hangar)", ylim=(-0.03, 1.03),
              title=f"({'b' if N == 4 else 'c'}) N = {N} : détection selon le rayon")
    ax[1].set_ylabel("part des sources détectées")
    legende_config(ax[0])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Cadre par défaut (σ = 10⁻³, I_c = 50 S), 100 tirages par point", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, "fig_detection_seuil")


def toutes(delta=P.DELTA_DEF):
    if (RDIR / "agrege_nsources_cercle.csv").exists():
        fig_nsources()
        fig_balayages()
        seuil = f"{P.SEUIL_DETECTION:g}".replace(".", ",")
        fig_balayages("detection_moyenne", "fig_detection", f"sources détectées (d < {seuil}), moyenne", log=False)
    if (RDIR / "delta_grille.csv").exists():
        fig_delta()
    fig_reconstruction()
    fig_reconstruction_E()
    fig_detection_seuil()
    if SOURCES_DS.exists():
        fig_deux_sources(delta)
    print(f"  figures de synthèse dans {OUT}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pa = argparse.ArgumentParser()
    pa.add_argument("--delta", type=float, default=P.DELTA_DEF)
    toutes(pa.parse_args().delta)

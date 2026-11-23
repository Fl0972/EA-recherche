"""
Figures de la partie « Comparaison », tracées uniquement à partir des CSV de
results/ (on peut retracer sans recalculer : python -m compare.run --figures).

Chaque méthode a une couleur ET un marqueur fixes, identiques sur toutes les
figures (lisible en noir et blanc et pour les daltoniens). Largeur pensée
pour une pleine largeur A4 (≈ 17 cm).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare.protocol import METHODES as TOUTES, NOMS  # noqa: E402

METHODES = list(TOUTES)          # méthodes tracées (fixé par toutes())

COULEUR = {"prony": "#2a78d6", "ml_bic": "#eb6834", "fw": "#1baf7a", "descente": "#eda100"}
MARQUEUR = {"prony": "o", "ml_bic": "s", "fw": "^", "descente": "D"}
INK, MUTED, GRID, AXIS = "#0b0b0b", "#6b6a63", "#e1e0d9", "#c3c2b7"
LARGEUR = 7.0                                          # pouces ≈ 17,8 cm
E_MIN = 1e-4                                           # plancher de l'axe log de E
NS = [2, 4, 6, 10]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def nom_methode(m, config):
    if m == "prony" and config == "grille":
        return "Prony (cercle seul)"
    return NOMS[m]


def lire(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("E", "sigma", "ic_ratio", "temps", "d_moy", "r_moy", "d_max"):
            r[k] = float(r[k]) if r.get(k) not in (None, "") else np.nan
        r["N"] = int(r["N"])
        r["n_est"] = int(r["n_est"])
        r["N_exact"] = r["N_exact"] in ("True", "1", True)
    return rows


def stats(rows, m, N, xkey, xs, champ="E"):
    """Médiane, quartiles de ``champ`` et proportion N̂ = N pour chaque x."""
    med, q1, q3, pex = [], [], [], []
    for x in xs:
        v = [r for r in rows if r["methode"] == m and r["N"] == N and np.isclose(r[xkey], x)]
        if not v:
            med.append(np.nan); q1.append(np.nan); q3.append(np.nan); pex.append(np.nan)
            continue
        a = np.array([r[champ] for r in v], float)
        med.append(np.nanmedian(a)); q1.append(np.nanpercentile(a, 25)); q3.append(np.nanpercentile(a, 75))
        pex.append(np.mean([r["N_exact"] for r in v]))
    return np.array(med), np.array(q1), np.array(q3), np.array(pex)


def courbe(ax, x, med, q1, q3, m, ls="-", bande=True, label=None, alpha_bande=0.16):
    if bande:
        ax.fill_between(x, np.maximum(q1, E_MIN), np.maximum(q3, E_MIN), color=COULEUR[m],
                        alpha=alpha_bande, lw=0)
    ax.plot(x, np.maximum(med, E_MIN), ls, color=COULEUR[m], marker=MARQUEUR[m], ms=4.5,
            lw=1.6, label=label, mec="white", mew=0.6)


def sauver(fig, fdir: Path, nom: str):
    fig.savefig(fdir / f"{nom}.pdf")
    fig.savefig(fdir / f"{nom}.png", dpi=200)
    plt.close(fig)


def _legende(fig, handles, labels, ncol=4):
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0))


# ---- balayages a (bruit) et b (centrale) ------------------------------------------------
def fig_balayage(rows, config, xkey, xlabel, xlog, titre, fdir, nom):
    xs = sorted({r[xkey] for r in rows})
    cat = xkey == "ic_ratio"                 # valeurs de I_c / S : positions régulières
    xp = list(range(len(xs))) if cat else xs
    fig, axes = plt.subplots(2, 4, figsize=(LARGEUR, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for j, N in enumerate(NS):
        a, b = axes[0, j], axes[1, j]
        for m in METHODES:
            med, q1, q3, pex = stats(rows, m, N, xkey, xs)
            courbe(a, xp, med, q1, q3, m, label=nom_methode(m, config))
            if m != "descente":
                b.plot(xp, pex, "-", color=COULEUR[m], marker=MARQUEUR[m], ms=4, lw=1.4,
                       mec="white", mew=0.6)
        a.set_yscale("log")
        a.set_title(f"N = {N}")
        b.set_ylim(-0.05, 1.05)
        b.set_xlabel(xlabel)
        for ax in (a, b):
            style(ax)
            if xlog and not cat:
                ax.set_xscale("log")
        if cat:
            b.set_xticks(xp, [f"{x:g}" for x in xs], fontsize=7)
    axes[0, 0].set_ylabel("erreur E (médiane, Q1–Q3)")
    axes[1, 0].set_ylabel("proportion N̂ = N")
    ylo = min(ax.get_ylim()[0] for ax in axes[0])
    yhi = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(ylo, yhi)
    h, l = axes[0, 0].get_legend_handles_labels()
    _legende(fig, h, l)
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    sauver(fig, fdir, nom)


# ---- balayage c (nombre de sources) -----------------------------------------------------
def fig_nsources(rows, config, fdir, nom, titre):
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGEUR, 3.0))
    for m in METHODES:
        med, q1, q3, pex = [], [], [], []
        for N in NS:
            s = stats(rows, m, N, "N", [N])
            med.append(s[0][0]); q1.append(s[1][0]); q3.append(s[2][0]); pex.append(s[3][0])
        courbe(a, NS, np.array(med), np.array(q1), np.array(q3), m, label=nom_methode(m, config))
        if m != "descente":
            b.plot(NS, pex, "-", color=COULEUR[m], marker=MARQUEUR[m], ms=4.5, lw=1.6,
                   mec="white", mew=0.6)
    a.set_yscale("log")
    a.set_ylabel("erreur E (médiane, Q1–Q3)")
    b.set_ylabel("proportion N̂ = N")
    b.set_ylim(-0.05, 1.05)
    for ax in (a, b):
        ax.set_xlabel("nombre de sources N")
        ax.set_xticks(NS)
        style(ax)
    h, l = a.get_legend_handles_labels()
    _legende(fig, h, l)
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.97))
    sauver(fig, fdir, nom)


# ---- balayage d (temps) -----------------------------------------------------------------
def fig_temps(rows, config, fdir, nom, titre):
    fig, a = plt.subplots(figsize=(LARGEUR * 0.75, 3.6))
    for m in METHODES:
        med, q1, q3 = [], [], []
        for N in NS:
            s = stats(rows, m, N, "N", [N], champ="temps")
            med.append(s[0][0]); q1.append(s[1][0]); q3.append(s[2][0])
        a.fill_between(NS, q1, q3, color=COULEUR[m], alpha=0.16, lw=0)
        a.plot(NS, med, "-", color=COULEUR[m], marker=MARQUEUR[m], ms=4.5, lw=1.6,
               label=nom_methode(m, config), mec="white", mew=0.6)
    a.set_yscale("log")
    a.set_xlabel("nombre de sources N")
    a.set_ylabel("temps de calcul (s, médiane, Q1–Q3)")
    a.set_xticks(NS)
    style(a)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3, fontsize=8)
    a.set_title(titre.replace(" — ", chr(10)), fontsize=9.5)
    fig.tight_layout()
    sauver(fig, fdir, nom)


# ---- synthèse cercle / grille -----------------------------------------------------------
def fig_synthese_balayage(rc, rg, xkey, xlabel, xlog, titre, fdir, nom):
    """Cercle en traits pleins, cercle + grille en pointillés, même panneau."""
    xs = sorted({r[xkey] for r in rc + rg})
    cat = xkey == "ic_ratio"
    xp = list(range(len(xs))) if cat else xs
    fig, axes = plt.subplots(2, 4, figsize=(LARGEUR, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [1.6, 1]})
    for j, N in enumerate(NS):
        a, b = axes[0, j], axes[1, j]
        for rows, ls in ((rc, "-"), (rg, "--")):
            if not rows:
                continue
            for m in METHODES:
                med, q1, q3, pex = stats(rows, m, N, xkey, xs)
                courbe(a, xp, med, q1, q3, m, ls=ls, bande=False)
                if m != "descente":
                    b.plot(xp, pex, ls, color=COULEUR[m], marker=MARQUEUR[m], ms=3.5, lw=1.3,
                           mec="white", mew=0.5)
        a.set_yscale("log")
        a.set_title(f"N = {N}")
        b.set_ylim(-0.05, 1.05)
        b.set_xlabel(xlabel)
        for ax in (a, b):
            style(ax)
            if xlog and not cat:
                ax.set_xscale("log")
        if cat:
            b.set_xticks(xp, [f"{x:g}" for x in xs], fontsize=7)
    axes[0, 0].set_ylabel("erreur E (médiane)")
    axes[1, 0].set_ylabel("proportion N̂ = N")
    ylo = min(ax.get_ylim()[0] for ax in axes[0])
    yhi = max(ax.get_ylim()[1] for ax in axes[0])
    for ax in axes[0]:
        ax.set_ylim(ylo, yhi)
    _legende_synthese(fig)
    fig.suptitle(titre, fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 0.98))
    sauver(fig, fdir, nom)


def _legende_synthese(fig):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COULEUR[m], marker=MARQUEUR[m], ls="-", ms=4.5, mec="white")
         for m in METHODES]
    l = [NOMS[m] for m in METHODES]
    h += [Line2D([], [], color=MUTED, ls="-"), Line2D([], [], color=MUTED, ls="--")]
    l += ["cercle (64 capteurs)", "cercle + grille (Prony : cercle seul)"]
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))


def fig_synthese_nsources_temps(rc, rg, fdir, nom):
    """Erreur, N̂ = N et temps en fonction de N : cercle (plein) / grille (pointillés)."""
    fig, axes = plt.subplots(1, 3, figsize=(LARGEUR, 2.9))
    for rows, ls in ((rc, "-"), (rg, "--")):
        if not rows:
            continue
        for m in METHODES:
            vals = [[stats(rows, m, N, "N", [N], champ=c) for N in NS] for c in ("E", "temps")]
            medE = np.array([v[0][0] for v in vals[0]])
            pex = np.array([v[3][0] for v in vals[0]])
            medT = np.array([v[0][0] for v in vals[1]])
            kw = dict(color=COULEUR[m], marker=MARQUEUR[m], ms=4, lw=1.4, mec="white", mew=0.5)
            axes[0].plot(NS, np.maximum(medE, E_MIN), ls, **kw)
            if m != "descente":
                axes[1].plot(NS, pex, ls, **kw)
            axes[2].plot(NS, medT, ls, **kw)
    axes[0].set_yscale("log"); axes[0].set_ylabel("erreur E (médiane)")
    axes[1].set_ylim(-0.05, 1.05); axes[1].set_ylabel("proportion N̂ = N")
    axes[2].set_yscale("log"); axes[2].set_ylabel("temps médian (s)")
    for ax in axes:
        ax.set_xlabel("nombre de sources N")
        ax.set_xticks(NS)
        style(ax)
    _legende_synthese(fig)
    fig.suptitle("Cadre par défaut (I_c = 50 S, σ = 10⁻³) : cercle et cercle + grille", fontsize=10)
    fig.tight_layout(rect=(0, 0.16, 1, 0.97))
    sauver(fig, fdir, nom)


def toutes(rdir: Path, fdir: Path, methodes=None):
    """Toutes les figures à partir des CSV de rdir (seulement les méthodes ``methodes``)."""
    global METHODES
    METHODES[:] = [m for m in TOUTES if methodes is None or m in methodes]
    R = {(b, c): lire(rdir / f"{b}_{c}.csv") for b in ("bruit", "centrale", "nsources", "temps")
         for c in ("cercle", "grille")}
    titres = {"cercle": "64 capteurs sur le cercle", "grille": "64 capteurs sur le cercle + grille 16 × 32 dans D"}
    for c in ("cercle", "grille"):
        if R[("bruit", c)]:
            fig_balayage(R[("bruit", c)], c, "sigma", "bruit relatif σ", True,
                         f"Balayage en bruit (I_c = 50 S) — {titres[c]}", fdir, f"bruit_{c}")
        if R[("centrale", c)]:
            fig_balayage(R[("centrale", c)], c, "ic_ratio", "I_c / S", True,
                         f"Balayage en source centrale (σ = 10⁻³) — {titres[c]}", fdir,
                         f"centrale_{c}")
        if R[("nsources", c)]:
            fig_nsources(R[("nsources", c)], c, fdir, f"nsources_{c}",
                         f"Nombre de sources (I_c = 50 S, σ = 10⁻³) — {titres[c]}")
        if R[("temps", c)]:
            fig_temps(R[("temps", c)], c, fdir, f"temps_{c}", f"Temps de calcul — {titres[c]}")
    if R[("bruit", "cercle")] and R[("bruit", "grille")]:
        fig_synthese_balayage(R[("bruit", "cercle")], R[("bruit", "grille")], "sigma",
                              "bruit relatif σ", True,
                              "Bruit : cercle (traits pleins) et cercle + grille (pointillés)",
                              fdir, "synthese_bruit")
        fig_synthese_balayage(R[("centrale", "cercle")], R[("centrale", "grille")], "ic_ratio",
                              "I_c / S", True,
                              "Source centrale : cercle (traits pleins) et cercle + grille (pointillés)",
                              fdir, "synthese_centrale")
        fig_synthese_nsources_temps(R[("nsources", "cercle")], R[("nsources", "grille")], fdir,
                                    "synthese_nsources_temps")
    print(f"  figures écrites dans {fdir}")

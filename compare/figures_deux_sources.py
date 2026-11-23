"""
Figures de l'analyse à 2 sources (à partir de results/deux_sources/*.csv ;
la carte de la perte et la trajectoire sont recalculées, en quelques secondes).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from compare import protocol as P  # noqa: E402
from compare.figures import AXIS, COULEUR, GRID, INK, LARGEUR, MARQUEUR, MUTED, sauver, style  # noqa: E402

XS = [1, 3, 10, 30, 100, 300]
METH = ["descente", "gn", "ml_bic", "fw", "prony"]
NOM = {"descente": "Descente de gradient (N connu)", "gn": "Gauss–Newton (N connu, même départ)",
       "ml_bic": "ML + BIC", "fw": "Frank–Wolfe glissant", "prony": "Prony"}
COUL = {**COULEUR, "gn": "#6250d6"}
MARQ = {**MARQUEUR, "gn": "v"}
CAT = ["réussite", "source éteinte", "budget épuisé", "limite statistique", "bloquée au bord",
       "minimum local"]
CAT_COUL = {"réussite": "#1baf7a", "source éteinte": "#e87ba4", "budget épuisé": "#eda100",
            "limite statistique": "#a9a8a0", "bloquée au bord": "#6250d6", "minimum local": "#e34948"}
ORANO = {1: 0.95, 3: 0.85, 10: 0.55, 30: 0.20, 100: 0.05}     # synthèse Orano
CADRE_TXT = {"orano": "cas test d'Orano (σ = 0, sans source centrale)",
             "projet": "cadre du projet (σ = 10⁻³, I_c = 50 S)"}


def lire(chemin: Path):
    with open(chemin, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v in ("True", "False"):
                r[k] = v == "True"
            else:
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def taux(rows, champ):
    return float(np.mean([bool(r[champ]) for r in rows])) if rows else np.nan


# ---- 1. taux de réussite en fonction du rapport d'intensité --------------------------------
def fig_reussite(R, fdir):
    panneaux = [("orano", "cercle"), ("projet", "cercle"), ("projet", "grille")]
    titres = ["Cas test d'Orano\n(σ = 0, sans source centrale)", "Cadre du projet, cercle\n(σ = 10⁻³, I_c = 50 S)",
              "Cadre du projet, cercle + grille"]
    fig, axes = plt.subplots(1, 3, figsize=(LARGEUR, 3.1), sharey=True)
    for ax, (c, cfg), ti in zip(axes, panneaux, titres):
        for m in METH:
            y = [taux(sel(R, cadre=c, config=cfg, x=float(x)), f"ok_{m}") for x in XS]
            lab = NOM[m] + (" (cercle seul)" if (m == "prony" and cfg == "grille") else "")
            ax.plot(XS, y, "-", color=COUL[m], marker=MARQ[m], ms=4.5, lw=1.6, mec="white", mew=0.6,
                    label=lab)
        if c == "orano":
            ax.plot(list(ORANO), list(ORANO.values()), "x", color=INK, ms=7, mew=1.6,
                    label="synthèse Orano (descente)")
        ax.set_xscale("log")
        ax.set_xticks(XS, [str(x) for x in XS])
        ax.minorticks_off()
        ax.set_xlabel("rapport d'intensité x = I₁ / I₂")
        ax.set_title(ti, fontsize=9)
        ax.set_ylim(-0.04, 1.04)
        style(ax)
    axes[0].set_ylabel("réussite (N̂ = 2, chaque source à < 0,1)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    sauver(fig, fdir, "ds_reussite_rapport")


# ---- 2. causes des échecs de la descente ----------------------------------------------------
def fig_causes(R, fdir):
    panneaux = [("orano", "cercle"), ("projet", "cercle"), ("projet", "grille")]
    titres = ["Cas test d'Orano", "Projet, cercle", "Projet, cercle + grille"]
    fig, axes = plt.subplots(1, 3, figsize=(LARGEUR, 3.0), sharey=True)
    pos = np.arange(len(XS))
    for ax, (c, cfg), ti in zip(axes, panneaux, titres):
        bas = np.zeros(len(XS))
        for k in CAT:
            v = np.array([np.mean([r["cat_descente"] == k for r in sel(R, cadre=c, config=cfg, x=float(x))])
                          for x in XS])
            ax.bar(pos, v, bottom=bas, color=CAT_COUL[k], width=0.78, edgecolor="white", lw=0.8,
                   label=k)
            bas += v
        ax.set_xticks(pos, [str(x) for x in XS])
        ax.set_xlabel("rapport d'intensité x")
        ax.set_title(ti, fontsize=9)
        style(ax)
        ax.grid(False)
    axes[0].set_ylabel("proportion des tirages")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("Descente de gradient à 2 sources : issue de chaque tirage", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    sauver(fig, fdir, "ds_causes")


# ---- 3. conditionnement ---------------------------------------------------------------------
def fig_conditionnement(R, fdir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(LARGEUR, 3.1))
    rows = sel(R, cadre="orano", config="cercle")
    med = [np.median([r["kappa"] for r in sel(rows, x=float(x))]) for x in XS]
    q1 = [np.percentile([r["kappa"] for r in sel(rows, x=float(x))], 25) for x in XS]
    q3 = [np.percentile([r["kappa"] for r in sel(rows, x=float(x))], 75) for x in XS]
    a.fill_between(XS, q1, q3, color=COUL["descente"], alpha=0.18, lw=0)
    a.plot(XS, med, "-", color=COUL["descente"], marker="D", ms=4.5, lw=1.6, mec="white",
           label="κ(JᵀJ) à la vérité, médiane et Q1–Q3")
    pente = np.polyfit(np.log(XS[2:]), np.log(med[2:]), 1)[0]
    ref = med[2] * (np.array(XS, float) / XS[2]) ** 2
    a.plot(XS, ref, "--", color=MUTED, lw=1.2, label="pente 2 (κ ∝ x²)")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(XS, [str(x) for x in XS]); a.minorticks_off()
    a.set_xlabel("rapport d'intensité x")
    a.set_ylabel("conditionnement κ")
    a.set_title(f"Conditionnement (pente mesurée {pente:.2f} pour".replace(".", ",") + f" x ≥ 10)", fontsize=9)
    a.legend(frameon=False, fontsize=7.5)
    style(a)
    for k in CAT:
        v = [r for r in rows if r["cat_descente"] == k]
        if v:
            b.scatter([r["kappa"] for r in v], [r["it_descente"] for r in v], s=12, color=CAT_COUL[k],
                      label=k, alpha=0.85, lw=0)
    b.axhline(20000, color=MUTED, ls=":", lw=1)
    b.text(0.02, 21500, "budget 20 000", color=MUTED, fontsize=7.5,
           transform=b.get_yaxis_transform())
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("conditionnement κ à la vérité")
    b.set_ylabel("itérations de la descente")
    b.set_title("Itérations de la descente (cas test d'Orano)", fontsize=9)
    b.legend(frameon=False, fontsize=7, loc="lower right")
    style(b)
    fig.tight_layout()
    sauver(fig, fdir, "ds_conditionnement")
    return pente


# ---- 4. paysage de la perte et trajectoire ---------------------------------------------------
def fig_paysage(R, fdir, cadre="orano", x=30):
    from compare.deux_sources import CADRES, choisir_situation, situation_x
    from sparse_sources.orano import Probleme, descente_projetee
    t, cat = choisir_situation(R, cadre, float(x))
    sig, ic = CADRES[cadre]
    fig, axes = plt.subplots(1, 3, figsize=(LARGEUR, 4.3),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    xs = np.linspace(0, 10, 101)
    ys = np.linspace(0, 20, 201)
    G = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)
    infos = {}
    for ax, cfg in zip(axes[:2], ("cercle", "grille")):
        sit = situation_x(x, t, sig, ic, cfg)
        pb = Probleme(sit.X, P.LO, P.HI, sit.m)
        # perte en fonction de la position de la source FAIBLE (indice 1), le reste à la vérité
        L = np.empty(len(G))
        for k, g in enumerate(G):
            u, a = pb.normalise(np.array([sit.z[0], g]), sit.I)
            L[k] = pb.perte(u, a)
        L = L.reshape(len(ys), len(xs))
        im = ax.pcolormesh(xs, ys, np.log10(np.maximum(L, 1e-12)), cmap="Greys_r", shading="auto",
                           rasterized=True)
        d = descente_projetee(pb, sit.u0, np.full(2, 0.5), iter_max=20000, trajectoire=True)
        tr = d["trajectoire"]
        for i, c in enumerate((COUL["descente"], COUL["prony"])):
            ax.plot(tr[:, i, 0], tr[:, i, 1], "-", color=c, lw=1.4)
            ax.plot(*tr[0, i], "o", mfc="white", mec=c, ms=6)
            ax.plot(*tr[-1, i], "o", color=c, ms=6, mec="white")
        ax.plot(*sit.z.T, "*", color="#e34948", ms=13, mec="white", mew=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(0, 10); ax.set_ylim(0, 20)
        ax.set_title(f"{'cercle' if cfg == 'cercle' else 'cercle + grille'} : {d['iterations']} itér.",
                     fontsize=9)
        cb = fig.colorbar(im, ax=ax, fraction=0.07, pad=0.03)
        cb.set_label("log₁₀ perte (source faible déplacée)", fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)
        infos[cfg] = (sit, d)
    a = axes[2]
    for cfg, ls in (("cercle", "-"), ("grille", "--")):
        sit, d = infos[cfg]
        tr = d["trajectoire"]
        dist = np.linalg.norm(tr[:, :, None, :] - sit.z[None, None], axis=-1)   # (it, est, vraie)
        it = np.arange(len(tr)) + 1
        for j, c in enumerate((COUL["descente"], COUL["prony"])):
            a.plot(it, dist[:, :, j].min(1), ls, color=c, lw=1.4,
                   label=f"source {'forte' if j == 0 else 'faible'}, {cfg}")
    a.axhline(0.1, color=MUTED, ls=":", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("itération")
    a.set_ylabel("distance à la vraie source la plus proche")
    a.set_title("Convergence des positions", fontsize=9)
    a.legend(frameon=False, fontsize=7, loc="lower left")
    style(a)
    fig.suptitle(f"Perte d'Orano et trajectoire de la descente — 2 sources, x = {x}, tirage {t}\n"
                 f"{CADRE_TXT[cadre]} ; ronds blancs : départ, ronds pleins : arrivée, "
                 f"étoiles : vraies sources", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    sauver(fig, fdir, "ds_paysage")
    return t


# ---- 5. bassins d'attraction ------------------------------------------------------------------
def fig_bassins(B, fdir, rdir):
    from compare.deux_sources import CADRES, situation_x
    with open(rdir / "bassins_situations.json", encoding="utf-8") as f:
        sits = json.load(f)
    fig, axes = plt.subplots(1, len(sits), figsize=(LARGEUR, 4.4))
    cmap = ListedColormap([CAT_COUL[k] for k in CAT])
    for ax, s in zip(axes, sits):
        rows = sel(B, cadre=s["cadre"], x=float(s["x"]), tirage=float(s["tirage"]))
        ux = sorted({r["u1x"] for r in rows}); uy = sorted({r["u1y"] for r in rows})
        Z = np.full((len(uy), len(ux)), np.nan)
        OK = np.zeros_like(Z)
        for r in rows:
            Z[uy.index(r["u1y"]), ux.index(r["u1x"])] = CAT.index(r["cat_descente"])
            OK[uy.index(r["u1y"]), ux.index(r["u1x"])] = r["ok_gn"]
        xs = np.array(ux) * 10; ys = np.array(uy) * 20
        ax.pcolormesh(xs, ys, Z, cmap=cmap, vmin=-0.5, vmax=len(CAT) - 0.5, shading="nearest",
                      rasterized=True)
        gx, gy = np.meshgrid(xs, ys)
        ax.scatter(gx[OK == 0], gy[OK == 0], s=5, marker="x", color=INK, lw=0.7)
        sig, ic = CADRES[s["cadre"]]
        sit = situation_x(s["x"], s["tirage"], sig, ic, "cercle")
        ax.plot(*sit.z.T, "*", color="white", ms=13, mec=INK, mew=0.8)
        ax.plot(*(sit.u0[1] * (P.HI - P.LO)), "s", color="white", ms=6, mec=INK)
        frac = np.nanmean(Z == 0)
        ax.set_title(f"{'Orano' if s['cadre'] == 'orano' else 'projet'}, x = {s['x']:g}, "
                     f"tirage {s['tirage']}\ndescente {100 * frac:.0f} %, "
                     f"Gauss–Newton {100 * OK.mean():.0f} %", fontsize=8)
        ax.set_aspect("equal"); ax.set_xlim(0, 10); ax.set_ylim(0, 20)
        ax.tick_params(labelsize=7)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    h = [Patch(color=CAT_COUL[k], label=k) for k in CAT]
    h += [Line2D([], [], marker="x", ls="", color=INK, label="Gauss–Newton échoue"),
          Line2D([], [], marker="*", ls="", color="white", mec=INK, ms=10, label="vraies sources"),
          Line2D([], [], marker="s", ls="", color="white", mec=INK, label="départ fixe de la source 2")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.suptitle("Bassins d'attraction : issue de la descente selon la position de départ de la "
                 "source 1 (case colorée)", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    sauver(fig, fdir, "ds_bassins")


# ---- 6. séparation ---------------------------------------------------------------------------
def fig_separation(R, fdir):
    bins = [0, 2, 4, 8, 23]
    fig, axes = plt.subplots(1, 2, figsize=(LARGEUR, 2.9), sharey=True)
    for ax, (c, x) in zip(axes, (("orano", 1.0), ("orano", 30.0))):
        rows = sel(R, cadre=c, config="cercle", x=x)
        cen = []
        for m in ("descente", "gn", "ml_bic", "fw", "prony"):
            y = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                v = [r for r in rows if lo <= r["separation"] < hi]
                y.append(taux(v, f"ok_{m}") if len(v) >= 5 else np.nan)
            ax.plot(range(len(y)), y, "-", color=COUL[m], marker=MARQ[m], ms=4.5, lw=1.6, mec="white",
                    label=NOM[m])
        n = [len([r for r in rows if lo <= r["separation"] < hi]) for lo, hi in zip(bins[:-1], bins[1:])]
        ax.set_xticks(range(len(n)), [f"[{lo}, {hi})\nn = {k}" for lo, hi, k in zip(bins[:-1], bins[1:], n)],
                      fontsize=7.5)
        ax.set_xlabel("distance entre les deux sources")
        ax.set_title(f"Cas test d'Orano, x = {x:g}", fontsize=9)
        ax.set_ylim(-0.04, 1.04)
        style(ax)
    axes[0].set_ylabel("réussite")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    sauver(fig, fdir, "ds_separation")


# ---- 7. temps ------------------------------------------------------------------------------
def fig_temps(R, fdir):
    fig, axes = plt.subplots(1, 2, figsize=(LARGEUR, 2.9), sharey=True)
    for ax, (c, cfg) in zip(axes, (("orano", "cercle"), ("projet", "cercle"))):
        rows = sel(R, cadre=c, config=cfg)
        for m in METH:
            y = [np.median([r[f"t_{m}"] for r in sel(rows, x=float(x))]) for x in XS]
            ax.plot(XS, y, "-", color=COUL[m], marker=MARQ[m], ms=4.5, lw=1.6, mec="white", label=NOM[m])
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(XS, [str(x) for x in XS]); ax.minorticks_off()
        ax.set_xlabel("rapport d'intensité x")
        ax.set_title(CADRE_TXT[c], fontsize=8.5)
        style(ax)
    axes[0].set_ylabel("temps médian (s)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    sauver(fig, fdir, "ds_temps")


def toutes(rdir: Path, fdir: Path):
    fdir.mkdir(parents=True, exist_ok=True)
    out = {}
    if (rdir / "rapport.csv").exists():
        R = lire(rdir / "rapport.csv")
        fig_reussite(R, fdir)
        fig_causes(R, fdir)
        out["pente"] = fig_conditionnement(R, fdir)
        fig_separation(R, fdir)
        fig_temps(R, fdir)
        out["tirage_paysage"] = fig_paysage(R, fdir)
    if (rdir / "bassins.csv").exists():
        fig_bassins(lire(rdir / "bassins.csv"), fdir, rdir)
    print(f"  figures 2 sources dans {fdir}", out)
    return out

"""
Scénario du projet avec la grosse source centrale : comparaison des méthodes
et limite de Cramér-Rao.

    python sparse_sources/experiments_projet.py            (~15 min)
    python sparse_sources/experiments_projet.py --rapide
    python sparse_sources/experiments_projet.py --seulement methodes centrale

Données : exactement celles de scripts/reconstruct.py. Les sources sont tirées
selon config/ (nombre fixé à n, intensités uniformes sur le simplexe de somme
100, positions uniformes dans D). On ajoute la source centrale (intensité
50 × 100 = 5000, au centre de D, donc au centre du cercle des capteurs), puis
le bruit relatif y (1 + σ ε). La source centrale est ensuite retirée
(``subtract_known_sources``). Son bruit, lui, reste : s_i = σ y_i ≈ 40 σ par capteur.

Méthodes comparées, toutes sur les mêmes mesures :
    prony         cfg.invert, comme reconstruct.py (rank_tol de la config)
    sfw           Frank-Wolfe glissant, BLASSO pondéré α = 1, λ = 1,5 σ ||y||
                  (meilleure méthode de la première étude)
    ml_bic        maximum de vraisemblance glouton, bruit blanchi, K par BIC
                  (``greedy_ml``)
    ml_bic_multi  idem avec 3 candidats par ajout et un départ supplémentaire
                  depuis Prony ; on garde le meilleur critère

Expériences :
    methodes   n ∈ {1, 2, 3, 5}, σ ∈ {1e-4 … 1e-2} : réussite, erreur, temps,
               et borne de Cramér-Rao (CRB) de chaque tirage
    critere    BIC contre AIC (choix de K)
    centrale   effet de l'intensité de la source centrale (facteur 0 à 500)
    defaut     configuration par défaut du projet (N ~ Poisson(20), σ = 1 %) :
               combien de sources sont détectables ?

Sorties : sparse_sources/results/projet_*.csv et projet_*.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particle_sources.config import load_config  # noqa: E402
from particle_sources.config.cli import DEFAULT_CONFIG_DIR  # noqa: E402
from particle_sources.generation import sample_sources  # noqa: E402
from particle_sources.inversion import subtract_known_sources  # noqa: E402
from particle_sources.sensors import measure  # noqa: E402
from sparse_sources import (build_grid, cramer_rao, greedy_ml, make_setup,  # noqa: E402
                            match_sources, relative_noise_std, sliding_frank_wolfe)
from sparse_sources.experiments import (OUT, min_separation, save, success,  # noqa: E402
                                        summarize, write_csv)
from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

SETUP = make_setup(h=0.0, normalize=True)
GRID = build_grid(step=0.25, setup=SETUP)
SIGMAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
METHODS = ["prony", "sfw", "ml_bic", "ml_bic_multi"]


def scenario(n, sigma, seed, central_factor=None, count="fixed"):
    """Tirage du projet. Renvoie (cfg, sources, y brut, y sans la source centrale)."""
    over = [f"seed={seed}", f"sensors.noise.params.sigma={sigma}"]
    if count == "fixed":
        over += ["sources.count_law.name=fixed", f"sources.count_law.params={{n: {n}}}"]
    if central_factor is not None:
        over += ([f"central_source.intensity_factor={central_factor}"] if central_factor
                 else ["central_source.enabled=false"])
    cfg = load_config(DEFAULT_CONFIG_DIR, over)
    rng = cfg.rng()
    src = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                         cfg.intensity_law, cfg.position_law)
    y = measure(src + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)
    y_inv = subtract_known_sources(y, cfg.fixed_sources, cfg.sensors, cfg.eps)
    return cfg, src, y, y_inv


def run_all(cfg, y, m, sigma, methods=METHODS):
    """Lance les méthodes ; renvoie {méthode: (z, I, temps)}."""
    s = relative_noise_std(y, sigma)
    out = {}
    if "prony" in methods or "ml_bic_multi" in methods:
        t0 = time.perf_counter()
        est = cfg.invert(m, cfg.sensors)
        out["prony"] = (est.positions, est.intensities, time.perf_counter() - t0)
    if "sfw" in methods:
        fw = sliding_frank_wolfe(m, 1.5 * sigma * float(np.linalg.norm(y)), setup=SETUP,
                                 grid=GRID, tol=1e-5, max_iter=10)
        z = fw["z_debiased"]
        out["sfw"] = (z, SETUP.physical_intensities(z, fw["I_debiased"]), fw["time"])
    if "ml_bic" in methods:
        r = greedy_ml(m, s, setup=SETUP, grid=GRID)
        out["ml_bic"] = (r["z"], r["I"], r["time"])
    if "ml_aic" in methods:
        r = greedy_ml(m, s, setup=SETUP, grid=GRID, criterion="aic")
        out["ml_aic"] = (r["z"], r["I"], r["time"])
    if "ml_bic_multi" in methods:
        a = greedy_ml(m, s, setup=SETUP, grid=GRID, n_candidates=3)
        zp, Ip, tp = out["prony"]
        b = greedy_ml(m, s, setup=SETUP, grid=GRID, z_init=zp, I_init=Ip)
        best = min((a, b), key=lambda r: r["criterion"])
        out["ml_bic_multi"] = (best["z"], best["I"], a["time"] + b["time"] + tp)
    return {k: v for k, v in out.items() if k in methods}


def record(rows, base, src, results, crb):
    """Ajoute une ligne par méthode."""
    sep = min_separation(src.positions)
    tol = 0.25 * min(sep, 4.0)
    for meth, (zh, Ih, t) in results.items():
        res = match_sources(src.positions, src.intensities, zh, Ih)
        rows.append({**base, "methode": meth, "K": src.n, "n_est": res["n_est"],
                     "err_pos": res["err_pos"], "err_int": res["err_int"], "temps": t,
                     "sep_min": sep, "crb_max": crb, "faisable": bool(2 * crb < tol),
                     "reussite": success(res, src.positions)})


# ---- Expériences ------------------------------------------------------------------------
def exp_methods(n_seeds):
    """Réussite, erreur, temps et CRB selon n et σ."""
    print("\n[1] Méthodes sur le scénario du projet")
    rows = []
    for n in [1, 2, 3, 5]:
        for sigma in SIGMAS:
            for seed in range(n_seeds):
                cfg, src, y, m = scenario(n, sigma, seed)
                crb = float(cramer_rao(src.positions, src.intensities,
                                       relative_noise_std(y, sigma), SETUP)["pos_std"].max())
                record(rows, {"n": n, "sigma": sigma, "seed": seed}, src,
                       run_all(cfg, y, m, sigma), crb)
            print(f"  n = {n}, σ = {sigma:g} fait")
    write_csv(rows, OUT / "projet_methodes.csv")
    summ = summarize(rows, ["n", "sigma", "methode"],
                     ["n_est", "err_pos", "err_int", "temps", "crb_max"])
    for row in summ:   # réussite parmi les tirages « faisables » (2 × CRB < tolérance)
        sel = [r for r in rows if (r["n"], r["sigma"], r["methode"]) ==
               (row["n"], row["sigma"], row["methode"]) and r["faisable"]]
        row["part_faisable"] = len(sel) / row["n_tirages"]
        row["reussite_si_faisable"] = float(np.mean([r["reussite"] for r in sel])) if sel else np.nan
    write_csv(summ, OUT / "projet_methodes_resume.csv")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for j, n in enumerate([1, 2, 3, 5]):
        for k, meth in enumerate(METHODS):
            g = [[r for r in rows if r["n"] == n and r["sigma"] == s and r["methode"] == meth]
                 for s in SIGMAS]
            axes[0, j].plot(SIGMAS, [np.mean([r["reussite"] for r in x]) for x in g], "-o",
                            color=SERIES[k], lw=2, ms=5, label=meth)
            axes[1, j].plot(SIGMAS, [np.nanmedian([r["err_pos"] for r in x]) for x in g], "-o",
                            color=SERIES[k], lw=2, ms=5, label=meth)
        g = [[r for r in rows if r["n"] == n and r["sigma"] == s and r["methode"] == METHODS[0]]
             for s in SIGMAS]
        axes[0, j].plot(SIGMAS, [np.mean([r["faisable"] for r in x]) for x in g], "--",
                        color=INK_MUTED, lw=1.5, label="2 CRB < tolérance")
        axes[1, j].plot(SIGMAS, [np.median([r["crb_max"] for r in x]) for x in g], "--",
                        color=INK_MUTED, lw=1.5, label="borne de Cramér-Rao")
        axes[0, j].set_title(f"{n} source{'s' if n > 1 else ''}", fontsize=10)
        axes[0, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("σ (bruit relatif)")
        for i in range(2):
            axes[i, j].set_xscale("log")
            style_axes(axes[i, j])
    axes[0, 0].set_ylabel("taux de réussite")
    axes[1, 0].set_ylabel("erreur de position (médiane)")
    h, l = axes[1, 3].get_legend_handles_labels()
    h0, l0 = axes[0, 3].get_legend_handles_labels()
    axes[0, 3].legend(h + h0[-1:], l + l0[-1:], fontsize=8, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Scénario du projet (source centrale 5000 retirée, bruit relatif)", fontsize=11)
    fig.tight_layout()
    save(fig, "projet_methodes.png")
    return rows


def exp_criterion(n_seeds):
    """BIC contre AIC pour le choix du nombre de sources."""
    print("\n[2] Critère de choix de K : BIC / AIC")
    rows = []
    for n in [2, 3, 5]:
        for sigma in [1e-4, 1e-3]:
            for seed in range(n_seeds):
                cfg, src, y, m = scenario(n, sigma, seed)
                record(rows, {"n": n, "sigma": sigma, "seed": seed}, src,
                       run_all(cfg, y, m, sigma, ("ml_bic", "ml_aic")), np.nan)
    summ = summarize(rows, ["n", "sigma", "methode"], ["n_est", "err_pos"])
    for row in summ:
        sel = [r for r in rows if (r["n"], r["sigma"], r["methode"]) ==
               (row["n"], row["sigma"], row["methode"])]
        row["trop"] = float(np.mean([r["n_est"] > r["K"] for r in sel]))
        row["trop_peu"] = float(np.mean([r["n_est"] < r["K"] for r in sel]))
    write_csv(summ, OUT / "projet_critere.csv")
    for r in summ:
        print(f"  n={r['n']} σ={r['sigma']:g} {r['methode']:7s} réussite {r['taux_reussite']:.2f}"
              f"  trop {r['trop']:.2f}  trop peu {r['trop_peu']:.2f}")


def exp_central(n_seeds, n=3, sigma=1e-3):
    """Effet de l'intensité de la source centrale (son bruit reste après soustraction)."""
    print("\n[3] Intensité de la source centrale")
    rows = []
    factors = [0, 5, 50, 500]
    for f in factors:
        for seed in range(n_seeds):
            cfg, src, y, m = scenario(n, sigma, seed, central_factor=f)
            crb = float(cramer_rao(src.positions, src.intensities,
                                   relative_noise_std(y, sigma), SETUP)["pos_std"].max())
            record(rows, {"facteur": f, "seed": seed}, src,
                   run_all(cfg, y, m, sigma), crb)
        print(f"  facteur {f} fait")
    write_csv(rows, OUT / "projet_centrale.csv")
    summ = summarize(rows, ["facteur", "methode"], ["n_est", "err_pos", "crb_max"])
    write_csv(summ, OUT / "projet_centrale_resume.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = [max(f, 0.5) for f in factors]
    for k, meth in enumerate(METHODS):
        g = [[r for r in rows if r["facteur"] == f and r["methode"] == meth] for f in factors]
        axes[0].plot(xs, [np.mean([r["reussite"] for r in x]) for x in g], "-o",
                     color=SERIES[k], lw=2, ms=5, label=meth)
        axes[1].plot(xs, [np.nanmedian([r["err_pos"] for r in x]) for x in g], "-o",
                     color=SERIES[k], lw=2, ms=5, label=meth)
    g = [[r for r in rows if r["facteur"] == f and r["methode"] == METHODS[0]] for f in factors]
    axes[1].plot(xs, [np.median([r["crb_max"] for r in x]) for x in g], "--", color=INK_MUTED,
                 lw=1.5, label="borne de Cramér-Rao")
    axes[0].set_ylabel("taux de réussite")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("erreur de position (médiane)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(xs, ["0", "5", "50", "500"])
        ax.set_xlabel("intensité centrale / intensité totale des sources")
        style_axes(ax)
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle(f"{n} sources, σ = {sigma:g} : le bruit de la source centrale limite tout",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "projet_centrale.png")


def exp_default(n_seeds):
    """Configuration par défaut : N ~ Poisson(20), σ = 1 %."""
    print("\n[4] Configuration par défaut du projet (Poisson(20), σ = 1 %)")
    rows = []
    for seed in range(n_seeds):
        cfg, src, y, m = scenario(None, 1e-2, seed, count="config")
        s = relative_noise_std(y, 1e-2)
        crb = cramer_rao(src.positions, src.intensities, s, SETUP)["pos_std"]
        # sources « détectables » seules : CRB d'une source isolée < 0,5
        alone = np.array([cramer_rao(z[None], [I], s, SETUP)["pos_std"][0]
                          for z, I in zip(src.positions, src.intensities)])
        res = run_all(cfg, y, m, 1e-2)
        row = {"seed": seed, "K": src.n, "crb_conjointe_mediane": float(np.median(crb)),
               "detectables_seules": int(np.sum(alone < 0.5)),
               "I_max": float(src.intensities.max())}
        for meth, (zh, Ih, t) in res.items():
            row[f"n_est_{meth}"] = len(Ih)
        rows.append(row)
        print("  " + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                               for k, v in row.items()))
    write_csv(rows, OUT / "projet_defaut.csv")


def main():
    """Lance les expériences demandées."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rapide", action="store_true", help="moins de tirages")
    p.add_argument("--seulement", nargs="+",
                   choices=["methodes", "critere", "centrale", "defaut"])
    args = p.parse_args()
    n = 4 if args.rapide else 20
    todo = set(args.seulement or ["methodes", "critere", "centrale", "defaut"])
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    if "methodes" in todo:
        exp_methods(n)
    if "critere" in todo:
        exp_criterion(n)
    if "centrale" in todo:
        exp_central(n)
    if "defaut" in todo:
        exp_default(max(n // 2, 3))
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s. Résultats : {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

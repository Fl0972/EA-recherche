"""
Expériences numériques de sparse_sources.

    python sparse_sources/experiments.py            (tout, ~15 min)
    python sparse_sources/experiments.py --rapide   (moins de tirages, ~5 min)
    python sparse_sources/experiments.py --seulement hypotheses lambda

Géométries (même domaine D = [0,10]×[0,20] et 64 capteurs que config/) :
    plan_brut       capteurs dans le plan (h = 0), noyau brut 1/r² : le projet
    plan_normalise  même géométrie, noyau normalisé φ/||φ|| (BLASSO pondéré)
    hauteur_20      capteurs à h = 20 au-dessus du plan, noyau brut

Configurations de sources :
    separees        3 sources bien séparées
    proches         2 sources à distance 1
    loin            2 sources séparées, capteurs sur un cercle 2× plus grand

Bruit additif gaussien m = A(s*) + σ ε, σ = niveau × rms(A(s*)) ; λ = c σ √M.

Expériences :
    hypotheses   check_hypotheses pour chaque (géométrie, configuration) + cartes de η_V
    lambda       nombre de sources trouvées en fonction de c (λ = c σ √M)
    methodes     LASSO (sans / avec raffinement) vs Frank-Wolfe glissant
    sigma        erreur de position en fonction de σ (pente 1 attendue)
    lien         hypothèses (marge, conditionnement) vs taux de réussite
    prony        comparaison avec scripts/reconstruct.py (Prony) sur données
                 bruitées : scénario du projet (bruit relatif, source centrale
                 retirée), sources tirées selon la config

Sorties : sparse_sources/results/ (PNG + CSV).
"""

from __future__ import annotations

import argparse
import csv
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
from particle_sources.inversion import prony_circle, subtract_known_sources  # noqa: E402
from particle_sources.sensors import measure  # noqa: E402
from sparse_sources import (build_grid, check_hypotheses, forward, lasso_pipeline,  # noqa: E402
                            make_setup, match_sources, refine, sliding_frank_wolfe)
from sparse_sources.common import debias, max_certificate  # noqa: E402
from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
GRID_STEP = 0.25

CONFIGS = {
    "separees": (np.array([[3.0, 5.0], [7.0, 14.0], [5.0, 9.5]]), np.array([12.0, 8.0, 10.0])),
    "proches": (np.array([[5.0, 9.5], [5.0, 10.5]]), np.array([10.0, 10.0])),
    "loin": (np.array([[3.0, 7.0], [7.0, 13.0]]), np.array([10.0, 15.0])),
}
GEOMETRIES = {
    "plan_brut": dict(h=0.0, normalize=False),
    "plan_normalise": dict(h=0.0, normalize=True),
    "hauteur_20": dict(h=20.0, normalize=False),
}


# ---- Outils -----------------------------------------------------------------------------
_SETUPS, _GRIDS = {}, {}


def get_setup(geom: str, config: str):
    """Setup (et grille) d'une géométrie ; « loin » = capteurs sur un cercle 2× plus grand."""
    key = (geom, config == "loin")
    if key not in _SETUPS:
        _SETUPS[key] = make_setup(radius_factor=2.0 if config == "loin" else 1.0,
                                  **GEOMETRIES[geom])
        _GRIDS[key] = build_grid(step=GRID_STEP, setup=_SETUPS[key])
    return _SETUPS[key], _GRIDS[key]


def simulate(setup, z, I, level, rng):
    """m = A(s*) + σ ε avec σ = level × rms(A(s*)) (noyau physique, non normalisé)."""
    m0 = forward(z, I, setup.with_normalization(False))
    sigma = level * float(np.sqrt(np.mean(m0**2)))
    return m0 + sigma * rng.standard_normal(m0.shape), sigma


def min_separation(z) -> float:
    """Plus petite distance entre deux sources (inf si une seule)."""
    if len(z) < 2:
        return np.inf
    d = np.linalg.norm(z[:, None] - z[None], axis=-1)
    return float(d[np.triu_indices(len(z), 1)].min())


def success(res: dict, z_true) -> bool:
    """Bon nombre de sources et erreur de position < 0.25 × min(séparation, 4)."""
    return bool(res["exact"] and res["err_pos"] < 0.25 * min(min_separation(z_true), 4.0))


def run_methods(setup, grid, m, lam, sigma, methods=("lasso", "lasso_raffine", "sfw")):
    """
    Lance les méthodes demandées. Renvoie {méthode: (z, I physiques, temps, extra)}.
    LASSO et LASSO raffiné partagent la résolution sur grille (le temps du
    raffiné inclut celui du LASSO).
    """
    out = {}
    if "lasso" in methods or "lasso_raffine" in methods:
        t0 = time.perf_counter()
        las = lasso_pipeline(m, lam, grid=grid, setup=setup, refine_positions=False,
                             method="fista", max_iter=5000)
        t_lasso = time.perf_counter() - t0
        out["lasso"] = (las["z"], setup.physical_intensities(las["z"], las["I"]), t_lasso,
                        {"eta_max": las["eta_max"]})
        if "lasso_raffine" in methods:
            t0 = time.perf_counter()
            z, I, _ = refine(las["z_merged"], las["I_merged"], m, lam, setup)
            r = m - (I @ setup.phi(z) if len(I) else 0.0)
            _, eta = max_certificate(r, lam, grid, setup)
            zd, Id = debias(z, I, m, setup)
            out["lasso_raffine"] = (zd, setup.physical_intensities(zd, Id),
                                    t_lasso + time.perf_counter() - t0, {"eta_max": eta})
    if "sfw" in methods:
        # 10 itérations suffisent largement quand la condition tient (≈ K + 2) ;
        # sinon (noyau brut dans le plan) l'algorithme ajoute des Diracs sans
        # fin près des capteurs : on s'arrête et on le note (non certifié).
        sfw = sliding_frank_wolfe(m, lam, setup=setup, grid=grid, tol=1e-5, max_iter=10)
        zd = sfw["z_debiased"]
        out["sfw"] = (zd, setup.physical_intensities(zd, sfw["I_debiased"]), sfw["time"],
                      {"eta_max": sfw["eta_max"], "iter": sfw["n_iter"],
                       "converged": sfw["converged"]})
    return out


def write_csv(rows, path):
    """Écrit une liste de dicts en CSV (union des clés, dans l'ordre d'apparition)."""
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path.relative_to(ROOT)}")


def save(fig, name):
    """Enregistre une figure PNG dans results/."""
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path.relative_to(ROOT)}")


def summarize(rows, keys, fields):
    """Moyennes / médianes par groupe ``keys`` (liste de dicts)."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), []).append(r)
    out = []
    for g, rs in groups.items():
        row = dict(zip(keys, g))
        row["n_tirages"] = len(rs)
        row["taux_reussite"] = float(np.mean([r["reussite"] for r in rs]))
        for f in fields:
            vals = np.array([r[f] for r in rs], float)
            row[f"{f}_median"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
        out.append(row)
    return out


# ---- 1. Hypothèses -----------------------------------------------------------------------
def exp_hypotheses():
    """check_hypotheses pour chaque géométrie et configuration, cartes de η_V."""
    print("\n[1] Hypothèses de Duval-Peyré")
    rows = []
    for geom in GEOMETRIES:
        for name, (z, _) in CONFIGS.items():
            setup, _ = get_setup(geom, name)
            res = check_hypotheses(z, setup=setup, verbose=False, plot=True,
                                   save_path=OUT / f"certificat_{geom}_{name}.png")
            plt.close(res["figure"])
            rows.append({"geometrie": geom, "config": name, "K": res["K"],
                         "H1": res["H1"], "H2": res["H2"], "H3": res["H3"],
                         "non_degenere": res["non_degenerate"], "rang": res["rank"],
                         "cond": res["cond"], "cond_norm": res["cond_normalized"],
                         "marge": res["margin"], "marge_relative": res["margin_relative"],
                         "max_bord": res["boundary_max"],
                         "vp_hess_max": float(res["eigenvalues"].max())})
            print(f"  {geom:15s} {name:9s} H1={res['H1']!s:5} H2={res['H2']!s:5} "
                  f"H3={res['H3']!s:5} marge={res['margin']:.3g} "
                  f"(relative {res['margin_relative']:.3g}) cond={res['cond_normalized']:.3g}")
    write_csv(rows, OUT / "hypotheses.csv")
    return rows


# ---- 2. Balayage de λ ---------------------------------------------------------------------
def exp_lambda(n_seeds):
    """Nombre de sources trouvées en fonction de c, λ = c σ √M (σ = 1e-3 × rms)."""
    print("\n[2] Balayage de λ = c σ √M")
    cs = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    rows = []
    for geom in GEOMETRIES:
        for name, (z, I) in CONFIGS.items():
            setup, grid = get_setup(geom, name)
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                m, sigma = simulate(setup, z, I, 1e-3, rng)
                for c in cs:
                    lam = c * sigma * np.sqrt(setup.M)
                    for meth, (zh, Ih, t, ex) in run_methods(setup, grid, m, lam, sigma,
                                                            ("lasso", "sfw")).items():
                        rows.append({"geometrie": geom, "config": name, "seed": seed, "c": c,
                                     "methode": meth, "K": len(I), "n_est": len(Ih)})
        print(f"  {geom} fait")
    write_csv(rows, OUT / "balayage_lambda.csv")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for j, geom in enumerate(GEOMETRIES):
        for i, meth in enumerate(["sfw", "lasso"]):
            ax = axes[i, j]
            for k, name in enumerate(CONFIGS):
                sel = [r for r in rows if r["geometrie"] == geom and r["config"] == name
                       and r["methode"] == meth]
                y = [np.mean([r["n_est"] - r["K"] for r in sel if r["c"] == c]) for c in cs]
                ax.plot(cs, y, "-o", color=SERIES[k], lw=2, ms=5, label=name)
            ax.axhline(0, color=INK_MUTED, lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_title(f"{'Frank-Wolfe glissant' if meth == 'sfw' else 'LASSO (grille)'} "
                         f"— {geom}", fontsize=10)
            style_axes(ax)
            if i == 1:
                ax.set_xlabel("c   (λ = c σ √M)")
            if j == 0:
                ax.set_ylabel("sources trouvées − K")
    axes[0, 0].legend(fontsize=8, title="configuration", title_fontsize=8)
    fig.suptitle("Nombre de sources en fonction de λ (bruit 1e-3 × rms)", fontsize=11)
    fig.tight_layout()
    save(fig, "balayage_lambda.png")
    return rows


# ---- 3. Comparaison des méthodes --------------------------------------------------------------
def exp_methods(n_seeds, c=2.0):
    """LASSO, LASSO raffiné et Frank-Wolfe glissant : nombre, erreurs, temps."""
    print(f"\n[3] Comparaison des méthodes (c = {c:g})")
    rows = []
    for geom in GEOMETRIES:
        for name, (z, I) in CONFIGS.items():
            setup, grid = get_setup(geom, name)
            for level in [1e-4, 1e-3, 1e-2]:
                for seed in range(n_seeds):
                    rng = np.random.default_rng(100 + seed)
                    m, sigma = simulate(setup, z, I, level, rng)
                    lam = c * sigma * np.sqrt(setup.M)
                    for meth, (zh, Ih, t, ex) in run_methods(setup, grid, m, lam, sigma).items():
                        res = match_sources(z, I, zh, Ih)
                        rows.append({"geometrie": geom, "config": name, "niveau": level,
                                     "seed": seed, "methode": meth, "K": len(I),
                                     "n_est": res["n_est"], "err_pos": res["err_pos"],
                                     "err_int": res["err_int"], "temps": t,
                                     "eta_max": ex["eta_max"],
                                     "certifie": ex["eta_max"] <= 1 + 1e-3,
                                     "reussite": success(res, z)})
        print(f"  {geom} fait")
    write_csv(rows, OUT / "comparaison_methodes.csv")
    summ = summarize(rows, ["geometrie", "config", "niveau", "methode"],
                     ["n_est", "err_pos", "err_int", "temps"])
    write_csv(summ, OUT / "comparaison_methodes_resume.csv")
    return rows


# ---- 4. Erreur en fonction de σ --------------------------------------------------------------
def exp_sigma(n_seeds, c=2.0):
    """Erreur de position du Frank-Wolfe glissant en fonction de σ."""
    print("\n[4] Erreur de position en fonction de σ")
    levels = np.logspace(-5, -1.5, 8)
    rows = []
    for geom in GEOMETRIES:
        for name, (z, I) in CONFIGS.items():
            setup, grid = get_setup(geom, name)
            for level in levels:
                for seed in range(n_seeds):
                    rng = np.random.default_rng(1000 + seed)
                    m, sigma = simulate(setup, z, I, level, rng)
                    lam = c * sigma * np.sqrt(setup.M)
                    zh, Ih, t, ex = run_methods(setup, grid, m, lam, sigma, ("sfw",))["sfw"]
                    res = match_sources(z, I, zh, Ih)
                    rows.append({"geometrie": geom, "config": name, "niveau": level,
                                 "sigma": sigma, "seed": seed, "K": len(I),
                                 "n_est": res["n_est"], "err_pos": res["err_pos"],
                                 "err_int": res["err_int"], "reussite": success(res, z)})
        print(f"  {geom} fait")
    write_csv(rows, OUT / "erreur_sigma.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, geom in zip(axes, GEOMETRIES):
        for k, name in enumerate(CONFIGS):
            xs, ys = [], []
            for level in levels:
                sel = [r for r in rows if r["geometrie"] == geom and r["config"] == name
                       and r["niveau"] == level and r["n_est"] == r["K"]]
                if len(sel) >= max(1, n_seeds // 2):
                    xs.append(level)
                    ys.append(np.median([r["err_pos"] for r in sel]))
            if xs:
                ax.plot(xs, ys, "-o", color=SERIES[k], lw=2, ms=5, label=name)
        x0 = np.array([levels[0], levels[-1]])
        ax.plot(x0, 10 * x0, color=INK_MUTED, ls="--", lw=1, label="pente 1")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(geom, fontsize=10)
        ax.set_xlabel("niveau de bruit σ / rms(A(s*))")
        style_axes(ax)
    axes[0].set_ylabel("erreur de position (médiane, K correct)")
    # Légende commune, avec toutes les configurations (certains panneaux sont vides)
    handles = [plt.Line2D([], [], color=SERIES[k], marker="o", lw=2, label=name)
               for k, name in enumerate(CONFIGS)]
    handles.append(plt.Line2D([], [], color=INK_MUTED, ls="--", lw=1, label="pente 1"))
    axes[2].legend(handles=handles, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   frameon=False)
    fig.suptitle("Frank-Wolfe glissant : erreur de position en fonction du bruit "
                 "(points absents : K faux dans la majorité des tirages)", fontsize=11)
    fig.tight_layout()
    save(fig, "erreur_sigma.png")
    return rows


# ---- 5. Hypothèses ↔ réussite --------------------------------------------------------------
def exp_link(hyp_rows, meth_rows):
    """Relie marge / conditionnement au taux de réussite (σ = 1e-3, Frank-Wolfe)."""
    print("\n[5] Hypothèses et réussite de la reconstruction")
    rows = []
    for h in hyp_rows:
        sel = [r for r in meth_rows if r["geometrie"] == h["geometrie"]
               and r["config"] == h["config"] and r["niveau"] == 1e-3]
        row = dict(h)
        for meth in ["lasso", "lasso_raffine", "sfw"]:
            rs = [r["reussite"] for r in sel if r["methode"] == meth]
            row[f"reussite_{meth}"] = float(np.mean(rs)) if rs else np.nan
        rows.append(row)
        print(f"  {h['geometrie']:15s} {h['config']:9s} non dégénérée={h['non_degenere']!s:5} "
              f"marge rel.={h['marge_relative']:9.3g}  réussite SFW={row['reussite_sfw']:.2f}")
    write_csv(rows, OUT / "lien_hypotheses.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, geom in enumerate(GEOMETRIES):
        sel = [r for r in rows if r["geometrie"] == geom]
        for r in sel:
            ax.scatter(r["cond_norm"], r["reussite_sfw"], s=70, color=SERIES[k],
                       marker="o" if r["non_degenere"] else "X", edgecolors="white",
                       linewidths=1, zorder=3)
            ax.annotate(r["config"], (r["cond_norm"], r["reussite_sfw"]), fontsize=7,
                        xytext=(5, 4), textcoords="offset points", color=INK_MUTED)
        ax.scatter([], [], color=SERIES[k], label=geom)
    ax.scatter([], [], marker="o", color=INK_MUTED, label="condition vérifiée")
    ax.scatter([], [], marker="X", color=INK_MUTED, label="condition NON vérifiée")
    ax.set_xscale("log")
    ax.set_xlabel("conditionnement de Γ (colonnes normalisées)")
    ax.set_ylabel("taux de réussite (Frank-Wolfe, σ = 1e-3)")
    ax.set_ylim(-0.05, 1.08)
    style_axes(ax)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Hypothèses de Duval-Peyré et réussite de la reconstruction", fontsize=10)
    fig.tight_layout()
    save(fig, "lien_hypotheses.png")
    return rows


# ---- 6. Comparaison avec reconstruct.py (Prony) ---------------------------------------------
def exp_prony(n_seeds, c=1.5):
    """
    Scénario de scripts/reconstruct.py : sources tirées selon config/ (nombre
    fixé ici à n), source centrale (intensité 5000) mesurée puis retirée,
    bruit relatif gaussien du projet y (1 + σ ε). Mêmes mesures pour toutes
    les méthodes :
        prony_config   cfg.invert, exactement comme reconstruct.py (rank_tol de la config)
        prony_5sigma   Prony avec rank_tol = 5σ (réglage conseillé dans reconstruction.yaml)
        sfw_brut       Frank-Wolfe glissant, noyau brut (h = 0)
        sfw_normalise  Frank-Wolfe glissant, noyau normalisé
        lasso_normalise LASSO raffiné, noyau normalisé
    λ = c ||w||, ||w|| estimé par σ ||y|| (y : mesures brutes, source centrale comprise).
    """
    print("\n[6] Comparaison avec reconstruct.py (Prony), données bruitées")
    rows = []
    setups = {"brut": make_setup(h=0.0), "normalise": make_setup(h=0.0, normalize=True)}
    grids = {k: build_grid(step=GRID_STEP, setup=s) for k, s in setups.items()}
    for n in [2, 3, 5]:
        for sigma in [1e-4, 1e-3, 1e-2]:
            for seed in range(n_seeds):
                cfg = load_config(DEFAULT_CONFIG_DIR, [
                    f"seed={seed}", "sources.count_law.name=fixed",
                    f"sources.count_law.params={{n: {n}}}",
                    f"sensors.noise.params.sigma={sigma}"])
                rng = cfg.rng()
                src = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                                     cfg.intensity_law, cfg.position_law)
                y = measure(src + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)
                y_inv = subtract_known_sources(y, cfg.fixed_sources, cfg.sensors, cfg.eps) \
                    if cfg.subtract_fixed_sources else y
                noise_norm = sigma * float(np.linalg.norm(y))
                sig_vec = sigma * np.abs(y)             # écart type par capteur
                results = {}
                t0 = time.perf_counter()
                est = cfg.invert(y_inv, cfg.sensors)
                results["prony_config"] = (est.positions, est.intensities,
                                           time.perf_counter() - t0)
                t0 = time.perf_counter()
                est = prony_circle(y_inv, cfg.sensors, rank_tol=5 * sigma)
                results["prony_5sigma"] = (est.positions, est.intensities,
                                           time.perf_counter() - t0)
                lam = c * noise_norm
                for key, s in setups.items():
                    meths = ("sfw", "lasso_raffine") if key == "normalise" else ("sfw",)
                    for meth, (zh, Ih, t, _) in run_methods(s, grids[key], y_inv, lam, sig_vec,
                                                            meths).items():
                        tag = "sfw" if meth == "sfw" else "lasso"
                        results[f"{tag}_{key}"] = (zh, Ih, t)
                for meth, (zh, Ih, t) in results.items():
                    res = match_sources(src.positions, src.intensities, zh, Ih)
                    rows.append({"n": n, "sigma": sigma, "seed": seed, "methode": meth,
                                 "K": src.n, "n_est": res["n_est"], "err_pos": res["err_pos"],
                                 "err_int": res["err_int"],
                                 "err_int_total": res["err_int_total"], "temps": t,
                                 "sep_min": min_separation(src.positions),
                                 "reussite": success(res, src.positions)})
        print(f"  n = {n} fait")
    write_csv(rows, OUT / "comparaison_prony.csv")
    summ = summarize(rows, ["n", "sigma", "methode"], ["n_est", "err_pos", "err_int", "temps"])
    write_csv(summ, OUT / "comparaison_prony_resume.csv")

    methods = list(dict.fromkeys(r["methode"] for r in rows))
    sigmas = sorted({r["sigma"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, n in enumerate([2, 3, 5]):
        for k, meth in enumerate(methods):
            sel = [[r for r in rows if r["n"] == n and r["sigma"] == s and r["methode"] == meth]
                   for s in sigmas]
            rate = [np.mean([r["reussite"] for r in g]) for g in sel]
            err = [np.nanmedian([r["err_pos"] for r in g]) for g in sel]
            axes[0, j].plot(sigmas, rate, "-o", color=SERIES[k], lw=2, ms=5, label=meth)
            axes[1, j].plot(sigmas, err, "-o", color=SERIES[k], lw=2, ms=5, label=meth)
        axes[0, j].set_title(f"{n} sources tirées (config du projet)", fontsize=10)
        axes[0, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("σ (bruit relatif du projet)")
        for i in range(2):
            axes[i, j].set_xscale("log")
            style_axes(axes[i, j])
    axes[0, 0].set_ylabel("taux de réussite")
    axes[1, 0].set_ylabel("erreur de position (médiane)")
    axes[0, 2].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Comparaison avec reconstruct.py (Prony) : source centrale retirée, "
                 "bruit relatif", fontsize=11)
    fig.tight_layout()
    save(fig, "comparaison_prony.png")
    return rows


def main():
    """Lance les expériences demandées."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapide", action="store_true", help="moins de tirages")
    parser.add_argument("--seulement", nargs="*", default=None,
                        choices=["hypotheses", "lambda", "methodes", "sigma", "lien", "prony"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    todo = set(args.seulement or ["hypotheses", "lambda", "methodes", "sigma", "lien", "prony"])
    n = 2 if args.rapide else 4
    t0 = time.perf_counter()
    hyp = exp_hypotheses() if todo & {"hypotheses", "lien"} else None
    if "lambda" in todo:
        exp_lambda(max(1, n // 2))
    meth = exp_methods(n) if todo & {"methodes", "lien"} else None
    if "sigma" in todo:
        exp_sigma(n)
    if "lien" in todo:
        exp_link(hyp, meth)
    if "prony" in todo:
        exp_prony(2 * n)
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s. Résultats : {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

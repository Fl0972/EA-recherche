"""
Étude ciblée du noyau physique du projet, G(x, z) = 1 / ||x - z||²
(capteurs dans le plan des sources, h = 0, 64 capteurs sur le cercle
circonscrit à D = [0, 10] × [0, 20]).

    python sparse_sources/experiments_1r2.py            (~15 min)
    python sparse_sources/experiments_1r2.py --rapide   (~5 min)
    python sparse_sources/experiments_1r2.py --seulement alpha resolution

Le BLASSO brut (pénalité λ Σ I_k) échoue toujours pour ce noyau (voir
COMPTE_RENDU.md) ; on étudie le BLASSO pondéré

    min_{I ≥ 0, z}  ½ ||m - Σ_k I_k φ(z_k)||² + λ Σ_k I_k ||φ(z_k)||^α,

qui est le BLASSO pour le noyau ψ = φ / ||φ||^α (mêmes mesures, même
modèle 1/r² : seule la pénalité change). α = 0 : brut, α = 1 : normalisé.

Choix de λ : λ = c σ √M ||Ψ||, ||Ψ|| = max_D ||ψ(z)|| (norme de l'opérateur
pour la norme TV) ; le bruit seul donne alors un certificat ≤ 1/c partout
(≈, car ||w|| ≈ σ √M), quel que soit α. c = 2 par défaut.

Expériences :
    alpha       taux de condition vérifiée et de réussite en fonction de α et K
                (sources tirées selon config/ : positions uniformes, intensités
                uniformes sur le simplexe, somme 100)
    prediction  pouvoir prédictif de check_hypotheses : réussite selon le
                verdict, la marge relative et le conditionnement
    resolution  paire au centre : réussite et erreur selon la séparation d et σ
    distance    paire déplacée du centre vers le bord (vers les capteurs)
    capteurs    effet du nombre de capteurs M
    temps       temps de calcul (Frank-Wolfe, LASSO, hypothèses) selon K, M, pas

Sorties : sparse_sources/results/r2_*.png et r2_*.csv.
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

from particle_sources.generation import intensity_laws  # noqa: E402
from sparse_sources import (build_grid, check_hypotheses, forward, lasso_pipeline,  # noqa: E402
                            make_setup, match_sources, sliding_frank_wolfe)
from sparse_sources.experiments import min_separation, save, success, write_csv  # noqa: E402
from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
GRID_STEP = 0.25        # grille du Frank-Wolfe / LASSO
HYP_STEP = 0.1          # grille de check_hypotheses
TOTAL = 100.0           # intensité totale (config/simulation.yaml)
ALPHAS = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

_CACHE = {}


def setup_grid(alpha: float, n_sensors: int | None = None):
    """Setup 1/r² (h = 0) pondéré par ||φ||^α, sa grille et ||Ψ|| = max ||ψ||."""
    key = (alpha, n_sensors)
    if key not in _CACHE:
        S = make_setup(h=0.0, alpha=alpha, n_sensors=n_sensors)
        g = build_grid(step=GRID_STEP, setup=S)
        _CACHE[key] = (S, g, float(np.max(np.linalg.norm(g.G, axis=0))))
    return _CACHE[key]


def draw_sources(K: int, rng, margin: float = 0.25):
    """
    K sources selon les lois de config/ : positions uniformes dans D (à
    distance ≥ margin du bord, le théorème supposant les sources intérieures),
    intensités uniformes sur le simplexe de somme TOTAL.
    """
    z = rng.uniform([margin, margin], [10 - margin, 20 - margin], (K, 2))
    I = intensity_laws.uniform_simplex(K, TOTAL, rng)
    return z, np.asarray(I, float)


def noisy(S, z, I, level, rng):
    """m = A(s*) + σ ε, σ = level × rms(A(s*)) (noyau physique 1/r²)."""
    m0 = forward(z, I, S.with_power(0.0))
    sigma = level * float(np.sqrt(np.mean(m0**2)))
    return m0 + sigma * rng.standard_normal(m0.shape), sigma


def run_sfw(S, g, op_norm, m, sigma, c=2.0, max_iter=12):
    """Frank-Wolfe glissant avec λ = c σ √M ||Ψ|| ; renvoie (z, I physiques, sortie)."""
    lam = c * sigma * np.sqrt(S.M) * op_norm
    out = sliding_frank_wolfe(m, lam, setup=S, grid=g, tol=1e-5, max_iter=max_iter)
    z = out["z_debiased"]
    return z, S.physical_intensities(z, out["I_debiased"]), out


def evaluate(z_true, I_true, z, I):
    """Erreurs et réussite (bon K, erreur de position < 0.25 min(sép, 4))."""
    res = match_sources(z_true, I_true, z, I)
    res["reussite"] = success(res, z_true)
    return res


def rate(rows, key="reussite"):
    """Moyenne d'une colonne booléenne (nan si vide)."""
    return float(np.mean([r[key] for r in rows])) if rows else np.nan


# ---- 1. Balayage de α -----------------------------------------------------------------
def exp_alpha(n_draws, level=1e-3):
    """Taux de condition vérifiée et de réussite selon α et K."""
    print(f"\n[1] Pondération ||φ||^α (σ = {level:g} × rms, {n_draws} tirages par K)")
    rows = []
    Ks = [1, 2, 3, 4, 5]
    for K in Ks:
        rng = np.random.default_rng(10 + K)
        draws = [draw_sources(K, rng) for _ in range(n_draws)]
        for a in ALPHAS:
            S, g, opn = setup_grid(a)
            for j, (z, I) in enumerate(draws):
                hyp = check_hypotheses(z, setup=S, verbose=False, grid_step=HYP_STEP)
                m, sigma = noisy(S, z, I, level, np.random.default_rng(1000 * K + j))
                zh, Ih, out = run_sfw(S, g, opn, m, sigma)
                res = evaluate(z, I, zh, Ih)
                rows.append({"alpha": a, "K": K, "tirage": j, "sep_min": min_separation(z),
                             "non_degenere": hyp["non_degenerate"], "H1": hyp["H1"],
                             "H2": hyp["H2"], "H3": hyp["H3"],
                             "marge_relative": hyp["margin_relative"],
                             "cond_norm": hyp["cond_normalized"], "n_est": res["n_est"],
                             "err_pos": res["err_pos"], "err_int": res["err_int"],
                             "certifie": out["converged"], "temps": out["time"],
                             "reussite": res["reussite"]})
        print(f"  K = {K} fait")
    write_csv(rows, OUT / "r2_alpha.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for k, K in enumerate(Ks):
        nd = [rate([r for r in rows if r["alpha"] == a and r["K"] == K], "non_degenere")
              for a in ALPHAS]
        sc = [rate([r for r in rows if r["alpha"] == a and r["K"] == K]) for a in ALPHAS]
        axes[0].plot(ALPHAS, nd, "-o", color=SERIES[k], lw=2, ms=5, label=f"K = {K}")
        axes[1].plot(ALPHAS, sc, "-o", color=SERIES[k], lw=2, ms=5, label=f"K = {K}")
    axes[0].set_title("Condition de Duval-Peyré vérifiée (vraies positions)", fontsize=10)
    axes[1].set_title(f"Réussite du Frank-Wolfe glissant (σ = {level:g} × rms)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("α   (pénalité λ Σ I_k ||φ(z_k)||^α ; 0 = BLASSO brut)")
        ax.axvline(1.0, color=INK_MUTED, lw=1, ls="--")
        ax.set_ylim(-0.05, 1.05)
        style_axes(ax)
    axes[0].set_ylabel("fraction des tirages")
    axes[1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Noyau 1/r², capteurs dans le plan : effet de la pondération", fontsize=11)
    fig.tight_layout()
    save(fig, "r2_alpha.png")
    return rows


# ---- 2. Pouvoir prédictif des hypothèses ------------------------------------------------------
def exp_prediction(alpha_rows):
    """Réussite selon le verdict et la marge relative (réutilise les tirages de [1])."""
    print("\n[2] Pouvoir prédictif de check_hypotheses")
    rows = []
    for a in ALPHAS:
        sel = [r for r in alpha_rows if r["alpha"] == a]
        nd = [r for r in sel if r["non_degenere"]]
        dg = [r for r in sel if not r["non_degenere"]]
        rows.append({"alpha": a, "n": len(sel), "n_non_degenere": len(nd),
                     "reussite_si_non_degenere": rate(nd), "reussite_si_degenere": rate(dg)})
        print(f"  α = {a:4.2f} : condition vraie {len(nd):3d}/{len(sel)} ; réussite "
              f"{rate(nd):.2f} si vraie, {rate(dg):.2f} sinon")
    write_csv(rows, OUT / "r2_prediction.csv")

    sel = [r for r in alpha_rows if r["alpha"] in (1.0, 1.5)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, a in enumerate([1.0, 1.5]):
        rs = [r for r in sel if r["alpha"] == a]
        for ok, mk in [(True, "o"), (False, "X")]:
            pts = [r for r in rs if r["reussite"] == ok]
            x = [max(r["marge_relative"], -0.95) if np.isfinite(r["marge_relative"]) else -0.95
                 for r in pts]
            y = [r["cond_norm"] for r in pts]
            axes[0].scatter(x, y, s=36, marker=mk, color=SERIES[k], alpha=0.75,
                            edgecolors="white", linewidths=0.6,
                            label=f"α = {a:g}, {'réussite' if ok else 'échec'}")
    axes[0].axvline(0, color=INK_MUTED, lw=1, ls="--")
    axes[0].set_xscale("symlog", linthresh=1e-2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("marge relative de (H2) (tronquée à -0.95)")
    axes[0].set_ylabel("conditionnement de Γ (colonnes normalisées)")
    axes[0].set_title("Chaque point = un tirage (K = 1 à 5)", fontsize=10)
    style_axes(axes[0])
    axes[0].legend(fontsize=7, frameon=False)
    # Réussite selon le conditionnement, condition vraie seulement
    bins = np.logspace(0, 7, 8)
    for k, a in enumerate([1.0, 1.5]):
        rs = [r for r in sel if r["alpha"] == a and r["non_degenere"]]
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = [r for r in rs if lo <= r["cond_norm"] < hi]
            if len(b) >= 3:
                xs.append(np.sqrt(lo * hi))
                ys.append(rate(b))
        axes[1].plot(xs, ys, "-o", color=SERIES[k], lw=2, ms=5, label=f"α = {a:g}")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_xlabel("conditionnement de Γ (par décade)")
    axes[1].set_ylabel("taux de réussite (condition vraie)")
    axes[1].set_title("Condition vraie : la réussite dépend du conditionnement", fontsize=10)
    style_axes(axes[1])
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "r2_prediction.png")
    return rows


# ---- 3. Limite de résolution -------------------------------------------------------------
def exp_resolution(n_seeds, alpha=1.0):
    """Paire verticale centrée en (5, 10), intensités 10 et 10 : réussite selon d et σ."""
    print(f"\n[3] Limite de résolution (α = {alpha:g})")
    S, g, opn = setup_grid(alpha)
    ds = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    levels = [1e-5, 1e-4, 1e-3, 1e-2]
    rows = []
    for d in ds:
        z = np.array([[5.0, 10.0 - d / 2], [5.0, 10.0 + d / 2]])
        I = np.array([10.0, 10.0])
        hyp = check_hypotheses(z, setup=S, verbose=False, grid_step=HYP_STEP)
        for level in levels:
            for seed in range(n_seeds):
                m, sigma = noisy(S, z, I, level, np.random.default_rng(seed))
                zh, Ih, out = run_sfw(S, g, opn, m, sigma)
                res = evaluate(z, I, zh, Ih)
                rows.append({"d": d, "niveau": level, "seed": seed,
                             "non_degenere": hyp["non_degenerate"],
                             "marge_relative": hyp["margin_relative"],
                             "cond_norm": hyp["cond_normalized"], "n_est": res["n_est"],
                             "err_pos": res["err_pos"], "err_int": res["err_int"],
                             "reussite": res["reussite"]})
    write_csv(rows, OUT / "r2_resolution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, level in enumerate(levels):
        sc = [rate([r for r in rows if r["d"] == d and r["niveau"] == level]) for d in ds]
        axes[0].plot(ds, sc, "-o", color=SERIES[k], lw=2, ms=5, label=f"σ = {level:g} × rms")
        er = []
        for d in ds:
            b = [r["err_pos"] for r in rows if r["d"] == d and r["niveau"] == level
                 and r["n_est"] == 2]
            er.append(np.median(b) if len(b) >= max(1, n_seeds // 2) else np.nan)
        axes[1].plot(ds, er, "-o", color=SERIES[k], lw=2, ms=5, label=f"σ = {level:g} × rms")
    axes[0].set_ylabel("taux de réussite")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("erreur de position (médiane, K = 2 trouvé)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("séparation d des deux sources")
        style_axes(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Noyau 1/r², pondération α = {alpha:g} : limite de résolution d'une paire "
                 "au centre de D", fontsize=11)
    fig.tight_layout()
    save(fig, "r2_resolution.png")
    return rows


# ---- 4. Distance aux capteurs ------------------------------------------------------------
def exp_distance(n_seeds, alpha=1.0, level=1e-3):
    """Paire horizontale (d = 2) déplacée de y = 10 (centre) vers y = 19 (bord haut)."""
    print(f"\n[4] Distance aux capteurs (α = {alpha:g}, σ = {level:g} × rms)")
    S, g, opn = setup_grid(alpha)
    ys = [10.0, 12.0, 14.0, 16.0, 17.5, 18.5, 19.25]
    rows = []
    for y in ys:
        z = np.array([[4.0, y], [6.0, y]])
        I = np.array([10.0, 10.0])
        hyp = check_hypotheses(z, setup=S, verbose=False, grid_step=HYP_STEP)
        dist = float(np.min(np.linalg.norm(S.sensors[None] - z[:, None], axis=-1)))
        for seed in range(n_seeds):
            m, sigma = noisy(S, z, I, level, np.random.default_rng(seed))
            zh, Ih, out = run_sfw(S, g, opn, m, sigma)
            res = evaluate(z, I, zh, Ih)
            rows.append({"y": y, "dist_capteur": dist, "seed": seed,
                         "non_degenere": hyp["non_degenerate"],
                         "marge_relative": hyp["margin_relative"],
                         "cond_norm": hyp["cond_normalized"], "n_est": res["n_est"],
                         "err_pos": res["err_pos"], "reussite": res["reussite"]})
    write_csv(rows, OUT / "r2_distance.csv")
    for y in ys:
        b = [r for r in rows if r["y"] == y]
        print(f"  y = {y:5.2f} (capteur à {b[0]['dist_capteur']:.2f}) : condition "
              f"{b[0]['non_degenere']!s:5}, réussite {rate(b):.2f}, erreur médiane "
              f"{np.nanmedian([r['err_pos'] for r in b]):.3g}")
    return rows


# ---- 5. Nombre de capteurs ----------------------------------------------------------------
def exp_sensors(n_draws, alpha=1.0, level=1e-3):
    """Réussite selon M (capteurs sur le cercle circonscrit) et K."""
    print(f"\n[5] Nombre de capteurs (α = {alpha:g}, σ = {level:g} × rms)")
    Ms = [16, 32, 64, 128, 256]
    Ks = [1, 2, 3, 4, 5]
    rows = []
    for K in Ks:
        rng = np.random.default_rng(500 + K)
        draws = [draw_sources(K, rng) for _ in range(n_draws)]
        for M in Ms:
            S, g, opn = setup_grid(alpha, M)
            for j, (z, I) in enumerate(draws):
                m, sigma = noisy(S, z, I, level, np.random.default_rng(K * 100 + j))
                zh, Ih, out = run_sfw(S, g, opn, m, sigma)
                res = evaluate(z, I, zh, Ih)
                rows.append({"M": M, "K": K, "tirage": j, "n_est": res["n_est"],
                             "err_pos": res["err_pos"], "temps": out["time"],
                             "reussite": res["reussite"]})
        print(f"  K = {K} fait")
    write_csv(rows, OUT / "r2_capteurs.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for k, K in enumerate(Ks):
        sc = [rate([r for r in rows if r["M"] == M and r["K"] == K]) for M in Ms]
        ax.plot(Ms, sc, "-o", color=SERIES[k], lw=2, ms=5, label=f"K = {K}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(M) for M in Ms])
    ax.set_xlabel("nombre de capteurs M (cercle circonscrit)")
    ax.set_ylabel("taux de réussite")
    ax.set_ylim(-0.05, 1.05)
    style_axes(ax)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title(f"Noyau 1/r², α = {alpha:g}, σ = {level:g} × rms (σ relatif au rms, "
                 "donc ||w|| croît comme √M)", fontsize=9)
    fig.tight_layout()
    save(fig, "r2_capteurs.png")
    return rows


# ---- 6. Temps de calcul ---------------------------------------------------------------------
def exp_timing(n_rep, alpha=1.0, level=1e-3):
    """Temps des briques selon K, le pas de grille et M."""
    print("\n[6] Temps de calcul")
    rows = []
    S, g, opn = setup_grid(alpha)
    for K in [1, 2, 3, 5, 8]:
        rng = np.random.default_rng(900 + K)
        for rep in range(n_rep):
            z, I = draw_sources(K, rng)
            m, sigma = noisy(S, z, I, level, rng)
            t0 = time.perf_counter()
            check_hypotheses(z, setup=S, verbose=False)
            t_hyp = time.perf_counter() - t0
            zh, Ih, out = run_sfw(S, g, opn, m, sigma)
            rows.append({"brique": "sfw", "K": K, "M": S.M, "pas": GRID_STEP,
                         "temps": out["time"], "iterations": out["n_iter"]})
            rows.append({"brique": "check_hypotheses (pas 0.05)", "K": K, "M": S.M,
                         "pas": 0.05, "temps": t_hyp})
    rng = np.random.default_rng(7)
    z, I = draw_sources(3, rng)
    for step in [0.5, 0.25, 0.1]:
        gs = build_grid(step=step, setup=S)
        m, sigma = noisy(S, z, I, level, rng)
        lam = 2 * sigma * np.sqrt(S.M) * opn
        for method in ["active_set", "fista", "sklearn"]:
            t0 = time.perf_counter()
            lasso_pipeline(m, lam, grid=gs, setup=S, method=method, refine_positions=True)
            rows.append({"brique": f"lasso_pipeline ({method})", "K": 3, "M": S.M,
                         "pas": step, "P": gs.P, "temps": time.perf_counter() - t0})
    for M in [16, 64, 256]:
        S2, g2, opn2 = setup_grid(alpha, M)
        m, sigma = noisy(S2, z, I, level, rng)
        zh, Ih, out = run_sfw(S2, g2, opn2, m, sigma)
        rows.append({"brique": "sfw", "K": 3, "M": M, "pas": GRID_STEP, "temps": out["time"],
                     "iterations": out["n_iter"]})
    write_csv(rows, OUT / "r2_temps.csv")
    groups = {}
    for r in rows:
        groups.setdefault((r["brique"], r["K"], r["M"], r["pas"]), []).append(r["temps"])
    for (b, K, M, pas), ts in groups.items():
        print(f"  {b:32s} K = {K}  M = {M:3d}  pas = {pas:<5g} : {np.median(ts):.3f} s")
    return rows


def main():
    """Lance les expériences demandées."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapide", action="store_true", help="moins de tirages")
    parser.add_argument("--seulement", nargs="*", default=None,
                        choices=["alpha", "prediction", "resolution", "distance", "capteurs",
                                 "temps"])
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    todo = set(args.seulement or ["alpha", "prediction", "resolution", "distance", "capteurs",
                                  "temps"])
    n = 8 if args.rapide else 20
    t0 = time.perf_counter()
    alpha_rows = exp_alpha(n) if todo & {"alpha", "prediction"} else None
    if "prediction" in todo:
        exp_prediction(alpha_rows)
    if "resolution" in todo:
        exp_resolution(max(3, n // 4))
    if "distance" in todo:
        exp_distance(max(3, n // 4))
    if "capteurs" in todo:
        exp_sensors(max(4, n // 2))
    if "temps" in todo:
        exp_timing(3)
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s. Résultats : {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

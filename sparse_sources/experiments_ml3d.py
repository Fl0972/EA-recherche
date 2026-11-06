"""
Maximum de vraisemblance + BIC en 3D : sources dans un cube, capteurs non
coplanaires.

    python sparse_sources/experiments_ml3d.py            (~30 min)
    python sparse_sources/experiments_ml3d.py --rapide

Géométrie (comme scripts/duval_peyre_3d.py) : sources dans le cube
[0, 10]³, M = 160 capteurs à 1,5 × le rayon de la sphère circonscrite,
dispositions de ``particle_sources.duval_peyre.experiments_3d.layout`` :

    deux_cercles    cercles horizontaux aux hauteurs bas et haut du cube
    trois_cercles   bas, milieu, haut
    sphere          spirale de Fibonacci sur la sphère

(un seul cercle ne suffit pas : le problème n'est pas identifiable en 3D).

Données : K sources espacées (distance ≥ 3), positions uniformes dans le
cube, intensités uniformes sur le simplexe (somme 100), source centrale
5000 au centre du cube, bruit relatif y (1 + σ ε), source centrale retirée.
Même noyau que le projet, en 3D : G = 1 / ||x - z||².

Méthodes :
    ml_bic        ``ml3d.greedy_ml_nd`` (K inconnu, choisi par BIC)
    blasso_projet BLASSO du projet en 3D (``particle_sources.duval_peyre.blasso``,
                  Frank-Wolfe glissant + débiaisage, λ = 4 ||bruit|| ||Φ|| et
                  seuil 0,5 % comme dans scripts/duval_peyre_3d.py)

Sorties : sparse_sources/results/ml3d.csv, ml3d_resume.csv, ml3d.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particle_sources.duval_peyre import experiments_3d as ex3  # noqa: E402
from particle_sources.duval_peyre.blasso import (Estimate, debias,  # noqa: E402
                                                 operator_norm, sliding_frank_wolfe)
from particle_sources.duval_peyre.kernel import InverseSquareKernel  # noqa: E402
from particle_sources.model import Box  # noqa: E402
from sparse_sources.experiments import OUT, save, summarize, write_csv  # noqa: E402
from sparse_sources.ml3d import (cramer_rao_nd, greedy_ml_nd, match_nd,  # noqa: E402
                                 success_nd)
from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

BOX = Box.from_bounds([[0.0, 10.0]] * 3)
M = 160
RADIUS = 1.5
LAYOUTS = ["deux_cercles", "trois_cercles", "sphere"]
KS = [1, 2, 3, 5]
SIGMAS = [1e-4, 1e-3, 1e-2]
SEP_MIN = 3.0
I_CENTRAL = 5000.0
H_GRID = 0.5


def draw_sources(K, rng, sep=SEP_MIN):
    """K positions uniformes dans le cube à distance ≥ sep ; intensités ~ simplexe (somme 100)."""
    lo, hi = BOX.lo, BOX.hi
    while True:
        z = np.empty((0, 3))
        for _ in range(5000):
            p = lo + (hi - lo) * rng.random(3)
            if len(z) == 0 or np.min(np.linalg.norm(z - p, axis=1)) >= sep:
                z = np.vstack([z, p])
                if len(z) == K:
                    return z, 100.0 * rng.dirichlet(np.ones(K))


def run(n_draws):
    """Toutes les dispositions, K et σ ; mêmes sources pour toutes les dispositions."""
    sensors = {name: ex3.layout(name, BOX, M, RADIUS) for name in LAYOUTS}
    kernels = {name: InverseSquareKernel(S) for name, S in sensors.items()}
    norms = {name: operator_norm(k, BOX, H_GRID) for name, k in kernels.items()}
    center = np.asarray(BOX.center, float)[None]
    rows = []
    for K in KS:
        for sigma in SIGMAS:
            t0 = time.perf_counter()
            for d in range(n_draws):
                z0, I0 = draw_sources(K, np.random.default_rng(100 * K + d))
                for name, S in sensors.items():
                    k = kernels[name]
                    central = I_CENTRAL * k.phi(center)[0]
                    y = k.phi(z0).T @ I0 + central
                    s = sigma * np.abs(y)
                    noise = s * np.random.default_rng(10 * d + K).standard_normal(M)
                    m = y + noise - central
                    crb = float(cramer_rao_nd(z0, I0, s, S)["pos_std"].max())
                    base = {"disposition": name, "K": K, "sigma": sigma, "tirage": d,
                            "crb_max": crb, "faisable": bool(2 * crb < 0.25 * min(SEP_MIN, 4))}
                    r = greedy_ml_nd(m, s, S, BOX)
                    out = {"ml_bic": (r["z"], r["I"], r["time"])}
                    tic = time.perf_counter()
                    lam = 4.0 * float(np.linalg.norm(noise)) * norms[name]
                    est, _, _ = sliding_frank_wolfe(k, m, lam, BOX, H_GRID, max_iter=2 * K + 5)
                    deb = debias(k, m, est)
                    keep = deb.intensities > 0.005 * I0.sum()
                    out["blasso_projet"] = (deb.positions[keep], deb.intensities[keep],
                                            time.perf_counter() - tic)
                    for meth, (zh, Ih, tt) in out.items():
                        res = match_nd(z0, I0, zh, Ih)
                        rows.append({**base, "methode": meth, "n_est": res["n_est"],
                                     "err_pos": res["err_pos"], "err_int": res["err_int"],
                                     "temps": tt, "reussite": success_nd(res, z0)})
            print(f"  K = {K}, σ = {sigma:g} fait ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    return rows


def report(rows):
    """Résumé console + CSV + figure."""
    write_csv(rows, OUT / "ml3d.csv")
    summ = summarize(rows, ["disposition", "K", "sigma", "methode"],
                     ["n_est", "err_pos", "crb_max", "temps"])
    write_csv(summ, OUT / "ml3d_resume.csv")

    def sel(**kw):
        return [r for r in rows if all(r[a] == b for a, b in kw.items())]

    print(f"\n{'disposition':14s} {'K':>2s} {'σ':>7s} {'ML+BIC':>7s} {'BLASSO':>7s} "
          f"{'K ML':>5s} {'K BLASSO':>8s} {'err ML':>8s} {'CRB':>8s} {'faisab.':>7s} {'t ML':>6s}")
    for name in LAYOUTS:
        for K in KS:
            for sigma in SIGMAS:
                ml = sel(disposition=name, K=K, sigma=sigma, methode="ml_bic")
                bl = sel(disposition=name, K=K, sigma=sigma, methode="blasso_projet")
                print(f"{name:14s} {K:2d} {sigma:7.0e} "
                      f"{100 * np.mean([r['reussite'] for r in ml]):6.0f}% "
                      f"{100 * np.mean([r['reussite'] for r in bl]):6.0f}% "
                      f"{np.median([r['n_est'] for r in ml]):5.0f} "
                      f"{np.median([r['n_est'] for r in bl]):8.0f} "
                      f"{np.nanmedian([r['err_pos'] for r in ml]):8.3g} "
                      f"{np.median([r['crb_max'] for r in ml]):8.3g} "
                      f"{100 * np.mean([r['faisable'] for r in ml]):6.0f}% "
                      f"{np.median([r['temps'] for r in ml]):6.2f}")

    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(12, 3.9), sharey=True)
    for ax, sigma in zip(axes, SIGMAS):
        for k, name in enumerate(LAYOUTS):
            for meth, ls, mk in [("ml_bic", "-", "o"), ("blasso_projet", "--", "s")]:
                vals = [np.mean([r["reussite"] for r in sel(disposition=name, K=K, sigma=sigma,
                                                            methode=meth)]) for K in KS]
                ax.plot(KS, vals, ls, marker=mk, color=SERIES[k], lw=2, ms=5,
                        label=f"{name} ({'ML + BIC' if meth == 'ml_bic' else 'BLASSO'})")
        ax.set_title(f"σ = {sigma:g}", fontsize=10)
        ax.set_xticks(KS)
        ax.set_xlabel("nombre de sources K")
        ax.set_ylim(-0.05, 1.05)
        style_axes(ax)
    axes[0].set_ylabel("taux de réussite")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Sources en 3D (cube [0, 10]³, M = 160) : traits pleins = ML + BIC, "
                 "tirets = BLASSO du projet", fontsize=10, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "ml3d.png")


def main():
    """Lance l'expérience."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rapide", action="store_true", help="1 tirage par case")
    p.add_argument("--tirages", type=int, default=None)
    args = p.parse_args()
    n = args.tirages or (1 if args.rapide else 8)
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    report(run(n))
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    main()

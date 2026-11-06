"""
Tester le maximum de vraisemblance + BIC sur un tirage, en 2D (scénario du
projet) ou en 3D.

    python sparse_sources/tester_ml.py                         2D, 3 sources, σ = 1e-3
    python sparse_sources/tester_ml.py --sources 5 --sigma 1e-4 --graine 7
    python sparse_sources/tester_ml.py --positions "2,4,30;7,15,70" --sigma 1e-3
    python sparse_sources/tester_ml.py --sans-centrale --comparer
    python sparse_sources/tester_ml.py --dim 3 --capteurs sphere --sources 3

En 2D, les données sont exactement celles de scripts/reconstruct.py (config/,
source centrale 5000 ajoutée puis retirée, bruit relatif σ) ; --positions
impose les sources (« x,y,I;x,y,I »). En 3D : cube [0, 10]³, 160 capteurs
(deux_cercles, trois_cercles ou sphere), même bruit et même source centrale.

Affiche les sources vraies et estimées, la borne de Cramér-Rao de chaque
source, χ² / M (≈ 1 si le modèle explique les données) et l'évolution du
critère BIC ; enregistre une figure dans outputs/tester_ml/. --comparer
lance aussi Prony et le Frank-Wolfe (2D seulement).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

OUTDIR = ROOT / "outputs" / "tester_ml"


def parse_positions(text, dim):
    """'x,y,I;x,y,I' -> (z (K, dim), I (K,))."""
    vals = [[float(v) for v in p.split(",")] for p in text.split(";") if p.strip()]
    if any(len(v) != dim + 1 for v in vals):
        raise SystemExit(f"--positions : {dim + 1} nombres par source attendus")
    a = np.array(vals)
    return a[:, :dim], a[:, dim]


def data_2d(args):
    """Tirage 2D du projet (ou sources imposées) -> dict."""
    from sparse_sources import relative_noise_std
    from sparse_sources.experiments_projet import SETUP, scenario
    factor = 0 if args.sans_centrale else None
    cfg, src, y, m = scenario(args.sources, args.sigma, args.graine, central_factor=factor)
    z, I = src.positions, src.intensities
    if args.positions:
        z, I = parse_positions(args.positions, 2)
        raw = SETUP.with_power(0.0)
        c = 0.0 if args.sans_centrale else 5000.0 * raw.phi(
            np.asarray(SETUP.domain.center, float)[None])[0]
        y = raw.phi(z).T @ I + c
        rng = np.random.default_rng(args.graine)
        m = y * (1 + args.sigma * rng.standard_normal(len(y))) - c
    return {"cfg": cfg, "z": z, "I": I, "y": y, "m": m, "s": relative_noise_std(y, args.sigma),
            "sensors": SETUP.sensors, "box": SETUP.domain, "setup": SETUP}


def data_3d(args):
    """Tirage 3D (cube [0, 10]³) -> dict."""
    from particle_sources.duval_peyre import experiments_3d as ex3
    from particle_sources.duval_peyre.kernel import InverseSquareKernel
    from sparse_sources.experiments_ml3d import BOX, M, RADIUS, draw_sources
    S = ex3.layout(args.capteurs, BOX, M, RADIUS)
    k = InverseSquareKernel(S)
    rng = np.random.default_rng(args.graine)
    if args.positions:
        z, I = parse_positions(args.positions, 3)
    else:
        z, I = draw_sources(args.sources, rng)
    c = 0.0 if args.sans_centrale else 5000.0 * k.phi(np.asarray(BOX.center, float)[None])[0]
    y = k.phi(z).T @ I + c
    s = args.sigma * np.abs(y)
    m = y + s * rng.standard_normal(len(y)) - c
    return {"z": z, "I": I, "y": y, "m": m, "s": s, "sensors": S, "box": BOX}


def main():
    """Un tirage, une reconstruction, un résumé."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--sources", type=int, default=3, help="nombre de sources tirées")
    p.add_argument("--positions", default=None, help="sources imposées : 'x,y,I;x,y,I'")
    p.add_argument("--sigma", type=float, default=1e-3, help="bruit relatif")
    p.add_argument("--graine", type=int, default=0)
    p.add_argument("--sans-centrale", action="store_true", help="pas de source centrale")
    p.add_argument("--critere", default="bic", help="bic, aic ou κ numérique")
    p.add_argument("--capteurs", default="sphere", choices=["deux_cercles", "trois_cercles",
                                                            "sphere"], help="3D seulement")
    p.add_argument("--comparer", action="store_true", help="aussi Prony et Frank-Wolfe (2D)")
    args = p.parse_args()
    crit = args.critere if args.critere in ("bic", "aic") else float(args.critere)

    from sparse_sources.ml3d import cramer_rao_nd, greedy_ml_nd, match_nd, success_nd
    D = data_2d(args) if args.dim == 2 else data_3d(args)
    if args.dim == 2:
        from sparse_sources import greedy_ml
        from sparse_sources.experiments_projet import GRID
        r = greedy_ml(D["m"], D["s"], setup=D["setup"], grid=GRID, criterion=crit)
    else:
        r = greedy_ml_nd(D["m"], D["s"], D["sensors"], D["box"], criterion=crit)
    z, I = D["z"], D["I"]
    crb = cramer_rao_nd(z, I, D["s"], D["sensors"])["pos_std"]
    res = match_nd(z, I, r["z"], r["I"])
    M = len(D["m"])

    print(f"\n{args.dim}D, {len(I)} sources vraies, σ = {args.sigma:g}, "
          f"source centrale : {'non' if args.sans_centrale else 'oui (5000, retirée)'}, "
          f"M = {M} capteurs")
    print("\nSources vraies (et borne de Cramér-Rao sur la position) :")
    for zk, Ik, ck in zip(z, I, crb):
        print(f"  position {np.round(zk, 3)}  intensité {Ik:7.2f}  CRB {ck:.3g}")
    print(f"\nMaximum de vraisemblance + {str(args.critere).upper()} : K = {r['K']} "
          f"({r['time']:.2f} s), χ² / M = {r['chi2'] / M:.2f}")
    for zk, Ik in zip(r["z"], r["I"]):
        print(f"  position {np.round(zk, 3)}  intensité {Ik:7.2f}")
    print("\nÉvolution du critère : " + ", ".join(
        f"K={h['K']}: {h['criterion']:.1f}" for h in r["history"]))
    print(f"\nErreur de position max {res['err_pos']:.3g}, erreur d'intensité relative max "
          f"{res['err_int']:.3g} -> {'RÉUSSITE' if success_nd(res, z) else 'échec'} "
          f"(bon K et erreur < 0,25 × min(séparation, 4))")

    others = {}
    if args.comparer and args.dim == 2:
        from sparse_sources import sliding_frank_wolfe
        from sparse_sources.experiments_projet import GRID, SETUP
        est = D["cfg"].invert(D["m"], D["cfg"].sensors)
        others["Prony"] = (est.positions, est.intensities)
        fw = sliding_frank_wolfe(D["m"], 1.5 * args.sigma * float(np.linalg.norm(D["y"])),
                                 setup=SETUP, grid=GRID, tol=1e-5, max_iter=10)
        zf = fw["z_debiased"]
        others["Frank-Wolfe"] = (zf, SETUP.physical_intensities(zf, fw["I_debiased"]))
        for name, (zo, Io) in others.items():
            ro = match_nd(z, I, zo, Io)
            print(f"{name:12s} : K = {len(Io)}, erreur max {ro['err_pos']:.3g} -> "
                  f"{'RÉUSSITE' if success_nd(ro, z) else 'échec'}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.2, 7.0))
    ax = fig.add_subplot(111, projection="3d" if args.dim == 3 else None)
    S = D["sensors"]
    ax.scatter(*S.T, s=8, color=INK_MUTED, label="capteurs")
    ax.scatter(*z.T, s=60, facecolors="none", edgecolors=SERIES[0], linewidths=2,
               label="sources vraies")
    ax.scatter(*np.atleast_2d(r["z"]).T, s=30, marker="x", color=SERIES[1], linewidths=2,
               label="ML + BIC")
    for k, (name, (zo, _)) in enumerate(others.items()):
        ax.scatter(*np.atleast_2d(zo).T, s=25, marker="+", color=SERIES[2 + k], label=name)
    if args.dim == 2:
        for zk, ck in zip(z, crb):
            ax.add_patch(plt.Circle(zk, 2 * ck, fill=False, color=SERIES[0], lw=1, ls=":"))
        b = D["box"]
        ax.add_patch(plt.Rectangle(b.lo, *(np.asarray(b.hi) - b.lo), fill=False,
                                   color=INK_MUTED, lw=1))
        ax.set_aspect("equal")
        style_axes(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"{len(I)} sources, σ = {args.sigma:g} : K estimé = {r['K']}"
                 + (" (cercles : 2 × CRB)" if args.dim == 2 else ""), fontsize=10)
    path = OUTDIR / f"tirage_{args.dim}d_n{len(I)}_s{args.sigma:g}_g{args.graine}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nFigure : {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

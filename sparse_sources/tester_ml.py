"""
Tester le maximum de vraisemblance + BIC sur un tirage, en 2D (scénario du
projet) ou en 3D.

    python sparse_sources/tester_ml.py                         2D, 3 sources, σ = 1e-3
    python sparse_sources/tester_ml.py --sources 5 --sigma 1e-4 --graine 7
    python sparse_sources/tester_ml.py --positions "2,4,30;7,15,70" --sigma 1e-3
    python sparse_sources/tester_ml.py --sans-centrale --comparer
    python sparse_sources/tester_ml.py --dim 3 --capteurs sphere --sources 3
    python sparse_sources/tester_ml.py --rayon 2                cercle 2 fois plus grand
    python sparse_sources/tester_ml.py --interieur 8x16         + 128 capteurs dans D
    python sparse_sources/tester_ml.py --sans-cercle --fichier-capteurs mes_capteurs.txt
    python sparse_sources/tester_ml.py --sources 50 --k-fixe 50 ML à K imposé

Capteurs en 2D : cercle (--rayon, --nb-capteurs, --sans-cercle) + grille dans
D (--interieur, centres des cases) + capteurs placés à la main
(--fichier-capteurs, une ligne « x,y » par capteur ; D = [0, 10] × [0, 20],
source centrale en (5, 10)). Avec d'autres capteurs que ceux du projet, les
sources sont les mêmes (même graine) et les mesures sont recalculées.

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


def read_sensors(path):
    """Fichier texte de capteurs : une ligne « x,y » (ou « x y ») par capteur, # = commentaire."""
    pts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].replace(";", ",").replace(",", " ").split()
        if line:
            if len(line) != 2:
                raise SystemExit(f"{path} : 2 nombres (x, y) attendus par ligne")
            pts.append([float(v) for v in line])
    if not pts:
        raise SystemExit(f"{path} : aucun capteur")
    return np.array(pts)


def custom_geometry(args):
    """
    Setup 2D construit à partir des options de capteurs, ou None si ce sont
    les capteurs du projet (cercle circonscrit à D, 64 capteurs, h = 0).

    Capteurs = cercle (rayon --rayon × rayon circonscrit, --nb-capteurs
    capteurs ; retiré par --sans-cercle) + grille --interieur n×m à
    l'intérieur de D (centres des cases) + capteurs de --fichier-capteurs.
    """
    from sparse_sources import make_setup
    from sparse_sources.kernel import Setup
    if (args.rayon == 1.0 and args.nb_capteurs is None and not args.interieur
            and not args.fichier_capteurs and not args.sans_cercle and args.hauteur is None):
        return None
    h = 0.0 if args.hauteur is None else args.hauteur
    circle = make_setup(h=h, n_sensors=args.nb_capteurs, radius_factor=args.rayon)
    box = circle.domain
    parts = [] if args.sans_cercle else [circle.sensors]
    if args.interieur:
        nx, ny = (int(v) for v in (args.interieur.lower().split("x") * 2)[:2])
        lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
        xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
        ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
        parts.append(np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2))
    if args.fichier_capteurs:
        parts.append(read_sensors(args.fichier_capteurs))
    if not parts:
        raise SystemExit("aucun capteur : --sans-cercle demande --interieur ou --fichier-capteurs")
    sensors = np.vstack(parts)
    if (h == 0 and not args.sans_centrale
            and np.min(np.linalg.norm(sensors - np.asarray(box.center), axis=1)) < 1e-9):
        raise SystemExit("un capteur est sur la source centrale (h = 0) : déplacez-le "
                         "ou donnez --hauteur > 0")
    return Setup(sensors, box, h, None, normalize=True)


def search_grid(setup, step=0.25):
    """
    Grille de recherche de ML dont aucun nœud ne tombe sur un capteur (à
    hauteur nulle, le signal y serait infini) : on réduit un peu le pas si
    besoin.
    """
    from sparse_sources import build_grid
    for k in range(20):
        with np.errstate(divide="ignore", invalid="ignore"):   # essais rejetés : nœud sur un capteur
            g = build_grid(step=step * (1 - 0.03 * k), setup=setup)
        pts = g.points.reshape(-1, 2)
        d = np.min(np.linalg.norm(pts[:, None] - setup.sensors[None], axis=-1))
        if setup.h > 0 or d > 1e-6:
            return g
    raise SystemExit("impossible de construire une grille qui évite les capteurs")


def data_2d(args):
    """Tirage 2D du projet (ou sources imposées) -> dict."""
    from sparse_sources import build_grid, relative_noise_std
    from sparse_sources.experiments_projet import GRID, SETUP, scenario
    factor = 0 if args.sans_centrale else None
    cfg, src, y, m = scenario(args.sources, args.sigma, args.graine, central_factor=factor)
    z, I = src.positions, src.intensities
    if args.positions:
        z, I = parse_positions(args.positions, 2)
    geom = custom_geometry(args)
    if geom is not None:            # mêmes sources, autres capteurs : on recalcule les mesures
        raw = geom.with_power(0.0)
        c = 0.0 if args.sans_centrale else 5000.0 * raw.phi(
            np.asarray(geom.domain.center, float)[None])[0]
        y = raw.phi(z).T @ I + c
        rng = np.random.default_rng(args.graine)
        m = y * (1 + args.sigma * rng.standard_normal(len(y))) - c
        return {"cfg": cfg, "z": z, "I": I, "y": y, "m": m,
                "s": relative_noise_std(y, args.sigma), "sensors": geom.sensors,
                "box": geom.domain, "setup": geom, "grid": search_grid(geom), "projet": False}
    if args.positions:
        raw = SETUP.with_power(0.0)
        c = 0.0 if args.sans_centrale else 5000.0 * raw.phi(
            np.asarray(SETUP.domain.center, float)[None])[0]
        y = raw.phi(z).T @ I + c
        rng = np.random.default_rng(args.graine)
        m = y * (1 + args.sigma * rng.standard_normal(len(y))) - c
    return {"cfg": cfg, "z": z, "I": I, "y": y, "m": m, "s": relative_noise_std(y, args.sigma),
            "sensors": SETUP.sensors, "box": SETUP.domain, "setup": SETUP, "grid": GRID,
            "projet": True}


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
    sys.stderr.reconfigure(encoding="utf-8")
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
    g = p.add_argument_group("capteurs en 2D (par défaut : ceux du projet)")
    g.add_argument("--rayon", type=float, default=1.0,
                   help="rayon du cercle = RAYON × rayon du cercle circonscrit à D (défaut 1)")
    g.add_argument("--nb-capteurs", type=int, default=None, help="capteurs sur le cercle (64)")
    g.add_argument("--sans-cercle", action="store_true", help="pas de capteurs sur le cercle")
    g.add_argument("--interieur", default=None,
                   help="grille de capteurs dans D : 'n' (n×n) ou 'nxm', centres des cases")
    g.add_argument("--fichier-capteurs", default=None,
                   help="capteurs placés à la main : fichier texte, une ligne 'x,y' par capteur")
    g.add_argument("--hauteur", type=float, default=None,
                   help="hauteur des capteurs au-dessus du plan des sources (défaut 0)")
    k = p.add_argument_group("nombre de sources")
    k.add_argument("--k-max", type=int, default=None, help="nombre maximal de sources (10)")
    k.add_argument("--k-fixe", type=int, default=None,
                   help="impose exactement K sources (pas de BIC : ML à K connu)")
    args = p.parse_args()
    crit = args.critere if args.critere in ("bic", "aic") else float(args.critere)
    opts = {}
    if args.k_max is not None:
        opts["k_max"] = args.k_max
    if args.k_fixe is not None:     # κ = 0 : on ajoute jusqu'à K sans jamais s'arrêter ni retirer
        crit = 0.0
        opts.update(k_max=args.k_fixe, patience=10**9, backward=False)

    from sparse_sources.ml3d import cramer_rao_nd, greedy_ml_nd, match_nd, success_nd
    D = data_2d(args) if args.dim == 2 else data_3d(args)
    if args.dim == 2:
        from sparse_sources import cramer_rao, greedy_ml
        r = greedy_ml(D["m"], D["s"], setup=D["setup"], grid=D["grid"], criterion=crit, **opts)
    else:
        r = greedy_ml_nd(D["m"], D["s"], D["sensors"], D["box"], criterion=crit, **opts)
    z, I = D["z"], D["I"]
    if args.dim == 2:               # tient compte de la hauteur des capteurs
        crb = cramer_rao(z, I, D["s"], D["setup"])["pos_std"]
    else:
        crb = cramer_rao_nd(z, I, D["s"], D["sensors"])["pos_std"]
    res = match_nd(z, I, r["z"], r["I"])
    M = len(D["m"])

    print(f"\n{args.dim}D, {len(I)} sources vraies, σ = {args.sigma:g}, "
          f"source centrale : {'non' if args.sans_centrale else 'oui (5000, retirée)'}, "
          f"M = {M} capteurs")
    print("\nSources vraies (et borne de Cramér-Rao sur la position) :")
    for zk, Ik, ck in zip(z, I, crb):
        print(f"  position {np.round(zk, 3)}  intensité {Ik:7.2f}  CRB {ck:.3g}")
    label = (f"ML, K = {args.k_fixe} imposé" if args.k_fixe
             else f"ML + {str(args.critere).upper()}")
    print(f"\n{label} : K = {r['K']} "
          f"({r['time']:.2f} s), χ² / M = {r['chi2'] / M:.2f}")
    for zk, Ik in zip(r["z"], r["I"]):
        print(f"  position {np.round(zk, 3)}  intensité {Ik:7.2f}")
    print("\nÉvolution du critère : " + ", ".join(
        f"K={h['K']}: {h['criterion']:.1f}" for h in r["history"]))
    print(f"\nErreur de position max {res['err_pos']:.3g}, erreur d'intensité relative max "
          f"{res['err_int']:.3g} -> {'RÉUSSITE' if success_nd(res, z) else 'échec'} "
          f"(bon K et erreur < 0,25 × min(séparation, 4))")
    from sparse_sources.experiments_ml import found_within
    print(f"Vraies sources retrouvées à moins de 0,5 : "
          f"{found_within(np.atleast_2d(z), np.atleast_2d(r['z']))} sur {len(I)}")

    others = {}
    projet = D.get("projet", True)
    if args.comparer and args.dim == 2 and not projet:
        print("--comparer ignoré : Prony a besoin des capteurs du projet")
    if args.comparer and args.dim == 2 and projet:
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
               label=label)
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
    tag = "" if projet else f"_M{M}_r{args.rayon:g}"
    path = OUTDIR / f"tirage_{args.dim}d_n{len(I)}_s{args.sigma:g}_g{args.graine}{tag}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nFigure : {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

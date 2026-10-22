"""
BLASSO en 3D : sources dans un cube, tirées comme dans le reste du projet
(nombre ~ loi de la config, Poisson par défaut ; intensités et positions
selon les lois de la config), capteurs hors d'un même plan.

Pour chaque tirage et chaque disposition de capteurs :
  1. identifiabilité : rang de Gamma (4K attendu en 3D) ;
  2. condition de source non dégénérée de Duval-Peyré (H1, H2, H3) ;
  3. reconstruction par BLASSO (Frank-Wolfe glissant + débiaisage) sur
     données bruitées, et erreurs.

    python scripts/duval_peyre_3d.py
    python scripts/duval_peyre_3d.py --set sources.count_law.params.lam=5
    python scripts/duval_peyre_3d.py --dispositions sphere --capteurs 200 --tirages 5

La source centrale de la config (central_source) est placée au centre du cube
si elle est activée ; sa contribution, connue, est retirée des mesures avant
l'inversion, comme dans scripts/reconstruct.py. Sorties : outputs/duval_peyre_3d/.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particle_sources.config import load_config  # noqa: E402
from particle_sources.config.cli import DEFAULT_CONFIG_DIR  # noqa: E402
from particle_sources.duval_peyre import (InverseSquareKernel, diagnose,  # noqa: E402
                                          gamma_matrix)
from particle_sources.duval_peyre.blasso import (Estimate, debias, operator_norm,  # noqa: E402
                                                 sliding_frank_wolfe)
from particle_sources.duval_peyre.experiments import write_csv  # noqa: E402
from particle_sources.duval_peyre import experiments_3d as ex3  # noqa: E402
from particle_sources.generation import central_source, sample_sources  # noqa: E402
from particle_sources.model import Box, Sources  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def significant(est: Estimate, total: float, rel: float) -> Estimate:
    """Ne garde que les Diracs d'intensité > rel * total (les autres sont du « bruit de masse »)."""
    keep = est.intensities > rel * total
    return Estimate(est.positions[keep], est.intensities[keep])


def main():
    """Tirages -> (rang, certificat, BLASSO) pour chaque disposition -> console, CSV, figures."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR")
    parser.add_argument("--cube", default="0,10", help="bornes 'a,b' : cube [a,b]^3")
    parser.add_argument("--dispositions", default="un_cercle,deux_cercles,trois_cercles,sphere")
    parser.add_argument("--capteurs", type=int, default=160, help="nombre M de capteurs")
    parser.add_argument("--rayon", type=float, default=1.5,
                        help="rayon des capteurs / rayon de la sphère circonscrite au cube")
    parser.add_argument("--tirages", type=int, default=3)
    parser.add_argument("--bruit", default="1e-6,1e-4,1e-2",
                        help="niveaux de bruit relatif (gaussien additif)")
    parser.add_argument("--c", type=float, default=4.0, help="lambda = c ||bruit|| ||Phi||")
    parser.add_argument("--tol", type=float, default=0.5,
                        help="distance sous laquelle une source est « retrouvée »")
    parser.add_argument("--seuil", type=float, default=0.005,
                        help="Diracs d'intensité < seuil * intensité totale ignorés")
    parser.add_argument("--fusion", type=float, default=0.0,
                        help="fusionne les sources à moins de cette distance (0 = aucune fusion)")
    parser.add_argument("--sortie", default=Path(DEFAULT_CONFIG_DIR).parent / "outputs" / "duval_peyre_3d")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    a, b = (float(v) for v in args.cube.split(","))
    box = Box.from_bounds([[a, b]] * 3)
    rng = cfg.rng()
    layouts = args.dispositions.split(",")
    levels = [float(v) for v in args.bruit.split(",")]
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    h_grid = float(box.size[0]) / 20

    central = Sources.empty(3)
    cs = cfg.raw.get("central_source") or {}
    if cs.get("enabled", False):
        central = central_source(box, float(cs["intensity_factor"]) * cfg.total_intensity)
    print(f"Cube [{a:g}, {b:g}]^3, M = {args.capteurs} capteurs à {args.rayon:g} x rayon "
          f"circonscrit, graine {cfg.seed}, source centrale : "
          f"{'I = %g (retirée avant inversion)' % central.intensities[0] if central.n else 'non'}\n")

    rows = []
    first = {}
    for draw in range(args.tirages):
        src = sample_sources(box, cfg.count_law, cfg.total_intensity, rng,
                             cfg.intensity_law, cfg.position_law)
        if args.fusion > 0 and src.n > 1:
            n0 = src.n
            src = ex3.merge_close(src, args.fusion)
            print(f"Tirage {draw + 1} : {n0} sources tirées -> K = {src.n} après fusion "
                  f"à {args.fusion:g}", flush=True)
        else:
            print(f"Tirage {draw + 1} : K = {src.n} sources", flush=True)
        if src.n == 0:
            continue
        for name in layouts:
            S = ex3.layout(name, box, args.capteurs, args.rayon)
            k = InverseSquareKernel(S)
            t = time.time()
            # 1. identifiabilité
            sv = np.linalg.svd(gamma_matrix(k, src.positions), compute_uv=False)
            rank = int(np.sum(sv > sv[0] * 1e-10))
            # 2. certificat
            diag = diagnose(k, src.positions, box, h=h_grid)
            base = {"tirage": draw + 1, "disposition": name, "K": src.n, "M": k.m,
                    "rang_Gamma": rank, "rang_attendu": 4 * src.n, "H1": diag.h1, "H2": diag.h2,
                    "H3": diag.h3, "non_degenere": diag.nondegenerate, "marge": diag.margin,
                    "vp_hess_max": float(diag.eigenvalues[:, -1].max()), "cond": diag.cond}
            # 3. BLASSO
            y_true = k.forward(src.positions, src.intensities)
            y_central = k.forward(central.positions, central.intensities) if central.n else 0.0
            y_all = y_true + y_central
            nP = operator_norm(k, box, h_grid)
            line = []
            for lvl in levels:
                eps = lvl * np.linalg.norm(y_all) / np.sqrt(k.m) * rng.standard_normal(k.m)
                m = y_all + eps - y_central          # source centrale connue : retirée
                lam = args.c * np.linalg.norm(eps) * nP
                est, it, _ = sliding_frank_wolfe(k, m, lam, box, h_grid, max_iter=2 * src.n + 5)
                deb = significant(debias(k, m, est), src.total_intensity, args.seuil)
                res = ex3.match_many(src.positions, src.intensities, deb, args.tol)
                rows.append({**base, "niveau": lvl, "norme_bruit": float(np.linalg.norm(eps)),
                             "lambda": lam, "iter_sfw": it, "n_est_brut": est.n, **res})
                line.append(f"bruit {lvl:.0e}: {res['retrouvees']}/{src.n} retrouvées, "
                            f"{res['parasites']} parasites")
                if (name, lvl) not in first:
                    first[(name, lvl)] = (src, S, deb, diag, k)
            print(f"   {name:14s} rang {rank:3d}/{4 * src.n:<3d} H1={'V' if diag.h1 else 'F'} "
                  f"H2={'V' if diag.h2 else 'F'} H3={'V' if diag.h3 else 'F'} | "
                  + " ; ".join(line) + f"   ({time.time() - t:.0f} s)", flush=True)

    write_csv(rows, out / "resultats_3d.csv")
    print(f"\n  -> {out / 'resultats_3d.csv'}")

    # Résumé par disposition
    print("\nRésumé (sur tous les tirages) :")
    print("   disposition     rang plein  cond. vérifiée  " +
          "  ".join(f"retrouvées@{l:.0e}" for l in levels))
    for name in layouts:
        sub = [r for r in rows if r["disposition"] == name]
        if not sub:
            continue
        draws = {r["tirage"]: r for r in sub}.values()
        full = np.mean([r["rang_Gamma"] == r["rang_attendu"] for r in draws])
        ok = np.mean([r["non_degenere"] for r in draws])
        rec = [np.sum([r["retrouvees"] for r in sub if r["niveau"] == l]) /
               np.sum([r["K"] for r in sub if r["niveau"] == l]) for l in levels]
        print(f"   {name:14s} {full:9.0%}  {ok:12.0%}    " + "  ".join(f"{v:13.0%}" for v in rec))

    # Figures : taux de sources retrouvées, scènes 3D, coupe de eta_V
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in layouts:
        sub = [r for r in rows if r["disposition"] == name]
        rec = [np.sum([r["retrouvees"] for r in sub if r["niveau"] == l]) /
               max(np.sum([r["K"] for r in sub if r["niveau"] == l]), 1) for l in levels]
        par = [np.mean([r["parasites"] for r in sub if r["niveau"] == l]) for l in levels]
        line, = ax.semilogx(levels, rec, "o-", lw=2, ms=7, label=f"{name} (retrouvées)")
        ax.semilogx(levels, np.array(par) / max(np.mean([r["K"] for r in sub]), 1), "s--",
                    c=line.get_color(), lw=1, ms=5, mfc="none", label=f"{name} (parasites / K)")
    ax.set_xlabel("bruit relatif")
    ax.set_ylabel(f"proportion (retrouvée = à moins de {args.tol:g})")
    ax.set_title("BLASSO 3D : sources retrouvées et parasites", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "taux_retrouvees.png", dpi=130)
    plt.close(fig)
    print(f"  -> {out / 'taux_retrouvees.png'}")

    for name in layouts:
        if (name, levels[0]) not in first:
            continue
        src, S, deb, diag, k = first[(name, levels[0])]
        fig = ex3.plot_scene(src, S, box, deb, title=f"{name}, tirage 1, bruit {levels[0]:.0e}\n"
                             f"{deb.n} sources estimées (significatives) pour K = {src.n}")
        fig.savefig(out / f"scene_{name}.png", dpi=130)
        plt.close(fig)
        print(f"  -> {out / f'scene_{name}.png'}")
        if diag.h1:
            zk = src.positions[np.argmax(src.intensities)]
            fig = ex3.plot_eta_slice(k, diag.p, src.positions, box, zk[2],
                                     title=f"$\\eta_V$, coupe z = {zk[2]:.2f} ({name}, tirage 1)")
            fig.savefig(out / f"eta_V_coupe_{name}.png", dpi=130)
            plt.close(fig)
            print(f"  -> {out / f'eta_V_coupe_{name}.png'}")


if __name__ == "__main__":
    main()

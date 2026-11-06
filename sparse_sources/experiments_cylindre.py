"""
Capteurs sur la paroi d'un cylindre : est-ce que le Frank-Wolfe (BLASSO) marche
alors pour beaucoup de sources et beaucoup de bruit ?

    python sparse_sources/experiments_cylindre.py            (~30 min)
    python sparse_sources/experiments_cylindre.py --rapide

Les sources restent dans le plan de D. Les capteurs sont sur un ou plusieurs
cercles (« étages ») de même rayon que le cercle du projet, aux hauteurs h :
même noyau G = 1 / (||x_i - z||² + h_i²). Même nombre total de capteurs
(M = 60) pour toutes les géométries :

    cercle             1 étage  h = 0 (comme le projet)
    cylindre_bas       3 étages h = 0 ; 2,5 ; 5
    cylindre_haut      5 étages h = 0 ; 2,5 ; 5 ; 7,5 ; 10
    cylindre_sans_plan 3 étages h = 2,5 ; 5 ; 7,5 (aucun capteur dans le plan)

Données : scénario du projet. K sources (K fixé et connu), positions
uniformes dans D mais ESPACÉES (distance minimale 3), intensités uniformes
sur le simplexe de somme 100, source centrale 5000 au centre de D (sur l'axe
du cylindre), bruit relatif y (1 + σ ε), source centrale retirée.

Méthodes (K connu à l'avance) :
    sfw_K    Frank-Wolfe glissant, BLASSO pondéré α = 1, λ « optimal pour K » :
             on parcourt λ = t λ_max (t de 0,5 à 3e-4) et on garde, parmi les λ
             qui donnent exactement K sources, celui dont le résidu (après
             débiaisage) est le plus petit. Si aucun λ ne donne K sources, on
             prend le premier λ qui en donne plus et on garde les K plus
             intenses (puis débiaisage).
    ml_K     maximum de vraisemblance (``greedy_ml``) avec K sources imposées.

Pour chaque tirage on calcule aussi la borne de Cramér-Rao (pire source) et
la condition de Duval-Peyré (H1-H3, ``check_hypotheses`` sur les vraies
positions, α = 1).

Sorties : sparse_sources/results/cylindre.csv, cylindre_resume.csv,
cylindre.png.
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

from sparse_sources import (build_grid, check_hypotheses, cramer_rao, debias,  # noqa: E402
                            greedy_ml, make_cylinder_setup, match_sources,
                            relative_noise_std, sliding_frank_wolfe)
from sparse_sources.experiments import (OUT, min_separation, save, success,  # noqa: E402
                                        summarize, write_csv)
from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

M_TOTAL = 60
GEOMETRIES = {
    "cercle": [0.0],
    "cylindre_bas": [0.0, 2.5, 5.0],
    "cylindre_haut": [0.0, 2.5, 5.0, 7.5, 10.0],
    "cylindre_sans_plan": [2.5, 5.0, 7.5],
}
KS = [3, 5, 8]
SIGMAS = [1e-4, 1e-3, 1e-2]
SEP_MIN = 3.0
I_CENTRAL = 5000.0
LAMBDA_T = np.geomspace(0.5, 3e-4, 14)


def setups():
    """{nom: (setup α = 1, grille)} pour chaque géométrie, M = 60 capteurs."""
    out = {}
    for name, levels in GEOMETRIES.items():
        S = make_cylinder_setup(levels, M_TOTAL // len(levels), normalize=True)
        out[name] = (S, build_grid(step=0.25, setup=S))
    return out


def draw_sources(K, rng, box, sep=SEP_MIN):
    """K positions uniformes dans D à distance ≥ sep, intensités ~ simplexe (somme 100)."""
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    for _ in range(10000):
        z = np.empty((0, 2))
        while len(z) < K:
            p = lo + (hi - lo) * rng.random(2)
            if len(z) == 0 or np.min(np.linalg.norm(z - p, axis=1)) >= sep:
                z = np.vstack([z, p])
            elif rng.random() < 1e-3:
                break
        if len(z) == K:
            return z, 100.0 * rng.dirichlet(np.ones(K))
    raise RuntimeError("impossible de placer les sources")


def measurements(S, z, I, sigma, rng):
    """Mesures du scénario : sources + centrale, bruit relatif, centrale retirée."""
    raw = S.with_power(0.0)
    c = np.asarray(S.domain.center, float)[None]
    central = I_CENTRAL * raw.phi(c)[0]
    y = raw.phi(z).T @ I + central
    noisy = y * (1 + sigma * rng.standard_normal(len(y)))
    return noisy - central, relative_noise_std(y, sigma)


def sfw_fixed_k(S, G, m, K):
    """Frank-Wolfe avec le λ « optimal » pour K sources (voir docstring du module)."""
    lam_max = float(np.max(G.G.T @ m))
    best, fallback = None, None
    for t in LAMBDA_T:
        fw = sliding_frank_wolfe(m, t * lam_max, setup=S, grid=G, tol=1e-5,
                                 max_iter=K + 6)
        z, I = fw["z_debiased"], fw["I_debiased"]
        if len(I) == K:
            res = float(np.linalg.norm(m - S.phi(z).T @ I))
            if best is None or res < best[0]:
                best = (res, z, I, t)
        elif len(I) > K and fallback is None:
            fallback = (z, I, t)
            if best is not None:
                break
        if len(I) > K + 3:
            break
    if best is not None:
        _, z, I, t = best
        return z, S.physical_intensities(z, I), t, True
    if fallback is None:
        return np.empty((0, 2)), np.empty(0), np.nan, False
    z, I, t = fallback
    keep = np.argsort(I)[::-1][:K]
    z, I = debias(z[keep], I[keep], m, setup=S)
    return z, S.physical_intensities(z, I), t, False


def run(n_draws):
    """Toutes les géométries, K et σ, mêmes sources pour toutes les géométries."""
    geoms = setups()
    box = next(iter(geoms.values()))[0].domain
    rows = []
    for K in KS:
        for sigma in SIGMAS:
            t0 = time.perf_counter()
            for d in range(n_draws):
                rng = np.random.default_rng(1000 * K + d)
                z0, I0 = draw_sources(K, rng, box)
                sep = min_separation(z0)
                for g, (S, G) in geoms.items():
                    m, s = measurements(S, z0, I0, sigma, np.random.default_rng(7 * d + K))
                    crb = float(cramer_rao(z0, I0, s, S)["pos_std"].max())
                    hyp = check_hypotheses(z0, setup=S, verbose=False, plot=False)
                    base = {"geometrie": g, "K": K, "sigma": sigma, "tirage": d,
                            "sep_min": sep, "I_min": float(I0.min()), "crb_max": crb,
                            "faisable": bool(2 * crb < 0.25 * min(sep, 4.0)),
                            "condition": bool(hyp["non_degenerate"]),
                            "marge_relative": float(hyp["margin_relative"]),
                            "cond_gamma": float(hyp["cond_normalized"])}
                    tic = time.perf_counter()
                    zf, If, t_lam, exact = sfw_fixed_k(S, G, m, K)
                    t_fw = time.perf_counter() - tic
                    r_ml = greedy_ml(m, s, setup=S, grid=G, k_max=K, criterion=0.0,
                                     backward=False)
                    for meth, (zh, Ih, tt, extra) in {
                            "sfw_K": (zf, If, t_fw, {"lambda_sur_lmax": t_lam,
                                                     "lambda_donne_K": exact}),
                            "ml_K": (r_ml["z"], r_ml["I"], r_ml["time"], {})}.items():
                        res = match_sources(z0, I0, zh, Ih)
                        rows.append({**base, "methode": meth, "n_est": res["n_est"],
                                     "err_pos": res["err_pos"], "err_int": res["err_int"],
                                     "temps": tt, "reussite": success(res, z0), **extra})
            print(f"  K = {K}, σ = {sigma:g} fait ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    return rows


def report(rows):
    """Résumé console + CSV + figure."""
    write_csv(rows, OUT / "cylindre.csv")
    summ = summarize(rows, ["geometrie", "K", "sigma", "methode"],
                     ["err_pos", "crb_max", "temps"])
    write_csv(summ, OUT / "cylindre_resume.csv")

    def rate(sel, key="reussite"):
        return 100 * np.mean([r[key] for r in sel]) if sel else np.nan

    print(f"\n{'géométrie':20s} {'K':>2s} {'σ':>7s} {'FW':>5s} {'ML':>5s} {'cond.':>6s} "
          f"{'faisab.':>7s} {'CRB méd.':>9s} {'cond Γ méd.':>11s}")
    for g in GEOMETRIES:
        for K in KS:
            for sigma in SIGMAS:
                sel = [r for r in rows if (r["geometrie"], r["K"], r["sigma"]) == (g, K, sigma)]
                fw = [r for r in sel if r["methode"] == "sfw_K"]
                ml = [r for r in sel if r["methode"] == "ml_K"]
                print(f"{g:20s} {K:2d} {sigma:7.0e} {rate(fw):4.0f}% {rate(ml):4.0f}% "
                      f"{rate(fw, 'condition'):5.0f}% {rate(fw, 'faisable'):6.0f}% "
                      f"{np.median([r['crb_max'] for r in fw]):9.3g} "
                      f"{np.median([r['cond_gamma'] for r in fw]):11.3g}")

    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(12, 3.9), sharey=True)
    for ax, sigma in zip(axes, SIGMAS):
        for k, g in enumerate(GEOMETRIES):
            for meth, ls, mk in [("sfw_K", "-", "o"), ("ml_K", "--", "s")]:
                sel = [[r for r in rows if (r["geometrie"], r["K"], r["sigma"], r["methode"])
                        == (g, K, sigma, meth)] for K in KS]
                ax.plot(KS, [rate(x) / 100 for x in sel], ls, marker=mk, color=SERIES[k],
                        lw=2, ms=5,
                        label=f"{g} ({'Frank-Wolfe' if meth == 'sfw_K' else 'ML'})")
        ax.set_title(f"σ = {sigma:g}", fontsize=10)
        ax.set_xticks(KS)
        ax.set_xlabel("nombre de sources K (connu)")
        ax.set_ylim(-0.05, 1.05)
        style_axes(ax)
    axes[0].set_ylabel("taux de réussite")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.suptitle("Capteurs sur un cylindre (M = 60) : traits pleins = Frank-Wolfe, "
                 "tirets = maximum de vraisemblance", fontsize=10, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "cylindre.png")


def main():
    """Lance l'expérience."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rapide", action="store_true", help="2 tirages par case")
    p.add_argument("--tirages", type=int, default=None)
    args = p.parse_args()
    n = args.tirages or (2 if args.rapide else 15)
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    rows = run(n)
    report(rows)
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    main()

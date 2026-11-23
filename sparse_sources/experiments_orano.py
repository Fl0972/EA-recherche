"""
Méthode Orano (descente de gradient projetée, N connu) contre Gauss-Newton
(même perte, N connu) et greedy_ml du dépôt (ML + BIC, N inconnu), sur les
mêmes situations et les mêmes initialisations.

    python sparse_sources/experiments_orano.py                          N = 2, 3, 5, 10 ; sans bruit
    python sparse_sources/experiments_orano.py --sigma 1e-3 --centrale
    python sparse_sources/experiments_orano.py --sources 2 --rapport 1 10 100
    python sparse_sources/experiments_orano.py --sources 5 --figure

Situations : positions uniformes dans D, séparation minimale 0,5 ; intensités
de somme 100, tirées sur le simplexe (comme config/) ou, avec --rapport x,
réparties géométriquement de x·I à I (rapport max/min = x), dans un ordre
aléatoire. Mesures : y_brut (1 + σ ε), source centrale 5000 en (5, 10)
ajoutée puis retirée si --centrale. Graine fixe par tirage : la situation, le
bruit et l'initialisation ne dépendent que de (N, rapport, tirage).

Initialisation (descente et Gauss-Newton) : positions uniformes dans D,
intensités égales. greedy_ml n'utilise pas d'initialisation ni N ; sans bruit,
sa vraisemblance suppose un bruit relatif de 1e-6.

Évaluation : appariement optimal vraies / estimées (linear_sum_assignment).
Réussite = bon nombre de sources (intensité > 1e-6 × 100) et chaque vraie
source à moins de 0,1 de son estimée. On rapporte, par méthode : taux de
réussite, nombre moyen de sources retrouvées à 0,1, médiane (sur les tirages)
de l'erreur de position maximale, nombre d'itérations médian (descente :
itérations ; Gauss-Newton : évaluations du résidu ; ML : sources ajoutées),
temps médian.

Sorties : tableau dans le terminal, sparse_sources/results/orano_<cas>.csv
(une ligne par tirage et par méthode), et avec --figure
sparse_sources/results/orano_trajectoires_<cas>.png.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from sparse_sources import greedy_ml, relative_noise_std  # noqa: E402
from sparse_sources.experiments import OUT, write_csv  # noqa: E402
from sparse_sources.experiments_projet import GRID, SETUP  # noqa: E402
from sparse_sources.orano import (I_CENTRALE, I_TOTAL, Probleme, descente_projetee,  # noqa: E402
                                  gauss_newton)

LO, HI = np.asarray(SETUP.domain.lo, float), np.asarray(SETUP.domain.hi, float)
CENTRE = np.asarray(SETUP.domain.center, float)
CAPTEURS = SETUP.sensors                     # 64 capteurs du projet (cercle circonscrit)
METHODES = ["descente (Orano)", "Gauss-Newton", "ML + BIC"]


def graine(N, rapport, tirage, flux):
    """Graine reproductible pour (N, rapport, tirage) et un flux (situation, bruit, init)."""
    x = 0 if rapport is None else int(round(1000 * rapport))
    return np.random.default_rng([N, x, tirage, flux])


def tirer_situation(N, rng, rapport=None, sep_min=0.5):
    """Positions uniformes dans D (séparation ≥ sep_min), intensités de somme I_TOTAL."""
    for _ in range(100000):
        p = LO + rng.random((N, 2)) * (HI - LO)
        d = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(N) * 1e9
        if N == 1 or d.min() >= sep_min:
            break
    else:
        raise RuntimeError("séparation minimale impossible à respecter")
    if rapport is None:
        I = rng.dirichlet(np.ones(N))
    else:
        I = float(rapport) ** (-np.arange(N) / max(N - 1, 1))   # de 1 à 1/x : rapport max/min = x
        rng.shuffle(I)
    return p, I / I.sum() * I_TOTAL


def mesures(p, I, sigma, centrale, rng):
    """(y sans la source centrale, y brut sans bruit) aux capteurs."""
    d2 = ((CAPTEURS[:, None] - p[None]) ** 2).sum(-1)
    y = (I / d2).sum(1)
    yc = I_CENTRALE / ((CAPTEURS - CENTRE) ** 2).sum(1) if centrale else 0.0
    y_brut = y + yc
    y_bruite = y_brut * (1 + sigma * rng.standard_normal(len(y))) if sigma > 0 else y_brut
    return y_bruite - yc, y_brut


def evaluer(p_vrai, p_est, I_est, tol=0.1):
    """Appariement optimal ; renvoie (réussite, sources à tol, erreur de position max, n estimées)."""
    garde = np.asarray(I_est) > 1e-6 * I_TOTAL
    p_est = np.asarray(p_est).reshape(-1, 2)[garde]
    if len(p_est) == 0:
        return False, 0, np.inf, 0
    C = np.linalg.norm(p_vrai[:, None] - p_est[None], axis=-1)
    i, j = linear_sum_assignment(C)
    err = C[i, j]
    n_ok = int((err < tol).sum())
    return bool(len(p_est) == len(p_vrai) and n_ok == len(p_vrai)), n_ok, float(err.max()), len(p_est)


def un_tirage(N, rapport, t, args, trajectoire=False):
    """Situation, mesures, puis les trois méthodes. Renvoie (situation, {méthode: résultat})."""
    p, I = tirer_situation(N, graine(N, rapport, t, 0), rapport, args.sep_min)
    y, y_brut = mesures(p, I, args.sigma, args.centrale, graine(N, rapport, t, 1))
    rng0 = graine(N, rapport, t, 2)
    u0, a0 = rng0.random((N, 2)), np.full(N, 1.0 / N)            # même initialisation
    pb = Probleme(CAPTEURS, LO, HI, y)
    out = {"descente (Orano)": descente_projetee(pb, u0, a0, args.iter, trajectoire=trajectoire),
           "Gauss-Newton": gauss_newton(pb, u0, a0)}
    if not args.sans_ml:
        s = relative_noise_std(y_brut, args.sigma if args.sigma > 0 else 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = greedy_ml(y, s, setup=SETUP, grid=GRID, k_max=N + 10)
        out["ML + BIC"] = {"p": np.atleast_2d(r["z"]).reshape(-1, 2), "I": np.asarray(r["I"]),
                           "iterations": len(r["history"]) - 1, "temps": r["time"]}
    return (p, I, u0 * (HI - LO) + LO), out


def nom_cas(args):
    """Suffixe des fichiers de sortie selon les options."""
    s = f"s{args.sigma:g}" + ("_centrale" if args.centrale else "")
    if args.rapport:
        s += "_rapport" + "-".join(f"{x:g}" for x in args.rapport)
    return s + "_N" + "-".join(map(str, args.sources))


def experience(args):
    """Toutes les situations ; tableau, CSV."""
    rows = []
    rapports = args.rapport or [None]
    methodes = METHODES[:2] if args.sans_ml else METHODES
    print(f"\nσ = {args.sigma:g}, source centrale : {'oui' if args.centrale else 'non'}, "
          f"{args.tirages} tirages par cas, réussite : bon nombre et chaque source à < {args.tol}")
    print(f"{'N':>3} {'rapport':>8} | {'méthode':<17} | {'réussite':>8} | {'à 0,1 (moy.)':>12} | "
          f"{'err. pos. méd.':>14} | {'itér. méd.':>10} | {'temps méd.':>10}")
    for N in args.sources:
        for x in rapports:
            for t in range(args.tirages):
                (p, I, _), out = un_tirage(N, x, t, args)
                for m, r in out.items():
                    ok, n_ok, err, n_est = evaluer(p, r["p"], r["I"], args.tol)
                    rows.append({"N": N, "rapport": "simplexe" if x is None else x, "tirage": t,
                                 "methode": m, "reussite": ok, "trouvees_0_1": n_ok,
                                 "n_estimees": n_est, "err_pos_max": err,
                                 "iterations": r["iterations"], "temps": r["temps"],
                                 "sigma": args.sigma, "centrale": args.centrale})
            for m in methodes:
                sel = [r for r in rows if r["N"] == N and r["methode"] == m
                       and r["rapport"] == ("simplexe" if x is None else x)]
                print(f"{N:>3} {('simplexe' if x is None else f'{x:g}'):>8} | {m:<17} | "
                      f"{100 * np.mean([r['reussite'] for r in sel]):>7.0f}% | "
                      f"{np.mean([r['trouvees_0_1'] for r in sel]):>5.1f} / {N:<4} | "
                      f"{np.median([r['err_pos_max'] for r in sel]):>14.3g} | "
                      f"{np.median([r['iterations'] for r in sel]):>10.0f} | "
                      f"{np.median([r['temps'] for r in sel]):>9.2f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"orano_{nom_cas(args)}.csv"
    write_csv(rows, path)
    print(f"\nRésultats par tirage : {path.relative_to(ROOT)}")
    return rows


def figure(args):
    """Trajectoires de la descente sur un tirage (premier N, premier rapport, tirage --tirage-figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sparse_sources.plotting import INK, INK_MUTED, SERIES, style_axes
    N, x, t = args.sources[0], (args.rapport or [None])[0], args.tirage_figure
    (p, I, p0), out = un_tirage(N, x, t, args, trajectoire=True)
    d = out["descente (Orano)"]
    fig, ax = plt.subplots(figsize=(6.0, 8.5))
    ax.scatter(*CAPTEURS.T, s=8, color=INK_MUTED, label="capteurs")
    ax.plot(*np.c_[LO, [HI[0], LO[1]], HI, [LO[0], HI[1]], LO], color=INK_MUTED, lw=1)
    for i in range(N):
        ax.plot(*d["trajectoire"][:, i].T, "-", lw=1, color=SERIES[0], alpha=0.7,
                label="trajectoire de la descente" if i == 0 else None)
    ax.scatter(*p.T, s=40 + 300 * I / I.max(), marker="*", color=INK, label="vraies sources", zorder=4)
    ax.scatter(*p0.T, marker="x", s=50, color=SERIES[3], label="initialisation", zorder=4)
    ax.scatter(*d["p"].T, s=90, facecolors="none", edgecolors=SERIES[0], linewidths=2,
               label="fin de la descente", zorder=5)
    gn = out["Gauss-Newton"]
    ax.scatter(*gn["p"].T, s=40, marker="+", color=SERIES[1], linewidths=2, label="Gauss-Newton",
               zorder=5)
    if "ML + BIC" in out:
        ax.scatter(*out["ML + BIC"]["p"].T, s=30, marker="D", facecolors="none",
                   edgecolors=SERIES[2], linewidths=1.5, label="ML + BIC", zorder=5)
    res = {m: evaluer(p, r["p"], r["I"], args.tol) for m, r in out.items()}
    ax.set_aspect("equal")
    style_axes(ax)
    ax.set_title(f"{N} sources, rapport {'simplexe' if x is None else f'{x:g}'}, σ = {args.sigma:g}"
                 f"{', centrale' if args.centrale else ''}, tirage {t}\n"
                 + " ; ".join(f"{m} : {v[1]}/{N} à {args.tol}" for m, v in res.items())
                 + f"\ndescente : {d['iterations']} itérations", fontsize=9, color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    path = OUT / f"orano_trajectoires_{nom_cas(args)}_t{t}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"Figure : {path.relative_to(ROOT)}")


def main():
    """Arguments de la ligne de commande."""
    sys.stdout.reconfigure(encoding="utf-8")
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--sources", type=int, nargs="+", default=[2, 3, 5, 10], help="nombres de sources N")
    pa.add_argument("--rapport", type=float, nargs="+", default=None,
                    help="rapport x max/min des intensités (défaut : simplexe)")
    pa.add_argument("--tirages", type=int, default=20, help="situations par cas")
    pa.add_argument("--sigma", type=float, default=0.0, help="bruit relatif (0 : sans bruit)")
    pa.add_argument("--centrale", action="store_true", help="source centrale 5000 ajoutée puis retirée")
    pa.add_argument("--iter", type=int, default=20000, help="budget d'itérations de la descente")
    pa.add_argument("--tol", type=float, default=0.1, help="réussite : chaque source à moins de tol")
    pa.add_argument("--sep-min", type=float, default=0.5, help="séparation minimale entre sources")
    pa.add_argument("--sans-ml", action="store_true", help="ne pas lancer greedy_ml")
    pa.add_argument("--figure", action="store_true", help="trajectoires de la descente sur un tirage")
    pa.add_argument("--tirage-figure", type=int, default=0, help="tirage tracé avec --figure")
    args = pa.parse_args()
    figure(args) if args.figure else experience(args)


if __name__ == "__main__":
    main()

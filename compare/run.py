"""
Comparaison reproductible des quatre méthodes (Prony, ML + BIC, Frank–Wolfe
glissant, descente de gradient d'Orano).

    python -m compare.run --all                      calcul complet + figures + Typst
    python -m compare.run --all --quick              10 tirages, 3 points par balayage
    python -m compare.run --figures                  retrace depuis les CSV, sans recalculer
    python -m compare.run --all --n-trials 100 --alpha 0.75 --beta 0.25 --timeout 60

Balayages (pour chacune des deux configurations « cercle » et « grille ») :
    bruit      σ ∈ logspace(-4, -2, 7), I_c = 50 S, N ∈ {2, 4, 6, 10}
    centrale   I_c / S ∈ {0, 10, 25, 50, 100, 200}, σ = 1e-3, N ∈ {2, 4, 6, 10}
    nsources   N ∈ {2, 4, 6, 10}, cadre par défaut (I_c = 50 S, σ = 1e-3)
    temps      temps de calcul, mêmes tirages que « nsources »

Les situations ne dépendent que de (N, tirage) : un point commun à plusieurs
balayages (le cadre par défaut) n'est calculé qu'une fois.

Sorties :
    results/brut_<config>.csv            une ligne par (point, tirage, méthode), écrite au fil
                                         de l'eau : un calcul interrompu reprend où il en était
    results/<balayage>_<config>.csv      lignes de chaque balayage, E calculée avec α, β
    figures/comparaison/*.pdf, *.png     figures
    figures/comparaison/synthese.typ     tableau de synthèse
    rapport/comparaison.typ              squelette de la partie « Comparaison »
(en mode --quick : results/quick/, figures/comparaison/quick/, rapport/comparaison_quick.typ)
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # 1 thread par processus : pas de sur-souscription

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from compare import protocol as P  # noqa: E402

NS = [2, 4, 6, 10]
SIGMA_DEF, IC_DEF = 1e-3, 50.0
BALAYAGES = ["bruit", "centrale", "nsources", "temps"]
CHAMPS = ["config", "N", "sigma", "ic_ratio", "tirage", "methode", "statut", "temps",
          "d_moy", "d_max", "r_moy", "n_est", "fausses", "manquees", "N_exact",
          "iterations", "perte", "chi2", "message"]


def grilles(quick):
    """Valeurs de chaque balayage."""
    if quick:
        return {"sigmas": [1e-4, 1e-3, 1e-2], "ics": [0.0, 50.0, 200.0]}
    return {"sigmas": [float(f"{v:.6g}") for v in np.logspace(-4, -2, 7)],
            "ics": [0.0, 10.0, 25.0, 50.0, 100.0, 200.0]}


def points(balayage, g):
    """Liste de (N, σ, I_c/S) d'un balayage."""
    if balayage == "bruit":
        return [(N, s, IC_DEF) for N in NS for s in g["sigmas"]]
    if balayage == "centrale":
        return [(N, SIGMA_DEF, ic) for N in NS for ic in g["ics"]]
    return [(N, SIGMA_DEF, IC_DEF) for N in NS]


def cle(config, N, sigma, ic, tirage, methode):
    return (config, int(N), f"{float(sigma):.6g}", f"{float(ic):.6g}", int(tirage), methode)


def dossiers(quick):
    r = ROOT / "results" / ("quick" if quick else "")
    f = ROOT / "figures" / "comparaison" / ("quick" if quick else "")
    r.mkdir(parents=True, exist_ok=True)
    f.mkdir(parents=True, exist_ok=True)
    (ROOT / "rapport").mkdir(exist_ok=True)
    return r, f


def lire_brut(path):
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {cle(r["config"], r["N"], r["sigma"], r["ic_ratio"], r["tirage"], r["methode"]): r
                for r in csv.DictReader(fh)}


def calculer(args, rdir):
    """Lance toutes les tâches manquantes, écrit results/brut_<config>.csv au fil de l'eau."""
    from compare.pool import map_timeout
    g = grilles(args.quick)
    n_trials = 10 if args.quick else args.n_trials
    pts = sorted({p for b in BALAYAGES for p in points(b, g)})
    for config in args.configs:
        brut_path = rdir / f"brut_{config}.csv"
        deja = lire_brut(brut_path)
        taches = [dict(config=config, N=N, sigma=s, ic_ratio=ic, tirage=t, methode=m,
                       alpha=args.alpha, beta=args.beta)
                  for (N, s, ic) in pts for t in range(n_trials) for m in args.methodes
                  if cle(config, N, s, ic, t, m) not in deja]
        # les tâches longues (grand N) d'abord : meilleur équilibrage
        taches.sort(key=lambda t: (-t["N"], t["methode"] != "descente"))
        print(f"\n[{config}] {len(pts)} points × {n_trials} tirages × {len(args.methodes)} méthodes : "
              f"{len(taches)} tâches à faire ({len(deja)} déjà faites)", flush=True)
        if not taches:
            continue
        neuf = not brut_path.exists()
        fh = open(brut_path, "a", encoding="utf-8", newline="")
        w = csv.DictWriter(fh, CHAMPS, extrasaction="ignore")
        if neuf:
            w.writeheader()
        n_to = {m: 0 for m in args.methodes}

        def sur_resultat(i, row):
            if "__exception__" in row:
                t = row
                row = {**{k: t[k] for k in ("config", "N", "sigma", "ic_ratio", "tirage", "methode")},
                       "statut": "exception", "temps": np.nan, "message": t["__exception__"][:200],
                       **P.erreur_timeout(t["N"])}
                row["N_exact"] = False
            w.writerow(row)
            fh.flush()

        def sur_timeout(i, t):
            n_to[t["methode"]] += 1
            row = {**{k: t[k] for k in ("config", "N", "sigma", "ic_ratio", "tirage", "methode")},
                   "statut": "timeout", "temps": args.timeout, **P.erreur_timeout(t["N"]),
                   "N_exact": False}
            w.writerow(row)
            fh.flush()

        t0 = time.perf_counter()
        map_timeout(P.tache, taches, args.n_proc, args.timeout, sur_resultat, sur_timeout,
                    progression=max(len(taches) // 40, 1))
        fh.close()
        print(f"[{config}] terminé en {(time.perf_counter() - t0) / 60:.1f} min ; "
              f"timeouts : {n_to}", flush=True)


def ecrire_balayages(args, rdir):
    """results/<balayage>_<config>.csv, avec E = α d_moy + β r_moy."""
    g = grilles(args.quick)
    for config in P.CONFIGS:
        brut = lire_brut(rdir / f"brut_{config}.csv")
        if not brut:
            continue
        n_trials = 1 + max(int(k[4]) for k in brut)
        for b in BALAYAGES:
            rows = []
            for (N, s, ic) in points(b, g):
                for t in range(n_trials):
                    for m in args.methodes:
                        r = brut.get(cle(config, N, s, ic, t, m))
                        if r is None:
                            continue
                        r = dict(r)
                        r["balayage"] = b
                        r["E"] = args.alpha * float(r["d_moy"]) + args.beta * float(r["r_moy"])
                        rows.append(r)
            if not rows:
                continue
            with open(rdir / f"{b}_{config}.csv", "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, ["balayage", "E"] + CHAMPS, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--all", action="store_true", help="calcul + figures + tableau + Typst")
    pa.add_argument("--figures", action="store_true", help="seulement retracer depuis les CSV")
    pa.add_argument("--quick", action="store_true", help="10 tirages, 3 points par balayage")
    pa.add_argument("--n-trials", type=int, default=100)
    pa.add_argument("--alpha", type=float, default=0.75)
    pa.add_argument("--beta", type=float, default=0.25)
    pa.add_argument("--timeout", type=float, default=60.0, help="secondes par tirage et méthode")
    pa.add_argument("--n-proc", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    pa.add_argument("--configs", nargs="+", default=P.CONFIGS, choices=P.CONFIGS)
    pa.add_argument("--methodes", nargs="+", default=["prony", "ml_bic", "fw"], choices=P.METHODES,
                    help="méthodes comparées (la descente, lente, n'est pas incluse par défaut : "
                         "elle est étudiée à part dans compare.deux_sources)")
    args = pa.parse_args()
    if not (args.all or args.figures):
        pa.error("préciser --all ou --figures")
    rdir, fdir = dossiers(args.quick)
    t0 = time.perf_counter()
    if args.all:
        calculer(args, rdir)
    ecrire_balayages(args, rdir)
    from compare import figures, typst_out
    figures.toutes(rdir, fdir, args.methodes)
    typst_out.ecrire(rdir, fdir, args)
    print(f"\nDurée totale : {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

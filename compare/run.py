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
import json  # noqa: E402
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


SOURCES = ROOT / "results" / "sources"         # sources estimées, une ligne JSON par tirage
DELTAS = [0.0, 0.05, 0.1, 0.2, 0.5]
N_BOOT = 2000


def lire_brut(path):
    """Ancien format (CSV sans les sources estimées), gardé pour lire les archives."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        return {cle(r["config"], r["N"], r["sigma"], r["ic_ratio"], r["tirage"], r["methode"]): r
                for r in csv.DictReader(fh)}


def chemin_sources(config, quick=False):
    d = SOURCES / "quick" if quick else SOURCES
    d.mkdir(parents=True, exist_ok=True)
    return d / f"comparaison_{config}.jsonl"


def lire_sources(path):
    """{clé: ligne} depuis le fichier JSONL des sources estimées."""
    if not path.exists():
        return {}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for L in fh:
            if L.strip():
                r = json.loads(L)
                out[cle(r["config"], r["N"], r["sigma"], r["ic_ratio"], r["tirage"], r["methode"])] = r
    return out


def calculer(args, rdir):
    """Lance toutes les tâches manquantes ; chaque tirage (sources estimées comprises) est
    ajouté au fil de l'eau à results/sources/comparaison_<config>.jsonl."""
    from compare.pool import map_timeout
    g = grilles(args.quick)
    n_trials = 10 if args.quick else args.n_trials
    pts = sorted({p for b in BALAYAGES for p in points(b, g)})
    for config in args.configs:
        path = chemin_sources(config, args.quick)
        deja = lire_sources(path)
        taches = [dict(config=config, N=N, sigma=s, ic_ratio=ic, tirage=t, methode=m,
                       alpha=args.alpha, beta=args.beta, delta=args.delta)
                  for (N, s, ic) in pts for t in range(n_trials) for m in args.methodes
                  if cle(config, N, s, ic, t, m) not in deja
                  or (args.relancer_timeouts and deja[cle(config, N, s, ic, t, m)].get("statut") == "timeout")]
        # un tirage relancé est ajouté à la fin du fichier : à la lecture, la dernière ligne l'emporte
        taches.sort(key=lambda t: (-t["N"], t["methode"] != "descente"))
        print(f"\n[{config}] {len(pts)} points × {n_trials} tirages × {len(args.methodes)} méthodes : "
              f"{len(taches)} tâches à faire ({len(deja)} déjà faites)", flush=True)
        if not taches:
            continue
        fh = open(path, "a", encoding="utf-8")
        n_to = {m: 0 for m in args.methodes}

        def ecrire(row):
            fh.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
            fh.flush()

        def sur_resultat(i, row):
            if "__exception__" in row:
                t = row
                row = {**{k: t[k] for k in ("config", "N", "sigma", "ic_ratio", "tirage", "methode")},
                       "statut": "exception", "temps": None, "message": t["__exception__"][:200],
                       "z_est": [], "I_est": []}
            ecrire(row)

        def sur_timeout(i, t):
            n_to[t["methode"]] += 1
            ecrire({**{k: t[k] for k in ("config", "N", "sigma", "ic_ratio", "tirage", "methode")},
                    "statut": "timeout", "temps": args.timeout, "z_est": [], "I_est": []})

        t0 = time.perf_counter()
        map_timeout(P.tache, taches, args.n_proc, args.timeout, sur_resultat, sur_timeout,
                    progression=max(len(taches) // 40, 1))
        fh.close()
        print(f"[{config}] terminé en {(time.perf_counter() - t0) / 60:.1f} min ; "
              f"timeouts : {n_to}", flush=True)


def verite(r):
    """Vraies sources d'un tirage (enregistrées, ou régénérées depuis les graines)."""
    if r.get("z_vrai"):
        return np.array(r["z_vrai"], float), np.array(r["I_vrai"], float)
    sit = P.situation(int(r["N"]), int(r["tirage"]), float(r["sigma"]), float(r["ic_ratio"]), r["config"])
    return sit.z, sit.I


def ligne_metrique(r, alpha, beta, delta, cache_verite):
    """Ligne par tirage : nouvelle métrique (fusion, delta) et ancienne, depuis les sources."""
    k = (r["config"], int(r["N"]), f"{float(r['sigma']):.6g}", f"{float(r['ic_ratio']):.6g}",
         int(r["tirage"]))
    if k not in cache_verite:
        cache_verite[k] = verite(r)
    z, I = cache_verite[k]
    ze = np.array(r.get("z_est") or [], float).reshape(-1, 2)
    Ie = np.array(r.get("I_est") or [], float).reshape(-1)
    e = P.erreur(z, I, ze, Ie, alpha, beta, delta)
    a = P.erreur_ancienne(z, I, ze, Ie, alpha, beta)
    out = {k: r.get(k) for k in ("config", "N", "sigma", "ic_ratio", "tirage", "methode", "statut",
                                 "temps", "iterations", "perte", "chi2", "message")}
    out.update(e)
    out.update(delta=delta, N_exact=bool(len(Ie) == int(r["N"])),
               E_ancienne=a["E"], d_moy_ancienne=a["d_moy"], r_moy_ancienne=a["r_moy"],
               fausses_ancienne=a["fausses"])
    return out


CHAMPS_LIGNE = ["balayage", "config", "N", "sigma", "ic_ratio", "tirage", "methode", "statut", "temps",
                "delta", "E", "d_moy", "d_max", "r_moy", "masse_parasite", "n_est", "n_fusion",
                "N_fusion", "fausses", "manquees", "N_exact", "E_ancienne", "d_moy_ancienne",
                "r_moy_ancienne", "fausses_ancienne", "iterations", "perte", "chi2", "message"]


def bootstrap_moyenne(v, n_boot=N_BOOT, graine=0):
    """Moyenne et intervalle de confiance à 95 % (bootstrap percentile, n_boot rééchantillonnages)."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(graine)
    m = v[rng.integers(0, v.size, (n_boot, v.size))].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agreger(rows, xkey):
    """Une ligne par (N, x, méthode) : moyenne [IC 95 %], médiane, quartiles, etc."""
    groupes = {}
    for r in rows:
        groupes.setdefault((int(r["N"]), float(r[xkey]) if xkey else 0.0, r["methode"]), []).append(r)
    out = []
    for (N, x, m), v in sorted(groupes.items()):
        E = np.array([r["E"] for r in v], float)
        mp = np.array([r["masse_parasite"] for r in v], float)
        t = np.array([r["temps"] for r in v if r["statut"] == "ok"], float)
        Ea = np.array([r["E_ancienne"] for r in v], float)
        Em, Eb, Eh = bootstrap_moyenne(E)
        Mm, Mb, Mh = bootstrap_moyenne(mp)
        out.append({"N": N, "x": x, "methode": m, "tirages": len(v),
                    "E_moyenne": Em, "E_ic95_bas": Eb, "E_ic95_haut": Eh,
                    "E_mediane": float(np.median(E)), "E_q1": float(np.percentile(E, 25)),
                    "E_q3": float(np.percentile(E, 75)),
                    "masse_parasite_moyenne": Mm, "masse_parasite_ic95_bas": Mb,
                    "masse_parasite_ic95_haut": Mh,
                    "d_moy_moyenne": float(np.mean([r["d_moy"] for r in v])),
                    "r_moy_moyenne": float(np.mean([r["r_moy"] for r in v])),
                    "fausses_moyenne": float(np.mean([r["fausses"] for r in v])),
                    "N_exact": float(np.mean([r["N_exact"] for r in v])),
                    "temps_median": float(np.median(t)) if t.size else np.nan,
                    "timeouts": sum(r["statut"] == "timeout" for r in v),
                    "E_ancienne_mediane": float(np.median(Ea)), "E_ancienne_moyenne": float(Ea.mean())})
    return out


def ecrire_csv(path, rows, champs):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, champs, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def ecrire_balayages(args, rdir):
    """
    Depuis results/sources/ : results/<balayage>_<config>.csv (une ligne par tirage, nouvelle
    et ancienne métrique), results/agrege_<balayage>_<config>.csv (moyenne, IC 95 %, médiane)
    et results/delta_<config>.csv (agrégats en fonction de delta, cadre par défaut).
    """
    g = grilles(args.quick)
    for config in P.CONFIGS:
        src = lire_sources(chemin_sources(config, args.quick))
        if not src:
            continue
        n_trials = 1 + max(int(k[4]) for k in src)
        cache = {}
        for b in BALAYAGES:
            rows = []
            for (N, s, ic) in points(b, g):
                for t in range(n_trials):
                    for m in args.methodes:
                        r = src.get(cle(config, N, s, ic, t, m))
                        if r is not None:
                            rows.append({"balayage": b, **ligne_metrique(r, args.alpha, args.beta,
                                                                         args.delta, cache)})
            if not rows:
                continue
            ecrire_csv(rdir / f"{b}_{config}.csv", rows, CHAMPS_LIGNE)
            xkey = {"bruit": "sigma", "centrale": "ic_ratio"}.get(b)
            ag = agreger(rows, xkey)
            ecrire_csv(rdir / f"agrege_{b}_{config}.csv", ag, list(ag[0]))
        out = []
        for delta in DELTAS:
            rows = [ligne_metrique(src[cle(config, N, SIGMA_DEF, IC_DEF, t, m)], args.alpha, args.beta,
                                   delta, cache)
                    for N in NS for t in range(n_trials) for m in args.methodes
                    if cle(config, N, SIGMA_DEF, IC_DEF, t, m) in src]
            for a in agreger(rows, None):
                out.append({"delta": delta, **{k: v for k, v in a.items() if k != "x"}})
        if out:
            ecrire_csv(rdir / f"delta_{config}.csv", out, list(out[0]))


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
    pa.add_argument("--delta", type=float, default=P.DELTA_DEF,
                    help="rayon de fusion des sources (estimées et vraies) dans la métrique")
    pa.add_argument("--timeout", type=float, default=60.0, help="secondes par tirage et méthode")
    pa.add_argument("--relancer-timeouts", action="store_true",
                    help="relance aussi les tirages déjà enregistrés en timeout (avec --timeout)")
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
    if not args.quick:
        from compare import figures_synthese, annexe
        figures_synthese.toutes()
        annexe.main()
    print(f"\nDurée totale : {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

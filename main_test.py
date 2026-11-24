"""
TEST : compare les méthodes sur beaucoup de tirages. On choisit les méthodes et les valeurs
de chaque paramètre ; le programme lance TOUTES les combinaisons (produit cartésien), puis
écrit les résultats et une figure.

Exemples (depuis la racine du dépôt) :

    python main_test.py --N 2 4 --tirages 10
        petit test rapide : 2 et 4 sources, 10 tirages, cadre du projet, 3 méthodes

    python main_test.py --methodes ml_bic blasso --N 4 --sigma 1e-4 1e-3 1e-2 --capteurs grille
        effet du bruit sur ML + BIC et le BLASSO, avec la grille

    python main_test.py --methodes descente gauss_newton ml_bic --N 2 --x 1 10 100 --sigma 0 --ic 0 --capteurs cercle
        cas test d'Orano : deux sources de rapport d'intensité x, sans bruit ni source centrale

    python main_test.py --preset nombre_sources
        refait une étude du rapport ; les sources déjà calculées sont lues dans resultats/sources/ :
        rien n'est recalculé, les CSV et la figure sortent en quelques secondes dans
        resultats/etudes/nombre_sources/. Études disponibles :
            nombre_sources      N = 2, 4, 6, 10 ; cadre du projet (σ = 10⁻³, I_c = 50 S)
            bruit               σ de 10⁻⁴ à 10⁻² (7 valeurs), N = 2, 4, 6, 10
            source_centrale     I_c / S = 0, 10, 25, 50, 100, 200, N = 2, 4, 6, 10
            deux_sources        N = 2, x = 1 à 300, cadre du projet, avec descente et Gauss–Newton
            deux_sources_orano  idem dans le cas test d'Orano (σ = 0, sans source centrale, cercle)

Paramètres (chacun accepte une ou plusieurs valeurs) :
    --methodes   prony ml_bic blasso descente gauss_newton        (défaut prony ml_bic blasso)
    --N          nombres de sources                               (défaut 2 4 6 10)
    --sigma      bruits relatifs σ (0 : sans bruit)               (défaut 1e-3)
    --ic         sources centrales, en multiples de S = 100       (défaut 50)
    --capteurs   cercle, grille                                   (défaut cercle grille)
    --x          rapports d'intensité I₁/I₂ (seulement avec N = 2 : cas à deux sources)
    --tirages    nombre de tirages par combinaison                (défaut 100)
    --axe        paramètre en abscisse de la figure : N, sigma, ic ou x
                 (défaut : le premier qui prend plusieurs valeurs)

Calcul :
    --nom        nom du test : les résultats vont dans resultats/tests/<nom>/   (défaut « test »)
    --sources    fichier JSONL des sources (défaut resultats/tests/<nom>/sources.jsonl). Les
                 tâches déjà présentes dans ce fichier ne sont PAS recalculées : un calcul
                 interrompu reprend où il s'était arrêté, et on peut changer la métrique sans
                 rien relancer.
    --processus  nombre de processus de calcul en parallèle      (défaut : nombre de cœurs − 1)
    --timeout    secondes au plus par tâche ; au-delà, la tâche compte comme un échec complet
                 (toutes les sources manquées)                    (défaut 300)
    --relancer-timeouts   relance les tâches enregistrées en timeout
    --seuil      rayon de tolérance de la détection               (défaut 0,1)
    --delta      rayon de fusion des sources proches              (défaut 0,1)

Sorties dans resultats/tests/<nom>/ :
    sources.jsonl   vraies sources et sources estimées de chaque tâche
    tirages.csv     une ligne par tâche : E, détection, précision, masse parasite, temps, ...
    resume.csv      une ligne par combinaison (méthode, capteurs, N, σ, I_c, x) : médiane de E et
                    son IC 95 %, moyenne de E, taux de détection, précision, masse parasite,
                    proportion de N bien estimé, temps médian, nombre de timeouts
    figure.png      erreur E, détection, précision et temps en fonction du paramètre --axe

Indicateurs (voir localisation/metrique.py) : les sources proches de moins de δ sont fusionnées,
puis appariées aux vraies par l'algorithme hongrois.
    détection      part des vraies sources retrouvées à moins du seuil (0,1)
    précision      distance entre source estimée et vraie source, sur les sources détectées
    masse parasite intensité des sources estimées en trop, en fraction de S
    E              (1/N) Σ (0,75 d_i + 0,25 r_i), une source manquée ajoutant 4/N
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # un fil par processus : le parallélisme est par tâche

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from localisation import comparaison as C  # noqa: E402
from localisation import metrique  # noqa: E402
from localisation.figures import figure_comparaison  # noqa: E402
from localisation.methodes import METHODES, NOMS, nom_canonique  # noqa: E402
from localisation.pool import map_timeout  # noqa: E402

RACINE = Path(__file__).resolve().parent
SOURCES = RACINE / "resultats" / "sources"
SIGMAS = [float(f"{v:.6g}") for v in np.logspace(-4, -2, 7)]

# Études du rapport. Leurs sources sont déjà dans resultats/sources/ : rien n'est recalculé.
PRESETS = {
    "nombre_sources": dict(methodes=["prony", "ml_bic", "blasso"], N=[2, 4, 6, 10], sigma=[1e-3],
                           ic=[50.0], capteurs=["cercle", "grille"], x=[None], axe="N",
                           sources=SOURCES / "comparaison.jsonl"),
    "bruit": dict(methodes=["prony", "ml_bic", "blasso"], N=[2, 4, 6, 10], sigma=SIGMAS, ic=[50.0],
                  capteurs=["cercle", "grille"], x=[None], axe="sigma",
                  sources=SOURCES / "comparaison.jsonl"),
    "source_centrale": dict(methodes=["prony", "ml_bic", "blasso"], N=[2, 4, 6, 10], sigma=[1e-3],
                            ic=[0.0, 10.0, 25.0, 50.0, 100.0, 200.0], capteurs=["cercle", "grille"],
                            x=[None], axe="ic", sources=SOURCES / "comparaison.jsonl"),
    "deux_sources": dict(methodes=["descente", "gauss_newton", "ml_bic", "blasso", "prony"], N=[2],
                         sigma=[1e-3], ic=[50.0], capteurs=["cercle", "grille"],
                         x=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], axe="x",
                         sources=SOURCES / "deux_sources.jsonl"),
    "deux_sources_orano": dict(methodes=["descente", "gauss_newton", "ml_bic", "blasso", "prony"],
                               N=[2], sigma=[0.0], ic=[0.0], capteurs=["cercle"],
                               x=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0], axe="x",
                               sources=SOURCES / "deux_sources.jsonl"),
}


def arguments():
    pa = argparse.ArgumentParser(description="Comparaison des méthodes sur beaucoup de tirages.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Études toutes prêtes (--preset) : " + ", ".join(PRESETS)
                                        + ".\nVoir l'en-tête de main_test.py pour le détail.")
    pa.add_argument("--preset", choices=PRESETS, default=None,
                    help="refait une étude du rapport (les autres paramètres sont alors ignorés)")
    pa.add_argument("--methodes", nargs="+", default=["prony", "ml_bic", "blasso"])
    pa.add_argument("--N", nargs="+", type=int, default=[2, 4, 6, 10])
    pa.add_argument("--sigma", nargs="+", type=float, default=[1e-3])
    pa.add_argument("--ic", nargs="+", type=float, default=[50.0])
    pa.add_argument("--capteurs", nargs="+", choices=["cercle", "grille"], default=["cercle", "grille"])
    pa.add_argument("--x", nargs="+", type=float, default=None)
    pa.add_argument("--tirages", type=int, default=100)
    pa.add_argument("--axe", choices=["N", "sigma", "ic", "x"], default=None)
    pa.add_argument("--nom", default=None)
    pa.add_argument("--sources", default=None)
    pa.add_argument("--processus", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    pa.add_argument("--timeout", type=float, default=300.0)
    pa.add_argument("--relancer-timeouts", action="store_true")
    pa.add_argument("--seuil", type=float, default=metrique.SEUIL)
    pa.add_argument("--delta", type=float, default=metrique.DELTA)
    a = pa.parse_args()
    if a.preset:
        p = PRESETS[a.preset]
        a.methodes, a.N, a.sigma, a.ic, a.capteurs = p["methodes"], p["N"], p["sigma"], p["ic"], p["capteurs"]
        a.x, a.axe = p["x"], a.axe or p["axe"]
        a.sources = a.sources or str(p["sources"])
        a.nom = a.nom or a.preset
    else:
        a.x = [None] if a.x is None else a.x
    if any(x is not None for x in a.x) and a.N != [2]:
        pa.error("--x (rapport d'intensité) demande --N 2")
    try:
        a.methodes = [nom_canonique(m) for m in a.methodes]
    except ValueError as e:
        pa.error(f"{e} ; disponibles : {' '.join(METHODES)}")
    a.nom = a.nom or "test"
    a.dossier = RACINE / "resultats" / ("etudes" if a.preset else "tests") / a.nom
    a.sources = Path(a.sources) if a.sources else a.dossier / "sources.jsonl"
    if a.axe is None:
        valeurs = {"N": a.N, "sigma": a.sigma, "ic": a.ic, "x": a.x}
        a.axe = next((k for k, v in valeurs.items() if len(v) > 1), "N")
    return a


def relatif(p: Path) -> str:
    """Chemin relatif à la racine du dépôt (s'il est dedans)."""
    p = Path(p).resolve()
    return p.relative_to(RACINE).as_posix() if p.is_relative_to(RACINE) else str(p)


def calculer(a, taches):
    """Lance les tâches absentes du fichier de sources (ou en timeout si demandé)."""
    deja = C.lire_sources(a.sources)
    a_faire = [t for t in taches if C.cle(t) not in deja
               or (a.relancer_timeouts and deja[C.cle(t)]["statut"] == "timeout")]
    print(f"{len(taches)} tâches, dont {len(taches) - len(a_faire)} déjà dans {relatif(a.sources)}")
    if not a_faire:
        return
    print(f"{len(a_faire)} tâches à calculer avec {a.processus} processus "
          f"(timeout {a.timeout:g} s par tâche)", flush=True)
    a.sources.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with open(a.sources, "a", encoding="utf-8") as fh:
        map_timeout(C.tache, a_faire, a.processus, a.timeout,
                    lambda i, r: C.ecrire_ligne(fh, r if "__exception__" not in r
                                                else {**C.ligne_timeout(a_faire[i]), "statut": "exception"}),
                    lambda i, t: C.ecrire_ligne(fh, C.ligne_timeout(t)),
                    progression=max(1, len(a_faire) // 20))
    print(f"calcul terminé en {(time.perf_counter() - t0) / 60:.1f} min")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    a = arguments()
    taches = C.taches(a.methodes, a.N, a.sigma, a.ic, a.capteurs, a.tirages, a.x)
    print(f"\nTest « {a.nom} » : méthodes {', '.join(NOMS[m] for m in a.methodes)} ; N = {a.N} ; "
          f"σ = {a.sigma} ; I_c/S = {a.ic} ; capteurs {a.capteurs}"
          + (f" ; x = {a.x}" if a.x != [None] else "") + f" ; {a.tirages} tirages")
    calculer(a, taches)

    # indicateurs depuis les sources (seulement les tâches demandées)
    src = C.lire_sources(a.sources)
    lignes = [C.indicateurs(src[C.cle(t)], a.delta, a.seuil) for t in taches if C.cle(t) in src]
    resume = C.agreger(lignes)
    a.dossier.mkdir(parents=True, exist_ok=True)
    C.ecrire_csv(a.dossier / "tirages.csv", lignes)
    C.ecrire_csv(a.dossier / "resume.csv", resume)
    with open(a.dossier / "parametres.json", "w", encoding="utf-8") as fh:
        json.dump({k: (relatif(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, fh,
                  ensure_ascii=False, indent=1)
    figure_comparaison(resume, a.axe, a.dossier / "figure.png",
                       f"{a.nom} — {a.tirages} tirages par point ; détection : d < {a.seuil:g}".replace(".", ","))

    # résumé dans le terminal
    print(f"\n{'méthode':<26}{'capteurs':<9}{'N':>3} {'σ':>9} {'Ic/S':>6} {'x':>5}  "
          f"{'E médiane':>10} {'détection':>9} {'précision':>10} {'temps (s)':>9}")
    for r in resume:
        print(f"{NOMS[r['methode']]:<26}{r['capteurs']:<9}{r['N']:>3} {r['sigma']:>9.3g} {r['ic']:>6g} "
              f"{str(r['x']):>5}  {r['E_mediane']:>10.3g} {100 * r['detection']:>8.0f} % "
              f"{r['precision_d']:>10.2g} {r['temps_median']:>9.3g}"
              + (f"   ({r['timeouts']} timeouts)" if r["timeouts"] else ""))
    print(f"\nRésultats : {relatif(a.dossier)}/  (resume.csv, tirages.csv, figure.png)")


if __name__ == "__main__":
    main()

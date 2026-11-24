"""
SIMULATION : on tire (ou on choisit) des sources, on simule les mesures des capteurs, on
lance une ou plusieurs méthodes, et on obtient une image avec les vraies sources et les
sources retrouvées par chaque méthode, plus un résumé dans le terminal.

Exemples (depuis la racine du dépôt) :

    python main.py
        cadre du projet : 4 sources tirées au hasard, σ = 10⁻³, I_c = 50 S,
        capteurs cercle + grille, méthodes prony ml_bic blasso descente

    python main.py --N 6 --capteurs cercle --ic 0 --methodes ml_bic blasso --tirage 21
        6 sources, cercle seul, sans source centrale, deux méthodes, tirage n° 21

    python main.py --sources "2,5,60  8,15,30  5,3,10" --methodes ml_bic
        sources choisies à la main : "x,y,intensité" séparées par des espaces

    python main.py --N 2 --x 30 --sigma 0 --ic 0 --capteurs cercle --methodes descente gauss_newton
        cas test d'Orano : deux sources d'intensités 30:1, sans bruit ni source centrale

    python main.py --afficher
        ouvre aussi l'image dans une fenêtre

Options :
    --N            nombre de sources tirées au hasard                     (défaut 4)
    --tirage       numéro du tirage (graine) : même numéro = mêmes sources et même bruit (défaut 0)
    --sources      sources imposées "x,y,I  x,y,I ..." (remplace --N et le tirage des sources)
    --x            rapport d'intensité I₁/I₂ pour N = 2 (cas à deux sources d'Orano)
    --sigma        bruit relatif σ (0 : sans bruit)                        (défaut 1e-3)
    --ic           intensité de la source centrale, en multiples de S = 100 (défaut 50 ; 0 : aucune)
    --capteurs     cercle (64 capteurs) ou grille (cercle + grille 16 × 32) (défaut grille)
    --methodes     parmi prony ml_bic blasso descente gauss_newton         (défaut les 4 premières)
    --seuil        rayon de tolérance de la détection                     (défaut 0,1)
    --delta        rayon de fusion des sources proches                    (défaut 0,1)
    --sortie       chemin de l'image (défaut resultats/simulations/simulation_<paramètres>.png)
    --afficher     ouvre l'image à la fin

Le hangar est D = [0, 10] × [0, 20] ; les sources tirées au hasard ont des positions
uniformes dans D et des intensités aléatoires de somme S = 100. La descente et Gauss–Newton
reçoivent le vrai nombre de sources ; Prony, ML + BIC et le BLASSO l'estiment eux-mêmes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from localisation import metrique
from localisation.figures import figure_simulation, fr
from localisation.methodes import METHODES, NOMS, lancer, nom_canonique
from localisation.probleme import CAPTEURS, situation

RACINE = Path(__file__).resolve().parent


def lire_sources(texte: str):
    """ "x,y,I  x,y,I" → (z (n, 2), I (n,)). """
    try:
        v = np.array([[float(a) for a in s.split(",")] for s in texte.split()], float)
        assert v.ndim == 2 and v.shape[1] == 3
    except (ValueError, AssertionError):
        raise SystemExit('--sources : format attendu "x,y,intensité  x,y,intensité ..."')
    return v[:, :2], v[:, 2]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pa = argparse.ArgumentParser(description="Simulation : une situation, une ou plusieurs méthodes, "
                                             "une image.", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("Options :")[0])
    pa.add_argument("--N", type=int, default=4)
    pa.add_argument("--tirage", type=int, default=0)
    pa.add_argument("--sources", type=str, default=None)
    pa.add_argument("--x", type=float, default=None)
    pa.add_argument("--sigma", type=float, default=1e-3)
    pa.add_argument("--ic", type=float, default=50.0)
    pa.add_argument("--capteurs", choices=CAPTEURS, default="grille")
    pa.add_argument("--methodes", nargs="+", default=["prony", "ml_bic", "blasso", "descente"])
    pa.add_argument("--seuil", type=float, default=metrique.SEUIL)
    pa.add_argument("--delta", type=float, default=metrique.DELTA)
    pa.add_argument("--sortie", type=str, default=None)
    pa.add_argument("--afficher", action="store_true")
    a = pa.parse_args()
    try:
        methodes = [nom_canonique(m) for m in a.methodes]
    except ValueError as e:
        raise SystemExit(f"{e}\nméthodes disponibles : {' '.join(METHODES)}")

    z = I = None
    if a.sources:
        z, I = lire_sources(a.sources)
    sit = situation(a.N, a.tirage, a.sigma, a.ic, a.capteurs, x=a.x, z=z, I=I)

    print(f"\nSituation : {sit.N} sources, capteurs {a.capteurs} ({len(sit.X)}), σ = {a.sigma:g}, "
          f"I_c = {a.ic:g} S, tirage {a.tirage}")
    for k, (zz, ii) in enumerate(zip(sit.z, sit.I)):
        print(f"   vraie source {k + 1} : ({zz[0]:6.2f}, {zz[1]:6.2f}), intensité {ii:6.2f}")

    estimations, resumes = {}, {}
    for m in methodes:
        print(f"\n{NOMS[m]} ...", flush=True)
        out = lancer(m, sit)
        e = metrique.erreur(sit.z, sit.I, out["z"], out["I"], a.delta, a.seuil)
        n_det = round(e["detection"] * e["N_fusion"])
        print(f"   {len(out['I'])} sources estimées en {out['temps']:.2f} s")
        for zz, ii in zip(out["z"], out["I"]):
            d = np.min(np.linalg.norm(sit.z - zz, axis=1))
            print(f"   ({zz[0]:6.2f}, {zz[1]:6.2f}), intensité {ii:7.2f}   (à {d:.3f} de la vraie source la plus proche)")
        print(f"   erreur E = {e['E']:.4f} ; détectées (d < {a.seuil:g}) : {n_det}/{e['N_fusion']} ; "
              f"précision (distance moyenne des détectées) : {fr(e['d_det'])} ; "
              f"masse parasite : {100 * e['masse_parasite']:.2f} % de S")
        estimations[m] = out
        resumes[m] = (f"E = {fr(e['E'], 3)} ; {n_det}/{e['N_fusion']} détectées (d < {fr(a.seuil)}) ; "
                      f"{len(out['I'])} estimées\nprécision {fr(e['d_det'])} ; temps {fr(out['temps'])} s")

    if a.sortie:
        chemin = Path(a.sortie)
    else:
        src = "manuelles" if a.sources else f"N{sit.N}_tirage{a.tirage}"
        x = f"_x{a.x:g}" if a.x is not None else ""
        chemin = (RACINE / "resultats" / "simulations"
                  / f"simulation_{src}{x}_{a.capteurs}_sigma{a.sigma:g}_ic{a.ic:g}.png")
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure_simulation(sit, estimations, resumes, chemin)
    chemin = chemin.resolve()
    print(f"\nImage : {chemin.relative_to(RACINE).as_posix() if chemin.is_relative_to(RACINE) else chemin}")
    if a.afficher:
        import os
        import subprocess
        if sys.platform.startswith("win"):
            os.startfile(chemin)                                   # noqa: S606
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(chemin)])


if __name__ == "__main__":
    main()

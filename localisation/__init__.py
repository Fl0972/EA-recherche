"""
Localisation de sources ponctuelles dans le hangar, à partir de capteurs en 1/r².

Organisation :
    probleme.py      hangar, capteurs, source centrale, bruit, tirage d'une situation
    noyau.py         noyau 1/r² et dérivées, Setup (blanchiment, normalisation), grille
    optimisation.py  briques communes à ML + BIC et au BLASSO
    prony.py         méthode de Prony (cercle seulement)
    ml_bic.py        maximum de vraisemblance + BIC
    blasso.py        BLASSO par Frank–Wolfe glissant
    descente.py      descente de gradient d'Orano (+ Gauss–Newton témoin)
    methodes.py      registre : nom → méthode, ``lancer(nom, situation)``
    metrique.py      fusion, appariement, détection, précision, erreur E
    comparaison.py   lancer beaucoup de tirages, stocker les sources, agréger
    figures.py       figures (reconstruction, courbes de comparaison)
    pool.py          calcul parallèle avec timeout par tâche
"""

from .methodes import METHODES, NOMS, lancer
from .metrique import erreur
from .probleme import Situation, situation

__all__ = ["METHODES", "NOMS", "lancer", "erreur", "Situation", "situation"]

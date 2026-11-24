"""
Registre des méthodes : un nom → une fonction ``estimer(situation) -> {"z", "I", ...}``.

    prony         Prony (capteurs du cercle seulement), N estimé
    ml_bic        maximum de vraisemblance glouton + BIC, N estimé
    blasso        BLASSO par Frank–Wolfe glissant, N estimé
    descente      descente de gradient projetée d'Orano, N CONNU
    gauss_newton  Gauss–Newton sur la perte d'Orano, N CONNU (témoin de la descente)
"""

from __future__ import annotations

import time
import warnings

import numpy as np

from . import blasso, descente, ml_bic, prony
from .probleme import Situation

METHODES = {
    "prony": prony.estimer,
    "ml_bic": ml_bic.estimer,
    "blasso": blasso.estimer,
    "descente": descente.estimer,
    "gauss_newton": descente.estimer_gauss_newton,
}
NOMS = {"prony": "Prony", "ml_bic": "ML + BIC", "blasso": "BLASSO (Frank–Wolfe)",
        "descente": "Descente (Orano, N connu)", "gauss_newton": "Gauss–Newton (N connu)"}
ALIAS = {"fw": "blasso", "gn": "gauss_newton", "ml": "ml_bic", "orano": "descente"}


def nom_canonique(nom: str) -> str:
    nom = ALIAS.get(nom, nom)
    if nom not in METHODES:
        raise ValueError(f"méthode inconnue : {nom!r} ({', '.join(METHODES)})")
    return nom


def lancer(nom: str, sit: Situation) -> dict:
    """
    Applique la méthode ``nom`` à la situation. Renvoie {"z" (n, 2), "I" (n,), "temps", ...}.
    Le temps ne compte que la méthode (pas la génération des données).
    """
    f = METHODES[nom_canonique(nom)]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = f(sit)
    out["temps"] = time.perf_counter() - t0
    out["z"] = np.asarray(out["z"], float).reshape(-1, 2)
    out["I"] = np.asarray(out["I"], float).reshape(-1)
    return out

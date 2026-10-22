"""
Lois du nombre de sources N.

Une loi de comptage est n'importe quelle fonction  rng -> int.
Pour ajouter une loi, écrire une fabrique qui renvoie une telle fonction
(puis l'enregistrer dans `config/registry.py`).
"""

from typing import Callable

import numpy as np

CountLaw = Callable[[np.random.Generator], int]


def poisson(lam: float) -> CountLaw:
    """N ~ Poisson(lam) : E[N] = Var[N] = lam."""
    return lambda rng: int(rng.poisson(lam))


def binomial(n: int, p: float) -> CountLaw:
    """N ~ Binomiale(n, p) : au plus n sources, chacune présente avec proba p."""
    return lambda rng: int(rng.binomial(n, p))


def negative_binomial(r: float, p: float) -> CountLaw:
    """N ~ Binomiale négative(r, p). Utile pour une sur-dispersion par rapport à Poisson."""
    return lambda rng: int(rng.negative_binomial(r, p))


def fixed(n: int) -> CountLaw:
    """N = n de façon déterministe (pratique pour les tests)."""
    return lambda rng: n


def at_least_one(law: CountLaw) -> CountLaw:
    """Conditionne la loi à N >= 1 (par rejet : on retire tant que N = 0)."""
    def sample(rng):
        while (n := law(rng)) == 0:
            pass
        return n
    return sample

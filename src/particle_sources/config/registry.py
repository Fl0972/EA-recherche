"""
Correspondance entre les noms utilisés dans les fichiers YAML et le code.

Pour rendre une nouvelle loi / méthode utilisable depuis la config, il suffit
de l'ajouter au dictionnaire correspondant.

Chaque entrée associe un nom à une fabrique : appelée avec les `params` du
YAML, elle renvoie l'objet prêt à servir (une loi, un réseau de capteurs, un
modèle de bruit, une méthode d'inversion). Les objets sans paramètre sont
enveloppés dans un `lambda:` pour garder la même interface.
"""

from functools import partial

from ..generation import count_laws, intensity_laws, position_laws
from ..inversion import prony_circle
from ..sensors import CircleSensors, noise

COUNT_LAWS = {
    "poisson": count_laws.poisson,
    "binomial": count_laws.binomial,
    "negative_binomial": count_laws.negative_binomial,
    "fixed": count_laws.fixed,
}

INTENSITY_LAWS = {
    "uniform_simplex": lambda: intensity_laws.uniform_simplex,
    "dirichlet": intensity_laws.dirichlet,
}

POSITION_LAWS = {
    "uniform_in_box": lambda: position_laws.uniform_in_box,
}

# Les géométries reçoivent le domaine en premier argument
SENSOR_LAYOUTS = {
    "circumscribed_circle": lambda box, n, t0=0.0: CircleSensors.circumscribed(box, n, t0),
    "circle": lambda box, center, radius, n, t0=0.0: CircleSensors(center, radius, n, t0),
}

NOISE_MODELS = {
    "none": lambda: noise.no_noise,
    "gaussian_relative": noise.gaussian_relative,
    "poisson_counts": noise.poisson_counts,
}

# Une méthode construite est une fonction  (y, sensors) -> Sources
INVERSION_METHODS = {
    "prony": lambda **params: partial(prony_circle, **params),
}


def build(registry: dict, spec: dict, kind: str, *args):
    """
    Construit l'objet décrit par spec = {name: ..., params: {...}}.

    Paramètres
    ----------
    registry : dict
        Un des dictionnaires ci-dessus.
    spec : dict
        Section du YAML, par ex. {"name": "poisson", "params": {"lam": 1.0}}.
    kind : str
        Nom de la section, utilisé uniquement dans les messages d'erreur.
    *args
        Arguments passés avant les params (le domaine pour SENSOR_LAYOUTS).

    Lève ValueError, avec la liste des choix possibles, si le nom est inconnu
    ou si les paramètres ne correspondent pas à la fabrique.
    """
    name = spec.get("name")
    params = spec.get("params") or {}
    if name not in registry:
        raise ValueError(f"{kind} inconnu : {name!r}. Choix possibles : "
                         f"{', '.join(registry)}")
    try:
        return registry[name](*args, **params)
    except TypeError as e:
        raise ValueError(f"paramètres invalides pour {kind} {name!r} : {e}") from e

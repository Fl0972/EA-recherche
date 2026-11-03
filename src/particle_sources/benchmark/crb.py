"""
Borne de Cramér-Rao (CRB) : meilleure précision possible pour tout estimateur
non biaisé des positions et intensités, le nombre de sources étant connu.

    Cov(theta_hat) >= F^-1,   F = information de Fisher du modèle de bruit.

La contrainte sum I = I_tot (connue a priori) est prise en compte par la CRB
contrainte (Gorman & Hero, 1990) :
    C = F^-1 - F^-1 G^T (G F^-1 G^T)^-1 G F^-1,   G = d(sum I)/d theta.

Attention : c'est une borne locale, à N connu, pour estimateurs non biaisés.
Un estimateur qui exploite l'a priori (bayésien, réseau entraîné) peut faire
mieux là où l'information manque (sources faibles, très proches).
"""

import numpy as np

from ..inversion import InverseProblem
from ..model import Sources, intensity_at, intensity_jacobian


def crb(true: Sources, problem: InverseProblem,
        all_sources: Sources | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Écarts-types minimaux par source : (position, intensité), chacun (N,).
    Position : sqrt(var_x + var_y). `all_sources` = tout ce que voient les
    capteurs (par défaut true + sources connues), pour le niveau de bruit.
    Renvoie des zéros si le modèle n'a pas de bruit.
    """
    n, d = true.n, true.dim
    if n == 0:
        return np.empty(0), np.empty(0)
    if problem.noise.std(np.ones(1)) is None:
        return np.zeros(n), np.zeros(n)

    x_s = problem.sensors.positions
    seen = all_sources if all_sources is not None else true + problem.known
    mu = intensity_at(seen, x_s, problem.eps)
    J = intensity_jacobian(true, x_s, problem.eps)
    F_inv = np.linalg.pinv(problem.noise.fisher(mu, J))

    G = np.concatenate([np.zeros(n * d), np.ones(n)])[None, :]
    FG = F_inv @ G.T
    C = F_inv - FG @ FG.T / (G @ FG)

    var = np.clip(np.diag(C), 0, None)
    pos_std = np.sqrt(var[: n * d].reshape(n, d).sum(axis=1))
    int_std = np.sqrt(var[n * d :])
    return pos_std, int_std

"""
Affinage par maximum de vraisemblance (moindres carrés pondérés par le bruit)
et choix du nombre de sources par critère BIC.

On minimise   chi2(theta) = sum_k ((y_k - mu_k(theta)) / std_k)^2
avec mu = Phi(sources inconnues) + Phi(sources connues), sous les contraintes :
positions dans le domaine, intensités >= 0, somme des intensités = I_tot.
"""

import numpy as np
from scipy.optimize import least_squares

from ..model import Sources, intensity_at, intensity_jacobian
from .problem import InverseProblem
from .prony import prony_circle

# Tolérance relative sur la contrainte sum I = I_tot (pénalité dans chi2)
SUM_TOL = 1e-3
# Écart-type relatif utilisé quand le modèle n'a pas de bruit
NOISELESS_REL_STD = 1e-9
# Nombre max d'évaluations par ajustement (quelques cas dégénérés ne convergent pas)
MAX_NFEV = 100


def _weights(y: np.ndarray, problem: InverseProblem) -> np.ndarray:
    std = problem.noise.std(y)
    if std is None:                       # pas de bruit : quasi nul (précision machine)
        std = NOISELESS_REL_STD * np.abs(y)
    return 1 / np.maximum(std, 1e-12 * np.abs(y).max())


def chi2(y: np.ndarray, problem: InverseProblem, sources: Sources) -> float:
    mu = intensity_at(sources, problem.sensors.positions, problem.eps)
    r = (y - problem.known_contribution() - mu) * _weights(y, problem)
    return float(r @ r)


def refine(y: np.ndarray, problem: InverseProblem, init: Sources) -> Sources:
    """Maximum de vraisemblance à nombre de sources fixé, initialisé par `init`."""
    n, d = init.n, problem.box.dim
    if n == 0:
        return init
    w = _weights(y, problem)
    target = y - problem.known_contribution()
    x_s = problem.sensors.positions
    lo = np.concatenate([np.tile(problem.box.lo, n), np.zeros(n)])
    hi = np.concatenate([np.tile(problem.box.hi, n), np.full(n, np.inf)])
    margin = 1e-6 * problem.box.size.min()
    theta0 = np.concatenate([init.positions.ravel(), np.maximum(init.intensities, 1e-9)])
    theta0 = np.clip(theta0, lo + margin, hi - margin)
    sum_w = 1 / (SUM_TOL * problem.total_intensity)

    def unpack(theta):
        return Sources(theta[: n * d].reshape(n, d), theta[n * d :])

    def residuals(theta):
        s = unpack(theta)
        r = (target - intensity_at(s, x_s, problem.eps)) * w
        return np.append(r, sum_w * (s.total_intensity - problem.total_intensity))

    def jacobian(theta):
        J = -intensity_jacobian(unpack(theta), x_s, problem.eps) * w[:, None]
        row = np.concatenate([np.zeros(n * d), np.full(n, sum_w)])
        return np.vstack([J, row])

    res = least_squares(residuals, theta0, jac=jacobian, bounds=(lo, hi), x_scale="jac",
                        max_nfev=MAX_NFEV)
    return unpack(res.x)


def prony_mle(y: np.ndarray, problem: InverseProblem, max_sources: int = 6) -> Sources:
    """
    Pour chaque N = 0..max_sources : initialisation par Prony à N sources, puis
    maximum de vraisemblance. On garde le N qui minimise
        BIC = chi2 + 3 N log K.
    """
    y_unknown = y - problem.known_contribution()
    K = len(y)
    best, best_score = Sources.empty(problem.box.dim), chi2(y, problem, Sources.empty())
    tried = set()
    for n in range(1, max_sources + 1):
        init = prony_circle(y_unknown, problem.sensors, n_sources=n)
        # Prony écarte les solutions non physiques : init peut avoir < n sources
        if init.n == 0 or init.n in tried:
            continue
        tried.add(init.n)
        est = refine(y, problem, init)
        score = chi2(y, problem, est) + 3 * est.n * np.log(K)
        if score < best_score:
            best, best_score = est, score
    return best

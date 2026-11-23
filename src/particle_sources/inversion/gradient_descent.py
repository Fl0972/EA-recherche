"""
Descente de gradient simple sur les moindres carrés (méthode de référence de
l'industriel).

On minimise la perte relative, non pondérée,

    L(theta) = 1/2 * ||mu(theta) - y||^2 / ||y||^2,      mu = Phi(sources)

par  theta <- P(theta - pas * grad L),  où P projette sur les contraintes
(positions dans le domaine, intensités >= 0). Le nombre de sources est imposé.

Les paramètres sont normalisés : positions ramenées à [0, 1] dans le domaine,
intensités divisées par une intensité de référence. Sans cela, positions et
intensités ont des échelles si différentes que la descente ne converge même
pas pour deux sources égales.

Le pas est soit fixe, soit choisi par recherche linéaire d'Armijo (on tente
de doubler le pas à chaque itération, puis on le divise par deux tant que la
perte ne diminue pas assez). Il n'y a pas d'autre accélération (ni moment, ni
préconditionnement) : c'est ce qui rend la méthode sensible au rapport
d'intensité entre les sources.
"""

from dataclasses import dataclass

import numpy as np

from ..model import Box, Sources, intensity_at, intensity_jacobian
from .problem import InverseProblem

# Constante de diminution suffisante d'Armijo
ARMIJO_C = 1e-4
# Pas minimal de la recherche linéaire (en dessous : on considère la descente bloquée)
MIN_STEP = 1e-14


@dataclass(frozen=True)
class DescentResult:
    sources: Sources          # estimation finale
    losses: np.ndarray        # perte à chaque itération, (n_iter + 1,)
    trajectory: np.ndarray    # positions enregistrées, (n_records, N, d)
    record_iters: np.ndarray  # itérations correspondant à `trajectory`
    n_iter: int               # nombre d'itérations effectuées
    converged: bool           # perte < loss_tol avant max_iter


class _Normalization:
    """Passage entre sources et paramètres normalisés u = [positions, intensités]."""

    def __init__(self, box: Box, n: int, ref_intensity: float):
        self.box, self.n, self.ref = box, n, ref_intensity
        self.scale = np.concatenate([np.tile(box.size, n), np.full(n, ref_intensity)])

    def to_params(self, sources: Sources) -> np.ndarray:
        pos = (sources.positions - self.box.lo) / self.box.size
        return np.concatenate([pos.ravel(), sources.intensities / self.ref])

    def to_sources(self, u: np.ndarray) -> Sources:
        d = self.box.dim
        pos = self.box.lo + u[: self.n * d].reshape(self.n, d) * self.box.size
        return Sources(pos, u[self.n * d:] * self.ref)

    def project(self, u: np.ndarray) -> np.ndarray:
        nd = self.n * self.box.dim
        return np.concatenate([np.clip(u[:nd], 0.0, 1.0), np.maximum(u[nd:], 0.0)])


def gradient_descent(y: np.ndarray, problem: InverseProblem, init: Sources,
                     max_iter: int = 20000, step: str | float = "armijo",
                     loss_tol: float = 1e-20, max_move: float = 0.05,
                     record_every: int = 10) -> DescentResult:
    """
    Descente de gradient projetée à partir de `init` (dont le nombre de sources
    est conservé). `step` : "armijo" ou un pas fixe (en paramètres normalisés).
    `max_move` : déplacement maximal d'un paramètre normalisé par itération
    (sans cette limite, les premiers pas envoient les sources sur les bords du
    domaine, où la projection les fusionne).
    La contribution des sources connues (problem.known) est retirée de y.
    """
    target = y - problem.known_contribution()
    x_s, eps = problem.sensors.positions, problem.eps
    y_norm2 = float(target @ target)
    norm = _Normalization(problem.box, init.n,
                          init.total_intensity or problem.total_intensity)

    def residual(u):
        return intensity_at(norm.to_sources(u), x_s, eps) - target

    def loss_of(r):
        return 0.5 * float(r @ r) / y_norm2

    def grad_of(u, r):
        J = intensity_jacobian(norm.to_sources(u), x_s, eps) * norm.scale  # dérivées / u
        return J.T @ r / y_norm2

    u = norm.project(norm.to_params(init))
    r = residual(u)
    loss, grad = loss_of(r), grad_of(u, r)
    losses, trajectory, record_iters = [loss], [norm.to_sources(u).positions], [0]
    lr = 1.0 if step == "armijo" else float(step)

    def capped(lr):
        biggest = np.abs(grad).max()
        return min(lr, max_move / biggest) if biggest > 0 else lr

    it = 0
    while it < max_iter and loss >= loss_tol:
        it += 1
        if step == "armijo":
            lr = capped(2 * lr)
            while True:
                v = norm.project(u - lr * grad)
                r = residual(v)
                new_loss = loss_of(r)
                if new_loss <= loss - ARMIJO_C * float(grad @ (u - v)) or lr < MIN_STEP:
                    break
                lr /= 2
        else:
            v = norm.project(u - capped(lr) * grad)
            r = residual(v)
            new_loss = loss_of(r)
        u, loss, grad = v, new_loss, grad_of(v, r)
        losses.append(loss)
        if it % record_every == 0:
            trajectory.append(norm.to_sources(u).positions)
            record_iters.append(it)

    if record_iters[-1] != it:
        trajectory.append(norm.to_sources(u).positions)
        record_iters.append(it)
    return DescentResult(norm.to_sources(u), np.array(losses), np.array(trajectory),
                         np.array(record_iters), it, loss < loss_tol)


def curvature_spectrum(sources: Sources, problem: InverseProblem) -> np.ndarray:
    """
    Valeurs propres (croissantes) de la hessienne de la perte en `sources`,
    en paramètres normalisés, quand les mesures sont exactes (résidu nul) :
        H = J^T J / ||y||^2.
    Une descente de gradient avance d'un facteur ~ lambda_min / lambda_max par
    itération dans la direction la plus plate : le nombre d'itérations
    nécessaire croît comme le conditionnement lambda_max / lambda_min.
    """
    x_s, eps = problem.sensors.positions, problem.eps
    y = intensity_at(sources, x_s, eps)
    norm = _Normalization(problem.box, sources.n, sources.total_intensity)
    J = intensity_jacobian(sources, x_s, eps) * norm.scale
    return np.linalg.eigvalsh(J.T @ J / float(y @ y))


def random_init(box: Box, n_sources: int, total_intensity: float,
                rng: np.random.Generator) -> Sources:
    """Positions uniformes dans le domaine, intensités égales."""
    pos = box.lo + rng.random((n_sources, box.dim)) * box.size
    return Sources(pos, np.full(n_sources, total_intensity / n_sources))


def gradient_descent_method(y: np.ndarray, problem: InverseProblem, n_sources: int,
                            max_iter: int = 20000, step: str | float = "armijo",
                            seed: int | None = 0) -> Sources:
    """Méthode d'inversion (y, problem) -> Sources : init aléatoire, N imposé."""
    init = random_init(problem.box, n_sources, problem.total_intensity,
                       np.random.default_rng(seed))
    return gradient_descent(y, problem, init, max_iter, step, record_every=max_iter).sources

"""
Cas test de l'industriel : deux sources, l'une `x` fois plus intense que
l'autre, reconstruites par descente de gradient. D'après eux, la descente
retrouve les deux sources pour x = 10 mais plus pour x = 100.

On compare, pour chaque rapport x :
  - la descente de gradient (leur méthode) ;
  - Gauss-Newton (scipy least_squares) depuis la même initialisation : même
    problème, autre optimiseur, pour savoir si l'échec vient de l'optimiseur ;
  - Prony + maximum de vraisemblance (notre méthode, sans initialisation ni
    nombre de sources donné).

Deux expériences :
  - cas fixe : positions et initialisation données (config), un run par x ;
  - essais aléatoires : positions et initialisation tirées uniformément dans
    le domaine, taux de réussite par x.
"""

import time
from dataclasses import dataclass

import numpy as np

from ..config import registry as reg
from ..generation import bright_and_faint
from ..inversion import (InverseProblem, curvature_spectrum, gradient_descent, prony_mle,
                         random_init, refine)
from ..model import Box, Sources
from ..sensors import measure
from .metrics import match

GD_LABEL = "Descente de gradient"
GN_LABEL = "Gauss-Newton (même init.)"
PRONY_LABEL = "Prony + max. vraisemblance"
METHOD_LABELS = [GD_LABEL, GN_LABEL, PRONY_LABEL]


@dataclass(frozen=True)
class TwoSourcesSpec:
    positions: np.ndarray        # (2, 2) : source forte, puis faible
    ratios: list[float]          # rapports d'intensité forte / faible testés
    noise: dict                  # {name, params} (voir sensors.yaml)
    init_positions: np.ndarray   # (2, 2) : initialisation du cas fixe
    max_iter: int
    step: str | float
    loss_tol: float
    max_move: float
    success_tol: float           # erreur de position max pour réussir (chaque source)
    show_ratios: list[float]     # rapports dont on trace les trajectoires
    n_trials: int                # essais aléatoires par rapport (0 = aucun)
    trials_seed: int | None
    output_dir: str

    @classmethod
    def from_raw(cls, raw: dict) -> "TwoSourcesSpec":
        if "two_sources" not in raw:
            raise KeyError("section `two_sources` absente de la config "
                           "(config/two_sources.yaml)")
        t = raw["two_sources"]
        descent = t.get("descent") or {}
        trials = t.get("random_trials") or {}
        return cls(
            positions=np.asarray(t["positions"], dtype=float),
            ratios=[float(x) for x in t["ratios"]],
            noise=t.get("noise") or {"name": "none"},
            init_positions=np.asarray(t["init"]["positions"], dtype=float),
            max_iter=int(descent.get("max_iter", 20000)),
            step=descent.get("step", "armijo"),
            loss_tol=float(descent.get("loss_tol", 1e-20)),
            max_move=float(descent.get("max_move", 0.05)),
            success_tol=float(t.get("success_tol", 0.1)),
            show_ratios=[float(x) for x in t.get("show_ratios", [])],
            n_trials=int(trials.get("n", 0)),
            trials_seed=trials.get("seed"),
            output_dir=t.get("output_dir", "outputs/two_sources"),
        )


def source_errors(true: Sources, est: Sources) -> np.ndarray:
    """Erreur de position de chaque vraie source (inf si non associée)."""
    errors = np.full(true.n, np.inf)
    i_true, i_est = match(true, est)
    errors[i_true] = np.linalg.norm(true.positions[i_true] - est.positions[i_est], axis=1)
    return errors


def _run_methods(y, problem: InverseProblem, init: Sources, spec: TwoSourcesSpec):
    """Applique les trois méthodes. Renvoie {label: (estimation, temps_ms)} et la descente."""
    out = {}
    t0 = time.perf_counter()
    descent = gradient_descent(y, problem, init, spec.max_iter, spec.step, spec.loss_tol,
                               spec.max_move)
    out[GD_LABEL] = (descent.sources, 1e3 * (time.perf_counter() - t0))
    t0 = time.perf_counter()
    out[GN_LABEL] = (refine(y, problem, init), 1e3 * (time.perf_counter() - t0))
    t0 = time.perf_counter()
    out[PRONY_LABEL] = (prony_mle(y, problem), 1e3 * (time.perf_counter() - t0))
    return out, descent


def _problem(sensors, box: Box, noise, total_intensity: float) -> InverseProblem:
    return InverseProblem(sensors=sensors, box=box, noise=noise, known=Sources.empty(box.dim),
                          total_intensity=total_intensity)


def run_fixed_case(spec: TwoSourcesSpec, sensors, box: Box, total_intensity: float,
                   rng: np.random.Generator):
    """
    Cas fixe, un run par rapport x. Renvoie (lignes, descentes, vérités) :
      lignes : une par (x, méthode) avec err_strong, err_faint (erreurs de
               position), success, n_iter et final_loss (descente), time_ms,
               condition (conditionnement de la perte à la solution) ;
      descentes, vérités : {x: DescentResult}, {x: Sources}, pour les figures.
    """
    noise = reg.build(reg.NOISE_MODELS, spec.noise, "two_sources.noise")
    problem = _problem(sensors, box, noise, total_intensity)
    init = Sources(spec.init_positions, np.full(2, total_intensity / 2))
    rows, descents, truths = [], {}, {}
    for x in spec.ratios:
        true = bright_and_faint(spec.positions, x, total_intensity)
        y = measure(true, sensors, rng, noise)
        eig = curvature_spectrum(true, problem)
        results, descent = _run_methods(y, problem, init, spec)
        descents[x], truths[x] = descent, true
        for label, (est, ms) in results.items():
            err = source_errors(true, est)
            rows.append({
                "ratio": x, "method": label, "n_sources": est.n,
                "err_strong": float(err[0]), "err_faint": float(err[1]),
                "success": bool(est.n == 2 and np.all(err < spec.success_tol)),
                "n_iter": descent.n_iter if label == GD_LABEL else None,
                "final_loss": float(descent.losses[-1]) if label == GD_LABEL else None,
                "time_ms": ms, "condition": float(eig[-1] / eig[0]),
            })
    return rows, descents, truths


def run_random_trials(spec: TwoSourcesSpec, sensors, box: Box, total_intensity: float,
                      progress=None):
    """
    Pour chaque x, `n_trials` situations : positions et initialisation tirées
    uniformément dans le domaine (les mêmes pour tous les x et toutes les
    méthodes). Renvoie une ligne par (x, méthode) : success_rate et erreur
    médiane sur la source faible.
    """
    noise = reg.build(reg.NOISE_MODELS, spec.noise, "two_sources.noise")
    problem = _problem(sensors, box, noise, total_intensity)
    rng_pos, rng_init, rng_noise = (np.random.default_rng(s) for s in
                                    np.random.SeedSequence(spec.trials_seed).spawn(3))
    positions = [box.lo + rng_pos.random((2, box.dim)) * box.size for _ in range(spec.n_trials)]
    inits = [random_init(box, 2, total_intensity, rng_init) for _ in range(spec.n_trials)]

    rows = []
    for x in spec.ratios:
        success = {label: [] for label in METHOD_LABELS}
        faint = {label: [] for label in METHOD_LABELS}
        for pos, init in zip(positions, inits):
            true = bright_and_faint(pos, x, total_intensity)
            y = measure(true, sensors, rng_noise, noise)
            results, _ = _run_methods(y, problem, init, spec)
            for label, (est, _) in results.items():
                err = source_errors(true, est)
                success[label].append(est.n == 2 and np.all(err < spec.success_tol))
                faint[label].append(err[1])
        for label in METHOD_LABELS:
            rows.append({"ratio": x, "method": label, "n_trials": spec.n_trials,
                         "success_rate": float(np.mean(success[label])),
                         "median_err_faint": float(np.median(faint[label]))})
        if progress:
            progress(x)
    return rows

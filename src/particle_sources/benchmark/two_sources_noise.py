"""
Cas test à deux sources avec bruit de mesure, rapport d'intensité x ~ 100.

Sans bruit, la seule difficulté est l'optimisation (voir intensity_ratio.py).
Avec bruit, s'ajoute une limite d'information : la borne de Cramér-Rao (CRB)
donne l'erreur minimale de tout estimateur non biaisé. On mesure, pour chaque
niveau de bruit sigma (gaussien relatif) :
  - l'erreur de position de la source faible et de la source forte ;
  - le rapport erreur / CRB (≈ 0,7-0,8 : l'estimateur atteint la borne) ;
  - la détection (Prony + MV estime-t-il bien N = 2 ?) et la réussite.

Deux expériences :
  - géométrie fixe (two_sources.positions), plusieurs rapports x, plusieurs
    tirages de bruit ;
  - géométries aléatoires (positions et initialisation uniformes), x fixé.
"""

from dataclasses import dataclass

import numpy as np

from ..generation import bright_and_faint
from ..inversion import (InverseProblem, gradient_descent, prony_mle, random_init,
                         refine)
from ..model import Box, Sources
from ..sensors import measure, noise
from .crb import crb
from .intensity_ratio import GD_LABEL, GN_LABEL, PRONY_LABEL, TwoSourcesSpec, source_errors


@dataclass(frozen=True)
class NoiseStudySpec:
    ratios: list[float]          # rapports x (géométrie fixe)
    sigmas: list[float]          # niveaux de bruit relatif
    n_realizations: int          # tirages de bruit par (x, sigma), géométrie fixe
    descent_realizations: int    # idem pour la descente de gradient (lente)
    descent_ratio: float         # rapport x où l'on fait tourner la descente
    geometry_ratio: float        # rapport x des géométries aléatoires
    n_geometries: int            # géométries aléatoires par sigma (0 = aucune)
    seed: int | None
    output_dir: str

    @classmethod
    def from_raw(cls, raw: dict) -> "NoiseStudySpec":
        if "two_sources_noise" not in raw:
            raise KeyError("section `two_sources_noise` absente de la config "
                           "(config/two_sources_noise.yaml)")
        t = raw["two_sources_noise"]
        return cls(
            ratios=[float(x) for x in t["ratios"]],
            sigmas=[float(s) for s in t["sigmas"]],
            n_realizations=int(t.get("n_realizations", 50)),
            descent_realizations=int(t.get("descent_realizations", 0)),
            descent_ratio=float(t.get("descent_ratio", 100)),
            geometry_ratio=float(t.get("geometry_ratio", 100)),
            n_geometries=int(t.get("n_geometries", 0)),
            seed=t.get("seed"),
            output_dir=t.get("output_dir", "outputs/two_sources_noise"),
        )


def _problem(sensors, box: Box, sigma: float, total_intensity: float) -> InverseProblem:
    return InverseProblem(sensors=sensors, box=box, noise=noise.gaussian_relative(sigma),
                          known=Sources.empty(box.dim), total_intensity=total_intensity)


def _estimate(label, y, problem, init, base: TwoSourcesSpec) -> Sources:
    if label == PRONY_LABEL:
        return prony_mle(y, problem)
    if label == GN_LABEL:
        return refine(y, problem, init)
    return gradient_descent(y, problem, init, base.max_iter, base.step, base.loss_tol,
                            base.max_move, record_every=base.max_iter).sources


def _summary(errors: np.ndarray, n_found: np.ndarray, crb_pos: np.ndarray,
             success_tol: float, estimates_n: bool = True) -> dict:
    """
    errors : (M, 2) erreurs forte / faible ; n_found : (M,) ; crb_pos : (M, 2).
    `estimates_n` : la méthode estime-t-elle N (sinon détection sans objet : nan).
    """
    found = n_found == 2
    ok = found & np.all(errors < success_tol, axis=1)
    faint = errors[found, 1]
    ratio = faint / crb_pos[found, 1] if found.any() else np.empty(0)
    med = (lambda a: float(np.median(a)) if a.size else float("nan"))
    return {
        "detection": float(found.mean()) if estimates_n else float("nan"),
        "success": float(ok.mean()),
        "err_faint": med(faint),
        "err_strong": med(errors[found, 0]),
        "faint_over_crb": med(ratio),
        "crb_faint": float(np.median(crb_pos[:, 1])),
        "crb_strong": float(np.median(crb_pos[:, 0])),
    }


def run_fixed_geometry(spec: NoiseStudySpec, base: TwoSourcesSpec, sensors, box: Box,
                       total_intensity: float, progress=None) -> list[dict]:
    """Une ligne par (x, sigma, méthode), géométrie et initialisation du cas fixe."""
    init = Sources(base.init_positions, np.full(2, total_intensity / 2))
    rng = np.random.default_rng(spec.seed)
    rows = []
    for x in spec.ratios:
        true = bright_and_faint(base.positions, x, total_intensity)
        for sigma in spec.sigmas:
            problem = _problem(sensors, box, sigma, total_intensity)
            crb_pos = np.tile(crb(true, problem)[0], (spec.n_realizations, 1))
            labels = [PRONY_LABEL, GN_LABEL]
            if x == spec.descent_ratio and spec.descent_realizations:
                labels.append(GD_LABEL)
            ys = [measure(true, sensors, rng, problem.noise)
                  for _ in range(spec.n_realizations)]
            for label in labels:
                m = spec.descent_realizations if label == GD_LABEL else spec.n_realizations
                ests = [_estimate(label, y, problem, init, base) for y in ys[:m]]
                errors = np.array([source_errors(true, e) for e in ests])
                n_found = np.array([e.n for e in ests])
                rows.append({"ratio": x, "sigma": sigma, "method": label, "n": m,
                             **_summary(errors, n_found, crb_pos[:m], base.success_tol,
                                         label == PRONY_LABEL)})
            if progress:
                progress(x, sigma)
    return rows


def run_random_geometries(spec: NoiseStudySpec, base: TwoSourcesSpec, sensors, box: Box,
                          total_intensity: float, progress=None):
    """
    Géométries et initialisations aléatoires (mêmes pour tous les sigma), x fixé.
    Renvoie (lignes, crb_par_géométrie) : une ligne par (sigma, méthode), plus
    `localizable` = proportion de géométries où la CRB de la source faible est
    sous le seuil de réussite ; et, pour la figure, la CRB de la source faible
    divisée par sigma et la distance relative au centre rho / R de chaque géométrie.
    """
    x = spec.geometry_ratio
    rng_pos, rng_init, rng_noise = (np.random.default_rng(s) for s in
                                    np.random.SeedSequence(spec.seed).spawn(3))
    truths = [bright_and_faint(box.lo + rng_pos.random((2, box.dim)) * box.size, x,
                               total_intensity) for _ in range(spec.n_geometries)]
    inits = [random_init(box, 2, total_intensity, rng_init) for _ in truths]

    # Bruit gaussien relatif : Fisher ∝ (1/sigma^2 + 2), donc
    # CRB(sigma) = CRB_lin * sigma / sqrt(1 + 2 sigma^2), CRB_lin calculée une fois.
    tiny = 1e-6
    small = _problem(sensors, box, tiny, total_intensity)
    crb_lin = np.array([crb(t, small)[0] for t in truths]) / tiny
    rho = np.array([np.linalg.norm(t.positions[1] - box.center) for t in truths])
    rho = rho / box.circumradius

    rows = []
    for sigma in spec.sigmas:
        problem = _problem(sensors, box, sigma, total_intensity)
        crb_pos = crb_lin * sigma / np.sqrt(1 + 2 * sigma**2)
        ys = [measure(t, sensors, rng_noise, problem.noise) for t in truths]
        for label in (PRONY_LABEL, GN_LABEL):
            ests = [_estimate(label, y, problem, i, base) for y, i in zip(ys, inits)]
            errors = np.array([source_errors(t, e) for t, e in zip(truths, ests)])
            n_found = np.array([e.n for e in ests])
            rows.append({"ratio": x, "sigma": sigma, "method": label, "n": len(truths),
                         "localizable": float(np.mean(crb_pos[:, 1] < base.success_tol)),
                         **_summary(errors, n_found, crb_pos, base.success_tol,
                                         label == PRONY_LABEL)})
        if progress:
            progress(sigma)
    return rows, {"crb_faint_over_sigma": crb_lin[:, 1], "rho_faint": rho}

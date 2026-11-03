"""
Comparaison d'un ensemble de sources estimé à la vérité.

Les deux ensembles n'ont pas forcément le même nombre de sources ni le même
ordre : on associe estimées et vraies sources par l'algorithme hongrois
(distance minimale), puis on mesure les erreurs.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..model import Sources


def match(true: Sources, est: Sources) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i_true, i_est) de l'association de distance totale minimale."""
    if true.n == 0 or est.n == 0:
        return np.empty(0, int), np.empty(0, int)
    dist = np.linalg.norm(true.positions[:, None] - est.positions[None], axis=-1)
    return linear_sum_assignment(dist)


def ospa(true: Sources, est: Sources, cutoff: float, p: float = 2.0) -> float:
    """
    Distance OSPA (Schuhmacher et al., 2008) entre ensembles de points :
    erreur de position moyenne, chaque source manquante ou en trop comptant
    comme une erreur `cutoff`. 0 si identiques, `cutoff` au pire.
    """
    m, n = sorted((true.n, est.n))
    if n == 0:
        return 0.0
    i, j = match(true, est)
    d = np.linalg.norm(true.positions[i] - est.positions[j], axis=-1)
    total = np.sum(np.minimum(d, cutoff) ** p) + cutoff**p * (n - m)
    return float((total / n) ** (1 / p))


@dataclass
class SituationResult:
    n_true: int
    n_est: int
    ospa: float
    position_errors: np.ndarray    # distances, si n_est == n_true (sinon vide)
    intensity_errors: np.ndarray   # |I_est - I_true|, idem
    position_ratios: np.ndarray    # erreur / écart-type de Cramér-Rao, idem
    intensity_ratios: np.ndarray   # (sources de borne nulle exclues)


def _ratios(err: np.ndarray, bound: np.ndarray | None) -> np.ndarray:
    if bound is None:
        return np.empty(0)
    ok = bound > 0
    return err[ok] / bound[ok]


def evaluate(true: Sources, est: Sources, cutoff: float,
             crb_pos: np.ndarray | None = None,
             crb_int: np.ndarray | None = None) -> SituationResult:
    """crb_pos, crb_int : bornes de Cramér-Rao par vraie source (même ordre que true)."""
    pos_err = int_err = pos_ratio = int_ratio = np.empty(0)
    if est.n == true.n and true.n > 0:
        i, j = match(true, est)
        pos_err = np.linalg.norm(true.positions[i] - est.positions[j], axis=-1)
        int_err = np.abs(true.intensities[i] - est.intensities[j])
        pos_ratio = _ratios(pos_err, None if crb_pos is None else crb_pos[i])
        int_ratio = _ratios(int_err, None if crb_int is None else crb_int[i])
    return SituationResult(true.n, est.n, ospa(true, est, cutoff),
                           pos_err, int_err, pos_ratio, int_ratio)

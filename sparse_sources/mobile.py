"""
Capteur mobile : on dispose des capteurs fixes du projet et d'un capteur
que l'on peut déplacer où l'on veut dans D pour faire des mesures
supplémentaires, une par position.

Stratégie adaptative (ML + BIC à chaque étape) :
    1. ML + BIC sur les mesures déjà faites -> sources estimées (ẑ, Î) ;
    2. on choisit les ``batch`` positions suivantes une à une : chacune est
       celle, parmi une grille de candidats, qui fait le plus baisser la
       somme des variances de Cramér-Rao des positions estimées
       (plan « A-optimal »), le bruit prévu étant σ × (signal prévu, source
       centrale comprise) ; la matrice de covariance est mise à jour après
       chaque choix (formule de Sherman-Morrison) ;
    3. on mesure à ces positions, on relance ML + BIC en partant de
       l'estimation précédente, et on recommence.

Stratégies de référence, même nombre de mesures : positions tirées au hasard
dans D, ou grille régulière fixée à l'avance.
"""

from __future__ import annotations

import time

import numpy as np

from .kernel import Setup
from .lasso import build_grid
from .ml import cramer_rao, greedy_ml, relative_noise_std


def candidate_points(box, step=0.25):
    """
    Positions possibles du capteur mobile : centres des cases de pas ``step``
    de D. Elles ne tombent ni sur les nœuds de la grille de recherche de ML
    (signal infini) ni sur le centre de D (source centrale) si D/step est pair.
    """
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    n = np.maximum(np.round((hi - lo) / step).astype(int), 1)
    xs = lo[0] + (np.arange(n[0]) + 0.5) * (hi[0] - lo[0]) / n[0]
    ys = lo[1] + (np.arange(n[1]) + 0.5) * (hi[1] - lo[1]) / n[1]
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def serpentine(box, n_lines, step=0.25):
    """
    Trajet d'un capteur qui enregistre en continu : ``n_lines`` lignes
    parallèles au grand côté de D (aller-retour, « en serpentin »), une
    mesure tous les ``step`` le long de chaque ligne. Les lignes sont au
    centre de bandes de même largeur et les mesures au centre de segments de
    longueur ``step`` : aucune ne tombe sur un nœud de la grille de
    recherche de ML (pas 0,25) ni sur le centre de D si n_lines est pair.
    Renvoie les positions (N, 2) dans l'ordre du trajet.
    """
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    xs = lo[0] + (np.arange(n_lines) + 0.5) * (hi[0] - lo[0]) / n_lines
    n = max(int(round((hi[1] - lo[1]) / step)), 1)
    ys = lo[1] + (np.arange(n) + 0.5) * (hi[1] - lo[1]) / n
    return np.vstack([np.column_stack([np.full(n, x), ys if k % 2 == 0 else ys[::-1]])
                      for k, x in enumerate(xs)])


def _jacobian_rows(z, I, points, box):
    """Lignes de la matrice des dérivées pour des capteurs en ``points`` : (C, 3K)."""
    S = Setup(np.atleast_2d(points), box, 0.0)
    P, J = S.phi(z), S.grad(z)                            # (K, C), (K, C, 2)
    return np.concatenate([P.T, (I[:, None, None] * J).transpose(1, 0, 2)
                           .reshape(P.shape[1], -1)], 1)


def next_positions(z, I, sensors, noise_std, candidates, box, sigma, central, n=1,
                   center=None):
    """
    ``n`` positions A-optimales pour les sources estimées (z, I), choisies une
    à une. ``central`` : intensité de la source centrale (au centre de D),
    qui fixe en grande partie le bruit prévu. Renvoie (n, 2).
    """
    z, I = np.asarray(z, float).reshape(-1, 2), np.asarray(I, float).reshape(-1)
    cand = np.asarray(candidates, float)
    if len(I) == 0:                                       # rien d'estimé : au hasard
        return cand[np.random.default_rng(0).choice(len(cand), n, replace=False)]
    K = len(I)
    cov = cramer_rao(z, I, noise_std, Setup(sensors, box, 0.0))["cov"]
    G = _jacobian_rows(z, I, cand, box)                   # (C, 3K)
    c = np.asarray(box.center if center is None else center, float)[None]
    pred = Setup(cand, box, 0.0).phi(z).T @ I + central * Setup(cand, box, 0.0).phi(c)[0]
    s2 = (sigma * pred) ** 2
    chosen = []
    for _ in range(n):
        U = G @ cov                                       # (C, 3K)
        den = s2 + np.einsum("ij,ij->i", U, G)
        with np.errstate(invalid="ignore", divide="ignore"):
            gain = np.sum(U[:, K:] ** 2, 1) / den         # baisse de trace(cov positions)
        gain[~np.isfinite(gain)] = -np.inf                # candidat sur une source estimée
        gain[chosen] = -np.inf
        j = int(np.argmax(gain))
        chosen.append(j)
        u = U[j]
        cov = cov - np.outer(u, u) / den[j]
    return cand[chosen]


def adaptive_ml(measure, sensors0, m0, y0, box, sigma, n_moves, batch=8, central=5000.0,
                checkpoints=(), candidate_step=0.25, k_max=60, verbose=False):
    """
    Capteurs fixes ``sensors0`` (mesures m0, signal y0 pour le bruit), puis
    ``n_moves`` mesures du capteur mobile choisies par ``next_positions``.

    measure(points) -> (m, y) : mesures (source centrale retirée) et signal
    brut aux positions demandées (simulation). Renvoie la liste des états
    aux ``checkpoints`` (nombre de mesures mobiles) et à la fin :
    {n_mobile, z, I, K, chi2, sensors, time}.
    """
    t0 = time.perf_counter()
    sensors, m, y = np.asarray(sensors0, float), np.asarray(m0, float), np.asarray(y0, float)
    cand = candidate_points(box, candidate_step)

    def fit(z_init=None, I_init=None):
        S = Setup(sensors, box, 0.0, normalize=True)
        s = relative_noise_std(y, sigma)
        return greedy_ml(m, s, setup=S, grid=build_grid(step=0.25, setup=S), k_max=k_max,
                         z_init=z_init, I_init=I_init), s

    res, s = fit()
    out = []
    marks = sorted(set(checkpoints) | {n_moves})
    done = 0
    while done < n_moves:
        k = min(batch, n_moves - done)
        # bruit prévu aux capteurs déjà en place : celui des mesures (connu)
        pts = next_positions(res["z"], res["I"], sensors, s, cand, box, sigma, central, n=k)
        m_new, y_new = measure(pts)
        sensors = np.vstack([sensors, pts])
        m, y = np.concatenate([m, m_new]), np.concatenate([y, y_new])
        done += k
        res, s = fit(res["z"], res["I"])
        if verbose:
            print(f"  {done:3d} mesures mobiles : K = {res['K']}, χ²/M = "
                  f"{res['chi2'] / len(m):.2f}", flush=True)
        if done in marks:
            out.append({"n_mobile": done, "z": res["z"], "I": res["I"], "K": res["K"],
                        "chi2": res["chi2"], "sensors": sensors.copy(),
                        "time": time.perf_counter() - t0})
    return out

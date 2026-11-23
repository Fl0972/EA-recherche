"""
Expérience 4 : loi a posteriori par MCMC (Metropolis adaptatif), K fixé.

Modèle bayésien (le même que l'évidence de Laplace de ``chemins.py``) :
    vraisemblance  m_i ~ N(A(z, I)_i, s_i²)
    a priori       z_k uniforme sur D, I_k log-uniforme sur [0,1 ; 1000]
Paramètres échantillonnés : θ = (z_k, log I_k)_k (log I : a priori uniforme).

Algorithme : Metropolis à marche aléatoire gaussienne, covariance adaptée
(Haario et al. 2001) : pendant la chauffe, la covariance de la proposition
est la covariance empirique des échantillons × 2,38² / d. Départ : la
solution ML (on échantillonne autour du mode trouvé ; des départs
multiples révéleraient d'autres modes).

Ce que l'on mesure : la CALIBRATION des incertitudes. Pour chaque source,
on calcule l'ellipse de crédibilité à 90 % (covariance a posteriori) et on
regarde si la vraie position est dedans. Bien calibré ⇔ 90 % des vraies
sources dedans. On compare à l'ellipse de Cramér-Rao (approximation
gaussienne locale).

    python -m sparse_sources.apprentissage.mcmc
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import chi2 as chi2_law  # noqa: E402

from sparse_sources import cramer_rao  # noqa: E402

from .base import OUT, parallel  # noqa: E402
from .chemins import I_MAX, I_MIN  # noqa: E402
from .donnees import bases  # noqa: E402
from .rapport import save_csv  # noqa: E402
from .selection_k import pick, rule_choice  # noqa: E402
from .simulation import PROJET, Scene  # noqa: E402

Q90 = chi2_law.ppf(0.9, 2)


def log_post(theta, d, scene: Scene):
    """log π(θ | m) à une constante près ; θ = (x1, y1, logI1, x2, ...)."""
    t = theta.reshape(-1, 3)
    z, logI = t[:, :2], t[:, 2]
    if np.any(z < scene.lo) or np.any(z > scene.hi) or np.any(logI < np.log(I_MIN)) \
            or np.any(logI > np.log(I_MAX)):
        return -np.inf
    r = (d["m"] - scene.field(z, np.exp(logI))) / d["s"]
    return -0.5 * float(r @ r)


def metropolis(d, z0, I0, scene: Scene = PROJET, n_iter=40000, burn=15000, seed=0):
    """Échantillons a posteriori (n, K, 3) et taux d'acceptation."""
    rng = np.random.default_rng(seed)
    theta = np.concatenate([np.r_[z, np.log(max(i, 1.1 * I_MIN))] for z, i in zip(z0, I0)])
    dim = len(theta)
    lp = log_post(theta, d, scene)
    # échelle de départ : Cramér-Rao (sur z, log I)
    crb = cramer_rao(z0, I0, d["s"], scene.setup)
    K = len(I0)
    sd = np.concatenate([np.r_[np.sqrt(np.diag(crb["cov"])[K + 2 * k:K + 2 * k + 2]),
                               np.sqrt(crb["cov"][k, k]) / max(I0[k], 1e-6)] for k in range(K)])
    C0 = np.diag(np.clip(sd, 1e-6, 5.0) ** 2)
    C = C0.copy()
    log_scale = np.log(2.38**2 / dim)
    chain, acc, acc_win, acc_after = [], 0, 0, 0
    L = np.linalg.cholesky(np.exp(log_scale) * C)
    for it in range(n_iter):
        prop = theta + L @ rng.standard_normal(dim)
        lq = log_post(prop, d, scene)
        ok = np.log(rng.random()) < lq - lp
        if ok:
            theta, lp = prop, lq
            acc += 1
            acc_win += 1
            if it >= burn:
                acc_after += 1
        chain.append(theta.copy())
        # adaptation pendant la chauffe, toutes les 500 itérations :
        #   covariance = covariance empirique (mélangée à celle de Cramér-Rao),
        #   échelle réglée pour viser 23 % d'acceptation (Robbins-Monro)
        if it < burn and it % 500 == 499:
            log_scale += (acc_win / 500 - 0.234) * 2.0
            acc_win = 0
            if it >= 2000:
                X = np.array(chain[it // 2:])
                C = 0.9 * np.cov(X.T) + 0.1 * C0
            try:
                L = np.linalg.cholesky(np.exp(log_scale) * C + 1e-14 * np.eye(dim))
            except np.linalg.LinAlgError:
                pass
    X = np.array(chain[burn:]).reshape(-1, K, 3)
    return X, acc_after / (n_iter - burn)


def one_case(args):
    i, d, z0, I0 = args
    K = len(I0)
    t = time.perf_counter()
    X, acc = metropolis(d, z0, I0, seed=i)
    dt = time.perf_counter() - t
    # appariement estimation ↔ vraies sources
    D = np.linalg.norm(d["z"][:, None] - z0[None], axis=-1)
    a, b = linear_sum_assignment(D)
    crb = cramer_rao(z0, I0, d["s"], PROJET.setup)["cov"]
    out = []
    for ti, k in zip(a, b):
        S = np.cov(X[:, k, :2].T)
        mu = X[:, k, :2].mean(0)
        e = d["z"][ti] - mu
        in_post = float(e @ np.linalg.solve(S, e)) <= Q90
        Sc = crb[K + 2 * k:K + 2 * k + 2, K + 2 * k:K + 2 * k + 2]
        e2 = d["z"][ti] - z0[k]
        in_crb = float(e2 @ np.linalg.solve(Sc, e2)) <= Q90
        out.append({"i": i, "n": d["n"], "sigma": d["sigma"], "K_est": K,
                    "dans_ellipse_post": in_post, "dans_ellipse_crb": in_crb,
                    "sd_post": float(np.sqrt(np.trace(S))), "sd_crb": float(np.sqrt(np.trace(Sc))),
                    "erreur": float(np.linalg.norm(e2)), "acceptation": acc, "temps": dt})
    return out


def main():
    t0 = time.perf_counter()
    _, _, test, pte = bases()
    cases = []
    for i, (d, p) in enumerate(zip(test, pte)):
        if i % 3:                 # un tirage sur trois (200 tirages)
            continue
        z, I = rule_choice(p, "bic")
        if len(I) == d["n"]:      # calibration à K correct
            cases.append((i, d, z, I))
    rows = [r for out in parallel(one_case, cases) for r in out]
    save_csv(rows, OUT / "mcmc.csv")
    print(f"  {len(cases)} tirages, {len(rows)} sources, {time.perf_counter() - t0:.0f} s")
    summ = []
    for key, sel in [("toutes", rows)] + [(f"sigma={s:g}", [r for r in rows if r["sigma"] == s])
                                          for s in (1e-4, 1e-3, 3e-3)] + \
            [(f"n={n}", [r for r in rows if r["n"] == n]) for n in range(1, 6)]:
        if not sel:
            continue
        summ.append({"groupe": key, "sources": len(sel),
                     "couverture_post_90": float(np.mean([r["dans_ellipse_post"] for r in sel])),
                     "couverture_crb_90": float(np.mean([r["dans_ellipse_crb"] for r in sel])),
                     "sd_post_median": float(np.median([r["sd_post"] for r in sel])),
                     "sd_crb_median": float(np.median([r["sd_crb"] for r in sel])),
                     "erreur_mediane": float(np.median([r["erreur"] for r in sel])),
                     "acceptation": float(np.mean([r["acceptation"] for r in sel])),
                     "temps_moyen": float(np.mean([r["temps"] for r in sel]))})
    save_csv(summ, OUT / "mcmc_resume.csv")
    for s in summ:
        print("  ", s)


if __name__ == "__main__":
    main()

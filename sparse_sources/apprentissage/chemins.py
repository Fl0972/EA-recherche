"""
Calcul (parallèle, mis en cache) des chemins ML K = 0..K_max pour une base
de tirages, avec pour chaque K les quantités utiles au choix de K :

    chi2      ||r_w||²
    logdetF   log det de la matrice de Fisher (paramètres physiques I, z)
    logprior  log π(θ̂) : z uniforme sur D, I log-uniforme sur [0,1 ; 1000]

Évidence de Laplace (approximation gaussienne de la vraisemblance autour du
maximum, Bishop §4.4 / MacKay ch. 28) :

    log p(m | K) ≈ -χ²/2 + log π(θ̂) + (3K/2) log 2π - ½ log det F + log K!

(K! : les K! permutations des sources donnent la même vraisemblance). Le
BIC est la version grossière de cette formule (log det F ≈ 3K log M).
"""

from __future__ import annotations

import pickle
import sys
from math import lgamma
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from .base import OUT, Fitter, parallel  # noqa: E402
from .simulation import PROJET, Scene  # noqa: E402

I_MIN, I_MAX = 0.1, 1000.0


def fisher_logdet(scene: Scene, s, z, I_phys):
    """log det(Dᵀ D), D = dérivées du modèle blanchi par rapport à (I, z)."""
    K = len(I_phys)
    if K == 0:
        return 0.0
    S0 = scene.setup.with_power(0.0).with_weights(1.0 / s)
    P = S0.phi(z)                 # K, M
    J = S0.grad(z)                # K, M, 2
    M = P.shape[1]
    D = np.concatenate([P.T, (I_phys[:, None, None] * J).transpose(1, 0, 2).reshape(M, 2 * K)], 1)
    sign, ld = np.linalg.slogdet(D.T @ D)
    return float(ld) if sign > 0 else -np.inf


def log_prior(scene: Scene, I_phys):
    """z uniforme sur D, I log-uniforme sur [I_MIN, I_MAX] (indépendants)."""
    K = len(I_phys)
    area = float(np.prod(scene.hi - scene.lo))
    I = np.clip(I_phys, I_MIN, I_MAX)
    return float(-K * np.log(area) - np.sum(np.log(I * np.log(I_MAX / I_MIN))))


def laplace_evidence(row, k):
    """log p(m | K) approchée pour l'élément k du chemin."""
    K = row["K"][k]
    return (-0.5 * row["chi2"][k] + row["logprior"][k] + 1.5 * K * np.log(2 * np.pi)
            - 0.5 * row["logdetF"][k] + lgamma(K + 1))


def compute_path(args):
    """Chemin d'un tirage (fonction de haut niveau : appelée par joblib)."""
    d, scene, k_max = args
    F = Fitter(d["m"], d["s"], scene)
    rows = {"K": [], "chi2": [], "logdetF": [], "logprior": [], "z": [], "I": []}
    for z, I, c in F.path(k_max):
        Ip = F.physical(z, I)
        rows["K"].append(len(Ip))
        rows["chi2"].append(c)
        rows["logdetF"].append(fisher_logdet(scene, d["s"], z, Ip))
        rows["logprior"].append(log_prior(scene, Ip))
        rows["z"].append(z)
        rows["I"].append(Ip)
    return rows


def paths_for(data, name, scene: Scene = PROJET, k_max=7, cache=True):
    """Chemins pour une liste de tirages, mis en cache dans results/<name>.pkl."""
    f = OUT / f"{name}.pkl"
    if cache and f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    res = parallel(compute_path, [(d, scene, k_max) for d in data])
    if cache:
        with open(f, "wb") as fh:
            pickle.dump(res, fh)
    return res

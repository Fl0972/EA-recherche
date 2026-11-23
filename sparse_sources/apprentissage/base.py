"""
Briques communes : ajustement par maximum de vraisemblance (ML) à K fixé,
« chemin » glouton K = 0, 1, ..., K_max, évaluation d'une reconstruction,
base de test fixe et calcul parallèle.

Le chemin ML est la boucle de ``sparse_sources.ml.greedy_ml`` (ajout d'un
Dirac au maximum de corrélation blanchie, puis glissement sans pénalité de
toutes les sources), mais on garde la solution de CHAQUE K au lieu de
s'arrêter : un critère de choix de K (BIC, évidence bayésienne, classifieur
appris...) peut ensuite choisir sur ce chemin.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from sparse_sources.common import (Grid, joint_descent, match_sources,  # noqa: E402
                                   max_certificate, prune_and_merge)
from sparse_sources.experiments import min_separation, success  # noqa: E402

from .simulation import PROJET, Scene  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


class Fitter:
    """Ajustement ML (bruit blanchi) pour un tirage donné : m, s, scène."""

    def __init__(self, m, s, scene: Scene = PROJET):
        self.scene = scene
        self.m = np.asarray(m, float)
        self.s = np.asarray(s, float)
        self.S = scene.setup.with_weights(1.0 / self.s)
        self.mw = self.S.whiten(self.m)
        g = scene.grid
        self.grid = Grid(g.xs, g.ys, g.points, g.step, self.S.phi(g.flat).T)
        self.merge = 1e-3 * float(np.linalg.norm(scene.hi - scene.lo))

    def chi2(self, z, I):
        """χ² = ||m_w - A_w(z, I)||² (I : intensités du noyau normalisé)."""
        r = self.mw - (I @ self.S.phi(z) if len(I) else 0.0)
        return float(r @ r)

    def slide(self, z, I):
        """Maximum de vraisemblance local à K fixé (L-BFGS-B, sans pénalité)."""
        if len(I) == 0:
            return z, I
        I = np.maximum(I, 1e-3 * max(float(I.max()), 1e-12))
        z, I, _ = joint_descent(z, I, self.mw, 0.0, self.S)
        return prune_and_merge(z, I, self.merge)

    def fit_from(self, z0):
        """Départ en z0 (positions), intensités par NNLS, puis glissement."""
        z0 = np.clip(np.asarray(z0, float).reshape(-1, 2), self.scene.lo, self.scene.hi)
        I0, _ = nnls(self.S.phi(z0).T, self.mw, maxiter=50 * len(z0) + 100)
        z, I = self.slide(z0, I0)
        return z, I, self.chi2(z, I)

    def residual(self, z, I):
        """Résidu blanchi."""
        return self.mw - (I @ self.S.phi(z) if len(I) else 0.0)

    def add_one(self, z, I):
        """Ajoute un Dirac au maximum de la corrélation blanchie et réajuste tout."""
        r = self.residual(z, I)
        z_new = max_certificate(r, 1.0, self.grid, self.S, 5)[0]
        return self.fit_from(np.vstack([z, z_new]) if len(z) else z_new[None])

    def path(self, k_max=7):
        """
        Chemin glouton : liste de (z, I, χ²) pour K = 0, ..., k_max
        (K effectif après fusion éventuelle de Diracs confondus).
        """
        z, I = np.zeros((0, 2)), np.zeros(0)
        out = [(z, I, self.chi2(z, I))]
        for _ in range(k_max):
            z, I, c = self.add_one(z, I)
            out.append((z, I, c))
        return out

    def physical(self, z, I):
        """Intensités physiques."""
        return self.S.physical_intensities(z, I)


def evaluate(d, z, I):
    """Réussite (même critère que les études précédentes), erreurs."""
    res = match_sources(d["z"], d["I"], z, I)
    return {"n_est": res["n_est"], "exact": res["exact"], "err_pos": res["err_pos"],
            "err_int": res["err_int"], "reussite": success(res, d["z"]),
            "sep_min": min_separation(d["z"])}


def test_set(scene: Scene = PROJET, ks=(1, 2, 3, 4, 5), sigmas=(1e-4, 1e-3, 3e-3),
             n_per=40, seed=2026):
    """Base de test fixe (graine différente de celles d'entraînement)."""
    rng = np.random.default_rng(seed)
    return [scene.draw(k, s, rng) for k in ks for s in sigmas for _ in range(n_per)]


def parallel(fn, items, n_jobs=None):
    """map parallèle (joblib, processus)."""
    from joblib import Parallel, delayed
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 1)
    return Parallel(n_jobs=n_jobs, batch_size=4)(delayed(fn)(x) for x in items)

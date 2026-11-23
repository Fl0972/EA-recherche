"""
Tests de la méthode Orano (sparse_sources/orano.py) : gradient et jacobien
contre différences finies, décroissance de la perte, cas test à 2 sources.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from sparse_sources.experiments_orano import (CAPTEURS, HI, LO, evaluer, mesures,
                                              tirer_situation, un_tirage)
from sparse_sources.orano import Probleme, descente_projetee


@pytest.fixture
def probleme():
    """3 sources, mesures légèrement perturbées pour que le résidu ne soit pas nul."""
    rng = np.random.default_rng(0)
    p, I = tirer_situation(3, rng)
    y, _ = mesures(p, I, 0.0, False, rng)
    return Probleme(CAPTEURS, LO, HI, 1.3 * y + 0.01), rng


def test_gradient_differences_finies(probleme):
    """∇L analytique (paramètres normalisés) = différences finies centrées, écart relatif < 1e-6."""
    pb, rng = probleme
    u, a = rng.random((3, 2)), rng.random(3) / 3
    _, gu, ga = pb.perte_gradient(u, a)
    g = np.r_[gu.ravel(), ga]
    th = np.r_[u.ravel(), a]

    def f(t):
        return pb.perte(t[:6].reshape(3, 2), t[6:])

    h = 1e-6
    fd = np.array([(f(th + h * e) - f(th - h * e)) / (2 * h) for e in np.eye(9)])
    assert np.linalg.norm(fd - g) / np.linalg.norm(g) < 1e-6


def test_jacobien_differences_finies(probleme):
    """Jacobien du résidu (Gauss-Newton) = différences finies centrées."""
    pb, rng = probleme
    u, a = rng.random((3, 2)), rng.random(3) / 3
    J = pb.jacobien_residu(u, a)
    th = np.r_[u.ravel(), a]

    def r(t):
        return (pb.modele(*pb.physique(t[:6].reshape(3, 2), t[6:])) - pb.y) / np.sqrt(pb.ny2)

    h = 1e-6
    Jfd = np.array([(r(th + h * e) - r(th - h * e)) / (2 * h) for e in np.eye(9)]).T
    assert np.linalg.norm(Jfd - J) / np.linalg.norm(J) < 1e-6


def test_perte_decroissante_et_projection(probleme):
    """Armijo : la perte ne remonte jamais ; les positions restent dans D, intensités ≥ 0."""
    pb, rng = probleme
    u0, a0 = rng.random((3, 2)), np.full(3, 1 / 3)
    pertes = []
    for n in [1, 5, 20, 100, 500]:
        r = descente_projetee(pb, u0, a0, iter_max=n)
        pertes.append(r["perte"])
        assert np.all(r["p"] >= LO) and np.all(r["p"] <= HI) and np.all(r["I"] >= 0)
    assert all(b <= a + 1e-15 for a, b in zip(pertes, pertes[1:]))


def test_cas_deux_sources_rapport_10():
    """Cas test d'Orano, 2 sources, rapport 10, sans bruit : descente et Gauss-Newton réussissent."""
    args = SimpleNamespace(sep_min=0.5, sigma=0.0, centrale=False, iter=20000, sans_ml=True)
    (p, I, _), out = un_tirage(2, 10.0, 1, args)
    for m in ["descente (Orano)", "Gauss-Newton"]:
        ok, n_ok, err, _ = evaluer(p, out[m]["p"], out[m]["I"])
        assert ok, f"{m} : {n_ok}/2 sources à 0,1, erreur max {err:.3g}"

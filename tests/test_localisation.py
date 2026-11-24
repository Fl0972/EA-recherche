"""Tests rapides (python -m pytest) : noyau, méthodes sans bruit, métrique, reproductibilité."""

import numpy as np
import pytest

from localisation import lancer, metrique, situation
from localisation.noyau import InverseSquareKernel, WeightedKernel
from localisation.probleme import capteurs


def test_derivees_du_noyau():
    """Gradient et hessienne analytiques = différences finies (noyau brut et normalisé)."""
    X = capteurs("cercle")
    z = np.array([3.1, 7.4])
    for k in (InverseSquareKernel(X), WeightedKernel(InverseSquareKernel(X), 1.0)):
        h = 1e-5
        e = np.eye(2) * h
        g_fd = np.stack([(k.phi(z + e[a]) - k.phi(z - e[a])) / (2 * h) for a in range(2)], -1)
        H_fd = np.stack([(k.grad(z + e[a]) - k.grad(z - e[a])) / (2 * h) for a in range(2)], -1)
        assert np.max(np.abs(k.grad(z) - g_fd)) < 1e-6 * np.max(np.abs(k.grad(z)))
        assert np.max(np.abs(k.hess(z) - H_fd)) < 1e-5 * np.max(np.abs(k.hess(z)))


def test_situation_reproductible():
    """Même (N, tirage) : mêmes sources, quels que soient σ, I_c et les capteurs."""
    a = situation(4, 7, 1e-3, 50, "cercle")
    b = situation(4, 7, 1e-2, 0, "grille")
    assert np.allclose(a.z, b.z) and np.allclose(a.I, b.I) and np.isclose(a.I.sum(), 100)
    assert np.allclose(a.m, situation(4, 7, 1e-3, 50, "cercle").m)


@pytest.mark.parametrize("methode", ["prony", "ml_bic", "blasso", "gauss_newton"])
def test_methodes_sans_bruit(methode):
    """Sans bruit ni source centrale, deux sources bien séparées sont retrouvées."""
    sit = situation(2, 0, 0.0, 0.0, "cercle", z=[[2.0, 5.0], [8.0, 15.0]], I=[60.0, 40.0])
    out = lancer(methode, sit)
    e = metrique.erreur(sit.z, sit.I, out["z"], out["I"])
    assert e["detection"] == 1.0 and e["d_max"] < 1e-2


def test_metrique_fusion_et_detection():
    z, I = np.array([[1.0, 1.0], [5.0, 5.0]]), np.array([50.0, 50.0])
    # la première source est coupée en deux morceaux à 0,02 l'un de l'autre : pas d'erreur
    ze = np.array([[0.99, 1.0], [1.01, 1.0], [5.0, 5.0]])
    Ie = np.array([25.0, 25.0, 50.0])
    e = metrique.erreur(z, I, ze, Ie)
    assert e["detection"] == 1.0 and e["E"] < 1e-12 and e["masse_parasite"] == 0
    # une source manquée : d = 5, r = 1, donc E = (0,75·5 + 0,25·1) / 2 = 2
    e = metrique.erreur(z, I, ze[2:], Ie[2:])
    assert e["detection"] == 0.5 and np.isclose(e["E"], 2.0)
    # une source à 0,3 : manquée à 0,1, détectée à 0,5
    e1 = metrique.erreur(z, I, [[1.3, 1.0], [5.0, 5.0]], I, seuil=0.1)
    e5 = metrique.erreur(z, I, [[1.3, 1.0], [5.0, 5.0]], I, seuil=0.5)
    assert e1["detection"] == 0.5 and e5["detection"] == 1.0

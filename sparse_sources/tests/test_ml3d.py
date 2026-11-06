"""Maximum de vraisemblance + BIC en dimension quelconque (ml3d)."""
import numpy as np
import pytest

from particle_sources.duval_peyre import experiments_3d as ex3
from particle_sources.duval_peyre.kernel import InverseSquareKernel
from particle_sources.model import Box
from sparse_sources import cramer_rao, greedy_ml, make_setup
from sparse_sources.ml3d import (box_grid, cramer_rao_nd, greedy_ml_nd, match_nd,
                                 success_nd)

CUBE = Box.from_bounds([[0.0, 10.0]] * 3)
Z3 = np.array([[2.0, 3.0, 4.0], [7.0, 6.0, 3.0], [5.0, 8.0, 8.0]])
I3 = np.array([30.0, 50.0, 20.0])


def data(sensors, z, I, sigma, seed=0, central=0.0, box=CUBE):
    """Mesures du scénario (source centrale facultative, retirée) et écart type du bruit."""
    k = InverseSquareKernel(sensors)
    c = central * k.phi(np.asarray(box.center, float)[None])[0]
    y = k.phi(z).T @ I + c
    s = sigma * np.abs(y)
    return y + s * np.random.default_rng(seed).standard_normal(len(y)) - c, s


def test_box_grid():
    """Grille : bords compris, pas respecté."""
    P = box_grid(CUBE, 2.5)
    assert P.shape == (125, 3)
    np.testing.assert_allclose(P.min(0), 0.0)
    np.testing.assert_allclose(P.max(0), 10.0)


@pytest.mark.parametrize("layout", ["sphere", "deux_cercles"])
def test_recovers_three_sources_3d(layout):
    """3 sources en 3D, σ = 1e-4 : K exact, erreur de l'ordre de la borne de Cramér-Rao."""
    S = ex3.layout(layout, CUBE, 160, 1.5)
    m, s = data(S, Z3, I3, 1e-4, seed=1)
    r = greedy_ml_nd(m, s, S, CUBE)
    res = match_nd(Z3, I3, r["z"], r["I"])
    crb = cramer_rao_nd(Z3, I3, s, S)["pos_std"].max()
    assert r["K"] == 3 and success_nd(res, Z3)
    assert res["err_pos"] < 5 * crb
    assert r["chi2"] < 160 + 4 * np.sqrt(2 * 160)       # compatible avec le bruit


def test_with_central_source_3d():
    """Source centrale 5000 retirée (son bruit reste) : 2 sources retrouvées à σ = 1e-4."""
    S = ex3.layout("sphere", CUBE, 160, 1.5)
    z, I = Z3[:2], np.array([40.0, 60.0])
    m, s = data(S, z, I, 1e-4, seed=2, central=5000.0)
    r = greedy_ml_nd(m, s, S, CUBE)
    assert r["K"] == 2 and success_nd(match_nd(z, I, r["z"], r["I"]), z)


def test_noise_only_gives_no_source():
    """Bruit seul : le BIC ne retient (presque) rien."""
    S = ex3.layout("sphere", CUBE, 160, 1.5)
    s = np.full(160, 1e-3)
    m = s * np.random.default_rng(3).standard_normal(160)
    assert greedy_ml_nd(m, s, S, CUBE)["K"] <= 1


def test_imposed_k():
    """κ = 0 et k_max = K : exactement K sources."""
    S = ex3.layout("trois_cercles", CUBE, 120, 1.5)
    m, s = data(S, Z3, I3, 1e-3, seed=4)
    r = greedy_ml_nd(m, s, S, CUBE, k_max=3, criterion=0.0, backward=False)
    assert r["K"] == 3


def test_single_ring_is_not_identifiable():
    """Un seul cercle en 3D : la borne de Cramér-Rao est infinie (symétrie haut / bas)."""
    S = ex3.layout("un_cercle", CUBE, 160, 1.5)
    s = np.full(160, 1e-3)
    assert np.all(np.isinf(cramer_rao_nd(Z3, I3, s, S)["pos_std"]))


def test_matches_2d_implementation():
    """En 2D, ml3d donne la même borne que ml.cramer_rao et la même solution que greedy_ml."""
    S2 = make_setup(h=0.0, normalize=True)
    z = np.array([[3.0, 6.0], [7.0, 15.0]])
    I = np.array([40.0, 60.0])
    m, s = data(S2.sensors, z, I, 1e-4, seed=5, box=S2.domain)
    np.testing.assert_allclose(cramer_rao_nd(z, I, s, S2.sensors)["pos_std"],
                               cramer_rao(z, I, s, S2)["pos_std"], rtol=1e-6)
    a = greedy_ml_nd(m, s, S2.sensors, S2.domain)
    b = greedy_ml(m, s, setup=S2)
    assert a["K"] == b["K"] == 2
    assert match_nd(b["z"], b["I"], a["z"], a["I"])["err_pos"] < 1e-4
    assert abs(a["chi2"] - b["chi2"]) < 1e-3 * b["chi2"]


def test_cramer_rao_scales_with_noise():
    """La borne est proportionnelle à l'écart type du bruit."""
    S = ex3.layout("sphere", CUBE, 100, 1.5)
    a = cramer_rao_nd(Z3, I3, np.full(100, 1e-3), S)["pos_std"]
    b = cramer_rao_nd(Z3, I3, np.full(100, 1e-2), S)["pos_std"]
    np.testing.assert_allclose(b, 10 * a, rtol=1e-8)


def test_match_nd():
    """Appariement : distance max et erreur relative d'intensité."""
    res = match_nd(Z3, I3, Z3[::-1] + 0.1, I3[::-1] * 1.1)
    assert res["exact"] and res["err_pos"] == pytest.approx(0.1 * np.sqrt(3))
    assert res["err_int"] == pytest.approx(0.1)

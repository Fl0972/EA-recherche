"""
Tests du blanchiment (Setup.weights), du maximum de vraisemblance glouton
(greedy_ml) et de la borne de Cramér-Rao, dans le régime du projet : bruit
relatif dominé par la source centrale (≈ 40 par capteur, retirée des mesures).
"""

import numpy as np
import pytest

from sparse_sources import (build_grid, cramer_rao, forward, greedy_ml, match_sources,
                            relative_noise_std)
from sparse_sources.kernel import check_derivatives

CENTRAL = 40.0          # contribution de la source centrale (5000 / 125) à chaque capteur


@pytest.fixture(scope="module")
def grid(setup_normalized):
    """Grille de pas 0,25 pour le noyau normalisé."""
    return build_grid(step=0.25, setup=setup_normalized)


def _data(S, z, I, sigma, seed=0):
    """Mesures du projet : y = (A(s) + central)(1 + σ ε), puis retrait de la source centrale."""
    clean = forward(z, I, S.with_power(0.0))
    y = (clean + CENTRAL) * (1 + sigma * np.random.default_rng(seed).standard_normal(S.M))
    return y - CENTRAL, relative_noise_std(y, sigma)


def test_whitened_derivatives(setup_normalized):
    """Dérivées du noyau blanchi puis normalisé : accord avec les différences finies."""
    w = np.linspace(0.3, 3.0, setup_normalized.M)
    S = setup_normalized.with_weights(w)
    for z in [[3.0, 7.0], [5.0, 10.0], [9.0, 2.0]]:
        err = check_derivatives(z, S)
        assert err["grad"] < 1e-6 and err["hess"] < 1e-5


def test_weights_api(setup_plane, setup_normalized):
    """Validation des poids, whiten, retour sans poids, invariance de ψ par poids constants."""
    S = setup_plane
    with pytest.raises(ValueError):
        S.with_weights(np.zeros(S.M))
    with pytest.raises(ValueError):
        S.with_weights(np.ones(S.M + 1))
    w = np.full(S.M, 3.0)
    z = np.array([[2.0, 5.0], [6.0, 12.0]])
    assert np.allclose(S.with_weights(w).phi(z), 3 * S.phi(z))
    assert np.allclose(S.with_weights(w).whiten(np.ones(S.M)), 3.0)
    assert np.allclose(S.with_weights(w).with_weights(None).phi(z), S.phi(z))
    # ψ = φ_w / ||φ_w|| ne dépend pas d'un poids constant ; les intensités physiques non plus
    N, Nw = setup_normalized, setup_normalized.with_weights(w)
    assert np.allclose(Nw.phi(z), N.phi(z))
    I = np.array([1.0, 2.0])
    assert np.allclose(Nw.physical_intensities(z, Nw.normalized_intensities(z, I)), I)


@pytest.mark.parametrize("sigma", [1e-4, 1e-3])
def test_greedy_ml_two_sources(setup_normalized, grid, sigma):
    """Deux sources, bruit de la source centrale : bon K, erreur de l'ordre de la CRB."""
    S = setup_normalized
    z, I = np.array([[3.0, 6.0], [7.0, 14.0]]), np.array([30.0, 20.0])
    m, s = _data(S, z, I, sigma)
    out = greedy_ml(m, s, setup=S, grid=grid)
    res = match_sources(z, I, out["z"], out["I"])
    crb = cramer_rao(z, I, s, S)["pos_std"].max()
    assert res["exact"]
    assert res["err_pos"] < 5 * crb
    assert out["chi2"] < S.M + 5 * np.sqrt(2 * S.M)      # résidu compatible avec le bruit


def test_greedy_ml_three_sources_low_noise(setup_normalized, grid):
    """Trois sources tirées, σ = 1e-5 : nombre exact, positions précises."""
    S = setup_normalized
    z = np.array([[2.0, 4.0], [8.0, 9.0], [4.0, 16.0]])
    I = np.array([40.0, 25.0, 35.0])
    m, s = _data(S, z, I, 1e-5, seed=3)
    out = greedy_ml(m, s, setup=S, grid=grid)
    res = match_sources(z, I, out["z"], out["I"])
    assert res["exact"] and res["err_pos"] < 0.02 and res["err_int"] < 0.02


def test_greedy_ml_options(setup_normalized, grid):
    """Critères aic / numérique, candidats multiples, départ imposé (hors de D ignoré)."""
    S = setup_normalized
    z, I = np.array([[3.0, 6.0], [7.0, 14.0]]), np.array([30.0, 20.0])
    m, s = _data(S, z, I, 1e-4)
    for kw in [dict(criterion="aic"), dict(criterion=10.0), dict(n_candidates=3),
               dict(z_init=np.array([[3.2, 6.1], [7.1, 13.8], [50.0, 50.0]]),
                    I_init=np.array([25.0, 25.0, 1.0]))]:
        out = greedy_ml(m, s, setup=S, grid=grid, **kw)
        assert match_sources(z, I, out["z"], out["I"])["exact"], kw


def test_greedy_ml_pure_noise(setup_normalized, grid):
    """Bruit seul : BIC ne retient (presque) aucune source."""
    S = setup_normalized
    s = np.full(S.M, 0.04)
    m = s * np.random.default_rng(5).standard_normal(S.M)
    assert greedy_ml(m, s, setup=S, grid=grid)["K"] <= 1


def test_cramer_rao_scaling_and_efficiency(setup_normalized, grid):
    """CRB ∝ s ; une source : l'erreur rms du ML est proche de la CRB (efficacité)."""
    S = setup_normalized
    z, I = np.array([[3.0, 7.0]]), np.array([50.0])
    s = np.full(S.M, 0.04)
    c1 = cramer_rao(z, I, s, S)["pos_std"][0]
    assert cramer_rao(z, I, 10 * s, S)["pos_std"][0] == pytest.approx(10 * c1)
    assert cramer_rao(z, I, s, S.with_weights(np.full(S.M, 7.0)))["pos_std"][0] == \
        pytest.approx(c1)                                # poids/pondération du setup ignorés
    clean = forward(z, I, S.with_power(0.0))
    errs = []
    for seed in range(30):
        m = clean + s * np.random.default_rng(100 + seed).standard_normal(S.M)
        out = greedy_ml(m, s, setup=S, grid=grid, k_max=1)
        errs.append(np.linalg.norm(out["z"][0] - z[0]))
    rms = np.sqrt(np.mean(np.square(errs)))
    assert 0.6 * c1 < rms < 1.5 * c1

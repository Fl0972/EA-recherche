"""
Tests de sparse_sources (pytest sparse_sources/tests).

Les configurations « favorables » utilisent soit le noyau normalisé
(h = 0, géométrie du projet), soit des capteurs hors du plan (h = 20) : avec
le noyau brut 1/r² et des capteurs dans le plan, la condition de source non
dégénérée échoue déjà pour une seule source (voir test dédié et README).
"""

import numpy as np
import pytest

from particle_sources.duval_peyre import diagnose
from sparse_sources import (build_grid, check_hypotheses, debias, forward,
                            hypotheses_after_fit, lasso_pipeline, match_sources,
                            merge_peaks, refine,
                            sliding_frank_wolfe, solve_lasso)
from sparse_sources.common import certificate, certificate_grad, max_certificate
from sparse_sources.hypotheses import eta_V, eta_V_grad, precertificate
from sparse_sources.kernel import check_derivatives
from sparse_sources.lasso import lasso_objective

POINTS = [[3.0, 7.0], [5.1, 10.0], [9.7, 19.4], [0.2, 0.3]]


# ---- Dérivées du noyau ----------------------------------------------------------------
@pytest.mark.parametrize("name", ["setup_plane", "setup_normalized", "setup_high"])
@pytest.mark.parametrize("z", POINTS)
def test_derivatives_vs_finite_differences(name, z, request):
    """∇φ et ∇²φ analytiques = différences finies centrées (noyau brut et normalisé)."""
    err = check_derivatives(z, request.getfixturevalue(name))
    assert err["grad"] < 1e-7
    assert err["hess"] < 1e-6


def test_kernel_is_project_kernel(setup_plane):
    """φ est bien G = 1/r² du projet (``intensity_at``)."""
    z, I = np.array([[2.0, 3.0], [6.0, 15.0]]), np.array([1.5, 4.0])
    direct = I @ (1.0 / np.sum((setup_plane.sensors[None] - z[:, None]) ** 2, axis=-1))
    assert np.allclose(forward(z, I, setup_plane), direct)


def test_certificate_gradient(setup_plane):
    """∇η du certificat = différences finies."""
    rng = np.random.default_rng(0)
    r = rng.standard_normal(setup_plane.M)
    z, h = np.array([4.0, 9.0]), 1e-6
    fd = [(certificate(r, 0.3, z + e, setup_plane) - certificate(r, 0.3, z - e, setup_plane))
          / (2 * h) for e in np.eye(2) * h]
    assert np.allclose(certificate_grad(r, 0.3, z, setup_plane), fd, rtol=1e-6)


# ---- Pré-certificat ---------------------------------------------------------------------
@pytest.mark.parametrize("name", ["setup_plane", "setup_normalized", "setup_high"])
def test_precertificate_interpolates(name, request):
    """η_V(z_k) = 1 et ∇η_V(z_k) = 0 à la précision numérique."""
    S = request.getfixturevalue(name)
    z = np.array([[3.0, 5.0], [7.0, 15.0], [5.0, 10.0]])
    p, _ = precertificate(z, S)
    assert np.allclose(eta_V(p, z, S), 1.0, atol=1e-10)
    # Échelle naturelle de ∇η_V : Σ_i |p_i| ||∇G(x_i, z_k)||
    scale = np.max(np.abs(p) @ np.linalg.norm(S.grad(z), axis=-1).T)
    assert np.max(np.abs(eta_V_grad(p, z, S))) < 1e-10 * scale


# ---- Hypothèses ---------------------------------------------------------------------
@pytest.mark.parametrize("name", ["setup_normalized", "setup_high"])
def test_single_well_seen_source_is_nondegenerate(name, request):
    """Une source au centre, bien vue : H1, H2, H3 vraies."""
    res = check_hypotheses([[5.1, 10.0]], setup=request.getfixturevalue(name), verbose=False)
    assert res["H1"] and res["H2"] and res["H3"] and res["non_degenerate"]
    assert res["interior"]
    assert np.all(res["eigenvalues"] < 0)


def test_raw_kernel_in_plane_fails(setup_plane):
    """
    Noyau brut, capteurs dans le plan : G(x_i, .) est sous-harmonique sur D
    (ΔG = 4 (r² - h²) / (r² + h²)³ > 0), η_V n'a pas de maximum strict
    intérieur : H3 échoue (et H2, η_V > 1 près des capteurs).
    """
    res = check_hypotheses([[5.1, 10.0]], setup=setup_plane, verbose=False)
    assert res["H1"] and not res["H3"] and not res["H2"]


@pytest.mark.parametrize("name,center,d", [
    ("setup_normalized", [3.0, 6.0], 0.05),
    ("setup_high", [7.0, 16.0], 0.2),
    ("setup_high", [5.0, 10.0], 0.05),
])
def test_two_very_close_sources_fail(name, center, d, request):
    """
    Deux sources très proches : H2 ou H3 échoue. Remarque : pour d ≈ 0.2 avec
    le noyau normalisé, la condition tient encore numériquement (marge
    relative ≈ 0.6) mais cond(Γ) ~ 1e6 : constantes du théorème minuscules.
    """
    z = [[center[0], center[1] - d / 2], [center[0], center[1] + d / 2]]
    res = check_hypotheses(z, setup=request.getfixturevalue(name), verbose=False)
    assert not (res["H2"] and res["H3"])
    assert not res["non_degenerate"]


def test_source_on_boundary_flagged(setup_normalized):
    """Une source sur le bord de D est signalée (interior = False)."""
    res = check_hypotheses([[0.0, 10.0]], setup=setup_normalized, verbose=False)
    assert not res["interior"]


@pytest.mark.parametrize("name,z", [
    ("setup_plane", [[5.0, 7.0], [5.0, 13.0]]),
    ("setup_normalized", [[5.0, 7.0], [5.0, 13.0]]),
    ("setup_normalized", [[2.0, 3.0], [8.0, 4.0], [5.0, 16.0]]),
    ("setup_high", [[5.1, 10.0]]),
])
def test_agrees_with_project_diagnose(name, z, request):
    """Même verdict et même marge que ``particle_sources.duval_peyre.diagnose``."""
    S = request.getfixturevalue(name)
    res = check_hypotheses(z, setup=S, verbose=False)
    ref = diagnose(S.kernel, z, S.domain)
    assert (res["H1"], res["H2"], res["H3"]) == (ref.h1, ref.h2, ref.h3)
    assert res["margin"] == pytest.approx(ref.margin, abs=1e-5)


def test_verbose_plot_and_after_fit(setup_normalized, tmp_path, capsys):
    """Le mode verbeux affiche un résumé et enregistre la carte ; hypotheses_after_fit."""
    path = tmp_path / "eta.png"
    res = check_hypotheses([[5.0, 7.0], [5.0, 13.0]], lam=1e-3, sigma=1e-4,
                           setup=setup_normalized, save_path=path)
    assert path.exists() and "VÉRIFIÉE" in capsys.readouterr().out
    assert np.isfinite(res["noise_over_lambda"])
    res2 = hypotheses_after_fit([[5.0, 7.0], [5.0, 13.0], [1.0, 1.0]], [1.0, 2.0, 0.0],
                                setup=setup_normalized, verbose=False)
    assert res2["K"] == 2 and res2["reliable"]


# ---- LASSO ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def offgrid(setup_normalized):
    """Une source hors grille (milieu de deux nœuds), sans bruit, grille de pas 0.2."""
    S = setup_normalized
    grid = build_grid(step=0.2, setup=S)
    m = forward([[5.1, 10.0]], [10.0], S.with_normalization(False))
    lam = 1e-2 * np.abs(grid.G.T @ m).max()
    return S, grid, m, lam


def test_lasso_offgrid_two_peaks_then_one(offgrid):
    """Avant regroupement : deux nœuds voisins actifs ; après : une seule source."""
    S, grid, m, lam = offgrid
    c, _, _ = solve_lasso(m, lam, grid, "fista", S, tol=1e-6, max_iter=50_000)
    active = np.flatnonzero(c > 1e-3 * c.max())
    assert len(active) == 2
    assert np.linalg.norm(grid.flat[active[0]] - grid.flat[active[1]]) <= 0.2 + 1e-9
    z, I, _ = merge_peaks(c, grid, rel_threshold=1e-3)
    assert len(I) == 1
    assert np.allclose(z[0], [5.1, 10.0], atol=0.02)


def test_lasso_solvers_agree(offgrid):
    """FISTA, sklearn (λ converti : α = λ / M) et ensemble actif : même objectif."""
    S, grid, m, lam = offgrid
    objs = {}
    for method in ["active_set", "fista", "sklearn"]:
        c, _, info = solve_lasso(m, lam, grid, method, S)
        assert np.all(c >= 0)
        objs[method] = lasso_objective(grid.G, m, lam, c)
    ref = objs["active_set"]
    # sklearn s'arrête sur max_iter (problème très mal conditionné) : 1e-3
    for method, rel in [("fista", 1e-4), ("sklearn", 1e-3)]:
        assert objs[method] == pytest.approx(ref, rel=rel), method
        assert objs[method] >= ref - 1e-12           # la référence est l'optimum


def test_refine_and_debias_recover_offgrid_source(offgrid):
    """Regroupement + raffinement + débiaisage : position et intensité exactes."""
    S, grid, m, lam = offgrid
    out = lasso_pipeline(m, 1e-4 * lam, grid=grid, setup=S, method="active_set")
    I_true = S.normalized_intensities([[5.1, 10.0]], [10.0])
    assert len(out["I"]) == 1
    assert np.allclose(out["z"][0], [5.1, 10.0], atol=1e-3)
    assert out["I"][0] == pytest.approx(I_true[0], rel=1e-3)
    assert out["certified"]


def test_debias_exact_without_noise(setup_high):
    """Positions exactes : le débiaisage NNLS redonne les intensités exactes."""
    z, I = np.array([[3.0, 5.0], [7.0, 15.0]]), np.array([4.0, 9.0])
    m = forward(z, I, setup_high)
    z2, I2 = debias(z, 0.5 * I, m, setup_high)
    assert np.allclose(I2, I, rtol=1e-8)


# ---- Frank-Wolfe glissant --------------------------------------------------------------
@pytest.mark.parametrize("z_true,I_true", [
    ([[5.0, 7.0], [5.0, 13.0]], [10.0, 20.0]),
    ([[3.0, 5.0], [7.0, 15.0]], [15.0, 8.0]),
])
def test_sfw_noise_free_recovery(setup_normalized, z_true, I_true):
    """Sans bruit : sources retrouvées à 1e-4 près, arrêt avec max η ≤ 1."""
    S = setup_normalized
    z_true, I_true = np.array(z_true), np.array(I_true)
    m = forward(z_true, I_true, S.with_normalization(False))
    grid = build_grid(step=0.2, setup=S)
    # λ petit mais pas trop : en deçà de ~1e-6 λ_max, η = φ·r/λ amplifie
    # l'erreur d'optimisation de L-BFGS et le critère η ≤ 1 + 1e-6 n'est plus
    # atteignable numériquement (voir README, limites).
    # Le biais de position du BLASSO est O(λ) (≈ 15 λ/λ_max ici) : λ = 3e-6 λ_max.
    lam = 3e-6 * np.abs(grid.G.T @ m).max()
    out = sliding_frank_wolfe(m, lam, setup=S, grid=grid, tol=1e-6)
    assert out["converged"] and out["eta_max"] <= 1 + 1e-6
    z, I = out["z_debiased"], S.physical_intensities(out["z_debiased"], out["I_debiased"])
    assert len(I) == len(I_true)
    order = [int(np.argmin(np.linalg.norm(z - zt, axis=1))) for zt in z_true]
    assert np.max(np.linalg.norm(z[order] - z_true, axis=1)) < 1e-4
    assert np.allclose(I[order], I_true, rtol=1e-4)
    # Historique : K croissant de 0 à 2, max η décroissant jusqu'à ≤ 1
    hist = out["history"]
    assert hist[0]["K"] == 0 and hist[-1]["eta_max"] <= 1 + 1e-6


def test_sfw_from_lasso_initialization(offgrid):
    """Initialisation par la sortie du LASSO : converge immédiatement (1 itération)."""
    S, grid, m, lam = offgrid
    lasso = lasso_pipeline(m, lam, grid=grid, setup=S, method="active_set")
    out = sliding_frank_wolfe(m, lam, z_init=lasso["z_lambda"], I_init=lasso["I_lambda"],
                              setup=S, grid=grid)
    assert out["converged"] and out["n_iter"] <= 2


def test_max_certificate_finds_boundary_max(setup_plane):
    """Le maximum de η peut être sur le bord de D (proche des capteurs)."""
    m = forward([[5.0, 10.0]], [10.0], setup_plane)
    z_star, val = max_certificate(m, 1.0, 0.25, setup_plane)
    lo, hi = setup_plane.domain.lo, setup_plane.domain.hi
    assert np.any(np.isclose(z_star, lo) | np.isclose(z_star, hi))
    grid = build_grid(step=0.25, setup=setup_plane)
    assert val >= (m @ grid.G).max() - 1e-12


def test_refine_keeps_bounds(setup_high):
    """refine reste dans D et garde des intensités ≥ 0."""
    m = forward([[0.3, 19.5]], [5.0], setup_high)
    z, I, _ = refine([[0.5, 19.0]], [3.0], m, 1e-6, setup_high)
    lo, hi = setup_high.domain.lo, setup_high.domain.hi
    assert np.all(z >= lo - 1e-12) and np.all(z <= hi + 1e-12) and np.all(I >= 0)


# ---- Données bruitées, plusieurs niveaux de bruit -----------------------------------------
NOISY_Z = np.array([[3.0, 5.0], [7.0, 15.0]])
NOISY_I = np.array([15.0, 8.0])
LEVELS = [1e-4, 1e-3, 1e-2]


def _noisy(S, level, seed=0):
    """m = A(s*) + σ ε, σ = level × rms(A(s*)) (noyau physique)."""
    m0 = forward(NOISY_Z, NOISY_I, S.with_normalization(False))
    sigma = level * float(np.sqrt(np.mean(m0**2)))
    rng = np.random.default_rng(seed)
    return m0 + sigma * rng.standard_normal(m0.shape), sigma


@pytest.fixture(scope="module")
def grid_normalized(setup_normalized):
    """Grille de pas 0.25 pour le noyau normalisé."""
    return build_grid(step=0.25, setup=setup_normalized)


@pytest.mark.parametrize("level", LEVELS)
def test_sfw_noisy_recovery(setup_normalized, grid_normalized, level):
    """
    Bruit 1e-4 à 1e-2 (relatif au rms), λ = 2 σ √M, configuration non
    dégénérée : bon nombre de sources, certificat ≤ 1, erreurs de position et
    d'intensité de l'ordre du bruit, résidu de l'ordre de M σ².
    """
    S = setup_normalized
    m, sigma = _noisy(S, level)
    lam = 2 * sigma * np.sqrt(S.M)
    out = sliding_frank_wolfe(m, lam, setup=S, grid=grid_normalized, tol=1e-5, sigma=sigma)
    assert out["converged"]
    I = S.physical_intensities(out["z_debiased"], out["I_debiased"])
    res = match_sources(NOISY_Z, NOISY_I, out["z_debiased"], I)
    assert res["exact"]
    assert res["err_pos"] < 300 * level        # ≈ 30 σ_rel observé, marge ×10
    assert res["err_int"] < 100 * level
    assert 0.3 < out["diagnostics"]["ratio"] < 3


@pytest.mark.parametrize("level", LEVELS)
def test_lasso_noisy_recovery(setup_normalized, grid_normalized, level):
    """
    Même scénario pour le pipeline LASSO + raffinement : bon K et erreur de
    l'ordre du bruit. Le certificat n'est pas toujours ≤ 1 : un petit pic
    parasite du LASSO peut rester bloqué par le raffinement local (intensité
    ≈ 0 donc gradient en position ≈ 0) ; il est retiré au débiaisage, mais la
    solution pénalisée n'est alors pas certifiée (η_max ≈ 1.003 à σ = 1e-4).
    """
    S = setup_normalized
    m, sigma = _noisy(S, level, seed=1)
    lam = 2 * sigma * np.sqrt(S.M)
    out = lasso_pipeline(m, lam, grid=grid_normalized, setup=S, sigma=sigma,
                         method="active_set")
    I = S.physical_intensities(out["z"], out["I"])
    res = match_sources(NOISY_Z, NOISY_I, out["z"], I)
    assert res["exact"]
    assert res["err_pos"] < 300 * level
    assert out["eta_max"] < 1.01


def test_error_linear_in_noise(setup_normalized, grid_normalized):
    """
    Régime de petit bruit (condition vérifiée) : erreur de position ∝ σ.
    Avec λ = c σ √M fixé en c, l'erreur moyenne sur 4 tirages divisée par σ
    varie peu quand σ est divisé par 100.
    """
    S = setup_normalized
    ratios = []
    for level in [1e-5, 1e-4, 1e-3]:
        errs = []
        for seed in range(4):
            m, sigma = _noisy(S, level, seed)
            out = sliding_frank_wolfe(m, 2 * sigma * np.sqrt(S.M), setup=S,
                                      grid=grid_normalized, tol=1e-5)
            I = S.physical_intensities(out["z_debiased"], out["I_debiased"])
            res = match_sources(NOISY_Z, NOISY_I, out["z_debiased"], I)
            assert res["exact"]
            errs.append(res["err_pos"])
        ratios.append(np.mean(errs) / level)
    assert max(ratios) / min(ratios) < 3


def test_high_noise_degrades(setup_normalized, grid_normalized):
    """Bruit fort (σ = 0.3 × rms) : l'erreur est bien plus grande qu'à 1e-3."""
    S = setup_normalized
    errs = {}
    for level in [1e-3, 0.3]:
        m, sigma = _noisy(S, level, seed=2)
        out = sliding_frank_wolfe(m, 2 * sigma * np.sqrt(S.M), setup=S, grid=grid_normalized,
                                  tol=1e-5)
        I = S.physical_intensities(out["z_debiased"], out["I_debiased"])
        res = match_sources(NOISY_Z, NOISY_I, out["z_debiased"], I)
        errs[level] = res["err_pos"] if res["exact"] else np.inf
    assert errs[0.3] > 10 * errs[1e-3]


@pytest.mark.parametrize("level", LEVELS)
def test_hypotheses_report_noise_regime(setup_normalized, level):
    """check_hypotheses avec λ et σ : ||w|| / λ = 1 / c pour λ = c σ √M."""
    S = setup_normalized
    m, sigma = _noisy(S, level)
    lam = 2 * sigma * np.sqrt(S.M)
    res = check_hypotheses(NOISY_Z, lam=lam, sigma=sigma, setup=S, verbose=False)
    assert res["non_degenerate"]
    assert res["noise_over_lambda"] == pytest.approx(0.5)
    # Noyau normalisé : ||ψ|| = 1, perturbation du certificat ≤ ||w|| / λ
    assert res["noise_certificate_bound"] == pytest.approx(0.5, rel=1e-6)

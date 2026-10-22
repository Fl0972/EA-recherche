"""Tests du module duval_peyre : dérivées, pré-certificat, hypothèses H1-H3, BLASSO."""

import numpy as np
import pytest

from particle_sources.duval_peyre import (GaussianKernel, InverseSquareKernel,
                                          check_derivatives, diagnose, eta,
                                          gamma_matrix, precertificate)
from particle_sources.duval_peyre.blasso import (_positive_lasso, debias, match,
                                                 operator_norm, sliding_frank_wolfe)
from particle_sources.duval_peyre.certificate import eta_grad
from particle_sources.duval_peyre.experiments import circle_positions, pair
from particle_sources.model import Box, Sources, intensity_at

R = 1.0
SENSORS = circle_positions([0, 0], R, 64)
BOX = Box.from_bounds([[-0.6, 0.6], [-0.6, 0.6]])   # D strictement à l'intérieur du cercle


@pytest.mark.parametrize("kernel", [InverseSquareKernel(SENSORS, 0.0),
                                    InverseSquareKernel(SENSORS, 0.7),
                                    GaussianKernel(SENSORS, 0.5)])
def test_derivatives_match_finite_differences(kernel):
    """Gradient et hessienne analytiques = différences finies centrées."""
    for z in ([0.1, -0.2], [0.5, 0.4]):
        err = check_derivatives(kernel, z)
        assert err["grad"] < 1e-7 and err["hess"] < 1e-7


def test_kernel_is_project_model():
    """phi(z) est bien la colonne du modèle direct du projet (intensity_at)."""
    for eps in (0.0, 0.3):
        k = InverseSquareKernel(SENSORS, eps)
        z = np.array([0.2, -0.1])
        assert np.allclose(k.phi(z), intensity_at(Sources([z], [1.0]), SENSORS, eps))


def test_precertificate_constraints():
    """eta_V(z_k) = 1 et grad eta_V(z_k) = 0 à la précision machine."""
    k = InverseSquareKernel(SENSORS, 0.0)
    z = np.array([[0.3, 0.0], [-0.2, 0.3], [0.0, -0.4]])
    p = precertificate(gamma_matrix(k, z), len(z))
    assert np.allclose(eta(k, p, z), 1.0, atol=1e-12)
    assert np.max(np.abs(eta_grad(k, p, z))) < 1e-10


def test_single_source_at_center_closed_form():
    """
    K = 1 au centre du cercle des capteurs (h = 0) : par symétrie p_V est
    constant et eta_V(z) = R^2 / (R^2 - |z|^2) > 1 : z_1 est un MINIMUM de
    eta_V (hessienne 2/R^2 I) -> H1 vraie, H2 et H3 fausses.
    """
    k = InverseSquareKernel(SENSORS, 0.0)
    p = precertificate(gamma_matrix(k, [[0.0, 0.0]]), 1)
    pts = np.array([[0.3, 0.1], [-0.5, 0.2], [0.0, 0.55]])
    rho2 = np.sum(pts**2, axis=-1)
    assert np.allclose(eta(k, p, pts), R**2 / (R**2 - rho2), rtol=1e-10)
    d = diagnose(k, [[0.0, 0.0]], BOX, h=0.02)
    assert d.h1 and not d.h2 and not d.h3
    assert np.allclose(d.eigenvalues, 2 / R**2, rtol=1e-8)
    assert not d.nondegenerate


def test_single_source_deep_sensors_ok():
    """K = 1, capteurs en profondeur (h = 2R) : la condition tient."""
    d = diagnose(InverseSquareKernel(SENSORS, 2.0), [[0.1, 0.05]], BOX, h=0.02)
    assert d.h1 and d.h2 and d.h3 and d.nondegenerate
    assert np.all(d.eigenvalues < 0)


def test_gaussian_witness_ok():
    """Témoin : noyau gaussien, capteurs denses, 2 sources séparées -> condition vérifiée."""
    xs = np.linspace(-1, 1, 15)
    grid = np.stack(np.meshgrid(xs, xs), axis=-1).reshape(-1, 2)
    d = diagnose(GaussianKernel(grid, 0.15), [[-0.3, 0.0], [0.3, 0.1]], BOX, h=0.01)
    assert d.nondegenerate


def test_close_sources_fail():
    """Deux sources très proches : H2 ou H3 échoue (alors que séparées, ça tient)."""
    k = InverseSquareKernel(SENSORS, 2.0)
    assert diagnose(k, pair([0, 0], 0.8, angle=0.0), BOX, h=0.02).nondegenerate
    d = diagnose(k, pair([0, 0], 0.02, angle=0.0), BOX, h=0.01)
    assert not (d.h2 and d.h3)
    assert not d.nondegenerate


def test_too_few_sensors_h1():
    """M < 3K : H1 fausse."""
    d = diagnose(InverseSquareKernel(circle_positions([0, 0], R, 5), 2.0),
                 [[-0.3, 0.0], [0.3, 0.0]], BOX, h=0.05)
    assert not d.h1 and not d.nondegenerate


def test_positive_lasso_kkt():
    """Solution du LASSO positif : conditions KKT (A^T r <= lam, = lam sur le support)."""
    rng = np.random.default_rng(0)
    A = rng.uniform(0, 1, (20, 200))
    m = A[:, [3, 50]] @ [1.0, 2.0] + 0.01 * rng.standard_normal(20)
    lam = 0.1
    a = _positive_lasso(A, m, lam)
    corr = A.T @ (m - A @ a)
    assert np.all(a >= 0)
    assert np.all(corr <= lam + 1e-9)
    assert np.allclose(corr[a > 0], lam, atol=1e-9)


def test_sfw_recovers_when_condition_holds():
    """Condition vérifiée + bruit faible : exactement K sources, erreurs petites."""
    k = InverseSquareKernel(SENSORS, 2.0)
    z, I = pair([0, 0], 0.8, angle=0.0), np.array([1.0, 1.5])
    y = k.forward(z, I)
    eps = 1e-5 * np.linalg.norm(y) / 8 * np.random.default_rng(1).standard_normal(64)
    m = y + eps
    lam = np.linalg.norm(eps) * operator_norm(k, BOX, 0.02)
    est, _, _ = sliding_frank_wolfe(k, m, lam, BOX, 0.02)
    res = match(z, I, debias(k, m, est))
    assert res["exact"]
    assert res["err_pos"] < 1e-2 and res["err_int"] < 1e-2


# ---- 3D ----------------------------------------------------------------------------------
CUBE = Box.from_bounds([[0, 10]] * 3)


def test_3d_single_ring_not_identifiable():
    """
    Un seul cercle de capteurs : deux sources distinctes (rho, h, I) donnent
    les mêmes mesures, et rang(Gamma) = 3K au lieu de 4K.
    """
    from particle_sources.duval_peyre.experiments_3d import layout
    S = layout("un_cercle", CUBE, 64, 1.5)
    k = InverseSquareKernel(S)
    c, R = CUBE.center, 1.5 * CUBE.circumradius
    rho, h, I = 3.0, 2.0, 100.0
    A = (R**2 + rho**2 + h**2) / (2 * R * rho)
    rho2 = 4.0
    h2 = np.sqrt(2 * R * rho2 * A - R**2 - rho2**2)
    y1 = k.forward([c + [rho, 0, h]], [I])
    y2 = k.forward([c + [rho2, 0, h2]], [I * rho2 / rho])
    assert np.allclose(y1, y2, rtol=1e-12)
    z = np.array([[2.0, 3.0, 4.0], [7.0, 6.0, 5.5]])
    sv = np.linalg.svd(gamma_matrix(k, z), compute_uv=False)
    assert np.sum(sv > 1e-10 * sv[0]) == 3 * len(z)


def test_3d_two_rings_full_rank_but_h3_fails():
    """Deux cercles à des hauteurs différentes : H1 vraie, mais H3 fausse (1/r^2 sous-harmonique)."""
    from particle_sources.duval_peyre.experiments_3d import layout
    k = InverseSquareKernel(layout("deux_cercles", CUBE, 96, 1.5))
    d = diagnose(k, [[4.0, 6.0, 5.0]], CUBE, h=0.5)
    assert d.h1 and d.eigenvalues.shape == (1, 3)
    assert not d.h3 and not d.nondegenerate

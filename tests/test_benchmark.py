import numpy as np
import pytest

from particle_sources.benchmark import crb, evaluate, ospa
from particle_sources.generation import central_source, count_laws, sample_sources
from particle_sources.inversion import InverseProblem, prony_mle
from particle_sources.model import Box, Sources, intensity_at, intensity_jacobian
from particle_sources.sensors import CircleSensors, measure, noise

BOX = Box.from_bounds([[0, 10], [0, 20]])
SENSORS = CircleSensors.circumscribed(BOX, 64)


def _problem(noise_model, known=None):
    return InverseProblem(SENSORS, BOX, noise_model,
                          known if known is not None else Sources.empty(), 100.0)


def test_jacobian_matches_finite_differences():
    s = sample_sources(BOX, count_laws.fixed(3), 100.0, np.random.default_rng(0))
    J = intensity_jacobian(s, SENSORS.positions)
    theta = np.concatenate([s.positions.ravel(), s.intensities])
    h = 1e-6
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        f = lambda t: intensity_at(Sources(t[:6].reshape(3, 2), t[6:]), SENSORS.positions)
        assert np.allclose(J[:, k], (f(tp) - f(tm)) / (2 * h), rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prony_mle_exact_without_noise(seed):
    rng = np.random.default_rng(seed)
    true = sample_sources(BOX, count_laws.fixed(3), 100.0, rng)
    big = central_source(BOX, 5000.0)
    problem = _problem(noise.no_noise, big)
    est = prony_mle(measure(true + big, SENSORS), problem)
    r = evaluate(true, est, cutoff=2.0)
    assert r.n_est == 3
    assert r.position_errors.max() < 1e-4
    assert r.intensity_errors.max() < 1e-3


def test_ospa():
    a = Sources([[0.0, 0.0], [5.0, 5.0]], [1.0, 1.0])
    assert ospa(a, a, cutoff=2.0) == 0.0
    assert ospa(Sources.empty(), Sources.empty(), cutoff=2.0) == 0.0
    assert ospa(a, Sources.empty(), cutoff=2.0) == pytest.approx(2.0)
    shifted = Sources([[0.0, 0.1], [5.0, 5.0]], [1.0, 1.0])
    assert ospa(a, shifted, cutoff=2.0) == pytest.approx(np.sqrt(0.1**2 / 2))
    one = Sources([[0.0, 0.0]], [1.0])
    assert ospa(a, one, cutoff=2.0) == pytest.approx(np.sqrt(2.0**2 / 2))


def test_crb_scales_with_noise_and_is_attained():
    rng = np.random.default_rng(3)
    true = sample_sources(BOX, count_laws.fixed(2), 100.0, rng)
    p1, i1 = crb(true, _problem(noise.gaussian_relative(1e-4)))
    p2, i2 = crb(true, _problem(noise.gaussian_relative(1e-3)))
    assert np.allclose(p2 / p1, 10, rtol=1e-2)

    # À faible bruit, le maximum de vraisemblance atteint la borne
    # (le BIC ajoute parfois une source parasite minuscule : ~2 % des cas)
    problem = _problem(noise.gaussian_relative(1e-5))
    errors = []
    for _ in range(200):
        est = prony_mle(measure(true, SENSORS, rng, problem.noise), problem)
        r = evaluate(true, est, cutoff=2.0)
        if r.n_est == r.n_true:
            errors.append(r.position_errors)
    assert len(errors) > 180
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    p, _ = crb(true, problem)
    assert np.all(rmse / p > 0.7) and np.all(rmse / p < 1.4)

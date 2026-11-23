import numpy as np
import pytest

from particle_sources.benchmark import TwoSourcesSpec, source_errors
from particle_sources.generation import bright_and_faint
from particle_sources.inversion import (InverseProblem, curvature_spectrum, gradient_descent,
                                        refine)
from particle_sources.model import Box, Sources, intensity_at
from particle_sources.sensors import CircleSensors, noise

BOX = Box.from_bounds([[0, 10], [0, 20]])
SENSORS = CircleSensors.circumscribed(BOX, 64)
PROBLEM = InverseProblem(SENSORS, BOX, noise.no_noise, Sources.empty(), 100.0)
TRUE_POS = [[3, 6], [6.5, 15]]
INIT = Sources([[4, 9], [6, 11]], [50, 50])


def test_bright_and_faint():
    s = bright_and_faint(TRUE_POS, 10, 100.0)
    assert s.total_intensity == pytest.approx(100.0)
    assert s.intensities[0] / s.intensities[1] == pytest.approx(10)


def test_descent_recovers_equal_sources():
    true = bright_and_faint(TRUE_POS, 1, 100.0)
    y = intensity_at(true, SENSORS.positions)
    result = gradient_descent(y, PROBLEM, INIT, max_iter=2000)
    assert result.converged
    assert np.all(source_errors(true, result.sources) < 1e-6)
    assert result.losses[-1] < result.losses[0]


def test_descent_respects_constraints():
    true = bright_and_faint(TRUE_POS, 100, 100.0)
    y = intensity_at(true, SENSORS.positions)
    for step in ("armijo", 0.3):
        est = gradient_descent(y, PROBLEM, INIT, max_iter=200, step=step).sources
        assert est.n == 2
        assert np.all(est.positions >= BOX.lo) and np.all(est.positions <= BOX.hi)
        assert np.all(est.intensities >= 0)


def test_faint_source_is_missed_by_descent_but_not_by_gauss_newton():
    """Le constat de l'industriel : à x = 100, la descente manque la source faible."""
    true = bright_and_faint(TRUE_POS, 100, 100.0)
    y = intensity_at(true, SENSORS.positions)
    gd = gradient_descent(y, PROBLEM, INIT, max_iter=2000).sources
    gn = refine(y, PROBLEM, INIT)
    assert source_errors(true, gd)[1] > 1.0
    assert np.all(source_errors(true, gn) < 1e-6)


def test_conditioning_grows_with_ratio():
    cond = []
    for x in (10, 100):
        eig = curvature_spectrum(bright_and_faint(TRUE_POS, x, 100.0), PROBLEM)
        cond.append(eig[-1] / eig[0])
    assert cond[1] / cond[0] > 30          # ~ x^2 = 100 asymptotiquement


def test_spec_from_raw():
    raw = {"two_sources": {"positions": TRUE_POS, "ratios": [10, 100],
                           "init": {"positions": [[4, 9], [6, 11]]}}}
    spec = TwoSourcesSpec.from_raw(raw)
    assert spec.ratios == [10.0, 100.0]
    assert spec.n_trials == 0 and spec.step == "armijo"
    with pytest.raises(KeyError):
        TwoSourcesSpec.from_raw({})

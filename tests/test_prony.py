"""Tests de la reconstruction par la méthode de Prony."""

import numpy as np
import pytest

from particle_sources.generation import count_laws, sample_sources
from particle_sources.inversion import prony_circle
from particle_sources.model import Box
from particle_sources.sensors import CircleSensors, measure

BOX = Box.from_bounds([[0, 10], [0, 10]])


def _sorted(sources):
    """Positions et intensités triées, pour comparer deux ensembles de sources."""
    order = np.lexsort(sources.positions.T)
    return sources.positions[order], sources.intensities[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("t0", [0.0, 0.3])
def test_exact_recovery_without_noise(seed, t0):
    """Sans bruit, Prony retrouve exactement nombre, positions et intensités."""
    rng = np.random.default_rng(seed)
    true = sample_sources(BOX, count_laws.fixed(5), 100.0, rng)
    sensors = CircleSensors.circumscribed(BOX, 64, t0)
    estimate = prony_circle(measure(true, sensors, rng), sensors)

    assert estimate.n == true.n
    pos_est, I_est = _sorted(estimate)
    pos_true, I_true = _sorted(true)
    assert np.allclose(pos_est, pos_true, atol=1e-4)
    # Tolérance absolue : une source minuscule dans un coin (|u| proche de 1)
    # subit un léger aliasing sur son intensité relative.
    assert np.allclose(I_est, I_true, rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_subtracting_central_source_is_as_if_absent(seed):
    """Retirer la source centrale connue revient à l'inverser comme si elle était absente."""
    from particle_sources.generation import central_source
    from particle_sources.inversion import subtract_known_sources

    rng = np.random.default_rng(seed)
    true = sample_sources(BOX, count_laws.fixed(4), 100.0, rng)
    big = central_source(BOX, 5000.0)
    sensors = CircleSensors.circumscribed(BOX, 64)

    y_with = subtract_known_sources(measure(true + big, sensors), big, sensors)
    y_without = measure(true, sensors)
    assert np.allclose(y_with, y_without, rtol=1e-9)

    pos_a, I_a = _sorted(prony_circle(y_with, sensors))
    pos_b, I_b = _sorted(prony_circle(y_without, sensors))
    assert np.allclose(pos_a, pos_b, atol=1e-6)
    assert np.allclose(I_a, I_b, atol=1e-6)

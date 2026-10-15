import numpy as np
import pytest

from particle_sources.generation import count_laws, sample_sources
from particle_sources.inversion import prony_circle
from particle_sources.model import Box
from particle_sources.sensors import CircleSensors, measure

BOX = Box.from_bounds([[0, 10], [0, 10]])


def _sorted(sources):
    order = np.lexsort(sources.positions.T)
    return sources.positions[order], sources.intensities[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("t0", [0.0, 0.3])
def test_exact_recovery_without_noise(seed, t0):
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

"""Tests de la génération aléatoire des sources."""

import numpy as np
import pytest

from particle_sources.generation import count_laws, intensity_laws, sample_sources
from particle_sources.model import Box

BOX = Box.from_bounds([[0, 10], [0, 10]])


@pytest.mark.parametrize("seed", range(20))
def test_sources_inside_box_and_total_intensity(seed):
    """Sources dans le domaine, intensités positives de somme fixée."""
    rng = np.random.default_rng(seed)
    sources = sample_sources(BOX, count_laws.poisson(8.0), 100.0, rng)
    assert np.all((sources.positions >= BOX.lo) & (sources.positions <= BOX.hi))
    assert np.all(sources.intensities >= 0)
    if sources.n:
        assert sources.total_intensity == pytest.approx(100.0)


def test_zero_sources():
    """N = 0 donne un ensemble vide de forme correcte."""
    sources = sample_sources(BOX, count_laws.fixed(0), 100.0, np.random.default_rng(0))
    assert sources.n == 0
    assert sources.positions.shape == (0, 2)


def test_at_least_one():
    """`at_least_one` ne renvoie jamais N = 0."""
    rng = np.random.default_rng(0)
    law = count_laws.at_least_one(count_laws.poisson(0.1))
    assert all(law(rng) >= 1 for _ in range(100))


def test_uniform_simplex_mean():
    """Sur le simplexe uniforme, E[I_i] = total / n."""
    # Sur le simplexe uniforme, E[I_i] = total / n
    rng = np.random.default_rng(0)
    draws = np.array([intensity_laws.uniform_simplex(4, 1.0, rng) for _ in range(20000)])
    assert np.allclose(draws.mean(axis=0), 0.25, atol=0.01)

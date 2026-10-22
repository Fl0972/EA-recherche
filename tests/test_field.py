"""Tests du modèle direct (loi en 1/r^2, superposition) et des objets Box / Sources."""

import numpy as np
import pytest

from particle_sources.model import Box, Sources, intensity_at


def test_single_source_inverse_square():
    """Une source seule donne bien I / r^2."""
    sources = Sources([[0.0, 0.0]], [4.0])
    assert intensity_at(sources, [2.0, 0.0]) == pytest.approx(1.0)
    assert intensity_at(sources, [0.0, 4.0]) == pytest.approx(0.25)


def test_superposition_and_shape():
    """Phi est additif en les sources et respecte la forme (..., d) -> (...)."""
    a = Sources([[0.0, 0.0]], [1.0])
    b = Sources([[3.0, 1.0]], [2.0])
    ab = Sources([[0.0, 0.0], [3.0, 1.0]], [1.0, 2.0])
    x = np.random.default_rng(0).uniform(5, 6, size=(7, 3, 2))
    assert intensity_at(ab, x).shape == (7, 3)
    assert np.allclose(intensity_at(ab, x), intensity_at(a, x) + intensity_at(b, x))


def test_circumradius():
    """Rayon du cercle circonscrit et centre d'un carré."""
    box = Box.from_bounds([[0, 10], [0, 10]])
    assert box.circumradius == pytest.approx(5 * np.sqrt(2))
    assert np.allclose(box.center, [5, 5])


def test_sources_addition():
    """`a + b` réunit deux ensembles de sources."""
    a = Sources([[0.0, 0.0]], [1.0])
    b = Sources([[3.0, 1.0], [1.0, 1.0]], [2.0, 5.0])
    ab = a + b
    assert ab.n == 3 and ab.total_intensity == pytest.approx(8.0)
    x = [[5.0, 5.0]]
    assert np.allclose(intensity_at(ab, x), intensity_at(a, x) + intensity_at(b, x))
    assert (a + Sources.empty()).n == 1

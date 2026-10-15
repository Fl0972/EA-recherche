import numpy as np
import pytest

from particle_sources.generation import count_laws, intensity_laws, position_laws, sample_sources
from particle_sources.model import Box, Sources

BOX = Box.from_bounds([[0, 10], [0, 10]])


@pytest.mark.parametrize("seed", range(20))
def test_sources_inside_box_and_total_intensity(seed):
    rng = np.random.default_rng(seed)
    sources = sample_sources(BOX, count_laws.poisson(8.0), 100.0, rng)
    assert np.all((sources.positions >= BOX.lo) & (sources.positions <= BOX.hi))
    assert np.all(sources.intensities >= 0)
    if sources.n:
        assert sources.total_intensity == pytest.approx(100.0)


def test_zero_sources():
    sources = sample_sources(BOX, count_laws.fixed(0), 100.0, np.random.default_rng(0))
    assert sources.n == 0
    assert sources.positions.shape == (0, 2)


def test_at_least_one():
    rng = np.random.default_rng(0)
    law = count_laws.at_least_one(count_laws.poisson(0.1))
    assert all(law(rng) >= 1 for _ in range(100))


def test_uniform_simplex_mean():
    # Sur le simplexe uniforme, E[I_i] = total / n
    rng = np.random.default_rng(0)
    draws = np.array([intensity_laws.uniform_simplex(4, 1.0, rng) for _ in range(20000)])
    assert np.allclose(draws.mean(axis=0), 0.25, atol=0.01)


def test_disk_cluster():
    rng = np.random.default_rng(42)
    radius = 1.5
    law = position_laws.disk_cluster(radius=radius)
    pts = law(50, BOX, rng)
    assert pts.shape == (50, 2)
    assert np.all((pts >= BOX.lo) & (pts <= BOX.hi))
    # Vérifie que les points sont à distance <= radius du centre
    dists = np.linalg.norm(pts - BOX.center, axis=-1)
    assert np.all(dists <= radius + 1e-12)

    # Cas n = 0
    assert law(0, BOX, rng).shape == (0, 2)
    with pytest.raises(ValueError, match="radius doit être strictement positif"):
        position_laws.disk_cluster(radius=-1.0)


def test_gaussian_cluster():
    rng = np.random.default_rng(42)
    law = position_laws.gaussian_cluster(std=0.5)
    pts = law(100, BOX, rng)
    assert pts.shape == (100, 2)
    assert np.all((pts >= BOX.lo) & (pts <= BOX.hi))
    # La majorité des points sont proches du centre
    mean_dist = np.mean(np.linalg.norm(pts - BOX.center, axis=-1))
    assert mean_dist < 1.0

    assert law(0, BOX, rng).shape == (0, 2)
    with pytest.raises(ValueError, match="std doit être strictement positif"):
        position_laws.gaussian_cluster(std=0)


def test_clustered_center():
    rng = np.random.default_rng(42)
    radius = 1.0
    law = position_laws.clustered_center(radius=radius, cluster_ratio=0.8)
    n = 20
    pts = law(n, BOX, rng)
    assert pts.shape == (n, 2)
    assert np.all((pts >= BOX.lo) & (pts <= BOX.hi))

    # Les n_cluster = 16 premiers points sont dans le disque de rayon `radius`
    n_cluster = int(np.round(n * 0.8))
    cluster_dists = np.linalg.norm(pts[:n_cluster] - BOX.center, axis=-1)
    assert np.all(cluster_dists <= radius + 1e-12)

    assert law(0, BOX, rng).shape == (0, 2)


def test_cluster_boost_intensity():
    rng = np.random.default_rng(42)
    law = intensity_laws.cluster_boost(cluster_ratio=0.8, cluster_share=0.9)
    total = 500.0
    n = 20
    intensities = law(n, total, rng)
    assert len(intensities) == n
    assert np.all(intensities >= 0)
    assert intensities.sum() == pytest.approx(total)

    n_cluster = int(np.round(n * 0.8))  # 16
    cluster_sum = intensities[:n_cluster].sum()
    bg_sum = intensities[n_cluster:].sum()
    assert cluster_sum == pytest.approx(total * 0.9)
    assert bg_sum == pytest.approx(total * 0.1)

    # Cas limites
    assert len(law(0, total, rng)) == 0
    assert law(1, total, rng)[0] == pytest.approx(total)


def test_sources_addition():
    s1 = Sources(np.array([[1.0, 2.0]]), np.array([10.0]))
    s2 = Sources(np.array([[3.0, 4.0], [5.0, 6.0]]), np.array([20.0, 30.0]))
    combined = s1 + s2
    assert combined.n == 3
    assert combined.total_intensity == pytest.approx(60.0)
    assert np.allclose(combined.positions[0], [1.0, 2.0])
    assert np.allclose(combined.positions[1:], [[3.0, 4.0], [5.0, 6.0]])

    empty = Sources.empty(2)
    assert (empty + s1).n == 1
    assert (s1 + empty).n == 1


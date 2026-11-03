import numpy as np
import pytest

from particle_sources.config import load_config
from particle_sources.config.cli import DEFAULT_CONFIG_DIR
from particle_sources.dataset import generate_dataset, load_dataset, load_manifest
from particle_sources.model import Sources, intensity_at

OVERRIDES = ["seed=0", "sources.count_law.params={lam: 3.0}"]


def _make(tmp_path, name="ds", n=50, shard=20, extra=()):
    cfg = load_config(DEFAULT_CONFIG_DIR, [*OVERRIDES, *extra])
    return generate_dataset(cfg, tmp_path / name, n, shard, dtype="float64",
                            verbose=False), cfg


def test_shapes_and_labels(tmp_path):
    path, cfg = _make(tmp_path)
    data = load_dataset(path)
    K = len(cfg.sensors.positions)
    n_max = data["mask"].shape[1]

    assert data["measurements"].shape == (50, K)
    assert data["measurements_clean"].shape == (50, K)
    assert data["positions"].shape == (50, n_max, 2)
    assert np.array_equal(data["mask"].sum(axis=1), data["n_sources"])
    assert len(load_manifest(path)["shards"]) == 3          # 20 + 20 + 10

    for i in range(50):
        m = data["mask"][i]
        I = data["intensities"][i, m]
        if m.any():
            assert I.sum() == pytest.approx(cfg.total_intensity)
            assert np.all(np.diff(I) <= 0)                   # tri décroissant
        assert np.all(data["intensities"][i, ~m] == 0)


def test_clean_measurements_match_labels(tmp_path):
    path, cfg = _make(tmp_path)
    data = load_dataset(path)
    for i in range(0, 50, 7):
        m = data["mask"][i]
        sources = Sources(data["positions"][i, m], data["intensities"][i, m])
        expected = intensity_at(sources + cfg.fixed_sources, cfg.sensors.positions, cfg.eps)
        assert np.allclose(data["measurements_clean"][i], expected)


def test_reproducible(tmp_path):
    a = load_dataset(_make(tmp_path, "a")[0])
    b = load_dataset(_make(tmp_path, "b")[0])
    for key in a:
        assert np.array_equal(a[key], b[key])


def test_overwrite_protection_and_max_sources(tmp_path):
    _make(tmp_path)
    with pytest.raises(FileExistsError):
        _make(tmp_path)
    cfg = load_config(DEFAULT_CONFIG_DIR, OVERRIDES)
    path = generate_dataset(cfg, tmp_path / "pad", 10, 10, max_sources=30, verbose=False)
    assert load_dataset(path)["mask"].shape == (10, 30)

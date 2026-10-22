"""Tests de la configuration : chargement YAML, surcharges, erreurs, sauvegarde, source centrale."""

from pathlib import Path

import numpy as np
import pytest

from particle_sources.config import load_config, save_config
from particle_sources.config.cli import DEFAULT_CONFIG_DIR
from particle_sources.config.loader import read_config_dir
from particle_sources.generation import sample_sources
from particle_sources.sensors import measure


def test_default_config_runs_end_to_end():
    """La config par défaut permet d'enchaîner tirage, mesure et inversion."""
    cfg = load_config(DEFAULT_CONFIG_DIR)
    rng = cfg.rng()
    sources = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                             cfg.intensity_law, cfg.position_law)
    y = measure(sources, cfg.sensors, rng, cfg.noise, cfg.eps)
    estimate = cfg.invert(y, cfg.sensors)
    assert y.shape == (len(cfg.sensors.positions),)
    assert estimate.dim == 2


def test_overrides():
    """Les surcharges --set remplacent bien les valeurs des YAML."""
    cfg = load_config(DEFAULT_CONFIG_DIR, [
        "seed=42",
        "sources.count_law.name=fixed",
        "sources.count_law.params={n: 3}",
        "sensors.layout.params.n=16",
    ])
    assert cfg.seed == 42
    assert cfg.count_law(cfg.rng()) == 3
    assert len(cfg.sensors.positions) == 16


def test_unknown_name_gives_clear_error():
    """Un nom de loi inconnu donne une erreur listant les choix possibles."""
    with pytest.raises(ValueError, match="Choix possibles"):
        load_config(DEFAULT_CONFIG_DIR, ["sources.count_law.name=gauss"])


def test_bad_params_give_clear_error():
    """Des paramètres invalides donnent une erreur explicite."""
    with pytest.raises(ValueError, match="paramètres invalides"):
        load_config(DEFAULT_CONFIG_DIR, ["sources.count_law.params={mu: 3}"])


def test_save_and_reload(tmp_path: Path):
    """Une config sauvegardée puis relue redonne la même expérience."""
    cfg = load_config(DEFAULT_CONFIG_DIR, ["seed=7"])
    save_config(cfg, tmp_path / "used.yaml")
    assert read_config_dir(tmp_path) == cfg.raw
    reloaded = load_config(tmp_path)
    assert reloaded.seed == 7
    assert np.allclose(reloaded.sensors.positions, cfg.sensors.positions)


def test_scientific_notation_is_float():
    """`1e-8` est lu comme un float (et non une chaîne)."""
    cfg = load_config(DEFAULT_CONFIG_DIR, ["reconstruction.method.params.rank_tol=1e-8"])
    assert cfg.raw["reconstruction"]["method"]["params"]["rank_tol"] == 1e-8


def test_central_source_enabled():
    """La source centrale est au centre, d'intensité intensity_factor * total_intensity."""
    cfg = load_config(DEFAULT_CONFIG_DIR, [
        "central_source.enabled=true", "central_source.intensity_factor=50",
        "central_source.position=null", "sources.total_intensity=100",
    ])
    assert cfg.fixed_sources.n == 1
    assert np.allclose(cfg.fixed_sources.positions[0], cfg.box.center)
    assert cfg.fixed_sources.intensities[0] == pytest.approx(5000.0)


def test_central_source_disabled_and_not_in_draw():
    """La source centrale peut être désactivée et ne modifie jamais le tirage aléatoire."""
    # Même graine pour les deux configs (la config par défaut peut la tirer au hasard)
    cfg = load_config(DEFAULT_CONFIG_DIR, ["seed=1", "central_source.enabled=false"])
    assert cfg.fixed_sources.n == 0
    # La source fixe ne participe jamais au tirage aléatoire
    cfg_on = load_config(DEFAULT_CONFIG_DIR, ["seed=1", "central_source.enabled=true"])
    draw = lambda c: sample_sources(c.box, c.count_law, c.total_intensity, c.rng(),
                                    c.intensity_law, c.position_law)
    assert np.allclose(draw(cfg).positions, draw(cfg_on).positions)

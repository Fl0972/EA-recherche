"""Tests de la reconstruction par approximation rationnelle (AAA + moindres carrés)."""

import numpy as np
import pytest

from particle_sources.inversion.aaa_sources import reconstruire
from particle_sources.model import Sources, intensity_at
from particle_sources.sensors import CircleSensors

# Cas de référence : grosse source au centre + 3 petites sources
Z_REF = np.array([0, 0.5 + 0.2j, -0.3 + 0.45j, 0.1 - 0.6j])
I_REF = np.array([50.0, 1.0, 0.8, 1.2])


def _sources(z, I, c=(0.0, 0.0)):
    """Sources aux positions complexes z (relatives à c)."""
    z = np.asarray(z, dtype=complex)
    return Sources(np.asarray(c) + np.stack([z.real, z.imag], axis=-1), I)


def _mesures(true, sensors, sigma=0.0, seed=0):
    """Mesures exactes + bruit gaussien d'écart-type sigma * moyenne(phi)."""
    phi = intensity_at(true, sensors.positions)
    rng = np.random.default_rng(seed)
    return phi + sigma * phi.mean() * rng.standard_normal(phi.shape)


def _apparier(estimate, true):
    """Pour chaque vraie source, la source estimée la plus proche : (positions, intensités)."""
    d = np.linalg.norm(true.positions[:, None] - estimate.positions[None], axis=-1)
    j = np.argmin(d, axis=1)
    assert len(set(j)) == true.n, "deux vraies sources appariées à la même estimation"
    return estimate.positions[j], estimate.intensities[j]


@pytest.fixture
def capteurs_ref():
    return CircleSensors([0.0, 0.0], 1.0, 128, 0.3)


def test_reference_sans_bruit(capteurs_ref):
    """Sans bruit, les 4 sources sont retrouvées à 1e-6 près."""
    true = _sources(Z_REF, I_REF)
    estimate, _ = reconstruire(_mesures(true, capteurs_ref), capteurs_ref)
    assert estimate.n == 4
    pos, I = _apparier(estimate, true)
    assert np.allclose(pos, true.positions, atol=1e-6)
    assert np.allclose(I, true.intensities, atol=1e-6)


def test_reference_bruit_faible(capteurs_ref):
    """sigma = 1e-3 : 4 sources, positions à 0.02 près, intensités à 10 % près."""
    true = _sources(Z_REF, I_REF)
    estimate, _ = reconstruire(_mesures(true, capteurs_ref, 1e-3), capteurs_ref)
    assert estimate.n == 4
    pos, I = _apparier(estimate, true)
    assert np.all(np.linalg.norm(pos - true.positions, axis=-1) < 0.02)
    assert np.allclose(I, true.intensities, rtol=0.1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reference_bruit_fort(capteurs_ref, seed):
    """sigma = 1e-2 : pas de crash, L <= 6, au plus L/2 sources hors centre."""
    true = _sources(Z_REF, I_REF)
    _, diag = reconstruire(_mesures(true, capteurs_ref, 1e-2, seed), capteurs_ref)
    assert diag["L"] <= 6
    assert diag["n_hors_centre"] <= diag["L"] // 2


def test_changement_de_repere():
    """Centre c != 0 et t0 quelconque, sans bruit : positions absolues correctes."""
    c, R = np.array([3.0, -2.0]), 2.5
    sensors = CircleSensors(c, R, 128, 1.234)
    true = _sources(R * Z_REF, I_REF * R**2, c)
    estimate, _ = reconstruire(_mesures(true, sensors), sensors)
    assert estimate.n == 4
    pos, I = _apparier(estimate, true)
    assert np.allclose(pos, true.positions, atol=1e-6)
    assert np.allclose(I, true.intensities, rtol=1e-6)


def test_residu_trouve_les_sources_manquees(capteurs_ref):
    """
    7 sources autour d'une grosse source centrale, sigma = 1e-3 : une seule
    passe AAA en rate la plupart, la recherche sur le résidu les retrouve.
    """
    z = np.array([0, 0.6, 0.6j, -0.6, -0.6j, 0.45 + 0.45j, -0.45 - 0.45j, 0.3 - 0.5j])
    I = np.array([50, 1, 1.5, 0.7, 1.2, 0.9, 1.1, 0.6])
    true = _sources(z, I)
    y = _mesures(true, capteurs_ref, 1e-3)

    une_passe, _ = reconstruire(y, capteurs_ref)
    estimate, diag = reconstruire(y, capteurs_ref, iterations_residu=30)
    assert une_passe.n < true.n
    assert diag["ajoutees_residu"] > 0
    assert estimate.n == true.n
    pos, _ = _apparier(estimate, true)
    assert np.all(np.linalg.norm(pos - true.positions, axis=-1) < 0.1)


def test_residu_n_invente_pas_de_source(capteurs_ref):
    """Cas de référence (sigma = 1e-3) : la recherche sur le résidu n'ajoute rien."""
    true = _sources(Z_REF, I_REF)
    estimate, diag = reconstruire(_mesures(true, capteurs_ref, 1e-3), capteurs_ref,
                                  iterations_residu=30)
    assert estimate.n == 4
    assert diag["ajoutees_residu"] == 0


def test_une_seule_source_decentree(capteurs_ref):
    """Une source hors du centre, sans source centrale : on ne trouve qu'elle."""
    true = _sources([0.4 - 0.3j], [2.0])
    estimate, _ = reconstruire(_mesures(true, capteurs_ref), capteurs_ref)
    assert estimate.n == 1
    assert np.allclose(estimate.positions, true.positions, atol=1e-6)
    assert np.allclose(estimate.intensities, true.intensities, rtol=1e-6)

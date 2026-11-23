import numpy as np
import pytest

from particle_sources.benchmark import (NoiseStudySpec, TwoSourcesSpec, crb,
                                        run_fixed_geometry, run_random_geometries)
from particle_sources.generation import bright_and_faint
from particle_sources.inversion import InverseProblem
from particle_sources.model import Box, Sources
from particle_sources.sensors import CircleSensors, noise

BOX = Box.from_bounds([[0, 10], [0, 20]])
SENSORS = CircleSensors.circumscribed(BOX, 64)
POS = [[3, 6], [6.5, 15]]
BASE = TwoSourcesSpec.from_raw({"two_sources": {
    "positions": POS, "ratios": [100], "init": {"positions": [[4, 9], [6, 11]]}}})


def _spec(**kw):
    raw = {"ratios": [100], "sigmas": [1e-4], "n_realizations": 3, "n_geometries": 3,
           "seed": 0, **kw}
    return NoiseStudySpec.from_raw({"two_sources_noise": raw})


def test_faint_source_crb_grows_like_ratio():
    """La précision sur la source faible se dégrade proportionnellement à x."""
    problem = InverseProblem(SENSORS, BOX, noise.gaussian_relative(1e-3), Sources.empty(), 100.)
    c50 = crb(bright_and_faint(POS, 50, 100.), problem)[0]
    c200 = crb(bright_and_faint(POS, 200, 100.), problem)[0]
    assert c200[1] / c50[1] == pytest.approx(4, rel=0.15)
    assert c200[0] / c50[0] == pytest.approx(1, rel=0.1)      # source forte : inchangée


def test_fixed_geometry_reaches_crb_at_low_noise():
    rows = run_fixed_geometry(_spec(), BASE, SENSORS, BOX, 100.)
    prony = next(r for r in rows if r["method"].startswith("Prony"))
    assert prony["detection"] == 1.0 and prony["success"] == 1.0
    assert prony["faint_over_crb"] < 2
    gn = next(r for r in rows if r["method"].startswith("Gauss"))
    assert np.isnan(gn["detection"])                           # N imposé


def test_random_geometries_crb_is_linear_in_sigma():
    rows, geo = run_random_geometries(_spec(sigmas=[1e-4, 1e-3]), BASE, SENSORS, BOX, 100.)
    crb_by_sigma = {r["sigma"]: r["crb_faint"] for r in rows if r["method"].startswith("Prony")}
    assert crb_by_sigma[1e-3] / crb_by_sigma[1e-4] == pytest.approx(10, rel=1e-3)
    assert len(geo["rho_faint"]) == 3 and np.all(geo["rho_faint"] < 1)

"""Configuration pytest : backend matplotlib sans écran, setups partagés."""

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from sparse_sources import make_setup  # noqa: E402


@pytest.fixture(scope="session")
def setup_plane():
    """Géométrie du projet : 64 capteurs dans le plan (h = 0), noyau brut 1/r²."""
    return make_setup(h=0.0)


@pytest.fixture(scope="session")
def setup_normalized():
    """Même géométrie, noyau normalisé φ/||φ|| (BLASSO pondéré)."""
    return make_setup(h=0.0, normalize=True)


@pytest.fixture(scope="session")
def setup_high():
    """Capteurs à la hauteur h = 20 au-dessus du plan des sources, noyau brut."""
    return make_setup(h=20.0)

"""Capteurs à plusieurs hauteurs (paroi d'un cylindre) : même noyau 1 / (r² + h_i²)."""
import numpy as np
import pytest

from sparse_sources import Setup, make_cylinder_setup, make_setup
from sparse_sources.kernel import check_derivatives


def test_single_level_matches_scalar_height():
    """Un seul étage à la hauteur h = le Setup du projet avec eps = h."""
    A = make_cylinder_setup([4.0], 64, stagger=False)
    B = make_setup(h=4.0)
    Z = np.array([[2.0, 3.0], [7.5, 18.0]])
    np.testing.assert_allclose(A.phi(Z), B.phi(Z), rtol=1e-14)
    np.testing.assert_allclose(A.grad(Z), B.grad(Z), rtol=1e-14)


def test_kernel_formula_per_sensor():
    """φ_i(z) = 1 / (||x_i - z||² + h_i²) capteur par capteur."""
    S = make_cylinder_setup([0.0, 2.5, 5.0], 20)
    z = np.array([3.0, 7.0])
    expected = 1.0 / (np.sum((S.sensors - z) ** 2, axis=1) + S.heights ** 2)
    np.testing.assert_allclose(S.phi(z), expected, rtol=1e-14)
    assert S.M == 60 and sorted(set(S.heights)) == [0.0, 2.5, 5.0]


def test_up_and_down_are_equivalent():
    """Sources dans le plan : un étage à -h mesure la même chose qu'à +h."""
    A = make_cylinder_setup([-3.0, 6.0], 16)
    B = make_cylinder_setup([3.0, 6.0], 16)
    z = np.array([[4.0, 9.0]])
    np.testing.assert_allclose(A.phi(z), B.phi(z), rtol=1e-14)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_derivatives(alpha):
    """Gradient et hessienne analytiques contre différences finies."""
    S = make_cylinder_setup([0.0, 2.5, 7.5], 12).with_power(alpha)
    for z in ([3.0, 7.0], [0.5, 19.0]):
        err = check_derivatives(np.array(z), S)
        assert err["grad"] < 1e-6 and err["hess"] < 1e-6


def test_heights_kept_by_variants_and_validated():
    """with_power / with_normalization / with_weights gardent les hauteurs ; validation."""
    S = make_cylinder_setup([0.0, 5.0], 10)
    for T in (S.with_power(1.0), S.with_normalization(True), S.with_weights(np.ones(20))):
        np.testing.assert_array_equal(T.heights, S.heights)
    assert "5" in S.height_label
    with pytest.raises(ValueError):
        Setup(S.sensors, S.domain, heights=np.ones(3))
    with pytest.raises(ValueError):
        Setup(S.sensors, S.domain, heights=-np.ones(20))

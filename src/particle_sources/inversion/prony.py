"""
Reconstruction des sources par la méthode de Prony (matrix pencil), à partir
de capteurs équirépartis sur un cercle de centre c et de rayon R.

Avec z_i = x_i - c (en complexe), rho_i = |z_i| < R, le noyau de Poisson donne
pour les coefficients de Fourier de t -> Phi(c + R e^{it}) :

    c_n = sum_i w_i u_i^n   (n >= 0),   u_i = conj(z_i) / R,
                                        w_i = I_i / (R^2 - rho_i^2)

C'est une somme d'exponentielles : le rang de la matrice de Hankel [c_{j+k}]
donne N, ses "valeurs propres de pencil" donnent les u_i, puis les w_i par
moindres carrés. Exact sans bruit ; très sensible au bruit (les c_n décroissent
comme (rho/R)^n).
"""

import numpy as np

from ..model import Sources
from ..sensors import CircleSensors


def fourier_coefficients(y: np.ndarray, sensors: CircleSensors) -> np.ndarray:
    """c_0, ..., c_{K//2 - 1} estimés par FFT (aliasing en (rho_max/R)^K)."""
    K = sensors.n
    n = np.arange(K // 2)
    c = np.fft.fft(y)[: K // 2] / K
    return c * np.exp(-1j * n * sensors.t0)


def estimate_rank(singular_values: np.ndarray, rank_tol: float) -> int:
    """Nombre de valeurs singulières au-dessus de rank_tol * s_max."""
    return int(np.sum(singular_values > rank_tol * singular_values[0]))


def prony_circle(
    y: np.ndarray,
    sensors: CircleSensors,
    n_sources: int | None = None,
    rank_tol: float = 1e-8,
) -> Sources:
    """
    y : mesures des capteurs (K,).
    n_sources : nombre de sources, ou None pour l'estimer par le rang
        (seuil rank_tol, à augmenter avec le niveau de bruit).
    """
    c = fourier_coefficients(y, sensors)
    L = len(c) - 1
    M = L // 2
    H = np.array([c[j : j + M + 1] for j in range(L - M + 1)])  # Hankel

    U, s, _ = np.linalg.svd(H)
    N = n_sources if n_sources is not None else estimate_rank(s, rank_tol)
    if N == 0:
        return Sources.empty()
    Us = U[:, :N]
    u = np.linalg.eigvals(np.linalg.pinv(Us[:-1]) @ Us[1:])

    V = u[None, :] ** np.arange(L + 1)[:, None]                 # Vandermonde
    w = np.linalg.lstsq(V, c, rcond=None)[0].real

    z = np.conj(u) * sensors.radius
    intensities = w * (sensors.radius**2 - np.abs(z) ** 2)
    positions = sensors.center + np.stack([z.real, z.imag], axis=-1)

    # On écarte les solutions non physiques (hors du cercle ou d'intensité <= 0)
    keep = (np.abs(u) < 1) & (intensities > 1e-9 * np.abs(intensities).max())
    return Sources(positions[keep], intensities[keep])

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
from .problem import InverseProblem


def fourier_coefficients(y: np.ndarray, sensors: CircleSensors) -> np.ndarray:
    """c_0, ..., c_{K//2 - 1} estimés par FFT (aliasing en (rho_max/R)^K)."""
    K = sensors.n
    n = np.arange(K // 2)
    c = np.fft.fft(y)[: K // 2] / K
    return c * np.exp(-1j * n * sensors.t0)


def estimate_rank(singular_values: np.ndarray, rank_tol: float) -> int:
    """Nombre de valeurs singulières au-dessus de rank_tol * s_max."""
    return int(np.sum(singular_values > rank_tol * singular_values[0]))


def noise_threshold(noise_std: np.ndarray, shape: tuple[int, int]) -> float:
    """
    Niveau des valeurs singulières dû au seul bruit : chaque c_n a un bruit
    d'écart-type ~ ||std|| / K, et la norme d'une matrice m x p de bruit
    i.i.d. vaut ~ std_c (sqrt(m) + sqrt(p)). Marge de sécurité x2.
    """
    std_c = np.sqrt(np.sum(noise_std**2)) / len(noise_std)
    return 2.0 * std_c * (np.sqrt(shape[0]) + np.sqrt(shape[1]))


def prony_circle(
    y: np.ndarray,
    sensors: CircleSensors,
    n_sources: int | None = None,
    rank_tol: float | None = 1e-8,
    noise_std: np.ndarray | None = None,
) -> Sources:
    """
    y : mesures des capteurs (K,).
    Nombre de sources N, par ordre de priorité :
        n_sources            imposé ;
        rank_tol             seuil relatif sur les valeurs singulières ;
        noise_std            (si rank_tol est None) seuil absolu déduit du
                             niveau de bruit des capteurs (voir noise_threshold).
    """
    c = fourier_coefficients(y, sensors)
    L = len(c) - 1
    M = L // 2
    H = np.array([c[j : j + M + 1] for j in range(L - M + 1)])  # Hankel

    U, s, _ = np.linalg.svd(H)
    if n_sources is not None:
        N = min(n_sources, len(s) - 1)
    elif rank_tol is not None or noise_std is None:
        N = estimate_rank(s, rank_tol if rank_tol is not None else 1e-8)
    else:
        N = int(np.sum(s > noise_threshold(noise_std, H.shape)))
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


def prony(y: np.ndarray, problem: InverseProblem, n_sources: int | None = None,
          rank_tol: float | None = None) -> Sources:
    """
    Méthode « prony » pour un InverseProblem : retire les sources connues, puis
    prony_circle. rank_tol None = seuil automatique d'après le modèle de bruit.
    """
    y_unknown = y - problem.known_contribution()
    return prony_circle(y_unknown, problem.sensors, n_sources, rank_tol,
                        noise_std=problem.noise.std(y))

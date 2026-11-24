"""
Méthode de Prony (matrix pencil), à partir de capteurs équirépartis sur un cercle
de centre c et de rayon R. Elle n'utilise QUE les capteurs du cercle.

Avec z_i = x_i − c (en complexe), ρ_i = |z_i| < R, le noyau de Poisson donne pour
les coefficients de Fourier de t ↦ Φ(c + R e^{it}) :

    c_n = Σ_i w_i u_i^n   (n ≥ 0),   u_i = conj(z_i) / R,   w_i = I_i / (R² − ρ_i²)

C'est une somme d'exponentielles : le rang de la matrice de Hankel [c_{j+k}] donne
N, ses « valeurs propres de pencil » donnent les u_i, puis les w_i par moindres
carrés. Exact sans bruit ; très sensible au bruit (les c_n décroissent comme (ρ/R)^n).

Étapes :
    1. FFT des mesures             → c_0, ..., c_L
    2. SVD de la matrice de Hankel → N (rang, seuil relatif rank_tol) et sous-espace signal
    3. invariance par décalage     → pôles u_i
    4. moindres carrés Vandermonde → poids w_i
    5. retour aux variables physiques : x_i = c + R conj(u_i),  I_i = w_i (R² − ρ_i²)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import probleme as P

RANK_TOL = 0.005          # seuil de rang du projet (≈ 5 σ)


@dataclass(frozen=True)
class Cercle:
    """n capteurs s_k = center + radius (cos t_k, sin t_k), t_k = t0 + 2π k / n."""
    center: np.ndarray
    radius: float
    n: int
    t0: float = 0.0


def fourier_coefficients(y: np.ndarray, sensors: Cercle) -> np.ndarray:
    """c_0, ..., c_{K/2 − 1} par FFT ; exp(−i n t0) compense l'angle du premier capteur."""
    K = sensors.n
    n = np.arange(K // 2)
    c = np.fft.fft(y)[: K // 2] / K
    return c * np.exp(-1j * n * sensors.t0)


def estimate_rank(singular_values: np.ndarray, rank_tol: float) -> int:
    """Nombre de valeurs singulières au-dessus de rank_tol × s_max (= nombre de sources)."""
    return int(np.sum(singular_values > rank_tol * singular_values[0]))


def prony_circle(y: np.ndarray, sensors: Cercle, n_sources: int | None = None,
                 rank_tol: float = 1e-8):
    """
    Reconstruit les sources à partir des mesures y (K,) d'un cercle de capteurs.
    n_sources : nombre de sources imposé, ou None pour l'estimer par le rang.
    Renvoie (positions (n, 2), intensités (n,)) ; les solutions non physiques
    (hors du cercle ou d'intensité ≤ 0) sont écartées.
    """
    # 1. Coefficients de Fourier et matrice de Hankel H[j, k] = c_{j+k}
    c = fourier_coefficients(y, sensors)
    L = len(c) - 1
    M = L // 2
    H = np.array([c[j: j + M + 1] for j in range(L - M + 1)])

    # 2. Nombre de sources et sous-espace signal
    U, s, _ = np.linalg.svd(H)
    N = n_sources if n_sources is not None else estimate_rank(s, rank_tol)
    if N == 0:
        return np.zeros((0, 2)), np.zeros(0)
    # 3. Pôles u_i : valeurs propres du décalage restreint au sous-espace signal
    Us = U[:, :N]
    u = np.linalg.eigvals(np.linalg.pinv(Us[:-1]) @ Us[1:])

    # 4. Poids w_i : c_n = Σ_i w_i u_i^n par moindres carrés
    V = u[None, :] ** np.arange(L + 1)[:, None]
    w = np.linalg.lstsq(V, c, rcond=None)[0].real

    # 5. Retour aux positions et intensités
    z = np.conj(u) * sensors.radius
    intensities = w * (sensors.radius**2 - np.abs(z) ** 2)
    positions = sensors.center + np.stack([z.real, z.imag], axis=-1)

    keep = (np.abs(u) < 1) & (intensities > 1e-9 * np.abs(intensities).max())
    return positions[keep], intensities[keep]


def estimer(sit: P.Situation) -> dict:
    """Prony sur les 64 capteurs du cercle (seuil de rang RANK_TOL), N estimé."""
    cercle = Cercle(P.CENTRE, P.RAYON, P.N_CERCLE, 0.0)
    z, I = prony_circle(sit.m[:P.N_CERCLE], cercle, rank_tol=RANK_TOL)
    return {"z": np.asarray(z, float).reshape(-1, 2), "I": np.asarray(I, float)}

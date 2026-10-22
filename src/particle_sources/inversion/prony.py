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

Étapes de `prony_circle` :
    1. FFT des mesures             -> c_0, ..., c_L
    2. SVD de la matrice de Hankel -> N (rang) et sous-espace signal
    3. invariance par décalage     -> pôles u_i
    4. moindres carrés Vandermonde -> poids w_i
    5. retour aux variables physiques :
       x_i = c + R conj(u_i),  I_i = w_i (R^2 - rho_i^2)
"""

import numpy as np

from ..model import Sources
from ..sensors import CircleSensors


def fourier_coefficients(y: np.ndarray, sensors: CircleSensors) -> np.ndarray:
    """
    c_0, ..., c_{K//2 - 1} estimés par FFT (aliasing en (rho_max/R)^K).

    Le facteur exp(-i n t0) compense l'angle du premier capteur, pour que les
    c_n soient ceux de t -> Phi(c + R e^{it}) avec t mesuré depuis l'est.
    """
    K = sensors.n
    n = np.arange(K // 2)
    c = np.fft.fft(y)[: K // 2] / K
    return c * np.exp(-1j * n * sensors.t0)


def estimate_rank(singular_values: np.ndarray, rank_tol: float) -> int:
    """
    Nombre de valeurs singulières au-dessus de rank_tol * s_max.

    Sert à estimer le nombre de sources N : au-delà de la N-ième, les valeurs
    singulières ne contiennent plus que du bruit (ou des erreurs d'arrondi).
    """
    return int(np.sum(singular_values > rank_tol * singular_values[0]))


def prony_circle(
    y: np.ndarray,
    sensors: CircleSensors,
    n_sources: int | None = None,
    rank_tol: float = 1e-8,
) -> Sources:
    """
    Reconstruit les sources à partir des mesures d'un cercle de capteurs.

    Paramètres
    ----------
    y : np.ndarray, forme (K,)
        Mesures des capteurs.
    sensors : CircleSensors
        Capteurs équirépartis ayant produit `y`.
    n_sources : int ou None
        Nombre de sources, ou None pour l'estimer par le rang
        (seuil rank_tol, à augmenter avec le niveau de bruit).
    rank_tol : float
        Seuil relatif sur les valeurs singulières de la matrice de Hankel.

    Renvoie
    -------
    Sources
        Sources reconstruites. Les solutions non physiques (hors du cercle ou
        d'intensité <= 0) sont écartées : on peut donc en obtenir moins que N.
    """
    # 1. Coefficients de Fourier et matrice de Hankel H[j, k] = c_{j+k}
    c = fourier_coefficients(y, sensors)
    L = len(c) - 1
    M = L // 2
    H = np.array([c[j : j + M + 1] for j in range(L - M + 1)])  # Hankel

    # 2. Nombre de sources et sous-espace signal
    U, s, _ = np.linalg.svd(H)
    N = n_sources if n_sources is not None else estimate_rank(s, rank_tol)
    if N == 0:
        return Sources.empty()
    # 3. Pôles u_i : valeurs propres du décalage restreint au sous-espace signal
    Us = U[:, :N]
    u = np.linalg.eigvals(np.linalg.pinv(Us[:-1]) @ Us[1:])

    # 4. Poids w_i : c_n = sum_i w_i u_i^n résolu par moindres carrés
    V = u[None, :] ** np.arange(L + 1)[:, None]                 # Vandermonde
    w = np.linalg.lstsq(V, c, rcond=None)[0].real

    # 5. Retour aux positions et intensités
    z = np.conj(u) * sensors.radius
    intensities = w * (sensors.radius**2 - np.abs(z) ** 2)
    positions = sensors.center + np.stack([z.real, z.imag], axis=-1)

    # On écarte les solutions non physiques (hors du cercle ou d'intensité <= 0)
    keep = (np.abs(u) < 1) & (intensities > 1e-9 * np.abs(intensities).max())
    return Sources(positions[keep], intensities[keep])

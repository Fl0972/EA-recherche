"""
Le problème : hangar, capteurs, source centrale, bruit, et tirage d'une situation.

Hangar D = [0, 10] × [0, 20]. On cherche N sources ponctuelles (z_i, I_i > 0), de somme
S = 100. Un capteur en s_k mesure

    Φ(s_k) = Σ_i I_i / ||s_k − z_i||²   +   I_c / ||s_k − c||²

où la seconde partie est une source centrale CONNUE d'intensité I_c, en c = (5, 10).
Bruit relatif : le capteur lit y_k^brut (1 + σ ε_k), ε_k ~ N(0, 1). On retire le signal
(connu) de la source centrale, mais son bruit reste : l'écart type au capteur k vaut
σ_k = σ |y_k^brut|, dominé par la source centrale quand I_c ≫ S.

Capteurs :
    "cercle" : 64 capteurs sur le cercle de centre c passant par les coins de D ;
    "grille" : ces 64 capteurs + une grille 16 × 32 dans D (576 capteurs).

Graines : positions, intensités, bruit ε et départ de la descente ne dépendent que de
(N, tirage) : toutes les méthodes, toutes les dispositions de capteurs et toutes les
valeurs de σ et I_c voient les mêmes sources et le même ε.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .noyau import Box

LO = np.array([0.0, 0.0])
HI = np.array([10.0, 20.0])
DOMAINE = Box(LO, HI)
CENTRE = 0.5 * (LO + HI)
RAYON = float(np.linalg.norm(HI - LO) / 2)        # cercle passant par les coins de D
S_TOTAL = 100.0
N_CERCLE = 64
GRILLE = (16, 32)
SIGMA_VIDE = 1e-6        # σ = 0 : ML et BLASSO ont besoin d'un niveau de bruit non nul
CAPTEURS = ["cercle", "grille"]


# ---- capteurs ---------------------------------------------------------------------------
def capteurs_cercle(n: int = N_CERCLE) -> np.ndarray:
    """n capteurs équirépartis sur le cercle circonscrit à D (premier capteur à l'est)."""
    t = 2 * np.pi * np.arange(n) / n
    return CENTRE + RAYON * np.stack([np.cos(t), np.sin(t)], 1)


def capteurs_grille(nx: int = GRILLE[0], ny: int = GRILLE[1]) -> np.ndarray:
    """Grille nx × ny de capteurs dans D (centres des cases)."""
    xs = LO[0] + (np.arange(nx) + 0.5) * (HI[0] - LO[0]) / nx
    ys = LO[1] + (np.arange(ny) + 0.5) * (HI[1] - LO[1]) / ny
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)


def capteurs(nom: str) -> np.ndarray:
    """Positions des capteurs ; les 64 premiers sont toujours ceux du cercle."""
    if nom == "cercle":
        return capteurs_cercle()
    if nom == "grille":
        return np.vstack([capteurs_cercle(), capteurs_grille()])
    raise ValueError(f"capteurs inconnus : {nom!r} (cercle ou grille)")


def champ(X, z, I) -> np.ndarray:
    """Φ(s_k) = Σ_i I_i / ||s_k − z_i||² en chaque capteur de X."""
    z = np.asarray(z, float).reshape(-1, 2)
    if len(z) == 0:
        return np.zeros(len(X))
    d2 = ((X[:, None, :] - z[None]) ** 2).sum(-1)
    return (1.0 / d2) @ np.asarray(I, float)


# ---- situation --------------------------------------------------------------------------
def rng_de(N: int, tirage: int, flux: int) -> np.random.Generator:
    """Générateur reproductible pour (N, tirage) et un flux (0 situation, 1 bruit, 2 départ)."""
    return np.random.default_rng([int(N), int(tirage), int(flux), 20261123])


@dataclass
class Situation:
    z: np.ndarray          # (N, 2) vraies positions
    I: np.ndarray          # (N,) vraies intensités
    X: np.ndarray          # (K, 2) capteurs
    m: np.ndarray          # (K,) mesures, source centrale retirée
    s: np.ndarray          # (K,) écart type du bruit par capteur
    y_brut: np.ndarray     # (K,) mesures brutes bruitées (source centrale comprise)
    u0: np.ndarray         # (N, 2) départ de la descente (coordonnées normalisées dans D)
    N: int
    sigma: float
    ic: float              # intensité de la source centrale
    capteurs: str
    tirage: int
    x: float | None = None  # rapport d'intensité I_1 / I_2 (cas à deux sources)


def situation(N: int, tirage: int, sigma: float, ic_ratio: float, nom_capteurs: str,
              x: float | None = None, z=None, I=None) -> Situation:
    """
    Situation (N, tirage) observée avec le bruit relatif σ et la source centrale
    I_c = ic_ratio × S, par les capteurs ``nom_capteurs``.

    Par défaut positions uniformes dans D et intensités S · Dirichlet(1, …, 1).
    x : cas à deux sources, intensités imposées I = S (x, 1) / (x + 1) (N = 2).
    z, I : sources imposées à la main (N est alors len(I)).
    """
    if z is not None:
        z = np.asarray(z, float).reshape(-1, 2)
        I = np.asarray(I, float).reshape(-1)
        N = len(I)
    r0 = rng_de(N, tirage, 0)
    z_t = LO + (HI - LO) * r0.random((N, 2))
    I_t = S_TOTAL * r0.dirichlet(np.ones(N))
    if z is None:
        z, I = z_t, I_t
    if x is not None:
        if N != 2:
            raise ValueError("le rapport d'intensité x n'a de sens qu'avec N = 2")
        I = S_TOTAL * np.array([x, 1.0]) / (x + 1.0)
    eps = rng_de(N, tirage, 1).standard_normal(N_CERCLE + GRILLE[0] * GRILLE[1])
    u0 = rng_de(N, tirage, 2).random((N, 2))
    X = capteurs(nom_capteurs)
    ic = float(ic_ratio) * S_TOTAL
    yc = ic / ((X - CENTRE) ** 2).sum(1) if ic > 0 else np.zeros(len(X))
    y_brut = (champ(X, z, I) + yc) * (1 + sigma * eps[:len(X)])
    m = y_brut - yc
    s_eff = sigma if sigma > 0 else SIGMA_VIDE
    s = np.maximum(s_eff * np.abs(y_brut), 1e-12 * max(float(np.abs(y_brut).max()), 1.0))
    return Situation(z, I, X, m, s, y_brut, u0, int(N), float(sigma), ic, nom_capteurs,
                     int(tirage), None if x is None else float(x))

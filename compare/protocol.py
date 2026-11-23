"""
Protocole commun de la comparaison des méthodes (codé une seule fois).

Situation
    N sources, positions uniformes dans D = [0,10] × [0,20], intensités
    I = S · Dirichlet(1, …, 1) (somme S = 100), source centrale d'intensité
    I_c en c = (5, 10), bruit relatif σ, disposition des capteurs.
    Mesures : y_k^brut = Φ_θ(s_k) + I_c / ||s_k − c||², bruitées en
    y_k^brut (1 + σ ε_k) ; la contribution (connue) de la source centrale est
    retirée : m_k = y_k^brut (1 + σ ε_k) − I_c / ||s_k − c||² ; écart type du
    bruit σ_k = σ |y_k^brut (1 + σ ε_k)| (comme ``relative_noise_std``).

Graines (« mêmes situations et mêmes graines pour toutes les méthodes »)
    Les positions, les intensités, les ε_k et l'initialisation de la descente
    ne dépendent que de (N, tirage) : ni de σ, ni de I_c, ni de la
    disposition des capteurs, ni de la méthode. Pour un tirage donné, les
    64 capteurs du cercle ont donc exactement les mêmes mesures dans les deux
    configurations, et quand σ varie seule l'amplitude du bruit change
    (nombres aléatoires communs : courbes moins bruitées).

Méthodes (code inchangé du dépôt)
    prony      ``particle_sources.inversion.prony_circle`` (seuil de rang de
               config/reconstruction.yaml), capteurs du cercle seulement ;
    ml_bic     ``sparse_sources.ml.greedy_ml`` (maximum de vraisemblance
               blanchi, glouton, K par BIC), k_max = 20 ;
    fw         ``sparse_sources.blasso.sliding_frank_wolfe`` sur le noyau
               blanchi et normalisé (α = 1), λ = 1,5 √K_capteurs, au plus
               40 itérations (même budget pour tout N) ;
    descente   ``sparse_sources.orano.descente_projetee`` (méthode d'Orano),
               N connu, positions initiales uniformes dans D, intensités S/N,
               20 000 itérations au plus.

Erreur d'un tirage
    Appariement hongrois sur la distance euclidienne. Pour chaque vraie
    source i : d_i = ||ẑ_i − z_i||, r_i = |Î_i − I_i| / I_i, plafonnés à
    d_max = 5 et r_max = 1 ; une vraie source manquée compte (5, 1).
    E = (1/N) Σ_i (α d_i + β r_i). Les sources en trop (fausses) ne changent
    pas E et sont comptées à part.
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

LO = np.array([0.0, 0.0])
HI = np.array([10.0, 20.0])
CENTRE = 0.5 * (LO + HI)
RAYON = float(np.linalg.norm(HI - LO) / 2)        # cercle passant par les coins de D
S_TOTAL = 100.0
D_MAX, R_MAX = 5.0, 1.0                           # plafonds de d_i et r_i
N_CERCLE = 64
GRILLE = (16, 32)

METHODES = ["prony", "ml_bic", "fw", "descente"]
NOMS = {"prony": "Prony", "ml_bic": "ML + BIC", "fw": "Frank–Wolfe glissant",
        "descente": "Descente de gradient (N connu)"}
CONFIGS = ["cercle", "grille"]
NOMS_CONFIG = {"cercle": "cercle (64 capteurs)",
               "grille": "cercle + grille (64 + 512 capteurs)"}


# ---- capteurs ---------------------------------------------------------------------------
def capteurs_cercle(n=N_CERCLE):
    """n capteurs équirépartis sur le cercle de centre c passant par les coins de D (t0 = 0)."""
    t = 2 * np.pi * np.arange(n) / n
    return CENTRE + RAYON * np.stack([np.cos(t), np.sin(t)], 1)


def capteurs_grille(nx=GRILLE[0], ny=GRILLE[1]):
    """Grille nx × ny de capteurs dans D (centres des cases)."""
    xs = LO[0] + (np.arange(nx) + 0.5) * (HI[0] - LO[0]) / nx
    ys = LO[1] + (np.arange(ny) + 0.5) * (HI[1] - LO[1]) / ny
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)


@lru_cache(maxsize=None)
def capteurs(config: str) -> np.ndarray:
    """Positions des capteurs ; les 64 premiers sont toujours ceux du cercle."""
    if config == "cercle":
        return capteurs_cercle()
    if config == "grille":
        return np.vstack([capteurs_cercle(), capteurs_grille()])
    raise ValueError(config)


# ---- situations -------------------------------------------------------------------------
def rng_de(N, tirage, flux):
    """Générateur reproductible pour (N, tirage) et un flux (0 situation, 1 bruit, 2 init)."""
    return np.random.default_rng([int(N), int(tirage), int(flux), 20261123])


@dataclass
class Situation:
    z: np.ndarray          # (N, 2) vraies positions
    I: np.ndarray          # (N,) vraies intensités
    X: np.ndarray          # (K, 2) capteurs
    m: np.ndarray          # (K,) mesures, source centrale retirée
    s: np.ndarray          # (K,) écart type du bruit par capteur
    y_brut: np.ndarray     # (K,) mesures brutes bruitées (centrale comprise)
    u0: np.ndarray         # (N, 2) initialisation normalisée de la descente
    N: int
    sigma: float
    ic: float              # intensité de la source centrale


def champ(X, z, I):
    """Φ_θ(s_k) = Σ_i I_i / ||s_k − z_i||²."""
    z = np.asarray(z, float).reshape(-1, 2)
    if len(z) == 0:
        return np.zeros(len(X))
    d2 = ((X[:, None, :] - z[None]) ** 2).sum(-1)
    return (1.0 / d2) @ np.asarray(I, float)


def situation(N, tirage, sigma, ic_ratio, config) -> Situation:
    """Tirage (N, tirage) observé avec le bruit σ, la source centrale I_c = ic_ratio × S."""
    r0 = rng_de(N, tirage, 0)
    z = LO + (HI - LO) * r0.random((N, 2))
    I = S_TOTAL * r0.dirichlet(np.ones(N))
    eps = rng_de(N, tirage, 1).standard_normal(N_CERCLE + GRILLE[0] * GRILLE[1])
    u0 = rng_de(N, tirage, 2).random((N, 2))
    X = capteurs(config)
    ic = float(ic_ratio) * S_TOTAL
    yc = ic / ((X - CENTRE) ** 2).sum(1) if ic > 0 else np.zeros(len(X))
    y_brut = (champ(X, z, I) + yc) * (1 + sigma * eps[:len(X)])
    m = y_brut - yc
    s = np.maximum(sigma * np.abs(y_brut), 1e-12 * max(float(np.abs(y_brut).max()), 1.0))
    return Situation(z, I, X, m, s, y_brut, u0, int(N), float(sigma), ic)


# ---- objets coûteux, construits une fois par processus ------------------------------
@lru_cache(maxsize=None)
def setup(config: str):
    """Setup de sparse_sources (noyau normalisé α = 1, comme dans le reste du dépôt)."""
    from particle_sources.model import Box
    from sparse_sources.kernel import Setup
    return Setup(capteurs(config), Box.from_bounds([[LO[0], HI[0]], [LO[1], HI[1]]]), 0.0,
                 None, True, None)


@lru_cache(maxsize=None)
def grille_recherche(config: str):
    """Grille de recherche (pas 0,25) dont aucun nœud ne tombe sur un capteur."""
    from sparse_sources.tester_ml import search_grid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return search_grid(setup(config))


@lru_cache(maxsize=None)
def capteurs_prony():
    from particle_sources.sensors import CircleSensors
    return CircleSensors(CENTRE, RAYON, N_CERCLE, 0.0)


@lru_cache(maxsize=None)
def rank_tol_prony():
    from particle_sources.config import load_config
    from particle_sources.config.cli import DEFAULT_CONFIG_DIR
    import yaml
    with open(Path(DEFAULT_CONFIG_DIR) / "reconstruction.yaml", encoding="utf-8") as f:
        return float(yaml.safe_load(f)["reconstruction"]["method"]["params"]["rank_tol"])


# ---- méthodes ---------------------------------------------------------------------------
def lancer(methode: str, sit: Situation, config: str):
    """Renvoie (ẑ (n,2), Î (n,), infos) ; le temps est mesuré par l'appelant."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if methode == "prony":
            from particle_sources.inversion import prony_circle
            est = prony_circle(sit.m[:N_CERCLE], capteurs_prony(), rank_tol=rank_tol_prony())
            return np.asarray(est.positions, float).reshape(-1, 2), np.asarray(est.intensities, float), {}
        if methode == "ml_bic":
            from sparse_sources.ml import greedy_ml
            r = greedy_ml(sit.m, sit.s, setup=setup(config), grid=grille_recherche(config),
                          k_max=20)
            return np.asarray(r["z"]).reshape(-1, 2), np.asarray(r["I"]), {"chi2": r["chi2"]}
        if methode == "fw":
            from sparse_sources.blasso import sliding_frank_wolfe
            from sparse_sources.common import Grid
            S = setup(config).with_weights(1.0 / sit.s)
            g = grille_recherche(config)
            gw = Grid(g.xs, g.ys, g.points, g.step, S.phi(g.flat).T)
            lam = 1.5 * np.sqrt(len(sit.m))
            fw = sliding_frank_wolfe(S.whiten(sit.m), lam, setup=S, grid=gw, tol=1e-5,
                                     max_iter=40)
            z = np.atleast_2d(fw["z_debiased"]).reshape(-1, 2)
            I = S.physical_intensities(z, fw["I_debiased"]) if len(z) else np.zeros(0)
            return z, np.asarray(I, float), {}
        if methode == "descente":
            from sparse_sources.orano import Probleme, descente_projetee
            pb = Probleme(sit.X, LO, HI, sit.m)
            r = descente_projetee(pb, sit.u0, np.full(sit.N, 1.0 / sit.N), iter_max=20000)
            return r["p"], r["I"], {"iterations": r["iterations"], "perte": r["perte"]}
    raise ValueError(methode)


# ---- erreur -----------------------------------------------------------------------------
def erreur(z, I, z_est, I_est, alpha=0.75, beta=0.25):
    """
    Appariement hongrois, erreurs plafonnées. Renvoie un dict :
    E, d_moy, d_max, r_moy, n_est, fausses, manquees.
    """
    z = np.asarray(z, float).reshape(-1, 2)
    z_est = np.asarray(z_est, float).reshape(-1, 2)
    I_est = np.asarray(I_est, float).reshape(-1)
    N, n = len(z), len(z_est)
    d = np.full(N, D_MAX)
    r = np.full(N, R_MAX)
    if n and N:
        C = np.linalg.norm(z[:, None] - z_est[None], axis=-1)
        a, b = linear_sum_assignment(C)
        d[a] = np.minimum(C[a, b], D_MAX)
        r[a] = np.minimum(np.abs(I_est[b] - I[a]) / I[a], R_MAX)
    E = float(np.mean(alpha * d + beta * r))
    return {"E": E, "d_moy": float(d.mean()), "d_max": float(d.max()), "r_moy": float(r.mean()),
            "n_est": n, "fausses": max(n - N, 0), "manquees": max(N - n, 0)}


def erreur_timeout(N, alpha=0.75, beta=0.25):
    """Tirage en timeout : toutes les sources manquées."""
    return {"E": alpha * D_MAX + beta * R_MAX, "d_moy": D_MAX, "d_max": D_MAX, "r_moy": R_MAX,
            "n_est": 0, "fausses": 0, "manquees": int(N)}


# ---- une tâche ------------------------------------------------------------------------
def tache(t: dict) -> dict:
    """
    t = {config, N, sigma, ic_ratio, tirage, methode, alpha, beta}.
    Renvoie la ligne de résultat (sans le statut de timeout, géré par le pool).
    """
    sit = situation(t["N"], t["tirage"], t["sigma"], t["ic_ratio"], t["config"])
    t0 = time.perf_counter()
    try:
        z, I, info = lancer(t["methode"], sit, t["config"])
        statut = "ok"
    except Exception as e:                     # une méthode qui plante compte comme échec
        z, I, info = np.zeros((0, 2)), np.zeros(0), {"erreur": repr(e)[:200]}
        statut = "exception"
    dt = time.perf_counter() - t0
    row = {**{k: t[k] for k in ("config", "N", "sigma", "ic_ratio", "tirage", "methode")},
           "statut": statut, "temps": dt,
           **erreur(sit.z, sit.I, z, I, t["alpha"], t["beta"])}
    row["N_exact"] = bool(row["n_est"] == t["N"])
    for k in ("iterations", "perte", "chi2"):
        if k in info:
            row[k] = info[k]
    if "erreur" in info:
        row["message"] = info["erreur"]
    return row

"""
Simulateur rapide du scénario du projet, pour fabriquer des bases
d'apprentissage (des milliers de tirages en quelques secondes) et pour
changer la géométrie (capteurs, taille du domaine).

Par défaut (``Scene()``) il reproduit exactement le modèle de
``experiments_projet.scenario`` :
    - n sources, positions uniformes dans D = [0,10] × [0,20],
      intensités uniformes sur le simplexe de somme 100 ;
    - source centrale d'intensité 5000 au centre de D ;
    - 64 capteurs sur le cercle circonscrit à D ;
    - mesures y_i = Σ I_k / ||x_i - z_k||² (+ centrale), bruit relatif
      y (1 + σ ε) ;
    - m = y - (contribution de la source centrale), s_i = σ |y_i|.
Seul le flux aléatoire diffère (on n'utilise pas load_config, trop lent).
Vérifié : mêmes capteurs, même m = y - centrale (écart 1e-14).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field as dc_field
from functools import cached_property
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from particle_sources.model import Box  # noqa: E402
from sparse_sources import build_grid, make_setup  # noqa: E402
from sparse_sources.kernel import Setup  # noqa: E402

_DEFAULT = make_setup(h=0.0, normalize=True)


def circle_sensors(bounds, n=64, radius_factor=1.0, t0=0.0):
    """n capteurs sur le cercle de centre celui de D et de rayon facteur × rayon circonscrit."""
    b = np.asarray(bounds, float)
    c = b.mean(1)
    r = radius_factor * 0.5 * np.linalg.norm(b[:, 1] - b[:, 0])
    t = t0 + 2 * np.pi * np.arange(n) / n
    return c + r * np.stack([np.cos(t), np.sin(t)], 1)


@dataclass(eq=False)
class Scene:
    """
    Géométrie + loi des sources. Tous les paramètres ont la valeur du projet.

    sensors : (M, 2) ou None (64 capteurs sur le cercle circonscrit à D)
    bounds  : [[xmin, xmax], [ymin, ymax]]
    central : intensité de la source centrale (0 : pas de source centrale)
    central_pos : position (défaut : centre de D)
    """
    bounds: list = dc_field(default_factory=lambda: [[0.0, 10.0], [0.0, 20.0]])
    sensors: np.ndarray | None = None
    central: float = 5000.0
    central_pos: np.ndarray | None = None
    total: float = 100.0
    grid_step: float = 0.25

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, float)
        if self.sensors is None:
            self.sensors = circle_sensors(self.bounds)
        self.sensors = np.asarray(self.sensors, float)
        self.lo, self.hi = self.bounds[:, 0], self.bounds[:, 1]
        if self.central_pos is None:
            self.central_pos = 0.5 * (self.lo + self.hi)
        self.y_central = self.field(np.atleast_2d(self.central_pos), [self.central]) \
            if self.central else np.zeros(self.M)

    @property
    def M(self):
        """Nombre de capteurs."""
        return len(self.sensors)

    @cached_property
    def setup(self) -> Setup:
        """Setup de sparse_sources (noyau normalisé pour le choix des Diracs)."""
        return Setup(self.sensors, Box.from_bounds(self.bounds.tolist()), 0.0,
                     None, True, None)

    @cached_property
    def grid(self):
        """Grille de recherche, pas ``grid_step`` (adapté à la taille de D)."""
        step = self.grid_step * float(min(self.hi - self.lo)) / 10.0
        return build_grid(step=step, setup=self.setup)

    def field(self, z, I):
        """Mesures exactes Σ I_k / ||x_i - z_k||², z (K,2), I (K,) -> (M,)."""
        z = np.asarray(z, float).reshape(-1, 2)
        if len(z) == 0:
            return np.zeros(self.M)
        d2 = ((self.sensors[:, None, :] - z[None]) ** 2).sum(-1)
        return (1.0 / d2) @ np.asarray(I, float)

    def draw(self, n, sigma, rng):
        """
        Un tirage. Renvoie un dict : z (n,2), I (n,), y (mesures brutes),
        m (centrale retirée), s (écart type du bruit par capteur), sigma, n.
        """
        z = self.lo + (self.hi - self.lo) * rng.random((n, 2))
        I = self.total * rng.dirichlet(np.ones(n)) if n else np.zeros(0)
        return self.measure(z, I, sigma, rng)

    def measure(self, z, I, sigma, rng):
        """Mesures bruitées de sources données (même dict que ``draw``)."""
        z = np.asarray(z, float).reshape(-1, 2)
        I = np.asarray(I, float).reshape(-1)
        y0 = self.field(z, I) + self.y_central
        y = y0 * (1 + sigma * rng.standard_normal(self.M))
        m = y - self.y_central
        s = np.maximum(sigma * np.abs(y), 1e-12 * max(np.abs(y).max(), 1.0))
        return {"z": z, "I": I, "y": y, "m": m, "s": s, "sigma": sigma, "n": len(I)}

    def dataset(self, n_samples, rng, k_values=(1, 2, 3, 4, 5),
                sigmas=(1e-4, 3e-4, 1e-3, 3e-3)):
        """Liste de tirages : K uniforme dans k_values, σ uniforme dans sigmas."""
        return [self.draw(int(rng.choice(k_values)), float(rng.choice(sigmas)), rng)
                for _ in range(n_samples)]


PROJET = Scene()
assert np.allclose(PROJET.sensors, _DEFAULT.sensors), "capteurs différents du projet"


def whitened(d):
    """Mesures blanchies m / s : l'entrée « naturelle » des modèles appris."""
    return d["m"] / d["s"]

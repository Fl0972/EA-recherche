"""
Noyau du modèle direct et ses dérivées, vu comme fonction de la position z
de la source.

Le noyau est celui du projet (``particle_sources.model.field.intensity_at``,
dérivées analytiques de ``particle_sources.duval_peyre.kernel``) :

    G(x_i, z) = 1 / (||x_i - z||^2 + h^2)

- x_i ∈ R² : projection du capteur i sur le plan des sources ;
- h ≥ 0    : hauteur des capteurs au-dessus du plan des sources. Un capteur à
  la hauteur h voit une source du plan à la distance sqrt(||x_i - z||^2 + h^2),
  d'où le noyau ci-dessus. C'est exactement le paramètre ``eps`` du projet
  (h = 0 : capteurs dans le plan, loi exacte en 1/r²).

    φ(z)      = (G(x_1, z), ..., G(x_M, z))        ∈ R^M
    ∇_z φ(z)  : ligne i = -2 (z - x_i) / q_i²     ∈ R^{M×2},  q_i = ||z - x_i||² + h²
    ∇²_z φ(z) : bloc i  = -2 Id / q_i² + 8 (z - x_i)(z - x_i)^T / q_i³   ∈ R^{M×2×2}

Un ``Setup`` regroupe tout ce qui définit l'opérateur A : capteurs, hauteur h
et domaine D. Par défaut il est lu dans ``config/`` (domaine de
``simulation.yaml``, capteurs de ``sensors.yaml``), pour être le même que
dans le reste du projet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from particle_sources.config import load_config
from particle_sources.config.cli import DEFAULT_CONFIG_DIR
from particle_sources.duval_peyre.kernel import InverseSquareKernel
from particle_sources.duval_peyre.kernel import check_derivatives as _check_derivatives
from particle_sources.model import Box
from particle_sources.sensors import CircleSensors


@dataclass(frozen=True)
class Setup:
    """
    Opérateur direct A et domaine D.

    Attributs
    ---------
    sensors : np.ndarray, forme (M, 2)
        Positions x_i des capteurs (projetées sur le plan des sources).
    domain : Box
        Domaine D des sources (rectangle).
    h : float
        Hauteur des capteurs au-dessus du plan (``eps`` du projet).
    layout : objet ou None
        Réseau de capteurs d'origine (``CircleSensors``), utile pour les
        méthodes qui en ont besoin (Prony). Facultatif.
    """
    sensors: np.ndarray
    domain: Box
    h: float = 0.0
    layout: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        """Convertit les capteurs en tableau float (M, 2)."""
        object.__setattr__(self, "sensors", np.atleast_2d(np.asarray(self.sensors, float)))
        if self.domain.dim != 2 or self.sensors.shape[1] != 2:
            raise ValueError("sparse_sources travaille en 2D (sources dans un plan)")

    @cached_property
    def kernel(self) -> InverseSquareKernel:
        """Noyau du projet, avec eps = h."""
        return InverseSquareKernel(self.sensors, eps=self.h)

    @property
    def M(self) -> int:
        """Nombre de capteurs M."""
        return len(self.sensors)

    # Raccourcis vers le noyau -------------------------------------------------------
    def phi(self, z) -> np.ndarray:
        """φ(z), forme (..., M) pour z de forme (..., 2)."""
        return self.kernel.phi(z)

    def grad(self, z) -> np.ndarray:
        """∇_z φ(z), forme (..., M, 2)."""
        return self.kernel.grad(z)

    def hess(self, z) -> np.ndarray:
        """∇²_z φ(z), forme (..., M, 2, 2)."""
        return self.kernel.hess(z)


def make_setup(config_dir=None, n_sensors: int | None = None, h: float | None = None,
               radius_factor: float = 1.0, domain=None, overrides=()) -> Setup:
    """
    Construit un ``Setup`` à partir de la configuration du projet.

    Paramètres
    ----------
    config_dir : chemin ou None
        Dossier de YAML (défaut : ``config/`` du dépôt).
    n_sensors : int ou None
        Nombre de capteurs ; None = celui de la config (64). Les capteurs sont
        alors replacés sur le cercle circonscrit à D (comme dans la config).
    h : float ou None
        Hauteur des capteurs ; None = ``sensors.eps`` de la config (0).
    radius_factor : float
        Rayon du cercle de capteurs = radius_factor × rayon du cercle
        circonscrit à D (1 = config du projet ; > 1 éloigne les capteurs).
    domain : [[xmin, xmax], [ymin, ymax]] ou None
        Domaine ; None = celui de la config.
    overrides : itérable de "clé=valeur"
        Surcharges passées à ``load_config``.
    """
    cfg = load_config(config_dir or DEFAULT_CONFIG_DIR, overrides)
    box = Box.from_bounds(domain) if domain is not None else cfg.box
    layout = cfg.sensors
    if n_sensors is not None or radius_factor != 1.0 or domain is not None:
        n = n_sensors if n_sensors is not None else layout.n
        t0 = getattr(layout, "t0", 0.0)
        layout = CircleSensors(box.center, radius_factor * box.circumradius, n, t0)
    return Setup(layout.positions, box, cfg.eps if h is None else float(h), layout)


_DEFAULT: Setup | None = None


def get_default_setup() -> Setup:
    """Setup utilisé quand une fonction reçoit ``setup=None`` (config du projet)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = make_setup()
    return _DEFAULT


def set_default_setup(setup: Setup) -> None:
    """Change le Setup par défaut (par ex. autres capteurs ou hauteur h)."""
    global _DEFAULT
    _DEFAULT = setup


def resolve(setup: Setup | None) -> Setup:
    """``setup`` s'il est donné, sinon le Setup par défaut."""
    return setup if setup is not None else get_default_setup()


# Fonctions ---------------------------------------------------------------------------
def phi(z, setup: Setup | None = None) -> np.ndarray:
    """φ(z) = (G(x_i, z))_i, forme (..., M)."""
    return resolve(setup).phi(z)


def grad_phi(z, setup: Setup | None = None) -> np.ndarray:
    """∇_z φ(z), forme (..., M, 2) : [i, a] = ∂G(x_i, z)/∂z_a."""
    return resolve(setup).grad(z)


def hess_phi(z, setup: Setup | None = None) -> np.ndarray:
    """∇²_z φ(z), forme (..., M, 2, 2)."""
    return resolve(setup).hess(z)


def check_derivatives(z, setup: Setup | None = None, step: float = 1e-5) -> dict:
    """
    Erreurs relatives max entre dérivées analytiques et différences finies
    centrées (pas ``step``) au point z. Attendu : ~1e-8 ou moins.
    """
    return _check_derivatives(resolve(setup).kernel, np.asarray(z, float), step)

"""
Modélisation de l'apparition aléatoire de sources émettant des particules.

Étapes :
  1. Nombre de sources N  ~ loi discrète modulaire (Poisson par défaut).
  2. Positions des sources ~ uniformes dans le domaine.
  3. Intensités (I_1, ..., I_N) ~ uniforme sur le simplexe  {I_i >= 0, sum I_i = I_tot}
     (c.-à-d. I = I_tot * Dirichlet(1, ..., 1)).
  4. Intensité totale mesurée en x :
         Phi(x) = sum_i I_i / ||x - x_i||^2
     (avec une régularisation optionnelle eps : I_i / (||x - x_i||^2 + eps^2)
      pour éviter la singularité sur les sources).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# --------------------------------------------------------------------------
# 1. Lois du nombre de sources (modulaires : toute fonction rng -> int)
# --------------------------------------------------------------------------
CountLaw = Callable[[np.random.Generator], int]


def poisson_law(lam: float) -> CountLaw:
    return lambda rng: int(rng.poisson(lam))


def binomial_law(n: int, p: float) -> CountLaw:
    return lambda rng: int(rng.binomial(n, p))


def negative_binomial_law(r: float, p: float) -> CountLaw:
    """Utile pour une sur-dispersion par rapport à Poisson."""
    return lambda rng: int(rng.negative_binomial(r, p))


def fixed_law(n: int) -> CountLaw:
    return lambda rng: n


def at_least_one(law: CountLaw) -> CountLaw:
    """Conditionne la loi à N >= 1 (par rejet)."""
    def sample(rng):
        while (n := law(rng)) == 0:
            pass
        return n
    return sample


# --------------------------------------------------------------------------
# 2-3. Tirage d'une configuration de sources
# --------------------------------------------------------------------------
def uniform_simplex(n: int, total: float, rng: np.random.Generator) -> np.ndarray:
    """Tirage uniforme sur le simplexe {x_i >= 0, sum x_i = total}."""
    if n == 0:
        return np.empty(0)
    return total * rng.dirichlet(np.ones(n))


@dataclass
class SourceField:
    positions: np.ndarray    # (N, d)
    intensities: np.ndarray  # (N,)
    eps: float = 0.0         # rayon de régularisation

    @property
    def n_sources(self) -> int:
        return len(self.intensities)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Intensité totale Phi(x) = sum_i I_i / (||x - x_i||^2 + eps^2).
        x : tableau de forme (..., d). Renvoie un tableau de forme (...).
        """
        x = np.asarray(x, dtype=float)
        if self.n_sources == 0:
            return np.zeros(x.shape[:-1])
        diff = x[..., None, :] - self.positions          # (..., N, d)
        r2 = np.sum(diff**2, axis=-1) + self.eps**2      # (..., N)
        with np.errstate(divide="ignore"):
            return np.sum(self.intensities / r2, axis=-1)


def sample_sources(
    count_law: CountLaw,
    total_intensity: float,
    domain: np.ndarray,
    rng: np.random.Generator,
    eps: float = 0.0,
) -> SourceField:
    """
    domain : tableau (d, 2) des bornes [min, max] par dimension.
    """
    domain = np.asarray(domain, dtype=float)
    n = count_law(rng)
    positions = rng.uniform(domain[:, 0], domain[:, 1], size=(n, len(domain)))
    intensities = uniform_simplex(n, total_intensity, rng)
    return SourceField(positions, intensities, eps)


# --------------------------------------------------------------------------
# Visualisation (2D)
# --------------------------------------------------------------------------
def plot_field(field: SourceField, domain, resolution: int = 400, ax=None):
    domain = np.asarray(domain, dtype=float)
    xs = np.linspace(*domain[0], resolution)
    ys = np.linspace(*domain[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = field(np.stack([X, Y], axis=-1))

    ax = ax or plt.gca()
    finite = Z[np.isfinite(Z) & (Z > 0)]
    if finite.size:
        norm = LogNorm(vmin=finite.min(), vmax=np.percentile(finite, 99.5))
        im = ax.pcolormesh(X, Y, Z, norm=norm, shading="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, label=r"$\Phi(x)$")
    if field.n_sources:
        sizes = 200 * field.intensities / field.intensities.max()
        ax.scatter(*field.positions.T, s=sizes, c="red", edgecolors="white",
                   label="sources (taille ∝ intensité)")
        ax.legend(loc="upper right")
    ax.set_title(f"{field.n_sources} sources — intensité totale = "
                 f"{field.intensities.sum():.3g}")
    ax.set_aspect("equal")
    return ax


# --------------------------------------------------------------------------
# Mesure le long du cercle circonscrit au domaine
# --------------------------------------------------------------------------
def circumscribed_circle(domain):
    """Centre et rayon du cercle dans lequel le rectangle `domain` est inscrit."""
    domain = np.asarray(domain, dtype=float)
    center = domain.mean(axis=1)
    radius = 0.5 * np.linalg.norm(domain[:, 1] - domain[:, 0])
    return center, radius


def circle_profile(field: SourceField, domain, n_points: int = 1000):
    """
    t in [0, 2pi) -> Phi(c + R (cos t, sin t)), t = 0 à l'est, sens trigonométrique.
    Renvoie (t, Phi(t)).
    """
    center, radius = circumscribed_circle(domain)
    t = np.linspace(0, 2 * np.pi, n_points)
    pts = center + radius * np.stack([np.cos(t), np.sin(t)], axis=-1)
    return t, field(pts)


def plot_field_and_profile(field: SourceField, domain, resolution: int = 400):
    center, radius = circumscribed_circle(domain)
    # Zone tracée = carré englobant le cercle, pour voir le cercle en entier
    extent = np.stack([center - radius, center + radius], axis=1)

    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(14, 6))
    plot_field(field, extent, resolution, ax=ax_map)

    t, phi = circle_profile(field, domain)
    lo, hi = np.asarray(domain, dtype=float).T
    ax_map.add_patch(plt.Rectangle(lo, *(hi - lo),
                                   fill=False, ec="white", ls="--", lw=1))
    ax_map.plot(center[0] + radius * np.cos(t), center[1] + radius * np.sin(t),
                c="orange", lw=2)
    ax_map.plot(center[0] + radius, center[1], "o", c="orange")
    ax_map.annotate("t = 0", (center[0] + radius, center[1]),
                    xytext=(-40, 6), textcoords="offset points", c="orange")

    ax_prof.plot(t, phi, c="orange")
    ax_prof.set_xlim(0, 2 * np.pi)
    ax_prof.set_xticks(np.linspace(0, 2 * np.pi, 5),
                       ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax_prof.set_xlabel("t")
    ax_prof.set_ylabel(r"$\Phi(c + R(\cos t, \sin t))$")
    ax_prof.set_title(f"Intensité mesurée sur le cercle circonscrit (R = {radius:.3g})")
    ax_prof.grid(alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(1)

    LAMBDA = 8.0          # paramètre de Poisson
    I_TOT = 100.0         # somme des intensités
    DOMAIN = [[0, 10], [0, 10]]

    field = sample_sources(poisson_law(LAMBDA), I_TOT, DOMAIN, rng, eps=0.05)

    print(f"N = {field.n_sources}")
    print("positions :\n", np.round(field.positions, 3))
    print("intensités :", np.round(field.intensities, 3))
    print("somme      :", field.intensities.sum())

    # Évaluation ponctuelle
    x0 = np.array([5.0, 5.0])
    print(f"Phi({x0}) = {field(x0):.4f}")

    plot_field_and_profile(field, DOMAIN)
    plt.show()

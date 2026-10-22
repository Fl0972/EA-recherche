"""Affichage des capteurs et des signaux qu'ils mesurent."""

import numpy as np

from ..sensors import CircleSensors

# Graduations de l'axe des angles : 0, pi/2, pi, 3pi/2, 2pi
ANGLE_TICKS =(np.linspace(0, 2 * np.pi, 5),
               ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])


def plot_circle_sensors(sensors: CircleSensors, ax, color: str = "orange"):
    """Cercle des capteurs, capteurs, et repère t = 0."""
    t = np.linspace(0, 2 * np.pi, 400)
    cx, cy = sensors.center
    ax.plot(cx + sensors.radius * np.cos(t), cy + sensors.radius * np.sin(t),
            c=color, lw=1.5)
    ax.plot(*sensors.positions.T, "o", c=color, ms=3, label="capteurs")
    p0 = sensors.positions[0]
    ax.annotate("t = 0", p0, xytext=(-40, 6), textcoords="offset points", c=color)


def plot_circle_profile(sensors: CircleSensors, y: np.ndarray, ax, **style):
    """Mesures y en fonction de l'angle t du capteur."""
    ax.plot(sensors.angles, y, **style)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks(*ANGLE_TICKS)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\Phi(c + R(\cos t, \sin t))$")
    ax.grid(alpha=0.3)

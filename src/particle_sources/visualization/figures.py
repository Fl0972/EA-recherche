"""Figures complètes, assemblées à partir des briques de tracé."""

import matplotlib.pyplot as plt
import numpy as np

from ..model import Box, Sources, intensity_at
from ..sensors import CircleSensors
from .field_map import plot_box, plot_intensity_map, plot_sources
from .sensors_plot import plot_circle_profile, plot_circle_sensors


def field_and_profile(sources: Sources, box: Box, sensors: CircleSensors,
                      measurements: np.ndarray | None = None,
                      reconstruction: Sources | None = None, eps: float = 0.05,
                      resolution: int = 400):
    """
    Gauche : carte de Phi, sources, domaine, capteurs (et reconstruction).
    Droite : profil exact sur le cercle (et mesures des capteurs).
    """
    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(14, 6))

    plot_intensity_map(sources, Box.around_ball(sensors.center, sensors.radius),
                       ax_map, eps, resolution)
    plot_box(box, ax_map)
    plot_circle_sensors(sensors, ax_map)
    plot_sources(sources, ax_map, label="sources (taille ∝ intensité)")
    if reconstruction is not None:
        plot_sources(reconstruction, ax_map, c="none", edgecolors="cyan",
                     marker="s", linewidths=1.5, label="reconstruction")
    ax_map.set_title(f"{sources.n} sources — intensité totale = "
                     f"{sources.total_intensity:.3g}")
    ax_map.legend(loc="upper right", fontsize=8)

    dense = CircleSensors(sensors.center, sensors.radius, 1000, sensors.t0)
    plot_circle_profile(dense, intensity_at(sources, dense.positions), ax_prof,
                        c="orange", label="Phi exacte")
    if measurements is not None:
        ax_prof.plot(sensors.angles, measurements, ".", c="k", ms=4,
                     label=f"mesures ({sensors.n} capteurs)")
    ax_prof.set_title(f"Intensité sur le cercle des capteurs (R = {sensors.radius:.3g})")
    ax_prof.legend()

    fig.tight_layout()
    return fig

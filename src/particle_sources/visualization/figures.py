"""Figures complètes, assemblées à partir des briques de tracé."""

import matplotlib.pyplot as plt
import numpy as np

from ..model import Box, Sources, intensity_at
from ..sensors import CircleSensors
from .field_map import plot_box, plot_intensity_map, plot_sources
from .sensors_plot import plot_circle_profile, plot_circle_sensors


def field_and_profile(sources: Sources, box: Box, sensors: CircleSensors,
                      measurements: np.ndarray | None = None,
                      reconstruction: Sources | None = None,
                      fixed_sources: Sources | None = None, eps: float = 0.05,
                      resolution: int = 400):
    """
    Gauche : carte de Phi, sources, domaine, capteurs (et reconstruction).
    Droite : profil exact sur le cercle (et mesures des capteurs).
    `fixed_sources` (ex. source centrale) contribue au champ mais est tracée à part.

    Paramètres
    ----------
    sources : Sources
        Sources tirées (vérité terrain).
    box : Box
        Domaine des sources.
    sensors : CircleSensors
        Capteurs ; la carte couvre le carré contenant leur cercle.
    measurements : np.ndarray (K,) ou None
        Mesures à superposer au profil exact.
    reconstruction : Sources ou None
        Sources estimées (carrés cyan).
    fixed_sources : Sources ou None
        Sources fixes (étoile blanche).
    eps, resolution
        Régularisation et finesse de la carte (voir `plot_intensity_map`).

    Renvoie
    -------
    matplotlib.figure.Figure
    """
    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(14, 6))
    fixed_sources = fixed_sources if fixed_sources is not None else Sources.empty(box.dim)
    all_sources = sources + fixed_sources
    # Tailles des marqueurs relatives aux sources tirées (la source fixe les écraserait)
    ref = sources.intensities.max() if sources.n else None

    plot_intensity_map(all_sources, Box.around_ball(sensors.center, sensors.radius),
                       ax_map, eps, resolution)
    plot_box(box, ax_map)
    plot_circle_sensors(sensors, ax_map)
    plot_sources(sources, ax_map, ref_intensity=ref, label="sources (taille ∝ intensité)")
    if fixed_sources.n:
        ax_map.scatter(*fixed_sources.positions.T, s=250, marker="*", c="white",
                       edgecolors="black", label="source fixe")
    if reconstruction is not None:
        plot_sources(reconstruction, ax_map, ref_intensity=ref, c="none",
                     edgecolors="cyan", marker="s", linewidths=1.5,
                     label="reconstruction")
    ax_map.set_title(f"{sources.n} sources — intensité totale = "
                     f"{sources.total_intensity:.3g}")
    ax_map.legend(loc="upper right", fontsize=8)

    # Profil exact : 1000 points fictifs sur le même cercle
    dense = CircleSensors(sensors.center, sensors.radius, 1000, sensors.t0)
    plot_circle_profile(dense, intensity_at(all_sources, dense.positions), ax_prof,
                        c="orange", label="Phi exacte")
    if measurements is not None:
        ax_prof.plot(sensors.angles, measurements, ".", c="k", ms=4,
                     label=f"mesures ({sensors.n} capteurs)")
    ax_prof.set_title(f"Intensité sur le cercle des capteurs (R = {sensors.radius:.3g})")
    ax_prof.legend()

    fig.tight_layout()
    return fig

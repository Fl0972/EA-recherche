"""
Génère des sources, simule des mesures bruitées, puis reconstruit les sources.
Paramètres : dossier config/.

    python scripts/reconstruct.py
    python scripts/reconstruct.py --set sensors.noise.params.sigma=0.01
"""

from pathlib import Path

import matplotlib.pyplot as plt

from particle_sources.config import save_config
from particle_sources.config.cli import parse_config_args
from particle_sources.generation import sample_sources
from particle_sources.sensors import measure
from particle_sources.visualization import field_and_profile


def main():
    cfg = parse_config_args(__doc__)
    rng = cfg.rng()

    sources = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                             cfg.intensity_law, cfg.position_law)
    y = measure(sources, cfg.sensors, rng, cfg.noise, cfg.eps)
    estimate = cfg.invert(y, cfg.sensors)

    print("Vraies sources :", sources, sep="\n")
    print("Reconstruction :", estimate, sep="\n")

    viz = cfg.visualization
    fig = field_and_profile(sources, cfg.box, cfg.sensors, measurements=y,
                            reconstruction=estimate, eps=viz.eps,
                            resolution=viz.resolution)
    if viz.save_path:
        fig.savefig(viz.save_path, dpi=120)
        save_config(cfg, Path(viz.save_path).with_suffix(".yaml"))
    else:
        plt.show()


if __name__ == "__main__":
    main()

"""
Tire une configuration de sources et affiche l'intensité sur le domaine
et sur le cercle des capteurs. Paramètres : dossier config/.

    python scripts/simulate.py
    python scripts/simulate.py --set seed=3 --set sources.count_law.params.lam=4
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
    # Les capteurs voient aussi les sources fixes (source centrale)
    y = measure(sources + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)
    print(sources)
    if cfg.fixed_sources.n:
        print("Source(s) fixe(s) :", cfg.fixed_sources, sep="\n")

    viz = cfg.visualization
    fig = field_and_profile(sources, cfg.box, cfg.sensors, measurements=y,
                            fixed_sources=cfg.fixed_sources,
                            eps=viz.eps, resolution=viz.resolution)
    if viz.save_path:
        fig.savefig(viz.save_path, dpi=120)
        save_config(cfg, Path(viz.save_path).with_suffix(".yaml"))
    else:
        plt.show()


if __name__ == "__main__":
    main()

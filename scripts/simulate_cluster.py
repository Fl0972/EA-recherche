"""
Simulation d'une configuration avec au centre un gros cluster émettant
beaucoup plus de radiation, et profil sur le cercle des capteurs.

Exemples d'utilisation :
    # Lancer la simulation avec les paramètres par défaut du cluster :
    python scripts/simulate_cluster.py

    # Modifier la taille du cluster (rayon r = 0.8 ou 3.0) :
    python scripts/simulate_cluster.py --set sources.position_law.params.radius=0.8
    python scripts/simulate_cluster.py --set sources.position_law.params.radius=3.0

    # Modifier l'intensité totale ou la proportion émise par le cluster :
    python scripts/simulate_cluster.py --set sources.total_intensity=1000.0
    python scripts/simulate_cluster.py --set sources.intensity_law.params.cluster_share=0.98

    # Changer le nombre total de sources ou la graine aléatoire :
    python scripts/simulate_cluster.py --set sources.count_law.params.n=40 --set seed=42
"""

from pathlib import Path

import matplotlib.pyplot as plt

from particle_sources.config import save_config
from particle_sources.config.cli import parse_config_args
from particle_sources.generation import sample_sources
from particle_sources.sensors import measure
from particle_sources.visualization import field_and_profile

CLUSTER_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "experiments" / "cluster"


def main():
    cfg = parse_config_args(__doc__, default_config_dir=CLUSTER_CONFIG_DIR)
    rng = cfg.rng()

    sources = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                             cfg.intensity_law, cfg.position_law)
    y = measure(sources, cfg.sensors, rng, cfg.noise, cfg.eps)
    print(sources)

    viz = cfg.visualization
    fig = field_and_profile(sources, cfg.box, cfg.sensors, measurements=y,
                            eps=viz.eps, resolution=viz.resolution)
    if viz.save_path:
        fig.savefig(viz.save_path, dpi=120)
        save_config(cfg, Path(viz.save_path).with_suffix(".yaml"))
    else:
        plt.show()


if __name__ == "__main__":
    main()


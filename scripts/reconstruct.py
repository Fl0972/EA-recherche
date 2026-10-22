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
from particle_sources.inversion import subtract_known_sources
from particle_sources.sensors import measure
from particle_sources.visualization import field_and_profile


def main():
    """
    Config -> tirage des sources -> mesures (sources fixes comprises)
    -> retrait éventuel des sources fixes -> inversion -> comparaison
    vraies sources / reconstruction (console + figure).
    """
    cfg = parse_config_args(__doc__)
    rng = cfg.rng()

    sources = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                             cfg.intensity_law, cfg.position_law)
    # Les capteurs voient aussi les sources fixes (source centrale)
    y = measure(sources + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)

    # Les sources fixes sont connues : on peut retirer leur contribution
    y_inv = y
    if cfg.subtract_fixed_sources:
        y_inv = subtract_known_sources(y, cfg.fixed_sources, cfg.sensors, cfg.eps)
    estimate = cfg.invert(y_inv, cfg.sensors)

    print("Vraies sources :", sources, sep="\n")
    if cfg.fixed_sources.n:
        print("Source(s) fixe(s) :", cfg.fixed_sources, sep="\n")
    label = " (sources fixes retirées)" if cfg.subtract_fixed_sources else ""
    print(f"Reconstruction{label} :", estimate, sep="\n")

    viz = cfg.visualization
    fig = field_and_profile(sources, cfg.box, cfg.sensors, measurements=y,
                            fixed_sources=cfg.fixed_sources,
                            reconstruction=estimate, eps=viz.eps,
                            resolution=viz.resolution)
    if viz.save_path:
        fig.savefig(viz.save_path, dpi=120)
        save_config(cfg, Path(viz.save_path).with_suffix(".yaml"))
    else:
        plt.show()


if __name__ == "__main__":
    main()

"""
Génération en masse de situations (sources aléatoires + mesures des capteurs).

Les sources d'une situation ont un nombre variable N : elles sont stockées
dans des tableaux rembourrés à N_max, avec un masque. Dans chaque situation,
les sources sont triées par intensité décroissante (ordre canonique, utile
pour l'apprentissage).
"""

import numpy as np

from ..config import ExperimentConfig
from ..generation import sample_sources
from ..model import intensity_at


def generate_samples(cfg: ExperimentConfig, n: int, rng: np.random.Generator,
                     store_clean: bool = True,
                     max_sources: int | None = None) -> dict[str, np.ndarray]:
    """
    Tire `n` situations. Renvoie un dictionnaire de tableaux :
        measurements        (n, K)        mesures bruitées
        measurements_clean  (n, K)        mesures sans bruit (si store_clean)
        n_sources           (n,)          nombre de sources tirées
        positions           (n, N_max, d) positions (0 au-delà de n_sources)
        intensities         (n, N_max)    intensités (0 au-delà de n_sources)
        mask                (n, N_max)    True pour les vraies sources
    Les sources fixes (source centrale) sont dans les mesures, pas dans les labels.
    """
    sensor_positions = cfg.sensors.positions
    K = len(sensor_positions)
    measurements = np.empty((n, K))
    clean = np.empty((n, K))
    drawn = []

    for i in range(n):
        sources = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                                 cfg.intensity_law, cfg.position_law)
        clean[i] = intensity_at(sources + cfg.fixed_sources, sensor_positions, cfg.eps)
        measurements[i] = cfg.noise(clean[i], rng)
        drawn.append(sources)

    n_sources = np.array([s.n for s in drawn], dtype=np.int32)
    n_max = int(n_sources.max(initial=0))
    if max_sources is not None:
        if n_max > max_sources:
            raise ValueError(f"une situation a {n_max} sources > max_sources = {max_sources}")
        n_max = max_sources

    positions = np.zeros((n, n_max, cfg.box.dim))
    intensities = np.zeros((n, n_max))
    for i, s in enumerate(drawn):
        order = np.argsort(-s.intensities)
        positions[i, : s.n] = s.positions[order]
        intensities[i, : s.n] = s.intensities[order]
    mask = np.arange(n_max) < n_sources[:, None]

    out = {
        "measurements": measurements,
        "n_sources": n_sources,
        "positions": positions,
        "intensities": intensities,
        "mask": mask,
    }
    if store_clean:
        out["measurements_clean"] = clean
    return out

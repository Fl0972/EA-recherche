"""
Génère un jeu de données de situations (sources + mesures des capteurs) pour
l'apprentissage. Paramètres : dossier config/ (notamment dataset.yaml).

    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --set dataset.n_samples=100000 \\
        --set dataset.output_dir=data/bruit_1pc --set sensors.noise.params.sigma=0.01

Lecture en Python :
    from particle_sources.dataset import load_dataset
    data = load_dataset("data/dataset")   # dict de tableaux numpy
"""

import time
from pathlib import Path

from particle_sources.config.cli import PROJECT_ROOT, parse_config_args
from particle_sources.dataset import generate_dataset


def main():
    cfg = parse_config_args(__doc__)
    ds = cfg.dataset
    if ds is None:
        raise SystemExit("section `dataset` absente de la config (config/dataset.yaml)")

    output_dir = Path(ds.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print(f"Génération de {ds.n_samples} situations -> {output_dir}")
    start = time.perf_counter()
    generate_dataset(cfg, output_dir, ds.n_samples, ds.shard_size,
                     store_clean=ds.store_clean, dtype=ds.dtype,
                     max_sources=ds.max_sources, overwrite=ds.overwrite)
    print(f"Terminé en {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()

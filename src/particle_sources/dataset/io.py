"""
Écriture et lecture d'un jeu de données sur disque.

Structure d'un dossier de jeu de données :
    config.yaml        config complète utilisée (surcharges comprises)
    manifest.json      méta-données : taille, graine, fichiers, capteurs...
    shard_00000.npz    situations 0 .. shard_size-1
    shard_00001.npz    ...
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import ExperimentConfig, save_config
from .generate import generate_samples

FLOAT_KEYS = ("measurements", "measurements_clean", "positions", "intensities")
PADDED_KEYS = ("positions", "intensities", "mask")


def generate_dataset(cfg: ExperimentConfig, output_dir, n_samples: int,
                     shard_size: int, store_clean: bool = True,
                     dtype: str = "float32", max_sources: int | None = None,
                     overwrite: bool = False, verbose: bool = True) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} existe déjà (dataset.overwrite: true pour écraser)")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Un flux aléatoire indépendant par fichier, tous dérivés de la graine
    seed_seq = np.random.SeedSequence(cfg.seed)
    n_shards = -(-n_samples // shard_size)
    shard_seeds = seed_seq.spawn(n_shards)

    shards = []
    for k, ss in enumerate(shard_seeds):
        n = min(shard_size, n_samples - k * shard_size)
        arrays = generate_samples(cfg, n, np.random.default_rng(ss), store_clean, max_sources)
        for key in FLOAT_KEYS:
            if key in arrays:
                arrays[key] = arrays[key].astype(dtype)
        name = f"shard_{k:05d}.npz"
        np.savez_compressed(output_dir / name, **arrays)
        shards.append({"file": name, "n_samples": n,
                       "max_sources": int(arrays["mask"].shape[1])})
        if verbose:
            print(f"  {name} : {n} situations ({sum(s['n_samples'] for s in shards)}"
                  f"/{n_samples})")

    save_config(cfg, output_dir / "config.yaml")
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "n_samples": n_samples,
        "seed_entropy": seed_seq.entropy,   # permet de reproduire même si seed: null
        "dtype": dtype,
        "store_clean": store_clean,
        "sensor_positions": cfg.sensors.positions.tolist(),
        "fixed_sources": {"positions": cfg.fixed_sources.positions.tolist(),
                          "intensities": cfg.fixed_sources.intensities.tolist()},
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                              encoding="utf-8")
    return output_dir


def load_manifest(directory) -> dict:
    return json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))


def load_dataset(directory) -> dict[str, np.ndarray]:
    """
    Charge tous les fichiers d'un jeu de données et les concatène (les
    tableaux de sources sont rembourrés au N_max global).
    Ajoute `sensor_positions` (K, d).
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    parts = [dict(np.load(directory / s["file"])) for s in manifest["shards"]]
    n_max = max(p["mask"].shape[1] for p in parts)

    for p in parts:
        pad = n_max - p["mask"].shape[1]
        if pad:
            for key in PADDED_KEYS:
                widths = [(0, 0), (0, pad)] + [(0, 0)] * (p[key].ndim - 2)
                p[key] = np.pad(p[key], widths)

    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    data["sensor_positions"] = np.array(manifest["sensor_positions"])
    return data

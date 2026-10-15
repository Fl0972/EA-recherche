"""Options de ligne de commande communes à tous les scripts."""

import argparse
from pathlib import Path

from .loader import ExperimentConfig, load_config

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def parse_config_args(description: str | None = None,
                      default_config_dir: Path | str = DEFAULT_CONFIG_DIR) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=default_config_dir,
                        help=f"dossier de fichiers YAML (défaut : {default_config_dir})")
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                        help="surcharge un paramètre, ex. --set seed=3 "
                             "--set sensors.noise.params.sigma=0.01")
    args = parser.parse_args()
    return load_config(args.config, args.set)

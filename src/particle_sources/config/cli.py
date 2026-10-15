"""Options de ligne de commande communes à tous les scripts."""

import argparse
from pathlib import Path

from .loader import ExperimentConfig, load_config

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def parse_config_args(description: str | None = None) -> ExperimentConfig:
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR,
                        help="dossier de fichiers YAML (défaut : config/)")
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                        help="surcharge un paramètre, ex. --set seed=3 "
                             "--set sensors.noise.params.sigma=0.01")
    args = parser.parse_args()
    return load_config(args.config, args.set)

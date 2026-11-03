"""Options de ligne de commande communes à tous les scripts."""

import argparse
from pathlib import Path

from .loader import ExperimentConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


def parse_args(description: str | None = None) -> argparse.Namespace:
    """Renvoie args.config (dossier) et args.set (liste de surcharges)."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR,
                        help="dossier de fichiers YAML (défaut : config/)")
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                        help="surcharge un paramètre, ex. --set seed=3 "
                             "--set sensors.noise.params.sigma=0.01")
    return parser.parse_args()


def parse_config_args(description: str | None = None) -> ExperimentConfig:
    args = parse_args(description)
    return load_config(args.config, args.set)


def resolve_output(path) -> Path:
    """Chemin relatif -> relatif à la racine du projet."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path

"""Options de ligne de commande communes à tous les scripts."""

import argparse
from pathlib import Path

from .loader import ExperimentConfig, load_config

# Dossier config/ à la racine du dépôt (src/particle_sources/config/cli.py -> ../../../config)
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[3] / "config"


def parse_config_args(description: str | None = None) -> ExperimentConfig:
    """
    Lit les options --config / --set de la ligne de commande et charge la config.

    --config DOSSIER   dossier de fichiers YAML (défaut : config/ du dépôt)
    --set CLE=VALEUR   surcharge un paramètre ; répétable

    Paramètres
    ----------
    description : str ou None
        Texte d'aide affiché par --help (en pratique le docstring du script).

    Renvoie
    -------
    ExperimentConfig
        Configuration prête à l'emploi (objets construits).
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR,
                        help="dossier de fichiers YAML (défaut : config/)")
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                        help="surcharge un paramètre, ex. --set seed=3 "
                             "--set sensors.noise.params.sigma=0.01")
    args = parser.parse_args()
    return load_config(args.config, args.set)

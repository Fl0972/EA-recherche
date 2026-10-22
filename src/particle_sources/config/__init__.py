"""
Configuration des expériences à partir de fichiers YAML.

loader    lecture des YAML, surcharges, construction de `ExperimentConfig`
registry  correspondance noms YAML -> fonctions / classes du code
cli       options --config / --set communes aux scripts
"""

from .loader import ExperimentConfig, load_config, save_config

__all__ = ["ExperimentConfig", "load_config", "save_config"]

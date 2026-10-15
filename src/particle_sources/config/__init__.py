"""Configuration des expériences à partir de fichiers YAML."""

from .loader import ExperimentConfig, load_config, save_config

__all__ = ["ExperimentConfig", "load_config", "save_config"]

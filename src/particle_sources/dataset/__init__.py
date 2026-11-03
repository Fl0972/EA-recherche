"""Génération et lecture de jeux de données pour l'apprentissage."""

from .generate import generate_samples
from .io import generate_dataset, load_dataset, load_manifest

__all__ = ["generate_dataset", "generate_samples", "load_dataset", "load_manifest"]

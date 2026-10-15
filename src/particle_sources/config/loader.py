"""
Chargement de la configuration d'une expérience.

Tous les fichiers *.yaml d'un dossier sont fusionnés (chaque clé de premier
niveau ne doit apparaître que dans un seul fichier), puis les objets sont
construits à partir des noms via `registry`.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from ..generation import central_source
from ..generation import count_laws as _count_laws
from ..generation.count_laws import CountLaw
from ..generation.intensity_laws import IntensityLaw
from ..generation.position_laws import PositionLaw
from ..model import Box, Sources
from ..sensors.noise import NoiseModel
from . import registry as reg


@dataclass(frozen=True)
class VisualizationConfig:
    eps: float
    resolution: int
    save_path: str | None


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict                       # config brute, telle que lue (pour la sauvegarder)
    seed: int | None
    box: Box
    total_intensity: float
    count_law: CountLaw
    intensity_law: IntensityLaw
    position_law: PositionLaw
    fixed_sources: Sources          # sources toujours présentes, hors tirage
    sensors: object                 # objet exposant .positions
    noise: NoiseModel
    eps: float
    invert: Callable[[np.ndarray, object], Sources]
    subtract_fixed_sources: bool    # retirer les sources fixes avant l'inversion
    visualization: VisualizationConfig

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


# ---- Lecture des fichiers -----------------------------------------------------
class _Loader(yaml.SafeLoader):
    """SafeLoader qui lit aussi `1e-8` comme un float (PyYAML exige `1.0e-8`)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$|^[-+]?\.(?:inf|Inf|INF)$"
               r"|^\.(?:nan|NaN|NAN)$"),
    list("-+0123456789."),
)


def _parse(text: str):
    return yaml.load(text, Loader=_Loader)


def read_config_dir(config_dir) -> dict:
    config_dir = Path(config_dir)
    files = sorted(config_dir.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"aucun fichier .yaml dans {config_dir}")
    raw: dict = {}
    for path in files:
        content = _parse(path.read_text(encoding="utf-8")) or {}
        for key in content:
            if key in raw:
                raise ValueError(f"clé {key!r} définie dans plusieurs fichiers ({path.name})")
        raw.update(content)
    return raw


def apply_override(raw: dict, override: str) -> None:
    """Applique une surcharge 'a.b.c=valeur' (valeur interprétée en YAML)."""
    key, sep, value = override.partition("=")
    if not sep:
        raise ValueError(f"surcharge invalide {override!r}, attendu clé=valeur")
    *parents, leaf = key.strip().split(".")
    node = raw
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = _parse(value)


def _get(d: dict, key: str, where: str):
    if key not in d:
        raise KeyError(f"paramètre manquant dans la config : {where}{key}")
    return d[key]


# ---- Construction des objets --------------------------------------------------
def load_config(config_dir, overrides=()) -> ExperimentConfig:
    raw = read_config_dir(config_dir)
    for override in overrides:
        apply_override(raw, override)

    box = Box.from_bounds(_get(raw, "domain", ""))
    src = _get(raw, "sources", "")
    sens = _get(raw, "sensors", "")
    rec = _get(raw, "reconstruction", "")
    viz = _get(raw, "visualization", "")

    count_law = reg.build(reg.COUNT_LAWS, _get(src, "count_law", "sources."), "count_law")
    if src.get("at_least_one", False):
        count_law = _count_laws.at_least_one(count_law)

    total_intensity = float(_get(src, "total_intensity", "sources."))
    fixed_sources = Sources.empty(box.dim)
    central = raw.get("central_source") or {}
    if central.get("enabled", False):
        fixed_sources = central_source(
            box, float(_get(central, "intensity_factor", "central_source.")) * total_intensity,
            central.get("position"))

    return ExperimentConfig(
        raw=raw,
        seed=raw.get("seed"),
        box=box,
        total_intensity=total_intensity,
        count_law=count_law,
        intensity_law=reg.build(reg.INTENSITY_LAWS,
                                _get(src, "intensity_law", "sources."), "intensity_law"),
        position_law=reg.build(reg.POSITION_LAWS,
                               _get(src, "position_law", "sources."), "position_law"),
        fixed_sources=fixed_sources,
        sensors=reg.build(reg.SENSOR_LAYOUTS,
                          _get(sens, "layout", "sensors."), "sensors.layout", box),
        noise=reg.build(reg.NOISE_MODELS, _get(sens, "noise", "sensors."), "sensors.noise"),
        eps=float(sens.get("eps", 0.0)),
        invert=reg.build(reg.INVERSION_METHODS,
                         _get(rec, "method", "reconstruction."), "reconstruction.method"),
        subtract_fixed_sources=bool(rec.get("subtract_fixed_sources", False)),
        visualization=VisualizationConfig(
            eps=float(viz.get("eps", 0.05)),
            resolution=int(viz.get("resolution", 400)),
            save_path=viz.get("save_path"),
        ),
    )


def save_config(cfg: ExperimentConfig, path) -> None:
    """Enregistre la config effectivement utilisée (surcharges comprises)."""
    Path(path).write_text(yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")

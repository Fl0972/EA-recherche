"""
Banc d'essai : compare des méthodes d'inversion en faisant varier un
paramètre de la config (niveau de bruit, nombre de capteurs, ...).

Pour chaque valeur du paramètre, les mêmes situations (mêmes sources) sont
tirées : seules les mesures changent. Chaque méthode est évaluée sur ces
situations, et la borne de Cramér-Rao est calculée en référence.
"""

import json
import time
from dataclasses import dataclass

import numpy as np

from ..config import load_config
from ..config import registry as reg
from ..generation import sample_sources
from ..model import intensity_at
from .crb import crb
from .metrics import evaluate

CRB_LABEL = "Borne de Cramér-Rao"


@dataclass(frozen=True)
class BenchmarkSpec:
    n_situations: int
    sweep_key: str               # paramètre de config que l'on fait varier
    sweep_values: list
    methods: list[dict]          # [{label, name, params}, ...]
    ospa_cutoff: float
    output_dir: str

    @classmethod
    def from_raw(cls, raw: dict) -> "BenchmarkSpec":
        if "benchmark" not in raw:
            raise KeyError("section `benchmark` absente de la config (config/benchmark.yaml)")
        b = raw["benchmark"]
        return cls(
            n_situations=int(b["n_situations"]),
            sweep_key=b["sweep"]["key"],
            sweep_values=list(b["sweep"]["values"]),
            methods=list(b["methods"]),
            ospa_cutoff=float(b.get("ospa_cutoff", 1.0)),
            output_dir=b.get("output_dir", "outputs/benchmark"),
        )


def _median(values) -> float:
    values = np.concatenate([np.atleast_1d(v) for v in values]) if values else np.empty(0)
    return float(np.median(values)) if values.size else float("nan")


def run_benchmark(config_dir, overrides, spec: BenchmarkSpec,
                  verbose: bool = True) -> list[dict]:
    """
    Renvoie une ligne par (valeur du paramètre, méthode) :
        n_accuracy        proportion de situations où N est bien estimé
        ospa              distance OSPA moyenne (pénalise aussi les erreurs sur N)
        position_error    erreur de position médiane (situations à N correct)
        intensity_error   erreur d'intensité médiane (idem)
        position_ratio    médiane de erreur / borne de Cramér-Rao, source par
        intensity_ratio   source (idem). ~0.7-0.8 : la méthode atteint la borne.
        time_ms           temps moyen par situation
    Pour la borne de Cramér-Rao : écarts-types minimaux médians (toutes situations).
    """
    base = load_config(config_dir, overrides)
    # Même graine pour toutes les valeurs : mêmes sources d'une valeur à l'autre
    entropy = np.random.SeedSequence(base.seed).entropy
    rows = []

    for value in spec.sweep_values:
        cfg = load_config(config_dir, [*overrides, f"{spec.sweep_key}={json.dumps(value)}"])
        methods = {m["label"]: reg.build(reg.INVERSION_METHODS, m, f"méthode {m['label']}")
                   for m in spec.methods}
        rng_src, rng_noise = (np.random.default_rng(s)
                              for s in np.random.SeedSequence(entropy).spawn(2))
        results = {label: [] for label in methods}
        elapsed = dict.fromkeys(methods, 0.0)
        crb_pos, crb_int = [], []

        for _ in range(spec.n_situations):
            true = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng_src,
                                  cfg.intensity_law, cfg.position_law)
            seen = true + cfg.fixed_sources
            y = cfg.noise(intensity_at(seen, cfg.sensors.positions, cfg.eps), rng_noise)

            p, i = crb(true, cfg.problem, seen)
            crb_pos.append(p)
            crb_int.append(i)
            for label, method in methods.items():
                start = time.perf_counter()
                est = method(y, cfg.problem)
                elapsed[label] += time.perf_counter() - start
                results[label].append(evaluate(true, est, spec.ospa_cutoff, p, i))

        for label, res in results.items():
            rows.append({
                "sweep_value": value,
                "method": label,
                "n_accuracy": float(np.mean([r.n_est == r.n_true for r in res])),
                "ospa": float(np.mean([r.ospa for r in res])),
                "position_error": _median([r.position_errors for r in res]),
                "intensity_error": _median([r.intensity_errors for r in res]),
                "position_ratio": _median([r.position_ratios for r in res]),
                "intensity_ratio": _median([r.intensity_ratios for r in res]),
                "time_ms": 1e3 * elapsed[label] / spec.n_situations,
            })
        rows.append({"sweep_value": value, "method": CRB_LABEL,
                     "n_accuracy": float("nan"), "ospa": float("nan"),
                     "position_error": _median(crb_pos),
                     "intensity_error": _median(crb_int),
                     "position_ratio": float("nan"), "intensity_ratio": float("nan"),
                     "time_ms": float("nan")})
        if verbose:
            summary = "  ".join(f"{r['method']}: N ok {r['n_accuracy']:.0%}, "
                                f"erreur/borne {r['position_ratio']:.2f}"
                                for r in rows[-len(methods) - 1 : -1])
            print(f"  {spec.sweep_key} = {value}  |  {summary}")
    return rows

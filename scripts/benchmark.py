"""
Compare les méthodes d'inversion (et la borne de Cramér-Rao) en faisant varier
un paramètre de la config. Paramètres : config/benchmark.yaml.

    python scripts/benchmark.py
    python scripts/benchmark.py --set benchmark.n_situations=50
    python scripts/benchmark.py --set benchmark.sweep.key=sensors.layout.params.n \\
        --set "benchmark.sweep.values=[16, 32, 64, 128]" \\
        --set benchmark.output_dir=outputs/benchmark_capteurs
"""

import csv
import time

import matplotlib

matplotlib.use("Agg")  # figure enregistrée, pas affichée

from particle_sources.benchmark import CRB_LABEL, BenchmarkSpec, run_benchmark
from particle_sources.config import load_config, save_config
from particle_sources.config.cli import parse_args, resolve_output
from particle_sources.visualization.benchmark_plot import plot_benchmark


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    spec = BenchmarkSpec.from_raw(cfg.raw)
    out = resolve_output(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{spec.n_situations} situations x {len(spec.sweep_values)} valeurs de "
          f"{spec.sweep_key} x {len(spec.methods)} méthodes")
    start = time.perf_counter()
    rows = run_benchmark(args.config, args.set, spec)
    print(f"Terminé en {time.perf_counter() - start:.0f} s")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_benchmark(rows, spec.sweep_key, CRB_LABEL).savefig(out / "benchmark.png", dpi=120)
    save_config(cfg, out / "config.yaml")
    print(f"Résultats : {out}")


if __name__ == "__main__":
    main()

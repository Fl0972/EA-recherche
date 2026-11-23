"""
Reproduit le cas test de l'industriel : deux sources, l'une x fois plus
intense que l'autre, reconstruites par descente de gradient (réussite
annoncée pour x = 10, échec pour x = 100). Compare avec Gauss-Newton depuis la
même initialisation et avec Prony + maximum de vraisemblance.
Paramètres : config/two_sources.yaml.

    python scripts/two_sources.py
    python scripts/two_sources.py --set two_sources.random_trials.n=0
    python scripts/two_sources.py --set two_sources.noise.name=gaussian_relative \\
        --set "two_sources.noise.params={sigma: 0.001}" \\
        --set two_sources.output_dir=outputs/two_sources_bruit
"""

import csv
import time

import matplotlib

matplotlib.use("Agg")  # figure enregistrée, pas affichée

from particle_sources.benchmark import TwoSourcesSpec, run_fixed_case, run_random_trials
from particle_sources.config import load_config, save_config
from particle_sources.config.cli import parse_args, resolve_output
from particle_sources.visualization.intensity_ratio_plot import plot_two_sources_study


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_fixed(rows):
    print(f"{'x':>6}  {'méthode':<28} {'err. forte':>10} {'err. faible':>11} "
          f"{'réussite':>8} {'itérations':>10} {'temps':>9}")
    for r in rows:
        n_iter = "" if r["n_iter"] is None else str(r["n_iter"])
        print(f"{r['ratio']:>6g}  {r['method']:<28} {r['err_strong']:>10.2e} "
              f"{r['err_faint']:>11.2e} {'oui' if r['success'] else 'non':>8} "
              f"{n_iter:>10} {r['time_ms']:>7.0f}ms")


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    spec = TwoSourcesSpec.from_raw(cfg.raw)
    out = resolve_output(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Cas fixe : {len(spec.ratios)} rapports d'intensité x 3 méthodes")
    fixed_rows, descents, truths = run_fixed_case(spec, cfg.sensors, cfg.box,
                                                  cfg.total_intensity, cfg.rng())
    print_fixed(fixed_rows)
    write_csv(fixed_rows, out / "fixed_case.csv")

    random_rows = []
    if spec.n_trials:
        print(f"\nEssais aléatoires : {spec.n_trials} par rapport")
        start = time.perf_counter()
        random_rows = run_random_trials(
            spec, cfg.sensors, cfg.box, cfg.total_intensity,
            progress=lambda x: print(f"  x = {x:g} fait ({time.perf_counter() - start:.0f} s)"))
        for r in random_rows:
            print(f"{r['ratio']:>6g}  {r['method']:<28} réussite {r['success_rate']:>5.0%}  "
                  f"erreur médiane source faible {r['median_err_faint']:.2e}")
        write_csv(random_rows, out / "random_trials.csv")

    shown = [x for x in spec.show_ratios if x in descents]
    fig = plot_two_sources_study(fixed_rows, descents, truths, random_rows, cfg.box,
                                 shown, spec.success_tol)
    fig.savefig(out / "two_sources.png", dpi=120)
    save_config(cfg, out / "config.yaml")
    print(f"\nRésultats : {out}")


if __name__ == "__main__":
    main()

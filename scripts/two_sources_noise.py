"""
Cas test à deux sources avec bruit de mesure (rapport d'intensité x ~ 100).
Compare l'erreur des méthodes à la borne de Cramér-Rao, en géométrie fixe
puis sur des géométries aléatoires.
Paramètres : config/two_sources_noise.yaml (et two_sources.yaml).

    python scripts/two_sources_noise.py
    python scripts/two_sources_noise.py --set two_sources_noise.n_geometries=0
"""

import csv
import time

import matplotlib

matplotlib.use("Agg")  # figure enregistrée, pas affichée

import numpy as np

from particle_sources.benchmark import (NoiseStudySpec, TwoSourcesSpec, run_fixed_geometry,
                                        run_random_geometries)
from particle_sources.config import load_config, save_config
from particle_sources.config.cli import parse_args, resolve_output
from particle_sources.visualization.two_sources_noise_plot import plot_noise_study


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows, extra=()):
    print(f"{'x':>5} {'sigma':>7}  {'méthode':<27} {'détect.':>7} {'réuss.':>6} "
          f"{'err. faible':>11} {'CRB faible':>10} {'err./CRB':>8} {'err. forte':>10}"
          + "".join(f" {k:>11}" for k in extra))
    for r in rows:
        print(f"{r['ratio']:>5g} {r['sigma']:>7.0e}  {r['method']:<27} {r['detection']:>7.0%} "
              f"{r['success']:>6.0%} {r['err_faint']:>11.2e} {r['crb_faint']:>10.2e} "
              f"{r['faint_over_crb']:>8.2f} {r['err_strong']:>10.2e}"
              + "".join(f" {r[k]:>11.0%}" for k in extra))


def main():
    args = parse_args(__doc__)
    cfg = load_config(args.config, args.set)
    base = TwoSourcesSpec.from_raw(cfg.raw)
    spec = NoiseStudySpec.from_raw(cfg.raw)
    out = resolve_output(spec.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    elapsed = lambda: f"{time.perf_counter() - start:.0f} s"

    print(f"Géométrie fixe : x = {spec.ratios}, sigma = {spec.sigmas}, "
          f"{spec.n_realizations} tirages de bruit")
    fixed = run_fixed_geometry(spec, base, cfg.sensors, cfg.box, cfg.total_intensity,
                               progress=lambda x, s: print(f"  x = {x:g}, sigma = {s:g} "
                                                           f"({elapsed()})"))
    print_rows(fixed)
    write_csv(fixed, out / "fixed_geometry.csv")

    random_rows, geo = [], {}
    if spec.n_geometries:
        print(f"\n{spec.n_geometries} géométries aléatoires, x = {spec.geometry_ratio:g}")
        random_rows, geo = run_random_geometries(
            spec, base, cfg.sensors, cfg.box, cfg.total_intensity,
            progress=lambda s: print(f"  sigma = {s:g} ({elapsed()})"))
        print_rows(random_rows, extra=("localizable",))
        write_csv(random_rows, out / "random_geometries.csv")
        np.savez(out / "geometries.npz", **geo)

    fig = plot_noise_study(fixed, random_rows, geo, base.success_tol)
    fig.savefig(out / "two_sources_noise.png", dpi=120)
    save_config(cfg, out / "config.yaml")
    print(f"\nTerminé en {elapsed()}. Résultats : {out}")


if __name__ == "__main__":
    main()

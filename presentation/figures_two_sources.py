"""
Figures de la synthèse « cas test à deux sources » (synthese_deux_sources.typ).

Refait le cas fixe (~1 min) pour les trajectoires ; les essais aléatoires
(~16 min) sont relus dans outputs/two_sources/random_trials.csv, produit par
scripts/two_sources.py. Les chiffres cités dans le texte sont enregistrés
dans figures/deux_sources.json.

    python presentation/figures_two_sources.py
"""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from particle_sources.benchmark import TwoSourcesSpec, run_fixed_case, run_random_trials
from particle_sources.config import load_config
from particle_sources.config.cli import DEFAULT_CONFIG_DIR, resolve_output
from particle_sources.generation import bright_and_faint
from particle_sources.model import Sources
from particle_sources.visualization import intensity_ratio_plot as P

from figures import FIG_DIR, _save

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def _read_random_rows(spec, cfg):
    path = resolve_output(spec.output_dir) / "random_trials.csv"
    if not path.exists():
        print("random_trials.csv absent : calcul des essais aléatoires (~16 min)")
        return run_random_trials(spec, cfg.sensors, cfg.box, cfg.total_intensity)
    with open(path, encoding="utf-8") as f:
        return [{**r, "ratio": float(r["ratio"]), "n_trials": int(r["n_trials"]),
                 "success_rate": float(r["success_rate"]),
                 "median_err_faint": float(r["median_err_faint"])}
                for r in csv.DictReader(f)]


def main():
    cfg = load_config(DEFAULT_CONFIG_DIR)
    spec = TwoSourcesSpec.from_raw(cfg.raw)
    box, sensors = cfg.box, cfg.sensors

    print("Cas fixe...")
    fixed, descents, truths = run_fixed_case(spec, sensors, box, cfg.total_intensity,
                                             cfg.rng())
    random_rows = _read_random_rows(spec, cfg)

    # 1. Schéma du cas test
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    init = Sources(spec.init_positions, [cfg.total_intensity / 2] * 2)
    P.plot_setup(ax, bright_and_faint(spec.positions, 10, cfg.total_intensity), init,
                 box, sensors)
    _save(fig, "deux_sources_schema.png")

    # 2. Trajectoires de la descente, x = 10 et x = 100
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2))
    for ax, x in zip(axes, (10.0, 100.0)):
        P.plot_descent_path(ax, truths[x], descents[x], box,
                            P.descent_title(fixed, descents, x))
    _save(fig, "deux_sources_trajectoires.png")

    # 3. Paysage de la perte vu par la source faible, x = 10 et x = 100
    fig, axes = plt.subplots(1, 2, figsize=(9, 6.2), constrained_layout=True)
    for ax, x in zip(axes, (10.0, 100.0)):
        im = P.plot_loss_landscape(ax, truths[x], box, sensors)
        ax.set_title(f"x = {x:g}")
    fig.colorbar(im, ax=axes, label="perte relative", shrink=0.8)
    _save(fig, "deux_sources_paysage.png")

    # 4. Perte au fil des itérations
    fig, ax = plt.subplots(figsize=(6.5, 4))
    P.plot_losses(ax, descents)
    _save(fig, "deux_sources_perte.png")

    # 5. Taux de réussite (essais aléatoires) et conditionnement
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    P.plot_success(axes[0], random_rows)
    axes[0].set_title(f"Réussite sur {random_rows[0]['n_trials']} situations aléatoires")
    P.plot_conditioning(axes[1], fixed)
    axes[1].set_title("Conditionnement de la perte (cas fixe)")
    fig.tight_layout()
    _save(fig, "deux_sources_reussite.png")

    numbers = {"fixed": fixed, "random": random_rows,
               "positions": spec.positions.tolist(), "init": spec.init_positions.tolist(),
               "max_iter": spec.max_iter, "success_tol": spec.success_tol,
               "n_sensors": len(sensors.positions)}
    (FIG_DIR / "deux_sources.json").write_text(json.dumps(numbers, indent=1),
                                               encoding="utf-8")
    print(f"Figures : {FIG_DIR}")


if __name__ == "__main__":
    main()

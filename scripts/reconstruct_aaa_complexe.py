"""
Génère des sources, simule des mesures bruitées, puis reconstruit les sources
par approximation rationnelle (AAA + moindres carrés), source centrale comprise.
Paramètres : dossier config/ (la section `reconstruction` est ignorée).

Affiche aussi le spectre des mesures, pour voir d'où vient L : jusqu'à
quelle fréquence chaque source reste visible au-dessus du bruit.

    python scripts/reconstruct_aaa_complexe.py
    python scripts/reconstruct_aaa_complexe.py --set seed=3
    python scripts/reconstruct_aaa_complexe.py --set sensors.noise.params.sigma=0.001
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from particle_sources.config import save_config
from particle_sources.config.cli import parse_config_args
from particle_sources.generation import sample_sources
from particle_sources.inversion.aaa_sources import reconstruire
from particle_sources.sensors import measure
from particle_sources.visualization import field_and_profile

S = 4.0  # seuil de détection, en multiples du bruit (paramètre `s` de reconstruire)
# Nombre maximal de sources ajoutées une à une en cherchant dans le résidu,
# après la passe AAA (0 = AAA seul, au plus L/2 sources hors centre)
ITERATIONS_RESIDU = 60


def visibilite(sources, sensors, sigma, s=S):
    """
    Pour chaque source : son poids w_i, son rapport rho_i / R, et la dernière
    fréquence n_i où sa contribution w_i (rho_i/R)^n dépasse s * sigma :

        n_i = log(s sigma / w_i) / log(rho_i / R)

    (0 si elle est déjà sous le bruit, inf pour une source exactement au centre,
    qui n'apparaît dans aucune fréquence négative).
    """
    R = sensors.radius
    rho = np.linalg.norm(sources.positions - sensors.center, axis=-1)
    w = sources.intensities / (R**2 - rho**2)
    with np.errstate(divide="ignore"):
        n = np.log(s * sigma / w) / np.log(rho / R)
    n = np.where(rho == 0, np.inf, np.maximum(n, 0))
    return w, rho / R, n


def figure_spectre(y, sources, sensors, diag, s=S):
    """
    |c_{-n}| mesurés (échelle log), seuil s*sigma, coupure L, et la
    contribution théorique w_i (rho_i/R)^n de chaque vraie source hors centre.
    """
    K = len(y)
    n = np.arange(1, K // 2)
    X = np.fft.fft(y) / K
    w, ratio, _ = visibilite(sources, sensors, diag["sigma"], s)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(n, np.abs(X[K - n]), "k.", label=r"mesuré $|\hat c_{-n}|$")
    for i, (wi, qi) in enumerate(zip(w, ratio)):
        if qi > 0:
            ax.semilogy(n, wi * qi**n, "--", lw=1,
                        label=rf"source {i} : $\rho/R$ = {qi:.2f}")
    ax.axhline(s * diag["sigma"], color="r", label=rf"seuil {s:g}$\hat\sigma$")
    ax.axvline(diag["L"] + 0.5, color="g", label=f"L = {diag['L']}")
    ax.set_xlabel("fréquence n")
    ax.set_ylabel("amplitude")
    ax.set_title(f"Spectre des mesures : au plus L//2 = {diag['L'] // 2} "
                 "sources résolubles hors centre")
    floor = max(np.abs(X[K - n]).min(), 1e-3 * s * diag["sigma"])
    ax.set_ylim(floor / 2, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    """
    Config -> tirage des sources -> mesures (sources fixes comprises)
    -> reconstruction AAA sur les mesures brutes -> comparaison
    vraies sources / reconstruction (console + figures).

    On ne retire pas la source centrale : la méthode l'estime elle-même
    (coefficient c_0), ce qui évite de laisser son bruit dans le résidu.
    """
    cfg = parse_config_args(__doc__)
    print(f"Graine : {cfg.seed}   (rejouer avec --set seed={cfg.seed})")
    rng = cfg.rng()

    sources = sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                             cfg.intensity_law, cfg.position_law)
    y = measure(sources + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)

    estimate, diag = reconstruire(y, cfg.sensors, s=S,
                                  iterations_residu=ITERATIONS_RESIDU)

    print("Vraies sources :", sources, sep="\n")
    if cfg.fixed_sources.n:
        print("Source(s) fixe(s) :", cfg.fixed_sources, sep="\n")

    print(f"\nBruit estimé sigma = {diag['sigma']:.3g} ; seuil {S:g} sigma "
          f"= {S * diag['sigma']:.3g}")
    print(f"L = {diag['L']} fréquences utiles -> au plus {diag['L'] // 2} "
          "sources résolubles hors centre")
    w, ratio, n_vis = visibilite(sources, cfg.sensors, diag["sigma"])
    print(f"{np.sum(n_vis >= 1)} source(s) sur {sources.n} dépassent le bruit "
          "sur au moins une fréquence :")
    for i, (wi, qi, ni) in enumerate(zip(w, ratio, n_vis)):
        if ni >= 1:
            print(f"  source {i} : w = {wi:.3g}, rho/R = {qi:.2f} "
                  f"-> visible jusqu'à n ~ {ni:.1f}")

    print(f"\nPasse AAA : {len(diag['initial'][0]) - 1} source(s) hors centre ; "
          f"recherche sur le résidu : {diag['ajoutees_residu']} ajoutée(s)")
    print("Reconstruction (source centrale comprise) :", estimate, sep="\n")

    viz = cfg.visualization
    fig = field_and_profile(sources, cfg.box, cfg.sensors, measurements=y,
                            fixed_sources=cfg.fixed_sources,
                            reconstruction=estimate, eps=viz.eps,
                            resolution=viz.resolution)
    fig_spectre = figure_spectre(y, sources, cfg.sensors, diag)
    if viz.save_path:
        path = Path(viz.save_path)
        fig.savefig(path, dpi=120)
        fig_spectre.savefig(path.with_name(path.stem + "_spectre" + path.suffix), dpi=120)
        save_config(cfg, path.with_suffix(".yaml"))
    else:
        plt.show()


if __name__ == "__main__":
    main()

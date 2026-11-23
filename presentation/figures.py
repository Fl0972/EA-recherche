"""
Figures de la présentation, produites en faisant tourner les simulations.
Chaque fonction enregistre une image dans presentation/figures/ et renvoie
les chiffres utiles au texte des transparents.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from particle_sources.benchmark import BenchmarkSpec, CRB_LABEL, evaluate, match, run_benchmark
from particle_sources.config import load_config
from particle_sources.config.cli import DEFAULT_CONFIG_DIR
from particle_sources.generation import sample_sources
from particle_sources.inversion import prony, prony_mle
from particle_sources.model import Box, Sources, intensity_at
from particle_sources.sensors import measure
from particle_sources.visualization import (field_and_profile, plot_box, plot_circle_sensors,
                                            plot_intensity_map, plot_sources)

FIG_DIR = Path(__file__).resolve().parent / "figures"
ORANGE, BLUE, GREY = "#E07A1F", "#1F5FA0", "#888888"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def _cfg(*overrides):
    return load_config(DEFAULT_CONFIG_DIR, list(overrides))


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw(cfg, rng):
    return sample_sources(cfg.box, cfg.count_law, cfg.total_intensity, rng,
                          cfg.intensity_law, cfg.position_law)


def _pick_seed(cfg, n, min_dist=2.5, min_intensity=4.0):
    """Première graine donnant n sources bien séparées et pas trop faibles."""
    for seed in range(1000):
        s = _draw(cfg, np.random.default_rng(seed))
        if s.n != n or s.intensities.min() < min_intensity:
            continue
        d = np.linalg.norm(s.positions[:, None] - s.positions[None], axis=-1)
        if d[np.triu_indices(n, 1)].min() >= min_dist:
            return seed
    raise RuntimeError("aucune graine trouvée")


# ---------------------------------------------------------------------------
def situation():
    """Une situation complète : carte d'intensité + profil mesuré."""
    cfg = _cfg("sources.count_law.params={lam: 4.0}", "sensors.noise.params.sigma=0.001")
    seed = _pick_seed(cfg, 4)
    rng = np.random.default_rng(seed)
    sources = _draw(cfg, rng)
    y = measure(sources + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)
    fig = field_and_profile(sources, cfg.box, cfg.sensors, measurements=y,
                            fixed_sources=cfg.fixed_sources, eps=0.05, resolution=300)
    return _save(fig, "situation.png")


def prony_noise():
    """Prony exact sans bruit, fragile avec 0,1 % de bruit."""
    base = ["central_source.enabled=false", "sources.count_law.name=fixed",
            "sources.count_law.params={n: 5}"]
    cfg_clean = _cfg(*base, "sensors.noise.name=none", "sensors.noise.params={}")
    cfg_noisy = _cfg(*base, "sensors.noise.params.sigma=0.001")
    seed = _pick_seed(cfg_clean, 5, min_dist=2.0, min_intensity=2.0)
    rng = np.random.default_rng(seed)
    sources = _draw(cfg_clean, rng)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    results = {}
    for ax, cfg, title in [(axes[0], cfg_clean, "Sans bruit"),
                           (axes[1], cfg_noisy, "0,1 % de bruit")]:
        y = measure(sources, cfg.sensors, rng, cfg.noise)
        est = prony(y, cfg.problem)
        results[title] = (sources.n, est.n)
        extent = Box.around_ball(cfg.sensors.center, cfg.sensors.radius)
        plot_intensity_map(sources, extent, ax, eps=0.05, resolution=250)
        plot_box(cfg.box, ax)
        plot_circle_sensors(cfg.sensors, ax, color=ORANGE)
        ref = sources.intensities.max()
        plot_sources(sources, ax, ref_intensity=ref, label="vraies sources")
        plot_sources(est, ax, ref_intensity=ref, c="none", edgecolors="cyan",
                     marker="s", linewidths=1.8, label="Prony")
        ax.set_title(f"{title} : {est.n} sources retrouvées sur {sources.n}", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "prony_bruit.png"), results


def fourier_decay():
    """Décroissance des coefficients de Fourier face au niveau de bruit."""
    cfg = _cfg("central_source.enabled=false", "sources.count_law.name=fixed",
               "sources.count_law.params={n: 4}", "sensors.noise.name=none",
               "sensors.noise.params={}")
    seed = _pick_seed(cfg, 4, min_dist=2.0, min_intensity=5.0)
    sources = _draw(cfg, np.random.default_rng(seed))
    sensors = cfg.sensors
    R, K = sensors.radius, sensors.n
    n = np.arange(K // 2)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    z = (sources.positions - sensors.center) @ np.array([1, 1j])
    for k, (zk, I) in enumerate(zip(z, sources.intensities)):
        rho = abs(zk)
        w = I / (R**2 - rho**2)
        ax.semilogy(n, w * (rho / R) ** n, lw=1.8,
                    label=fr"source {k + 1} : $\rho/R$ = {rho / R:.2f}, I = {I:.0f}")
    y = intensity_at(sources, sensors.positions)
    for sigma, ls in [(1e-4, ":"), (1e-3, "--"), (1e-2, "-.")]:
        floor = np.sqrt(np.sum((sigma * y) ** 2)) / K
        ax.axhline(floor, c="k", ls=ls, lw=1, label=fr"bruit $\sigma$ = {sigma:g}")
    ax.set_xlabel("ordre n du coefficient de Fourier")
    ax.set_ylabel(r"$|w_i\,u_i^{\,n}|$ (contribution de la source i à $c_n$)")
    ax.set_title("L'information décroît exponentiellement avec n")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return _save(fig, "fourier_decroissance.png")


def central_source():
    """Effet de la source centrale sur les mesures (bruit relatif 0,1 %)."""
    sigma = 1e-3
    cfg_on = _cfg("sources.count_law.params={lam: 4.0}", f"sensors.noise.params.sigma={sigma}")
    cfg_off = _cfg("sources.count_law.params={lam: 4.0}", f"sensors.noise.params.sigma={sigma}",
                   "central_source.enabled=false")
    seed = _pick_seed(cfg_on, 4)
    sources = _draw(cfg_on, np.random.default_rng(seed))
    sensors = cfg_on.sensors
    t = sensors.angles
    signal = intensity_at(sources, sensors.positions)
    known = cfg_on.problem.known_contribution()

    rng = np.random.default_rng(0)
    y_on = cfg_on.noise(signal + known, rng)
    y_off = cfg_off.noise(signal, rng)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(t, y_on, ".", c="k", ms=4, label="mesures")
    ax.plot(t, signal + known, c=ORANGE, lw=1.5, label="valeur exacte")
    ax.axhline(known[0], c=GREY, ls="--", lw=1, label="contribution de la source centrale")
    ax.set_title("Mesures brutes : la source centrale ajoute une constante")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t, y_on - known - signal, "o-", c=ORANGE, ms=3, lw=0.8,
            label="avec source centrale (soustraite)")
    ax.plot(t, y_off - signal, "o-", c=BLUE, ms=3, lw=0.8, label="sans source centrale")
    ax.set_title("Bruit restant sur les sources inconnues (mesure - signal exact)")
    ax.legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks(np.linspace(0, 2 * np.pi, 5), ["0", r"$\pi/2$", r"$\pi$",
                                                       r"$3\pi/2$", r"$2\pi$"])
        ax.set_xlabel("angle t du capteur")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    ratio = float(np.median((signal + known) / signal))
    return _save(fig, "source_centrale.png"), float(known[0]), ratio


def benchmark(n_situations=200):
    """Banc d'essai bruit pour lambda = 1 (config actuelle) et lambda = 3."""
    out = {}
    for lam in (1.0, 3.0):
        overrides = [f"sources.count_law.params={{lam: {lam}}}",
                     f"benchmark.n_situations={n_situations}"]
        cfg = load_config(DEFAULT_CONFIG_DIR, overrides)
        spec = BenchmarkSpec.from_raw(cfg.raw)
        print(f"  banc d'essai lambda = {lam}")
        out[lam] = run_benchmark(DEFAULT_CONFIG_DIR, overrides, spec, verbose=False)

    rows = out[3.0]
    methods = [m for m in dict.fromkeys(r["method"] for r in rows) if m != CRB_LABEL]
    colors = dict(zip(methods, [BLUE, ORANGE]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("n_accuracy", "Proportion de N correct"),
              ("position_ratio", "Erreur de position / borne de Cramér-Rao"),
              ("ospa", "Distance OSPA moyenne (c = 2)")]
    for ax, (key, title) in zip(axes, panels):
        for m in methods:
            pts = sorted((r["sweep_value"], r[key]) for r in rows if r["method"] == m)
            ax.plot(*zip(*pts), "o-", c=colors[m], label=m)
        ax.set_xscale("log")
        ax.set_xlabel(r"bruit relatif $\sigma$")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylim(0, 1.02)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[1].axhspan(0.67, 0.83, color=GREY, alpha=0.25, label="niveau de la borne")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.4, 3)
    axes[2].set_yscale("log")
    axes[0].legend(fontsize=9)
    axes[1].legend(fontsize=9)
    fig.suptitle(r"N ~ Poisson($\lambda$ = 3), 64 capteurs, 200 situations par niveau de bruit",
                 y=1.02)
    fig.tight_layout()
    return _save(fig, "banc_essai.png"), out


def n_correct(bench: dict):
    """Version épurée du banc d'essai : proportion de N correct selon le bruit."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    styles = {("1.0", True): ("#B8B8B8", "-"), ("1.0", False): ("#B8B8B8", "--"),
              ("3.0", True): (BLUE, "-"), ("3.0", False): ("#7A7A7A", "--")}
    for lam in ("1.0", "3.0"):
        for mle in (True, False):
            pts = sorted((r["sweep_value"], r["n_accuracy"]) for r in bench[lam]
                         if r["method"].startswith("Prony +") == mle and r["method"].startswith("Prony"))
            color, ls = styles[(lam, mle)]
            label = f"λ = {lam[0]}, {'Prony + max. vraisemblance' if mle else 'Prony'}"
            ax.plot(*zip(*pts), ls, marker="o", c=color, lw=3 if (lam, mle) == ("3.0", True) else 2)
            pct = int(pts[-1][1] * 100 + 0.5 + 1e-9)  # arrondi au plus proche, comme le document
            ax.annotate(f"{label} : {pct} %", pts[-1], xytext=(10, -4),
                        textcoords="offset points", color=color, fontsize=11,
                        weight="bold" if (lam, mle) == ("3.0", True) else "normal")
    ax.set_xscale("log")
    ax.set_xlim(6e-6, 1e-1)
    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2], ["0,001 %", "0,01 %", "0,1 %", "1 %"])
    ax.set_xlabel("bruit relatif σ")
    ax.set_ylim(0, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Proportion de situations où le nombre de sources est correct", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, "n_correct.png")


def missed_sources(n_situations=300, sigma=1e-3):
    """Quelles sources la meilleure méthode rate-t-elle ?"""
    cfg = _cfg("sources.count_law.params={lam: 3.0}", f"sensors.noise.params.sigma={sigma}")
    rng = np.random.default_rng(1)
    all_I, missed_I = [], []
    under = over = 0
    for _ in range(n_situations):
        true = _draw(cfg, rng)
        y = measure(true + cfg.fixed_sources, cfg.sensors, rng, cfg.noise, cfg.eps)
        est = prony_mle(y, cfg.problem)
        all_I.extend(true.intensities)
        if est.n < true.n:
            under += 1
            i, _ = match(true, est)
            missed_I.extend(np.delete(true.intensities, i))
        elif est.n > true.n:
            over += 1
    all_I, missed_I = np.array(all_I), np.array(missed_I)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.logspace(-3, 2, 26)
    ax.hist(all_I, bins=bins, color=GREY, alpha=0.6, label="toutes les sources")
    ax.hist(missed_I, bins=bins, color=ORANGE, alpha=0.9, label="sources manquées")
    ax.set_xscale("log")
    ax.set_xlabel(r"intensité de la source ($I_{tot}$ = 100)")
    ax.set_ylabel("nombre de sources")
    ax.set_title(fr"Sources manquées par Prony + MV ($\lambda$ = 3, $\sigma$ = {sigma:g})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    stats = {"n": n_situations, "under": under, "over": over,
             "median_missed": float(np.median(missed_I)) if missed_I.size else float("nan"),
             "median_all": float(np.median(all_I)),
             "frac_small": float(np.mean(all_I < 1.0))}
    return _save(fig, "sources_manquees.png"), stats

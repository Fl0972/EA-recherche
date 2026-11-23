"""
Étude du maximum de vraisemblance glouton + BIC (``greedy_ml``) sur le
scénario du projet (source centrale 5000 retirée, bruit relatif σ).

    python sparse_sources/experiments_ml.py            (~5 min)
    python sparse_sources/experiments_ml.py --rapide
    python sparse_sources/experiments_ml.py --seulement detection resolution

Expériences :
    sigma_faux   le niveau de bruit σ est mal connu : la méthode reçoit
                 c × σ (c de 0,5 à 2). Effet sur K et sur la réussite.
    detection    une source seule d'intensité I (0,25 à 20) : probabilité que
                 le BIC la détecte, comparée à la théorie. Ajouter une source
                 bien placée fait baisser χ² d'environ SNR² + 3, avec
                 SNR = I ||φ_w(z)|| ; le BIC l'accepte si la baisse dépasse
                 3 log M, soit SNR > ~3,1 pour M = 64.
    resolution   deux sources d'intensité 50 à distance d : réussite et erreur
                 en fonction de d, comparées à la borne de Cramér-Rao.
    capteurs     nombre M de capteurs sur le cercle du projet (32 à 1024).
    beaucoup     jusqu'à 50 sources : nombre de directions de l'espace des
                 paramètres que les capteurs mesurent, et ce que ML retrouve.
    geometrie    où placer les capteurs (cercle agrandi, 1024 capteurs, grille
                 dans D) : directions mesurables et borne de Cramér-Rao.
    mobile       capteur mobile : 64 capteurs du cercle + N mesures placées
                 où l'on veut (adaptatif A-optimal, au hasard, grille).
    continu      capteur qui enregistre en continu le long d'un trajet en
                 serpentin, 50 sources : CRB, puis ML + BIC (long, ~1 h).
    frank_wolfe  Frank-Wolfe normalisé (réglage du projet et blanchi) avec 20 et
                 50 sources et des capteurs dans D (~5 min).

Sorties : sparse_sources/results/ml_*.csv et ml_*.png.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sparse_sources import (build_grid, cramer_rao, greedy_ml, make_setup,  # noqa: E402
                            match_sources, relative_noise_std)
from sparse_sources.experiments import OUT, save, success, write_csv  # noqa: E402
from sparse_sources.experiments_projet import GRID, SETUP, scenario  # noqa: E402
from sparse_sources.plotting import INK_MUTED, SERIES, style_axes  # noqa: E402

I_CENTRAL = 5000.0
RAW = SETUP.with_power(0.0)
CENTER = np.asarray(SETUP.domain.center, float)[None]


def synth(z, I, sigma, rng):
    """Mesures du projet pour des sources données : (m sans la centrale, s, y brut)."""
    central = I_CENTRAL * RAW.phi(CENTER)[0]
    y = (RAW.phi(np.atleast_2d(z)).T @ np.atleast_1d(I) if len(I) else 0.0) + central
    noisy = y * (1 + sigma * rng.standard_normal(len(y)))
    return noisy - central, relative_noise_std(y, sigma), y


def exp_wrong_sigma(n_seeds):
    """Le niveau de bruit fourni à la méthode est c × σ."""
    print("\n[1] Niveau de bruit mal connu")
    factors = [0.5, 0.8, 1.0, 1.25, 2.0]
    rows = []
    for n, sigma in [(2, 1e-3), (3, 1e-4), (3, 1e-3)]:
        for seed in range(n_seeds):
            cfg, src, y, m = scenario(n, sigma, seed)
            for c in factors:
                r = greedy_ml(m, relative_noise_std(y, c * sigma), setup=SETUP, grid=GRID)
                res = match_sources(src.positions, src.intensities, r["z"], r["I"])
                rows.append({"n": n, "sigma": sigma, "facteur": c, "seed": seed,
                             "n_est": res["n_est"], "err_pos": res["err_pos"],
                             "chi2_sur_M": r["chi2"] / SETUP.M * c**2,
                             "reussite": success(res, src.positions)})
        print(f"  n = {n}, σ = {sigma:g} fait")
    write_csv(rows, OUT / "ml_sigma_faux.csv")

    print(f"  {'cas':>14s} " + " ".join(f"c={c:<4g}" for c in factors))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (n, sigma) in enumerate([(2, 1e-3), (3, 1e-4), (3, 1e-3)]):
        g = [[r for r in rows if (r["n"], r["sigma"], r["facteur"]) == (n, sigma, c)]
             for c in factors]
        succ = [np.mean([r["reussite"] for r in x]) for x in g]
        kmean = [np.mean([r["n_est"] for r in x]) for x in g]
        print(f"  n={n}, σ={sigma:<6g} réussite " + " ".join(f"{100 * v:4.0f}%" for v in succ)
              + " | K moyen " + " ".join(f"{v:4.1f}" for v in kmean))
        lab = f"{n} sources, σ = {sigma:g}"
        axes[0].plot(factors, succ, "-o", color=SERIES[k], lw=2, ms=5, label=lab)
        axes[1].plot(factors, kmean, "-o", color=SERIES[k], lw=2, ms=5, label=lab)
        axes[1].axhline(n, color=SERIES[k], lw=1, ls=":")
    axes[0].set_ylabel("taux de réussite")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("nombre de sources estimé (moyenne)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(factors, [f"{c:g}" for c in factors])
        ax.set_xlabel("σ fourni / σ vrai")
        style_axes(ax)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Maximum de vraisemblance + BIC avec un niveau de bruit mal connu",
                 fontsize=10, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "ml_sigma_faux.png")


def exp_detection(n_seeds, sigma=1e-3):
    """Une source seule d'intensité I : détectée (K = 1) ou non ?"""
    print(f"\n[2] Seuil de détection (σ = {sigma:g})")
    intensities = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
    lo, hi = SETUP.domain.lo, SETUP.domain.hi
    rows = []
    for I in intensities:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            z = lo + (hi - lo) * rng.random(2)
            m, s, _ = synth(z[None] if I else np.zeros((0, 2)), [I] if I else [], sigma, rng)
            snr = float(I * np.linalg.norm(RAW.phi(z) / s))
            r = greedy_ml(m, s, setup=SETUP, grid=GRID)
            err = float(np.min(np.linalg.norm(r["z"] - z, axis=1))) if r["K"] else np.nan
            rows.append({"I": I, "seed": seed, "snr": snr, "K": r["K"],
                         "detectee": bool(r["K"] >= 1),
                         "localisee": bool(r["K"] >= 1 and err < 1.0), "err_pos": err,
                         "crb": float(cramer_rao(z[None], [I], s, SETUP)["pos_std"][0])
                         if I else np.nan})
        sel = [r for r in rows if r["I"] == I]
        print(f"  I = {I:5g} : SNR médian {np.median([r['snr'] for r in sel]):6.2f}, "
              f"K ≥ 1 : {100 * np.mean([r['detectee'] for r in sel]):3.0f} %, "
              f"à moins de 1 : {100 * np.mean([r['localisee'] for r in sel]):3.0f} %, "
              f"K > 1 : {100 * np.mean([r['K'] > 1 for r in sel]):3.0f} %, "
              f"CRB médiane {np.nanmedian([r['crb'] for r in sel]) if I else np.nan:.3g}")
    write_csv(rows, OUT / "ml_detection.csv")

    rows_s = [r for r in rows if r["I"] > 0]
    snr = np.array([r["snr"] for r in rows_s])
    det = np.array([r["detectee"] for r in rows_s], float)
    loc = np.array([r["localisee"] for r in rows_s], float)
    edges = np.geomspace(max(snr.min(), 0.1), snr.max(), 14)
    mids, rate, rate_loc = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (snr >= a) & (snr < b)
        if sel.sum() >= 5:
            mids.append(np.sqrt(a * b))
            rate.append(det[sel].mean())
            rate_loc.append(loc[sel].mean())
    from scipy.stats import ncx2
    thr = 3 * np.log(SETUP.M) - 3      # baisse de χ² attendue : SNR² + 3 ; seuil 3 log M
    xs = np.geomspace(edges[0], edges[-1], 200)
    theory = ncx2.sf(3 * np.log(SETUP.M), 3, xs**2)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(xs, theory, "-", color=INK_MUTED, lw=1.5,
            label="théorie : P(χ²₃(SNR²) > 3 log M)")
    ax.plot(mids, rate, "o", color=SERIES[0], ms=6, label="mesuré : K ≥ 1")
    ax.plot(mids, rate_loc, "s", color=SERIES[2], ms=6, label="mesuré : K ≥ 1 et erreur < 1")
    ax.axvline(np.sqrt(max(thr, 0)), color=SERIES[1], ls=":", lw=1.5,
               label=f"SNR² + 3 = 3 log M (SNR ≈ {np.sqrt(max(thr, 0)):.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("SNR = I ||φ(z) / s||")
    ax.set_ylabel("probabilité de détection")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_axes(ax)
    ax.set_title(f"Détection d'une source seule par le BIC (σ = {sigma:g}, source centrale)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "ml_detection.png")


def exp_resolution(n_seeds):
    """Deux sources d'intensité 50 à distance d."""
    print("\n[3] Résolution de deux sources")
    ds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    sigmas = [1e-5, 1e-4, 1e-3]
    lo, hi = SETUP.domain.lo, SETUP.domain.hi
    rows = []
    for sigma in sigmas:
        for d in ds:
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                for _ in range(1000):
                    c = lo + (hi - lo) * rng.random(2)
                    t = rng.uniform(0, np.pi)
                    u = 0.5 * d * np.array([np.cos(t), np.sin(t)])
                    z = np.array([c - u, c + u])
                    if np.all((z >= lo) & (z <= hi)):
                        break
                I = np.array([50.0, 50.0])
                m, s, _ = synth(z, I, sigma, rng)
                crb = float(cramer_rao(z, I, s, SETUP)["pos_std"].max())
                r = greedy_ml(m, s, setup=SETUP, grid=GRID)
                res = match_sources(z, I, r["z"], r["I"])
                rows.append({"sigma": sigma, "d": d, "seed": seed, "n_est": res["n_est"],
                             "err_pos": res["err_pos"], "crb_max": crb,
                             "reussite": success(res, z)})
            sel = [r for r in rows if (r["sigma"], r["d"]) == (sigma, d)]
            print(f"  σ = {sigma:g}, d = {d:3g} : réussite "
                  f"{100 * np.mean([r['reussite'] for r in sel]):3.0f} %, K = 1 : "
                  f"{100 * np.mean([r['n_est'] == 1 for r in sel]):3.0f} %, CRB médiane "
                  f"{np.median([r['crb_max'] for r in sel]):.3g}")
    write_csv(rows, OUT / "ml_resolution.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, sigma in enumerate(sigmas):
        g = [[r for r in rows if (r["sigma"], r["d"]) == (sigma, d)] for d in ds]
        axes[0].plot(ds, [np.mean([r["reussite"] for r in x]) for x in g], "-o",
                     color=SERIES[k], lw=2, ms=5, label=f"σ = {sigma:g}")
        axes[1].plot(ds, [np.nanmedian([r["err_pos"] for r in x]) for x in g], "-o",
                     color=SERIES[k], lw=2, ms=5, label=f"erreur, σ = {sigma:g}")
        axes[1].plot(ds, [np.median([r["crb_max"] for r in x]) for x in g], ":",
                     color=SERIES[k], lw=1.5, label=f"CRB, σ = {sigma:g}")
    axes[0].set_ylabel("taux de réussite")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("erreur de position (médiane)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("distance d entre les deux sources")
        style_axes(ax)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.suptitle("Deux sources d'intensité 50 (source centrale du projet)", fontsize=10,
                 color=INK_MUTED)
    fig.tight_layout()
    save(fig, "ml_resolution.png")


def exp_sensors(n_seeds):
    """Scénario du projet avec M capteurs sur le même cercle."""
    print("\n[4] Nombre de capteurs")
    Ms = [32, 64, 256, 1024]
    cases = [(3, 1e-3), (5, 1e-4), (3, 1e-2)]
    rows = []
    for M in Ms:
        S = make_setup(h=0.0, normalize=True, n_sensors=M)
        G = build_grid(step=0.25, setup=S)
        for n, sigma in cases:
            for seed in range(n_seeds):
                cfg, src, y, m = scenario(n, sigma, seed,
                                          extra=[f"sensors.layout.params={{n: {M}, t0: 0.0}}"])
                s = relative_noise_std(y, sigma)
                crb = float(cramer_rao(src.positions, src.intensities, s, S)["pos_std"].max())
                r = greedy_ml(m, s, setup=S, grid=G)
                res = match_sources(src.positions, src.intensities, r["z"], r["I"])
                rows.append({"M": M, "n": n, "sigma": sigma, "seed": seed, "crb_max": crb,
                             "n_est": res["n_est"], "err_pos": res["err_pos"],
                             "temps": r["time"], "reussite": success(res, src.positions)})
            sel = [r for r in rows if (r["M"], r["n"], r["sigma"]) == (M, n, sigma)]
            print(f"  M = {M:4d}, n = {n}, σ = {sigma:g} : réussite "
                  f"{100 * np.mean([r['reussite'] for r in sel]):3.0f} %, CRB médiane "
                  f"{np.median([r['crb_max'] for r in sel]):.3g}, temps médian "
                  f"{np.median([r['temps'] for r in sel]):.2f} s")
    write_csv(rows, OUT / "ml_capteurs.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, (n, sigma) in enumerate(cases):
        g = [[r for r in rows if (r["M"], r["n"], r["sigma"]) == (M, n, sigma)] for M in Ms]
        lab = f"{n} sources, σ = {sigma:g}"
        axes[0].plot(Ms, [np.mean([r["reussite"] for r in x]) for x in g], "-o",
                     color=SERIES[k], lw=2, ms=5, label=lab)
        axes[1].plot(Ms, [np.median([r["crb_max"] for r in x]) for x in g], "-o",
                     color=SERIES[k], lw=2, ms=5, label=lab)
    axes[0].set_ylabel("taux de réussite (ML + BIC)")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("borne de Cramér-Rao (médiane)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ms, [str(M) for M in Ms])
        ax.set_xlabel("nombre de capteurs M (même cercle)")
        style_axes(ax)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Plus de capteurs : la borne baisse comme 1/√M", fontsize=10, color=INK_MUTED)
    fig.tight_layout()
    save(fig, "ml_capteurs.png")


def measurable_directions(z, I, sensors, rel=1e-6):
    """
    Nombre de directions de l'espace des paramètres (3K) que les capteurs
    mesurent : valeurs singulières de D (colonnes normalisées, voir
    ``cramer_rao``) supérieures à rel × la plus grande. Renvoie (n, cond).
    """
    from particle_sources.duval_peyre.kernel import InverseSquareKernel
    k = InverseSquareKernel(sensors)
    K, M = len(I), len(sensors)
    D = np.hstack([k.phi(z).T, (k.grad(z) * I[:, None, None]).transpose(1, 0, 2)
                   .reshape(M, 2 * K)])
    sv = np.linalg.svd(D / np.linalg.norm(D, axis=0), compute_uv=False)
    return int(np.sum(sv > rel * sv[0])), float(sv[0] / sv[-1])


def found_within(z_true, z_hat, tol=0.5):
    """Nombre de vraies sources ayant une source estimée à moins de tol (appariement 1-1)."""
    from scipy.optimize import linear_sum_assignment
    if len(z_hat) == 0:
        return 0
    C = np.linalg.norm(z_true[:, None] - np.asarray(z_hat)[None], axis=-1)
    a, b = linear_sum_assignment(C)
    return int(np.sum(C[a, b] < tol))


def exp_many(n_seeds):
    """Beaucoup de sources (jusqu'à 50) : ce que les mesures contiennent, ce que ML retrouve."""
    print("\n[5] Beaucoup de sources")
    lo, hi = SETUP.domain.lo, SETUP.domain.hi
    Ks = [1, 3, 5, 10, 20, 30, 50]
    rows = []
    for M in [64, 1024]:
        S = make_setup(h=0.0, n_sensors=M)
        for K in Ks:
            for seed in range(n_seeds):
                r = np.random.default_rng(seed)
                z = lo + (hi - lo) * r.random((K, 2))
                I = 100.0 * r.dirichlet(np.ones(K))
                n, cond = measurable_directions(z, I, S.sensors)
                sep = float(np.min(np.linalg.norm(z[:, None] - z[None], axis=-1)
                                   + np.eye(K) * 1e9)) if K > 1 else np.inf
                rows.append({"M": M, "K": K, "seed": seed, "parametres": 3 * K,
                             "directions_mesurables": n, "cond": cond, "sep_min": sep})
            sel = [x for x in rows if (x["M"], x["K"]) == (M, K)]
            print(f"  M = {M:4d}, K = {K:2d} : {3 * K:3d} paramètres, directions mesurables "
                  f"(sv > 1e-6) {np.median([x['directions_mesurables'] for x in sel]):.0f}, "
                  f"séparation min médiane {np.median([x['sep_min'] for x in sel]):.2f}")
    write_csv(rows, OUT / "ml_beaucoup_directions.csv")

    print("  ML + BIC (64 capteurs, source centrale, k_max = 60) :")
    runs = []
    for K in [10, 20, 50]:
        for sigma in [1e-4, 1e-3]:
            for seed in range(n_seeds):
                r = np.random.default_rng(seed)
                z = lo + (hi - lo) * r.random((K, 2))
                I = 100.0 * r.dirichlet(np.ones(K))
                m, s, _ = synth(z, I, sigma, r)
                res = greedy_ml(m, s, setup=SETUP, grid=GRID, k_max=60)
                runs.append({"K": K, "sigma": sigma, "seed": seed, "n_est": res["K"],
                             "trouvees_a_0_5": found_within(z, res["z"]),
                             "chi2_sur_M": res["chi2"] / SETUP.M, "temps": res["time"]})
            sel = [x for x in runs if (x["K"], x["sigma"]) == (K, sigma)]
            print(f"    K = {K:2d}, σ = {sigma:g} : K estimé {np.median([x['n_est'] for x in sel]):.0f}, "
                  f"vraies sources retrouvées à 0,5 près "
                  f"{np.median([x['trouvees_a_0_5'] for x in sel]):.0f}, temps médian "
                  f"{np.median([x['temps'] for x in sel]):.1f} s")
    write_csv(runs, OUT / "ml_beaucoup_ml.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for k, M in enumerate([64, 1024]):
        g = [[x for x in rows if (x["M"], x["K"]) == (M, K)] for K in Ks]
        ax.plot(Ks, [np.median([x["directions_mesurables"] for x in y]) for y in g], "-o",
                color=SERIES[k], lw=2, ms=5, label=f"directions mesurables, M = {M}")
    ax.plot(Ks, [3 * K for K in Ks], ":", color=INK_MUTED, lw=1.5, label="paramètres 3K")
    ax.set_xlabel("nombre de sources K")
    ax.set_ylabel("nombre de directions")
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    ax.set_title("Ce que mesure le cercle de capteurs (sans bruit, seuil 1e-6)", fontsize=10)
    fig.tight_layout()
    save(fig, "ml_beaucoup.png")


def interior_grid(nx, ny, box=None):
    """Grille nx × ny de capteurs à l'intérieur de D (centres des cases, évite le centre de D)."""
    box = box or SETUP.domain
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


def geometries():
    """Dispositions de capteurs comparées par ``exp_geometry`` : nom -> (M, 2)."""
    circle = SETUP.sensors
    return {"cercle projet": circle,
            "cercle rayon x2": make_setup(h=0.0, radius_factor=2.0).sensors,
            "cercle 1024": make_setup(h=0.0, n_sensors=1024).sensors,
            "cercle + grille 4x8": np.vstack([circle, interior_grid(4, 8)]),
            "cercle + grille 8x16": np.vstack([circle, interior_grid(8, 16)]),
            "cercle + grille 16x32": np.vstack([circle, interior_grid(16, 32)])}


def exp_geometry(n_seeds):
    """
    Où placer les capteurs : cercle agrandi, plus de capteurs, capteurs dans D.
    Pour K sources tirées comme dans config/ : directions mesurables (sans
    bruit) et borne de Cramér-Rao avec le bruit du projet (source centrale).
    """
    from sparse_sources.kernel import Setup
    print("\n[6] Disposition des capteurs")
    lo, hi = SETUP.domain.lo, SETUP.domain.hi
    rows = []
    for name, sens in geometries().items():
        raw = Setup(sens, SETUP.domain, 0.0)
        central = I_CENTRAL * raw.phi(CENTER)[0]
        for K in [3, 10, 20, 50]:
            for seed in range(n_seeds):
                r = np.random.default_rng(seed)
                z = lo + (hi - lo) * r.random((K, 2))
                I = 100.0 * r.dirichlet(np.ones(K))
                y = raw.phi(z).T @ I + central
                row = {"geometrie": name, "M": len(sens), "K": K, "seed": seed,
                       "parametres": 3 * K,
                       "directions_mesurables": measurable_directions(z, I, sens)[0]}
                for sigma in [1e-4, 1e-3]:
                    crb = cramer_rao(z, I, relative_noise_std(y, sigma), raw)["pos_std"]
                    row[f"crb_mediane_{sigma:g}"] = float(np.median(crb))
                    row[f"part_crb_0_1_{sigma:g}"] = float(np.mean(crb < 0.1))
                rows.append(row)
            sel = [x for x in rows if (x["geometrie"], x["K"]) == (name, K)]
            med = lambda key: np.median([x[key] for x in sel])  # noqa: E731
            print(f"  {name:22s} M = {len(sens):4d}, K = {K:2d} : directions "
                  f"{med('directions_mesurables'):.0f} / {3 * K}, CRB médiane "
                  f"{med('crb_mediane_0.0001'):.3g} (σ=1e-4), {med('crb_mediane_0.001'):.3g} "
                  f"(σ=1e-3), sources à CRB < 0,1 : {100 * med('part_crb_0_1_0.001'):.0f} % "
                  f"(σ=1e-3)")
    write_csv(rows, OUT / "ml_geometrie.csv")

    names = list(geometries())
    Ks = [3, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for k, name in enumerate(names):
        g = [[x for x in rows if (x["geometrie"], x["K"]) == (name, K)] for K in Ks]
        ax.plot(Ks, [np.median([x["directions_mesurables"] for x in y]) for y in g], "-o",
                color=SERIES[k % len(SERIES)], lw=2, ms=5, label=name)
    ax.plot(Ks, [3 * K for K in Ks], ":", color=INK_MUTED, lw=1.5, label="paramètres 3K")
    ax.set_xlabel("nombre de sources K")
    ax.set_ylabel("directions mesurables")
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    ax.set_title("Directions mesurées selon la disposition des capteurs (sans bruit)",
                 fontsize=10)
    fig.tight_layout()
    save(fig, "ml_geometrie.png")


def exp_mobile(n_seeds, Ks=(10, 20), sigma=1e-3):
    """
    Capteur mobile : 64 capteurs du cercle + N mesures d'un capteur que l'on
    place où l'on veut dans D. Trois façons de choisir les N positions :
    adaptatif (A-optimal sur les sources estimées par ML + BIC, par paquets
    de 8), au hasard, grille régulière. Positions prises parmi les mêmes
    candidats (centres des cases de pas 0,25).
    """
    from sparse_sources.kernel import Setup
    from sparse_sources.mobile import adaptive_ml, candidate_points
    from sparse_sources.ml3d import match_nd, success_nd
    print("\n[7] Capteur mobile")
    box = SETUP.domain
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    cand = candidate_points(box)
    Ns = {8: (2, 4), 32: (4, 8), 72: (6, 12), 128: (8, 16)}
    rows = []

    def record(K, seed, strat, n, z, I, z_hat, I_hat, t):
        res = match_nd(z, I, z_hat, I_hat)
        rows.append({"K": K, "seed": seed, "strategie": strat, "n_mobile": n,
                     "n_est": len(I_hat), "trouvees_0_5": found_within(z, z_hat, 0.5),
                     "trouvees_0_1": found_within(z, z_hat, 0.1),
                     "reussite": success_nd(res, z), "temps": t})

    for K in Ks:
        for seed in range(n_seeds):
            r = np.random.default_rng(seed)
            z = lo + (hi - lo) * r.random((K, 2))
            I = 100.0 * r.dirichlet(np.ones(K))
            noise = np.random.default_rng(1000 + seed)

            def measure(pts):
                raw = Setup(np.atleast_2d(pts), box, 0.0)
                c = I_CENTRAL * raw.phi(CENTER)[0]
                y = raw.phi(z).T @ I + c
                return y * (1 + sigma * noise.standard_normal(len(y))) - c, y

            m0, y0 = measure(SETUP.sensors)

            def fit(sensors, m, y):
                S = Setup(sensors, box, 0.0, normalize=True)
                t0 = time.perf_counter()
                out = greedy_ml(m, relative_noise_std(y, sigma), setup=S,
                                grid=build_grid(step=0.25, setup=S), k_max=K + 15)
                return out, time.perf_counter() - t0

            res, t = fit(SETUP.sensors, m0, y0)
            for strat in ["adaptatif", "hasard", "grille"]:
                record(K, seed, strat, 0, z, I, res["z"], res["I"], t)
            for st in adaptive_ml(measure, SETUP.sensors, m0, y0, box, sigma, max(Ns),
                                  checkpoints=Ns, k_max=K + 15):
                record(K, seed, "adaptatif", st["n_mobile"], z, I, st["z"], st["I"], st["time"])
            for n, (nx, ny) in Ns.items():
                pick = np.random.default_rng(2000 + seed).choice(len(cand), n, replace=False)
                grid_pts = interior_grid(nx, ny)
                snap = cand[np.argmin(np.linalg.norm(grid_pts[:, None] - cand[None], axis=-1), 1)]
                for strat, pts in [("hasard", cand[pick]), ("grille", snap)]:
                    m1, y1 = measure(pts)
                    res, t = fit(np.vstack([SETUP.sensors, pts]), np.concatenate([m0, m1]),
                                 np.concatenate([y0, y1]))
                    record(K, seed, strat, n, z, I, res["z"], res["I"], t)
            print(f"  K = {K}, tirage {seed} : " + ", ".join(
                f"{x['strategie']} N={x['n_mobile']} -> {x['trouvees_0_1']}"
                for x in rows if (x["K"], x["seed"]) == (K, seed) and x["n_mobile"] in (32, 128)),
                flush=True)
        for strat in ["adaptatif", "hasard", "grille"]:
            line = []
            for n in [0, *Ns]:
                sel = [x for x in rows if (x["K"], x["strategie"], x["n_mobile"]) == (K, strat, n)]
                line.append(f"N={n}: {np.mean([x['trouvees_0_1'] for x in sel]):.1f}")
            print(f"  K = {K}, {strat:9s} : sources retrouvées à 0,1 près (moyenne) " + ", ".join(line))
    write_csv(rows, OUT / "ml_mobile.csv")

    fig, axes = plt.subplots(1, len(Ks), figsize=(4.2 * len(Ks), 3.6), squeeze=False)
    for ax, K in zip(axes[0], Ks):
        for k, strat in enumerate(["adaptatif", "hasard", "grille"]):
            xs = [0, *Ns]
            ys = [np.mean([x["trouvees_0_1"] for x in rows
                           if (x["K"], x["strategie"], x["n_mobile"]) == (K, strat, n)]) for n in xs]
            ax.plot(xs, ys, "-o", color=SERIES[k], lw=2, ms=5, label=strat)
        ax.axhline(K, color=INK_MUTED, ls=":", lw=1.5)
        ax.set_xlabel("mesures du capteur mobile N")
        ax.set_ylabel("sources retrouvées à 0,1 près")
        ax.set_title(f"{K} sources, σ = {sigma:g}", fontsize=10)
        style_axes(ax)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "ml_mobile.png")


def exp_continuous(n_seeds, K=50, sigma=1e-3, ml_seeds=(0,)):
    """
    Capteur qui enregistre en continu le long d'un trajet en serpentin
    (``mobile.serpentine``), en plus des 64 capteurs du cercle. Pour K
    sources : directions mesurables et CRB (n_seeds tirages), puis ML + BIC
    sur les tirages ``ml_seeds`` (long : 10 à 30 min par reconstruction).
    """
    from sparse_sources.kernel import Setup
    from sparse_sources.mobile import serpentine
    from sparse_sources.ml3d import match_nd, success_nd
    print(f"\n[8] Capteur en enregistrement continu, {K} sources")
    box = SETUP.domain
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    paths = {"cercle seul": np.zeros((0, 2)),
             "grille fixe 16x32": interior_grid(16, 32)}
    for nl, st in [(4, 0.25), (8, 0.25), (8, 0.1), (16, 0.25)]:
        paths[f"serpentin {nl} lignes, pas {st:g}"] = serpentine(box, nl, st)
    rows = []
    for name, pts in paths.items():
        sens = np.vstack([SETUP.sensors, pts])
        raw = Setup(sens, box, 0.0)
        central = I_CENTRAL * raw.phi(CENTER)[0]
        for seed in range(n_seeds):
            r = np.random.default_rng(seed)
            z = lo + (hi - lo) * r.random((K, 2))
            I = 100.0 * r.dirichlet(np.ones(K))
            y = raw.phi(z).T @ I + central
            crb = cramer_rao(z, I, relative_noise_std(y, sigma), raw)["pos_std"]
            rows.append({"trajet": name, "mesures_mobiles": len(pts),
                         "longueur_trajet": float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                         if len(pts) > 1 else 0.0,
                         "seed": seed, "directions_mesurables": measurable_directions(z, I, sens)[0],
                         "crb_mediane": float(np.nanmedian(crb)),
                         "part_crb_0_1": float(np.mean(crb < 0.1))})
        sel = [x for x in rows if x["trajet"] == name]
        print(f"  {name:26s} {len(pts):5d} mesures, trajet {sel[0]['longueur_trajet']:5.0f} : "
              f"directions {np.median([x['directions_mesurables'] for x in sel]):.0f} / {3 * K}, "
              f"CRB médiane {np.median([x['crb_mediane'] for x in sel]):.3g}, sources à CRB < 0,1 : "
              f"{100 * np.mean([x['part_crb_0_1'] for x in sel]):.0f} %", flush=True)
    write_csv(rows, OUT / "ml_continu_crb.csv")

    runs = []
    for name in ["serpentin 4 lignes, pas 0.25", "serpentin 8 lignes, pas 0.25"]:
        pts = paths[name]
        sens = np.vstack([SETUP.sensors, pts])
        S = Setup(sens, box, 0.0, normalize=True)
        raw = S.with_power(0.0)
        central = I_CENTRAL * raw.phi(CENTER)[0]
        for seed in ml_seeds:
            r = np.random.default_rng(seed)
            z = lo + (hi - lo) * r.random((K, 2))
            I = 100.0 * r.dirichlet(np.ones(K))
            y = raw.phi(z).T @ I + central
            m = y * (1 + sigma * r.standard_normal(len(y))) - central
            s = relative_noise_std(y, sigma)
            crb = cramer_rao(z, I, s, raw)["pos_std"]
            res = greedy_ml(m, s, setup=S, grid=build_grid(step=0.25, setup=S), k_max=K + 15)
            mt = match_nd(z, I, res["z"], res["I"])
            runs.append({"trajet": name, "M": S.M, "seed": seed, "n_est": res["K"],
                         "chi2_sur_M": res["chi2"] / S.M,
                         "trouvees_0_5": found_within(z, res["z"], 0.5),
                         "trouvees_0_1": found_within(z, res["z"], 0.1),
                         "crb_inf_0_1": int(np.sum(crb < 0.1)),
                         "reussite": success_nd(mt, z), "temps": res["time"]})
            x = runs[-1]
            print(f"  ML + BIC, {name}, tirage {seed} : K estimé {x['n_est']}, χ²/M "
                  f"{x['chi2_sur_M']:.2f}, retrouvées à 0,5 : {x['trouvees_0_5']}, à 0,1 : "
                  f"{x['trouvees_0_1']} (CRB < 0,1 : {x['crb_inf_0_1']}), {x['temps']:.0f} s",
                  flush=True)
    write_csv(runs, OUT / "ml_continu_ml.csv")


def exp_fw_many(seeds=(0,), sigma=1e-3):
    """
    Frank-Wolfe glissant normalisé avec beaucoup de sources et des capteurs
    dans D : réglage du projet (λ = 1,5 σ ||y||) et version blanchie
    (capteurs divisés par s_i, λ = 1,5 √M). Mêmes tirages et même bruit que
    les reconstructions ML + BIC de COMPTE_RENDU_CAPTEURS.md (graine 0).
    """
    import warnings
    from sparse_sources import sliding_frank_wolfe
    from sparse_sources.kernel import Setup
    from sparse_sources.mobile import serpentine
    print("\n[9] Frank-Wolfe normalisé, beaucoup de sources")
    box = SETUP.domain
    lo, hi = np.asarray(box.lo, float), np.asarray(box.hi, float)
    cases = [(20, "grille 8x16", interior_grid(8, 16)), (50, "cercle seul", np.zeros((0, 2))),
             (50, "grille 16x32", interior_grid(16, 32)),
             (50, "serpentin 8 lignes", serpentine(box, 8, 0.25))]
    rows = []
    for K, name, pts in cases:
        sens = np.vstack([SETUP.sensors, pts])
        base = Setup(sens, box, 0.0, normalize=True)
        raw = base.with_power(0.0)
        for seed in seeds:
            r = np.random.default_rng(seed)
            z = lo + (hi - lo) * r.random((K, 2))
            I = 100.0 * r.dirichlet(np.ones(K))
            c = I_CENTRAL * raw.phi(CENTER)[0]
            y = raw.phi(z).T @ I + c
            m = y * (1 + sigma * r.standard_normal(len(y))) - c
            s = relative_noise_std(y, sigma)
            crb = cramer_rao(z, I, s, raw)["pos_std"]
            for variant in ["projet", "blanchi"]:
                if variant == "projet":
                    S, mm, lam = base, m, 1.5 * sigma * float(np.linalg.norm(y))
                else:
                    S = Setup(sens, box, 0.0, normalize=True, weights=1.0 / s)
                    mm, lam = S.whiten(m), 1.5 * np.sqrt(len(m))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fw = sliding_frank_wolfe(mm, lam, setup=S, grid=build_grid(step=0.25, setup=S),
                                             tol=1e-5, max_iter=K + 30)
                zf = np.atleast_2d(fw["z_debiased"]).reshape(-1, 2)
                rows.append({"K": K, "capteurs": name, "M": len(sens), "seed": seed,
                             "reglage": variant, "n_est": len(zf),
                             "trouvees_0_5": found_within(z, zf, 0.5),
                             "trouvees_0_1": found_within(z, zf, 0.1),
                             "crb_inf_0_1": int(np.sum(crb < 0.1)), "temps": fw["time"]})
                x = rows[-1]
                print(f"  K = {K}, {name:18s}, {variant:7s} : K estimé {x['n_est']}, retrouvées à "
                      f"0,5 : {x['trouvees_0_5']}, à 0,1 : {x['trouvees_0_1']} (CRB < 0,1 : "
                      f"{x['crb_inf_0_1']}), {x['temps']:.0f} s", flush=True)
    write_csv(rows, OUT / "ml_fw_beaucoup.csv")


def main():
    """Lance les expériences demandées."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--rapide", action="store_true", help="moins de tirages")
    p.add_argument("--seulement", nargs="+", choices=["sigma_faux", "detection", "resolution", "capteurs",
                                                  "beaucoup", "geometrie", "mobile", "continu",
                                  "frank_wolfe"])
    args = p.parse_args()
    n = 4 if args.rapide else 20
    todo = set(args.seulement or ["sigma_faux", "detection", "resolution", "capteurs",
                                  "beaucoup", "geometrie", "mobile", "continu",
                                  "frank_wolfe"])
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    if "sigma_faux" in todo:
        exp_wrong_sigma(n)
    if "detection" in todo:
        exp_detection(2 * n)
    if "resolution" in todo:
        exp_resolution(n)
    if "capteurs" in todo:
        exp_sensors(n)
    if "beaucoup" in todo:
        exp_many(max(n // 4, 2))
    if "geometrie" in todo:
        exp_geometry(max(n // 4, 2))
    if "mobile" in todo:
        exp_mobile(max(n // 4, 2))
    if "continu" in todo:
        exp_continuous(max(n // 4, 2))
    if "frank_wolfe" in todo:
        exp_fw_many()
    print(f"\nTerminé en {time.perf_counter() - t0:.0f} s.")


if __name__ == "__main__":
    main()

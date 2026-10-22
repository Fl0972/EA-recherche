"""
Expériences : carte de eta_V, balayages de la condition de source non
dégénérée, et validation du BLASSO sur données bruitées.

Chaque fonction de balayage renvoie une liste de lignes (dict) prête pour un
CSV ; les fonctions `plot_*` renvoient une figure matplotlib. La géométrie
(domaine D, cercle des capteurs) est celle de la config du projet ; la
profondeur h est le paramètre `eps` du noyau, G = 1 / (r^2 + h^2) : c'est la
loi en 1/distance^2 pour des capteurs placés à la hauteur h au-dessus du plan
des sources (h = 0 : capteurs dans le plan, modèle par défaut du projet).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from ..model import Box
from .blasso import debias, lasso_grid, match, operator_norm, sliding_frank_wolfe
from .certificate import Diagnostic, diagnose
from .kernel import InverseSquareKernel


# ---- Géométrie --------------------------------------------------------------------
def circle_positions(center, radius, n, t0=0.0, aperture=2 * np.pi) -> np.ndarray:
    """
    n capteurs sur l'arc d'ouverture `aperture` du cercle (center, radius),
    à partir de l'angle t0. aperture = 2 pi : cercle complet (comme CircleSensors).
    """
    full = np.isclose(aperture, 2 * np.pi)
    t = t0 + aperture * np.arange(n) / (n if full else n - 1)
    return np.asarray(center) + radius * np.stack([np.cos(t), np.sin(t)], axis=-1)


def pair(center, d, angle=np.pi / 2) -> np.ndarray:
    """Deux sources séparées de d, centrées en `center`, alignées selon `angle`."""
    u = np.array([np.cos(angle), np.sin(angle)])
    return np.asarray(center, float) + np.outer([-0.5, 0.5], d * u)


def write_csv(rows, path) -> None:
    """Écrit une liste de dicts (mêmes clés) en CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def threshold(xs, ok, from_above=True):
    """
    Seuil d'un balayage monotone attendu : plus petite valeur x telle que la
    condition tienne pour tout x' >= x (from_above) ; nan si elle ne tient pas
    à la plus grande valeur.
    """
    xs, ok = np.asarray(xs), np.asarray(ok, bool)
    order = np.argsort(xs)[::-1] if from_above else np.argsort(xs)
    thr = np.nan
    for i in order:
        if not ok[i]:
            break
        thr = xs[i]
    return thr


# ---- Figure : carte de eta_V ----------------------------------------------------------
def plot_certificate(diag: Diagnostic, box: Box, sensors: np.ndarray, ax=None, title=None):
    """
    Carte de chaleur de eta_V sur D (palette divergente centrée sur 1, écrêtée
    pour rester lisible), niveau 1 en noir, capteurs et sources marqués.
    Nécessite un diagnostic calculé avec keep_grid=True.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 8))
    g = diag.grid
    vals = g["eta"]
    lo = min(np.percentile(vals, 1), 0.9)
    hi = max(np.percentile(vals, 99), 1.1)
    im = ax.pcolormesh(g["xs"], g["ys"], np.clip(vals, lo, hi), shading="auto",
                       cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=1.0, vmin=lo, vmax=hi))
    ax.contour(g["xs"], g["ys"], vals, levels=[1.0], colors="k", linewidths=1.2)
    ax.add_patch(Rectangle(box.lo, *box.size, fill=False, ec="0.3", lw=1))
    ax.plot(*sensors.T, "o", c="darkorange", ms=3, label="capteurs")
    ax.plot(*diag.z.T, "*", c="lime", mec="k", ms=14, label="sources $z_k$")
    if diag.margin < 0:
        ax.plot(*diag.argmax, "x", c="k", ms=10, mew=2, label=r"max de $\eta_V$ hors boules")
    ax.set_aspect("equal")
    ax.set_xlim(min(box.lo[0], sensors[:, 0].min()) - 0.5, max(box.hi[0], sensors[:, 0].max()) + 0.5)
    ax.set_ylim(min(box.lo[1], sensors[:, 1].min()) - 0.5, max(box.hi[1], sensors[:, 1].max()) + 0.5)
    verdict = "condition vérifiée" if diag.nondegenerate else "condition NON vérifiée"
    flags = f"H1={'V' if diag.h1 else 'F'}  H2={'V' if diag.h2 else 'F'}  H3={'V' if diag.h3 else 'F'}"
    ax.set_title((title + "\n" if title else "") + f"{verdict} ({flags}), marge = {diag.margin:.3g}",
                 fontsize=10)
    plt.colorbar(im, ax=ax, label=r"$\eta_V$ (écrêté)", shrink=0.8)
    ax.legend(loc="lower right", fontsize=8)
    return ax


# ---- Balayages -------------------------------------------------------------------------
def sweep_separation(box, sensors, center, distances, depths, h=0.08):
    """a) Deux sources de séparation d (verticales, centrées en `center`), pour chaque profondeur."""
    rows = []
    for depth in depths:
        k = InverseSquareKernel(sensors, depth)
        for d in distances:
            diag = diagnose(k, pair(center, d), box, h=h)
            rows.append({"balayage": "separation", "profondeur": depth, "d": d, **diag.as_row()})
    return rows


def sweep_depth(box, sensors, configs: dict, depths, h=0.08):
    """b1) Profondeur h des capteurs, pour plusieurs configurations nommées {nom: z}."""
    rows = []
    for name, z in configs.items():
        for depth in depths:
            diag = diagnose(InverseSquareKernel(sensors, depth), z, box, h=h)
            rows.append({"balayage": "profondeur", "config": name, "profondeur": depth,
                         **diag.as_row()})
    return rows


def sweep_offset(box, sensors, center, d, offsets, depth, h=0.08):
    """
    b2) Une paire de séparation d translatée de `offset` vers le haut de D :
    on se rapproche des capteurs (distance au capteur le plus proche en sortie).
    """
    rows = []
    k = InverseSquareKernel(sensors, depth)
    for off in offsets:
        z = pair(np.asarray(center) + [0, off], d)
        dist = float(np.min(np.linalg.norm(z[:, None] - sensors[None], axis=-1)))
        rows.append({"balayage": "distance_capteurs", "profondeur": depth, "decalage": off,
                     "dist_capteur": dist, **diagnose(k, z, box, h=h).as_row()})
    return rows


def sweep_sensors(box, center, radius, z, ms, depth, apertures, h=0.08):
    """c) Nombre M de capteurs et ouverture de l'arc qui les porte (2 pi = cercle complet)."""
    rows = []
    for ap in apertures:
        for m in ms:
            S = circle_positions(center, radius, m, t0=np.pi / 2 - ap / 2, aperture=ap)
            diag = diagnose(InverseSquareKernel(S, depth), z, box, h=h)
            rows.append({"balayage": "capteurs", "profondeur": depth, "ouverture": ap,
                         **diag.as_row()})
    return rows


def plot_sweep(rows, x, group, xlabel, title, logx=False):
    """
    Deux panneaux partageant l'axe x : marge (en haut, ligne y = 0 = seuil) et
    conditionnement de Gamma (en bas, échelle log). Une courbe par valeur de
    `group` ; points pleins = condition vérifiée, creux = non vérifiée.
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    for gval in dict.fromkeys(r[group] for r in rows):
        sub = [r for r in rows if r[group] == gval]
        xs = np.array([r[x] for r in sub])
        mg = np.array([r["marge"] for r in sub])
        ok = np.array([r["non_degenere"] for r in sub])
        label = f"{group} = {gval:.3g}" if isinstance(gval, float) else f"{group} = {gval}"
        line, = a1.plot(xs, mg, "-", lw=2, label=label)
        a1.plot(xs[ok], mg[ok], "o", c=line.get_color(), ms=6)
        a1.plot(xs[~ok], mg[~ok], "o", mfc="none", c=line.get_color(), ms=6)
        a2.semilogy(xs, [r["cond"] for r in sub], "-", lw=2, c=line.get_color(), label=label)
    a1.axhline(0, c="k", lw=0.8)
    a1.set_yscale("symlog", linthresh=1e-5)
    a1.set_ylabel("marge 1 - max $\\eta_V$ hors boules\n(échelle symlog)")
    a1.set_title(title + "\npoints pleins : condition vérifiée ; creux : non vérifiée", fontsize=10)
    a2.set_ylabel(r"cond($\Gamma$)")
    a2.set_xlabel(xlabel)
    for a in (a1, a2):
        a.grid(alpha=0.3)
        if logx:
            a.set_xscale("log")
    a1.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ---- Validation BLASSO sur données bruitées -------------------------------------------------
def noise_experiment(box, sensors, z, intensities, depth, noise_levels, constants,
                     n_rep=5, h=0.1, seed=0, name=""):
    """
    m = A(s*) + eps, eps ~ N(0, sigma^2 I) avec sigma = niveau * ||A(s*)|| / sqrt(M)
    (niveau = bruit relatif). lambda = c * ||eps|| * ||Phi|| (c sans dimension,
    voir `blasso.operator_norm`). Pour chaque (niveau, c, tirage) : LASSO sur
    grille, Frank-Wolfe glissant, débiaisage ; erreurs mesurées par `match`.
    """
    k = InverseSquareKernel(sensors, depth)
    z = np.atleast_2d(z)
    I = np.asarray(intensities, float)
    y = k.forward(z, I)
    nP = operator_norm(k, box, h)
    rng = np.random.default_rng(seed)
    diag = diagnose(k, z, box, h=0.08)
    rows = []
    for level in noise_levels:
        for rep in range(n_rep):
            eps = level * np.linalg.norm(y) / np.sqrt(k.m) * rng.standard_normal(k.m)
            m = y + eps
            for c in constants:
                lam = c * np.linalg.norm(eps) * nP
                grid_est, _ = lasso_grid(k, m, lam, box, h)
                # Au plus 4K + 2 Diracs : au-delà, l'échec (trop de sources) est acquis
                sfw, it, eta_max = sliding_frank_wolfe(k, m, lam, box, h, max_iter=4 * len(z) + 2)
                deb = debias(k, m, sfw)
                row = {"config": name, "profondeur": depth, "non_degenere": diag.nondegenerate,
                       "niveau": level, "norme_bruit": float(np.linalg.norm(eps)), "c": c,
                       "lambda": lam, "tirage": rep, "iter_sfw": it}
                for tag, est in [("grille", grid_est), ("sfw", sfw), ("debiaise", deb)]:
                    row.update({f"{key}_{tag}": v for key, v in match(z, I, est).items()})
                rows.append(row)
    return rows, diag


def plot_noise(rows, title):
    """
    Erreurs (médiane sur les tirages) en fonction de ||bruit|| en log-log, une
    courbe par constante c, avec une droite de pente 1 de référence ; plus la
    proportion de tirages où exactement K sources sont retrouvées.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for c in dict.fromkeys(r["c"] for r in rows):
        sub = [r for r in rows if r["c"] == c]
        levels = sorted(dict.fromkeys(r["niveau"] for r in sub))
        nb = [np.median([r["norme_bruit"] for r in sub if r["niveau"] == l]) for l in levels]
        med = lambda key: [np.nanmedian([r[key] for r in sub if r["niveau"] == l]) for l in levels]
        exact = [np.mean([r["exact_sfw"] for r in sub if r["niveau"] == l]) for l in levels]
        axes[0].loglog(nb, med("err_pos_sfw"), "o-", lw=2, ms=6, label=f"c = {c:g}")
        axes[1].loglog(nb, med("err_int_debiaise"), "o-", lw=2, ms=6, label=f"c = {c:g}")
        axes[2].semilogx(nb, exact, "o-", lw=2, ms=6, label=f"c = {c:g}")
    nb_all = np.array(sorted(r["norme_bruit"] for r in rows))
    for ax, key in [(axes[0], "err_pos_sfw"), (axes[1], "err_int_debiaise")]:
        ref = np.nanmedian([r[key] for r in rows if r["niveau"] == min(x["niveau"] for x in rows)])
        if np.isfinite(ref):
            x0 = np.median([r["norme_bruit"] for r in rows if r["niveau"] == min(x["niveau"] for x in rows)])
            ax.loglog(nb_all, ref * nb_all / x0, "k--", lw=1, label="pente 1")
    axes[0].set_ylabel("erreur de position max (SFW)")
    axes[1].set_ylabel("erreur relative d'intensité max (SFW + débiaisage)")
    axes[2].set_ylabel("proportion de tirages avec exactement K sources")
    axes[2].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.set_xlabel(r"$\|\varepsilon\|$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig

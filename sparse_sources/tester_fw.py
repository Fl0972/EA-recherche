"""
Tester le Frank-Wolfe glissant (BLASSO) ou le maximum de vraisemblance + BIC
(--methode ml) sur le cercle de capteurs du projet, sur plusieurs tirages,
avec figures.

    python sparse_sources/tester_fw.py                          mode « centrale » (défaut)
    python sparse_sources/tester_fw.py --mode normalisation
    python sparse_sources/tester_fw.py --graines 0 1 2 3 4 5 6 7 8 9 --sigma 1e-4
    python sparse_sources/tester_fw.py --sources 2 3 5 8 --blanchi
    python sparse_sources/tester_fw.py --certificats            affiche aussi les certificats
    python sparse_sources/tester_fw.py --sans-affichage         enregistre sans ouvrir de fenêtre
    python sparse_sources/tester_fw.py --interieur 16x32 --blanchi --sources 2 3 5 10 20
    python sparse_sources/tester_fw.py --interieur 16x32 --blanchi --sources 30 --graines 0 --cas 4
                                      un seul cas : 30 sources, avec bruit et source centrale
    python sparse_sources/tester_fw.py --methode ml             mêmes tests avec ML + BIC
    python sparse_sources/tester_fw.py --methode ml --interieur 16x32 --sources 50 --graines 0 --cas 4
                                      + 512 capteurs dans D (mesures recalculées)

Modes (4 cas chacun, pour 2, 3 et 5 sources par défaut) :
    centrale       noyau normalisé (α = 1) ; sans bruit / avec bruit,
                   sans / avec la grosse source centrale (5000, au centre de D,
                   connue et retirée des mesures, mais dont le bruit reste).
                   Sans bruit, elle est retirée exactement : les deux cas
                   « sans bruit » doivent donner le même résultat.
    normalisation  avec la source centrale ; sans bruit / avec bruit, noyau
                   brut (α = 0) / normalisé (α = 1).

Données : celles de scripts/reconstruct.py (config/ : 64 capteurs sur le
cercle circonscrit à D, bruit relatif σ). Pour une graine donnée, les
sources sont les mêmes dans tous les cas.

Choix de λ :
    avec bruit : λ = 1,5 σ ||y|| (réglage du projet), ou λ = 1,5 √M si
                 --blanchi (chaque capteur divisé par son écart type de bruit) ;
    sans bruit : λ = 10⁻⁴ λ_max, λ_max = max_z ⟨ψ(z), m⟩ (plus petit λ qui
                 donne la solution nulle).

ML + BIC (--methode ml) : toujours blanchi (--blanchi est sans effet), nombre
de sources choisi par le BIC, k_max = K + 10. Sans bruit, la vraisemblance a
besoin d'un niveau de bruit : on suppose σ = 10⁻⁶. La carte « certificat »
est alors la corrélation du résidu blanchi final avec le noyau, divisée par
√(3 log M) : au-dessus de 1, le BIC accepterait une source de plus.

Réussite : bon nombre de sources et erreur de position < 0,25 × min(séparation, 4).

Sorties dans outputs/tester_fw/ (et affichées à la fin, sauf --sans-affichage) :
    <nom>_reconstructions_g<graine>.png   une case par (K, cas) pour ce tirage
    <nom>_certificats_g<graine>.png       certificat final η(z) = ⟨ψ(z), r⟩ / λ
    <nom>_resume.png                      taux de réussite par cas et par K
avec <nom> = [ml_]mode, suivi de _grille<nxm>, _blanchi, _K…, _cas… selon les options.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import matplotlib  # noqa: E402

if "--sans-affichage" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sparse_sources import build_grid, certificate, relative_noise_std, sliding_frank_wolfe  # noqa: E402
from sparse_sources.common import match_sources  # noqa: E402
from sparse_sources.experiments import success  # noqa: E402
from sparse_sources.experiments_projet import SETUP, scenario  # noqa: E402
from sparse_sources.plotting import INK, INK_MUTED, SERIES, certificate_map, style_axes  # noqa: E402

OUTDIR = ROOT / "outputs" / "tester_fw"


def cases_for(mode, sigma):
    """Liste des cas : (bruit σ, exposant α, source centrale ?, libellé)."""
    if mode == "centrale":
        return [(0.0, 1.0, False, "sans bruit, sans centrale"),
                (0.0, 1.0, True, "sans bruit, avec centrale"),
                (sigma, 1.0, False, f"σ = {sigma:g}, sans centrale"),
                (sigma, 1.0, True, f"σ = {sigma:g}, avec centrale")]
    return [(0.0, 0.0, True, "sans bruit, brut (α = 0)"),
            (0.0, 1.0, True, "sans bruit, normalisé (α = 1)"),
            (sigma, 0.0, True, f"σ = {sigma:g}, brut (α = 0)"),
            (sigma, 1.0, True, f"σ = {sigma:g}, normalisé (α = 1)")]


def interior_sensors(spec):
    """Grille 'nxm' de capteurs dans D (centres des cases), ou tableau vide."""
    if not spec:
        return np.zeros((0, 2))
    nx, ny = (int(v) for v in (spec.lower().split("x") * 2)[:2])
    lo, hi = np.asarray(SETUP.domain.lo, float), np.asarray(SETUP.domain.hi, float)
    xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)


METHODS = {"fw": "Frank-Wolfe glissant", "ml": "ML + BIC"}
NOISE_FREE_SIGMA = 1e-6   # ML sans bruit : niveau de bruit supposé (la vraisemblance en a besoin)


def search_grid_for(S, inner):
    """Grille de recherche ; avec des capteurs dans D, aucun nœud sur un capteur."""
    if inner is not None and len(inner):
        from sparse_sources.tester_ml import search_grid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return search_grid(S)
    return build_grid(step=0.25, setup=S)


def run_ml(K, sigma, z, I, y, m, S, inner):
    """
    ML + BIC (``greedy_ml``, toujours blanchi). Carte affichée à la place du
    certificat : corrélation entre le résidu blanchi final et le noyau
    blanchi normalisé, divisée par √(3 log M). Le BIC accepte une source de
    plus à peu près quand elle dépasse 1 : η < 1 partout veut dire que le
    BIC ne voit plus rien à ajouter.
    """
    from sparse_sources import greedy_ml
    s = relative_noise_std(y, sigma if sigma > 0 else NOISE_FREE_SIGMA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = greedy_ml(m, s, setup=S, grid=search_grid_for(S, inner), k_max=K + 10)
    zf, If = np.atleast_2d(r["z"]).reshape(-1, 2), np.asarray(r["I"])
    res = match_sources(z, I, zf, If)
    Sw = S.with_weights(1.0 / s)
    rw = Sw.whiten(m) - (Sw.normalized_intensities(zf, If) @ Sw.phi(zf) if len(If) else 0.0)
    thr = np.sqrt(3 * np.log(len(m)))
    xs = np.linspace(S.domain.lo[0], S.domain.hi[0], 101)
    ys = np.linspace(S.domain.lo[1], S.domain.hi[1], 201)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eta = certificate(rw, thr, np.stack(np.meshgrid(xs, ys), -1), Sw)
    eta = np.where(np.isfinite(eta), eta, np.nan)
    return {"z": z, "I": I, "z_est": zf, "I_est": If, "ok": success(res, z), "res": res,
            "eta_max": float(np.nanmax(eta)), "time": r["time"], "xs": xs, "ys": ys,
            "eta": eta, "sensors": S.sensors,
            "found_0_1": found_within(z, zf, 0.1), "found_0_5": found_within(z, zf, 0.5)}


def run_case(K, sigma, seed, alpha, central, whiten, inner=None, method="fw"):
    """
    Un tirage, une reconstruction (``method`` : 'fw' ou 'ml'). ``inner`` :
    capteurs supplémentaires dans D (les mesures sont alors recalculées pour
    cercle + capteurs intérieurs). Renvoie un dict (sources vraies, estimées,
    certificat...).
    """
    cfg, src, y, m = scenario(K, sigma, seed, central_factor=None if central else 0)
    z, I = src.positions, src.intensities
    if inner is not None and len(inner):
        from sparse_sources.kernel import Setup
        raw = Setup(np.vstack([SETUP.sensors, inner]), SETUP.domain, 0.0)
        c = 5000.0 * raw.phi(np.asarray(SETUP.domain.center, float)[None])[0] if central else 0.0
        y = raw.phi(z).T @ I + c
        m = y * (1 + sigma * np.random.default_rng(seed).standard_normal(len(y))) - c
        S = raw.with_power(alpha)
    else:
        S = SETUP.with_power(alpha)
    if method == "ml":
        return run_ml(K, sigma, z, I, y, m, S, inner)
    s = relative_noise_std(y, sigma) if sigma > 0 else None
    if whiten and s is not None:
        S = S.with_weights(1.0 / s)
        mm = S.whiten(m)
    else:
        mm = m
    grid = search_grid_for(S, inner)
    if sigma > 0:
        lam = 1.5 * np.sqrt(len(m)) if whiten else 1.5 * sigma * float(np.linalg.norm(y))
    else:
        lam = 1e-4 * float(np.max(mm @ grid.G))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fw = sliding_frank_wolfe(mm, lam, setup=S, grid=grid, tol=1e-5, max_iter=3 * K + 10)
    zf = np.atleast_2d(fw["z_debiased"]).reshape(-1, 2)
    If = S.physical_intensities(zf, fw["I_debiased"]) if len(zf) else np.zeros(0)
    res = match_sources(z, I, zf, If)
    zp, Ip = np.atleast_2d(fw["z"]).reshape(-1, 2), np.asarray(fw["I"])
    r = mm - (Ip @ S.phi(zp) if len(Ip) else 0.0)
    xs = np.linspace(S.domain.lo[0], S.domain.hi[0], 101)
    ys = np.linspace(S.domain.lo[1], S.domain.hi[1], 201)
    eta = certificate(r, lam, np.stack(np.meshgrid(xs, ys), -1), S)
    return {"z": z, "I": I, "z_est": zf, "I_est": If, "ok": success(res, z), "res": res,
            "eta_max": float(fw["eta_max"]), "time": fw["time"], "xs": xs, "ys": ys,
            "eta": eta, "sensors": S.sensors,
            "found_0_1": found_within(z, zf, 0.1), "found_0_5": found_within(z, zf, 0.5)}


def found_within(z_true, z_hat, tol):
    """Nombre de vraies sources ayant une source estimée à moins de tol (appariement 1-1)."""
    from scipy.optimize import linear_sum_assignment
    if len(z_hat) == 0:
        return 0
    C = np.linalg.norm(np.asarray(z_true)[:, None] - np.asarray(z_hat)[None], axis=-1)
    a, b = linear_sum_assignment(C)
    return int(np.sum(C[a, b] < tol))


def draw_domain(ax):
    """Contour de D."""
    b = SETUP.domain
    ax.plot(*np.c_[b.lo, [b.hi[0], b.lo[1]], b.hi, [b.lo[0], b.hi[1]], b.lo],
            color=INK_MUTED, lw=1)


def fig_reconstructions(results, Ks, labels, seed, mode, title, meth):
    """Une case par (K, cas) pour un tirage."""
    nr, nc = len(Ks), len(labels)
    size = (3.3 * nc, 4.4 * nr) if nr * nc > 1 else (7.5, 10.0)
    fig, axes = plt.subplots(nr, nc, figsize=size, squeeze=False)
    for i, K in enumerate(Ks):
        for j, label in enumerate(labels):
            r, ax = results[(seed, K, label)], axes[i][j]
            ax.scatter(*r["sensors"].T, s=8 if len(r["sensors"]) <= 64 else 3, color=INK_MUTED,
                       label="capteurs")
            draw_domain(ax)
            ax.scatter(*SETUP.domain.center, marker="*", s=90, color=INK_MUTED,
                       label="source centrale (retirée)") if "avec centrale" in label else None
            ax.scatter(*r["z"].T, s=30 + 3 * r["I"], facecolors="none", edgecolors=SERIES[0],
                       linewidths=2, label="sources vraies")
            if len(r["z_est"]):
                ax.scatter(*r["z_est"].T, s=40, marker="x", color=SERIES[1], linewidths=2,
                           label=METHODS[meth])
            ax.set_aspect("equal")
            style_axes(ax)
            ax.set_title(f"{K} sources, {label}\n{r['res']['n_est']} estimées, "
                         f"{'réussite' if r['ok'] else 'échec'} ; vraies retrouvées : "
                         f"{r['found_0_1']}/{K} à 0,1, {r['found_0_5']}/{K} à 0,5",
                         fontsize=8.5 if nr * nc > 1 else 10,
                         color=INK if r["ok"] else SERIES[1])
    handles, names = [], []
    for ax in axes.flat:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=8)
    fig.suptitle(f"{METHODS[meth]}, {title}, tirage {seed}\n"
                 "(taille des cercles ∝ intensité)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUTDIR / f"{mode}_reconstructions_g{seed}.png", dpi=120)
    return fig


def fig_certificates(results, Ks, labels, seed, mode, title, meth):
    """Certificat final η pour un tirage."""
    nr, nc = len(Ks), len(labels)
    box = SETUP.domain
    fig, axes = plt.subplots(nr, nc, figsize=(3.6 * nc, 4.8 * nr), squeeze=False)
    for i, K in enumerate(Ks):
        for j, label in enumerate(labels):
            r = results[(seed, K, label)]
            certificate_map(axes[i][j], r["xs"], r["ys"], r["eta"], r["sensors"], r["z"], box,
                            title=f"{K} sources, {label}\nmax η = {r['eta_max']:.3f}",
                            estimates=r["z_est"])
            axes[i][j].set_xlim(box.lo[0] - 1, box.hi[0] + 1)
            axes[i][j].set_ylim(box.lo[1] - 1, box.hi[1] + 1)
            axes[i][j].title.set_fontsize(8.5)
    what = ("Certificat final η(z) = ⟨ψ(z), r⟩ / λ" if meth == "fw" else
            "Corrélation du résidu final ⟨ψ_w(z), r_w⟩ / √(3 log M) (BIC : nouvelle source si > 1)")
    fig.suptitle(f"{what}\n{METHODS[meth]}, {title}, tirage {seed} (contour noir : η = 1)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{mode}_certificats_g{seed}.png", dpi=120)
    return fig


def fig_summary(rates, Ks, labels, n_seeds, mode, title, meth):
    """Taux de réussite par cas et par nombre de sources."""
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    width = 0.8 / len(labels)
    x = np.arange(len(Ks))
    for j, label in enumerate(labels):
        v = [100 * rates[(K, label)] for K in Ks]
        bars = ax.bar(x + (j - (len(labels) - 1) / 2) * width, v, width * 0.92,
                      color=SERIES[j % len(SERIES)], label=label)
        ax.bar_label(bars, fmt="%.0f", fontsize=7, padding=2)
    ax.set_xticks(x, [f"{K} sources" for K in Ks])
    ax.set_ylabel("réussite (%)")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    style_axes(ax)
    ax.set_title(f"{METHODS[meth]}, {title} :\nréussite sur {n_seeds} tirages",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{mode}_resume.png", dpi=120)
    return fig


def main():
    """Lance toutes les combinaisons, affiche les tableaux et les figures."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methode", choices=["fw", "ml"], default="fw",
                   help="fw : Frank-Wolfe glissant (défaut) ; ml : maximum de vraisemblance + BIC")
    p.add_argument("--mode", choices=["centrale", "normalisation"], default="centrale")
    p.add_argument("--sources", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--sigma", type=float, default=1e-3, help="bruit relatif des cas bruités")
    p.add_argument("--graines", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="tirages (une figure par tirage)")
    p.add_argument("--blanchi", action="store_true",
                   help="cas bruités : capteurs divisés par l'écart type de leur bruit")
    p.add_argument("--cas", type=int, nargs="+", choices=[1, 2, 3, 4], default=None,
                   help="ne garder que ces cas (numéros dans l'ordre du tableau, "
                        "ex. 4 = avec bruit et avec centrale en mode centrale)")
    p.add_argument("--interieur", default=None,
                   help="ajoute une grille de capteurs dans D, 'n' (n×n) ou 'nxm', ex. 16x32")
    p.add_argument("--certificats", action="store_true",
                   help="affiche aussi les cartes de certificat (toujours enregistrées)")
    p.add_argument("--sans-affichage", action="store_true", help="n'ouvre pas les figures")
    args = p.parse_args()

    cases = cases_for(args.mode, args.sigma)
    if args.cas:
        cases = [cases[i - 1] for i in args.cas]
    labels = [c[3] for c in cases]
    inner = interior_sensors(args.interieur)
    M = len(SETUP.sensors) + len(inner)
    whiten = args.blanchi and args.methode == "fw"      # ML est toujours blanchi
    title = (f"cercle + grille {args.interieur} ({M} capteurs)" if len(inner)
             else "cercle du projet (64 capteurs)") + (", blanchi" if whiten else "")
    tag = (("ml_" if args.methode == "ml" else "") + args.mode
           + (f"_grille{args.interieur}" if len(inner) else "")
           + ("_blanchi" if whiten else "")
           + ("" if args.sources == [2, 3, 5] else "_K" + "-".join(map(str, args.sources)))
           + ("_cas" + "-".join(map(str, args.cas)) if args.cas else ""))
    results = {}
    print(f"\n{METHODS[args.methode]}, {title}, mode « {args.mode} »")
    for seed in args.graines:
        print(f"\nTirage {seed}")
        print(f"{'K':>2} | {'cas':32s} | {'K estimé':>8} | {'err. pos. max':>13} | "
              f"{'err. int. max':>13} | {'max η':>6} | {'résultat':8} | trouvées à 0,1 / 0,5 | temps")
        for K in args.sources:
            for sigma, alpha, central, label in cases:
                r = run_case(K, sigma, seed, alpha, central, whiten, inner, args.methode)
                results[(seed, K, label)] = r
                res = r["res"]
                print(f"{K:>2} | {label:32s} | {res['n_est']:>8} | {res['err_pos']:>13.3g} | "
                      f"{res['err_int']:>13.3g} | {r['eta_max']:>6.3f} | "
                      f"{'RÉUSSITE' if r['ok'] else 'échec':8} | {r['found_0_1']:>8} / {r['found_0_5']:<9} | "
                      f"{r['time']:.2f} s", flush=True)

    rates = {(K, label): np.mean([results[(s, K, label)]["ok"] for s in args.graines])
             for K in args.sources for label in labels}
    print(f"\nRésumé : taux de réussite sur {len(args.graines)} tirages")
    print(f"{'cas':32s} | " + " | ".join(f"{K} sources" for K in args.sources))
    for label in labels:
        print(f"{label:32s} | " + " | ".join(f"{100 * rates[(K, label)]:8.0f} %"
                                             for K in args.sources))
    print("err. int. max : erreur relative d'intensité maximale sur les paires (vraie, estimée).")
    print("max η : maximum du certificat final ; ≈ 1 si Frank-Wolfe a convergé."
          if args.methode == "fw" else
          "max η : corrélation max du résidu / √(3 log M) ; > 1 : le BIC verrait une source de plus.")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    keep = []
    for seed in args.graines:
        keep.append(fig_reconstructions(results, args.sources, labels, seed, tag, title,
                                        args.methode))
        figc = fig_certificates(results, args.sources, labels, seed, tag, title, args.methode)
        if args.certificats:
            keep.append(figc)
        else:
            plt.close(figc)
    keep.append(fig_summary(rates, args.sources, labels, len(args.graines), tag, title,
                            args.methode))
    print(f"\nFigures enregistrées dans {OUTDIR.relative_to(ROOT)} "
          f"({tag}_reconstructions_g*.png, {tag}_certificats_g*.png, {tag}_resume.png)")
    if not args.sans_affichage:
        print("Fermez les fenêtres pour terminer.")
        plt.show()


if __name__ == "__main__":
    main()

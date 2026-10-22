"""
Théorème de Duval-Peyré pour le modèle du projet : condition de source non
dégénérée (H1, H2, H3), balayages, puis validation du BLASSO sur données
bruitées. Géométrie (domaine D, capteurs) : dossier config/ ; la profondeur h
des capteurs est le paramètre eps du noyau G = 1 / (r^2 + h^2).

    python scripts/duval_peyre.py certificat --sources "5,7;5,13"
    python scripts/duval_peyre.py certificat --sources "5,7;5,13" --profondeur 12
    python scripts/duval_peyre.py balayages
    python scripts/duval_peyre.py blasso
    python scripts/duval_peyre.py tout
    python scripts/duval_peyre.py              (sans argument = tout)

Options communes : --config DOSSIER, --set CLE=VALEUR (comme les autres
scripts), --profondeur H (défaut : sensors.eps de la config), --sortie DOSSIER
(défaut : outputs/duval_peyre). Figures en PNG, résultats en CSV.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from particle_sources.config import load_config  # noqa: E402
from particle_sources.config.cli import DEFAULT_CONFIG_DIR  # noqa: E402
from particle_sources.duval_peyre import InverseSquareKernel, diagnose  # noqa: E402
from particle_sources.duval_peyre import experiments as ex  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")


def parse_sources(text: str) -> np.ndarray:
    """'x1,y1;x2,y2' -> tableau (K, 2)."""
    return np.array([[float(v) for v in p.split(",")] for p in text.split(";") if p.strip()])


def save(fig, out: Path, name: str):
    """Enregistre une figure PNG dans `out` et l'affiche dans la console."""
    path = out / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  -> {path}")


# ---- Sous-commandes ----------------------------------------------------------------
def cmd_certificat(cfg, args, out):
    """Diagnostic d'une configuration donnée + carte de eta_V."""
    z = parse_sources(args.sources)
    S = cfg.sensors.positions
    diag = diagnose(InverseSquareKernel(S, args.profondeur), z, cfg.box, keep_grid=True)
    print(diag)
    ax = ex.plot_certificate(diag, cfg.box, S, title=f"profondeur h = {args.profondeur:g}")
    ax.figure.tight_layout()
    save(ax.figure, out, f"certificat_h{args.profondeur:g}.png")
    ex.write_csv([{"profondeur": args.profondeur, **diag.as_row()}], out / "certificat.csv")


def cmd_balayages(cfg, args, out):
    """Balayages a), b), c) de la condition de source non dégénérée."""
    box, S = cfg.box, cfg.sensors.positions
    c, R = cfg.sensors.center, cfg.sensors.radius
    deep = args.profondeur_ref
    t = time.time()

    # Carte de référence : même paire, capteurs dans le plan et en profondeur
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    for ax, depth in zip(axes, [args.profondeur, deep]):
        diag = diagnose(InverseSquareKernel(S, depth), ex.pair(c, 6.0), box, keep_grid=True)
        ex.plot_certificate(diag, box, S, ax=ax, title=f"paire d = 6, profondeur h = {depth:g}")
    fig.tight_layout()
    save(fig, out, "carte_eta_V.png")

    # a) séparation
    dists = np.geomspace(0.05, 16, 28)
    depths_a = sorted({args.profondeur, 0.75 * deep, deep, 1.5 * deep})
    rows_a = ex.sweep_separation(box, S, c, dists, depths_a)
    print("\na) Séparation de deux sources (centrées en "
          f"{np.round(c, 2)}) -- seuil d* au-delà duquel la condition tient :")
    for depth in depths_a:
        sub = [r for r in rows_a if r["profondeur"] == depth]
        thr = ex.threshold([r["d"] for r in sub], [r["non_degenere"] for r in sub])
        print(f"   h = {depth:6.3g} : d* = {thr:.3g}" if np.isfinite(thr)
              else f"   h = {depth:6.3g} : jamais vérifiée sur [{dists[0]:.2g}, {dists[-1]:.3g}]")
    save(ex.plot_sweep(rows_a, "d", "profondeur", "séparation d", "a) Séparation de 2 sources",
                       logx=True), out, "balayage_separation.png")

    # b1) profondeur
    depths_b = np.linspace(0, 2.5 * deep, 31)
    configs = {"K=1 centre": c[None], "K=1 excentrée": (c + [0, 6])[None],
               "K=2 d=6": ex.pair(c, 6.0), "K=3": c + np.array([[-2, -4], [2, 0], [-1, 5]])}
    rows_b = ex.sweep_depth(box, S, configs, depths_b)
    print("\nb) Profondeur h des capteurs -- seuil h* au-delà duquel la condition tient :")
    for name in configs:
        sub = [r for r in rows_b if r["config"] == name]
        thr = ex.threshold([r["profondeur"] for r in sub], [r["non_degenere"] for r in sub])
        print(f"   {name:15s}: h* = {thr:.3g}" if np.isfinite(thr)
              else f"   {name:15s}: jamais vérifiée pour h <= {depths_b[-1]:.3g}")
    save(ex.plot_sweep(rows_b, "profondeur", "config", "profondeur h",
                       "b) Profondeur des capteurs (G = 1/(r² + h²))"), out, "balayage_profondeur.png")

    # b2) distance aux capteurs dans le plan
    offs = np.linspace(0, 7, 15)
    rows_o = ex.sweep_offset(box, S, c, 6.0, offs, deep)
    print(f"\nb') Paire d = 6 rapprochée du bord haut de D (h = {deep:g}) :")
    for r in rows_o:
        print(f"   décalage {r['decalage']:4.1f}  dist. capteur {r['dist_capteur']:5.2f}  "
              f"{'OK ' if r['non_degenere'] else 'NON'}  marge {r['marge']:+.3g}")
    save(ex.plot_sweep(rows_o, "dist_capteur", "profondeur", "distance au capteur le plus proche",
                       "b') Rapprochement des capteurs dans le plan"), out, "balayage_distance.png")

    # c) nombre et disposition des capteurs
    z = ex.pair(c, 6.0)
    ms = np.unique(np.round(np.geomspace(6, 8 * 6, 12)).astype(int))
    aps = [2 * np.pi, np.pi, np.pi / 2]
    rows_c = ex.sweep_sensors(box, c, R, z, ms, deep, aps)
    for r in rows_c:
        r["ouverture_deg"] = round(np.degrees(r["ouverture"]))
    print(f"\nc) Capteurs (paire d = 6, h = {deep:g}) -- M minimal pour que la condition tienne :")
    for ap in aps:
        sub = [r for r in rows_c if r["ouverture"] == ap]
        thr = ex.threshold([r["M"] for r in sub], [r["non_degenere"] for r in sub])
        print(f"   arc de {np.degrees(ap):3.0f}° : M* = {thr:.0f}" if np.isfinite(thr)
              else f"   arc de {np.degrees(ap):3.0f}° : jamais vérifiée pour M <= {ms[-1]}")
    save(ex.plot_sweep(rows_c, "M", "ouverture_deg", "nombre de capteurs M (3K = 6)",
                       "c) Nombre et disposition des capteurs", logx=True), out, "balayage_capteurs.png")

    ex.write_csv(rows_a + rows_b + rows_o + rows_c, out / "balayages.csv")
    print(f"  -> {out / 'balayages.csv'}   ({time.time() - t:.0f} s)")


def cmd_blasso(cfg, args, out):
    """Validation sur données bruitées : configurations où la condition tient / échoue."""
    box, S, c = cfg.box, cfg.sensors.positions, cfg.sensors.center
    deep = args.profondeur_ref
    levels = np.geomspace(1e-5, 3e-2, 8)
    consts = [0.5, 1.0, 4.0]
    cases = [
        ("tient : paire d = 6, h = %g" % deep, ex.pair(c, 6.0), deep),
        ("échoue (sources proches) : paire d = 1, h = %g" % deep, ex.pair(c, 1.0), deep),
        ("échoue (capteurs dans le plan) : paire d = 6, h = %g" % args.profondeur,
         ex.pair(c, 6.0), args.profondeur),
    ]
    all_rows = []
    for i, (name, z, depth) in enumerate(cases):
        t = time.time()
        rows, diag = ex.noise_experiment(box, S, z, [100.0, 150.0], depth, levels, consts,
                                         n_rep=args.tirages, name=name, seed=i)
        all_rows += rows
        print(f"\n{name}  -- condition {'vérifiée' if diag.nondegenerate else 'NON vérifiée'} "
              f"(marge {diag.margin:+.3g})  ({time.time() - t:.0f} s)")
        print("   niveau    ||eps||   c    #K exact   err_pos(sfw)  err_I(débiaisé)  #sources(grille)")
        for lvl in levels:
            for cc in consts:
                sub = [r for r in rows if r["niveau"] == lvl and r["c"] == cc]
                print(f"   {lvl:8.1e} {np.median([r['norme_bruit'] for r in sub]):9.2e} {cc:4g} "
                      f"{np.mean([r['exact_sfw'] for r in sub]):8.0%}   "
                      f"{np.nanmedian([r['err_pos_sfw'] for r in sub]):11.3e}  "
                      f"{np.nanmedian([r['err_int_debiaise'] for r in sub]):12.3e}   "
                      f"{np.median([r['n_est_grille'] for r in sub]):6.0f}")
        save(ex.plot_noise(rows, f"BLASSO -- {name}"), out, f"blasso_{i + 1}.png")
    ex.write_csv(all_rows, out / "blasso.csv")
    print(f"  -> {out / 'blasso.csv'}")


def main():
    """Lit les arguments, charge la config du projet et lance la sous-commande."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("commande", nargs="?", default="tout",
                        choices=["certificat", "balayages", "blasso", "tout"],
                        help="étape à lancer (défaut : tout)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR, help="dossier de YAML")
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                        help="surcharge de la config (répétable)")
    parser.add_argument("--sources", default="5,7;5,13",
                        help="positions 'x1,y1;x2,y2;...' pour `certificat`")
    parser.add_argument("--profondeur", type=float, default=None,
                        help="profondeur h = eps du noyau (défaut : sensors.eps de la config)")
    parser.add_argument("--profondeur-ref", type=float, default=None,
                        help="profondeur « favorable » des balayages et du BLASSO "
                             "(défaut : rayon du cercle des capteurs)")
    parser.add_argument("--tirages", type=int, default=5, help="tirages de bruit par niveau")
    parser.add_argument("--sortie", default=Path(DEFAULT_CONFIG_DIR).parent / "outputs" / "duval_peyre")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.profondeur is None:
        args.profondeur = cfg.eps
    if args.profondeur_ref is None:
        args.profondeur_ref = round(float(cfg.sensors.radius))
    out = Path(args.sortie)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Domaine D = {cfg.box.lo} x {cfg.box.hi}, M = {len(cfg.sensors.positions)} capteurs, "
          f"profondeur h = {args.profondeur:g} (référence {args.profondeur_ref:g})\n")

    if args.commande in ("certificat", "tout"):
        cmd_certificat(cfg, args, out)
    if args.commande in ("balayages", "tout"):
        cmd_balayages(cfg, args, out)
    if args.commande in ("blasso", "tout"):
        cmd_blasso(cfg, args, out)


if __name__ == "__main__":
    main()

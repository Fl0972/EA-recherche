"""
Expérience 5 : position des capteurs, nombre de capteurs, taille et forme
du domaine.

Pour chaque géométrie, deux mesures :
    crb       borne de Cramér-Rao médiane (erreur de position minimale
              possible, pire source) sur 300 tirages a priori à 3 sources,
              σ = 1e-3 — c'est une propriété des MESURES, pas de l'algorithme ;
    reussite  taux de réussite de ML + BIC (``greedy_ml``) sur des tirages
              K ∈ {2, 3, 5}, σ ∈ {1e-4, 1e-3}, 30 tirages par case.
Toutes les géométries gardent la source centrale (5000 au centre de D).

Remarque théorique (vérifiée par l'expérience « taille ») : si on multiplie
toutes les longueurs (domaine ET capteurs) par a, toutes les mesures sont
divisées par a², source centrale comprise. Le bruit étant relatif, les
mesures blanchies m/s sont inchangées : le problème est le MÊME, seules les
erreurs de position sont multipliées par a. La taille du domaine ne compte
donc pas, seules la forme et la position relative des capteurs comptent.

Placement optimal (plan d'expérience bayésien D-optimal) : parmi des
positions candidates, on choisit gloutonnement les 64 capteurs qui
maximisent la moyenne, sur des tirages a priori de 3 sources, de
log det(Fisher). On le fait avec des candidats HORS de D (capteurs autour
de la zone), et partout.

    python -m sparse_sources.apprentissage.geometrie
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sparse_sources import cramer_rao, greedy_ml  # noqa: E402

from .base import OUT, evaluate, parallel  # noqa: E402
from .rapport import save_csv  # noqa: E402
from .simulation import Scene, circle_sensors  # noqa: E402

B = np.array([[0.0, 10.0], [0.0, 20.0]])
M = 64


def ellipse(bounds, n=M, factor=1.0):
    """Ellipse passant par les 4 coins de D (axes ∝ côtés × √2)."""
    b = np.asarray(bounds, float)
    c, half = b.mean(1), 0.5 * (b[:, 1] - b[:, 0])
    t = 2 * np.pi * np.arange(n) / n
    return c + factor * np.sqrt(2) * half * np.stack([np.cos(t), np.sin(t)], 1)


def rectangle(bounds, n=M, margin=0.0):
    """n capteurs équirépartis sur le bord du rectangle D élargi de ``margin``."""
    b = np.asarray(bounds, float)
    lo, hi = b[:, 0] - margin, b[:, 1] + margin
    w, h = hi - lo
    L = 2 * (w + h)
    out = []
    for s in (np.arange(n) + 0.5) * L / n:
        if s < w:
            out.append([lo[0] + s, lo[1]])
        elif s < w + h:
            out.append([hi[0], lo[1] + s - w])
        elif s < 2 * w + h:
            out.append([hi[0] - (s - w - h), hi[1]])
        else:
            out.append([lo[0], hi[1] - (s - 2 * w - h)])
    return np.array(out)


def inside_grid(bounds, n_side=8):
    """Grille n_side × n_side de capteurs DANS D."""
    b = np.asarray(bounds, float)
    xs = b[0, 0] + (np.arange(n_side) + 0.5) * (b[0, 1] - b[0, 0]) / n_side
    ys = b[1, 0] + (np.arange(n_side) + 0.5) * (b[1, 1] - b[1, 0]) / n_side
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1)


def random_inside(bounds, n=M, seed=0):
    b = np.asarray(bounds, float)
    return b[:, 0] + (b[:, 1] - b[:, 0]) * np.random.default_rng(seed).random((n, 2))


# ---- plan d'expérience D-optimal ------------------------------------------------------
def d_optimal(candidates, n=M, n_prior=200, K=3, sigma=1e-3, seed=0, bounds=B):
    """
    Choix glouton de n capteurs parmi ``candidates`` maximisant
    E_prior[log det F]. Fisher additive par capteur : F = Σ_i g_i g_iᵀ / s_i²,
    g_i = dérivées de la mesure i par rapport à (I, z). La source centrale
    entre dans s_i = σ y_i.
    """
    rng = np.random.default_rng(seed)
    scene0 = Scene(bounds=bounds, sensors=candidates)
    G = []   # pour chaque tirage a priori : (C, 3K) gradients blanchis
    for _ in range(n_prior):
        d = scene0.draw(K, 0.0, rng)
        z, I = d["z"], d["I"]
        diff = candidates[:, None, :] - z[None]                 # C, K, 2
        q = (diff**2).sum(-1)                                   # C, K
        dI = 1.0 / q
        dz = (2 * I[None, :, None] * diff / q[..., None] ** 2).reshape(len(candidates), 2 * K)
        y = dI @ I + scene0.y_central
        G.append(np.concatenate([dI, dz], 1) / (sigma * y)[:, None])
    G = np.stack(G)                                            # P, C, 3K
    dim = G.shape[2]
    F = np.tile(1e-8 * np.eye(dim), (n_prior, 1, 1))
    chosen = []
    for _ in range(n):
        Finv = np.linalg.pinv(F, hermitian=True)
        # gain de log det en ajoutant le capteur c : log(1 + g_cᵀ F⁻¹ g_c)
        gain = np.log1p(np.einsum("pci,pij,pcj->pc", G, Finv, G)).mean(0)
        gain[chosen] = -np.inf
        c = int(np.argmax(gain))
        chosen.append(c)
        F = F + np.einsum("pi,pj->pij", G[:, c], G[:, c])
    return candidates[chosen]


def candidates_outside(bounds=B, step=0.5, extent=6.0):
    b = np.asarray(bounds, float)
    xs = np.arange(b[0, 0] - extent, b[0, 1] + extent + 1e-9, step)
    ys = np.arange(b[1, 0] - extent, b[1, 1] + extent + 1e-9, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = np.all((P > b[:, 0] - 0.25) & (P < b[:, 1] + 0.25), axis=1)
    return P[~inside]


def candidates_everywhere(bounds=B, step=0.5, extent=6.0):
    b = np.asarray(bounds, float)
    xs = np.arange(b[0, 0] - extent, b[0, 1] + extent + 1e-9, step) + 0.25
    ys = np.arange(b[1, 0] - extent, b[1, 1] + extent + 1e-9, step) + 0.25
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], 1)


# ---- évaluation --------------------------------------------------------------------------
def crb_median(scene: Scene, K=3, sigma=1e-3, n=300, seed=7):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        d = scene.draw(K, sigma, rng)
        v.append(float(cramer_rao(d["z"], d["I"], d["s"], scene.setup)["pos_std"].max()))
    return float(np.median(v)), float(np.mean(np.array(v) < 0.25))


def run_one(args):
    name, scene, d = args
    r = greedy_ml(d["m"], d["s"], setup=scene.setup, grid=scene.grid)
    return {"geometrie": name, "n": d["n"], "sigma": d["sigma"], **evaluate(d, r["z"], r["I"]),
            "temps": r["time"]}


def evaluate_geometries(geoms, n_per=30, ks=(2, 3, 5), sigmas=(1e-4, 1e-3), scale=1.0):
    rows, summ = [], []
    jobs = []
    for name, scene in geoms.items():
        rng = np.random.default_rng(123)
        for k in ks:
            for s in sigmas:
                for _ in range(n_per):
                    jobs.append((name, scene, scene.draw(k, s, rng)))
    res = parallel(run_one, jobs)
    rows += res
    for name, scene in geoms.items():
        sel = [r for r in res if r["geometrie"] == name]
        crb, frac = crb_median(scene)
        row = {"geometrie": name, "M": scene.M, "crb_median_3src": crb,
               "part_crb<0.25": frac, "reussite_moyenne": float(np.mean([r["reussite"] for r in sel]))}
        for k in ks:
            for s in sigmas:
                row[f"n={k}_sigma={s:g}"] = float(np.mean([r["reussite"] for r in sel
                                                           if r["n"] == k and r["sigma"] == s]))
        summ.append(row)
        print(f"  {name:28s} CRB médiane {crb:7.3f}  réussite {row['reussite_moyenne']:.3f}  "
              + " ".join(f"{row[f'n={k}_sigma={s:g}']:.2f}" for k in ks for s in sigmas))
    return rows, summ


def main():
    t0 = time.perf_counter()
    print("\n[a] Position des capteurs (64 capteurs, D = 10 × 20)")
    print("  calcul des plans D-optimaux...")
    opt_out = d_optimal(candidates_outside())
    opt_all = d_optimal(candidates_everywhere())
    np.save(OUT / "capteurs_optimaux_hors_D.npy", opt_out)
    np.save(OUT / "capteurs_optimaux_partout.npy", opt_all)
    geoms = {
        "cercle (projet)": Scene(),
        "cercle x0.8 (coupe les coins)": Scene(sensors=circle_sensors(B, M, 0.8)),
        "cercle x1.5": Scene(sensors=circle_sensors(B, M, 1.5)),
        "cercle x2": Scene(sensors=circle_sensors(B, M, 2.0)),
        "ellipse par les coins": Scene(sensors=ellipse(B)),
        "bord de D (marge 0,5)": Scene(sensors=rectangle(B, margin=0.5)),
        "bord de D (marge 2)": Scene(sensors=rectangle(B, margin=2.0)),
        "deux cercles x1 et x1.5": Scene(sensors=np.vstack([circle_sensors(B, 32, 1.0),
                                                           circle_sensors(B, 32, 1.5, np.pi / 32)])),
        "grille 8x8 dans D": Scene(sensors=inside_grid(B)),
        "aleatoire dans D": Scene(sensors=random_inside(B)),
        "D-optimal hors de D": Scene(sensors=opt_out),
        "D-optimal partout": Scene(sensors=opt_all),
    }
    rows, summ = evaluate_geometries(geoms)
    save_csv(rows, OUT / "geometrie_capteurs.csv")
    save_csv(summ, OUT / "geometrie_capteurs_resume.csv")

    fig, axes = plt.subplots(2, 6, figsize=(18, 7.5))
    for ax, (name, sc), s in zip(axes.ravel(), geoms.items(), summ):
        ax.add_patch(plt.Rectangle((0, 0), 10, 20, fill=False, lw=1.2, color="0.3"))
        ax.plot(*sc.sensors.T, "o", ms=3, color="#2a6fb0")
        ax.plot(5, 10, "*", ms=10, color="#c0392b")
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nCRB {s['crb_median_3src']:.2f}, réussite {s['reussite_moyenne']:.2f}",
                     fontsize=8)
        ax.set_xlim(-12, 22)
        ax.set_ylim(-8, 28)
        ax.tick_params(labelsize=6)
    fig.suptitle("64 capteurs (points bleus), domaine D (rectangle), source centrale (étoile)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "geometrie_capteurs.png", dpi=130)
    plt.close(fig)

    print("\n[b] Nombre de capteurs (cercle du projet)")
    geoms = {f"cercle, M={n}": Scene(sensors=circle_sensors(B, n)) for n in (16, 32, 64, 128, 256)}
    geoms["D-optimal hors D, M=32"] = Scene(sensors=d_optimal(candidates_outside(), n=32))
    rows_b, summ_b = evaluate_geometries(geoms)
    save_csv(summ_b, OUT / "geometrie_nombre_resume.csv")

    print("\n[c] Taille et forme du domaine (64 capteurs sur le cercle circonscrit)")
    geoms = {}
    for a in (0.5, 1.0, 2.0, 4.0):
        b = B * a
        geoms[f"taille x{a:g} ({10*a:g} x {20*a:g})"] = Scene(bounds=b, sensors=circle_sensors(b))
    for name, b in [("carre 14,1 x 14,1", [[0, 14.142], [0, 14.142]]),
                    ("allonge 5 x 40", [[0, 5.0], [0, 40.0]]),
                    ("tres allonge 2,5 x 80", [[0, 2.5], [0, 80.0]])]:
        geoms[name] = Scene(bounds=b, sensors=circle_sensors(b))
        geoms[name + ", ellipse"] = Scene(bounds=b, sensors=ellipse(b))
    rows_c, summ_c = evaluate_geometries(geoms)
    for s in summ_c:   # erreur relative à la taille : CRB / sqrt(aire)
        b = geoms[s["geometrie"]].bounds
        s["crb_sur_racine_aire"] = s["crb_median_3src"] / float(np.sqrt(np.prod(b[:, 1] - b[:, 0])))
    save_csv(summ_c, OUT / "geometrie_domaine_resume.csv")
    print(f"  total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()

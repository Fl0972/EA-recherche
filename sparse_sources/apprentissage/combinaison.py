"""
Expérience 7 : les deux leviers ensemble — meilleure géométrie de capteurs
et meilleure méthode (ML + BIC avec mouvements de division).

Mêmes tirages que l'expérience géométrie (K ∈ {2, 3, 5}, σ ∈ {1e-4, 1e-3},
30 tirages par case), sur trois géométries.

    python -m sparse_sources.apprentissage.combinaison
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sparse_sources import greedy_ml  # noqa: E402

from .base import OUT, evaluate, parallel  # noqa: E402
from .geometrie import B, rectangle  # noqa: E402
from .rapport import pivot, save_csv  # noqa: E402
from .recherche import global_search  # noqa: E402
from .simulation import Scene  # noqa: E402


def job(args):
    geo, scene, i, d = args
    g = greedy_ml(d["m"], d["s"], setup=scene.setup, grid=scene.grid)
    z, I, *_ = global_search(d, split=True, r_prior=0, r_nn=0, scene=scene, seed=i)
    base = {"geometrie": geo, "n": d["n"], "sigma": d["sigma"]}
    return [{**base, "methode": "ML + BIC", **evaluate(d, g["z"], g["I"])},
            {**base, "methode": "ML + BIC + division", **evaluate(d, z, I)}]


def main():
    geoms = {"cercle (projet)": Scene(),
             "bord de D (marge 0,5)": Scene(sensors=rectangle(B, margin=0.5)),
             "D-optimal hors de D": Scene(sensors=np.load(OUT / "capteurs_optimaux_hors_D.npy"))}
    jobs = []
    for geo, sc in geoms.items():
        rng = np.random.default_rng(123)
        i = 0
        for k in (2, 3, 5):
            for s in (1e-4, 1e-3):
                for _ in range(30):
                    jobs.append((geo, sc, i, sc.draw(k, s, rng)))
                    i += 1
    rows = [r for out in parallel(job, jobs) for r in out]
    for r in rows:
        r["cle"] = f"{r['geometrie']} | {r['methode']}"
    save_csv(rows, OUT / "combinaison.csv")
    tab = pivot(rows, "cle", ["n", "sigma"], "reussite")
    save_csv(tab, OUT / "combinaison_resume.csv")
    for r in tab:
        print("  ", r["cle"].ljust(48), f"moyenne {r['moyenne']:.3f}  ",
              " ".join(f"{v:.2f}" for k, v in r.items() if k.startswith("n=")))


if __name__ == "__main__":
    main()

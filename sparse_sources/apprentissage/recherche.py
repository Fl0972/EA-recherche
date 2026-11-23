"""
Expérience 6 : recherche globale stochastique du maximum de vraisemblance.

À K fixé, la vraisemblance a plusieurs maxima locaux ; le ML glouton n'en
explore qu'un. Ici, pour chaque K = 1, 2, ..., on lance le ML depuis
plusieurs départs et on garde le plus petit χ² :

    glouton   meilleure solution K-1 + Dirac au maximum de corrélation
    division  meilleure solution K-1 où une source est remplacée par deux
              sources voisines (écart ±0,7 selon x ou y) : sépare des sources
              proches que le glouton a confondues
    a priori  meilleure solution K-1 + une source tirée uniformément dans D
              (R départs)
    reseau    K positions tirées dans la carte du réseau, vue comme une loi
              de probabilité (échantillonnage d'importance, R départs)

K est choisi par BIC (arrêt après 2 hausses). Variantes comparées :
    glouton (= chemin ML seul), +division, +a priori, +reseau, tout.

    python -m sparse_sources.apprentissage.recherche
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from .base import OUT, Fitter, evaluate, parallel  # noqa: E402
from .rapport import pivot, save_csv  # noqa: E402
from .simulation import PROJET, Scene  # noqa: E402

VARIANTS = {
    "glouton": dict(split=False, r_prior=0, r_nn=0),
    "glouton+division": dict(split=True, r_prior=0, r_nn=0),
    "glouton+a priori (R=8)": dict(split=False, r_prior=8, r_nn=0),
    "glouton+reseau (R=8)": dict(split=False, r_prior=0, r_nn=8),
    "tout": dict(split=True, r_prior=8, r_nn=8),
}


def global_search(d, heat=None, pts=None, split=True, r_prior=8, r_nn=8, k_max=8,
                  patience=2, scene: Scene = PROJET, seed=0):
    """Renvoie (z, I physiques, K, nombre d'ajustements, origine de chaque solution)."""
    rng = np.random.default_rng(seed)
    F = Fitter(d["m"], d["s"], scene)
    kappa = 3 * np.log(scene.M)
    z, I = np.zeros((0, 2)), np.zeros(0)
    best = (F.chi2(z, I), z, I)
    n_fits, fails, origin = 0, 0, []
    p = None
    if heat is not None and r_nn:
        p = np.clip(heat, 0, None) + 1e-4 * heat.max()
        p = p / p.sum()
    for K in range(1, k_max + 1):
        cands = [(*F.add_one(z, I), "glouton")]
        if split and len(I):
            for j in range(len(I)):
                for e in ([0.7, 0], [0, 0.7]):
                    zz = np.vstack([np.delete(z, j, 0), z[j] + e, z[j] - np.array(e)])
                    cands.append((*F.fit_from(zz), "division"))
        for _ in range(r_prior):
            new = scene.lo + (scene.hi - scene.lo) * rng.random(2)
            cands.append((*F.fit_from(np.vstack([z, new]) if len(z) else new[None]), "a priori"))
        if p is not None:
            for _ in range(r_nn):
                idx = rng.choice(len(p), K, replace=False, p=p)
                zz = pts[idx] + 0.25 * rng.standard_normal((K, 2))
                cands.append((*F.fit_from(zz), "reseau"))
        n_fits += len(cands)
        cands = [c for c in cands if len(c[1]) == K] or cands
        zK, IK, cK, src = min(cands, key=lambda t: t[2])
        z, I = zK, IK
        if cK + kappa * len(IK) < best[0] + kappa * len(best[2]) - 1e-9:
            best, fails = (cK, zK, IK), 0
            origin.append(src)
        else:
            fails += 1
            if fails >= patience:
                break
    _, z, I = best
    return z, F.physical(z, I), len(I), n_fits, origin


def job(args):
    i, d, heat, pts, name, kw = args
    t = time.perf_counter()
    z, I, K, nf, origin = global_search(d, heat, pts, seed=i, **kw)
    return {"methode": name, "i": i, "n": d["n"], "sigma": d["sigma"], **evaluate(d, z, I),
            "temps": time.perf_counter() - t, "ajustements": nf,
            "origine": "/".join(origin)}


def main(variants=VARIANTS):
    from .donnees import bases
    from .reseau import predict, train as train_net
    t0 = time.perf_counter()
    _, _, test, _ = bases()
    net, maps = train_net()
    H, _ = predict(net, maps, test)
    jobs = [(i, d, H[i], maps.pts, name, kw) for name, kw in variants.items()
            for i, d in enumerate(test)]
    rows = parallel(job, jobs)
    save_csv(rows, OUT / "recherche.csv")
    tab = pivot(rows, "methode", ["n", "sigma"], "reussite")
    for r in tab:
        sel = [x for x in rows if x["methode"] == r["methode"]]
        r["temps_moyen"] = float(np.mean([x["temps"] for x in sel]))
        r["ajustements_moyens"] = float(np.mean([x["ajustements"] for x in sel]))
    save_csv(tab, OUT / "recherche_resume.csv")
    for r in tab:
        print("  ", r["methode"].ljust(24), f"moyenne {r['moyenne']:.3f}  ",
              " ".join(f"{v:.2f}" for k, v in r.items() if k.startswith("n=")),
              f" {r['temps_moyen']:.1f} s")
    print(f"  total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()

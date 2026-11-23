"""
Expérience 3 : réseau de neurones + maximum de vraisemblance.

Le ML glouton peut rester bloqué dans un minimum local : il ajoute les
sources une par une au maximum de corrélation, et deux sources proches
ressemblent d'abord à une seule. Le réseau (``reseau.py``) propose
directement les K positions d'un coup (pics de sa carte). Pour chaque K :

    solution(K) = meilleure (plus petit χ²) entre
        - le chemin glouton habituel,
        - le ML lancé depuis les K plus hauts pics du réseau,
        - le ML lancé depuis la solution K-1 améliorée + un pic du réseau.

Puis on choisit K : BIC, classe prédite par le réseau, ou classifieur
appris sur le profil du chemin amélioré (gradient boosting, cf. expérience 1).
On compare aussi au réseau seul (pics + intensités par moindres carrés,
sans ML).

    python -m sparse_sources.apprentissage.hybride
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from .base import OUT, Fitter, evaluate, parallel  # noqa: E402
from .donnees import bases  # noqa: E402
from .rapport import pivot, save_csv  # noqa: E402
from .reseau import predict, train as train_net  # noqa: E402
from .selection_k import CLASSIFIERS, path_features, pick, rule_choice  # noqa: E402
from .simulation import PROJET, Scene  # noqa: E402

K_MAX = 7


def improved_path(args):
    """Chemin amélioré (même format que ``chemins.compute_path`` : K, chi2, z, I physiques)."""
    d, row, peaks, scene = args
    F = Fitter(d["m"], d["s"], scene)
    out = {"K": [0], "chi2": [F.chi2(np.zeros((0, 2)), np.zeros(0))],
           "z": [np.zeros((0, 2))], "I": [np.zeros(0)], "source": ["glouton"]}
    prev = (np.zeros((0, 2)), np.zeros(0))          # solution K-1 améliorée (noyau normalisé)
    for K in range(1, K_MAX + 1):
        cands = []
        # 1. glouton (recalculé dans le noyau normalisé à partir des positions du chemin)
        k = [i for i, kk in enumerate(row["K"]) if kk == K]
        if k:
            z = row["z"][k[0]]
            cands.append((*F.fit_from(z), "glouton"))
        # 2. K plus hauts pics du réseau
        if len(peaks) >= K:
            cands.append((*F.fit_from(peaks[:K]), "reseau"))
        # 3. solution K-1 améliorée + chaque pic du réseau non encore utilisé (au plus 3)
        if len(prev[1]):
            for p in peaks[:3]:
                if np.min(np.linalg.norm(prev[0] - p, axis=1)) > 0.5:
                    cands.append((*F.fit_from(np.vstack([prev[0], p])), "mixte"))
        cands = [c for c in cands if len(c[1]) == K]
        if not cands:
            continue
        z, I, c, src = min(cands, key=lambda t: t[2])
        prev = (z, I)
        out["K"].append(K)
        out["chi2"].append(c)
        out["z"].append(z)
        out["I"].append(F.physical(z, I))
        out["source"].append(src)
    return out


def nn_only(d, peaks, K, scene: Scene = PROJET):
    """Réseau seul : K plus hauts pics, intensités par NNLS (pas de ML)."""
    z = peaks[:K] if len(peaks) else (0.5 * (scene.lo + scene.hi))[None]
    A = scene.field  # mesures exactes d'une source unité
    G = np.stack([A(zz[None], [1.0]) for zz in z], 1) / d["s"][:, None]
    I, _ = nnls(G, d["m"] / d["s"])
    return z, I


def features_by_K(row, sigma):
    """Profil du chemin indexé par K = 0..K_MAX (K manquant : valeur du K précédent)."""
    c = np.empty(K_MAX + 1)
    last = row["chi2"][0]
    for K in range(K_MAX + 1):
        if K in row["K"]:
            last = row["chi2"][row["K"].index(K)]
        c[K] = last
    return path_features({"chi2": c}, sigma)


def references(d):
    """greedy_ml officiel (avec étape arrière) et ML parti des vraies positions (oracle)."""
    from sparse_sources import greedy_ml
    g = greedy_ml(d["m"], d["s"], setup=PROJET.setup, grid=PROJET.grid)
    F = Fitter(d["m"], d["s"])
    z, I, c = F.fit_from(d["z"])
    # oracle complet : vraies positions puis choix de K par BIC entre cette solution et greedy
    bic_o = c + 3 * len(I) * np.log(PROJET.M)
    best = (g["z"], g["I"]) if g["criterion"] < bic_o else (z, F.physical(z, I))
    return (g["z"], g["I"]), (z, F.physical(z, I)), best


def improved_for(data, paths, name, net, maps):
    f = OUT / f"{name}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    H, PK = predict(net, maps, data)
    peaks = [maps.peaks(h)[0] for h in H]
    res = parallel(improved_path, [(d, r, p, PROJET) for d, r, p in zip(data, paths, peaks)])
    with open(f, "wb") as fh:
        pickle.dump((res, peaks, PK), fh)
    return res, peaks, PK


def main():
    t0 = time.perf_counter()
    train, ptr, test, pte = bases()
    net, maps = train_net()
    itr, _, _ = improved_for(train, ptr, "chemins_ameliores_train", net, maps)
    ite, peaks, PK = improved_for(test, pte, "chemins_ameliores_test", net, maps)
    print(f"  chemins améliorés : {time.perf_counter() - t0:.0f} s")

    rows = []
    def rec(meth, i, z, I):
        d = test[i]
        rows.append({"methode": meth, "n": d["n"], "sigma": d["sigma"], "i": i,
                     **evaluate(d, z, I)})

    refs = parallel(references, test)
    for i, (g, o, ob) in enumerate(refs):
        rec("greedy_ml (reference)", i, *g)
        rec("ML depart vrai, K vrai", i, *o)
        rec("ML depart vrai + bic", i, *ob)
    k_nn = PK.argmax(1) + 1
    for i, d in enumerate(test):
        rec("bic (glouton)", i, *rule_choice(pte[i], "bic"))
        rec("reseau seul", i, *nn_only(d, peaks[i], k_nn[i]))
        rec("reseau+ML, K du reseau", i, *pick(ite[i], k_nn[i]))
        rec("reseau+ML+bic", i, *rule_choice(ite[i], "bic"))
        rec("reseau+ML, K oracle", i, *pick(ite[i], d["n"]))

    for clf_name in ("boosting", "mlp", "foret"):
        Xtr = np.stack([features_by_K(p, d["sigma"]) for p, d in zip(itr, train)])
        Xte = np.stack([features_by_K(p, d["sigma"]) for p, d in zip(ite, test)])
        clf = CLASSIFIERS[clf_name]().fit(Xtr, [d["n"] for d in train])
        pred = clf.predict(Xte)
        for i in range(len(test)):
            rec(f"reseau+ML+{clf_name}", i, *pick(ite[i], int(pred[i])))

    src = [s for r in ite for s in r["source"][1:]]
    print("  origine des solutions retenues :",
          {s: round(src.count(s) / len(src), 3) for s in set(src)})
    print("  précision du réseau sur K :", np.mean(k_nn == np.array([d["n"] for d in test])))
    save_csv(rows, OUT / "hybride.csv")
    tab = pivot(rows, "methode", ["n", "sigma"], "reussite")
    save_csv(tab, OUT / "hybride_resume.csv")
    for r in tab:
        print("  ", r["methode"].ljust(24), f"moyenne {r['moyenne']:.3f}  ",
              " ".join(f"{v:.2f}" for k, v in r.items() if k not in ("methode", "moyenne")))
    print(f"  total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()

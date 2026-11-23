"""
Expérience 1 : choisir le nombre de sources K sur le chemin ML.

Le chemin ML donne, pour chaque K = 0..7, la meilleure solution trouvée et
son χ². Il reste à choisir K. On compare :

    règles probabilistes (sans apprentissage)
        bic        χ² + 3K log M                (référence)
        aic        χ² + 3K · 2
        laplace    évidence bayésienne de Laplace (``chemins.laplace_evidence``)
    classifieurs appris (sklearn) : entrée = profil du chemin
        (log des gains de χ² à chaque ajout, log χ²/M, log σ), sortie = K vrai
        logistique, lda, knn, svm (noyau gaussien), arbre, foret,
        boosting (HistGradientBoosting), mlp (perceptron multicouche)

Entraînement : 3000 tirages simulés ; test : 600 autres tirages
(K = 1..5, σ ∈ {1e-4, 1e-3, 3e-3}, 40 tirages par case).

    python -m sparse_sources.apprentissage.selection_k
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from .base import OUT, evaluate  # noqa: E402
from .chemins import laplace_evidence  # noqa: E402
from .donnees import bases  # noqa: E402
from .rapport import pivot, save_csv  # noqa: E402

M = 64


def path_features(row, sigma):
    """Profil du chemin : gains de χ² successifs, χ²/M, log σ."""
    c = np.asarray(row["chi2"], float)
    gains = np.log1p(np.maximum(c[:-1] - c[1:], 0.0))
    return np.concatenate([gains, np.log(c / M), [np.log10(sigma)]])


def pick(row, K):
    """Élément du chemin dont le nombre de sources est K (ou le plus proche)."""
    Ks = np.asarray(row["K"])
    k = int(np.argmin(np.abs(Ks - K) + 1e-3 * np.arange(len(Ks))))
    return row["z"][k], row["I"][k]


def rule_choice(row, rule):
    """Index du chemin choisi par une règle (on exclut K = 0, toujours ≥ 1 source)."""
    K = np.asarray(row["K"])
    c = np.asarray(row["chi2"])
    if rule == "bic":
        score = c + 3 * K * np.log(M)
    elif rule == "aic":
        score = c + 3 * K * 2
    elif rule == "laplace":
        score = -np.array([laplace_evidence(row, k) for k in range(len(K))])
    score = np.where(K >= 1, score, np.inf)
    k = int(np.argmin(score))
    return row["z"][k], row["I"][k]


CLASSIFIERS = {
    "logistique": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
    "lda": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
    "svm": lambda: make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale")),
    "arbre": lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
    "foret": lambda: RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
    "boosting": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                       random_state=0),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000,
                                                                 alpha=1e-3, random_state=0)),
}


def main():
    t0 = time.perf_counter()
    train, ptr, test, pte = bases()
    Xtr = np.stack([path_features(p, d["sigma"]) for p, d in zip(ptr, train)])
    ytr = np.array([d["n"] for d in train])
    Xte = np.stack([path_features(p, d["sigma"]) for p, d in zip(pte, test)])
    yte = np.array([d["n"] for d in test])

    rows = []
    def record(meth, i, z, I, extra=None):
        d = test[i]
        rows.append({"methode": meth, "n": d["n"], "sigma": d["sigma"], "i": i,
                     **evaluate(d, z, I), **(extra or {})})

    for rule in ("bic", "aic", "laplace"):
        for i, p in enumerate(pte):
            record(rule, i, *rule_choice(p, rule))
    # oracle : on connaît le vrai K (borne haute de toute règle de choix sur ce chemin)
    for i, p in enumerate(pte):
        record("oracle_K", i, *pick(p, test[i]["n"]))

    for name, make in CLASSIFIERS.items():
        t = time.perf_counter()
        clf = make().fit(Xtr, ytr)
        pred = clf.predict(Xte)
        dt = time.perf_counter() - t
        for i, p in enumerate(pte):
            record(name, i, *pick(p, int(pred[i])))
        print(f"  {name:10s} précision sur K : {np.mean(pred == yte):.3f}  ({dt:.1f} s)")

    for rule in ("bic", "aic", "laplace", "oracle_K"):
        sel = [r for r in rows if r["methode"] == rule]
        print(f"  {rule:10s} précision sur K : {np.mean([r['exact'] for r in sel]):.3f}")

    save_csv(rows, OUT / "selection_k.csv")
    tab = pivot(rows, "methode", ["n", "sigma"], "reussite")
    save_csv(tab, OUT / "selection_k_resume.csv")
    for r in tab:
        print("  ", r["methode"].ljust(11), f"moyenne {r['moyenne']:.3f}  ",
              " ".join(f"{v:.2f}" for k, v in r.items() if k not in ("methode", "moyenne")))
    print(f"  total {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()

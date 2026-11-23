"""
Expérience 2 : régression directe, K connu.

Un modèle appris lit les 64 mesures et renvoie directement les positions
(et intensités) des sources. Pour K = 2, les sources sont rangées par
intensité décroissante (sinon la cible n'est pas une fonction des mesures).

Modèles (sklearn) : acp_lineaire (ACP 20 composantes + régression ridge),
knn, noyau (régression ridge à noyau gaussien = méthode RKHS), svr, foret,
boosting, mlp. Comparaison : maximum de vraisemblance (chemin ML à K fixé),
borne de Cramér-Rao, et « mlp + ML » (le ML part de la prédiction du MLP).

Entrée : x = m / s normalisé (x / ||x||, log ||x||) ; σ = 1e-3 fixé.
Aussi : ACP des mesures (combien de composantes portent l'information ?).

    python -m sparse_sources.apprentissage.regression_directe
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.multioutput import MultiOutputRegressor  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

from sparse_sources import cramer_rao  # noqa: E402

from .base import OUT, Fitter, evaluate, parallel  # noqa: E402
from .rapport import save_csv  # noqa: E402
from .simulation import PROJET  # noqa: E402

SIGMA = 1e-3


def feat(d):
    x = d["m"] / d["s"]
    n = np.linalg.norm(x)
    return np.concatenate([x / n, [np.log(n)]])


def target(d):
    """Sources rangées par intensité décroissante : (x1, y1, log I1, x2, ...)."""
    o = np.argsort(-d["I"])
    return np.concatenate([np.r_[d["z"][k], np.log(d["I"][k])] for k in o])


def decode(t, K):
    t = t.reshape(K, 3)
    return np.clip(t[:, :2], PROJET.lo, PROJET.hi), np.exp(t[:, 2])


MODELS = {
    "acp_lineaire": lambda: make_pipeline(StandardScaler(), PCA(20), Ridge(1e-3)),
    "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10, weights="distance")),
    "noyau_rkhs": lambda: make_pipeline(StandardScaler(), KernelRidge(alpha=1e-3, kernel="rbf",
                                                                      gamma=1.0 / 65 / 4)),
    "svr": lambda: make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=10.0, epsilon=0.01))),
    "foret": lambda: RandomForestRegressor(100, min_samples_leaf=2, random_state=0, n_jobs=-1),
    "boosting": lambda: MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=500,
                                                                           random_state=0)),
    "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=500,
                                                                early_stopping=True,
                                                                random_state=0)),
}
N_TRAIN = {"noyau_rkhs": 6000, "svr": 6000}   # méthodes à noyau : coût en N² ou N³


def ml_fit(args):
    """ML à K fixé : meilleur de (chemin glouton, départ en z0 si donné)."""
    d, K, z0 = args
    F = Fitter(d["m"], d["s"])
    z, I = np.zeros((0, 2)), np.zeros(0)
    for _ in range(K):
        z, I, c = F.add_one(z, I)
    best = (c, z, I)
    if z0 is not None:
        z1, I1, c1 = F.fit_from(z0)
        if len(I1) == K and c1 < best[0]:
            best = (c1, z1, I1)
    return best[1], F.physical(best[1], best[2])


def run_K(K, n_train=20000, n_test=300):
    rng = np.random.default_rng(10 + K)
    train = [PROJET.draw(K, SIGMA, rng) for _ in range(n_train)]
    test = [PROJET.draw(K, SIGMA, rng) for _ in range(n_test)]
    Xtr, Ytr = np.stack([feat(d) for d in train]), np.stack([target(d) for d in train])
    Xte = np.stack([feat(d) for d in test])
    rows = []

    def rec(meth, preds, t):
        for d, (z, I) in zip(test, preds):
            rows.append({"K": K, "methode": meth, "temps_par_tirage": t / len(test),
                         **evaluate(d, z, I)})

    preds_mlp = None
    for name, make in MODELS.items():
        t = time.perf_counter()
        n = N_TRAIN.get(name, n_train)
        mdl = make().fit(Xtr[:n], Ytr[:n])
        t_fit = time.perf_counter() - t
        t = time.perf_counter()
        P = mdl.predict(Xte)
        preds = [decode(p, K) for p in P]
        rec(name, preds, time.perf_counter() - t)
        if name == "mlp":
            preds_mlp = preds
        e = np.nanmedian([r["err_pos"] for r in rows if r["methode"] == name])
        print(f"  K={K} {name:13s} erreur médiane {e:.3f}  (apprentissage {t_fit:.0f} s, N={n})")

    t = time.perf_counter()
    rec("ML", parallel(ml_fit, [(d, K, None) for d in test]), time.perf_counter() - t)
    t = time.perf_counter()
    rec("mlp+ML", parallel(ml_fit, [(d, K, p[0]) for d, p in zip(test, preds_mlp)]),
        time.perf_counter() - t)
    crb = [float(cramer_rao(d["z"], d["I"], d["s"], PROJET.setup)["pos_std"].max()) for d in test]
    for d, c in zip(test, crb):
        rows.append({"K": K, "methode": "borne_CR", "err_pos": c, "reussite": np.nan,
                     "exact": True, "n_est": K})
    return rows, Xtr


def pca_study(X):
    """Variance expliquée cumulée de l'ACP des mesures blanchies normalisées."""
    p = PCA().fit(X[:, :-1])
    cum = np.cumsum(p.explained_variance_ratio_)
    return [{"composantes": k + 1, "variance_cumulee": float(cum[k])} for k in range(20)]


def main():
    all_rows = []
    for K in (1, 2):
        rows, Xtr = run_K(K)
        all_rows += rows
        if K == 2:
            save_csv(pca_study(Xtr), OUT / "acp_variance.csv")
    save_csv(all_rows, OUT / "regression_directe.csv")
    summ = []
    for K in (1, 2):
        for meth in dict.fromkeys(r["methode"] for r in all_rows if r["K"] == K):
            sel = [r for r in all_rows if r["K"] == K and r["methode"] == meth]
            summ.append({"K": K, "methode": meth,
                         "erreur_mediane": float(np.nanmedian([r["err_pos"] for r in sel])),
                         "erreur_moyenne": float(np.nanmean([r["err_pos"] for r in sel])),
                         "reussite": float(np.nanmean([r["reussite"] for r in sel])),
                         "temps_par_tirage": float(np.mean([r.get("temps_par_tirage", np.nan)
                                                            for r in sel]))})
    save_csv(summ, OUT / "regression_directe_resume.csv")
    for s in summ:
        print(f"  K={s['K']} {s['methode']:13s} médiane {s['erreur_mediane']:.4f} "
              f"moyenne {s['erreur_moyenne']:.4f} réussite {s['reussite']:.2f} "
              f"temps {s['temps_par_tirage']*1e3:.2f} ms")


if __name__ == "__main__":
    main()

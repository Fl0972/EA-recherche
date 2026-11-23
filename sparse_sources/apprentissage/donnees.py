"""
Bases fixes de l'étude (mises en cache) :
    entrainement  3000 tirages, K uniforme dans 1..5, σ ∈ {1e-4, 3e-4, 1e-3, 3e-3}
    test          5 K × 3 σ (1e-4, 1e-3, 3e-3) × 40 tirages = 600, autre graine
"""
from __future__ import annotations

import pickle

import numpy as np

from .base import OUT, test_set
from .chemins import paths_for
from .simulation import PROJET


def bases(n_train=3000):
    """(train, chemins_train, test, chemins_test)."""
    f = OUT / "bases.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            train, test = pickle.load(fh)
    else:
        train = PROJET.dataset(n_train, np.random.default_rng(1))
        test = test_set()
        with open(f, "wb") as fh:
            pickle.dump((train, test), fh)
    return train, paths_for(train, "chemins_train"), test, paths_for(test, "chemins_test")


if __name__ == "__main__":
    import time
    t = time.perf_counter()
    tr, ptr, te, pte = bases()
    print(len(tr), len(te), f"{time.perf_counter() - t:.0f} s")

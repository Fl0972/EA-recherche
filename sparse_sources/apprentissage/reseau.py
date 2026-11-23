"""
Réseau de neurones (perceptron multicouche, PyTorch) qui lit les 64 mesures
et prédit :
    - une carte de « présence » des sources sur une grille grossière de D
      (pas 0,5) : la cible est Σ_k exp(-||g - z_k||² / (2 w²)), w = 0,5 ;
    - le nombre de sources K (classification 1..5).

Entrée : mesures blanchies x = m / s, mises à l'échelle : (x / ||x||, log ||x||,
log σ). Les maxima locaux de la carte donnent des positions candidates, qui
servent de point de départ au maximum de vraisemblance (« NN + ML »).

Idée : c'est l'« inférence amortie » — l'apprentissage est fait une fois sur
des simulations, puis la prédiction est instantanée. Le réseau ne remplace
pas la vraisemblance : il propose de bons départs, le ML fait la précision.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from .base import OUT  # noqa: E402
from .simulation import PROJET, Scene  # noqa: E402

STEP, WIDTH = 0.5, 0.5
K_CLASSES = 5


def features(d):
    """Entrée du réseau pour un tirage (66,)."""
    x = d["m"] / d["s"]
    n = np.linalg.norm(x)
    return np.concatenate([x / n, [np.log(n), np.log10(d["sigma"])]])


class Maps:
    """Grille grossière de D et cartes cibles."""

    def __init__(self, scene: Scene = PROJET, step=STEP):
        self.scene = scene
        self.xs = np.arange(scene.lo[0], scene.hi[0] + 1e-9, step)
        self.ys = np.arange(scene.lo[1], scene.hi[1] + 1e-9, step)
        gx, gy = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.pts = np.stack([gx.ravel(), gy.ravel()], 1)
        self.shape = gx.shape

    def target(self, z):
        """Carte cible (P,)."""
        if len(z) == 0:
            return np.zeros(len(self.pts))
        d2 = ((self.pts[:, None] - z[None]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * WIDTH**2)).sum(1).clip(0, 1)

    def peaks(self, h, n_max=8, thr=0.1):
        """Maxima locaux (8 voisins) de la carte, triés par hauteur."""
        H = h.reshape(self.shape)
        P = np.pad(H, 1, constant_values=-np.inf)
        nb = np.stack([P[1 + dx:P.shape[0] - 1 + dx, 1 + dy:P.shape[1] - 1 + dy]
                       for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy])
        mask = (H >= nb.max(0)) & (H > thr)
        idx = np.argwhere(mask)
        vals = H[mask]
        order = np.argsort(-vals)[:n_max]
        out = []
        for i, j in idx[order]:      # affinage : barycentre local 3×3
            i0, i1, j0, j1 = max(i - 1, 0), min(i + 2, H.shape[0]), max(j - 1, 0), min(j + 2, H.shape[1])
            w = np.clip(H[i0:i1, j0:j1], 0, None)
            gx, gy = np.meshgrid(self.xs[i0:i1], self.ys[j0:j1], indexing="ij")
            out.append([(w * gx).sum() / w.sum(), (w * gy).sum() / w.sum()])
        return np.array(out).reshape(-1, 2), vals[order]


class Net(nn.Module):
    """MLP : 66 → 512 → 512 → 512 → (carte P, logits K)."""

    def __init__(self, n_in, n_map, hidden=512):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.map = nn.Linear(hidden, n_map)
        self.k = nn.Linear(hidden, K_CLASSES)

    def forward(self, x):
        h = self.body(x)
        return self.map(h), self.k(h)


def train(scene: Scene = PROJET, n_train=150_000, epochs=40, seed=0, name="reseau",
          sigmas=(1e-4, 3e-4, 1e-3, 3e-3), verbose=True):
    """Entraîne (ou recharge) le réseau. Renvoie (net, maps)."""
    f = OUT / f"{name}.pt"
    maps = Maps(scene)
    net = Net(66, len(maps.pts))
    if f.exists():
        net.load_state_dict(torch.load(f))
        net.eval()
        return net, maps
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 100)
    t0 = time.perf_counter()
    data = scene.dataset(n_train, rng, sigmas=sigmas)
    X = torch.tensor(np.stack([features(d) for d in data]), dtype=torch.float32)
    Yn = np.empty((n_train, len(maps.pts)), np.float32)
    for i, d in enumerate(data):
        Yn[i] = maps.target(d["z"])
    Y = torch.from_numpy(Yn)
    Kt = torch.tensor([d["n"] - 1 for d in data])
    if verbose:
        print(f"  données : {n_train} tirages, {time.perf_counter() - t0:.0f} s")
    n_val = n_train // 20
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n_train - n_val) + n_val
        for b in perm.split(512):
            pm, pk = net(X[b])
            loss = bce(pm, Y[b]) * 10 + ce(pk, Kt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            net.eval()
            with torch.no_grad():
                pm, pk = net(X[:n_val])
                acc = (pk.argmax(1) == Kt[:n_val]).float().mean().item()
                lv = bce(pm, Y[:n_val]).item()
            print(f"  époque {ep + 1}: perte carte (val) {lv:.4f}, précision K (val) {acc:.3f},"
                  f" {time.perf_counter() - t0:.0f} s")
    net.eval()
    torch.save(net.state_dict(), f)
    return net, maps


def predict(net, maps, data):
    """Cartes (N, P) et probabilités de K (N, 5)."""
    X = torch.tensor(np.stack([features(d) for d in data]), dtype=torch.float32)
    with torch.no_grad():
        pm, pk = net(X)
    return torch.sigmoid(pm).numpy(), torch.softmax(pk, 1).numpy()

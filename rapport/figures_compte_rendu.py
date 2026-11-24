# A lancer depuis le dossier rapport/ : python figures_compte_rendu.py
"""Figures du compte rendu (partie 2). Lit results/sources/comparaison_*.jsonl via metriques.py (sources.pkl).
Fusion des sources à moins de 0,1 (intensités additionnées, barycentre pondéré), appariement hongrois,
source détectée si la source appariée est à moins de 0,5."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
DELTA, SEUIL = 0.1, 0.5
def fusion(z, I):
    z = np.asarray(z, float).reshape(-1, 2); I = np.asarray(I, float)
    if len(z) < 2: return z, I
    lab = fcluster(linkage(z, "single"), DELTA, criterion="distance")
    zz, II = [], []
    for g in np.unique(lab):
        m = lab == g; w = I[m]; s = w.sum()
        zz.append((z[m] * w[:, None]).sum(0) / s if s > 0 else z[m].mean(0)); II.append(s)
    return np.array(zz), np.array(II)
rows = []
for cfg in ["cercle", "grille"]:
    recs, verite = {}, {}
    for l in open(f"../results/sources/comparaison_{cfg}.jsonl"):
        d = json.loads(l); k = (d["N"], d["sigma"], d["ic_ratio"], d["tirage"])
        recs[k + (d["methode"],)] = d
        if "z_vrai" in d: verite[k] = (d["z_vrai"], d["I_vrai"])
    print(cfg, "points sans vérité :", sum(1 for k in recs if k[:4] not in verite))
    for k, d in recs.items():
        zv, Iv = fusion(*verite[k[:4]])
        if d.get("z_est"):
            ze, Ie = fusion(d["z_est"], d["I_est"])
        else:
            ze, Ie = np.zeros((0, 2)), np.zeros(0)
        dist = np.full(len(zv), np.inf); rel = np.full(len(zv), np.nan)
        if len(ze):
            C = np.linalg.norm(zv[:, None] - ze[None], axis=2)
            r, c = linear_sum_assignment(C)
            dist[r] = C[r, c]; rel[r] = np.abs(Ie[c] - Iv[r]) / Iv[r]
        for i in range(len(zv)):
            rows.append(dict(config=cfg, N=d["N"], sigma=d["sigma"], ic=d["ic_ratio"], tirage=d["tirage"],
                             methode=d["methode"], temps=d["temps"], statut=d["statut"], I=Iv[i], S=Iv.sum(),
                             dc=np.linalg.norm(zv[i] - np.array([5, 10])), d=dist[i], r=rel[i], det=dist[i] < SEUIL))
df = pd.DataFrame(rows)



plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6,
    "lines.linewidth": 1.8, "lines.markersize": 4.5, "savefig.bbox": "tight"})
C = {"prony": "#2a6fdb", "ml_bic": "#e2583e", "fw": "#169f78"}
M = {"prony": "o", "ml_bic": "s", "fw": "^"}
L = {"prony": "Prony", "ml_bic": "ML + BIC", "fw": "Frank–Wolfe glissant"}
METH = ["prony", "fw", "ml_bic"]
OUT = "images/"
def save(f, n): f.savefig(OUT + n + ".png", dpi=220); f.savefig(OUT + n + ".pdf"); plt.close(f)
def legende(fig, ax, extra=True, y=-0.06, ncol=5):
    h = [Line2D([], [], color=C[m], marker=M[m], label=L[m]) for m in METH]
    if extra:
        h += [Line2D([], [], color="gray", label="cercle (64 capteurs)"),
              Line2D([], [], color="gray", ls="--", label="cercle + grille (576 capteurs)")]
    fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))
def courbe(a, x, s, m, cfg):
    if cfg == "grille" and m == "prony": return
    a.plot(x, s, marker=M[m], color=C[m], ls="-" if cfg == "cercle" else "--", mfc=C[m] if cfg == "cercle" else "white")

# ---- Figure : détection et précision en fonction du bruit ----
b = df[df.ic == 50]
fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
for j, n in enumerate([4, 6, 10]):
    for cfg in ["cercle", "grille"]:
        for m in METH:
            s = b[(b.config == cfg) & (b.methode == m) & (b.N == n)]
            courbe(ax[0, j], *zip(*s.groupby("sigma").det.mean().items()), m, cfg)
            courbe(ax[1, j], *zip(*s[s.det].groupby("sigma").d.median().items()), m, cfg)
    ax[0, j].set(title=f"N = {n}", ylim=(-0.03, 1.03))
    ax[1, j].set(xscale="log", yscale="log", xlabel="bruit relatif σ", ylim=(2e-5, 0.5))
    for a in ax[:, j]: a.axvline(1e-3, color="gray", lw=0.8, ls="-.")
ax[0, 0].set_ylabel("sources détectées\n(à moins de 0,5)"); ax[1, 0].set_ylabel("erreur de position médiane\ndes sources détectées")
legende(fig, ax, y=-0.05)
fig.tight_layout(); save(fig, "cr_bruit")

# ---- Figure : cadre par défaut (N, intensité, temps) ----
d = df[(df.sigma == 1e-3) & (df.ic == 50)]
fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
bins = [0, 0.02, 0.05, 0.1, 0.2, 1]; lab = ["< 2 %", "2–5 %", "5–10 %", "10–20 %", "> 20 %"]
d4 = d[d.N >= 4].assign(b=lambda t: pd.cut(t.I / t.S, bins, labels=lab))
t = d.drop_duplicates(["config", "methode", "N", "tirage"])
for cfg in ["cercle", "grille"]:
    for m in METH:
        s = d[(d.config == cfg) & (d.methode == m)]
        courbe(ax[0], *zip(*s.groupby("N").det.mean().items()), m, cfg)
        s = d4[(d4.config == cfg) & (d4.methode == m)]
        courbe(ax[1], range(5), s.groupby("b", observed=False).det.mean().values, m, cfg)
        s = t[(t.config == cfg) & (t.methode == m)]
        courbe(ax[2], *zip(*s.groupby("N").temps.median().items()), m, cfg)
ax[0].set(title="(a) Détection selon le nombre de sources", xlabel="nombre de sources N", ylabel="sources détectées", xticks=[2, 4, 6, 10], ylim=(-0.03, 1.03))
ax[1].set(title="(b) Détection selon l'intensité (N = 4 à 10)", xlabel="intensité de la source / S", ylim=(-0.03, 1.03))
ax[1].set_xticks(range(5), lab)
ax[2].set(title="(c) Temps de calcul (médiane)", xlabel="nombre de sources N", ylabel="temps (s)", yscale="log", xticks=[2, 4, 6, 10])
legende(fig, ax, y=-0.12)
fig.tight_layout(); save(fig, "cr_defaut")

# ---- Figure : exemple de reconstruction ----
def charge(cfg, key):
    rec, ver = {}, None
    for l in open(f"../results/sources/comparaison_{cfg}.jsonl"):
        r = json.loads(l)
        if (r["N"], r["sigma"], r["ic_ratio"], r["tirage"]) == key:
            rec[r["methode"]] = r
            if "z_vrai" in r: ver = (r["z_vrai"], r["I_vrai"])
    return rec, ver
from scipy.optimize import linear_sum_assignment
key = (6, 1e-3, 50.0, 21)
fig, ax = plt.subplots(1, 2, figsize=(10, 5.6))
R = np.sqrt(125); th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
for a, cfg, titre in [(ax[0], "cercle", "(a) Cercle seul (64 capteurs)"), (ax[1], "grille", "(b) Cercle + grille (576 capteurs)")]:
    rec, (zv, Iv) = charge(cfg, key)
    zv, Iv = fusion(zv, Iv)
    a.plot(5 + R * np.cos(th), 10 + R * np.sin(th), ".", color="#9a9a9a", ms=3, zorder=1)
    if cfg == "grille":
        gx, gy = np.meshgrid(np.linspace(10 / 32, 10 - 10 / 32, 16), np.linspace(20 / 64, 20 - 20 / 64, 32))
        a.plot(gx, gy, ".", color="#c8c8c8", ms=1.5, zorder=1)
    a.plot([0, 10, 10, 0, 0], [0, 0, 20, 20, 0], "k", lw=1)
    a.plot(5, 10, "*", color="k", ms=11, zorder=3)
    bilan = []
    for z, I in zip(zv, Iv):
        a.add_patch(plt.Circle(z, 0.5, fill=False, ls=":", color="k", lw=0.9, zorder=2))
        a.plot(*z, "o", mfc="none", mec="k", ms=3 + 0.9 * np.sqrt(I), mew=1.4, zorder=2)
        a.annotate(f"{I:.0f}", z, zorder=6, xytext=(8, 6), textcoords="offset points", fontsize=7.5)
    for m in ["ml_bic", "fw", "prony"]:
        r = rec[m]
        ze, Ie = fusion(r["z_est"], r["I_est"]) if r["z_est"] else (np.zeros((0, 2)), np.zeros(0))
        if len(ze):
            Cm = np.linalg.norm(zv[:, None] - ze[None], axis=2); i, k = linear_sum_assignment(Cm)
            ndet = int((Cm[i, k] < 0.5).sum())
        else: ndet = 0
        bilan.append(f"{L[m]} : {ndet}/{len(zv)}")
        a.scatter(ze[:, 0], ze[:, 1], marker=M[m], s=18 + 3 * Ie, color=C[m], alpha=0.85, zorder=4,
                  edgecolor="white", lw=0.5)
    a.set(title=titre, aspect="equal", xlim=(-6.5, 16.5), ylim=(-1.6, 21.6))
    a.text(0.5, -0.12, "détectées :  " + "   ·   ".join(bilan), transform=a.transAxes, ha="center", fontsize=8.5)
    a.grid(False)
h = [Line2D([], [], marker="o", mfc="none", mec="k", ls="", ms=8, label="vraie source (nombre : intensité)"),
     Line2D([], [], color="k", ls=":", label="zone de détection (rayon 0,5)"),
     Line2D([], [], marker="*", color="k", ls="", ms=10, label="source centrale"),
     Line2D([], [], marker=".", color="#9a9a9a", ls="", label="capteurs")] + \
    [Line2D([], [], marker=M[m], color=C[m], ls="", ms=7, label=L[m]) for m in ["ml_bic", "fw", "prony"]]
fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); save(fig, "cr_exemple")

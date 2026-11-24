import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6, "lines.linewidth": 1.8, "lines.markersize": 4.5, "savefig.bbox": "tight"})
C = {"desc": "#e0a100", "gn": "#6a4fd6", "ml": "#e2583e", "fw": "#169f78", "prony": "#2a6fdb", "orano": "black"}
M = {"desc": "D", "gn": "v", "ml": "s", "fw": "^", "prony": "o"}
L = {"desc": "Descente de gradient (N connu)", "gn": "Gauss–Newton (N connu, même départ)",
     "ml": "ML + BIC", "fw": "Frank–Wolfe glissant", "prony": "Prony"}
out = "/home/claude/synth/images/"
def save(fig, nom):
    fig.savefig(out + nom + ".png", dpi=220); fig.savefig(out + nom + ".pdf"); plt.close(fig)

# ---------- Figure 1 : cas à deux sources ----------
x = [1, 3, 10, 30, 100, 300]
fig, ax = plt.subplots(1, 3, figsize=(10, 3.0))
a = ax[0]
for k, v in [("gn", [100,100,100,100,100,99]), ("ml", [100]*6), ("fw", [100]*6),
             ("prony", [97,95,89,69,39,22]), ("desc", [71,68,52,30,13,2])]:
    a.plot(x, np.array(v)/100, marker=M[k], color=C[k], label=L[k])
a.scatter([1,3,10,30,100], [.95,.85,.55,.20,.05], marker="x", s=45, color="k", zorder=5, label="Résultats d'Orano (descente)")
a.set(xscale="log", title="(a) Cas test d'Orano : σ = 0, sans source centrale",
      xlabel="rapport d'intensité x = I₁ / I₂", ylabel="réussite", ylim=(-0.04, 1.05))
a = ax[1]
for k, cer, gri in [("desc", [30,30,22,13,3,1], [0]*6), ("ml", [43,38,25,15,7,2], [100,100,99,98,95,85]),
                    ("fw", [13,13,10,9,6,3], [91,88,83,79,66,36])]:
    a.plot(x, np.array(cer)/100, marker=M[k], color=C[k])
    a.plot(x, np.array(gri)/100, marker=M[k], color=C[k], ls="--", mfc="white")
a.plot([], [], color="gray", label="cercle (64 capteurs)"); a.plot([], [], color="gray", ls="--", label="cercle + grille (576)")
a.legend(loc="center right", frameon=False)
a.set(xscale="log", title="(b) Cadre du projet : σ = 10⁻³, I_c = 50 S", xlabel="rapport d'intensité x", ylim=(-0.04, 1.05))
a = ax[2]
med = np.array([7126,5404,8458,46916,4.03e5,3.62e6]); q1 = [1502,1584,1076,2917,13739,93539]; q3 = [34485,33450,1.15e5,5.55e5,5.77e6,5.16e7]
a.fill_between(x, q1, q3, color=C["desc"], alpha=0.18, lw=0)
a.plot(x, med, marker="D", color=C["desc"], label="κ médian (Q1–Q3)")
xx = np.array([1, 300]); a.plot(xx, 8458*(xx/10)**2, "k--", lw=1, label="pente 2 : κ ∝ x²")

a.set(xscale="log", yscale="log", title="(c) Conditionnement de la perte à la solution", xlabel="rapport d'intensité x", ylabel="κ = λ_max / λ_min")
a.legend(loc="lower right", frameon=False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
fig.tight_layout(); save(fig, "fig_deux_sources")

# ---------- Figure 2 : nombre de sources et temps ----------
N = [2, 4, 6, 10]
fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
E = {"prony": ([.453,1.84,2.36,2.71], None), "ml": ([.0932,1.24,1.85,2.59], [5.9e-4,1.53e-3,3.28e-3,6.71e-3]),
     "fw": ([.302,1.75,2.22,2.76], [9.97e-4,7.96e-3,2.63e-2,6.8e-2])}
P = {"prony": ([16,19,20,2], None), "ml": ([92,23,0,0], [99,93,93,81]), "fw": ([82,17,1,0], [89,57,27,21])}
T = {"prony": ([1.31e-3,1.26e-3,1.3e-3,1.27e-3], None), "ml": ([.214,.357,.463,.611], [.551,1.06,1.83,6.76]),
     "fw": ([.0669,.105,.138,.161], [.349,.606,1.07,3.58])}
for a, D, yl, log in [(ax[0], E, "erreur E (médiane)", True), (ax[1], P, "bon nombre de sources (N̂ = N)", False), (ax[2], T, "temps par tirage (s)", True)]:
    for k in ["prony", "ml", "fw"]:
        cer, gri = D[k]
        cer = np.array(cer, float) / (1 if log else 100)
        a.plot(N, cer, marker=M[k], color=C[k], label=L[k] if a is ax[0] else None)
        if gri is not None:
            a.plot(N, np.array(gri, float) / (1 if log else 100), marker=M[k], color=C[k], ls="--", mfc="white")
    a.set(xticks=N, xlabel="nombre de sources N", ylabel=yl)
    if log: a.set_yscale("log")
ax[0].axhline(1, color="gray", lw=0.8, ls=":"); ax[0].text(5.2, 0.6, "≈ 1 source sur 4 manquée", fontsize=7, color="gray")
ax[0].set_title("(a) Erreur"); ax[1].set_title("(b) Estimation du nombre de sources"); ax[2].set_title("(c) Temps de calcul")
ax[1].set_ylim(-0.04, 1.05)
ax[0].plot([], [], color="gray", label="cercle (64 capteurs)"); ax[0].plot([], [], color="gray", ls="--", label="cercle + grille (576 capteurs)")
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1))
fig.tight_layout(); save(fig, "fig_nsources")

# ---------- Figure 3 : bruit et source centrale ----------
sig = [1e-4, 2.15e-4, 4.64e-4, 1e-3, 2.15e-3, 4.64e-3, 1e-2]
bruit = {4: {"prony": [1.23,1.23,1.39,1.84,2.26,2.38,2.65], "ml": [.314,1.02,1.13,1.24,1.5,2.19,2.34], "fw": [1.13,1.21,1.36,1.75,2.17,2.4,2.85],
             "mlg": [1.52e-4,3.28e-4,7.07e-4,1.53e-3,3.22e-3,6.87e-3,1.53e-2], "fwg": [8.39e-4,1.75e-3,3.71e-3,7.96e-3,1.81e-2,3.62e-2,.726]},
         6: {"prony": [2.01,1.93,2.2,2.36,2.57,2.87,2.98], "ml": [1.08,1.48,1.62,1.85,2.26,2.73,2.97], "fw": [1.6,1.74,1.95,2.22,2.54,3.0,3.32],
             "mlg": [3.19e-4,6.9e-4,1.5e-3,3.28e-3,7.03e-3,1.5e-2,4.2e-2], "fwg": [4.22e-3,7.71e-3,1.5e-2,2.63e-2,5.84e-2,.567,.747]},
         10: {"prony": [2.63,2.62,2.64,2.71,2.9,3.07,3.2], "ml": [1.87,2.11,2.32,2.59,2.71,3.0,3.28], "fw": [2.14,2.36,2.56,2.76,3.0,3.3,3.45],
              "mlg": [7.78e-4,1.67e-3,3.49e-3,6.71e-3,1.51e-2,3.36e-2,9.49e-2], "fwg": [1.21e-2,2.01e-2,3.49e-2,6.8e-2,.246,.584,1.13]}}
rap = [0, 10, 25, 50, 100, 200]
cent = {4: {"prony": [1.23,1.24,1.47,1.84,2.23,2.29], "ml": [.0556,1.02,1.15,1.24,1.47,2.1], "fw": [.788,1.23,1.37,1.75,2.17,2.33],
            "mlg": [1.69e-4,4.52e-4,8.57e-4,1.53e-3,2.74e-3,5.5e-3], "fwg": [4.56e-4,2.08e-3,4.42e-3,7.96e-3,1.51e-2,3.13e-2]},
        6: {"prony": [1.97,1.98,2.24,2.36,2.58,2.72], "ml": [.774,1.46,1.62,1.85,2.23,2.48], "fw": [1.23,1.79,1.98,2.22,2.48,2.97],
            "mlg": [2.31e-4,8.49e-4,1.77e-3,3.28e-3,6.12e-3,1.17e-2], "fwg": [9.33e-4,9.13e-3,1.63e-2,2.63e-2,5.29e-2,.238]},
        10: {"prony": [2.63,2.62,2.71,2.71,2.92,3.08], "ml": [1.55,2.11,2.34,2.59,2.68,2.97], "fw": [1.88,2.36,2.57,2.76,2.97,3.17],
             "mlg": [3.4e-4,1.83e-3,3.52e-3,6.71e-3,1.2e-2,2.6e-2], "fwg": [3.23e-3,2.05e-2,4.01e-2,6.8e-2,.151,.509]}}
fig, ax = plt.subplots(2, 3, figsize=(10, 5.4), sharey=True)
xc = np.arange(len(rap))
for j, n in enumerate([4, 6, 10]):
    for row, (X, D) in enumerate([(sig, bruit[n]), (xc, cent[n])]):
        a = ax[row, j]
        for k in ["prony", "ml", "fw"]:
            a.plot(X, D[k], marker=M[k], color=C[k], label=L[k] if (row == 0 and j == 0) else None)
        for k in ["ml", "fw"]:
            a.plot(X, D[k + "g"], marker=M[k], color=C[k], ls="--", mfc="white")
        a.set_yscale("log"); a.set_ylim(5e-5, 6)
        a.axhline(1, color="gray", lw=0.8, ls=":")
        if row == 0:
            a.set(xscale="log", xlabel="bruit relatif σ", title=f"N = {n}")
            a.axvline(1e-3, color="gray", lw=0.8, ls="-.")
        else:
            a.set_xticks(xc, [str(r) for r in rap]); a.set(xlabel="I_c / S")
            a.axvline(3, color="gray", lw=0.8, ls="-.")
ax[0, 0].set_ylabel("erreur E (médiane)\nbalayage en bruit, I_c = 50 S"); ax[1, 0].set_ylabel("erreur E (médiane)\nsource centrale, σ = 10⁻³")
ax[0, 0].plot([], [], color="gray", label="cercle (64 capteurs)"); ax[0, 0].plot([], [], color="gray", ls="--", label="cercle + grille (576 capteurs)")
ax[0, 0].plot([], [], color="gray", ls="-.", lw=0.8, label="cadre par défaut")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(); save(fig, "fig_balayages")

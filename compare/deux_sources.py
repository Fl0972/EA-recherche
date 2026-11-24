"""
Analyse approfondie du cas à 2 sources : pourquoi la descente de gradient
(méthode d'Orano) échoue déjà, et ce que font les autres méthodes.

    python -m compare.deux_sources --all [--n-trials 100] [--n-proc 2]
    python -m compare.deux_sources --figures          retrace depuis les CSV

Expériences (situations du protocole commun : positions des 2 sources
uniformes dans D, mêmes graines que compare.run) :

  rapport   intensités imposées I = S (x, 1) / (x + 1), x ∈ {1, 3, 10, 30, 100, 300},
            dans deux cadres : « orano » (σ = 0, pas de source centrale, le cas test
            d'Orano) et « projet » (σ = 10⁻³, I_c = 50 S). Méthodes : descente (N connu),
            Gauss–Newton (même perte, même initialisation, N connu), ML + BIC,
            Frank–Wolfe, Prony. Pour la descente on garde de quoi classer les échecs.
  bassins   une situation, la position initiale de la source 1 parcourt une grille
            de D (la source 2 part toujours du même point) : issue de la descente
            et de Gauss–Newton depuis chaque point de départ.
  paysage   une situation : carte de la perte d'Orano en fonction de la position
            d'une source (l'autre et les intensités à leur vraie valeur), avec la
            trajectoire de la descente, sur le cercle et sur cercle + grille.

Réussite (critère d'Orano) : N̂ = 2 et chaque source à moins de 0,1.
Classement d'un échec de la descente (dans cet ordre) :
  source éteinte      une intensité a été projetée à 0 : ∂L/∂z_i ∝ I_i, la source ne bouge plus ;
  budget épuisé       20 000 itérations sans converger ;
  limite statistique  la perte finale est ≤ à celle du minimum voisin de la vérité
                      (Gauss–Newton parti de la vérité) : c'est le minimum lui-même qui est
                      faux, aucun algorithme minimisant cette perte ne ferait mieux ;
  bloquée au bord     une source finit sur le bord de D (projection active) ;
  minimum local       arrêt avant le budget sur une perte plus grande que le bon minimum.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from compare import protocol as P  # noqa: E402

RDIR = ROOT / "results" / "deux_sources"
FDIR = ROOT / "figures" / "deux_sources"
XS = [1, 3, 10, 30, 100, 300]
CADRES = {"orano": (0.0, 0.0), "projet": (1e-3, 50.0)}       # (σ, I_c / S)
TOL = 0.1
SOURCES = ROOT / "results" / "sources" / "deux_sources.jsonl"   # sources estimées, tirage par tirage
SIGMA_VIDE = 1e-6        # ML et FW sans bruit : la vraisemblance a besoin d'un niveau de bruit
CATEGORIES = ["réussite", "source éteinte", "budget épuisé", "limite statistique",
              "bloquée au bord", "minimum local"]


# ---- situations à intensités imposées --------------------------------------------------
def situation_x(x, tirage, sigma, ic_ratio, config):
    """Mêmes positions et bruit que protocol.situation(2, tirage, …), intensités S (x, 1)/(x+1)."""
    sit = P.situation(2, tirage, sigma, ic_ratio, config)
    I = P.S_TOTAL * np.array([x, 1.0]) / (x + 1.0)
    eps = P.rng_de(2, tirage, 1).standard_normal(P.N_CERCLE + P.GRILLE[0] * P.GRILLE[1])
    X = sit.X
    ic = ic_ratio * P.S_TOTAL
    yc = ic / ((X - P.CENTRE) ** 2).sum(1) if ic > 0 else np.zeros(len(X))
    y_brut = (P.champ(X, sit.z, I) + yc) * (1 + sigma * eps[:len(X)])
    m = y_brut - yc
    s_eff = sigma if sigma > 0 else SIGMA_VIDE
    s = np.maximum(s_eff * np.abs(y_brut), 1e-12 * max(float(np.abs(y_brut).max()), 1.0))
    return P.Situation(sit.z, I, X, m, s, y_brut, sit.u0, 2, sigma, ic)


def reussite(z, z_est, I_est, N=2):
    """Ancien critère : (N̂ = N et chaque source à moins de TOL, erreur de position max)."""
    z_est = np.asarray(z_est, float).reshape(-1, 2)
    e = P.erreur_ancienne(z, np.ones(len(z)), z_est, np.ones(len(z_est)))
    return bool(e["n_est"] == N and e["d_max"] < TOL), e["d_max"]


def reussite_fusion(z, I, z_est, I_est, delta=P.DELTA_DEF):
    """Critère après fusion : chaque vraie source appariée à moins de TOL, quel que soit N̂."""
    e = P.erreur(z, I, z_est, I_est, delta=delta)
    return bool(e["d_max"] < TOL), e["d_max"]


def _sources(out, nom, z, I):
    out[f"zest_{nom}"] = np.asarray(z, float).reshape(-1, 2).tolist()
    out[f"Iest_{nom}"] = np.asarray(I, float).reshape(-1).tolist()


# ---- une tâche « rapport » ---------------------------------------------------------------
def tache_rapport(t):
    from sparse_sources.orano import Probleme, descente_projetee, gauss_newton
    sig, ic = CADRES[t["cadre"]]
    sit = situation_x(t["x"], t["tirage"], sig, ic, t["config"])
    pb = Probleme(sit.X, P.LO, P.HI, sit.m)
    a0 = np.full(2, 0.5)
    out = {k: t[k] for k in ("cadre", "x", "tirage", "config")}
    sep = float(np.linalg.norm(sit.z[0] - sit.z[1]))
    out["separation"] = sep
    out["z_vrai"], out["I_vrai"] = sit.z.tolist(), sit.I.tolist()
    # vérité et minimum voisin de la vérité (Gauss–Newton parti de la vérité)
    u_v, a_v = pb.normalise(sit.z, sit.I)
    out["perte_verite"] = pb.perte(u_v, a_v)
    ref = gauss_newton(pb, u_v, a_v)
    out["perte_ref"] = ref["perte"]
    out["d_ref"] = reussite(sit.z, ref["p"], ref["I"])[1]
    _sources(out, "ref", ref["p"], ref["I"])
    J = pb.jacobien_residu(u_v, a_v)
    sv = np.linalg.svd(J, compute_uv=False)
    out["kappa"] = float((sv[0] / sv[-1]) ** 2)                  # conditionnement de JᵀJ
    # descente
    t0 = time.perf_counter()
    d = descente_projetee(pb, sit.u0, a0, iter_max=20000, trajectoire=False)
    out["t_descente"] = time.perf_counter() - t0
    ok, dmax = reussite(sit.z, d["p"], d["I"])
    _sources(out, "descente", d["p"], d["I"])
    out.update(ok_descente=ok, d_descente=dmax, it_descente=d["iterations"],
               perte_descente=d["perte"], Imin_descente=float(d["I"].min()))
    u_f, _ = pb.normalise(d["p"], d["I"])
    out["bord_descente"] = bool(np.any((u_f <= 1e-9) | (u_f >= 1 - 1e-9)))
    out["cat_descente"] = categorie(out)
    # Gauss–Newton, même initialisation
    t0 = time.perf_counter()
    g = gauss_newton(pb, sit.u0, a0)
    out["t_gn"] = time.perf_counter() - t0
    out["ok_gn"], out["d_gn"] = reussite(sit.z, g["p"], g["I"])
    _sources(out, "gn", g["p"], g["I"])
    # méthodes qui estiment N
    for m in ("ml_bic", "fw", "prony"):
        t0 = time.perf_counter()
        z, I, _ = P.lancer(m, sit, t["config"])
        out[f"t_{m}"] = time.perf_counter() - t0
        out[f"ok_{m}"], out[f"d_{m}"] = reussite(sit.z, z, I)
        out[f"n_{m}"] = len(I)
        _sources(out, m, z, I)
    for m in ("ref", "descente", "gn", "ml_bic", "fw", "prony"):
        out[f"okf_{m}"], out[f"df_{m}"] = reussite_fusion(sit.z, sit.I, out[f"zest_{m}"], out[f"Iest_{m}"])
    return out


def categorie(o):
    if o["ok_descente"]:
        return "réussite"
    if o["Imin_descente"] < 1e-6 * P.S_TOTAL:
        return "source éteinte"
    if o["it_descente"] >= 20000:
        return "budget épuisé"
    if o["perte_descente"] <= o["perte_ref"] * (1 + 1e-6) + 1e-15:
        return "limite statistique"
    if o["bord_descente"]:
        return "bloquée au bord"
    return "minimum local"


# ---- une tâche « bassin » ------------------------------------------------------------------
def tache_bassin(t):
    from sparse_sources.orano import Probleme, descente_projetee, gauss_newton
    sig, ic = CADRES[t["cadre"]]
    sit = situation_x(t["x"], t["tirage"], sig, ic, t["config"])
    pb = Probleme(sit.X, P.LO, P.HI, sit.m)
    u0 = np.array([t["u1"], sit.u0[1]])
    a0 = np.full(2, 0.5)
    d = descente_projetee(pb, u0, a0, iter_max=20000)
    u_v, a_v = pb.normalise(sit.z, sit.I)
    ref = gauss_newton(pb, u_v, a_v)
    o = {"ok_descente": reussite(sit.z, d["p"], d["I"])[0], "Imin_descente": float(d["I"].min()),
         "it_descente": d["iterations"], "perte_descente": d["perte"], "perte_ref": ref["perte"]}
    u_f, _ = pb.normalise(d["p"], d["I"])
    o["bord_descente"] = bool(np.any((u_f <= 1e-9) | (u_f >= 1 - 1e-9)))
    g = gauss_newton(pb, u0, a0)
    return {**{k: t[k] for k in ("cadre", "x", "tirage", "config")},
            "u1x": float(t["u1"][0]), "u1y": float(t["u1"][1]),
            "cat_descente": categorie(o), "it_descente": d["iterations"],
            "ok_gn": reussite(sit.z, g["p"], g["I"])[0]}


def tache(t):
    return tache_bassin(t) if t["type"] == "bassin" else tache_rapport(t)


# ---- exécution ------------------------------------------------------------------------------
def _cle(r):
    return (r["cadre"], float(r["x"]), int(r["tirage"]), r["config"])


def executer(taches, chemin, n_proc, timeout=300, jsonl=None):
    """
    Lance les tâches ; si ``jsonl`` est donné, chaque résultat y est ajouté au fil de l'eau
    (sources estimées comprises) et les tâches déjà présentes ne sont pas relancées.
    """
    from compare.pool import map_timeout
    RDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fh = None
    if jsonl is not None:
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        if jsonl.exists():
            with open(jsonl, encoding="utf-8") as f:
                rows = [json.loads(L) for L in f if L.strip()]
        faites = {_cle(r) for r in rows}
        taches = [t for t in taches if _cle(t) not in faites]
        print(f"   {len(rows)} tâches déjà faites, {len(taches)} à faire", flush=True)
        fh = open(jsonl, "a", encoding="utf-8")

    def ok(i, r):
        if "__exception__" not in r:
            rows.append(r)
            if fh is not None:
                fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
                fh.flush()
        else:
            print("   exception :", r["__exception__"][:200], flush=True)

    def to(i, t):
        print("   timeout :", {k: t[k] for k in t if k != "u1"}, flush=True)

    if taches:
        map_timeout(tache, taches, n_proc, timeout, ok, to, progression=max(len(taches) // 20, 1))
    if fh is not None:
        fh.close()
    cles = []
    for r in rows:
        cles += [k for k in r if k not in cles and not k.startswith(("zest_", "Iest_", "z_vrai", "I_vrai"))]
    with open(chemin, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cles, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def calcul_rapport(n_trials, n_proc):
    taches = [{"type": "rapport", "cadre": c, "x": x, "tirage": t, "config": "cercle"}
              for c in CADRES for x in XS for t in range(n_trials)]
    # cadre du projet sur cercle + grille aussi (mêmes tirages)
    taches += [{"type": "rapport", "cadre": "projet", "x": x, "tirage": t, "config": "grille"}
               for x in XS for t in range(n_trials)]
    taches.sort(key=lambda t: -t["x"])
    return executer(taches, RDIR / "rapport.csv", n_proc, jsonl=SOURCES)


def choisir_situation(rows, cadre, x, config="cercle"):
    """Un tirage où la descente échoue mais où le bon minimum est retrouvable (d_ref < 0,1)."""
    rows = sorted(rows, key=lambda r: int(float(r["tirage"])))      # choix indépendant de l'ordre de calcul
    for cat in ("minimum local", "budget épuisé", "source éteinte", "bloquée au bord"):
        c = [r for r in rows if r["cadre"] == cadre and float(r["x"]) == x and r["config"] == config
             and r["cat_descente"] == cat and float(r["d_ref"]) < TOL]
        if c:
            return int(c[0]["tirage"]), cat
    return 0, "?"


def calcul_bassins(rows_rapport, n_proc, n=(12, 24)):
    sel = []
    for cadre, x in (("orano", 1), ("orano", 30), ("projet", 1)):
        t, cat = choisir_situation(rows_rapport, cadre, x)
        sel.append((cadre, x, t, cat))
    gx = (np.arange(n[0]) + 0.5) / n[0]
    gy = (np.arange(n[1]) + 0.5) / n[1]
    taches = [{"type": "bassin", "cadre": c, "x": x, "tirage": t, "config": "cercle",
               "u1": np.array([a, b])} for c, x, t, _ in sel for a in gx for b in gy]
    rows = executer(taches, RDIR / "bassins.csv", n_proc)
    with open(RDIR / "bassins_situations.json", "w", encoding="utf-8") as f:
        json.dump([{"cadre": c, "x": x, "tirage": t, "categorie_init_protocole": cat}
                   for c, x, t, cat in sel], f, ensure_ascii=False, indent=1)
    return rows


def lire_csv(chemin):
    with open(chemin, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pa = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument("--all", action="store_true")
    pa.add_argument("--figures", action="store_true")
    pa.add_argument("--n-trials", type=int, default=100)
    pa.add_argument("--n-proc", type=int, default=2)
    pa.add_argument("--rapport-seul", action="store_true",
                    help="ne pas recalculer les bassins d'attraction (inchangés par la métrique)")
    args = pa.parse_args()
    t0 = time.perf_counter()
    if args.all:
        print("[rapport]", flush=True)
        rr = calcul_rapport(args.n_trials, args.n_proc)
        print(f"  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if not args.rapport_seul:
            print("[bassins]", flush=True)
            calcul_bassins(rr, args.n_proc)
        print(f"  {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    from compare import figures_deux_sources as F
    F.toutes(RDIR, FDIR)
    print(f"Durée : {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""
Comparaison des méthodes sur beaucoup de tirages.

Une « tâche » = (capteurs, N, σ, I_c/S, x, tirage, méthode). Pour chaque tâche, on
enregistre dans un fichier JSONL (une ligne par tâche) les vraies sources et les sources
estimées. Les indicateurs (``metrique.erreur``) sont recalculés depuis ce fichier : changer
la métrique, le rayon de fusion ou le rayon de tolérance ne demande jamais de relancer les
méthodes, et un calcul interrompu reprend où il s'était arrêté.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

from . import metrique
from .methodes import lancer, nom_canonique
from .probleme import situation

N_BOOT = 2000


# ---- tâches et stockage ----------------------------------------------------------------
def fmt(v) -> str:
    return "" if v is None or v == "" else f"{float(v):.6g}"


def cle(r) -> tuple:
    """Identifiant d'une tâche (ou d'une ligne du fichier de sources)."""
    return (r["capteurs"], int(r["N"]), fmt(r["sigma"]), fmt(r["ic"]), fmt(r.get("x")),
            int(r["tirage"]), nom_canonique(r["methode"]))


def taches(methodes, Ns, sigmas, ics, capteurs, n_tirages, xs=(None,), premier_tirage=0):
    """Produit cartésien de tous les paramètres."""
    out = []
    for c, N, s, ic, x, t, m in itertools.product(capteurs, Ns, sigmas, ics, xs,
                                                  range(premier_tirage, premier_tirage + n_tirages),
                                                  methodes):
        out.append({"capteurs": c, "N": int(N), "sigma": float(s), "ic": float(ic),
                    "x": None if x is None else float(x), "tirage": int(t),
                    "methode": nom_canonique(m)})
    return out


def tache(t: dict) -> dict:
    """Exécute une tâche (fonction de module : appelée dans les processus de calcul)."""
    sit = situation(t["N"], t["tirage"], t["sigma"], t["ic"], t["capteurs"], x=t.get("x"))
    try:
        out = lancer(t["methode"], sit)
        statut = "ok"
    except Exception as e:                      # une méthode qui plante compte comme échec
        out = {"z": np.zeros((0, 2)), "I": np.zeros(0), "temps": np.nan, "erreur": repr(e)[:200]}
        statut = "exception"
    ligne = {**t, "statut": statut, "temps": out["temps"],
             "z_vrai": np.round(sit.z, 10).tolist(), "I_vrai": np.round(sit.I, 10).tolist(),
             "z_est": out["z"].tolist(), "I_est": out["I"].tolist()}
    for k in ("iterations", "perte", "chi2", "erreur"):
        if k in out:
            ligne[k] = out[k]
    return ligne


def ligne_timeout(t: dict) -> dict:
    sit = situation(t["N"], t["tirage"], t["sigma"], t["ic"], t["capteurs"], x=t.get("x"))
    return {**t, "statut": "timeout", "temps": np.nan, "z_vrai": sit.z.tolist(),
            "I_vrai": sit.I.tolist(), "z_est": [], "I_est": []}


def lire_sources(path) -> dict:
    """{clé: ligne} ; si une tâche apparaît plusieurs fois, la dernière ligne l'emporte."""
    path = Path(path)
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for L in fh:
                if L.strip():
                    r = json.loads(L)
                    out[cle(r)] = r
    return out


def ecrire_ligne(fh, ligne: dict):
    fh.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    fh.flush()


# ---- indicateurs par tirage et agrégation ---------------------------------------------
def indicateurs(r: dict, delta=metrique.DELTA, seuil=metrique.SEUIL) -> dict:
    """Une ligne par tirage : paramètres + indicateurs (recalculés depuis les sources)."""
    e = metrique.erreur(r["z_vrai"], r["I_vrai"], r["z_est"] or np.zeros((0, 2)), r["I_est"] or [],
                        delta, seuil)
    e.pop("d")
    return {"capteurs": r["capteurs"], "N": int(r["N"]), "sigma": float(r["sigma"]),
            "ic": float(r["ic"]), "x": "" if r.get("x") is None else float(r["x"]),
            "tirage": int(r["tirage"]), "methode": nom_canonique(r["methode"]),
            "statut": r["statut"], "temps": r.get("temps"), **e,
            "tous_detectes": bool(e["detection"] == 1.0),
            "N_exact": bool(e["n_est"] == int(r["N"]))}


def bootstrap(v, stat=np.mean, n_boot=N_BOOT, graine=0):
    """Statistique et intervalle de confiance à 95 % (bootstrap percentile) ; ignore les nan."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(graine)
    b = stat(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    return float(stat(v)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


PARAMS = ["capteurs", "N", "sigma", "ic", "x", "methode"]


def agreger(lignes: list[dict]) -> list[dict]:
    """Une ligne par point (capteurs, N, σ, I_c, x, méthode)."""
    groupes = {}
    for r in lignes:
        groupes.setdefault(tuple(r[k] for k in PARAMS), []).append(r)
    out = []
    for k, v in sorted(groupes.items(), key=lambda kv: tuple(str(a) for a in kv[0])):
        col = lambda nom: np.array([r[nom] for r in v], float)
        Em, Eb, Eh = bootstrap(col("E"), np.median)
        Mm, Mb, Mh = bootstrap(col("E"), np.mean)
        Dm, Db, Dh = bootstrap(col("detection"), np.mean)
        Pm, Pb, Ph = bootstrap(col("d_det"), np.median)
        t = np.array([r["temps"] for r in v if r["statut"] == "ok"], float)
        out.append({**dict(zip(PARAMS, k)), "tirages": len(v),
                    "E_mediane": Em, "E_mediane_ic_bas": Eb, "E_mediane_ic_haut": Eh,
                    "E_moyenne": Mm, "E_moyenne_ic_bas": Mb, "E_moyenne_ic_haut": Mh,
                    "detection": Dm, "detection_ic_bas": Db, "detection_ic_haut": Dh,
                    "tous_detectes": float(col("tous_detectes").mean()),
                    "precision_d": Pm, "precision_d_ic_bas": Pb, "precision_d_ic_haut": Ph,
                    "precision_r": float(np.nanmedian(col("r_det"))) if np.isfinite(col("r_det")).any()
                    else np.nan,
                    "masse_parasite": float(col("masse_parasite").mean()),
                    "sources_en_trop": float(col("fausses").mean()),
                    "N_exact": float(col("N_exact").mean()),
                    "temps_median": float(np.median(t)) if t.size else np.nan,
                    "timeouts": sum(r["statut"] == "timeout" for r in v),
                    "exceptions": sum(r["statut"] == "exception" for r in v)})
    return out


def ecrire_csv(path, lignes: list[dict]):
    if not lignes:
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, list(lignes[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(lignes)

"""Tableaux de résultats (CSV) communs aux expériences."""

from __future__ import annotations

import csv

import numpy as np


def save_csv(rows, path):
    """Écrit une liste de dicts (colonnes = union des clés, dans l'ordre d'apparition)."""
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def pivot(rows, by, cols, field):
    """
    Tableau croisé : une ligne par valeur de ``by`` (méthode), une colonne par
    combinaison de ``cols`` (par ex. n, σ), valeur = moyenne de ``field`` ;
    plus la colonne « moyenne » (sur toutes les lignes).
    """
    groups = sorted({tuple(r[c] for c in cols) for r in rows})
    out = []
    for b in dict.fromkeys(r[by] for r in rows):
        sel = [r for r in rows if r[by] == b]
        row = {by: b}
        for g in groups:
            v = [float(r[field]) for r in sel if tuple(r[c] for c in cols) == g]
            row["_".join(f"{c}={x:g}" for c, x in zip(cols, g))] = float(np.mean(v)) if v else np.nan
        row["moyenne"] = float(np.mean([float(r[field]) for r in sel]))
        out.append(row)
    return out

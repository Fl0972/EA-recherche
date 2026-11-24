"""
Mesure de l'erreur d'une reconstruction.

1. Fusion : les sources estimées distantes de moins de δ (0,1 par défaut) sont fusionnées
   (regroupement hiérarchique à lien simple ; intensité = somme, position = barycentre
   pondéré), et de même les vraies sources. Une source coupée en deux morceaux voisins
   n'est donc pas une erreur.
2. Appariement hongrois entre sources (fusionnées) estimées et vraies.
3. Indicateurs, pour les N' vraies sources fusionnées :
   - d_i = ||ẑ_i − z_i|| ≤ 5, r_i = |Î_i − I_i| / I_i ≤ 1 ; source manquée = (5, 1) ;
   - taux de détection : part des vraies sources avec d_i < seuil (0,1 par défaut, le
     « rayon de tolérance ») ;
   - précision : d_i et r_i moyens sur les seules sources détectées ;
   - masse parasite : intensité des sources estimées non appariées, rapportée à S ;
   - score global E = (1/N') Σ (0,75 d_i + 0,25 r_i) : une source manquée ajoute 4/N'.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment

DELTA = 0.1              # rayon de fusion
SEUIL = 0.1              # rayon de tolérance de la détection
D_MAX, R_MAX = 5.0, 1.0  # plafonds de d_i et r_i
ALPHA, BETA = 0.75, 0.25


def fusion(z, I, delta: float = DELTA):
    """Fusionne les sources à moins de δ (lien simple) : intensité = somme, position = barycentre."""
    z = np.asarray(z, float).reshape(-1, 2)
    I = np.asarray(I, float).reshape(-1)
    if delta <= 0 or len(I) <= 1:
        return z, I
    lab = fcluster(linkage(z, "single"), t=delta, criterion="distance")
    zs, Is = [], []
    for k in np.unique(lab):
        sel = lab == k
        w = np.clip(I[sel], 0, None)
        Is.append(I[sel].sum())
        zs.append(w @ z[sel] / w.sum() if w.sum() > 0 else z[sel].mean(0))
    return np.array(zs).reshape(-1, 2), np.array(Is)


def erreur(z, I, z_est, I_est, delta: float = DELTA, seuil: float = SEUIL) -> dict:
    """
    Indicateurs d'un tirage (vraies sources z, I ; estimées z_est, I_est). Renvoie :
    E, detection, d_det, r_det (précision), masse_parasite, d_moy, r_moy, d_max,
    n_est (nombre brut de sources estimées), n_fusion, N_fusion, fausses, d (par vraie source).
    """
    zt, It = fusion(z, I, delta)
    ze, Ie = fusion(z_est, I_est, delta)
    N, n = len(zt), len(ze)
    d = np.full(N, D_MAX)
    r = np.full(N, R_MAX)
    apparie = np.zeros(n, bool)
    if n and N:
        C = np.linalg.norm(zt[:, None] - ze[None], axis=-1)
        a, b = linear_sum_assignment(C)
        d[a] = np.minimum(C[a, b], D_MAX)
        r[a] = np.minimum(np.abs(Ie[b] - It[a]) / It[a], R_MAX)
        apparie[b] = True
    S = float(np.sum(I))
    det = d < seuil
    return {"E": float(np.mean(ALPHA * d + BETA * r)),
            "detection": float(det.mean()) if N else np.nan,
            "d_det": float(d[det].mean()) if det.any() else np.nan,
            "r_det": float(r[det].mean()) if det.any() else np.nan,
            "masse_parasite": float(np.clip(Ie[~apparie], 0, None).sum() / S) if S > 0 else 0.0,
            "d_moy": float(d.mean()), "r_moy": float(r.mean()), "d_max": float(d.max()),
            "n_est": int(len(np.asarray(I_est).reshape(-1))), "n_fusion": n, "N_fusion": N,
            "fausses": int(n - apparie.sum()), "d": d}

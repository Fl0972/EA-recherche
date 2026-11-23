"""
Méthode de l'équipe industrielle (Orano) : descente de gradient projetée sur
les positions et intensités de N sources (N connu), et Gauss-Newton à région
de confiance sur la même perte, pour comparaison.

Conventions (celles du reste de sparse_sources) :
    D = [0, 10] × [0, 20], capteurs s_k sur le cercle circonscrit à D ;
    modèle direct      Φ_θ(s) = Σ_i I_i / |s − p_i|² ;
    perte              L(θ) = ½ Σ_k (Φ_θ(s_k) − y_k)² / |y|²   (moindres carrés NON pondérés) ;
    paramètres normalisés θ = (u, a) : p_i = lo + u_i ⊙ (hi − lo), u_i ∈ [0, 1]²,
                       I_i = a_i × I_total, a_i ≥ 0.

Gradient exact : ∂Φ/∂I_i = 1/|s − p_i|², ∂Φ/∂p_i = I_i · 2 (s − p_i) / |s − p_i|⁴,
puis règle de la chaîne : ∂L/∂u = ∂L/∂p ⊙ (hi − lo), ∂L/∂a = ∂L/∂I × I_total.

Descente projetée : θ_{n+1} = P_C(θ_n − η_n ∇L), C = D^N × R_+^N ; pas d'Armijo
(départ au double du pas précédent, division par 2 jusqu'à
L(new) ≤ L(old) − c ∇L · (old − new), c = 1e-4) ; déplacement des positions
limité à 5 % du domaine par itération (sur chaque coordonnée normalisée) ;
budget 20 000 itérations ; arrêt si L < 1e-14 ou si le pas devient négligeable.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

I_TOTAL = 100.0          # intensité totale des sources recherchées
I_CENTRALE = 5000.0      # source centrale du projet


class Probleme:
    """Capteurs, domaine, mesures y (source centrale retirée) et |y|²."""

    def __init__(self, capteurs, lo, hi, y):
        self.S = np.asarray(capteurs, float)                     # (K, 2)
        self.lo = np.asarray(lo, float)
        self.echelle = np.asarray(hi, float) - self.lo           # (2,)
        self.y = np.asarray(y, float)
        self.ny2 = float(self.y @ self.y)

    # ---- passage paramètres normalisés <-> physiques
    def physique(self, u, a):
        """(u, a) normalisés -> (p, I) physiques."""
        return self.lo + u * self.echelle, a * I_TOTAL

    def normalise(self, p, I):
        """(p, I) physiques -> (u, a) normalisés."""
        return (np.asarray(p, float) - self.lo) / self.echelle, np.asarray(I, float) / I_TOTAL

    # ---- modèle direct, perte, gradient (vectorisés sur capteurs et sources)
    def modele(self, p, I):
        """Φ(s_k) pour chaque capteur, (K,)."""
        d2 = ((self.S[:, None, :] - p[None, :, :]) ** 2).sum(-1)  # (K, N)
        return (I / d2).sum(1)

    def perte(self, u, a):
        """L(u, a)."""
        p, I = self.physique(u, a)
        r = self.modele(p, I) - self.y
        return 0.5 * float(r @ r) / self.ny2

    def perte_gradient(self, u, a):
        """L et ∇L par rapport aux paramètres normalisés : (L, g_u (N, 2), g_a (N,))."""
        p, I = self.physique(u, a)
        diff = self.S[:, None, :] - p[None, :, :]                 # s_k − p_i, (K, N, 2)
        d2 = (diff ** 2).sum(-1)                                  # (K, N)
        r = (I / d2).sum(1) - self.y                              # résidu (K,)
        L = 0.5 * float(r @ r) / self.ny2
        g_I = (r @ (1.0 / d2)) / self.ny2                         # ∂L/∂I_i
        g_p = np.einsum("k,kn,knj->nj", r, 2 * I / d2 ** 2, diff) / self.ny2   # ∂L/∂p_i
        return L, g_p * self.echelle, g_I * I_TOTAL               # règle de la chaîne

    def jacobien_residu(self, u, a):
        """Jacobien du résidu normalisé (Φ − y)/|y| par rapport à (u, a), (K, 3N)."""
        p, I = self.physique(u, a)
        diff = self.S[:, None, :] - p[None, :, :]
        d2 = (diff ** 2).sum(-1)
        Jp = (2 * I / d2 ** 2)[:, :, None] * diff * self.echelle  # (K, N, 2)
        Ja = I_TOTAL / d2                                         # (K, N)
        return np.hstack([Jp.reshape(len(self.S), -1), Ja]) / np.sqrt(self.ny2)


def descente_projetee(pb: Probleme, u0, a0, iter_max=20000, c=1e-4, dep_max=0.05,
                      tol_perte=1e-14, tol_pas=1e-14, eta0=1e-2, trajectoire=False):
    """
    Descente de gradient projetée, pas d'Armijo (méthode Orano). Renvoie un
    dict : p, I (physiques), perte, iterations, temps, trajectoire (positions
    physiques à chaque itération si demandé, (n+1, N, 2)).
    """
    t0 = time.perf_counter()
    u, a = np.array(u0, float), np.array(a0, float)
    L, gu, ga = pb.perte_gradient(u, a)
    eta = eta0
    traj = [pb.physique(u, a)[0]] if trajectoire else None
    n = 0
    for n in range(1, iter_max + 1):
        eta *= 2.0                                               # double du pas précédent
        bloque = False
        while True:
            du = np.clip(-eta * gu, -dep_max, dep_max)           # positions : au plus 5 % du domaine
            u_new = np.clip(u + du, 0.0, 1.0)                    # projection sur D
            a_new = np.maximum(a - eta * ga, 0.0)                # projection sur R_+
            L_new = pb.perte(u_new, a_new)
            dec = float((gu * (u - u_new)).sum() + ga @ (a - a_new))
            if L_new <= L - c * dec:
                break
            eta /= 2.0
            if eta < 1e-30:                                      # pas négligeable : on s'arrête
                bloque = True
                break
        if bloque:
            n -= 1
            break
        pas = np.sqrt(((u_new - u) ** 2).sum() + ((a_new - a) ** 2).sum())
        u, a = u_new, a_new
        L, gu, ga = pb.perte_gradient(u, a)
        if trajectoire:
            traj.append(pb.physique(u, a)[0])
        if L < tol_perte or pas < tol_pas:
            break
    p, I = pb.physique(u, a)
    return {"p": p, "I": I, "perte": L, "iterations": n, "temps": time.perf_counter() - t0,
            "trajectoire": np.array(traj) if trajectoire else None}


def gauss_newton(pb: Probleme, u0, a0, max_nfev=2000):
    """Gauss-Newton à région de confiance (scipy least_squares, 'trf', bornes), même perte."""
    t0 = time.perf_counter()
    N = len(a0)
    ny = np.sqrt(pb.ny2)

    def residu(th):
        u, a = th[:2 * N].reshape(N, 2), th[2 * N:]
        return (pb.modele(*pb.physique(u, a)) - pb.y) / ny

    def jac(th):
        return pb.jacobien_residu(th[:2 * N].reshape(N, 2), th[2 * N:])

    lb = np.zeros(3 * N)
    ub = np.r_[np.ones(2 * N), np.full(N, np.inf)]
    th0 = np.clip(np.r_[np.ravel(u0), a0], lb, ub - 1e-12)
    res = least_squares(residu, th0, jac=jac, bounds=(lb, ub), method="trf",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
    u, a = res.x[:2 * N].reshape(N, 2), res.x[2 * N:]
    p, I = pb.physique(u, a)
    return {"p": p, "I": I, "perte": 0.5 * float(res.fun @ res.fun), "iterations": int(res.nfev),
            "temps": time.perf_counter() - t0}

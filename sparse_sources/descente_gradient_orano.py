"""
Méthode de l'équipe industrielle (Orano) : descente de gradient projetée, étendue à N sources.

Reproduit le cas test de la synthèse « Cas test à deux sources : descente de gradient et
rapport d'intensité » (F. Guillaumey, nov. 2026) :
  - modèle direct  Phi(s) = sum_i I_i / |s - p_i|^2, 64 capteurs sur le cercle circonscrit à D = [0,10] x [0,20] ;
  - perte          L = 1/2 * sum_k (Phi(s_k) - y_k)^2 / |y|^2 ;
  - paramètres normalisés : positions ramenées à [0,1] dans D, intensités divisées par l'intensité totale ;
  - descente projetée sur C = D^N x R_+^N, pas d'Armijo (départ au double du pas précédent, division par 2
    jusqu'à L(new) <= L - c * grad . (old - new), c = 1e-4), déplacement limité à 5 % du domaine par itération ;
  - budget : 20 000 itérations. Le nombre de sources N est supposé connu (comme dans la méthode d'Orano).

Comparaison optionnelle (--gn) : Gauss-Newton avec région de confiance (scipy least_squares), même perte,
même initialisation.

Exemples :
  python descente_gradient_orano.py                                  # N = 2, 3, 5, 10 ; intensités tirées sur le simplexe
  python descente_gradient_orano.py --sources 2 --rapport 1 3 10 30 100 --gn   # reproduit le tableau 1 de la synthèse
  python descente_gradient_orano.py --sources 3 5 --rapport 10 --sigma 1e-3 --centrale --gn
  python descente_gradient_orano.py --sources 5 --figure             # trajectoires sur un tirage
"""

import argparse
import csv
import time

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

# ----------------------------------------------------------------------------- géométrie du projet
LX, LY = 10.0, 20.0                       # domaine D = [0, LX] x [0, LY]
CENTRE = np.array([LX / 2, LY / 2])
RAYON = np.hypot(LX / 2, LY / 2)          # cercle circonscrit, R = sqrt(125)
ECHELLE = np.array([LX, LY])              # positions normalisées u in [0,1]^2 -> p = u * ECHELLE
I_TOTAL = 100.0                           # intensité totale des sources recherchées
I_CENTRALE = 5000.0


def capteurs(K=64):
    t = 2 * np.pi * np.arange(K) / K
    return CENTRE + RAYON * np.c_[np.cos(t), np.sin(t)]


# ----------------------------------------------------------------------------- modèle direct
def modele(S, p, I):
    """Phi(s_k) pour chaque capteur. S : (K,2), p : (N,2), I : (N,)."""
    d2 = ((S[:, None, :] - p[None, :, :]) ** 2).sum(-1)          # (K,N)
    return (I / d2).sum(1)


def perte_et_gradient(S, y, ny2, u, a):
    """Perte normalisée et gradient par rapport aux paramètres normalisés (u, a)."""
    p, I = u * ECHELLE, a * I_TOTAL
    diff = S[:, None, :] - p[None, :, :]                           # s_k - p_i, (K,N,2)
    d2 = (diff ** 2).sum(-1)                                       # (K,N)
    r = (I / d2).sum(1) - y                                        # résidu (K,)
    L = 0.5 * (r @ r) / ny2
    # dPhi/dI_i = 1/d2 ; dPhi/dp_i = I_i * 2 (s - p_i) / d2^2
    g_I = (r @ (1.0 / d2)) / ny2                                   # (N,)
    g_p = np.einsum('k,kn,knj->nj', r, 2 * I / d2 ** 2, diff) / ny2  # (N,2)
    return L, g_p * ECHELLE, g_I * I_TOTAL                         # règle de la chaîne vers (u, a)


def perte(S, y, ny2, u, a):
    r = modele(S, u * ECHELLE, a * I_TOTAL) - y
    return 0.5 * (r @ r) / ny2


# ----------------------------------------------------------------------------- descente de gradient projetée
def descente_gradient(S, y, u0, a0, iter_max=20000, c=1e-4, pas_max_pos=0.05,
                      tol_perte=1e-14, tol_pas=1e-13, garder_trajectoire=False):
    """Descente de gradient projetée avec règle d'Armijo (méthode Orano)."""
    ny2 = y @ y
    u, a = u0.copy(), a0.copy()
    L, gu, ga = perte_et_gradient(S, y, ny2, u, a)
    eta = 1e-2
    traj = [u * ECHELLE] if garder_trajectoire else None
    n = 0
    for n in range(1, iter_max + 1):
        eta *= 2.0                                                 # on part du double du pas précédent
        gmax = np.abs(gu).max()
        if gmax > 0:
            eta = min(eta, pas_max_pos / gmax)                     # déplacement <= 5 % du domaine
        while True:
            u_new = np.clip(u - eta * gu, 0.0, 1.0)                # projection : sources dans D
            a_new = np.maximum(a - eta * ga, 0.0)                  # intensités positives
            L_new = perte(S, y, ny2, u_new, a_new)
            descente = gu.ravel() @ (u - u_new).ravel() + ga @ (a - a_new)
            if L_new <= L - c * descente or eta < 1e-30:
                break
            eta /= 2.0
        pas = np.sqrt(((u_new - u) ** 2).sum() + ((a_new - a) ** 2).sum())
        u, a = u_new, a_new
        L, gu, ga = perte_et_gradient(S, y, ny2, u, a)
        if garder_trajectoire:
            traj.append(u * ECHELLE)
        if L < tol_perte or pas < tol_pas:
            break
    return dict(p=u * ECHELLE, I=a * I_TOTAL, perte=L, iterations=n,
                trajectoire=np.array(traj) if garder_trajectoire else None)


# ----------------------------------------------------------------------------- Gauss-Newton (comparaison)
def gauss_newton(S, y, u0, a0):
    """Gauss-Newton à région de confiance sur la même perte (scipy least_squares, méthode trf)."""
    N = len(a0)
    ny = np.sqrt(y @ y)

    def residu(th):
        u, a = th[:2 * N].reshape(N, 2), th[2 * N:]
        return (modele(S, u * ECHELLE, a * I_TOTAL) - y) / ny

    def jac(th):
        u, a = th[:2 * N].reshape(N, 2), th[2 * N:]
        p, I = u * ECHELLE, a * I_TOTAL
        diff = S[:, None, :] - p[None, :, :]
        d2 = (diff ** 2).sum(-1)
        Jp = (2 * I / d2 ** 2)[:, :, None] * diff * ECHELLE        # (K,N,2)
        JI = I_TOTAL / d2                                           # (K,N)
        return np.hstack([Jp.reshape(len(S), 2 * N), JI]) / ny

    th0 = np.r_[u0.ravel(), a0]
    lb = np.r_[np.zeros(2 * N), np.zeros(N)]
    ub = np.r_[np.ones(2 * N), np.full(N, np.inf)]
    res = least_squares(residu, np.clip(th0, lb, ub - 1e-12), jac=jac, bounds=(lb, ub),
                        method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    u, a = res.x[:2 * N].reshape(N, 2), res.x[2 * N:]
    return dict(p=u * ECHELLE, I=a * I_TOTAL, perte=0.5 * (res.fun @ res.fun), iterations=res.nfev)


# ----------------------------------------------------------------------------- situations de test
def tirer_situation(N, rng, rapport=None, sep_min=0.5):
    """Positions uniformes dans D (séparation minimale sep_min) et intensités de somme I_TOTAL.
    rapport=None : intensités uniformes sur le simplexe (comme config/).
    rapport=x    : intensités géométriquement réparties de x*I à I (rapport max/min = x)."""
    for _ in range(10000):
        p = rng.uniform(0, 1, (N, 2)) * ECHELLE
        d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)) + np.eye(N) * 1e9
        if d.min() >= sep_min:
            break
    if rapport is None:
        I = rng.dirichlet(np.ones(N))
    else:
        I = rapport ** (-np.arange(N) / max(N - 1, 1))
        rng.shuffle(I)
    return p, I / I.sum() * I_TOTAL


def mesures(S, p, I, sigma, centrale, rng):
    """Mesures y_k : bruit relatif gaussien sur le signal brut, source centrale retirée."""
    y = modele(S, p, I)
    yc = I_CENTRALE / ((S - CENTRE) ** 2).sum(1) if centrale else 0.0
    y_brut = y + yc
    if sigma > 0:
        y_brut = y_brut * (1 + sigma * rng.standard_normal(len(S)))
    return y_brut - yc


def erreurs(p_vrai, I_vrai, p_est, I_est):
    """Appariement optimal vraies / estimées ; renvoie les erreurs de position et d'intensité."""
    C = np.sqrt(((p_vrai[:, None] - p_est[None]) ** 2).sum(-1))
    i, j = linear_sum_assignment(C)
    return C[i, j], np.abs(I_vrai[i] - I_est[j])


# ----------------------------------------------------------------------------- expérience
def experience(args):
    S = capteurs(args.capteurs)
    lignes = []
    rapports = args.rapport if args.rapport else [None]
    print(f"{'N':>3} {'rapport':>8} | {'méthode':<16} | {'réussite':>8} | {'itér. méd.':>10} | "
          f"{'err. pos. méd.':>14} | {'sources à 0,1':>13} | {'temps méd.':>10}")
    for N in args.sources:
        for x in rapports:
            res = {'descente': [], 'gauss-newton': []}
            for t in range(args.tirages):
                rng = np.random.default_rng(1000 * N + t)
                p, I = tirer_situation(N, rng, x, args.sep_min)
                y = mesures(S, p, I, args.sigma, args.centrale, rng)
                u0 = rng.uniform(0, 1, (N, 2))                     # même initialisation pour les deux méthodes
                a0 = np.full(N, 1.0 / N)
                methodes = [('descente', lambda: descente_gradient(S, y, u0, a0, args.iter))]
                if args.gn:
                    methodes.append(('gauss-newton', lambda: gauss_newton(S, y, u0, a0)))
                for nom, f in methodes:
                    t0 = time.perf_counter()
                    r = f()
                    dt = time.perf_counter() - t0
                    ep, ei = erreurs(p, I, r['p'], r['I'])
                    res[nom].append(dict(ok=bool((ep < args.tol).all()), it=r['iterations'],
                                         ep=ep.max(), n_ok=int((ep < args.tol).sum()), t=dt))
            for nom, R in res.items():
                if not R:
                    continue
                ligne = dict(N=N, rapport='simplexe' if x is None else x, methode=nom,
                             reussite=np.mean([r['ok'] for r in R]),
                             iterations=np.median([r['it'] for r in R]),
                             err_pos=np.median([r['ep'] for r in R]),
                             sources_ok=np.mean([r['n_ok'] for r in R]),
                             temps=np.median([r['t'] for r in R]))
                lignes.append(ligne)
                print(f"{N:>3} {str(ligne['rapport']):>8} | {nom:<16} | {100 * ligne['reussite']:>7.0f}% | "
                      f"{ligne['iterations']:>10.0f} | {ligne['err_pos']:>14.3g} | "
                      f"{ligne['sources_ok']:>6.1f} / {N:<4} | {ligne['temps']:>9.2f}s")
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(lignes[0]))
            w.writeheader()
            w.writerows(lignes)
        print(f"\nRésultats écrits dans {args.csv}")


def figure(args):
    import matplotlib.pyplot as plt
    S = capteurs(args.capteurs)
    N = args.sources[0]
    x = args.rapport[0] if args.rapport else None
    rng = np.random.default_rng(1000 * N)
    p, I = tirer_situation(N, rng, x, args.sep_min)
    y = mesures(S, p, I, args.sigma, args.centrale, rng)
    u0 = rng.uniform(0, 1, (N, 2))
    r = descente_gradient(S, y, u0, np.full(N, 1.0 / N), args.iter, garder_trajectoire=True)
    ep, _ = erreurs(p, I, r['p'], r['I'])
    fig, ax = plt.subplots(figsize=(5.5, 8))
    ax.plot(*S.T, '.', color='gray', ms=4)
    ax.add_patch(plt.Rectangle((0, 0), LX, LY, fill=False, ls='--', color='gray'))
    for i in range(N):
        ax.plot(*r['trajectoire'][:, i].T, '-', lw=1)
    ax.scatter(*p.T, s=40 + 300 * I / I.max(), marker='*', c='k', label='vraies sources', zorder=3)
    ax.plot(*(u0 * ECHELLE).T, 'x', c='C3', label='initialisation')
    ax.plot(*r['p'].T, 'o', mfc='none', c='C0', ms=9, label='fin de la descente')
    ax.set_aspect('equal')
    ax.set_title(f"{N} sources, rapport {x if x else 'simplexe'} : {int((ep < args.tol).sum())}/{N} "
                 f"retrouvées à {args.tol}\n{r['iterations']} itérations")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=8)
    nom = f"descente_orano_{N}sources.png"
    fig.savefig(nom, dpi=150, bbox_inches='tight')
    print(f"Figure écrite dans {nom}")


if __name__ == '__main__':
    pa = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pa.add_argument('--sources', type=int, nargs='+', default=[2, 3, 5, 10], help='nombres de sources N')
    pa.add_argument('--rapport', type=float, nargs='+', default=None,
                    help='rapport x entre la plus forte et la plus faible (défaut : intensités sur le simplexe)')
    pa.add_argument('--tirages', type=int, default=20, help='situations aléatoires par cas')
    pa.add_argument('--sigma', type=float, default=0.0, help='bruit relatif gaussien (0 : sans bruit)')
    pa.add_argument('--centrale', action='store_true', help='ajouter la source centrale (5000) puis la retirer')
    pa.add_argument('--capteurs', type=int, default=64)
    pa.add_argument('--iter', type=int, default=20000, help='budget d’itérations')
    pa.add_argument('--tol', type=float, default=0.1, help='réussite : chaque source à moins de tol')
    pa.add_argument('--sep-min', type=float, default=0.5, help='séparation minimale entre sources tirées')
    pa.add_argument('--gn', action='store_true', help='comparer avec Gauss-Newton')
    pa.add_argument('--csv', type=str, default=None, help='fichier CSV de sortie')
    pa.add_argument('--figure', action='store_true', help='tracer les trajectoires sur un tirage')
    args = pa.parse_args()
    figure(args) if args.figure else experience(args)

"""
Reconstruction des sources par approximation rationnelle (AAA) puis moindres
carrés, à partir de capteurs équirépartis sur un cercle de centre c et de
rayon R. Alternative à `prony_circle`, plus robuste au bruit et à une grosse
source au centre.

On identifie R^2 à C et on note, pour chaque source, z_i = x_i - c,
rho_i = |z_i|, w_i = I_i / (R^2 - rho_i^2) > 0 et m_i = w_i z_i.
Sur le cercle, conj(z) = R^2 / z, et chaque terme du signal se décompose en

    I_i / |z - z_i|^2 = w_i z_i / (z - z_i)          (pôle intérieur : la source)
                      + w_i R^2 / (R^2 - conj(z_i) z) (artefact, pôle extérieur)

Le premier terme ne contient que les fréquences négatives e^{-int}, n >= 1 ;
le second ne contient que la constante et les fréquences positives. En ne
gardant que les fréquences négatives des mesures, on obtient donc

    C(z) = sum_{n >= 1} c_{-n} e^{-int} = sum_{z_i != 0} m_i / (z - z_i),
    c_{-n} = sum_i w_i (z_i / R)^n,

une fonction rationnelle dont les pôles sont les sources hors du centre et
les résidus m_i = w_i z_i (d'où w_i réel > 0 : arg m_i = arg z_i). La source
centrale (z_i = 0) n'apparaît pas dans C ; on la retrouve par
c_0 = sum_i w_i :  I_centre = R^2 (c_0 - sum_{z_i != 0} w_i).

Étapes de `reconstruire` :
    1. FFT des mesures, niveau de bruit sigma, coupure L     -> `transformee_C`
    2. AAA sur C aux capteurs -> pôles et résidus
    3. filtrage physique, intensités initiales              -> `estimation_initiale`
    4. moindres carrés sur les mesures brutes               -> `raffiner`
    5. nettoyage des sources négligeables

Limite fondamentale : seules L fréquences dépassent le bruit, et on ne peut
pas résoudre plus de L/2 sources hors du centre.
"""

import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import least_squares

from ..model import Sources
from ..sensors import CircleSensors


# --------------------------------------------------------------------------
# Modèle direct (en complexe, relatif au centre du cercle)
# --------------------------------------------------------------------------

def modele_direct(z_capteurs: np.ndarray, z_sources: np.ndarray,
                  intensites: np.ndarray) -> np.ndarray:
    """
    Phi_k = sum_i I_i / |z_k - z_i|^2, forme (K,).

    z_capteurs (K,) et z_sources (N,) sont des complexes dans le même repère.
    """
    r2 = np.abs(z_capteurs[:, None] - z_sources[None, :]) ** 2
    return r2 ** -1 @ intensites


def _jacobien(z_capteurs, z_sources, intensites):
    """
    Dérivées de `modele_direct` par rapport à (Re z_i, Im z_i, I_i),
    colonnes rangées comme dans `raffiner` : forme (K, 3N).
    """
    d = z_capteurs[:, None] - z_sources[None, :]    # (K, N)
    inv_r2 = 1.0 / np.abs(d) ** 2
    dx = 2 * intensites * d.real * inv_r2**2
    dy = 2 * intensites * d.imag * inv_r2**2
    return np.stack([dx, dy, inv_r2], axis=-1).reshape(len(z_capteurs), -1)


# --------------------------------------------------------------------------
# 1. Transformée : fréquences négatives des mesures
# --------------------------------------------------------------------------

def transformee_C(phi: np.ndarray, s: float = 4.0):
    """
    Fréquences négatives des mesures, et niveau de bruit.

    C est construit sur TOUTES les fréquences négatives, sans coupure à L :
    tronquer à L une source éloignée du centre (rho/R proche de 1, série lente)
    laisse une queue de l'ordre de s*sigma/(1 - rho/R), qui n'est plus une
    fraction rationnelle à un seul pôle ; AAA la remplaçait alors par deux
    pôles, rejetés ensuite par le filtre physique. Le bruit des hautes
    fréquences est absorbé par la tolérance d'AAA (voir `reconstruire`).

    Paramètres
    ----------
    phi : np.ndarray, forme (K,)
        Mesures brutes des capteurs (dans l'ordre de `CircleSensors.angles`).
    s : float
        Seuil de détection, en multiples du niveau de bruit.

    Renvoie
    -------
    C : np.ndarray complexe, forme (K,)
        C aux capteurs : sum_{n=1}^{K/2-1} c_{-n} e^{-i n t_k}. Le déphasage
        dû à t0 se compense de lui-même (fft puis ifft sur les mêmes angles).
    c0 : float
        Coefficient constant c_0 = sum_i w_i.
    L : int
        Nombre de premières fréquences négatives consécutives au-dessus de
        s*sigma : fréquences utiles, d'où au plus L/2 sources résolubles.
    sigma : float
        Niveau de bruit estimé sur les coefficients de Fourier.
    """
    K = len(phi)
    X = np.fft.fft(phi) / K
    c0 = float(X[0].real)
    n = np.arange(1, K // 2)
    X_neg = X[K - n]                                    # X_{-1}, ..., X_{-(K/2-1)}

    # Niveau de bruit : médiane des hautes fréquences, n dans [K/4, K/2)
    sigma = float(np.median(np.abs(X_neg[n >= K // 4])))

    # Coupure : premières fréquences consécutives au-dessus du bruit
    au_dessus = np.abs(X_neg) > s * sigma
    L = int(np.argmin(au_dessus)) if not au_dessus.all() else len(au_dessus)

    Y = np.zeros(K, dtype=complex)
    Y[K - n] = X_neg
    C = K * np.fft.ifft(Y)
    return C, c0, L, sigma


# --------------------------------------------------------------------------
# 2. AAA (Nakatsukasa, Sète, Trefethen 2018)
# --------------------------------------------------------------------------

def _aaa_maison(Z, F, rtol, max_termes=100):
    """
    Algorithme AAA, utilisé si SciPy < 1.15 (pas de `scipy.interpolate.AAA`).

    Forme barycentrique r(z) = sum_j w_j f_j/(z - z_j) / sum_j w_j/(z - z_j) ;
    à chaque étape on ajoute le point de plus grande erreur, et les poids sont
    le dernier vecteur singulier droit de la matrice de Loewner.

    Renvoie (pôles, résidus).
    """
    Z = np.asarray(Z, dtype=complex)
    F = np.asarray(F, dtype=complex)
    libre = np.ones(len(Z), dtype=bool)
    R = np.full_like(F, F.mean())
    tol = rtol * np.abs(F).max()
    zj, fj = np.empty(0, complex), np.empty(0, complex)

    for _ in range(min(max_termes, len(Z) - 1)):
        j = np.argmax(np.where(libre, np.abs(F - R), -1.0))
        zj, fj = np.append(zj, Z[j]), np.append(fj, F[j])
        libre[j] = False
        Cauchy = 1.0 / (Z[libre, None] - zj[None, :])
        Loewner = (F[libre, None] - fj[None, :]) * Cauchy
        wj = np.linalg.svd(Loewner)[2][-1].conj()
        R = F.copy()
        R[libre] = (Cauchy @ (wj * fj)) / (Cauchy @ wj)
        if np.abs(F - R).max() <= tol:
            break

    # Pôles : valeurs propres généralisées de la matrice en flèche (m+1)x(m+1)
    m = len(zj)
    E = np.zeros((m + 1, m + 1), dtype=complex)
    E[0, 1:] = wj
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zj)
    B = np.eye(m + 1, dtype=complex)
    B[0, 0] = 0.0
    poles = eigvals(E, B)
    poles = poles[np.isfinite(poles)]

    # Résidus : n(p) / d'(p)
    Cp = 1.0 / (poles[:, None] - zj[None, :])
    residus = (Cp @ (wj * fj)) / -(Cp**2 @ wj)
    return poles, residus


def poles_et_residus(z: np.ndarray, C: np.ndarray, rtol: float):
    """
    Approximation rationnelle AAA de C sur les points z ; renvoie (pôles, résidus).

    Utilise `scipy.interpolate.AAA` (SciPy >= 1.15), sinon `_aaa_maison`.
    """
    try:
        from scipy.interpolate import AAA
    except ImportError:
        return _aaa_maison(z, C, rtol)
    r = AAA(z, C, rtol=rtol)
    return r.poles(), r.residues()


# --------------------------------------------------------------------------
# 3. Estimation initiale
# --------------------------------------------------------------------------

def estimation_initiale(poles, residus, c0, R, sigma, L, s=4.0,
                        marge_poles=0.98, tol_angle=0.5):
    """
    Sources initiales à partir des pôles et résidus de C.

    Un pôle p de résidu m est gardé si |p| < marge_poles * R, si w = m / p vérifie
    Re w > 0 et |arg w| < tol_angle (w doit être réel positif), et si
    |m| > s * sigma. On en garde au plus L // 2 (les plus gros résidus).
    La source centrale est ajoutée en dernier, en z = 0.

    Renvoie
    -------
    z : np.ndarray complexe (M + 1,)
        Positions relatives au centre ; la dernière est la source centrale.
    I : np.ndarray (M + 1,)
        Intensités initiales.
    """
    poles = np.asarray(poles, dtype=complex)
    residus = np.asarray(residus, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = residus / poles
    garde = ((np.abs(poles) < marge_poles * R)
             & (w.real > 0) & (np.abs(np.angle(w)) < tol_angle)
             & (np.abs(residus) > s * sigma))
    p, m, w = poles[garde], residus[garde], w[garde]

    ordre = np.argsort(-np.abs(m))[: L // 2]
    p, w = p[ordre], w[ordre].real

    I = w * (R**2 - np.abs(p) ** 2)
    I_centre = max(R**2 * (c0 - w.sum()), 0.0)
    return np.append(p, 0.0), np.append(I, I_centre)


# --------------------------------------------------------------------------
# 4. Raffinement par moindres carrés
# --------------------------------------------------------------------------

def raffiner(phi, z_capteurs, z0, I0, R, marge_bornes=0.99):
    """
    Moindres carrés non linéaires sur les mesures brutes phi.

    Inconnues (Re z_i, Im z_i, I_i) pour chaque source, centre compris.
    Bornes : |Re z_i|, |Im z_i| <= marge_bornes * R et I_i >= 0. Le point de
    départ est projeté strictement à l'intérieur des bornes.

    Renvoie (z, I, résultat de `least_squares`).
    """
    N = len(z0)
    b = marge_bornes * R
    lo = np.tile([-b, -b, 0.0], N)
    hi = np.tile([b, b, np.inf], N)
    x0 = np.stack([z0.real, z0.imag, I0], axis=-1).ravel()
    delta = 1e-6 * R
    x0 = np.clip(x0, lo + delta, hi - delta)

    def decoupe(x):
        x = x.reshape(N, 3)
        return x[:, 0] + 1j * x[:, 1], x[:, 2]

    res = least_squares(
        lambda x: modele_direct(z_capteurs, *decoupe(x)) - phi,
        x0, jac=lambda x: _jacobien(z_capteurs, *decoupe(x)),
        bounds=(lo, hi), x_scale="jac", ftol=1e-14, xtol=1e-14, gtol=1e-14,
    )
    z, I = decoupe(res.x)
    return z, I, res


# --------------------------------------------------------------------------
# 5. Recherche itérative sur le résidu
# --------------------------------------------------------------------------

def grille_candidats(R, marge_poles=0.98, pas=0.02):
    """Points z (complexes, relatifs au centre) d'une grille de pas `pas * R` dans le disque |z| < marge_poles * R."""
    t = np.arange(-1, 1 + pas / 2, pas) * R
    g = (t[None, :] + 1j * t[:, None]).ravel()
    return g[np.abs(g) < marge_poles * R]


def meilleur_candidat_residu(residu, G, z_grille):
    """
    Filtre adapté positif (à la manière de CLEAN) : pour chaque point de grille
    x, l'intensité I >= 0 qui explique le mieux le résidu par une seule source
    en x est I = <r, g_x> / <g_x, g_x>, avec g_x(k) = 1/|z_k - x|^2, et elle
    réduit la somme des carrés de <r, g_x>^2 / <g_x, g_x>.

    Contrairement à AAA, ce critère ne voit que des sources positives : il
    n'est pas trompé par les « dipôles » (+ et -) que laisse dans le résidu une
    source légèrement mal placée.

    G : matrice (P, K) des g_x pour les P points de grille.
    Renvoie (z, I, gain) du meilleur point, ou None si aucun I > 0.
    """
    proj = G @ residu
    norme2 = np.einsum("pk,pk->p", G, G)
    gain = np.where(proj > 0, proj**2 / norme2, 0.0)
    j = int(np.argmax(gain))
    if gain[j] <= 0:
        return None
    return z_grille[j], proj[j] / norme2[j], gain[j]


def chasse_residu(phi, z_capteurs, z, I, R, sigma, iterations,
                  marge_poles=0.98, marge_bornes=0.99, penalite=None):
    """
    Ajoute les sources une à une, tant que le résidu en contient.

    À chaque itération : résidu = phi - modèle actuel, meilleur candidat du
    résidu par filtre adapté (`meilleur_candidat_residu`), puis
    raffinement de TOUTES les sources ensemble. Le candidat n'est gardé que si
    la somme des carrés des résidus baisse de plus de 3 * penalite * sigma_m^2
    (3 inconnues par source ; penalite = ln K par défaut, critère BIC), où
    sigma_m^2 = K sigma^2 / ln 2 est la variance du bruit par capteur déduite
    de sigma (médiane de |X| pour un bruit gaussien complexe). Sinon, ou s'il
    n'y a plus de candidat, on s'arrête.

    La source centrale reste en dernière position de z.

    Renvoie (z, I, nombre de sources ajoutées).
    """
    K = len(phi)
    penalite = np.log(K) if penalite is None else penalite
    sigma_m2 = K * sigma**2 / np.log(2)
    cout = np.sum((modele_direct(z_capteurs, z, I) - phi) ** 2)
    ajoutees = 0
    if iterations <= 0:
        return z, I, ajoutees

    z_grille = grille_candidats(R, marge_poles)
    G = 1.0 / np.abs(z_capteurs[None, :] - z_grille[:, None]) ** 2

    for _ in range(iterations):
        if 3 * (len(z) + 1) >= K:          # plus d'inconnues que de mesures
            break
        candidat = meilleur_candidat_residu(phi - modele_direct(z_capteurs, z, I),
                                            G, z_grille)
        if candidat is None:
            break
        p, Ip, _ = candidat

        z_essai = np.concatenate([z[:-1], [p], z[-1:]])
        I_essai = np.concatenate([I[:-1], [Ip], I[-1:]])
        z2, I2, res = raffiner(phi, z_capteurs, z_essai, I_essai, R, marge_bornes)
        cout2 = 2 * res.cost
        if cout - cout2 <= 3 * penalite * sigma_m2:
            break
        z, I, cout = z2, I2, cout2
        ajoutees += 1
    return z, I, ajoutees


# --------------------------------------------------------------------------
# Méthode complète
# --------------------------------------------------------------------------

def reconstruire(phi, sensors: CircleSensors, s=4.0, marge_poles=0.98,
                 tol_angle=0.5, marge_bornes=0.99, seuil_final=0.05,
                 seuil_nul=1e-6, iterations_residu=0, penalite=None):
    """
    Reconstruit les sources (centre compris) à partir des mesures d'un cercle.

    Paramètres
    ----------
    phi : np.ndarray, forme (K,)
        Mesures brutes (sans retirer la source centrale).
    sensors : CircleSensors
        Capteurs équirépartis ayant produit `phi`.
    s : float
        Seuil de détection en multiples du bruit (coupure L, résidus, AAA).
    marge_poles : float
        Pôles gardés si |p| < marge_poles * R.
    tol_angle : float
        Tolérance (radians) sur arg(w), qui doit être nul.
    marge_bornes : float
        Bornes du raffinement : |Re z|, |Im z| <= marge_bornes * R.
    seuil_final : float
        On écarte les sources d'intensité < seuil_final * (2e plus grande
        intensité) ; la 2e et non la 1re, qui est en général la source centrale.
    seuil_nul : float
        On écarte aussi les sources d'intensité < seuil_nul * (plus grande
        intensité) : sans cela, une source centrale absente (intensité ~ 0)
        serait gardée quand il n'y a qu'une seule autre source.
    iterations_residu : int
        Nombre maximal de sources ajoutées une à une par `chasse_residu`
        après la passe AAA principale (0 = pas de recherche sur le résidu,
        et au plus L/2 sources hors centre).
    penalite : float ou None
        Pénalité par inconnue du critère d'arrêt de `chasse_residu`
        (None = ln K, critère BIC). Plus grande = moins de sources ajoutées.

    Les sources finales hors du cercle des capteurs (|z| >= R) sont écartées.

    Renvoie
    -------
    Sources
        Sources reconstruites (positions absolues).
    dict
        Diagnostic : L, sigma, c0, pôles AAA, estimation initiale,
        ajoutees_residu (sources ajoutées par la recherche sur le résidu),
        n_hors_centre (sources trouvées hors de la source centrale), coût final.
    """
    phi = np.asarray(phi, dtype=float)
    R, c = sensors.radius, sensors.center
    z_capteurs = R * np.exp(1j * sensors.angles)

    C, c0, L, sigma = transformee_C(phi, s)
    diag = {"L": L, "sigma": sigma, "c0": c0, "poles_aaa": np.empty(0, complex)}

    if L == 0:
        poles = residus = np.empty(0, complex)
    else:
        # Bruit sur C en chaque capteur : somme de K/2 coefficients de niveau sigma
        rtol = max(s * sigma * np.sqrt(len(phi) / 2) / np.abs(C).max(), 1e-13)
        poles, residus = poles_et_residus(z_capteurs, C, rtol)
        diag["poles_aaa"] = poles

    z0, I0 = estimation_initiale(poles, residus, c0, R, sigma, L, s,
                                 marge_poles, tol_angle)
    diag["initial"] = (z0, I0)
    z, I, res = raffiner(phi, z_capteurs, z0, I0, R, marge_bornes)

    z, I, diag["ajoutees_residu"] = chasse_residu(
        phi, z_capteurs, z, I, R, sigma, iterations_residu, marge_poles,
        marge_bornes, penalite)
    diag["cout"] = 0.5 * float(np.sum((modele_direct(z_capteurs, z, I) - phi) ** 2))

    # Nettoyage des sources négligeables ou hors du cercle des capteurs
    tri = np.sort(I)[::-1]
    ref = tri[1] if len(tri) > 1 else tri[0]
    garde = ((I >= seuil_final * ref) & (I >= seuil_nul * tri[0]) & (I > 0)
             & (np.abs(z) < R))
    est_centre = np.arange(len(I)) == len(I) - 1
    diag["n_hors_centre"] = int(np.sum(garde & ~est_centre))

    z, I = z[garde], I[garde]
    positions = c + np.stack([z.real, z.imag], axis=-1)
    return Sources(positions, I), diag


def aaa_circle(y, sensors: CircleSensors, **params) -> Sources:
    """
    Même interface que `prony_circle` : (mesures, capteurs) -> Sources.
    Les `params` sont ceux de `reconstruire`.
    """
    return reconstruire(y, sensors, **params)[0]

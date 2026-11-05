"""
Vérification des hypothèses du théorème de Duval & Peyré (« Exact support
recovery for sparse spikes deconvolution », FoCM 2015) pour le BLASSO positif.

Pour des sources z_1, ..., z_K ∈ D :

    Γ = [φ(z_1) … φ(z_K) | ∂₁φ(z_1) … ∂₁φ(z_K) | ∂₂φ(z_1) … ∂₂φ(z_K)] ∈ R^{M×3K}

    p_V = argmin ||p||  s.c.  Γ^T p = (1, …, 1, 0, …, 0)   (K uns, 2K zéros)
        = pinv(Γ^T) (1_K, 0_{2K})
    η_V(z) = Σ_i p_V,i G(x_i, z)          (pré-certificat « vanishing derivatives »)

Par construction η_V(z_k) = 1 et ∇η_V(z_k) = 0. La condition de source non
dégénérée est :

    (H1) Γ est de rang plein 3K (il faut M ≥ 3K) ;
    (H2) η_V(z) < 1 pour tout z ∈ D \\ {z_1, …, z_K} (bord de D compris) ;
    (H3) ∇²η_V(z_k) est définie négative pour tout k.

Sous cette condition (et z_k à l'intérieur de D), il existe λ₀, c₀ > 0 tels
que pour λ ≤ λ₀ et ||w|| / λ ≤ c₀, le BLASSO a exactement K Diracs, à
distance O(||w|| + λ) des vraies sources (intensités et positions).

En pratique (H2) est testée hors de petites boules de rayon ρ autour des z_k
(η_V → 1 en z_k) ; à l'intérieur des boules, c'est (H3) qui garantit le
maximum strict, et on contrôle en plus qu'aucun nœud de grille n'y dépasse 1.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .common import grid_local_maxima, make_grid, maximize_on_domain
from .kernel import Setup, resolve

TOL = 1e-6   # η_V < 1 « strictement » : marge > TOL


def gamma_matrix(z, setup: Setup | None = None) -> np.ndarray:
    """Γ = [φ(z_k) | ∂₁φ(z_k) | ∂₂φ(z_k)], forme (M, 3K)."""
    setup = resolve(setup)
    z = np.asarray(z, float).reshape(-1, 2)
    g = setup.grad(z)                                    # (K, M, 2)
    return np.hstack([setup.phi(z).T, g[:, :, 0].T, g[:, :, 1].T])


def precertificate(z, setup: Setup | None = None):
    """
    p_V = pinv(Γ^T) (1_K, 0_2K) (norme minimale), forme (M,), et Γ.
    """
    gamma = gamma_matrix(z, setup)
    K = gamma.shape[1] // 3
    u = np.zeros(3 * K)
    u[:K] = 1.0
    return np.linalg.pinv(gamma.T) @ u, gamma


def eta_V(p, Z, setup: Setup | None = None) -> np.ndarray:
    """η_V(z) = Σ_i p_i G(x_i, z), Z de forme (..., 2)."""
    return resolve(setup).phi(Z) @ p


def eta_V_grad(p, Z, setup: Setup | None = None) -> np.ndarray:
    """∇η_V, forme (..., 2)."""
    return np.einsum("i,...ia->...a", p, resolve(setup).grad(Z))


def eta_V_hess(p, Z, setup: Setup | None = None) -> np.ndarray:
    """∇²η_V, forme (..., 2, 2)."""
    return np.einsum("i,...iab->...ab", p, resolve(setup).hess(Z))


def _dist_to(points, z):
    """Distance de chaque point (..., 2) au z_k le plus proche."""
    return np.min(np.linalg.norm(np.asarray(points)[..., None, :] - z, axis=-1), axis=-1)


def _boundary_max(f, grad, box, z, rho, step, n_starts=3):
    """
    Max de f sur le bord de D hors des boules : chaque côté est échantillonné
    au pas step / 4, puis L-BFGS-B le long du côté depuis les meilleurs points.
    Renvoie (valeur, point).
    """
    best = (-np.inf, np.full(2, np.nan))
    for a in range(2):                       # coordonnée fixée
        b = 1 - a                            # coordonnée libre
        for side in (box.lo[a], box.hi[a]):
            t = np.linspace(box.lo[b], box.hi[b], int(np.ceil(box.size[b] / (step / 4))) + 1)
            pts = np.empty((len(t), 2))
            pts[:, a], pts[:, b] = side, t
            vals = f(pts)
            vals[_dist_to(pts, z) < rho] = -np.inf
            order = [i for i in np.argsort(-vals)[:n_starts] if np.isfinite(vals[i])]
            for i in order:
                if vals[i] > best[0]:
                    best = (float(vals[i]), pts[i].copy())
                fixed = lambda s, a=a, b=b, side=side: np.array(
                    [side, s[0]] if a == 0 else [s[0], side])
                res = minimize(lambda s: -float(f(fixed(s))), [t[i]],
                               jac=lambda s: -np.array([grad(fixed(s))[b]]),
                               method="L-BFGS-B", bounds=[(box.lo[b], box.hi[b])])
                x = fixed(res.x)
                if _dist_to(x, z) >= rho and -res.fun > best[0]:
                    best = (float(-res.fun), x)
    return best


def check_hypotheses(z_sources, lam: float | None = None, sigma=None,
                     grid_step: float | None = None, exclusion_radius: float | None = None,
                     verbose: bool = True, setup: Setup | None = None, plot: bool | None = None,
                     save_path=None, n_refine: int = 10, estimates=None) -> dict:
    """
    Vérifie la condition de source non dégénérée de Duval-Peyré pour les
    sources z_sources (K, 2) (vraies positions en simulation, positions
    estimées sur données réelles).

    Paramètres
    ----------
    z_sources : (K, 2)
    lam, sigma : facultatifs. Si les deux sont donnés, on rapporte le rapport
        bruit / λ, ||w|| / λ ≈ σ √M / λ, et une borne de la perturbation du
        certificat par le bruit, ||Φ||·||w|| / λ avec ||Φ|| = max_D ||φ(z)||
        (indicateur heuristique du régime « petit bruit », pas une hypothèse
        du théorème : les constantes λ₀, c₀ ne sont pas calculées).
    grid_step : pas de la grille de (H2) ; défaut min(taille de D) / 200
    exclusion_radius : rayon ρ des boules exclues ; défaut
        min(0.25 × séparation minimale, 0.01 × diagonale de D)
    verbose : affiche le résumé (et trace la carte si plot n'est pas False)
    plot : trace la carte de η_V (défaut : = verbose)
    save_path : enregistre la figure (PNG) si donné
    n_refine : nombre de maxima locaux de la grille affinés par L-BFGS-B
    estimates : positions estimées à superposer sur la carte (facultatif)

    Renvoie un dict
    ---------------
    H1, H2, H3, non_degenerate (= H1 and H2 and H3), interior,
    K, M, rank, n_cols (= 3K), cond, cond_normalized, singular_values,
    interp_error (max |η_V(z_k) - 1|), grad_error (max ||∇η_V(z_k)||, relatif),
    margin (= 1 - max η_V hors boules), argmax, boundary_max, boundary_argmax,
    ball_excess, exclusion_radius, eigenvalues (K, 2), hessians (K, 2, 2),
    min_dist_boundary, p (pré-certificat), noise_over_lambda,
    noise_certificate_bound, figure (si tracée).
    """
    setup = resolve(setup)
    box = setup.domain
    z = np.asarray(z_sources, float).reshape(-1, 2)
    K, M = len(z), setup.M
    diag = float(np.linalg.norm(box.size))
    if grid_step is None:
        grid_step = float(min(box.size)) / 200
    if exclusion_radius is None:
        sep = np.inf
        if K > 1:
            d = np.linalg.norm(z[:, None] - z[None], axis=-1)
            sep = float(d[np.triu_indices(K, 1)].min())
        exclusion_radius = min(0.25 * sep, 0.01 * diag)
    rho = float(exclusion_radius)
    warnings = []

    # 5. Sources à l'intérieur de D (le théorème suppose z_k dans l'intérieur)
    dist_bd = np.minimum(z - box.lo, box.hi - z).min(axis=1)
    min_dist_boundary = float(dist_bd.min())
    interior = bool(min_dist_boundary > 1e-9 * diag)
    if not interior:
        warnings.append("au moins une source est sur le bord de D ou à l'extérieur")

    # 1. (H1) rang et conditionnement de Γ
    p, gamma = precertificate(z, setup)
    sv = np.linalg.svd(gamma, compute_uv=False)
    rank = int(np.sum(sv > max(gamma.shape) * np.finfo(float).eps * sv[0]))
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    gn = gamma / np.linalg.norm(gamma, axis=0)
    svn = np.linalg.svd(gn, compute_uv=False)
    cond_n = float(svn[0] / svn[-1]) if svn[-1] > 0 else np.inf
    n_cols = 3 * K
    if M < n_cols:
        warnings.append(f"M = {M} < 3K = {n_cols} : (H1) ne peut pas tenir")
    H1 = bool(M >= n_cols and rank == n_cols)

    # 2. Pré-certificat : contraintes η_V(z_k) = 1, ∇η_V(z_k) = 0
    f = lambda x: eta_V(p, x, setup)
    g = lambda x: eta_V_grad(p, x, setup)
    interp_error = float(np.max(np.abs(f(z) - 1)))
    scale = np.max(np.sum(np.abs(p)[None, :, None] * np.abs(setup.grad(z)), axis=1))
    grad_error = float(np.max(np.linalg.norm(g(z), axis=-1)) / max(scale, 1e-300))

    # 3. (H2) grille fine hors des boules, raffinement local, bord de D
    grid = make_grid(box, grid_step, setup)
    vals = (p @ grid.G).reshape(grid.shape)
    dist = _dist_to(grid.points, z)
    out = np.where(dist >= rho, vals, -np.inf)
    i = np.unravel_index(np.argmax(out), out.shape)
    best_val, best_pt = float(out[i]), grid.points[i].copy()
    starts = [grid.points[tuple(j)] for j in grid_local_maxima(out, n_refine)]
    for v, x in maximize_on_domain(lambda x: float(f(x)), g, box, starts):
        if _dist_to(x, z) >= rho and v > best_val:
            best_val, best_pt = v, x
    b_val, b_pt = _boundary_max(f, g, box, z, rho, grid_step)
    if b_val > best_val:
        best_val, best_pt = b_val, b_pt
    margin = 1.0 - best_val
    inside = (dist < rho) & (dist > 1e-12)
    ball_excess = float(np.max(vals[inside] - 1)) if inside.any() else -np.inf
    H2 = bool(margin > TOL and ball_excess <= TOL)

    # 4. (H3) hessiennes de η_V aux sources
    hessians = eta_V_hess(p, z, setup)
    eig = np.linalg.eigvalsh(hessians)
    H3 = bool(np.all(eig < 0))

    res = {
        "H1": H1, "H2": H2, "H3": H3, "non_degenerate": H1 and H2 and H3,
        "interior": interior, "K": K, "M": M, "rank": rank, "n_cols": n_cols,
        "cond": cond, "cond_normalized": cond_n, "singular_values": sv,
        "interp_error": interp_error, "grad_error": grad_error,
        "margin": margin, "argmax": np.asarray(best_pt), "boundary_max": b_val,
        "boundary_argmax": b_pt, "ball_excess": ball_excess, "exclusion_radius": rho,
        "eigenvalues": eig, "hessians": hessians, "min_dist_boundary": min_dist_boundary,
        "p": p, "z": z, "warnings": warnings,
        "noise_over_lambda": np.nan, "noise_certificate_bound": np.nan,
    }

    # Régime de bruit (indicatif)
    if lam is not None and sigma is not None:
        noise = float(np.sqrt(np.sum(np.broadcast_to(np.asarray(sigma, float) ** 2, (M,)))))
        op_norm = float(np.max(np.linalg.norm(grid.G, axis=0)))
        res["noise_over_lambda"] = noise / lam
        res["noise_certificate_bound"] = op_norm * noise / lam

    if verbose:
        print(format_report(res))
    if plot if plot is not None else verbose:
        import matplotlib.pyplot as plt
        from .plotting import certificate_map, place_legend

        fig, ax = plt.subplots(figsize=(6.4, 7.2))
        tag = "VÉRIFIÉE" if res["non_degenerate"] else "NON vérifiée"
        certificate_map(ax, grid.xs, grid.ys, vals, setup.sensors, z, box,
                        title=f"Pré-certificat η_V, K = {K}, h = {setup.h:g}"
                              f"{', noyau normalisé' if setup.normalize else ''}\n"
                              f"condition {tag} (marge = {margin:.3g})",
                        estimates=estimates)
        ax.scatter(*res["argmax"], marker="D", s=40, color="#eda100", zorder=6,
                   label="max hors boules")
        place_legend(ax)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=130)
        res["figure"] = fig
    return res


def format_report(res: dict) -> str:
    """Résumé lisible du dict renvoyé par ``check_hypotheses``."""
    ok = {True: "VRAIE", False: "FAUSSE"}
    L = [f"Condition de source non dégénérée : K = {res['K']} sources, M = {res['M']} capteurs"]
    L.append(f"  (H1) {ok[res['H1']]} : rang(Γ) = {res['rank']} / {res['n_cols']}, "
             f"cond(Γ) = {res['cond']:.3e} (colonnes normalisées : {res['cond_normalized']:.3e})")
    L.append(f"  Pré-certificat : max |η_V(z_k) - 1| = {res['interp_error']:.1e}, "
             f"max ||∇η_V(z_k)|| (relatif) = {res['grad_error']:.1e}")
    L.append(f"  (H2) {ok[res['H2']]} : marge 1 - max η_V = {res['margin']:.4g} "
             f"(atteint en {np.round(res['argmax'], 3)}, boules de rayon "
             f"{res['exclusion_radius']:.3g})")
    L.append(f"       max sur le bord de D : {res['boundary_max']:.4g} "
             f"en {np.round(res['boundary_argmax'], 3)}")
    if res["ball_excess"] > TOL:
        L.append(f"       η_V dépasse 1 de {res['ball_excess']:.3g} tout près d'une source")
    L.append(f"  (H3) {ok[res['H3']]} : valeurs propres de ∇²η_V(z_k)")
    for zk, ev in zip(res["z"], res["eigenvalues"]):
        L.append(f"       z = {np.round(zk, 3)} : {np.array2string(ev, precision=4)}")
    L.append(f"  Sources à l'intérieur de D : {'oui' if res['interior'] else 'NON'} "
             f"(distance min au bord {res['min_dist_boundary']:.3g})")
    if np.isfinite(res["noise_over_lambda"]):
        L.append(f"  Bruit : ||w|| / λ ≈ {res['noise_over_lambda']:.3g}, perturbation du "
                 f"certificat ≤ {res['noise_certificate_bound']:.3g} (indicatif)")
    for w in res["warnings"]:
        L.append(f"  ATTENTION : {w}")
    L.append("  => condition " + ("VÉRIFIÉE : le théorème de Duval-Peyré s'applique."
                                  if res["non_degenerate"] else
                                  "NON vérifiée : le théorème ne s'applique pas."))
    return "\n".join(L)


def hypotheses_after_fit(z_hat, I_hat=None, **kwargs) -> dict:
    """
    Diagnostic de fiabilité d'une reconstruction : ``check_hypotheses`` sur
    les positions ESTIMÉES (les sources d'intensité nulle sont écartées si
    I_hat est donné). Si la condition tient en ẑ, le BLASSO est stable autour
    de la solution trouvée ; sinon, le support estimé est à prendre avec
    précaution (sources trop proches, trop près du bord, mal vues).
    Ajoute la clé ``reliable`` (= non_degenerate and interior).
    """
    z_hat = np.asarray(z_hat, float).reshape(-1, 2)
    if I_hat is not None:
        z_hat = z_hat[np.asarray(I_hat, float).reshape(-1) > 0]
    res = check_hypotheses(z_hat, **kwargs)
    res["reliable"] = bool(res["non_degenerate"] and res["interior"])
    return res

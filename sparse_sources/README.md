# sparse_sources — LASSO, BLASSO (Frank-Wolfe glissant) et hypothèses de Duval–Peyré

Localisation de sources ponctuelles positives s = Σ I_k δ_{z_k} (z_k ∈ D ⊂ R²)
à partir de M capteurs, avec le noyau du projet

    G(x_i, z) = 1 / (‖x_i − z‖² + h²),   m = Σ_k I_k φ(z_k) + w,   φ(z) = (G(x_i, z))_i

(h = hauteur des capteurs au-dessus du plan des sources ; h = 0 dans le projet,
c'est le paramètre `eps` de `config/sensors.yaml`). Domaine et capteurs sont lus
dans `config/` : D = [0, 10] × [0, 20], 64 capteurs sur le cercle circonscrit.
Aucun fichier existant n'a été modifié ; le noyau et ses dérivées viennent de
`particle_sources.duval_peyre.kernel`.

## Organisation

| Fichier | Contenu |
|---|---|
| `kernel.py` | `Setup` (capteurs, h, D, noyau normalisé ou non), `make_setup`, φ, ∇φ, ∇²φ, `check_derivatives` |
| `common.py` | `forward`, `certificate`, `max_certificate`, `debias`, `diagnostics`, `match_sources`, grille, descente locale |
| `lasso.py` | `build_grid`, `solve_lasso` (FISTA / sklearn / ensemble actif), `merge_peaks`, `refine`, `lasso_pipeline` |
| `blasso.py` | `sliding_frank_wolfe` |
| `hypotheses.py` | `check_hypotheses`, `hypotheses_after_fit` |
| `ml.py` | `greedy_ml` (maximum de vraisemblance glouton, K par BIC), `cramer_rao`, `relative_noise_std` |
| `ml3d.py` | `greedy_ml_nd`, `cramer_rao_nd` : la même méthode en dimension quelconque (sources en 3D) |
| `tester_ml.py` | tester ML + BIC sur un tirage (2D ou 3D) : sources vraies / estimées, CRB, χ², figure |
| `plotting.py` | style des figures |
| `experiments.py` | expériences → `results/` (PNG + CSV) |
| `experiments_1r2.py` | étude du noyau 1/r² → `results/r2_*` |
| `experiments_projet.py` | scénario du projet (source centrale) → `results/projet_*` |
| `experiments_ml.py` | études de ML + BIC (σ mal connu, détection, résolution, capteurs) → `results/ml_*` |
| `experiments_cylindre.py` | capteurs sur la paroi d'un cylindre (`make_cylinder_setup`), K connu → `results/cylindre*` |
| `experiments_ml3d.py` | sources en 3D → `results/ml3d*` |
| `RAPPORT_ML_BIC.md` | rapport complet sur le maximum de vraisemblance + BIC |
| `tests/` | pytest : `pytest sparse_sources/tests` |

## Appels

Toutes les fonctions prennent un argument facultatif `setup=` ; sans lui, elles
utilisent la géométrie de `config/` (h = 0, noyau brut).

```python
import numpy as np
from sparse_sources import (make_setup, forward, build_grid, lasso_pipeline,
                            sliding_frank_wolfe, check_hypotheses,
                            hypotheses_after_fit, match_sources)

S = make_setup(normalize=True)          # géométrie du projet, BLASSO pondéré (voir plus bas)
# make_setup(h=20.0)                    # capteurs à 20 au-dessus du plan
# make_setup(n_sensors=128, radius_factor=2.0)

# Données simulées : bruit additif σ
z_true, I_true = np.array([[3.0, 5.0], [7.0, 15.0]]), np.array([15.0, 8.0])
m0 = forward(z_true, I_true, S.with_normalization(False))   # mesures physiques
sigma = 1e-3 * np.sqrt(np.mean(m0**2))
m = m0 + sigma * np.random.default_rng(0).standard_normal(S.M)
lam = 2 * sigma * np.sqrt(S.M)                               # λ = c σ √M

# 1. Hypothèses (vraies positions en simulation)
res = check_hypotheses(z_true, lam=lam, sigma=sigma, setup=S)   # résumé + carte de η_V
res["non_degenerate"], res["margin"], res["margin_relative"], res["eigenvalues"]

# 2. LASSO positif sur grille
grid = build_grid(step=0.25, setup=S)                        # G : M × P
out = lasso_pipeline(m, lam, grid=grid, setup=S, sigma=sigma, verbose=True)
#   method="fista" (défaut) | "sklearn" | "active_set" ; refine_positions=True/False
out["z"], out["I"], out["eta_max"], out["diagnostics"]

# 3. Frank-Wolfe glissant (départ 0, ou z_init/I_init = sortie du LASSO)
fw = sliding_frank_wolfe(m, lam, setup=S, grid=grid, sigma=sigma, verbose=True)
z_hat = fw["z_debiased"]
I_hat = S.physical_intensities(z_hat, fw["I_debiased"])     # intensités physiques
fw["history"]          # [{iter, K, eta_max, objective}, ...]

# 4. Scénario du projet (source centrale retirée, bruit relatif) : maximum de
#    vraisemblance glouton, bruit blanchi, nombre de sources choisi par BIC
from sparse_sources import greedy_ml, relative_noise_std, cramer_rao
# y_brut : mesures AVEC la source centrale ; m = y_brut - contribution centrale
s = relative_noise_std(y_brut, sigma)           # écart type du bruit par capteur
ml = greedy_ml(m, s, setup=S, grid=grid)         # ml["z"], ml["I"], ml["K"], ml["chi2"]
cramer_rao(z_true, I_true, s)["pos_std"]         # erreur minimale possible (simulation)
# en 3D : from sparse_sources.ml3d import greedy_ml_nd ; greedy_ml_nd(m, s, capteurs, cube)

# 5. Fiabilité a posteriori : hypothèses aux positions estimées
hypotheses_after_fit(z_hat, I_hat, setup=S)["reliable"]
match_sources(z_true, I_true, z_hat, I_hat)                  # erreurs
```

Formules principales (détaillées dans les docstrings) :

- certificat : η(z) = (1/λ) Σ_i r_i G(x_i, z), r = m − A(s) ; s est solution du
  BLASSO ⇔ η ≤ 1 sur D et η(z_k) = 1 ;
- LASSO : min_{c ≥ 0} ½‖m − Gc‖² + λ‖c‖₁ ; pour sklearn, α = λ / M ;
- raffinement / glissement : min_{I ≥ 0, z ∈ D} ½‖m − Σ I_k φ(z_k)‖² + λ Σ I_k
  (L-BFGS-B, gradient analytique) ;
- débiaisage : NNLS à positions fixées ;
- diagnostics : ‖r‖² comparé à Mσ² (σ scalaire ou vecteur par capteur) ;
- hypothèses : Γ = [φ(z_k) | ∂₁φ(z_k) | ∂₂φ(z_k)] ∈ R^{M×3K},
  p_V = pinv(Γᵀ)(1_K, 0_{2K}), η_V = Σ p_V,i G(x_i, ·) ;
  (H1) rang Γ = 3K, (H2) η_V < 1 hors des z_k (bord compris), (H3) ∇²η_V(z_k) ≺ 0.

## Noyau normalisé (ajout par rapport à la demande)

`make_setup(normalize=True)` remplace φ par ψ = φ/‖φ‖. Le BLASSO avec ψ est
le BLASSO pondéré min ½‖m − A(s)‖² + λ Σ_k I_k ‖φ(z_k)‖ : les intensités
manipulées sont Ĩ_k = I_k ‖φ(z_k)‖ (`physical_intensities` /
`normalized_intensities` pour convertir), les mesures m sont inchangées.
Toute la chaîne (LASSO, Frank-Wolfe, hypothèses, appliquées au noyau ψ)
fonctionne à l'identique. C'est ce qui rend la méthode utilisable avec la
géométrie du projet (voir Résultats).

## Lancer

```bash
pytest sparse_sources/tests                  # 86 tests, ~70 s
python sparse_sources/tester_ml.py --comparer  # un tirage : ML + BIC, Prony, Frank-Wolfe
python sparse_sources/experiments.py         # expériences générales (~6 min) → sparse_sources/results/
python sparse_sources/experiments_1r2.py     # étude du noyau 1/r² (~6,5 min) → results/r2_*
python sparse_sources/experiments_projet.py  # scénario du projet, Cramér-Rao → results/projet_*
python sparse_sources/experiments_ml.py      # études de ML + BIC (~4 min) → results/ml_*
python sparse_sources/experiments_cylindre.py  # capteurs sur un cylindre (~25 min)
python sparse_sources/experiments_ml3d.py    # sources en 3D (~10 min)
python sparse_sources/experiments.py --rapide --seulement hypotheses prony
```

Les PNG sont ignorés par le `.gitignore` du dépôt (`*.png`) : ils sont produits
localement par `experiments.py` ; les CSV sont versionnés.

## Résultats (détails, chiffres et figures : `COMPTE_RENDU.md`)

- **Noyau 1/r², capteurs dans le plan (projet), BLASSO standard : échec
  systématique.** La condition de Duval–Peyré est fausse dans 100 % des cas
  et on obtient des dizaines de sources parasites près des capteurs. Cause :
  Δ(1/r²) = 4/r⁴ > 0 (G sous-harmonique) et ‖φ‖ varie d'un facteur 215 sur D.
- **BLASSO pondéré** (`make_setup(normalize=True)` ou `alpha=1`) : 1 ou 2
  sources retrouvées dans 95 à 100 % des tirages à σ = 10⁻³ × rms, en ~0,05 s.
  L'erreur de position est proportionnelle à σ (pente 1 sur trois décades).
- **`check_hypotheses` prédit la réussite** : 94 % de réussite si la condition
  est vraie, 2 à 6 % sinon (100 tirages par α).
- **Comparaison avec Prony** (`reconstruct.py`, scénario du projet) : Prony
  est meilleur à σ = 10⁻⁴, le BLASSO pondéré à σ = 10⁻³ pour 2 sources ;
  tout échoue pour 5 sources ou σ = 10⁻².
- **Maximum de vraisemblance + BIC** (`greedy_ml`) : meilleure méthode sur le
  scénario du projet, erreur au niveau de la borne de Cramér-Rao, détection
  conforme à la théorie du BIC ; généralisée en 3D (`ml3d.py`). Voir
  `RAPPORT_ML_BIC.md`.
- **Capteurs sur un cylindre** : n'aident pas le Frank-Wolfe, même avec le λ
  optimal pour un K connu et des sources espacées.

## Limites observées

- Séparation minimale ≈ 0,5 (σ = 10⁻⁵) à 4 (σ = 10⁻²), soit environ d* ∝ σ^{1/4}.
- Zone aveugle : paires à moins de ~4–5 d'un capteur (le cercle de capteurs
  passe par les coins de D).
- K ≥ 3 sources tirées au hasard : 25 % de réussite à K = 3, ≤ 5 % à K = 5.
- Capteurs très hors du plan (h = 20) : condition vraie mais cond(Γ) ≈ 10⁵,
  inexploitable numériquement. La condition est qualitative.
- FISTA est lent sur grille fine (27 s au pas de 0,1, contre 0,17 s pour l'ensemble actif).
- Pour un λ < ~10⁻⁶ λ_max, le critère η ≤ 1 + 10⁻⁶ n'est plus atteignable numériquement.

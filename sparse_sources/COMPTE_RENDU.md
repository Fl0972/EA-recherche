# Compte rendu — Localisation de sources ponctuelles par parcimonie (LASSO, BLASSO) et hypothèses de Duval–Peyré

*Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/`. Toutes les
figures citées sont produites par `experiments.py` et `experiments_1r2.py` dans
`sparse_sources/results/` ; les tableaux chiffrés viennent des CSV du même dossier.*

---

## 0. Résumé

- On a implémenté trois outils : un **LASSO positif sur grille** (FISTA,
  sklearn, ensemble actif), un **BLASSO sans grille par Frank-Wolfe glissant**
  et une fonction **`check_hypotheses`** qui vérifie la condition de source non
  dégénérée de Duval–Peyré (H1, H2, H3). Le code est testé (62 tests pytest).
- **Pour le noyau physique du projet, G = 1/‖x − z‖² (capteurs dans le plan),
  le BLASSO standard ne fonctionne jamais** : la condition de Duval–Peyré est
  fausse dans 100 % des 100 tirages testés, et la reconstruction échoue dans
  100 % des cas (des dizaines de sources parasites près des capteurs). Ce
  n'est pas un problème d'algorithme : G(x, ·) est **sous-harmonique**
  (Δ(1/r²) = 4/r⁴ > 0), donc un certificat ne peut pas avoir de maximum strict
  à l'intérieur de D, et la pénalité Σ I_k favorise les sources près des
  capteurs (‖φ‖ varie d'un facteur 215 sur D).
- **Remède : BLASSO pondéré**, pénalité λ Σ I_k ‖φ(z_k)‖^α (même modèle
  1/r², mêmes mesures). Avec **α ≈ 1 à 1,25** :
  - une ou deux sources sont retrouvées dans 95 à 100 % des tirages
    (σ = 10⁻³ × rms) en ~0,05 s ;
  - l'erreur de position est **exactement proportionnelle au bruit** (pente 1
    sur trois décades), comme le prédit le théorème ;
  - **`check_hypotheses` prédit la réussite** : 94 % de réussite quand la
    condition est vraie, 2 à 6 % quand elle est fausse.
- **Limites** : sources proches (séparation minimale ≈ 0,5 à 4 selon le bruit),
  **zone aveugle près des capteurs** (paires à moins de ~4–5 d'un capteur),
  K ≥ 3 sources tirées au hasard (25 % de réussite à K = 3, 0 à 5 % à K = 5).
- **Comparaison avec `reconstruct.py` (Prony)** dans le scénario réel du projet
  (source centrale de 5000, bruit relatif) : Prony est meilleur à très faible
  bruit (σ = 10⁻⁴), le BLASSO pondéré est meilleur à σ = 10⁻³ pour 2 sources
  (75 % contre 25 %) ; tous échouent à σ = 10⁻² ou pour 5 sources. La source
  centrale domine le bruit (40 par capteur contre un signal utile ~7 en rms).

---

## 1. Cadre et notations

- Domaine des sources : D = [0, 10] × [0, 20] (`config/simulation.yaml`).
- M = 64 capteurs équirépartis sur le **cercle circonscrit** à D (centre
  (5, 10), rayon R ≈ 11,18), dans le plan des sources (`config/sensors.yaml`).
  Ce cercle passe **par les coins de D** : certains capteurs sont à ~0,3 d'un coin.
- Sources : s = Σ_k I_k δ_{z_k}, I_k ≥ 0, K inconnu.
- Noyau du projet (`particle_sources.model.field.intensity_at`) :

      G(x_i, z) = 1 / (‖x_i − z‖² + h²),    φ(z) = (G(x_1, z), …, G(x_M, z)) ∈ R^M

  h = 0 dans le projet (paramètre `eps`). Un capteur placé à la hauteur h
  au-dessus du plan voit exactement ce noyau : h sert à étudier des capteurs
  hors du plan.
- Mesures : m = A(s) + w, A(s) = Σ_k I_k φ(z_k), bruit gaussien.
  Dans `sparse_sources`, bruit **additif** σ = niveau × rms(A(s)) ; pour la
  comparaison avec Prony, bruit **relatif** du projet y(1 + σε).

### Problèmes résolus

- **BLASSO positif** (sur les mesures, sans grille) :
  min_{s ≥ 0} ½‖m − A(s)‖² + λ‖s‖_TV, ‖s‖_TV = Σ I_k.
- **LASSO positif sur grille** (positions restreintes aux nœuds y_j) :
  min_{c ≥ 0} ½‖m − Gc‖² + λ Σ c_j, G_ij = G(x_i, y_j) (M × P).
- **BLASSO pondéré** (ajout) : pénalité λ Σ I_k ‖φ(z_k)‖^α ; c'est le BLASSO
  pour le noyau ψ = φ/‖φ‖^α avec les intensités Ĩ_k = I_k‖φ(z_k)‖^α.

### Certificat et optimalité

Pour le résidu r = m − A(s), η(z) = (1/λ) Σ_i r_i G(x_i, z).
s est solution du BLASSO **si et seulement si** η ≤ 1 sur D et η(z_k) = 1 aux
sources. C'est le critère d'arrêt du Frank-Wolfe et le test « certifié » du LASSO.

### Théorème de Duval–Peyré (2015) et hypothèses vérifiées

Γ = [φ(z_1)…φ(z_K) | ∂₁φ(z_1)…∂₁φ(z_K) | ∂₂φ(z_1)…∂₂φ(z_K)] ∈ R^{M×3K},
p_V = pinv(Γᵀ)(1_K, 0_{2K}) (norme minimale), η_V(z) = Σ_i p_V,i G(x_i, z).

- (H1) Γ de rang 3K (il faut M ≥ 3K) ;
- (H2) η_V(z) < 1 pour z ∈ D hors des z_k (bord de D compris) ;
- (H3) ∇²η_V(z_k) définie négative.

Si (H1–H3) tiennent et les z_k sont intérieurs, il existe λ₀, c₀ > 0 tels que
pour λ ≤ λ₀ et ‖w‖/λ ≤ c₀, le BLASSO a **exactement K** Diracs, à distance
O(‖w‖) des vraies sources. Le théorème est **qualitatif** : λ₀ et c₀ ne sont pas
calculés. On mesure donc aussi la marge et le conditionnement (§ 5.3).

---

## 2. Méthodes : ce qui a été implémenté

### 2.1 Noyau (`kernel.py`)

- Réutilise `InverseSquareKernel` du projet : φ, ∇φ (M×2) et ∇²φ (M×2×2) **analytiques** :
  ∇G = −2(z − x)/q², ∇²G = −2 Id/q² + 8(z − x)(z − x)ᵀ/q³, q = ‖z − x‖² + h².
- `Setup` regroupe capteurs, h, D et la pondération α ; `make_setup()` lit `config/`.
- `WeightedKernel` : ψ = φ/‖φ‖^α avec dérivées analytiques (quotient, formules
  dans la docstring). Toutes les dérivées sont vérifiées contre des différences
  finies (erreur relative ≤ 10⁻⁹).

### 2.2 Briques communes (`common.py`)

| Fonction | Rôle |
|---|---|
| `forward(z, I)` | A(s) = Σ I_k φ(z_k) |
| `certificate`, `max_certificate` | η sur des points ; max sur D : grille fine puis L-BFGS-B borné depuis les 5 meilleurs maxima locaux (le bord de D est géré par les bornes) |
| `joint_descent` | min_{I ≥ 0, z ∈ D} ½‖m − Σ I_k φ(z_k)‖² + λ Σ I_k, L-BFGS-B, gradient analytique, intensités mises à l'échelle |
| `debias` | NNLS (`scipy.optimize.nnls`) à positions fixées, sans pénalité |
| `diagnostics` | ‖r‖² comparé à Mσ² (σ scalaire ou par capteur), attendu (M − 3K)/M après ajustement |
| `match_sources` | appariement hongrois, erreurs de position et d'intensité |

### 2.3 LASSO positif sur grille (`lasso.py`)

1. `build_grid(D, step)` : grille et matrice G (M × P). Pas de 0,25 → P = 3321 ; pas de 0,1 → P = 20 301.
2. `solve_lasso` :
   - **FISTA** (codé) : pas 1/L, L = ‖G‖₂², prox = max(· − λ/L, 0),
     redémarrage adaptatif (O'Donoghue–Candès), arrêt sur les conditions KKT.
     Ajout d'un **polissage** : LASSO exact restreint au support trouvé par
     FISTA, accepté seulement s'il vérifie KKT sur **toute** la grille ;
   - **sklearn** `Lasso(positive=True, fit_intercept=False)` avec
     **α_sklearn = λ/M** (sklearn minimise (1/2n)‖y − Xw‖² + α‖w‖₁, n = M) ;
   - **ensemble actif** exact (algorithme du projet, `duval_peyre.blasso`), comme référence.
3. `merge_peaks` : composantes connexes des nœuds actifs (rayon 1,5 pas) →
   I = Σ c_j, z = barycentre pondéré.
4. `refine` : K fixé, descente conjointe (I, z) sans grille.
5. `lasso_pipeline` : grille → LASSO → regroupement → raffinement → test
   max η ≤ 1 → débiaisage → diagnostics.

### 2.4 Frank-Wolfe glissant (`blasso.py`)

Denoyelle, Duval, Peyré, Soubies (2019). Départ μ = 0, ou une initialisation donnée (par ex. la sortie du LASSO). À chaque itération :

- a) on calcule r = m − A(μ) ;
- b) on cherche z* = argmax η ;
- c) on s'arrête si η(z*) ≤ 1 + tol ;
- d) on ajoute un Dirac en z* ;
- e) on réajuste les intensités par un LASSO positif exact sur K colonnes ;
- f) **glissement** : optimisation conjointe de (I, z), L-BFGS-B borné à D ;
- g) on supprime les Diracs nuls et on fusionne ceux à moins de 10⁻³ × diagonale.

La sortie contient l'historique (K, max η, objectif) puis le débiaisage.

### 2.5 Hypothèses (`hypotheses.py`)

`check_hypotheses(z, lam, sigma, grid_step, exclusion_radius, verbose, setup, …)` :

1. Γ, rang par SVD et conditionnement (brut et colonnes normalisées). Avertissement si M < 3K.
2. p_V par pseudo-inverse. On contrôle η_V(z_k) = 1 et ∇η_V(z_k) = 0 (≈ 10⁻¹⁵).
3. (H2) : grille fine (pas = min(taille D)/200) hors de boules de rayon ρ
   (défaut min(¼ séparation, 1 % diagonale)), raffinement L-BFGS-B depuis 10
   maxima locaux, maximisation 1D sur chacun des 4 côtés de D, et contrôle
   que η_V ne dépasse pas 1 à l'intérieur des boules.
4. (H3) : valeurs propres de la hessienne 2×2 de η_V en chaque z_k.
5. Sources intérieures à D (distance au bord).
6. Dictionnaire de sortie : H1, H2, H3, `non_degenerate`, rang, cond, marge,
   **marge relative**, valeurs propres, bornes de bruit si λ et σ sont donnés.
7. Mode verbeux : résumé et carte de η_V (capteurs, sources, contour η_V = 1).

**Choix numérique important (H2).** La marge 1 − max η_V hors des boules est
petite par nature (de 10⁻⁷ à 10⁻⁵ ici), car au bord d'une boule
η_V ≈ 1 − ½|λ_max|ρ². Un seuil fixe (10⁻⁶ dans le code d'origine du projet) donne
des verdicts arbitraires. On utilise donc :

- un seuil au niveau du bruit numérique, max(10⁻⁹, 100 × erreur d'interpolation) ;
- une **marge relative** = marge / (½|λ_max(∇²η_V)|ρ²). Elle vaut environ 1
  quand η_V décroît comme le prédit la hessienne, elle est ≪ 1 quand η_V est
  presque plat, et elle est < 0 quand (H2) échoue.

`hypotheses_after_fit(z_hat, I_hat)` applique la même vérification aux
positions **estimées** et renvoie `reliable`.

### 2.6 Choix de λ

- Expériences génériques : λ = c σ √M (c = 2 par défaut ; balayage c ∈ [0,25 ; 16]).
- Étude 1/r² : **λ = c σ √M ‖Ψ‖**, ‖Ψ‖ = max_D ‖ψ(z)‖. Avec ce choix, le bruit
  seul ne peut pas faire dépasser 1/c au certificat, quel que soit α. Avec le
  noyau brut, ‖Φ‖ ≈ 13,7, donc λ = c σ √M seul ne protège pas des sources
  parasites près des coins.

---

## 3. Structure du code

```
sparse_sources/
├── __init__.py          API publique
├── kernel.py            Setup, make_setup, WeightedKernel, φ, ∇φ, ∇²φ
├── common.py            forward, certificat, max_certificate, descente, debias, diagnostics
├── lasso.py             build_grid, FISTA / sklearn / ensemble actif, merge_peaks, refine, pipeline
├── blasso.py            sliding_frank_wolfe
├── hypotheses.py        check_hypotheses, hypotheses_after_fit, format_report
├── plotting.py          style des figures (palette fixe, carte de certificat)
├── experiments.py       expériences générales (3 géométries, comparaison Prony)
├── experiments_1r2.py   étude ciblée du noyau 1/r²
├── tests/               62 tests pytest
├── results/             CSV (versionnés) et PNG (ignorés par .gitignore)
├── README.md            notice courte
└── COMPTE_RENDU.md      ce document
```

Dépendances : `sparse_sources` importe `particle_sources` (noyau, `Box`,
`CircleSensors`, config, Prony, génération). **Aucun fichier existant n'a été
modifié.** Toutes les fonctions prennent un argument `setup=` facultatif ;
sans lui, elles utilisent la géométrie de `config/`.

Commits (un par partie) : point de départ, common, lasso, blasso, hypothèses,
tests, tests bruit, expériences, pondération α et étude 1/r², compte rendu.

---

## 4. Validation (tests)

`pytest sparse_sources/tests` : **62 tests, ~28 s**. Les 67 tests existants du
projet passent toujours.

| Ce qui est testé | Résultat |
|---|---|
| ∇φ, ∇²φ (noyaux brut, h = 20, pondérés α = 0,5 ; 1 ; 1,5 ; 2) contre différences finies | erreur ≤ 10⁻⁹ |
| φ = noyau du projet (`intensity_at`) | identique |
| η_V(z_k) = 1, ∇η_V(z_k) = 0 | ≤ 10⁻¹⁰ |
| Une source bien vue (α = 1, ou h = 20) | H1, H2, H3 vraies |
| Noyau brut 1/r² dans le plan, une source | H3 et H2 fausses (sous-harmonicité) |
| Deux sources très proches | H2 ou H3 fausse |
| Accord avec `duval_peyre.diagnose` du projet | mêmes verdicts, même marge (10⁻⁵) |
| LASSO, source hors grille | 2 nœuds voisins → 1 source après regroupement |
| FISTA / sklearn / ensemble actif | même objectif (10⁻⁴ ; 10⁻³ pour sklearn) |
| Frank-Wolfe sans bruit | positions à < 10⁻⁴ (5 × 10⁻⁵), arrêt avec max η ≤ 1 + 10⁻⁶ |
| Bruit 10⁻⁴, 10⁻³, 10⁻² × rms (Frank-Wolfe et LASSO) | bon K, erreur ≤ 300 σ_rel, ‖r‖²/Mσ² ∈ [0,3 ; 3] |
| Linéarité erreur / bruit (10⁻⁵ → 10⁻³) | rapport erreur/σ constant à un facteur 3 près |
| Bruit fort (0,3 × rms) | dégradation nette |

---

## 5. Résultats

### 5.1 Trois géométries, trois configurations (`experiments.py`)

Géométries : **plan_brut** (projet : h = 0, α = 0) ; **plan_normalise**
(h = 0, α = 1) ; **hauteur_20** (capteurs à h = 20, α = 0). Configurations :
3 sources séparées ; 2 sources à distance 1 ; 2 sources avec capteurs sur un
cercle 2 fois plus grand (« loin »).

**Hypothèses** (`hypotheses.csv`, cartes `certificat_*.png`) :

| géométrie | config | H1 | H2 | H3 | marge | marge relative | cond(Γ) normalisé |
|---|---|---|---|---|---|---|---|
| plan_brut | séparées | ✔ | ✘ | ✘ | −41,8 | < 0 | 122 |
| plan_brut | proches | ✔ | ✘ | ✘ | −13,7 | < 0 | 3 980 |
| plan_brut | loin | ✔ | ✘ | ✘ | −0,50 | < 0 | 94 |
| plan_normalise | séparées | ✔ | ✔ | ✔ | 8,5 × 10⁻⁶ | 1,24 | 92 |
| plan_normalise | proches | ✔ | ✔ | ✔ | 1,1 × 10⁻⁶ | 0,70 | 3 950 |
| plan_normalise | loin | ✔ | ✔ | ✔ | 6,5 × 10⁻⁶ | 1,32 | 85 |
| hauteur_20 | séparées | ✔ | ✔ | ✔ | 8,4 × 10⁻⁷ | 1,42 | **127 000** |
| hauteur_20 | proches | ✔ | ✔ | ✔ | 1,5 × 10⁻⁷ | 0,70 | **293 000** |
| hauteur_20 | loin | ✔ | ✘ | ✘ | −0,012 | < 0 | 507 |

**Comparaison des méthodes** (c = 2, 4 tirages, `comparaison_methodes_resume.csv`), taux de réussite (bon K et erreur < ¼ séparation) :

| géométrie / config | σ/rms | LASSO | LASSO + raffinement | Frank-Wolfe |
|---|---|---|---|---|
| plan_brut, toutes | 10⁻⁴ à 10⁻² | 0 % (23 à 40 sources) | 0 % | 0 % |
| plan_normalise, séparées | 10⁻⁴ / 10⁻³ / 10⁻² | 0 / 0 / 25 % | 50 / 75 / 75 % | **100 / 100 / 75 %** |
| plan_normalise, proches | 10⁻⁴ / 10⁻³ / 10⁻² | 50 / 25 / 0 % | 100 / 25 / 0 % | **100 / 25 / 0 %** |
| plan_normalise, loin | 10⁻⁴ / 10⁻³ / 10⁻² | 50 / 50 / 75 % | 50 / 100 / 25 % | **100 / 100 / 25 %** |
| hauteur_20, séparées et proches | 10⁻⁴ à 10⁻² | 0 % (1 seule source) | 0 % | 0 % |

Le LASSO sans raffinement surestime souvent K, à cause d'un petit pic voisin
non fusionné. Le raffinement corrige la plupart des cas, et le Frank-Wolfe est
le plus fiable.

**Erreur de position en fonction de σ** (`erreur_sigma.png`, Frank-Wolfe, plan_normalise) :
l'erreur est **proportionnelle à σ** de 10⁻⁵ à 10⁻² × rms, avec un rapport
err / σ_rel constant : ≈ 72 (séparées), ≈ 97 (loin), ≈ 330 (proches). C'est
exactement la loi O(‖w‖) du théorème, et la constante grandit quand les
sources se rapprochent. Le régime linéaire s'arrête là où K devient faux.

**Balayage de λ** (`balayage_lambda.png`) : avec α = 1, K est exact pour c de
0,5 à 4 environ. Un c trop petit (< 0,5) ajoute des sources parasites, un c trop
grand (> 4) fusionne les sources proches. Avec le noyau brut, on trouve 20 à 35
sources de trop pour tout c (le plateau à +7 ou +8 du Frank-Wolfe vient du
plafond de 10 itérations).

### 5.2 Étude ciblée du noyau 1/r² (`experiments_1r2.py`)

Sources tirées comme dans `config/` (positions uniformes à ≥ 0,25 du bord,
intensités uniformes sur le simplexe, somme 100), 20 tirages par K,
σ = 10⁻³ × rms, λ = 2σ√M‖Ψ‖.

**Effet de la pondération α** (`r2_alpha.png`) :

| α | condition vraie (K = 1 / 2 / 3 / 4 / 5) | réussite Frank-Wolfe (K = 1 / 2 / 3 / 4 / 5) | temps médian |
|---|---|---|---|
| 0 (brut) | 0 / 0 / 0 / 0 / 0 % | 0 / 0 / 0 / 0 / 0 % | 0,59 s (non convergé) |
| 0,5 | 0 / 0 / 0 / 0 / 0 % | 0 / 0 / 0 / 0 / 0 % | 1,29 s |
| 0,75 | 35 / 25 / 0 / 0 / 0 % | 35 / 25 / 0 / 0 / 0 % | 0,21 s |
| **1** | 100 / 100 / 20 / 20 / 0 % | **100 / 95 / 25 / 15 / 5 %** | **0,05 s** |
| **1,25** | 100 / 100 / 25 / 25 / 0 % | **100 / 95 / 30 / 15 / 0 %** | 0,04 s |
| 1,5 | 100 / 100 / 35 / 30 / 0 % | 100 / 95 / 25 / 5 / 0 % | 0,04 s |
| 2 | 100 / 90 / 50 / 30 / 0 % | 100 / 80 / 20 / 5 / 0 % | 0,04 s |

Il faut α ≥ 1 pour une seule source. C'est démontrable : avec α = 1,
η = ⟨ψ(z), ψ(z₀)⟩ ≤ 1 par Cauchy–Schwarz, avec égalité seulement en z₀.
Au-delà de α ≈ 1,25, la condition est plus souvent vraie, mais avec un
conditionnement plus mauvais, et la réussite baisse. **Recommandation : α = 1
(ou 1,25).**

**Pouvoir prédictif de `check_hypotheses`** (`r2_prediction.png`, 100 tirages par α) :

| α | condition vraie | réussite si vraie | réussite si fausse |
|---|---|---|---|
| 0,75 | 12 / 100 | 92 % | 1 % |
| **1** | 48 / 100 | **94 %** | **6 %** |
| 1,25 | 50 / 100 | 94 % | 2 % |
| 1,5 | 53 / 100 | 83 % | 2 % |
| 2 | 54 / 100 | 74 % | 2 % |

Le signe de la marge relative sépare presque parfaitement les réussites des
échecs. La vérification des hypothèses est donc un **outil de diagnostic
fiable**, utilisable a posteriori via `hypotheses_after_fit`.

**Limite de résolution** (`r2_resolution.png`, paire verticale au centre, α = 1, 5 tirages) :

| σ / rms | séparation minimale résolue (réussite = 100 %) | erreur à d = 2 |
|---|---|---|
| 10⁻⁵ | 0,5 | 1,3 × 10⁻³ |
| 10⁻⁴ | 1 | 1,3 × 10⁻² |
| 10⁻³ | 1,5 | 0,14 |
| 10⁻² | 4 | — (K faux à d = 2) |

Soit environ d* ∝ σ^{1/4} : le bruit admissible décroît comme une puissance
élevée de la séparation (≈ d⁴), ce qui est typique de la super-résolution de
sources positives proches. À d = 0,25, la condition est vraie mais
cond(Γ) = 2,5 × 10⁵ et aucun niveau de bruit testé ne permet la reconstruction.
À séparation fixée, l'erreur reste proportionnelle à σ et décroît environ
comme d^{-1,5}.

**Distance aux capteurs** (paire horizontale de séparation 2, α = 1, σ = 10⁻³) :

| y (centre = 10) | distance au capteur le plus proche | condition | réussite |
|---|---|---|---|
| 10 à 14 | 10,2 à 7,1 | vraie | 100 % |
| 16 | 5,1 | vraie | 60 % |
| 17,5 / 18,5 / 19,25 | 3,6 / 2,6 / 1,9 | **fausse** | **0 %** |

Une **source seule** reste reconstructible près des capteurs (la condition tient
jusqu'à 1,7 d'un capteur). C'est la **paire** qui échoue :
`r2_certificat_pres_capteurs.png` montre η_V > 1 le long d'un arc qui relie les
deux sources, parallèle à l'arc de capteurs. Près d'un capteur, ψ(z) est dominé
par ce capteur, et deux sources à la même distance des mêmes capteurs ont des
signatures presque identiques. Elles sont proches au sens du noyau, même à 2
unités l'une de l'autre.

**Nombre de capteurs** (`r2_capteurs.png`, 10 tirages par K) : M = 16 à 256
change peu de chose (K = 1 : 100 % partout ; K = 2 : 90 % à 100 % ; K = 3 :
20 à 40 % ; K = 5 : 0 %). Les échecs sont **géométriques** (sources proches ou
près des capteurs) et non dus au bruit ; ajouter des capteurs sur le **même
cercle** n'apporte pas d'information nouvelle.

### 5.3 La condition est qualitative : le cas h = 20

Avec des capteurs à h = 20 au-dessus du plan, G(x, ·) n'est plus
sous-harmonique sur D (ΔG = 4(r² − h²)/(r² + h²)³ < 0 pour r < h), et la
condition est **vraie** pour les configurations séparées et proches. Pourtant
**la reconstruction échoue même sans bruit** : on obtient une seule source au
lieu de 3, ou des Diracs parasites quand λ → 0. Le noyau est si lisse (h plus
grand que la taille de D) que les colonnes sont presque colinéaires :
cond(Γ) ≈ 10⁵. Les constantes λ₀ et c₀ du théorème sont alors hors de portée
de la double précision et des tolérances de L-BFGS. **À retenir : (H1–H3) ne
suffisent pas en pratique ; il faut regarder cond(Γ) et la marge relative.**

### 5.4 Comparaison avec `scripts/reconstruct.py` (Prony), données bruitées

Scénario identique à `reconstruct.py` : n sources tirées selon `config/`, source
centrale d'intensité 5000 mesurée puis retirée, bruit relatif y(1 + σε),
8 tirages par case. Méthodes sur les **mêmes mesures** :
- `prony_config` : `cfg.invert`, exactement comme dans le script ;
- `prony_5sigma` : Prony avec rank_tol = 5σ (réglage conseillé en commentaire de la config) ;
- `sfw_brut`, `sfw_normalise`, `lasso_normalise` : λ = 1,5 σ‖y‖ (estimation de ‖w‖).

Réussite (bon K et erreur < ¼ séparation) et erreur de position médiane (`comparaison_prony_resume.csv`, `comparaison_prony.png`) :

| n | σ | Prony (config) | Prony (5σ) | Frank-Wolfe brut | LASSO normalisé | Frank-Wolfe normalisé |
|---|---|---|---|---|---|---|
| 2 | 10⁻⁴ | 88 % (0,02) | 25 % | 0 % | **100 %** (0,05) | 88 % (0,05) |
| 2 | 10⁻³ | 25 % (0,41) | 25 % | 0 % | **75 %** (0,50) | 62 % (0,50) |
| 2 | 10⁻² | 0 % | 12 % | 0 % | 12 % | 12 % |
| 3 | 10⁻⁴ | **62 %** (0,16) | 25 % | 0 % | 25 % | 25 % |
| 3 | 10⁻³ | 25 % | 25 % | 0 % | 0 % | 0 % |
| 5 | 10⁻⁴ | 12 % | **25 %** | 0 % | 0 % | 0 % |
| 3 et 5 | 10⁻² | 0 % | 0 % | 0 % | 0 % | 0 % |

Temps : Prony ≈ 1 ms ; Frank-Wolfe normalisé 5 à 160 ms ; LASSO normalisé ≈ 0,6 s ; Frank-Wolfe brut 0,1 à 0,6 s.

Lecture :
- **Prony est meilleur à très faible bruit et pour plus de sources.** Il est
  exact sans bruit, mais très sensible au bruit, car les coefficients de Fourier
  décroissent comme (ρ/R)ⁿ.
- **Le BLASSO pondéré est plus robuste à bruit moyen** pour 2 sources.
- Le BLASSO brut ne marche jamais (§ 6.1).
- **Le scénario est très défavorable.** La source centrale apporte 40 à chaque
  capteur, alors que le signal des sources utiles vaut ~7 en rms. Le bruit
  relatif est donc dominé par la source centrale : σ = 10⁻³ correspond à un bruit
  de ~6 × 10⁻³ du signal utile. Comme elle est au centre du cercle, ce bruit est
  quasi uniforme, et un blanchiment n'aiderait pas.
- Dans la config par défaut, K suit une loi de Poisson(20) : **20 sources sont
  hors de portée de toutes les méthodes testées** (0 % dès K = 5).

---

## 6. Où ça ne marche pas, et pourquoi

### 6.1 BLASSO standard avec le noyau 1/r² (cas du projet) : échec systématique

- **Constat** : condition fausse dans 100 % des cas (100 tirages aléatoires
  et 3 configurations fixes), 0 % de réussite. On obtient 20 à 40 sources parasites
  collées au bord de D, en particulier aux coins, avec des intensités fausses
  (30 à 90 % d'erreur).
- **Cause 1, sous-harmonicité.** En 2D, Δ_z(1/‖x − z‖²) = 4/‖x − z‖⁴ > 0 :
  chaque G(x_i, ·) est sous-harmonique sur D. Dans le cas symétrique
  (p_V ≥ 0), η_V est sous-harmonique et n'a **pas de maximum strict
  intérieur**. La hessienne en z_k a une trace positive (H3 fausse) et le
  maximum est au bord (H2 fausse, max η_V entre 17 et 43 près des coins).
- **Cause 2, biais de la pénalité.** ‖φ(z)‖ varie d'un facteur 215 sur D
  (0,064 au centre, 13,7 près des coins). Pour la pénalité λ Σ I_k, une source
  près d'un capteur produit 215 fois plus de signal par unité de masse : le
  BLASSO « paie » moins cher pour remplacer une source centrale par plusieurs
  sources de bord. La solution du LASSO sur grille est d'ailleurs un **anneau**
  ou des sources de bord, même sans bruit (vérifié : FISTA, sklearn et
  l'ensemble actif donnent le même objectif à 10⁻⁷ près).
- **Remède** : la pondération α ≈ 1 (§ 5.2).

### 6.2 Sources proches

Limite de résolution d* ≈ 0,5 à 4 selon le bruit (§ 5.2). Le conditionnement de
Γ explose (≈ 4 000 à d = 1, ≈ 250 000 à d = 0,25) et la constante de l'erreur
aussi (err/σ ≈ 330 à d = 1 contre ≈ 72 pour des sources bien séparées).

### 6.3 Zone aveugle près des capteurs

Le cercle de capteurs passe par les coins de D. Les **paires** situées à moins
de ~4–5 d'un capteur ne sont plus séparables (§ 5.2). Comme les positions
uniformes tombent souvent près du bord, cela explique en grande partie les
faibles taux pour K ≥ 3 tirées au hasard.

### 6.4 Nombre de sources

Avec des positions aléatoires dans D, la probabilité d'avoir une paire proche
ou une paire près des capteurs croît vite avec K : 25 % de réussite à K = 3,
15 % à K = 4, 0 à 5 % à K = 5. Les K ≈ 20 de la config par défaut sont hors de
portée.

### 6.5 Noyau trop lisse (capteurs hors du plan, h grand)

Condition vraie mais cond(Γ) ≈ 10⁵ : inexploitable numériquement (§ 5.3).

### 6.6 Problèmes numériques rencontrés

- **FISTA lent** sur grille fine : les colonnes voisines sont presque
  colinéaires. Sans polissage, il reste 16 nœuds actifs au lieu de 2 après
  50 000 itérations. Le préconditionnement diagonal n'aide pas. Avec un pas de
  0,1, le pipeline FISTA prend 27 s, contre 0,17 s pour l'ensemble actif.
- **sklearn** s'arrête sur son nombre maximal d'itérations (écart d'objectif ~3 × 10⁻⁴).
- **LASSO + raffinement** : un petit pic parasite du LASSO peut rester bloqué
  dans un minimum local. Son intensité est ≈ 0, donc son gradient en position
  est ≈ 0. Il est retiré au débiaisage, mais la solution n'est pas certifiée
  (η_max ≈ 1,003). Le Frank-Wolfe initialisé par le LASSO corrige ce cas.
- **λ très petit** (< ~10⁻⁶ λ_max) : η = φ·r/λ amplifie l'erreur
  d'optimisation, et le critère η ≤ 1 + 10⁻⁶ n'est plus atteignable. Le biais
  de position du BLASSO est O(λ) (≈ 15 λ/λ_max sans bruit).
- **Seuil de (H2)** : un seuil fixe sur la marge donne des verdicts arbitraires,
  d'où la marge relative (§ 2.5).

---

## 7. Temps de calcul

Machine de développement (Windows, Python 3.14, NumPy 2.3, SciPy 1.16), un cœur, médianes.

| Brique | Paramètres | Temps |
|---|---|---|
| `make_grid` (matrice G) | pas 0,1 (P = 20 301) | 0,07 s |
| Frank-Wolfe glissant, α = 1 | K = 1 / 2 / 3 / 5 / 8 (M = 64, grille 0,25) | 0,004 / 0,017 / 0,045 / 0,10 / 0,44 s |
| Frank-Wolfe glissant, α = 1, K = 3 | M = 16 / 64 / 256 | 0,05 / 0,05 / 0,06 s |
| Frank-Wolfe glissant, noyau brut | plafonné à 10 itérations, non convergé | 0,4 à 1,3 s |
| `lasso_pipeline` ensemble actif, K = 3 | pas 0,5 / 0,25 / 0,1 | 0,03 / 0,04 / 0,17 s |
| `lasso_pipeline` FISTA, K = 3 | pas 0,5 / 0,25 / 0,1 | 0,86 / 2,0 / 27 s |
| `lasso_pipeline` sklearn, K = 3 | pas 0,5 / 0,25 / 0,1 | 1,5 / 1,9 / 16 s |
| `check_hypotheses` | grille 0,05 (80 000 points), K = 1 à 8 | 0,25 à 0,30 s |
| Prony (`prony_circle`) | M = 64 | ~0,001 s |
| `pytest sparse_sources/tests` | 62 tests | 28 s |
| `experiments.py` | tout | 336 s (≈ 6 min) |
| `experiments_1r2.py` | tout (20 tirages par K) | 391 s (≈ 6,5 min) |

Le Frank-Wolfe glissant est de loin le plus rapide quand la condition tient
(≈ K + 1 itérations). Le coût du LASSO sur grille est dominé par le solveur :
FISTA est 15 à 150 fois plus lent que l'ensemble actif.

---

## 8. Pistes d'amélioration

**Modélisation et pénalité**
1. **Adopter le BLASSO pondéré (α = 1) pour le noyau 1/r²** et l'utiliser par
   défaut. Étudier une pondération adaptative, par exemple ℓ1 repondéré
   (Candès–Wakin–Boyd), pour réduire le biais en O(λ) sans perdre la stabilité.
2. **Ne pas soustraire la source centrale** : l'intégrer au dictionnaire, avec
   une position connue et une intensité inconnue, ou utiliser une attache aux
   données adaptée au bruit (log-vraisemblance de Poisson/KL pour
   `poisson_counts`, pondération 1/σ_i² pour le bruit relatif).
3. **Choix automatique de λ** : principe de discrépance (‖r‖² ≈ Mσ², déjà
   calculé dans `diagnostics`), chemin de régularisation, ou validation croisée
   sur les capteurs.

**Géométrie des capteurs** (c'est le facteur limitant principal)
4. **Éloigner les capteurs des coins de D**, avec un cercle plus grand que le
   cercle circonscrit (`radius_factor`), pour supprimer la zone aveugle (§ 6.3).
   Utiliser `check_hypotheses` sur des configurations types pour comparer des
   géométries, car c'est un critère fiable (94 % / 6 %).
5. **Diversifier les capteurs** plutôt que les multiplier sur le même cercle
   (plusieurs cercles, capteurs à l'intérieur, hauteur modérée h ~ 2–5). On
   veut casser la sous-harmonicité sans rendre le noyau trop lisse (h = 20 est
   trop lisse).

**Algorithmes**
6. **Hybride Prony → Frank-Wolfe** : Prony (ou AAA, déjà dans le projet) est
   excellent à faible bruit et coûte 1 ms. Utiliser sa sortie comme `z_init`
   du Frank-Wolfe glissant, puis certifier le résultat par max η ≤ 1 et
   `hypotheses_after_fit`.
7. **LASSO sur grille** : remplacer FISTA par l'ensemble actif ou une descente
   par coordonnées avec *screening*, ou utiliser une grille multi-résolution.
   Toujours enchaîner LASSO → Frank-Wolfe plutôt que LASSO → raffinement local.
8. **Précision** : une mise à l'échelle des variables et des tolérances de
   L-BFGS adaptées à λ permettraient d'exploiter les petits λ (§ 6.6).

**Théorie**
9. **Séparation dans la métrique du noyau** : la bonne notion de « sources
   proches » n'est pas la distance euclidienne mais la distance induite par
   ψ (métrique de Fisher–Rao, cf. Poon–Keriven–Peyré). Elle expliquerait
   quantitativement la zone aveugle et la loi d* ∝ σ^{1/4}.
10. **Constantes quantitatives** : relier λ₀ et c₀ à la marge relative et à
    cond(Γ) pour prédire le niveau de bruit admissible, et pas seulement « oui
    ou non ».
11. **Sources positives proches** : comparer la loi observée d* ∝ σ^{1/4} avec
    les résultats sur la super-résolution de pics positifs proches (Denoyelle,
    Duval, Peyré 2017, pré-certificat η_W).

---

## 9. Écarts par rapport à la demande initiale

- Toutes les fonctions acceptent un argument supplémentaire `setup=` (géométrie, h, α).
- `debias(z, I_init, m)` prend aussi les mesures m (nécessaires au NNLS).
- `check_hypotheses` a des paramètres en plus (`setup`, `plot`, `save_path`,
  `tol`, `estimates`) et renvoie en plus la marge relative et des indicateurs de bruit.
- Ajouts non demandés, jugés nécessaires :
  - la **pondération α** ;
  - le solveur exact à **ensemble actif** (référence) ;
  - le **polissage** de FISTA ;
  - le seuil numérique et la **marge relative** de (H2) ;
  - l'étude ciblée **1/r²** ;
  - la comparaison avec **Prony**.
- Les **PNG** ne sont pas versionnés (le `.gitignore` du dépôt ignore `*.png`) :
  relancer les deux scripts d'expériences pour les régénérer. Les CSV sont versionnés.

## 10. Reproduire

```bash
pip install -e ".[dev]"  scikit-learn       # sklearn uniquement pour la comparaison LASSO
pytest sparse_sources/tests                  # 62 tests, ~30 s
python sparse_sources/experiments.py         # ~6 min → results/
python sparse_sources/experiments_1r2.py     # ~6,5 min → results/r2_*
```

Toutes les graines sont fixées dans les scripts : les résultats sont reproductibles.

## Références

- V. Duval, G. Peyré, *Exact support recovery for sparse spikes deconvolution*, Found. Comput. Math., 2015.
- Q. Denoyelle, V. Duval, G. Peyré, E. Soubies, *The sliding Frank-Wolfe algorithm and its application to super-resolution microscopy*, Inverse Problems, 2019.
- Q. Denoyelle, V. Duval, G. Peyré, *Support recovery for sparse super-resolution of positive measures*, J. Fourier Anal. Appl., 2017.
- C. Poon, N. Keriven, G. Peyré, *The geometry of off-the-grid compressed sensing*, Found. Comput. Math., 2023.
- A. Beck, M. Teboulle, *A fast iterative shrinkage-thresholding algorithm*, SIAM J. Imaging Sci., 2009 ; B. O'Donoghue, E. Candès, *Adaptive restart for accelerated gradient schemes*, 2015.
- E. Candès, M. Wakin, S. Boyd, *Enhancing sparsity by reweighted ℓ1 minimization*, 2008.

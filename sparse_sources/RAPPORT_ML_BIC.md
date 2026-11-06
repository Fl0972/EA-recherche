---
title: "Maximum de vraisemblance glouton et critère BIC pour la localisation de sources ponctuelles"
subtitle: "Rapport sur la méthode greedy_ml (2D) et sa généralisation en 3D (greedy_ml_nd)"
lang: fr
---

# Maximum de vraisemblance + BIC : rapport complet

Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/`. Ce
rapport complète le compte rendu général (`COMPTE_RENDU.md`) : il est
entièrement consacré à la méthode qui donne les meilleurs résultats sur le
scénario du projet, le **maximum de vraisemblance glouton avec choix du nombre
de sources par le critère BIC** (abrégé « ML + BIC »). Les chiffres viennent
des fichiers CSV de `sparse_sources/results/`, produits par les scripts cités
dans chaque section.

Les formules sont écrites en LaTeX (`$...$` et `$$...$$`) ; le document se
convertit en LaTeX ou en PDF avec Pandoc (section 10).

## 0. Résumé

1. **La méthode.** On ajoute les sources une à une ; à chaque étape, toutes
   les positions et intensités sont réajustées **sans pénalité** en maximisant
   la vraisemblance gaussienne (bruit de chaque capteur pris en compte), et le
   nombre de sources est choisi par le critère **BIC** : une source n'est
   gardée que si elle fait baisser $\chi^2$ de plus de $3\log M\approx 12{,}5$
   (en 2D, 64 capteurs). Aucun paramètre à régler.
2. **C'est la meilleure méthode testée** sur le scénario du projet : elle bat
   Prony et le Frank-Wolfe dans toutes les cases (par exemple 80 % de réussite
   contre 55 % et 40 % pour 3 sources à $\sigma = 10^{-4}$ ; 50 % contre 5 % et
   0 % pour 5 sources).
3. **Elle est optimale** : son erreur vaut 0,7 à 0,9 fois la borne de
   Cramér–Rao (CRB), en 2D comme en 3D, et elle réussit dans 90 à 100 % des
   tirages où la CRB le permet. On ne peut pas faire beaucoup mieux en
   changeant d'algorithme.
4. **La théorie du BIC est vérifiée** : une source est détectée dès que son
   rapport signal sur bruit dépasse environ 3 (probabilité mesurée égale à la
   prédiction à quelques points près), avec 2 % de fausses alarmes sur du
   bruit seul. Surestimer le bruit est sans danger ; le sous-estimer d'un
   facteur 2 crée des sources parasites.
5. **Généralisation en 3D** (`ml3d.py`) : même méthode, 4 paramètres par
   source. Avec des capteurs sur une sphère, 1 à 2 sources sont retrouvées à
   $\sigma = 10^{-3}$ malgré la source centrale ; le BLASSO 3D du projet
   échoue partout. Un seul cercle de capteurs ne suffit pas en 3D (problème
   non identifiable).
6. **Ce qui limite, ce sont les données** : le bruit de la source centrale
   (CRB multipliée par 42 par rapport au même problème sans elle), le nombre
   de sources et leur proximité. Plus de capteurs aide lentement (CRB en
   $1/\sqrt M$) ; des capteurs en hauteur sur un cylindre n'aident pas.

| situation (2D, source centrale du projet) | ce que ML + BIC retrouve |
|---|---|
| $\sigma = 10^{-4}$ | 1 à 3 sources de façon fiable (80 à 100 %), 5 sources une fois sur deux |
| $\sigma = 10^{-3}$ | 1 à 2 sources (75 à 100 %), 3 sources dans 40 % des cas |
| $\sigma = 10^{-2}$ | 1 source ; au-delà, la CRB dépasse la taille du domaine |
| séparation minimale | 1 à $\sigma = 10^{-5}$, 2 à $\sigma = 10^{-4}$, 4 à $\sigma = 10^{-3}$ |
| intensité minimale (source isolée, $\sigma = 10^{-3}$) | environ 2 pour la détecter, 10 pour la localiser à 1 près |

## 1. Le problème

### 1.1 Modèle

On cherche $K$ sources ponctuelles de positions $z_k\in D$ et d'intensités
$I_k > 0$ à partir des mesures de $M$ capteurs placés en $x_1,\dots,x_M$. Le
noyau est celui du projet, la loi en $1/r^2$ :

$$
G(x_i, z) = \frac{1}{\lVert x_i - z\rVert^2 + h_i^2},
\qquad
y_i = \sum_{k=1}^{K} I_k\,G(x_i, z_k) + I_c\,G(x_i, c) .
$$

- En 2D, $D = [0,10]\times[0,20]$ et les $M = 64$ capteurs sont sur le cercle
  circonscrit à $D$, dans le plan des sources ($h_i = 0$). La hauteur $h_i$
  sert seulement à l'étude des capteurs sur un cylindre (section 6.6).
- En 3D, $D = [0,10]^3$, $h_i = 0$ et les capteurs sont hors d'un même plan
  (section 6.7).
- $I_c\,G(x_i, c)$ est la **source centrale** du projet : intensité
  $I_c = 50\times 100 = 5000$, au centre $c$ de $D$.

On note $\varphi(z) = \big(G(x_1,z),\dots,G(x_M,z)\big)\in\mathbb{R}^M$ la
signature d'une source d'intensité 1 placée en $z$, et
$A(z, I) = \sum_k I_k\,\varphi(z_k)$.

### 1.2 Bruit et source centrale

Le bruit du projet est **relatif** : le capteur $i$ mesure
$y_i(1 + \sigma\varepsilon_i)$, avec $\varepsilon_i\sim\mathcal N(0,1)$
indépendants. La contribution de la source centrale est connue et retirée :

$$
m_i = y_i(1 + \sigma\varepsilon_i) - I_c\,G(x_i, c)
    = A(z, I)_i + w_i,
\qquad w_i\sim\mathcal N\big(0,\,s_i^2\big),\quad s_i = \sigma\,y_i .
$$

Le signal de la source centrale disparaît, **pas son bruit** : en 2D,
$I_c\,G(x_i, c) = 5000/125 = 40$ pour chaque capteur, alors que le signal
utile des petites sources vaut environ 7 en moyenne quadratique. L'écart type
du bruit, $s_i\approx 40\,\sigma$, est donc dominé par la source centrale.
C'est ce qui rend le problème difficile.

### 1.3 Pourquoi pas le BLASSO

Le BLASSO (étudié dans le compte rendu général) résout

$$
\min_{K,\,z,\,I\ge 0}\ \tfrac12\big\lVert m - A(z, I)\big\rVert^2
+ \lambda\sum_k I_k\,\lVert\varphi(z_k)\rVert^{\alpha} .
$$

Il a trois défauts sur ce problème :

1. **un seul paramètre $\lambda$ règle deux choses** : le nombre de sources
   (il faut $\lambda$ assez grand pour rejeter le bruit) et le biais des
   positions et intensités (il faut $\lambda$ petit pour ne pas les fausser) ;
2. il ne réussit que si la **condition de Duval–Peyré** (H2) est vraie : un
   certificat qui reste sous 1 hors des sources. Avec le noyau $1/r^2$, elle
   est fausse dès que deux sources sont proches ou qu'une source est près d'un
   capteur ou d'un bord, ce qui arrive souvent à partir de 3 sources ;
3. il ignore la structure du bruit (la variance $s_i^2$ est connue).

Le maximum de vraisemblance + BIC **sépare** les deux rôles de $\lambda$ :
positions et intensités sont ajustées sans pénalité, et le nombre de sources
est choisi par un critère qui ne dépend que de $M$.

## 2. La méthode

### 2.1 Vraisemblance et blanchiment

Le bruit étant gaussien, indépendant, de variance $s_i^2$ connue, la
log-vraisemblance vaut, à une constante près,

$$
\log L(z, I) = -\frac12\sum_{i=1}^{M}\frac{\big(m_i - A(z, I)_i\big)^2}{s_i^2}
= -\frac12\,\chi^2(z, I),
\qquad
\chi^2(z, I) = \big\lVert m_w - A_w(z, I)\big\rVert^2 ,
$$

avec les mesures et le modèle **blanchis** : $(m_w)_i = m_i/s_i$ et
$A_w(z, I)_i = A(z, I)_i/s_i$. Chaque capteur compte d'autant moins que son
bruit est grand. En pratique, $s_i = \sigma\,y_i$ est calculé sur les
mesures brutes, source centrale comprise (`relative_noise_std`).

Si le modèle est juste, $\chi^2$ au vrai paramètre suit une loi
$\chi^2_M$ : il vaut environ $M$. C'est un contrôle simple de la
reconstruction (« $\chi^2/M\approx 1$ »).

### 2.2 À nombre de sources fixé : moindres carrés non linéaires

Pour $K$ fixé, l'estimateur du maximum de vraisemblance est

$$
(\hat z, \hat I)_K = \operatorname*{arg\,min}_{z_k\in D,\ I_k\ge 0}\ \chi^2(z, I) .
$$

C'est un problème de moindres carrés non linéaires en $z$ (linéaire en $I$).
On le résout par **descente conjointe** des positions et intensités
(L-BFGS-B, contraintes $z_k\in D$ et $I_k\ge 0$, gradient analytique :
$\partial_{I_k}\chi^2 = -2\,\varphi_w(z_k)^{\top} r_w$ et
$\nabla_{z_k}\chi^2 = -2\,I_k\,\nabla\varphi_w(z_k)^{\top} r_w$, avec
$r_w = m_w - A_w(z, I)$ le résidu blanchi). Il n'y a **aucune pénalité** : pas
de biais en $O(\lambda)$.

Le problème n'est pas convexe ; d'où la construction gloutonne de la
section 2.4, qui fournit un bon point de départ.

### 2.3 Choix du nombre de sources : le critère BIC

$\chi^2(\hat z, \hat I)_K$ diminue toujours quand $K$ augmente : une source de
plus « explique » toujours un peu de bruit. On choisit donc $K$ en minimisant

$$
C(K) = \chi^2\big(\hat z, \hat I\big)_K + \kappa\, p\, K,
\qquad p = d + 1 \ \text{paramètres par source},
$$

avec $d$ la dimension ($p = 3$ en 2D : deux coordonnées et une intensité ;
$p = 4$ en 3D) et

- $\kappa = \log M$ : **BIC** (critère d'information bayésien, Schwarz 1978) ;
- $\kappa = 2$ : **AIC** (Akaike 1974).

**Pourquoi $\log M$.** Le BIC approche la probabilité a posteriori du modèle à
$K$ sources : par la méthode de Laplace,
$\log p(m\mid K)\approx \log L(\hat\theta_K) - \tfrac{pK}{2}\log M$. Chaque
paramètre « coûte » $\tfrac12\log M$ en log-vraisemblance, soit $\log M$ en
$\chi^2$. Le BIC est **consistant** : si le vrai modèle fait partie des
candidats, il est choisi avec une probabilité qui tend vers 1 quand $M$
grandit ; l'AIC, lui, surestime $K$ avec une probabilité non nulle.

**Le seuil, concrètement.** Une source supplémentaire est acceptée si elle fait
baisser $\chi^2$ de plus de $p\,\log M$, soit $3\log 64\approx 12{,}5$ en 2D
avec 64 capteurs.

- Une **fausse source**, qui n'explique que du bruit, fait baisser $\chi^2$
  d'environ $p = 3$ en moyenne (loi $\chi^2_3$). Elle est presque toujours
  rejetée. (La recherche du meilleur emplacement augmente un peu cette baisse,
  d'où un taux de fausses détections faible mais non nul.)
- Une **vraie source** de rapport signal sur bruit
  $\mathrm{SNR} = I\,\lVert\varphi_w(z)\rVert$ fait baisser $\chi^2$ d'environ
  $\mathrm{SNR}^2 + 3$ (loi $\chi^2_3$ décentrée de paramètre
  $\mathrm{SNR}^2$). Elle est acceptée si $\mathrm{SNR}^2 + 3 > 12{,}5$, soit
  $\mathrm{SNR}\gtrsim 3$. La section 6.2 vérifie cette prédiction.

Il n'y a **aucun paramètre à régler** : $\kappa$ ne dépend que de $M$, et le
niveau de bruit $\sigma$ est celui du capteur (la section 6.4 étudie ce qui se
passe s'il est mal connu).

### 2.4 Algorithme glouton

```text
entrées : mesures m, écarts types s, capteurs, domaine D
K = 0, meilleur = (C(0), aucune source)
répéter
    r_w = résidu blanchi
    nouvelle position z* = argmax_z < phi_w(z) / ||phi_w(z)||, r_w >
                           (grille de D, puis L-BFGS-B depuis les meilleurs noeuds)
    intensités par moindres carrés positifs (NNLS)
    glissement : descente conjointe de toutes les positions et intensités
    suppression des intensités nulles, fusion des sources confondues
    si C(K) < C(meilleur) : meilleur = solution courante
    sinon, si C n'a pas baissé depuis 2 ajouts : arrêt
étape arrière : retirer une source tant que C baisse
sortie : meilleure solution (z, I, K)
```

- Le choix de la nouvelle position utilise la **corrélation normalisée**
  (noyau $\varphi_w/\lVert\varphi_w\rVert$, c'est-à-dire $\alpha = 1$ ; en 2D,
  il faut pour cela passer un `Setup` normalisé, `make_setup(normalize=True)`,
  ce que font tous les scripts ; `greedy_ml_nd` normalise toujours) : sans
  normalisation, le maximum serait toujours collé à un capteur, où
  $\lVert\varphi\rVert$ explose. C'est le seul endroit où la normalisation
  intervient.
- La **patience de 2 ajouts** évite de s'arrêter sur un palier ; l'**étape
  arrière** corrige les ajouts inutiles (par exemple une source placée au
  milieu de deux vraies sources, puis rendue inutile par les suivantes).
- Coût : $K_{\max}$ ajouts, chacun avec une descente conjointe sur $pK$
  variables ; 0,1 à 0,5 s en 2D pour 1 à 5 sources, 1 à 5 s en 3D.

### 2.5 Borne de Cramér–Rao

La borne de Cramér–Rao (CRB) donne l'erreur minimale de tout estimateur sans
biais. Avec $\theta = (I_1,\dots,I_K, z_1,\dots,z_K)$ et la matrice des
dérivées du modèle

$$
\mathbf{D} = \Big[\varphi(z_1),\dots,\varphi(z_K)\;\Big\vert\;I_1\nabla\varphi(z_1),\dots,I_K\nabla\varphi(z_K)\Big]\in\mathbb{R}^{M\times pK},
$$

la matrice de Fisher vaut $F = \mathbf{D}^{\top}\operatorname{diag}(s_i^{-2})\,\mathbf{D}$
et, pour tout estimateur sans biais,

$$
\mathbb{E}\,\lVert\hat z_k - z_k\rVert^2 \;\ge\; \sum_{j=1}^{d}\big(F^{-1}\big)_{z_k^{(j)} z_k^{(j)}} =: \mathrm{CRB}_k^2 .
$$

Dans les tableaux, « CRB » désigne $\max_k\mathrm{CRB}_k$ (la source la moins
bien déterminée). Deux propriétés guident la lecture des résultats :

- la CRB est **proportionnelle au bruit** ($\sigma$, donc à $I_c$) et
  décroît comme $1/\sqrt M$ avec le nombre de capteurs ;
- le maximum de vraisemblance est **asymptotiquement efficace** : quand le
  bruit est petit, son erreur atteint la CRB. Si une méthode atteint la CRB,
  **on ne peut pas faire mieux en changeant d'algorithme** ; il faut changer
  les données.

Si $F$ est singulière, le problème n'est pas identifiable (par exemple un seul
cercle de capteurs en 3D, section 6.7) : la CRB est infinie.

### 2.6 Généralisation en dimension quelconque (3D)

Rien dans les sections 2.1 à 2.5 ne dépend de la dimension. `ml3d.py` reprend
la méthode pour des sources dans $\mathbb{R}^d$ :

- noyau `InverseSquareKernel` du projet (il accepte déjà des capteurs en
  dimension quelconque) ;
- $p = d + 1$ paramètres par source dans le BIC ;
- grille de recherche de $D$ en $d$ dimensions (pas = côté / 20 en 3D, soit
  $21^3 = 9261$ nœuds) ;
- même descente conjointe, même étape arrière, même CRB.

**Identifiabilité en 3D.** Avec des sources en 3D et un seul cercle de
capteurs horizontal, une source au-dessus du plan du cercle et sa symétrique
en dessous donnent exactement les mêmes mesures : le problème n'est pas
identifiable, et $F$ est singulière. Il faut des capteurs non coplanaires
(deux ou trois cercles à des hauteurs différentes, ou une sphère). Le test
`test_single_ring_is_not_identifiable` le vérifie.

En 2D, `greedy_ml_nd` donne exactement la même solution que `greedy_ml` et la
même borne que `cramer_rao` (test `test_matches_2d_implementation`).

## 3. Implémentation

| fichier | contenu |
|---|---|
| `ml.py` | `greedy_ml` (2D, utilise `Setup`), `cramer_rao`, `relative_noise_std` |
| `ml3d.py` | `greedy_ml_nd`, `cramer_rao_nd`, `match_nd`, `success_nd` (dimension quelconque) |
| `kernel.py` | `Setup` : blanchiment (`with_weights`), hauteur par capteur (`heights`), `make_cylinder_setup` |
| `tester_ml.py` | **script pour tester la méthode à la main** sur un tirage (2D ou 3D) |
| `experiments_projet.py` | comparaison avec Prony et Frank-Wolfe sur le scénario du projet |
| `experiments_ml.py` | études propres à ML + BIC : $\sigma$ mal connu, détection, résolution, nombre de capteurs |
| `experiments_cylindre.py` | capteurs sur la paroi d'un cylindre, $K$ connu |
| `experiments_ml3d.py` | sources en 3D |
| `tests/test_ml.py`, `tests/test_ml3d.py`, `tests/test_cylinder.py` | tests pytest |

**Utilisation en 2D :**

```python
from sparse_sources import greedy_ml, cramer_rao, relative_noise_std, make_setup

S = make_setup(h=0.0, normalize=True)       # capteurs du projet
s = relative_noise_std(y_brut, sigma)       # s_i = sigma * y_i (source centrale comprise)
r = greedy_ml(m, s, setup=S)                # m : mesures, source centrale retirée
r["z"], r["I"], r["K"], r["chi2"] / S.M     # positions, intensités, K, chi2 / M
cramer_rao(r["z"], r["I"], s, S)["pos_std"] # précision attendue de chaque source
```

**Utilisation en 3D :**

```python
from sparse_sources.ml3d import greedy_ml_nd, cramer_rao_nd
r = greedy_ml_nd(m, s, sensors, box)        # sensors (M, 3), box : Box 3D
```

**Paramètres** (valeurs par défaut conseillées) : `criterion="bic"` ;
`k_max=10` ; `patience=2` ; `backward=True`. Pour imposer $K$ :
`k_max=K, criterion=0.0, backward=False`.

## 4. Tests

### 4.1 Tests automatiques (pytest)

`pytest sparse_sources/tests` lance 86 tests (environ 70 s), dont 24
consacrés à ML + BIC et à ses extensions. Tous passent.

**`tests/test_ml.py`** (2D, 8 tests) :

| test | ce qui est vérifié |
|---|---|
| `test_whitened_derivatives` | dérivées du noyau blanchi et normalisé contre différences finies |
| `test_weights_api` | validation des poids, blanchiment, invariance de $\psi$ par des poids constants |
| `test_greedy_ml_two_sources` ($\sigma = 10^{-4}$ et $10^{-3}$) | 2 sources avec la source centrale : $K$ exact, erreur inférieure à 5 CRB, $\chi^2$ compatible avec le bruit |
| `test_greedy_ml_three_sources_low_noise` | 3 sources tirées, $\sigma = 10^{-5}$ : $K$ exact, positions précises |
| `test_greedy_ml_options` | critères AIC et $\kappa$ numérique, candidats multiples, départ imposé (points hors de $D$ ignorés) |
| `test_greedy_ml_pure_noise` | bruit seul : au plus une source |
| `test_cramer_rao_scaling_and_efficiency` | CRB proportionnelle au bruit ; sur 30 tirages, erreur rms du ML entre 0,6 et 1,5 fois la CRB (**efficacité**) |

**`tests/test_ml3d.py`** (dimension quelconque, 10 tests) :

| test | ce qui est vérifié |
|---|---|
| `test_box_grid` | grille de $D$ : bords compris, pas respecté |
| `test_recovers_three_sources_3d` (sphère, deux cercles) | 3 sources en 3D, $\sigma = 10^{-4}$ : $K$ exact, erreur inférieure à 5 CRB, $\chi^2$ compatible avec le bruit |
| `test_with_central_source_3d` | source centrale de 5000 retirée, 2 sources retrouvées |
| `test_noise_only_gives_no_source` | bruit seul : au plus une source |
| `test_imposed_k` | $\kappa = 0$, `k_max` $= K$ : exactement $K$ sources |
| `test_single_ring_is_not_identifiable` | un seul cercle en 3D : CRB infinie |
| `test_matches_2d_implementation` | en 2D, même solution que `greedy_ml` (à $10^{-4}$ près) et même CRB que `cramer_rao` |
| `test_cramer_rao_scales_with_noise` | CRB exactement proportionnelle au bruit |
| `test_match_nd` | appariement hongrois et erreurs |

**`tests/test_cylinder.py`** (capteurs à plusieurs hauteurs, 6 tests) :
noyau capteur par capteur $1/(\lVert x_i - z\rVert^2 + h_i^2)$, un seul
étage identique à l'ancien noyau, équivalence des hauteurs $+h$ et $-h$,
dérivées ($\alpha = 0$ et $1$), conservation des hauteurs par
`with_power`, `with_normalization` et `with_weights`, validation des entrées.

### 4.2 Tester la méthode à la main : `tester_ml.py`

```bash
python sparse_sources/tester_ml.py                          # 2D, 3 sources, sigma = 1e-3
python sparse_sources/tester_ml.py --sources 5 --sigma 1e-4 --graine 7
python sparse_sources/tester_ml.py --positions "2,4,30;7,15,70" --sigma 1e-3
python sparse_sources/tester_ml.py --sans-centrale --comparer
python sparse_sources/tester_ml.py --dim 3 --capteurs sphere --sources 3
```

Le script tire un jeu de sources (exactement comme `scripts/reconstruct.py`
en 2D), lance ML + BIC et affiche :

- les sources vraies avec leur CRB, puis les sources estimées ;
- $\chi^2/M$, qui doit être proche de 1 ;
- l'évolution du critère $C(K)$ à chaque ajout, ce qui montre où le BIC s'arrête ;
- l'erreur et le verdict (réussite : bon $K$ et erreur de position inférieure
  à $\tfrac14\min(d_{\min}, 4)$, le critère de tout le dossier) ;
- avec `--comparer`, les résultats de Prony et du Frank-Wolfe sur les mêmes
  mesures.

Il enregistre une figure dans `outputs/tester_ml/` (sources vraies, estimées,
cercles de rayon $2\,\mathrm{CRB}$). Exemple de sortie (2D, 3 sources,
$\sigma = 10^{-3}$, graine 0) :

```text
2D, 3 sources vraies, sigma = 0.001, source centrale : oui (5000, retirée), M = 64 capteurs

Sources vraies (et borne de Cramér-Rao sur la position) :
  position [6.37  5.396]  intensité   15.87  CRB 0.349
  position [0.41  0.331]  intensité   17.79  CRB 0.00415
  position [ 8.133 18.255]  intensité   66.34  CRB 0.0111

Maximum de vraisemblance + BIC : K = 3 (0.22 s), chi2 / M = 0.82
  position [0.406 0.322]  intensité   17.50
  position [ 8.14  18.269]  intensité   65.68
  position [5.813 5.721]  intensité   18.51

Évolution du critère : K=0: 735304.6, K=1: 238755.5, K=2: 997.1, K=3: 89.8,
                       K=4: 97.7, K=5: 102.9

Erreur de position max 0.645, erreur d'intensité relative max 0.167 -> RÉUSSITE
```

Le critère chute fortement jusqu'à $K = 3$, puis remonte : chaque source de
plus coûte $3\log 64\approx 12{,}5$ et ne fait baisser $\chi^2$ que de
quelques unités. La source la plus faible, au centre de $D$ (près de la
source centrale), est la moins bien localisée ; c'est exactement ce
qu'annonce sa CRB (0,35, contre 0,004 et 0,011 pour les deux autres).

## 5. Critère de réussite et protocole

Sauf mention contraire : sources tirées comme dans `config/` (positions
uniformes, intensités uniformes sur le simplexe de somme 100), source
centrale de 5000 ajoutée puis retirée, bruit relatif $\sigma$, mêmes mesures
pour toutes les méthodes. Une reconstruction est **réussie** si le nombre de
sources est exact et si l'erreur de position maximale (appariement hongrois)
est inférieure à $\tfrac14\min(d_{\min}, 4)$, où $d_{\min}$ est la plus petite
distance entre deux sources. Un tirage est **faisable** si
$2\,\mathrm{CRB} < \tfrac14\min(d_{\min}, 4)$ : une méthode optimale a alors de
bonnes chances de réussir.

## 6. Résultats

### 6.1 Comparaison avec Prony et le Frank-Wolfe (scénario du projet)

`experiments_projet.py`, 20 tirages par case, mêmes mesures pour toutes les
méthodes. Prony est la méthode de `scripts/reconstruct.py` ; le Frank-Wolfe
est le BLASSO pondéré ($\alpha = 1$, $\lambda = 1{,}5\,\sigma\lVert y\rVert$),
la meilleure variante du BLASSO trouvée dans le compte rendu général.

**Taux de réussite** (`projet_methodes_resume.csv`) :

| $n$ | $\sigma$ | Prony | Frank-Wolfe | **ML + BIC** | tirages faisables | ML + BIC sur les faisables |
|---|---|---|---|---|---|---|
| 1 | $10^{-4}$ | 100 % | 100 % | **100 %** | 100 % | 100 % |
| 1 | $10^{-3}$ | 15 % | 100 % | **100 %** | 100 % | 100 % |
| 1 | $10^{-2}$ | 0 % | 100 % | **100 %** | 85 % | 100 % |
| 2 | $10^{-4}$ | 85 % | 90 % | **95 %** | 85 % | 94 % |
| 2 | $10^{-3}$ | 15 % | 70 % | **75 %** | 80 % | 94 % |
| 2 | $3\times 10^{-3}$ | 0 % | 40 % | **75 %** | 50 % | 90 % |
| 2 | $10^{-2}$ | 5 % | 15 % | **30 %** | 20 % | 100 % |
| 3 | $10^{-4}$ | 55 % | 40 % | **80 %** | 75 % | 100 % |
| 3 | $3\times 10^{-4}$ | 25 % | 20 % | **65 %** | 35 % | 100 % |
| 3 | $10^{-3}$ | 10 % | 5 % | **40 %** | 25 % | 100 % |
| 5 | $10^{-4}$ | 5 % | 0 % | **50 %** | 20 % | 100 % |
| 5 | $3\times 10^{-4}$ | 5 % | 0 % | **15 %** | 5 % | 100 % |

**Erreur de position médiane et CRB** :

| $n$ | $\sigma$ | Prony | Frank-Wolfe | ML + BIC | CRB médiane |
|---|---|---|---|---|---|
| 1 | $10^{-3}$ | 0,059 | 0,023 | **0,022** | 0,025 |
| 1 | $10^{-2}$ | 0,84 | 0,23 | **0,23** | 0,25 |
| 2 | $10^{-4}$ | 0,014 | 0,035 | **0,014** | 0,016 |
| 2 | $10^{-3}$ | 0,32 | 0,29 | **0,13** | 0,16 |
| 2 | $3\times 10^{-3}$ | 1,6 | 0,66 | **0,36** | 0,49 |
| 3 | $10^{-4}$ | 0,14 | 0,72 | **0,15** | 0,22 |
| 5 | $10^{-4}$ | 0,88 | 1,6 | **0,60** | 1,04 |

À retenir :

1. ML + BIC est **la meilleure méthode dans toutes les cases**, ou à égalité
   avec la meilleure, et l'écart grandit avec le nombre de sources et le
   bruit (50 % contre 5 % et 0 % pour 5 sources à $\sigma = 10^{-4}$).
2. **Son erreur atteint la CRB** : 0,7 à 0,9 fois la CRB médiane pour 1 et
   2 sources. (La CRB du tableau porte sur la pire source, l'erreur médiane
   sur un tirage typique ; une erreur légèrement sous la CRB est donc normale.)
3. **Sur les tirages faisables, elle réussit dans 90 à 100 % des cas.** Ses
   échecs sont presque tous des tirages où la CRB dépasse la tolérance :
   aucune méthode ne pourrait les réussir de façon fiable.

**BIC contre AIC** (`projet_critere.csv`) : le BIC est meilleur dans toutes
les cases testées (par exemple 80 % contre 60 % pour 3 sources à
$\sigma = 10^{-4}$). L'AIC ajoute des sources parasites dans 10 à 20 % des
tirages ; les erreurs du BIC sont surtout des sources **manquantes** à fort
bruit, c'est-à-dire des sources trop faibles pour être vues (section 6.2).

**Effet de la source centrale** (3 sources, $\sigma = 10^{-3}$,
`projet_centrale_resume.csv`) :

| intensité centrale / 100 | ML + BIC | CRB médiane |
|---|---|---|
| 0 (pas de source centrale) | **100 %** | 0,052 |
| 5 | 70 % | 0,30 |
| 50 (projet) | 40 % | 2,2 |
| 500 | 0 % | 21 |

Sans source centrale, la méthode réussit toujours ; avec celle du projet, la
CRB est multipliée par environ 42. **La limite est le bruit de la source
centrale, pas l'algorithme.**

### 6.2 Seuil de détection : la théorie du BIC est vérifiée

`experiments_ml.py`, expérience `detection` : une seule source d'intensité
$I$ (0 à 20) à une position uniforme dans $D$, source centrale du projet,
$\sigma = 10^{-3}$, 40 tirages par intensité (`ml_detection.csv`,
`ml_detection.png`). La section 2.3 prédit que la source est retenue avec la
probabilité $P\big(\chi^2_3(\mathrm{SNR}^2) > 3\log M\big)$, avec
$\mathrm{SNR} = I\,\lVert\varphi(z)/s\rVert$.

| SNR | tirages | détectée ($K\ge 1$) | théorie | détectée et à moins de 1 |
|---|---|---|---|---|
| 0 à 1 | 38 | 3 % | 1 % | 0 % |
| 1 à 2 | 34 | 6 % | 6 % | 0 % |
| 2 à 3 | 21 | 29 % | 22 % | 0 % |
| 3 à 4 | 17 | 65 % | 63 % | 12 % |
| 4 à 6 | 32 | 94 % | 93 % | 25 % |
| 6 à 10 | 36 | 100 % | 100 % | 53 % |
| 10 à 20 | 46 | 100 % | 100 % | 89 % |
| plus de 20 | 96 | 100 % | 100 % | 100 % |

- **La probabilité de détection suit la théorie à quelques points près** : le
  seuil du BIC est bien $\mathrm{SNR}\approx 3$ (50 % de détection).
- **Fausses alarmes** : sur 40 tirages de bruit seul ($I = 0$), une seule
  source parasite (2 %). Le BIC ne crée presque jamais de source à partir du
  bruit.
- **Détecter n'est pas localiser.** Entre $\mathrm{SNR} = 3$ et 10, la source
  est vue mais sa position est imprécise : la CRB vaut environ $10/\mathrm{SNR}$
  (4,2 pour $I = 1$, 0,85 pour $I = 5$). Il faut $\mathrm{SNR}\gtrsim 20$ pour
  une erreur inférieure à 1 dans presque tous les cas.
- Dans le scénario du projet ($\sigma = 10^{-3}$), une source isolée
  d'intensité 2 est presque toujours détectée, mais il faut une intensité
  d'environ 10 pour la localiser à mieux que 1 près.

### 6.3 Résolution de deux sources proches

Expérience `resolution` : deux sources d'intensité 50 à distance $d$, centre
et orientation aléatoires dans $D$, source centrale du projet, 20 tirages
(`ml_resolution.csv`, `ml_resolution.png`).

| $d$ | $\sigma = 10^{-5}$ | $\sigma = 10^{-4}$ | $\sigma = 10^{-3}$ | CRB médiane ($\sigma = 10^{-3}$) |
|---|---|---|---|---|
| 0,5 | 80 % | 20 % | 10 % | 7,8 |
| 1 | **100 %** | 70 % | 15 % | 2,1 |
| 1,5 | 100 % | 90 % | 25 % | 1,5 |
| 2 | 100 % | **100 %** | 50 % | 0,91 |
| 3 | 100 % | 100 % | 90 % | 0,47 |
| 4 | 100 % | 100 % | **100 %** | 0,36 |
| 6 | 100 % | 100 % | 100 % | 0,20 |

- La **séparation minimale résolue** passe de 1 ($\sigma = 10^{-5}$) à 2
  ($\sigma = 10^{-4}$) puis 4 ($\sigma = 10^{-3}$). La CRB d'une paire croît
  très vite quand $d$ diminue (d'un facteur 2 à 4 chaque fois que $d$ est
  divisé par 2) : deux sources proches ont presque la même signature.
- Quand elle échoue, la méthode trouve **une seule source** au lieu de deux
  (85 % des tirages à $d = 0{,}5$, $\sigma = 10^{-3}$) : le BIC juge que la
  deuxième source n'apporte pas assez. C'est le comportement attendu, et il
  est préférable à l'invention de sources.
- L'erreur médiane reste **sous la CRB** de la pire source à toutes les
  distances (figure de droite).

### 6.4 Niveau de bruit mal connu

La méthode a besoin des écarts types $s_i = \sigma y_i$. Expérience
`sigma_faux` : la méthode reçoit $c\times\sigma$ au lieu de $\sigma$
(`ml_sigma_faux.csv`, `ml_sigma_faux.png`, 20 tirages).

| cas | $c = 0{,}5$ | $c = 0{,}8$ | $c = 1$ | $c = 1{,}25$ | $c = 2$ |
|---|---|---|---|---|---|
| 2 sources, $\sigma = 10^{-3}$ | 5 % ($K$ = 3,0) | 80 % | 75 % | 80 % | 80 % ($K$ = 1,8) |
| 3 sources, $\sigma = 10^{-4}$ | 25 % ($K$ = 4,1) | 70 % | 80 % | 75 % | 75 % ($K$ = 2,9) |
| 3 sources, $\sigma = 10^{-3}$ | 10 % ($K$ = 3,8) | 35 % | 40 % | 40 % | 35 % ($K$ = 2,4) |

(entre parenthèses : nombre moyen de sources estimé)

- **Surestimer le bruit est sans danger** : jusqu'à un facteur 2, la réussite
  ne bouge presque pas. Le seuil effectif devient plus exigeant
  ($\chi^2$ divisé par $c^2$) : on perd seulement les sources les plus
  faibles.
- **Sous-estimer le bruit est dangereux** : avec $c = 0{,}5$, $\chi^2$ est
  multiplié par 4, chaque petite structure du bruit semble significative, et
  la méthode ajoute en moyenne une source parasite.
- **Conseil** : en cas de doute, prendre $\sigma$ un peu trop grand. Le
  contrôle $\chi^2/M$ permet de vérifier le choix : avec le bon $\sigma$, il
  vaut environ $(M - pK)/M$, soit 0,86 pour 3 sources et 64 capteurs (0,78 à
  0,85 mesuré). Une valeur nettement supérieure à 1 indique un $\sigma$ trop
  petit ou une source manquante.

### 6.5 Nombre de capteurs

Expérience `capteurs` : même scénario, $M$ capteurs sur le même cercle,
20 tirages (`ml_capteurs.csv`, `ml_capteurs.png`).

| $M$ | 3 sources, $\sigma = 10^{-3}$ | 5 sources, $\sigma = 10^{-4}$ | 3 sources, $\sigma = 10^{-2}$ | temps (5 sources) |
|---|---|---|---|---|
| 32 | 40 % (CRB 3,7) | 35 % (CRB 1,7) | 0 % (CRB 36) | 0,6 s |
| 64 (projet) | 40 % (CRB 2,2) | 50 % (CRB 1,0) | 0 % (CRB 22) | 0,5 s |
| 256 | 60 % (CRB 1,1) | 65 % (CRB 0,52) | 10 % (CRB 11) | 1,0 s |
| 1024 | **70 %** (CRB 0,55) | **75 %** (CRB 0,26) | 15 % (CRB 5,5) | 4,5 s |

- La CRB est **divisée par 2 quand $M$ est multiplié par 4** : elle décroît
  comme $1/\sqrt M$, car les bruits des capteurs sont indépendants et se
  moyennent. ML + BIC en profite directement.
- Le gain est réel mais lent : pour diviser l'erreur par 10, il faut 100 fois
  plus de capteurs. À fort bruit ($\sigma = 10^{-2}$), même 1024 capteurs ne
  suffisent pas.
- Ce gain suppose des bruits **indépendants** d'un capteur à l'autre. Une
  part de bruit commune à tous les capteurs ne diminuerait pas.

### 6.6 Capteurs sur la paroi d'un cylindre ($K$ connu)

`experiments_cylindre.py` (`cylindre.csv`, `cylindre_resume.csv`,
`cylindre.png`). Question posée : avec des capteurs répartis sur la paroi
d'un cylindre (plusieurs étages au-dessus du plan des sources), le
Frank-Wolfe marche-t-il pour beaucoup de sources et beaucoup de bruit ?

**Protocole.** Même noyau $1/(\lVert x_i - z\rVert^2 + h_i^2)$, avec une
hauteur $h_i$ par capteur (`make_cylinder_setup`). Le cylindre a pour axe la
verticale du centre de $D$ et pour rayon celui du cercle du projet. Même
nombre total de capteurs pour toutes les géométries, $M = 60$ :

| géométrie | étages (hauteurs $h$) |
|---|---|
| cercle | 1 étage, $h = 0$ (comme le projet) |
| cylindre bas | $h = 0$ ; 2,5 ; 5 (20 capteurs par étage) |
| cylindre haut | $h = 0$ ; 2,5 ; 5 ; 7,5 ; 10 (12 par étage) |
| cylindre sans plan | $h = 2{,}5$ ; 5 ; 7,5 (aucun capteur dans le plan) |

Comme les sources sont dans le plan, un capteur à $-h$ mesure la même chose
qu'à $+h$ : un cylindre symétrique de part et d'autre du plan revient à ces
étages avec deux fois plus de capteurs. Les étages sont décalés en angle.

Sources : $K\in\{3, 5, 8\}$, **$K$ connu à l'avance**, positions uniformes
mais **espacées** (distance minimale 3), intensités uniformes sur le
simplexe, source centrale de 5000, bruit relatif, 15 tirages par case. Les
mêmes sources sont utilisées pour toutes les géométries.

- **Frank-Wolfe à $\lambda$ optimal pour $K$** : BLASSO pondéré ($\alpha = 1$),
  14 valeurs de $\lambda$ de $0{,}5\,\lambda_{\max}$ à
  $3\times 10^{-4}\,\lambda_{\max}$ ; parmi celles qui donnent exactement $K$
  sources, on garde celle de plus petit résidu (après débiaisage). Si aucune
  ne donne $K$ sources (29 % des cas), on prend la première qui en donne plus
  et on garde les $K$ plus intenses.
- **ML à $K$ imposé** : `greedy_ml(k_max=K, criterion=0, backward=False)`.

**Taux de réussite (Frank-Wolfe / ML), condition de Duval–Peyré vraie, CRB
médiane** :

| $K$ | $\sigma$ | cercle | cylindre bas | cylindre haut | cylindre sans plan |
|---|---|---|---|---|---|
| 3 | $10^{-4}$ | **67 %** / 87 % | 53 % / 87 % | 53 % / **93 %** | 53 % / 87 % |
| 3 | $10^{-3}$ | 20 % / **53 %** | 20 % / 33 % | 20 % / 27 % | 7 % / 27 % |
| 3 | $10^{-2}$ | 7 % / 7 % | 7 % / 0 % | 0 % / 0 % | 7 % / 0 % |
| 5 | $10^{-4}$ | 0 % / **33 %** | 7 % / 20 % | 0 % / 7 % | 0 % / 27 % |
| 5 | $\ge 10^{-3}$ | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| 8 | tous | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % | 0 % / 0 % |
| condition vraie, $K$ = 3 / 5 / 8 | | 47 / 0 / 0 % | 47 / 7 / 0 % | 53 / 7 / 0 % | 47 / 7 / 0 % |
| CRB médiane, $K$ = 3 / 5 / 8 ($\sigma = 10^{-4}$) | | **0,16** / 4,5 / 19 | 0,20 / **2,5** / **6,8** | 0,29 / 3,6 / 11 | 0,28 / 3,7 / 17 |

**Réponse : non.** Le cylindre n'aide pas le Frank-Wolfe, ni pour beaucoup
de sources, ni pour beaucoup de bruit.

1. **Le Frank-Wolfe n'est jamais meilleur sur un cylindre que sur le cercle**,
   même avec le $\lambda$ choisi en connaissant $K$ et des sources espacées.
   Il plafonne à 67 % pour 3 sources à très faible bruit et tombe à 0 % dès 5
   sources, quelle que soit la géométrie.
2. **La condition de Duval–Peyré reste fausse** à partir de 5 sources (0 à
   7 % des tirages). Les étages en hauteur améliorent un peu la marge (marge
   relative médiane pour 3 sources : $-40$ pour le cercle, $-17$ pour le
   cylindre bas, $-9$ sans capteur dans le plan, $+1{,}0$ pour le cylindre
   haut), car le signal ne peut plus exploser près d'un
   capteur. Mais ce gain ne se traduit pas en réussite.
3. **Monter les capteurs dégrade l'information à petit nombre de sources** :
   un capteur à la hauteur $h$ reçoit moins de signal, et le reçoit plus
   « flou » (toutes les sources sont à peu près à la distance $h$). Pour
   3 sources, la CRB passe de 0,16 (cercle) à 0,20, 0,28 et 0,29. À 5 ou
   8 sources, la diversité des points de vue aide un peu (CRB divisée par 2
   à 3 avec le cylindre bas), mais la CRB reste très au-dessus de la
   tolérance : le problème reste insoluble pour toutes les méthodes.
4. **ML est au moins aussi bon que le Frank-Wolfe dans toutes les cases**
   (87 à 93 % contre 53 à 67 % pour 3 sources à $\sigma = 10^{-4}$).

Ce qui limite, c'est le bruit de la source centrale et le nombre de sources,
pas la position des capteurs en hauteur. Pour un vrai gain, il faudrait des
capteurs **plus proches des sources** (à l'intérieur de $D$) ou un bruit
plus faible, pas des capteurs plus loin du plan.

### 6.7 Généralisation en 3D

`experiments_ml3d.py` (`ml3d.csv`, `ml3d_resume.csv`, `ml3d.png`) : sources
dans le cube $[0,10]^3$, espacées d'au moins 3, intensités uniformes sur le
simplexe (somme 100), source centrale de 5000 au centre du cube, bruit
relatif $\sigma$, $M = 160$ capteurs à 1,5 fois le rayon de la sphère
circonscrite (géométrie de `scripts/duval_peyre_3d.py`), 8 tirages par case
(un tirage = 12,5 %). Comparaison avec le BLASSO 3D du projet
(`particle_sources.duval_peyre.blasso`, noyau brut, $\lambda = 4\lVert
w\rVert\,\lVert\Phi\rVert$ et seuil de 0,5 % comme dans
`scripts/duval_peyre_3d.py`).

**Taux de réussite de ML + BIC** (le BLASSO du projet est à 0 % dans toutes
les cases sauf une source à $\sigma = 10^{-3}$, 12 %) :

| $K$ | $\sigma$ | deux cercles | trois cercles | sphère | CRB médiane (sphère) |
|---|---|---|---|---|---|
| 1 | $10^{-4}$ | 100 % | 100 % | **100 %** | 0,006 |
| 1 | $10^{-3}$ | 100 % | 100 % | **100 %** | 0,059 |
| 1 | $10^{-2}$ | 88 % | 88 % | **100 %** | 0,59 |
| 2 | $10^{-4}$ | 100 % | 100 % | **100 %** | 0,049 |
| 2 | $10^{-3}$ | 62 % | 62 % | **88 %** | 0,49 |
| 2 | $10^{-2}$ | 0 % | 0 % | 0 % | 4,9 |
| 3 | $10^{-4}$ | 62 % | 75 % | **75 %** | 0,23 |
| 3 | $10^{-3}$ | 12 % | 25 % | 25 % | 2,3 |
| 5 | $10^{-4}$ | 0 % | 12 % | **50 %** | 1,2 |
| 5 | $10^{-3}$ | 0 % | 0 % | 0 % | 12 |

**Erreur médiane et CRB** (une source) : 0,044 contre 0,059 (sphère,
$\sigma = 10^{-3}$), 0,067 contre 0,085 (trois cercles), 0,074 contre 0,081
(deux cercles). L'erreur vaut 0,75 à 0,9 fois la CRB, comme en 2D : **la
méthode reste efficace en 3D.**

À retenir :

1. **La méthode se généralise sans modification** : mêmes étapes, $p = 4$
   paramètres par source dans le BIC, mêmes garanties (erreur au niveau de la
   CRB, presque pas de fausses sources).
2. **La sphère est la meilleure disposition** : à nombre de capteurs égal, sa
   CRB est 1,4 fois (une source) à plus de 20 fois (cinq sources) plus petite
   que celle des cercles (par exemple 1,2 contre 3,0 et 29 pour 5 sources à
   $\sigma = 10^{-4}$), car elle voit les
   sources sous toutes les directions.
3. **Le problème est plus dur qu'en 2D** : la source centrale est au milieu
   du cube, entourée par les sources, et une source a 4 paramètres au lieu de
   3. Avec la source centrale, on retrouve de façon fiable 1 ou 2 sources à
   $\sigma = 10^{-3}$, 2 ou 3 à $\sigma = 10^{-4}$. Les échecs suivent la CRB :
   presque aucun tirage n'est faisable au-delà.
4. **Le BLASSO du projet échoue partout**, avec 4 à 14 sources estimées : noyau
   brut ($\alpha = 0$), donc le même défaut qu'en 2D (fausses sources du côté
   des capteurs). Ce n'est pas une comparaison à $\lambda$ optimisé, mais
   elle reprend exactement le réglage du script du projet.
5. **Coût** : 0,2 à 0,7 s pour une source, 1 à 2 s pour deux, 3 s pour trois,
   7 à 11 s pour cinq (grille de $21^3$ nœuds, 160 capteurs).

### 6.8 Beaucoup de sources (jusqu'à 50)

Expérience `beaucoup` de `experiments_ml.py` (`ml_beaucoup_directions.csv`,
`ml_beaucoup_ml.csv`, `ml_beaucoup.png`, 5 tirages par cas). Sources tirées
comme dans `config/` : positions uniformes, intensités uniformes sur le
simplexe de somme 100.

**Ce que les mesures contiennent.** Avec $K$ sources, il y a $3K$ paramètres
à retrouver. On compte combien de « directions » de cet espace de paramètres
les capteurs mesurent vraiment : ce sont les valeurs singulières de la
matrice des dérivées du modèle (colonnes normalisées) qui dépassent
$10^{-6}$ fois la plus grande. Le calcul est fait **sans bruit** : un bruit
réaliste ($\sigma = 10^{-3}$) en laisserait encore moins.

| sources $K$ | paramètres $3K$ | directions mesurables, 64 capteurs | 1024 capteurs | plus petite distance entre sources |
|---|---|---|---|---|
| 1 | 3 | 3 | 3 | |
| 3 | 9 | 9 | 9 | 4,3 |
| 5 | 15 | 15 | 15 | 2,7 |
| 10 | 30 | 29 | 29 | 0,74 |
| 20 | 60 | 46 | 47 | 0,62 |
| 30 | 90 | 50 | 56 | 0,24 |
| 50 | 150 | 56 | 71 | 0,15 |

- Jusqu'à 5 sources, tous les paramètres sont mesurés. Au-delà, le nombre de
  directions mesurables **plafonne autour de 50 à 70**, même avec 16 fois
  plus de capteurs.
- La raison : vu depuis le cercle, le signal en $1/r^2$ est très lisse. Les
  capteurs voient bien les grandes structures (intensité totale, barycentre,
  étalement), mais les détails fins sont écrasés. Des capteurs
  supplémentaires sur le même cercle répètent presque la même information.
- Avec 50 sources, il y a de plus 150 inconnues pour 64 mesures : le problème
  n'a pas de solution unique, même sans bruit. Et les sources tirées au
  hasard sont à environ 0,15 les unes des autres, alors que même deux
  sources isolées ne sont séparées qu'à partir d'une distance de 2 à 4 avec
  le bruit du projet.

**Ce que ML + BIC retrouve** (64 capteurs, source centrale, médiane sur
5 tirages) :

| vraies sources | $\sigma$ | sources estimées | vraies sources retrouvées à moins de 0,5 | temps |
|---|---|---|---|---|
| 10 | $10^{-4}$ | 6 | 5 | 1,2 s |
| 10 | $10^{-3}$ | 4 | 2 | 0,3 s |
| 20 | $10^{-4}$ | 8 | 4 | 4,6 s |
| 20 | $10^{-3}$ | 6 | 3 | 0,9 s |
| 50 | $10^{-4}$ | 10 | 6 | 14 s |
| 50 | $10^{-3}$ | 7 | 4 | 1,5 s |

La méthode ne s'effondre pas : elle garde **les quelques sources que les
données permettent de voir** (les plus intenses et les plus isolées) et
n'invente pas les autres. Mais **aucune méthode ne peut retrouver 50 sources**
avec ces mesures. Pour s'en approcher, il faudrait changer les mesures :
capteurs proches des sources (à l'intérieur de $D$), bruit beaucoup plus
faible, ou information a priori (par exemple des positions possibles
connues, ce qui ne laisse que les intensités à estimer). On peut aussi viser
des grandeurs globales (intensité totale, barycentre, carte de densité
grossière) plutôt que chaque source.

## 7. Limites

1. **Le bruit doit être connu.** La méthode utilise $s_i = \sigma y_i$. Un
   $\sigma$ sous-estimé d'un facteur 2 crée des sources parasites
   (section 6.4). Si $\sigma$ est inconnu, on peut l'estimer par
   $\chi^2/(M - pK)$ sur une première reconstruction, ou le surestimer
   légèrement.
2. **Modèle gaussien.** La vraisemblance suppose un bruit gaussien
   indépendant. Pour un bruit de comptage (loi de Poisson, option
   `poisson_counts` du projet), il faudrait remplacer $\chi^2$ par la
   déviance de Poisson ; le reste de la méthode est inchangé.
3. **Optimum local.** La descente conjointe n'est pas convexe. Le départ
   glouton est bon en pratique (en partant des vraies positions, on retrouve
   la même solution dans les cas vérifiés), mais rien ne garantit l'optimum
   global, surtout pour des sources proches. Les variantes à plusieurs
   candidats ou à départ depuis Prony n'ont rien apporté (compte rendu
   général, section 5.5).
4. **Sources faibles ou proches.** Une source est vue si son SNR dépasse
   environ 3, et localisée à 1 près si son SNR dépasse environ 20 ; deux
   sources plus proches que la séparation minimale (section 6.3) sont
   fusionnées en une seule. C'est une limite des données, que la CRB
   quantifie.
5. **Coût.** Il croît avec $K$ (un glissement à chaque ajout) : moins d'une
   demi-seconde en 2D jusqu'à 5 sources, mais 4,5 s avec 1024 capteurs et 7
   à 11 s en 3D pour 5 sources. Pour $K\approx 20$ (configuration par défaut
   du projet), il faudrait limiter le glissement aux sources voisines de la
   nouvelle.
6. **Beaucoup de sources.** Au-delà d'une dizaine de sources, les capteurs
   du cercle ne mesurent qu'environ 50 à 70 directions de l'espace des
   paramètres (section 6.8) : 50 sources sont impossibles à retrouver,
   quelle que soit la méthode.
7. **Configuration par défaut du projet** ($N\sim\mathrm{Poisson}(20)$,
   $\sigma = 1\,\%$) : la CRB conjointe est comprise entre $10^4$ et $5\times 10^5$.
   Aucune méthode ne peut retrouver ces sources ; ML + BIC en retient 1 à 5,
   les plus intenses (compte rendu général, section 5.5).

## 8. Recommandations

| question | réponse |
|---|---|
| quelle méthode utiliser ? | `greedy_ml` (2D) ou `greedy_ml_nd` (3D), critère BIC |
| quel bruit lui donner ? | $s_i = \sigma y_i$ sur les mesures brutes, source centrale comprise (`relative_noise_std`) ; en cas de doute, un $\sigma$ un peu trop grand |
| comment savoir si le résultat est fiable ? | $\chi^2/M$ proche de $(M - pK)/M$, et CRB de chaque source (`cramer_rao`) petite devant la distance entre sources |
| $K$ est connu à l'avance ? | `k_max=K, criterion=0.0, backward=False` |
| comment améliorer les résultats ? | réduire le bruit de la source centrale (c'est le facteur dominant), ajouter des capteurs (gain en $1/\sqrt M$), en 3D préférer une sphère de capteurs |
| à éviter | BLASSO non pondéré ($\alpha = 0$), AIC, capteurs en hauteur, un seul cercle en 3D |

## 9. Pistes

1. **Estimer $\sigma$ en même temps** que les sources (maximum de
   vraisemblance en $\sigma$ : $\hat\sigma^2 = \frac{1}{M}\sum_i r_i^2/y_i^2$),
   pour ne plus dépendre d'une valeur fournie.
2. **Bruit de Poisson** : remplacer $\chi^2$ par la déviance de Poisson
   (même algorithme glouton, même BIC).
3. **Glissement local** pour les grands $K$ : ne réoptimiser que les
   sources proches de la dernière ajoutée, puis un glissement global à la
   fin.
4. **Intervalles de confiance** : la matrice $F^{-1}$ évaluée en la solution
   donne une ellipse d'incertitude par source ; l'afficher dans
   `tester_ml.py`.
5. **Plan d'expérience** : utiliser la CRB pour choisir la position des
   capteurs avant de les construire (par exemple des capteurs à l'intérieur
   de $D$, plus proches des sources).

## 10. Reproduire

```bash
pip install -e ".[dev]" scikit-learn
pytest sparse_sources/tests                      # tous les tests
python sparse_sources/experiments_projet.py      # comparaison des méthodes (environ 6 min)
python sparse_sources/experiments_ml.py          # études de ML + BIC
python sparse_sources/experiments_cylindre.py    # capteurs sur un cylindre
python sparse_sources/experiments_ml3d.py        # sources en 3D
python sparse_sources/tester_ml.py --help        # tester un tirage
```

Pour obtenir une version LaTeX ou PDF du rapport :

```bash
pandoc RAPPORT_ML_BIC.md -s -o RAPPORT_ML_BIC.tex
pandoc RAPPORT_ML_BIC.md -s --pdf-engine=xelatex -o RAPPORT_ML_BIC.pdf
```

## Références

- G. Schwarz, *Estimating the dimension of a model*, The Annals of Statistics,
  1978 (critère BIC).
- H. Akaike, *A new look at the statistical model identification*, IEEE
  Transactions on Automatic Control, 1974 (critère AIC).
- S. M. Kay, *Fundamentals of Statistical Signal Processing, Volume I:
  Estimation Theory*, Prentice Hall, 1993 (borne de Cramér–Rao, efficacité du
  maximum de vraisemblance).
- S. M. Kay, *Fundamentals of Statistical Signal Processing, Volume II:
  Detection Theory*, Prentice Hall, 1998 (détection, loi du $\chi^2$ décentrée).
- V. Duval, G. Peyré, *Exact support recovery for sparse spikes deconvolution*,
  Foundations of Computational Mathematics, 2015 (BLASSO, condition de source
  non dégénérée).
- Q. Denoyelle, V. Duval, G. Peyré, E. Soubies, *The sliding Frank-Wolfe
  algorithm and its application to super-resolution microscopy*, Inverse
  Problems, 2019.

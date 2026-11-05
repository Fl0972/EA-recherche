---
title: "Localisation de sources ponctuelles : LASSO, BLASSO, maximum de vraisemblance et hypothèses de Duval–Peyré"
subtitle: "Compte rendu du dossier sparse_sources"
lang: fr
---

# Compte rendu : localisation de sources ponctuelles

Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/`. Les
figures citées sont produites par les scripts `experiments.py`,
`experiments_1r2.py` et `experiments_projet.py` dans `sparse_sources/results/`.
Les tableaux chiffrés viennent des fichiers CSV du même dossier.

Les formules sont écrites en LaTeX (`$...$` et `$$...$$`). Le document se lit
directement dans un aperçu Markdown (VS Code, GitHub) et se convertit en LaTeX
ou en PDF avec Pandoc (voir la section 10).

## 0. Résumé

1. **Outils implémentés** : un LASSO positif sur grille (FISTA, sklearn,
   ensemble actif), un BLASSO sans grille par Frank-Wolfe glissant, une fonction
   `check_hypotheses` qui vérifie la condition de source non dégénérée de
   Duval–Peyré, et une méthode de **maximum de vraisemblance gloutonne** avec
   choix du nombre de sources par BIC (`greedy_ml`), accompagnée de la **borne
   de Cramér–Rao** (`cramer_rao`). Le code est couvert par 70 tests pytest.
2. **Le BLASSO standard échoue toujours avec le noyau du projet**
   $G(x,z) = 1/\lVert x-z\rVert^2$ (capteurs dans le plan) : la condition de
   Duval–Peyré est fausse dans 100 % des tirages. La cause est mathématique :
   $G(x,\cdot)$ est sous-harmonique et $\lVert\varphi\rVert$ varie d'un facteur
   215 sur le domaine.
3. **Le BLASSO pondéré** (pénalité $\lambda\sum_k I_k\lVert\varphi(z_k)\rVert^{\alpha}$,
   avec $\alpha\approx 1$) corrige ce défaut : 1 ou 2 sources sont retrouvées dans
   95 à 100 % des tirages, et `check_hypotheses` prédit la réussite (94 % de
   réussite si la condition est vraie, 2 à 6 % sinon).
4. **Dans le scénario réel du projet** (grosse source centrale d'intensité 5000
   retirée, bruit relatif), la meilleure méthode est le **maximum de vraisemblance
   glouton + BIC**. Elle bat Prony et le BLASSO dans presque toutes les cases,
   et son erreur est proche de la borne de Cramér–Rao, qui est la meilleure
   précision possible (section 5.5 ; tableau de choix des méthodes en
   section 11).
5. **La limite principale est physique, pas algorithmique** : le bruit de la
   source centrale reste dans les mesures après soustraction. Il multiplie
   l'erreur minimale possible par un facteur d'environ 40 par rapport au même
   problème sans source centrale (3 sources, $\sigma = 10^{-3}$ : 100 % de
   réussite sans source centrale, 40 % avec). Dans la configuration par
   défaut ($N\sim\mathrm{Poisson}(20)$, $\sigma = 1\,\%$), seules 0 à 4
   sources par tirage sont détectables, quelle que soit la méthode.

## 1. Cadre et notations

**Domaine et capteurs.** Les sources sont dans $D = [0,10]\times[0,20]$
(`config/simulation.yaml`). Les $M = 64$ capteurs sont équirépartis sur le
cercle circonscrit à $D$, de centre $c = (5,10)$ et de rayon
$R = \sqrt{125}\approx 11{,}18$, dans le plan des sources
(`config/sensors.yaml`). Ce cercle passe par les coins de $D$ : certains
capteurs sont à environ $0{,}3$ d'un coin.

**Sources et mesures.** Les sources forment une mesure positive

$$
s = \sum_{k=1}^{K} I_k\,\delta_{z_k}, \qquad I_k > 0,\quad z_k \in D,
$$

avec $K$ inconnu. Le noyau du projet (`particle_sources.model.field.intensity_at`)
est

$$
G(x_i, z) = \frac{1}{\lVert x_i - z\rVert^2 + h^2},
\qquad
\varphi(z) = \big(G(x_1,z),\dots,G(x_M,z)\big) \in \mathbb{R}^M .
$$

On a $h = 0$ dans le projet (paramètre `eps`). Un capteur placé à la hauteur
$h$ au-dessus du plan voit exactement ce noyau ; $h$ sert donc à étudier des
capteurs hors du plan. Les mesures sont

$$
m = A(s) + w, \qquad A(s) = \sum_{k=1}^{K} I_k\,\varphi(z_k).
$$

**Deux modèles de bruit.**

- Bruit additif (études génériques, sections 5.1 à 5.3) :
  $w \sim \mathcal{N}(0,\sigma^2 I_M)$, avec
  $\sigma = \text{niveau}\times\mathrm{rms}(A(s))$.
- Bruit relatif du projet (sections 5.4 et 5.5) :
  $y_i = y_i^{\mathrm{exact}}\,(1+\sigma\varepsilon_i)$, avec
  $\varepsilon_i\sim\mathcal{N}(0,1)$. Les mesures brutes $y$ contiennent la
  source centrale $I_c\,\varphi(c)$, avec $I_c = 50\times 100 = 5000$. Le
  projet la retire : $m = y - I_c\,\varphi(c)$. Le bruit, lui, reste, avec un
  écart type par capteur

$$
s_i = \sigma\, y_i \approx \sigma\Big(\frac{I_c}{R^2} + A(s)_i\Big) = \sigma\,(40 + A(s)_i).
$$

  La source centrale est au centre du cercle des capteurs : elle apporte
  exactement $I_c/R^2 = 40$ à chaque capteur. Le signal utile vaut environ 7 en
  moyenne quadratique. Le bruit est donc dominé par la source centrale.

### 1.1 Problèmes d'optimisation

**BLASSO positif** (sans grille) :

$$
\min_{s\ge 0}\; \tfrac12\,\lVert m - A(s)\rVert_2^2 + \lambda\,\lVert s\rVert_{\mathrm{TV}},
\qquad \lVert s\rVert_{\mathrm{TV}} = \sum_k I_k .
$$

**LASSO positif sur grille** (positions restreintes aux nœuds $y_1,\dots,y_P$) :

$$
\min_{c\in\mathbb{R}^P_{+}}\; \tfrac12\,\lVert m - \mathbf{G}c\rVert_2^2 + \lambda\sum_{j=1}^{P} c_j,
\qquad \mathbf{G}_{ij} = G(x_i, y_j)\in\mathbb{R}^{M\times P}.
$$

**BLASSO pondéré** (ajout par rapport à la demande) :

$$
\min_{K,\,I\ge 0,\,z}\; \tfrac12\,\Big\lVert m - \sum_k I_k\varphi(z_k)\Big\rVert_2^2
+ \lambda\sum_k I_k\,\lVert\varphi(z_k)\rVert^{\alpha}.
$$

### 1.2 Ce que représente $\alpha$

$\alpha \ge 0$ est **l'exposant de pondération de la pénalité**. Le BLASSO
pondéré est exactement le BLASSO standard pour le noyau modifié

$$
\psi(z) = \frac{\varphi(z)}{\lVert\varphi(z)\rVert^{\alpha}},
\qquad \tilde I_k = I_k\,\lVert\varphi(z_k)\rVert^{\alpha},
$$

car $\sum_k \tilde I_k\,\psi(z_k) = \sum_k I_k\,\varphi(z_k)$ : le modèle physique
et les mesures sont inchangés, seule la pénalité change.

- $\alpha = 0$ : BLASSO standard ; chaque unité d'intensité coûte $\lambda$,
  où que soit la source.
- $\alpha = 1$ : noyau normalisé, $\lVert\psi(z)\rVert = 1$ partout. Une source
  coûte $\lambda$ par unité de **signal** produit, et non par unité de masse.
- $\alpha > 1$ : on favorise encore plus les sources loin des capteurs.

Pourquoi c'est nécessaire avec le noyau en $1/r^2$ : $\lVert\varphi(z)\rVert$
vaut $0{,}064$ au centre de $D$ et $13{,}7$ près d'un coin, soit un rapport de
215. Avec $\alpha = 0$, une source près d'un capteur produit 215 fois plus de
signal par unité de pénalité ; le BLASSO remplace alors une source centrale par
des sources parasites au bord. Avec $\alpha = 1$, on montre par Cauchy–Schwarz
qu'une source seule est toujours bien identifiée (section 5.2). La valeur
conseillée est $\alpha = 1$ (ou $1{,}25$). Dans le code :
`make_setup(normalize=True)` ($\alpha = 1$) ou `make_setup(alpha=...)`.

### 1.3 Certificat et optimalité

Pour le résidu $r = m - A(s)$, le certificat dual est

$$
\eta(z) = \frac{1}{\lambda}\sum_{i=1}^{M} r_i\,G(x_i, z) = \frac{\langle\varphi(z), r\rangle}{\lambda}.
$$

La mesure $s$ est solution du BLASSO positif **si et seulement si**
$\eta(z)\le 1$ sur $D$ et $\eta(z_k) = 1$ en chaque source. C'est le critère
d'arrêt du Frank-Wolfe et le test « certifié » du LASSO.

### 1.4 Théorème de Duval–Peyré (2015) et hypothèses vérifiées

On pose

$$
\Gamma = \big[\varphi(z_1),\dots,\varphi(z_K)\;\big\vert\;\partial_1\varphi(z_1),\dots,\partial_1\varphi(z_K)\;\big\vert\;\partial_2\varphi(z_1),\dots,\partial_2\varphi(z_K)\big] \in \mathbb{R}^{M\times 3K},
$$

$$
p_V = (\Gamma^{\top})^{+}\begin{pmatrix}\mathbf{1}_K\\ \mathbf{0}_{2K}\end{pmatrix},
\qquad \eta_V(z) = \langle\varphi(z), p_V\rangle .
$$

Le pré-certificat $\eta_V$ est le certificat de norme minimale qui vaut 1 aux
sources avec un gradient nul. Les hypothèses sont :

- **(H1)** $\operatorname{rang}\Gamma = 3K$ (il faut $M\ge 3K$) ;
- **(H2)** $\eta_V(z) < 1$ pour tout $z\in D\setminus\{z_1,\dots,z_K\}$, bord compris ;
- **(H3)** $\nabla^2\eta_V(z_k)\prec 0$ pour tout $k$.

**Théorème.** Si (H1)–(H3) sont vraies et si les $z_k$ sont intérieurs à $D$,
il existe $\lambda_0, c_0 > 0$ tels que, pour $\lambda\le\lambda_0$ et
$\lVert w\rVert/\lambda\le c_0$, le BLASSO a exactement $K$ Diracs, à distance
$O(\lVert w\rVert)$ des vraies sources. Le théorème est **qualitatif** :
$\lambda_0$ et $c_0$ ne sont pas calculables. On mesure donc aussi la marge et
le conditionnement de $\Gamma$ (section 5.3).

## 2. Méthodes implémentées

### 2.1 Noyau (`kernel.py`)

On réutilise `InverseSquareKernel` du projet, avec des dérivées analytiques.
En notant $q = \lVert z-x\rVert^2 + h^2$ :

$$
\nabla_z G = -\frac{2(z-x)}{q^2},
\qquad
\nabla_z^2 G = -\frac{2\,\mathrm{Id}}{q^2} + \frac{8\,(z-x)(z-x)^{\top}}{q^3}.
$$

- `Setup` regroupe les capteurs, $h$, $D$, la pondération $\alpha$ et des
  **poids par capteur** $w_i$ (blanchiment, section 2.6) ; `make_setup()` lit
  `config/`.
- `WeightedKernel` calcule $\psi = \varphi/\lVert\varphi\rVert^{\alpha}$ avec ses
  dérivées analytiques ; `WhitenedKernel` calcule $(w_i\,G(x_i,z))_i$.
- Toutes les dérivées sont vérifiées contre des différences finies (erreur
  relative $\le 10^{-9}$).

### 2.2 Briques communes (`common.py`)

| Fonction | Rôle |
|---|---|
| `forward(z, I)` | $A(s) = \sum_k I_k\,\varphi(z_k)$ |
| `certificate`, `max_certificate` | $\eta$ en des points ; maximum sur $D$ : grille fine puis L-BFGS-B borné depuis les 5 meilleurs maxima locaux |
| `joint_descent` | $\min_{I\ge 0,\,z\in D}\tfrac12\lVert m-\sum_k I_k\varphi(z_k)\rVert^2+\lambda\sum_k I_k$ par L-BFGS-B, gradient analytique |
| `debias` | moindres carrés positifs (NNLS) à positions fixées, sans pénalité |
| `diagnostics` | $\lVert r\rVert^2$ comparé à $M\sigma^2$ |
| `match_sources` | appariement hongrois ; erreurs de position et d'intensité |

### 2.3 LASSO positif sur grille (`lasso.py`)

1. `build_grid(D, step)` construit la grille et la matrice $\mathbf{G}$
   ($M\times P$). Avec un pas de $0{,}25$, $P = 3321$ ; avec un pas de $0{,}1$,
   $P = 20\,301$.
2. `solve_lasso` propose trois solveurs.
   - **FISTA** : pas $1/L$ avec $L = \lVert\mathbf{G}\rVert_2^2$, opérateur
     proximal $c\mapsto\max(c-\lambda/L, 0)$, redémarrage adaptatif
     (O'Donoghue–Candès), arrêt sur les conditions KKT. On ajoute un
     **polissage** : LASSO exact restreint au support trouvé, accepté seulement
     s'il vérifie KKT sur toute la grille.
   - **sklearn** : `Lasso(positive=True, fit_intercept=False)` avec
     $\alpha_{\mathrm{sklearn}} = \lambda/M$, car sklearn minimise
     $\frac{1}{2M}\lVert m-\mathbf{G}c\rVert^2 + \alpha_{\mathrm{sklearn}}\lVert c\rVert_1$.
   - **Ensemble actif** exact (algorithme du projet), qui sert de référence.
3. `merge_peaks` regroupe les nœuds actifs voisins (rayon $1{,}5$ pas) :
   $I = \sum_j c_j$ et $z$ = barycentre pondéré.
4. `refine` : à $K$ fixé, descente conjointe en $(I, z)$ sans grille.
5. `lasso_pipeline` enchaîne grille, LASSO, regroupement, raffinement, test
   $\max\eta\le 1$, débiaisage et diagnostics.

### 2.4 Frank-Wolfe glissant (`blasso.py`)

Algorithme de Denoyelle, Duval, Peyré et Soubies (2019). On part de $\mu = 0$
(ou d'une initialisation donnée). À chaque itération :

1. on calcule le résidu $r = m - A(\mu)$ ;
2. on cherche $z^{\star} = \arg\max_D \eta$ ;
3. on s'arrête si $\eta(z^{\star})\le 1 + \mathrm{tol}$ : $\mu$ est alors
   certifiée optimale ;
4. on ajoute un Dirac en $z^{\star}$ ;
5. on réajuste les intensités par un LASSO positif exact sur les $K$ colonnes ;
6. **glissement** : optimisation conjointe de $(I, z)$ par L-BFGS-B borné à $D$ ;
7. on supprime les Diracs nuls et on fusionne ceux distants de moins de
   $10^{-3}$ fois la diagonale de $D$.

La sortie contient l'historique ($K$, $\max\eta$, objectif) puis le résultat
débiaisé.

### 2.5 Hypothèses (`hypotheses.py`)

`check_hypotheses(z, lam, sigma, ...)` :

1. calcule $\Gamma$, son rang (SVD) et son conditionnement ;
2. calcule $p_V$ par pseudo-inverse et contrôle $\eta_V(z_k) = 1$ et
   $\nabla\eta_V(z_k) = 0$ (à $10^{-15}$ près) ;
3. teste (H2) sur une grille fine hors de boules de rayon $\rho$ autour des
   sources, puis raffine par L-BFGS-B et maximise sur chacun des 4 côtés de $D$ ;
4. teste (H3) avec les valeurs propres de la hessienne $2\times 2$ de $\eta_V$ ;
5. renvoie H1, H2, H3, `non_degenerate`, le rang, le conditionnement, la marge
   et la **marge relative**, et trace la carte de $\eta_V$ si demandé.

**Choix numérique pour (H2).** Au bord d'une boule de rayon $\rho$,
$\eta_V\approx 1-\tfrac12\lvert\lambda_{\max}\rvert\rho^2$, où $\lambda_{\max}$
est la valeur propre de $\nabla^2\eta_V(z_k)$ la plus proche de 0. La marge
$1-\max\eta_V$ est donc petite par nature (de $10^{-7}$ à $10^{-5}$ ici), et un
seuil fixe donne des verdicts arbitraires. On utilise :

- un seuil au niveau du bruit numérique,
  $\max(10^{-9},\,100\times\text{erreur d'interpolation})$ ;
- une **marge relative**
  $\dfrac{1-\max\eta_V}{\tfrac12\lvert\lambda_{\max}\rvert\rho^2}$.
  Elle vaut environ 1 quand $\eta_V$ décroît comme le prédit la hessienne, elle
  est $\ll 1$ quand $\eta_V$ est presque plat, et elle est négative quand (H2)
  échoue.

`hypotheses_after_fit(z_hat, I_hat)` applique la même vérification aux
positions **estimées** et renvoie `reliable`.

### 2.6 Maximum de vraisemblance glouton et choix de $K$ par BIC (`ml.py`)

Cette méthode a été conçue pour le scénario du projet, où la grosse source
centrale fixe le niveau de bruit. Le BLASSO y a deux défauts :

1. la pénalité $\lambda\sum_k I_k$ biaise positions et intensités (biais
   $O(\lambda)$), alors qu'il faut $\lambda$ de l'ordre de $\lVert w\rVert$ pour
   éliminer les fausses sources ;
2. le nombre de sources n'est réglé qu'indirectement, par $\lambda$.

Le bruit étant gaussien de variance $s_i^2$ connue, la log-vraisemblance vaut,
à une constante près,

$$
\log L(I, z) = -\frac12\sum_{i=1}^{M}\frac{\big(m_i - A(s)_i\big)^2}{s_i^2}
= -\frac12\,\big\lVert m_w - A_w(s)\big\rVert_2^2,
\qquad (m_w)_i = \frac{m_i}{s_i},\quad A_w(s)_i = \frac{A(s)_i}{s_i}.
$$

C'est le **blanchiment** : chaque capteur est divisé par l'écart type de son
bruit (`Setup.with_weights(1/s)`). Pour le bruit du projet, $s_i = \sigma\,y_i$
(`relative_noise_std`), calculé sur les mesures **brutes**, source centrale
comprise.

**Algorithme `greedy_ml`.** On part de $K = 0$ et on répète :

1. **ajout** d'un Dirac au maximum de la corrélation
   $z\mapsto\langle\psi_w(z), r_w\rangle$ entre le résidu blanchi et le noyau
   blanchi normalisé ($\alpha = 1$). C'est la même étape que dans Frank-Wolfe ;
2. intensités par moindres carrés positifs, puis **glissement sans pénalité**
   ($\lambda = 0$) de toutes les positions et intensités. On obtient ainsi
   l'estimateur du maximum de vraisemblance à $K$ fixé (un minimum local) ;
3. calcul du **critère d'information**

$$
C(K) = \big\lVert r_w\big\rVert_2^2 + \kappa\cdot 3K,
\qquad \kappa = \log M \ (\text{BIC}) \quad\text{ou}\quad \kappa = 2 \ (\text{AIC}),
$$

   où $3K$ est le nombre de paramètres (deux coordonnées et une intensité par
   source).

On s'arrête quand $C$ n'a pas diminué depuis deux ajouts, et on garde le
meilleur $K$. Une **étape arrière** essaie ensuite de retirer chaque source,
tant que $C$ diminue. Options : `n_candidates` (plusieurs positions candidates
à chaque ajout), `z_init`/`I_init` (départ imposé, par exemple la sortie de
Prony).

### 2.7 Borne de Cramér–Rao (`ml.py`)

La borne de Cramér–Rao (CRB) donne l'erreur minimale de tout estimateur sans
biais. Avec $\theta = (I_1,\dots,I_K, z_1,\dots,z_K)$ et

$$
\mathbf{D} = \Big[\varphi(z_1),\dots,\varphi(z_K)\;\Big\vert\;I_1\nabla\varphi(z_1),\dots,I_K\nabla\varphi(z_K)\Big]\in\mathbb{R}^{M\times 3K},
$$

la matrice de Fisher vaut
$F = \mathbf{D}^{\top}\operatorname{diag}(s_i^{-2})\,\mathbf{D}$, et pour tout
estimateur sans biais

$$
\mathbb{E}\,\lVert\hat z_k - z_k\rVert^2 \;\ge\; \big(F^{-1}\big)_{z_k^{(1)} z_k^{(1)}} + \big(F^{-1}\big)_{z_k^{(2)} z_k^{(2)}} =: \mathrm{CRB}_k^2 .
$$

`cramer_rao(z, I, s)` renvoie $\mathrm{CRB}_k$ pour chaque source. Dans les
tableaux, « CRB » désigne $\max_k \mathrm{CRB}_k$. C'est une **limite physique** :
aucun algorithme ne peut faire mieux en moyenne.

### 2.8 Choix de $\lambda$

- Expériences génériques : $\lambda = c\,\sigma\sqrt{M}$, avec $c = 2$ par
  défaut (balayage $c\in[0{,}25;\,16]$).
- Étude du noyau $1/r^2$ : $\lambda = c\,\sigma\sqrt{M}\,\lVert\Psi\rVert$ avec
  $\lVert\Psi\rVert = \max_D\lVert\psi(z)\rVert$. Avec ce choix, le bruit seul ne
  peut pas faire dépasser $1/c$ au certificat, quel que soit $\alpha$.
- Scénario du projet : $\lambda = 1{,}5\,\sigma\lVert y\rVert$, une estimation de
  $1{,}5\,\lVert w\rVert$.

## 3. Structure du code

```text
sparse_sources/
|-- __init__.py             API publique
|-- kernel.py               Setup, make_setup, WeightedKernel, WhitenedKernel, phi et derivees
|-- common.py               forward, certificat, max_certificate, descente, debias, diagnostics
|-- lasso.py                build_grid, FISTA / sklearn / ensemble actif, merge_peaks, refine
|-- blasso.py               sliding_frank_wolfe
|-- hypotheses.py           check_hypotheses, hypotheses_after_fit, format_report
|-- ml.py                   greedy_ml (maximum de vraisemblance + BIC), cramer_rao
|-- plotting.py             style des figures
|-- experiments.py          experiences generales (3 geometries, comparaison Prony)
|-- experiments_1r2.py      etude ciblee du noyau 1/r^2
|-- experiments_projet.py   scenario du projet (source centrale), Cramer-Rao
|-- tests/                  70 tests pytest
|-- results/                CSV (versionnes) et PNG (ignores par .gitignore)
|-- README.md               notice courte
|-- COMPTE_RENDU.md         ce document
`-- COMPTE_RENDU.tex        ce document converti en LaTeX par Pandoc
```

`sparse_sources` importe `particle_sources` (noyau, `Box`, `CircleSensors`,
configuration, Prony, génération). **Aucun fichier existant du projet n'a été
modifié.** Toutes les fonctions acceptent un argument facultatif `setup=` ;
sans lui, elles utilisent la géométrie de `config/`.

## 4. Validation

`pytest sparse_sources/tests` exécute **70 tests** en environ 30 s. Les 67 tests
existants du projet passent toujours.

| Ce qui est testé | Résultat |
|---|---|
| $\nabla\varphi$, $\nabla^2\varphi$ (brut, $h = 20$, pondéré, blanchi) contre différences finies | erreur $\le 10^{-6}$ (en pratique $10^{-10}$) |
| $\varphi$ identique au noyau du projet (`intensity_at`) | oui |
| $\eta_V(z_k) = 1$ et $\nabla\eta_V(z_k) = 0$ | à $10^{-10}$ près |
| Une source bien vue ($\alpha = 1$, ou $h = 20$) | H1, H2, H3 vraies |
| Noyau brut $1/r^2$ dans le plan, une source | H2 et H3 fausses |
| Accord avec `duval_peyre.diagnose` du projet | mêmes verdicts |
| FISTA, sklearn, ensemble actif | même objectif ($10^{-4}$ ; $10^{-3}$ pour sklearn) |
| Frank-Wolfe sans bruit | positions à moins de $10^{-4}$ |
| Bruit $10^{-4}$, $10^{-3}$, $10^{-2}$ (Frank-Wolfe et LASSO) | bon $K$, erreur $\le 300\,\sigma_{\mathrm{rel}}$ |
| Erreur proportionnelle au bruit ($10^{-5}$ à $10^{-3}$) | rapport constant à un facteur 3 près |
| `greedy_ml`, 2 sources, bruit de la source centrale | bon $K$, erreur $< 5\times$ CRB, $\chi^2$ compatible avec le bruit |
| `greedy_ml`, bruit seul | au plus 1 source retenue |
| Efficacité : erreur rms du ML pour une source (30 tirages) | entre $0{,}6$ et $1{,}5$ fois la CRB |

## 5. Résultats

### 5.1 Trois géométries, trois configurations (`experiments.py`)

Géométries : **plan_brut** (projet : $h = 0$, $\alpha = 0$), **plan_normalise**
($h = 0$, $\alpha = 1$), **hauteur_20** ($h = 20$, $\alpha = 0$). Configurations :
3 sources séparées ; 2 sources à distance 1 ; 2 sources avec des capteurs sur un
cercle 2 fois plus grand (« loin »).

**Hypothèses** (`hypotheses.csv`, cartes `certificat_*.png`) :

| géométrie | config | H1 | H2 | H3 | marge | marge relative | cond($\Gamma$) normalisé |
|---|---|---|---|---|---|---|---|
| plan_brut | séparées | oui | non | non | $-41{,}8$ | $<0$ | 122 |
| plan_brut | proches | oui | non | non | $-13{,}7$ | $<0$ | 3 980 |
| plan_brut | loin | oui | non | non | $-0{,}50$ | $<0$ | 94 |
| plan_normalise | séparées | oui | oui | oui | $8{,}5\times 10^{-6}$ | 1,24 | 92 |
| plan_normalise | proches | oui | oui | oui | $1{,}1\times 10^{-6}$ | 0,70 | 3 950 |
| plan_normalise | loin | oui | oui | oui | $6{,}5\times 10^{-6}$ | 1,32 | 85 |
| hauteur_20 | séparées | oui | oui | oui | $8{,}4\times 10^{-7}$ | 1,42 | **127 000** |
| hauteur_20 | proches | oui | oui | oui | $1{,}5\times 10^{-7}$ | 0,70 | **293 000** |
| hauteur_20 | loin | oui | non | non | $-0{,}012$ | $<0$ | 507 |

**Comparaison des méthodes** ($c = 2$, 4 tirages). Taux de réussite, défini
comme « bon $K$ et erreur de position inférieure au quart de la séparation » :

| géométrie, config | $\sigma/\mathrm{rms}$ | LASSO | LASSO + raffinement | Frank-Wolfe |
|---|---|---|---|---|
| plan_brut, toutes | $10^{-4}$ à $10^{-2}$ | 0 % (23 à 40 sources) | 0 % | 0 % |
| plan_normalise, séparées | $10^{-4}$ / $10^{-3}$ / $10^{-2}$ | 0 / 0 / 25 % | 50 / 75 / 75 % | **100 / 100 / 75 %** |
| plan_normalise, proches | $10^{-4}$ / $10^{-3}$ / $10^{-2}$ | 50 / 25 / 0 % | 100 / 25 / 0 % | **100 / 25 / 0 %** |
| plan_normalise, loin | $10^{-4}$ / $10^{-3}$ / $10^{-2}$ | 50 / 50 / 75 % | 50 / 100 / 25 % | **100 / 100 / 25 %** |
| hauteur_20, séparées et proches | $10^{-4}$ à $10^{-2}$ | 0 % (une seule source) | 0 % | 0 % |

Le LASSO seul surestime souvent $K$, à cause de petits pics voisins non
fusionnés ; le raffinement corrige la plupart des cas, et le Frank-Wolfe est le
plus fiable.

**Erreur en fonction du bruit** (`erreur_sigma.png`, Frank-Wolfe,
plan_normalise) : l'erreur de position est **proportionnelle à $\sigma$** de
$10^{-5}$ à $10^{-2}$, avec un rapport $\text{err}/\sigma_{\mathrm{rel}}$
constant : environ 72 (séparées), 97 (loin) et 330 (proches). C'est la loi
$O(\lVert w\rVert)$ du théorème ; la constante grandit quand les sources se
rapprochent.

**Balayage de $\lambda$** (`balayage_lambda.png`) : avec $\alpha = 1$, $K$ est
exact pour $c$ entre $0{,}5$ et $4$ environ. Un $c$ trop petit ajoute des
sources parasites, un $c$ trop grand fusionne les sources proches. Avec le
noyau brut, on trouve 20 à 35 sources de trop pour tout $c$.

### 5.2 Étude ciblée du noyau $1/r^2$ (`experiments_1r2.py`)

Sources tirées comme dans `config/` (positions uniformes, intensités uniformes
sur le simplexe de somme 100), 20 tirages par $K$,
$\sigma = 10^{-3}\times\mathrm{rms}$, $\lambda = 2\sigma\sqrt{M}\lVert\Psi\rVert$.

**Effet de la pondération $\alpha$** (`r2_alpha.png`) :

| $\alpha$ | condition vraie ($K$ = 1 / 2 / 3 / 4 / 5) | réussite Frank-Wolfe ($K$ = 1 / 2 / 3 / 4 / 5) | temps médian |
|---|---|---|---|
| 0 (brut) | 0 / 0 / 0 / 0 / 0 % | 0 / 0 / 0 / 0 / 0 % | 0,59 s (non convergé) |
| 0,5 | 0 / 0 / 0 / 0 / 0 % | 0 / 0 / 0 / 0 / 0 % | 1,29 s |
| 0,75 | 35 / 25 / 0 / 0 / 0 % | 35 / 25 / 0 / 0 / 0 % | 0,21 s |
| **1** | 100 / 100 / 20 / 20 / 0 % | **100 / 95 / 25 / 15 / 5 %** | **0,05 s** |
| **1,25** | 100 / 100 / 25 / 25 / 0 % | **100 / 95 / 30 / 15 / 0 %** | 0,04 s |
| 1,5 | 100 / 100 / 35 / 30 / 0 % | 100 / 95 / 25 / 5 / 0 % | 0,04 s |
| 2 | 100 / 90 / 50 / 30 / 0 % | 100 / 80 / 20 / 5 / 0 % | 0,04 s |

Il faut $\alpha\ge 1$ pour une seule source. Pour $\alpha = 1$ et une source en
$z_0$, sans bruit et pour $\lambda\to 0$, le certificat vaut

$$
\eta(z) = \big\langle\psi(z), \psi(z_0)\big\rangle \le \lVert\psi(z)\rVert\,\lVert\psi(z_0)\rVert = 1
$$

par Cauchy–Schwarz, avec égalité seulement si $\psi(z) = \psi(z_0)$, donc en
$z = z_0$. Au-delà de $\alpha\approx 1{,}25$, la condition est plus souvent
vraie, mais avec un conditionnement plus mauvais, et la réussite baisse.
**Recommandation : $\alpha = 1$ (ou $1{,}25$).**

**Pouvoir prédictif de `check_hypotheses`** (`r2_prediction.png`, 100 tirages
par $\alpha$) :

| $\alpha$ | condition vraie | réussite si vraie | réussite si fausse |
|---|---|---|---|
| 0,75 | 12 / 100 | 92 % | 1 % |
| **1** | 48 / 100 | **94 %** | **6 %** |
| 1,25 | 50 / 100 | 94 % | 2 % |
| 1,5 | 53 / 100 | 83 % | 2 % |
| 2 | 54 / 100 | 74 % | 2 % |

Le signe de la marge relative sépare presque parfaitement les réussites des
échecs : la vérification des hypothèses est un **outil de diagnostic fiable**,
utilisable a posteriori avec `hypotheses_after_fit`.

**Limite de résolution** (`r2_resolution.png`, paire verticale au centre,
$\alpha = 1$, 5 tirages) :

| $\sigma/\mathrm{rms}$ | séparation minimale résolue (100 %) | erreur à $d = 2$ |
|---|---|---|
| $10^{-5}$ | 0,5 | $1{,}3\times 10^{-3}$ |
| $10^{-4}$ | 1 | $1{,}3\times 10^{-2}$ |
| $10^{-3}$ | 1,5 | 0,14 |
| $10^{-2}$ | 4 | ($K$ faux à $d = 2$) |

On observe environ $d^{\star}\propto\sigma^{1/4}$ : le bruit admissible
décroît comme $d^4$, ce qui est typique de la super-résolution de sources
positives proches. À $d = 0{,}25$, la condition est vraie mais
$\operatorname{cond}(\Gamma) = 2{,}5\times 10^5$, et aucun niveau de bruit testé
ne permet la reconstruction.

**Distance aux capteurs** (paire horizontale de séparation 2, $\alpha = 1$,
$\sigma = 10^{-3}$) :

| $y$ (centre = 10) | distance au capteur le plus proche | condition | réussite |
|---|---|---|---|
| 10 à 14 | 10,2 à 7,1 | vraie | 100 % |
| 16 | 5,1 | vraie | 60 % |
| 17,5 / 18,5 / 19,25 | 3,6 / 2,6 / 1,9 | **fausse** | **0 %** |

Une **source seule** reste reconstructible près des capteurs. C'est la
**paire** qui échoue : près d'un capteur, $\psi(z)$ est dominé par ce capteur,
et deux sources à la même distance des mêmes capteurs ont des signatures
presque identiques (`r2_certificat_pres_capteurs.png`).

**Nombre de capteurs** (`r2_capteurs.png`) : passer de $M = 16$ à $M = 256$
change peu de chose. Les échecs sont **géométriques** ; ajouter des capteurs sur
le **même cercle** n'apporte pas d'information nouvelle.

### 5.3 La condition est qualitative : le cas $h = 20$

Avec des capteurs à $h = 20$ au-dessus du plan, $G(x,\cdot)$ n'est plus
sous-harmonique sur $D$, car

$$
\Delta_z G = \frac{4\,(r^2-h^2)}{(r^2+h^2)^3} < 0 \quad\text{pour } r < h,
$$

et la condition est **vraie** pour les configurations séparées et proches.
Pourtant **la reconstruction échoue même sans bruit** : le noyau est si lisse
que les colonnes sont presque colinéaires, avec
$\operatorname{cond}(\Gamma)\approx 10^5$. **À retenir : (H1)–(H3) ne suffisent
pas en pratique ; il faut aussi regarder $\operatorname{cond}(\Gamma)$ et la
marge relative.**

### 5.4 Première comparaison avec `reconstruct.py` (Prony)

Scénario identique à `scripts/reconstruct.py` : $n$ sources tirées selon
`config/`, source centrale d'intensité 5000 mesurée puis retirée, bruit relatif,
8 tirages par case, mêmes mesures pour toutes les méthodes. Taux de réussite
(et erreur de position médiane) :

| $n$ | $\sigma$ | Prony (config) | Prony ($5\sigma$) | Frank-Wolfe brut | LASSO normalisé | Frank-Wolfe normalisé |
|---|---|---|---|---|---|---|
| 2 | $10^{-4}$ | 88 % (0,02) | 25 % | 0 % | **100 %** (0,05) | 88 % (0,05) |
| 2 | $10^{-3}$ | 25 % (0,41) | 25 % | 0 % | **75 %** (0,50) | 62 % (0,50) |
| 2 | $10^{-2}$ | 0 % | 12 % | 0 % | 12 % | 12 % |
| 3 | $10^{-4}$ | **62 %** (0,16) | 25 % | 0 % | 25 % | 25 % |
| 3 | $10^{-3}$ | 25 % | 25 % | 0 % | 0 % | 0 % |
| 5 | $10^{-4}$ | 12 % | **25 %** | 0 % | 0 % | 0 % |

Prony est meilleur à très faible bruit et pour plus de sources ; le BLASSO
pondéré est plus robuste à bruit moyen pour 2 sources. Cette comparaison a
motivé la recherche d'une meilleure méthode (section suivante).

### 5.5 Scénario du projet : maximum de vraisemblance et borne de Cramér–Rao (`experiments_projet.py`)

Données identiques à `scripts/reconstruct.py` : $n$ sources tirées selon
`config/` (intensités uniformes sur le simplexe de somme 100, positions
uniformes), source centrale $I_c = 5000$ ajoutée puis retirée, bruit relatif
$\sigma$. 20 tirages par case, mêmes mesures pour toutes les méthodes :

- `prony` : `cfg.invert`, exactement comme `reconstruct.py` ;
- `sfw` : Frank-Wolfe glissant, BLASSO pondéré $\alpha = 1$,
  $\lambda = 1{,}5\,\sigma\lVert y\rVert$ (meilleure méthode des sections précédentes) ;
- `ml_bic` : maximum de vraisemblance glouton, bruit blanchi, $K$ par BIC (section 2.6) ;
- `ml_bic_multi` : idem avec 3 candidats par ajout et un second départ depuis
  Prony ; on garde le meilleur critère.

Pour chaque tirage, on calcule aussi la borne de Cramér–Rao (CRB). Un tirage
est dit **faisable** si $2\,\mathrm{CRB} < \tfrac14\min(d_{\min}, 4)$, où
$d_{\min}$ est la plus petite distance entre deux sources : une méthode
optimale a alors de bonnes chances de passer le critère de réussite.

**Taux de réussite** (`projet_methodes_resume.csv`, `projet_methodes.png`) :

| $n$ | $\sigma$ | Prony | Frank-Wolfe | **ML + BIC** | ML multi | tirages faisables | ML + BIC sur les faisables |
|---|---|---|---|---|---|---|---|
| 1 | $10^{-4}$ | 100 % | 100 % | **100 %** | 100 % | 100 % | 100 % |
| 1 | $10^{-3}$ | 15 % | 100 % | **100 %** | 100 % | 100 % | 100 % |
| 1 | $10^{-2}$ | 0 % | 100 % | **100 %** | 100 % | 85 % | 100 % |
| 2 | $10^{-4}$ | 85 % | 90 % | **95 %** | 95 % | 85 % | 94 % |
| 2 | $3\times 10^{-4}$ | 40 % | 75 % | **85 %** | 85 % | 80 % | 94 % |
| 2 | $10^{-3}$ | 15 % | 70 % | **75 %** | 75 % | 80 % | 94 % |
| 2 | $3\times 10^{-3}$ | 0 % | 40 % | **75 %** | 75 % | 50 % | 90 % |
| 2 | $10^{-2}$ | 5 % | 15 % | **30 %** | 30 % | 20 % | 100 % |
| 3 | $10^{-4}$ | 55 % | 40 % | **80 %** | 75 % | 75 % | 100 % |
| 3 | $3\times 10^{-4}$ | 25 % | 20 % | **65 %** | 65 % | 35 % | 100 % |
| 3 | $10^{-3}$ | 10 % | 5 % | **40 %** | 35 % | 25 % | 100 % |
| 3 | $10^{-2}$ | 0 % | 0 % | 0 % | 0 % | 0 % | sans objet |
| 5 | $10^{-4}$ | 5 % | 0 % | **50 %** | 50 % | 20 % | 100 % |
| 5 | $3\times 10^{-4}$ | 5 % | 0 % | **15 %** | 15 % | 5 % | 100 % |
| 5 | $10^{-3}$ | 0 % | 0 % | 5 % | 5 % | 5 % | 100 % |

**Erreur de position médiane comparée à la CRB** (même fichier) :

| $n$ | $\sigma$ | Prony | Frank-Wolfe | ML + BIC | CRB médiane |
|---|---|---|---|---|---|
| 1 | $10^{-3}$ | 0,059 | 0,023 | **0,022** | 0,025 |
| 1 | $10^{-2}$ | 0,84 | 0,23 | **0,23** | 0,25 |
| 2 | $10^{-4}$ | 0,014 | 0,035 | **0,014** | 0,016 |
| 2 | $10^{-3}$ | 0,32 | 0,29 | **0,13** | 0,16 |
| 2 | $3\times 10^{-3}$ | 1,6 | 0,66 | **0,36** | 0,49 |
| 3 | $10^{-4}$ | 0,14 | 0,72 | **0,15** | 0,22 |
| 3 | $10^{-3}$ | 4,3 | 0,93 | **0,42** | 2,2 |
| 5 | $10^{-4}$ | 0,88 | 1,6 | **0,60** | 1,04 |

**Lecture.**

1. **Le maximum de vraisemblance + BIC est la meilleure méthode dans toutes les
   cases**, ou à égalité avec la meilleure. Le gain est net dès 2 sources :
   75 % contre 15 % pour Prony à $n = 2$, $\sigma = 10^{-3}$ ; 80 % contre 55 %
   à $n = 3$, $\sigma = 10^{-4}$ ; 50 % contre 5 % à $n = 5$, $\sigma = 10^{-4}$.
2. **Son erreur atteint la borne de Cramér–Rao.** Pour 1 et 2 sources,
   l'erreur médiane vaut $0{,}7$ à $0{,}9$ fois la CRB (la CRB porte sur la
   pire source, l'erreur médiane sur un tirage typique). La méthode est donc
   **efficace** au sens statistique : aucun estimateur sans biais ne peut faire
   mieux. Le test `test_cramer_rao_scaling_and_efficiency` le vérifie pour une
   source.
3. **Quand le problème est faisable, la méthode réussit** : 90 à 100 % de
   réussite sur les tirages faisables, dans toutes les cases. Ses échecs
   correspondent presque tous à des tirages où la CRB dépasse la tolérance :
   aucune méthode ne pourrait les réussir de manière fiable.
4. **D'où vient le gain par rapport au BLASSO** : surtout de la suppression de
   la pénalité lors du glissement (plus de biais $O(\lambda)$ sur les
   positions) et du choix de $K$ par BIC. Le blanchiment seul apporte peu ici :
   la source centrale, au centre du cercle, donne presque le même bruit à tous
   les capteurs (dans le prototype, le Frank-Wolfe blanchi et le non blanchi
   donnent des résultats très proches).
5. **Prony** est excellent pour une source à très faible bruit, mais il
   surestime $K$ dès que $\sigma\ge 10^{-3}$ (4 à 7 sources estimées), car son
   seuil de rang n'est pas adapté au bruit.
6. **Les variantes « multi »** (plusieurs candidats, départ depuis Prony)
   n'apportent rien : elles trouvent parfois un critère plus bas, mais une
   reconstruction moins bonne. La méthode simple suffit.
7. Quand la CRB est très grande (par exemple $n = 3$, $\sigma = 10^{-3}$ :
   CRB $= 2{,}2$), l'erreur des méthodes peut être inférieure à la CRB. Elles
   sont alors biaisées (régularisation implicite), et la borne indique surtout
   que la position de la pire source n'est pas déterminée par les données.

**Choix du nombre de sources : BIC contre AIC** (`projet_critere.csv`) :

| $n$ | $\sigma$ | réussite BIC | réussite AIC | trop de sources (BIC / AIC) | trop peu (BIC / AIC) |
|---|---|---|---|---|---|
| 2 | $10^{-4}$ | **95 %** | 85 % | 5 / 15 % | 0 / 0 % |
| 2 | $10^{-3}$ | **75 %** | 70 % | 5 / 15 % | 15 / 10 % |
| 3 | $10^{-4}$ | **80 %** | 60 % | 0 / 20 % | 0 / 0 % |
| 3 | $10^{-3}$ | **40 %** | 35 % | 0 / 10 % | 50 / 25 % |
| 5 | $10^{-4}$ | **50 %** | 40 % | 0 / 20 % | 35 / 25 % |

Le BIC ($\kappa = \log M\approx 4{,}2$) est meilleur partout ; l'AIC
($\kappa = 2$) ajoute des sources parasites. Les erreurs du BIC sont surtout des
sources **manquantes** à fort bruit : des sources faibles ou proches que les
données ne permettent pas de voir.

**Effet de la source centrale** (`projet_centrale_resume.csv`,
`projet_centrale.png`, $n = 3$, $\sigma = 10^{-3}$, 20 tirages ; le facteur
vaut $I_c/100$) :

| facteur | Prony | Frank-Wolfe | ML + BIC | erreur ML + BIC | CRB médiane |
|---|---|---|---|---|---|
| 0 (pas de source centrale) | 55 % | 35 % | **100 %** | 0,034 | 0,052 |
| 5 | 55 % | 35 % | **70 %** | 0,22 | 0,30 |
| 50 (projet) | 10 % | 5 % | **40 %** | 0,42 | 2,2 |
| 500 | 0 % | 0 % | 0 % | 1,2 | 21 |

**C'est le résultat principal.** Sans source centrale, le maximum de
vraisemblance réussit dans 100 % des cas. Avec la source centrale du projet, la
CRB est multipliée par environ 42. Le bruit de la source centrale, d'écart type
$\sigma I_c/R^2 = 40\,\sigma$ par capteur, reste dans les données après
soustraction. Il domine le signal des petites sources, qui vaut environ 7 en
moyenne quadratique : avec $\sigma = 10^{-3}$, le rapport signal sur bruit par
capteur n'est que d'environ 175.

**Configuration par défaut du projet** ($N\sim\mathrm{Poisson}(20)$,
$\sigma = 1\,\%$, 10 tirages, `projet_defaut.csv`) :

- les tirages contiennent 12 à 29 sources, d'intensité maximale 11 à 26 ;
- la CRB conjointe médiane est comprise entre $10^4$ et $5\times 10^5$ : les
  positions ne sont **pas déterminées** par les données ;
- même en supposant toutes les autres sources connues, une source n'a une CRB
  inférieure à $0{,}5$ que dans 0 à 4 cas par tirage ;
- ML + BIC retient 1 à 5 sources, Frank-Wolfe 1 à 4, Prony 0 à 9.

Dans cette configuration, on ne peut espérer retrouver que les quelques
sources les plus intenses et les mieux placées. Il faut soit beaucoup moins de
bruit, soit beaucoup moins de sources, soit une autre géométrie de capteurs
(section 8).

## 6. Où ça ne marche pas, et pourquoi

### 6.1 BLASSO standard avec le noyau $1/r^2$ : échec systématique

- **Constat** : la condition est fausse dans 100 % des cas, et la réussite est
  nulle. On obtient 20 à 40 sources parasites collées au bord de $D$, surtout
  aux coins.
- **Cause 1 : sous-harmonicité.** En dimension 2,
  $\Delta_z\big(1/\lVert x-z\rVert^2\big) = 4/\lVert x-z\rVert^4 > 0$ : chaque
  $G(x_i,\cdot)$ est sous-harmonique. Lorsque $p_V\ge 0$, $\eta_V$ est
  sous-harmonique et ne peut pas avoir de maximum strict à l'intérieur de $D$ :
  la trace de $\nabla^2\eta_V(z_k)$ est positive (H3 fausse) et le maximum est
  au bord (H2 fausse, $\max\eta_V$ entre 17 et 43 près des coins).
- **Cause 2 : biais de la pénalité.** $\lVert\varphi(z)\rVert$ varie d'un
  facteur 215 sur $D$. Le BLASSO « paie » moins cher pour remplacer une source
  centrale par plusieurs sources de bord ; la solution du LASSO sur grille est
  d'ailleurs un anneau de sources de bord, même sans bruit.
- **Remède** : la pondération $\alpha\approx 1$ (section 1.2).

### 6.2 Sources proches

La séparation minimale résolue va de $0{,}5$ à $4$ selon le bruit
($d^{\star}\propto\sigma^{1/4}$). Le conditionnement de $\Gamma$ explose
(environ 4 000 à $d = 1$ et 250 000 à $d = 0{,}25$).

### 6.3 Zone aveugle près des capteurs

Le cercle des capteurs passe par les coins de $D$. Les **paires** situées à
moins de 4 à 5 unités d'un capteur ne sont plus séparables. Comme les positions
uniformes tombent souvent près du bord, cela explique une grande partie des
échecs pour $K\ge 3$.

### 6.4 Bruit de la source centrale

C'est la limite dominante dans le scénario du projet (section 5.5). Retirer la
source centrale enlève son signal, mais pas son bruit, d'écart type
$40\,\sigma$ par capteur. Aucune méthode ne peut descendre sous la borne de
Cramér–Rao, qui croît proportionnellement à ce bruit.

### 6.5 Nombre de sources

Avec des positions aléatoires, la probabilité d'avoir une paire proche ou une
paire près des capteurs croît vite avec $K$. Dans la configuration par défaut
($K\approx 20$, $\sigma = 1\,\%$), la borne de Cramér–Rao conjointe dépasse
$10^4$ : le problème est hors de portée de toute méthode.

### 6.6 Noyau trop lisse (capteurs hors du plan, $h$ grand)

La condition est vraie mais $\operatorname{cond}(\Gamma)\approx 10^5$ : c'est
inexploitable numériquement (section 5.3).

### 6.7 Problèmes numériques rencontrés

- **FISTA est lent** sur une grille fine, car les colonnes voisines sont presque
  colinéaires : 27 s au pas de $0{,}1$ contre $0{,}17$ s pour l'ensemble actif.
  Le polissage sur le support corrige la non-convergence.
- **sklearn** s'arrête sur son nombre maximal d'itérations (écart d'objectif
  d'environ $3\times 10^{-4}$).
- **LASSO + raffinement** : un petit pic parasite peut rester bloqué dans un
  minimum local (intensité $\approx 0$, donc gradient en position $\approx 0$).
- **$\lambda$ très petit** (moins de $10^{-6}\lambda_{\max}$) : le critère
  $\eta\le 1+10^{-6}$ n'est plus atteignable numériquement.
- **Seuil de (H2)** : un seuil fixe donne des verdicts arbitraires, d'où la
  marge relative (section 2.5).
- **Maximum de vraisemblance** : le problème à $K$ fixé n'est pas convexe. Un
  meilleur critère ne garantit pas une meilleure reconstruction : partir de
  Prony ou essayer plusieurs candidats trouve parfois un optimum de critère plus
  bas mais plus loin de la vérité (section 5.5).

## 7. Temps de calcul

Machine de développement (Windows, Python 3.14, NumPy 2.3, SciPy 1.16), un
cœur, valeurs médianes.

| Brique | Paramètres | Temps |
|---|---|---|
| `make_grid` (matrice $\mathbf{G}$) | pas 0,1 ($P = 20\,301$) | 0,07 s |
| Frank-Wolfe glissant, $\alpha = 1$ | $K$ = 1 / 2 / 3 / 5 / 8 | 0,004 / 0,017 / 0,045 / 0,10 / 0,44 s |
| Frank-Wolfe glissant, noyau brut | plafonné à 10 itérations | 0,4 à 1,3 s |
| `lasso_pipeline`, ensemble actif, $K = 3$ | pas 0,5 / 0,25 / 0,1 | 0,03 / 0,04 / 0,17 s |
| `lasso_pipeline`, FISTA, $K = 3$ | pas 0,5 / 0,25 / 0,1 | 0,86 / 2,0 / 27 s |
| `lasso_pipeline`, sklearn, $K = 3$ | pas 0,5 / 0,25 / 0,1 | 1,5 / 1,9 / 16 s |
| `check_hypotheses` | grille 0,05, $K$ = 1 à 8 | 0,25 à 0,30 s |
| Prony (`prony_circle`) | $M = 64$ | environ 0,001 s |
| `greedy_ml` (ML + BIC) | $n$ = 1 / 2 / 3 / 5, $\sigma = 10^{-4}$ | 0,06 / 0,12 / 0,19 / 0,42 s |
| `greedy_ml`, 3 candidats + départ Prony | $n$ = 1 / 3 / 5 | 0,24 / 0,65 / 1,4 s |
| `cramer_rao` | $K\le 30$ | moins de 0,01 s |
| `pytest sparse_sources/tests` | 70 tests | environ 30 s |
| `experiments.py` | tout | 336 s |
| `experiments_1r2.py` | tout (20 tirages par $K$) | 391 s |
| `experiments_projet.py` | tout (20 tirages par case) | 366 s |

Le Frank-Wolfe glissant est le plus rapide quand la condition tient (environ
$K+1$ itérations). Le maximum de vraisemblance glouton est 3 à 10 fois plus
lent : il ajoute en général deux sources de trop avant de s'arrêter, puis fait
l'étape arrière. Il reste sous la demi-seconde pour 5 sources.

## 8. Pistes d'amélioration

**Modélisation**

1. **Utiliser le maximum de vraisemblance glouton + BIC** (`greedy_ml`) comme
   méthode par défaut dans le scénario du projet, et le BLASSO pondéré
   ($\alpha = 1$) quand on veut un certificat d'optimalité.
2. **Réduire le bruit de la source centrale** plutôt que chercher un meilleur
   algorithme : mesurer plus longtemps, protéger les capteurs de la source
   centrale (collimation), ou estimer son intensité conjointement si elle n'est
   pas connue exactement.
3. **Bruit de Poisson** (`poisson_counts`) : remplacer l'attache quadratique par
   la log-vraisemblance de Poisson (divergence de Kullback–Leibler) ; la même
   méthode gloutonne s'applique.

**Géométrie des capteurs** (facteur limitant principal après le bruit)

4. **Éloigner les capteurs des coins de $D$** (cercle plus grand,
   `radius_factor`) pour supprimer la zone aveugle. La borne de Cramér–Rao et
   `check_hypotheses` permettent de comparer des géométries avant de construire
   quoi que ce soit.
5. **Diversifier les capteurs** plutôt que les multiplier sur le même cercle
   (plusieurs cercles, capteurs à l'intérieur, hauteur modérée $h\sim 2$ à $5$).

**Algorithmes**

6. **Choix de $K$** : le BIC est un bon compromis ; un test du $\chi^2$ sur
   $\lVert r_w\rVert^2$ ou une validation croisée sur les capteurs sont des
   alternatives.
7. **LASSO sur grille** : remplacer FISTA par l'ensemble actif ou par une
   descente par coordonnées avec élimination (*screening*).

**Théorie**

8. **Séparation dans la métrique du noyau** (métrique de Fisher–Rao,
   Poon–Keriven–Peyré) pour expliquer quantitativement la zone aveugle et la
   loi $d^{\star}\propto\sigma^{1/4}$.
9. **Constantes quantitatives** : relier $\lambda_0$ et $c_0$ à la marge relative
   et à $\operatorname{cond}(\Gamma)$.

## 9. Écarts par rapport à la demande initiale

- Toutes les fonctions acceptent un argument supplémentaire `setup=`
  (géométrie, $h$, $\alpha$, poids des capteurs).
- `debias(z, I_init, m)` prend aussi les mesures $m$, nécessaires au NNLS.
- `check_hypotheses` a des paramètres supplémentaires et renvoie en plus la
  marge relative et des indicateurs de bruit.
- Ajouts non demandés, jugés nécessaires : la pondération $\alpha$, le solveur
  exact à ensemble actif, le polissage de FISTA, la marge relative de (H2),
  l'étude ciblée du noyau $1/r^2$, la comparaison avec Prony, le maximum de
  vraisemblance glouton, le blanchiment et la borne de Cramér–Rao.
- Les **PNG** ne sont pas versionnés (le `.gitignore` du dépôt ignore `*.png`) ;
  les scripts d'expériences les régénèrent. Les CSV sont versionnés.

## 10. Reproduire

```bash
pip install -e ".[dev]" scikit-learn
pytest sparse_sources/tests                   # 70 tests, environ 30 s
python sparse_sources/experiments.py          # environ 6 min
python sparse_sources/experiments_1r2.py      # environ 6,5 min
python sparse_sources/experiments_projet.py   # scenario du projet
```

Toutes les graines sont fixées : les résultats sont reproductibles.

Pour obtenir une version LaTeX ou PDF du compte rendu :

```bash
pandoc COMPTE_RENDU.md -s -o COMPTE_RENDU.tex
pandoc COMPTE_RENDU.md -s --pdf-engine=xelatex -o COMPTE_RENDU.pdf
```

## 11. Conclusion : quelle méthode utiliser ?

Taux de réussite sur le scénario du projet (section 5.5, source centrale
comprise, 20 tirages par case) :

| sources | bruit $\sigma$ | Prony | Frank-Wolfe ($\alpha = 1$) | **ML + BIC** |
|---|---|---|---|---|
| 1 | $10^{-4}$ (faible) | 100 % | 100 % | **100 %** |
| 1 | $10^{-3}$ | 15 % | 100 % | **100 %** |
| 2 | $10^{-4}$ (faible) | 85 % | 90 % | **95 %** |
| 2 | $3\times 10^{-3}$ | 0 % | 40 % | **75 %** |
| 3 | $10^{-4}$ (faible) | 55 % | 40 % | **80 %** |
| 3 | $10^{-3}$ | 10 % | 5 % | **40 %** |
| 5 | $10^{-4}$ (faible) | 5 % | 0 % | **50 %** |

- **Prony est bon quand le bruit est très faible.** C'est une méthode
  algébrique, exacte sans bruit. Mais dès que $\sigma$ atteint $10^{-3}$, il
  trouve trop de sources (4 à 7) et s'effondre, même avec une seule source
  (15 %).
- **Frank-Wolfe résiste bien au bruit, mais seulement pour 1 ou 2 sources.** À
  partir de 3 sources, il bute sur la condition (H2) : sources proches, sources
  près des capteurs.
- **Avec beaucoup de sources, les deux sont mauvais.** Prony fait un peu mieux à
  faible bruit (55 % contre 40 % pour 3 sources), mais avec 5 sources ils sont
  tous les deux proches de 0 %. Prony n'est donc pas « bon pour beaucoup de
  sources ».
- **ML + BIC fait au moins aussi bien que la meilleure des deux dans chaque
  case**, et nettement mieux dès 2 sources avec du bruit.

**Tableau de conclusion :**

| situation | meilleure méthode classique | méthode à utiliser | ce qui limite |
|---|---|---|---|
| bruit très faible | Prony | **ML + BIC** | rien en pratique (1 à 3 sources) |
| bruit moyen, 1 ou 2 sources | Frank-Wolfe ($\alpha = 1$) | **ML + BIC** | le bruit, au niveau de la borne de Cramér–Rao |
| 3 sources ou plus | aucune | **ML + BIC** | la condition (H2) pour le BLASSO, le bruit pour le ML |
| bruit fort, source centrale du projet | aucune | **ML + BIC** | la borne de Cramér–Rao : le problème n'est plus résoluble |

En pratique, la méthode à retenir est `greedy_ml` (maximum de vraisemblance +
BIC). Son erreur atteint la borne de Cramér–Rao : pour faire mieux, il faut
changer les données (moins de bruit sur la source centrale, autre géométrie
de capteurs), pas l'algorithme. Prony et Frank-Wolfe ne servent plus que de
points de comparaison.

## Références

- V. Duval, G. Peyré, *Exact support recovery for sparse spikes deconvolution*,
  Foundations of Computational Mathematics, 2015.
- Q. Denoyelle, V. Duval, G. Peyré, E. Soubies, *The sliding Frank-Wolfe
  algorithm and its application to super-resolution microscopy*, Inverse
  Problems, 2019.
- Q. Denoyelle, V. Duval, G. Peyré, *Support recovery for sparse
  super-resolution of positive measures*, Journal of Fourier Analysis and
  Applications, 2017.
- C. Poon, N. Keriven, G. Peyré, *The geometry of off-the-grid compressed
  sensing*, Foundations of Computational Mathematics, 2023.
- A. Beck, M. Teboulle, *A fast iterative shrinkage-thresholding algorithm for
  linear inverse problems*, SIAM Journal on Imaging Sciences, 2009.
- B. O'Donoghue, E. Candès, *Adaptive restart for accelerated gradient
  schemes*, Foundations of Computational Mathematics, 2015.
- G. Schwarz, *Estimating the dimension of a model*, Annals of Statistics, 1978
  (critère BIC).
- S. M. Kay, *Fundamentals of Statistical Signal Processing: Estimation
  Theory*, Prentice Hall, 1993 (borne de Cramér–Rao).

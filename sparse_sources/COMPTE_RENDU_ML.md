---
title: "Compte rendu : localisation de sources par maximum de vraisemblance et critère BIC"
subtitle: "Méthode, résultats et limites"
lang: fr
---

# Compte rendu : maximum de vraisemblance + BIC

Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/`.

Ce compte rendu présente la méthode qui donne les meilleurs résultats sur le
problème du projet : le **maximum de vraisemblance glouton avec choix du
nombre de sources par le critère BIC** (« ML + BIC »). Il explique l'idée, ce
que la méthode permet de retrouver, et où elle s'arrête. Les détails
techniques (dérivations, tableaux complets, liste des tests) sont dans
`RAPPORT_ML_BIC.md` ; la comparaison avec les autres méthodes (LASSO, BLASSO,
Prony) est dans `COMPTE_RENDU.md`. Tous les chiffres viennent des fichiers CSV
de `sparse_sources/results/`.

## 1. Le problème

On cherche des sources ponctuelles dans le rectangle $D = [0,10]\times[0,20]$ :
leur nombre $K$, leurs positions $z_k$ et leurs intensités $I_k$. On dispose
de $M = 64$ capteurs placés sur le cercle qui entoure $D$. Le capteur $i$, en
$x_i$, reçoit de chaque source un signal qui décroît comme l'inverse du carré
de la distance :

$$
y_i = \sum_{k=1}^{K} \frac{I_k}{\lVert x_i - z_k\rVert^2} \;+\; \frac{I_c}{\lVert x_i - c\rVert^2} .
$$

Le second terme est la **source centrale** du projet : une source connue,
très intense ($I_c = 5000$, contre une intensité totale de 100 pour les
autres sources), placée au centre $c$ de $D$.

Chaque mesure est bruitée : le capteur lit $y_i(1 + \sigma\varepsilon_i)$, avec
$\varepsilon_i$ gaussien centré réduit et $\sigma$ entre $10^{-4}$ et
$10^{-2}$. On retire ensuite la contribution connue de la source centrale.

**Ce qui rend le problème difficile.** Retirer la source centrale enlève son
signal, mais pas son bruit. Chaque capteur reçoit $5000/125 = 40$ de la
source centrale, contre environ 7 en moyenne pour toutes les autres sources
réunies. Le bruit restant, d'écart type environ $40\,\sigma$, est donc
dominé par la source centrale.

## 2. L'idée : deux questions séparées

La méthode répond séparément à deux questions.

**Où sont les sources et quelle est leur intensité ? Le maximum de
vraisemblance (ML).** Pour un nombre de sources $K$ donné, on cherche les
positions et intensités qui rendent les mesures observées **les plus
probables**. Avec un bruit gaussien, cela revient à minimiser l'écart entre
les mesures $m_i$ et le modèle, chaque capteur étant divisé par l'écart type
$s_i$ de son bruit :

$$
\chi^2 = \sum_{i=1}^{M}\left(\frac{m_i - \text{modèle}_i}{s_i}\right)^2 .
$$

Diviser par $s_i$ revient à faire moins confiance aux capteurs les plus
bruités. On ajuste positions et intensités **sans aucune pénalité** : il n'y a
pas de paramètre de régularisation qui fausserait le résultat.

**Combien y a-t-il de sources ? Le critère BIC.** Le ML seul ne suffit pas :
plus on ajoute de sources, plus $\chi^2$ baisse, même si les sources ajoutées
sont fausses, car une source de trop « explique » toujours un peu de bruit.
Le critère d'information bayésien (BIC) fait payer chaque source :

$$
C(K) = \chi^2 + 3K\log M .
$$

Chaque source a 3 paramètres ($x$, $y$, $I$) et chaque paramètre coûte
$\log M\approx 4{,}2$ pour 64 capteurs. **Une source n'est donc gardée que si
elle fait baisser $\chi^2$ de plus de $3\log 64\approx 12{,}5$.**

Pourquoi ce seuil fonctionne :

- une **fausse source**, qui n'explique que du bruit, fait baisser $\chi^2$
  d'environ 3 en moyenne (un par paramètre) : elle est rejetée ;
- une **vraie source** fait baisser $\chi^2$ d'environ
  $\mathrm{SNR}^2 + 3$, où $\mathrm{SNR}$ est son rapport signal sur bruit :
  elle est gardée dès que $\mathrm{SNR}\gtrsim 3$.

Il n'y a **aucun réglage** : le seuil ne dépend que du nombre de capteurs.

**Comparaison avec le BLASSO.** Le BLASSO (Frank-Wolfe) règle ces deux
questions avec un seul paramètre $\lambda$ : un $\lambda$ petit laisse passer
de fausses sources, un $\lambda$ grand décale les positions et sous-estime
les intensités. Il ne fonctionne en plus que si une condition mathématique
(la condition de Duval–Peyré) est vraie, ce qui est rare à partir de 3
sources. ML + BIC n'a aucune de ces deux contraintes.

## 3. L'algorithme

On construit la solution source par source (algorithme « glouton ») :

1. on part de zéro source ;
2. on cherche l'endroit où les mesures encore inexpliquées ressemblent le
   plus au signal d'une source, et on y place une nouvelle source ;
3. on réajuste **ensemble** toutes les positions et intensités pour
   minimiser $\chi^2$ (optimisation L-BFGS-B, gradient exact) ;
4. on calcule $C(K)$ ; s'il baisse, on recommence en 2 ; s'il remonte deux
   fois de suite, on s'arrête ;
5. étape arrière : on essaie de retirer chaque source, et on la retire si
   $C$ baisse ;
6. on renvoie la solution de plus petit $C(K)$.

Exemple réel (3 sources, $\sigma = 10^{-3}$, avec la source centrale) :

| $K$ | 0 | 1 | 2 | **3** | 4 | 5 |
|---|---|---|---|---|---|---|
| $C(K)$ | 735 305 | 238 756 | 997 | **89,8** | 97,7 | 102,9 |

Le critère chute jusqu'à $K = 3$, puis remonte : la 4e source fait baisser
$\chi^2$ de moins de 12,5. La méthode trouve 3 sources. Elle prend 0,1 à
0,5 seconde pour 1 à 5 sources.

## 4. Savoir si le résultat est fiable

La méthode fournit deux contrôles.

**Le $\chi^2$ final.** Si le modèle explique les mesures, $\chi^2/M$ vaut
environ $(M - 3K)/M$, soit 0,86 pour 3 sources et 64 capteurs. Une valeur
nettement supérieure à 1 signale une source manquée ou un bruit sous-estimé.

**La borne de Cramér–Rao (CRB).** C'est l'erreur de position minimale que
**n'importe quelle méthode** peut atteindre en moyenne, compte tenu du bruit.
Elle se calcule pour chaque source (`cramer_rao`). Si la CRB d'une source est
grande devant la distance aux autres sources, cette source ne peut pas être
localisée de façon fiable, quel que soit l'algorithme. C'est l'outil qui
permet de distinguer « la méthode est mauvaise » de « le problème est
impossible ».

## 5. Résultats sur le scénario du projet

Mêmes données que `scripts/reconstruct.py`, 20 tirages par case. Une
reconstruction est **réussie** si le nombre de sources est exact et si chaque
source est retrouvée à moins d'un quart de la distance entre les deux sources
les plus proches.

| sources | bruit $\sigma$ | Prony | Frank-Wolfe | **ML + BIC** |
|---|---|---|---|---|
| 1 | $10^{-3}$ | 15 % | 100 % | **100 %** |
| 1 | $10^{-2}$ | 0 % | 100 % | **100 %** |
| 2 | $10^{-4}$ | 85 % | 90 % | **95 %** |
| 2 | $10^{-3}$ | 15 % | 70 % | **75 %** |
| 2 | $3\times 10^{-3}$ | 0 % | 40 % | **75 %** |
| 3 | $10^{-4}$ | 55 % | 40 % | **80 %** |
| 3 | $10^{-3}$ | 10 % | 5 % | **40 %** |
| 5 | $10^{-4}$ | 5 % | 0 % | **50 %** |

1. **ML + BIC est la meilleure méthode dans toutes les cases**, et l'écart
   grandit avec le nombre de sources et le bruit.
2. **Elle est optimale** : son erreur vaut 0,7 à 0,9 fois la CRB. On ne peut
   donc pas faire beaucoup mieux en changeant d'algorithme.
3. **Ses échecs sont des cas impossibles** : sur les tirages où la CRB permet
   de réussir, elle réussit dans 90 à 100 % des cas.

**Le vrai frein est la source centrale.** Pour 3 sources à
$\sigma = 10^{-3}$, ML + BIC réussit dans **100 %** des cas sans source
centrale et dans **40 %** avec. La source centrale multiplie la CRB par
environ 42.

## 6. Ce que la méthode peut voir

**Une source faible** (une source seule, $\sigma = 10^{-3}$, source centrale) :

| intensité | 0,5 | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| détectée | 25 % | 48 % | 90 % | 100 % | 100 % | 100 % |
| localisée à moins de 1 | 18 % | 22 % | 40 % | 80 % | 98 % | 100 % |

La probabilité de détection suit exactement la prédiction théorique du BIC
(seuil $\mathrm{SNR}\approx 3$). Sur du bruit seul, la méthode invente une
source dans 2 % des cas seulement. Une source d'intensité 2 est presque
toujours vue, mais il en faut environ 10 pour la placer à mieux que 1 près.

**Deux sources proches** (intensité 50 chacune) : elles sont séparées à
partir d'une distance d'environ 1 à $\sigma = 10^{-5}$, 2 à
$\sigma = 10^{-4}$ et 4 à $\sigma = 10^{-3}$. En dessous, la méthode trouve une
seule source au lieu de deux, ce qui vaut mieux que d'en inventer.

## 7. Robustesse

**Bruit mal connu.** La méthode a besoin du niveau de bruit $\sigma$. Le
**surestimer** d'un facteur 2 ne coûte presque rien (on perd seulement les
sources les plus faibles). Le **sous-estimer** d'un facteur 2 fait apparaître
en moyenne une source parasite. En cas de doute, il faut donc prendre
$\sigma$ un peu trop grand.

**Plus de capteurs sur le même cercle.** La CRB baisse comme $1/\sqrt M$ :
quadrupler le nombre de capteurs divise l'erreur par 2. Pour 3 sources à
$\sigma = 10^{-3}$, la réussite passe de 40 % (64 capteurs) à 70 %
(1024 capteurs). Le gain est réel mais lent.

**Capteurs en hauteur, sur la paroi d'un cylindre.** Pas de gain : avec
3 sources, le cercle simple reste le meilleur, et à partir de 5 sources
toutes les géométries échouent, pour le Frank-Wolfe comme pour le ML.

## 8. En 3D

La méthode se généralise sans modification à des sources dans un cube
(`ml3d.py`) : chaque source a alors 4 paramètres, d'où un seuil de
$4\log M$. Sur le cube $[0,10]^3$ avec 160 capteurs et la source centrale :

- avec des capteurs sur une **sphère**, 2 sources sont retrouvées dans 100 %
  des cas à $\sigma = 10^{-4}$ et 88 % à $\sigma = 10^{-3}$, 5 sources dans
  50 % des cas à $\sigma = 10^{-4}$ ;
- l'erreur vaut 0,75 à 0,9 fois la CRB, comme en 2D ;
- le BLASSO 3D du projet échoue dans presque tous les cas ;
- **un seul cercle de capteurs ne suffit pas en 3D** : une source au-dessus
  du cercle et sa symétrique en dessous donnent les mêmes mesures. Il faut
  des capteurs hors d'un même plan (plusieurs cercles, ou une sphère, qui
  est la meilleure disposition).

## 9. Et avec beaucoup de sources ?

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

## 10. Conclusion

| situation (2D, source centrale du projet) | ce que ML + BIC retrouve |
|---|---|
| $\sigma = 10^{-4}$ | 1 à 3 sources de façon fiable, 5 sources une fois sur deux |
| $\sigma = 10^{-3}$ | 1 à 2 sources, 3 sources dans 40 % des cas |
| $\sigma = 10^{-2}$ | 1 source |
| 20 à 50 sources | seulement les quelques sources les plus intenses (section 9) |

1. **ML + BIC est la méthode à utiliser** : la meilleure dans tous les tests,
   sans paramètre à régler, et au niveau de la borne de Cramér–Rao.
2. **Ses limites sont celles des données**, pas de l'algorithme : bruit de la
   source centrale, nombre de sources, sources trop faibles ou trop proches.
3. **Pour aller plus loin, il faut changer les mesures** : réduire le bruit
   de la source centrale (c'est le facteur dominant), rapprocher des
   capteurs des sources, ou ajouter de l'information a priori (positions
   possibles connues, par exemple). La CRB permet d'évaluer chaque option
   avant de la mettre en œuvre.

## 11. Utilisation

```python
from sparse_sources import greedy_ml, cramer_rao, relative_noise_std, make_setup

S = make_setup(h=0.0, normalize=True)       # capteurs du projet
s = relative_noise_std(y_brut, sigma)       # bruit de chaque capteur (source centrale comprise)
r = greedy_ml(m, s, setup=S)                # m : mesures, source centrale retirée
r["z"], r["I"], r["K"]                      # positions, intensités, nombre de sources
r["chi2"] / S.M                             # contrôle : environ (M - 3K) / M
cramer_rao(r["z"], r["I"], s, S)["pos_std"] # précision de chaque source
```

Pour tester la méthode sur un tirage, avec une figure :

```bash
python sparse_sources/tester_ml.py                       # 3 sources, sigma = 1e-3
python sparse_sources/tester_ml.py --sources 5 --sigma 1e-4 --comparer
python sparse_sources/tester_ml.py --dim 3 --capteurs sphere
```

Pour reproduire les résultats :

```bash
pytest sparse_sources/tests                    # 86 tests
python sparse_sources/experiments_projet.py    # comparaison des méthodes (section 5)
python sparse_sources/experiments_ml.py        # sections 6, 7 et 9
python sparse_sources/experiments_cylindre.py  # capteurs sur un cylindre
python sparse_sources/experiments_ml3d.py      # 3D (section 8)
```

## Références

- G. Schwarz, *Estimating the dimension of a model*, The Annals of
  Statistics, 1978 (critère BIC).
- S. M. Kay, *Fundamentals of Statistical Signal Processing*, volumes I
  (estimation, borne de Cramér–Rao) et II (détection), Prentice Hall, 1993
  et 1998.
- V. Duval, G. Peyré, *Exact support recovery for sparse spikes
  deconvolution*, Foundations of Computational Mathematics, 2015 (BLASSO).

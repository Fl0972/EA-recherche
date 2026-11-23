---
title: "Compte rendu : apprentissage automatique et méthodes probabilistes pour localiser les sources"
lang: fr
---

# Apprentissage automatique et méthodes probabilistes pour localiser les sources

Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/apprentissage/`.
Chiffres tirés des CSV de `sparse_sources/apprentissage/results/`.

## En bref

On a cherché à battre la méthode de référence, **maximum de vraisemblance + BIC**
(« ML + BIC », `greedy_ml`), avec de l'apprentissage automatique (régression
logistique, LDA, k plus proches voisins, SVM, arbres, forêts aléatoires,
gradient boosting, perceptrons multicouches, régression ridge à noyau RKHS, ACP)
et avec des méthodes probabilistes (évidence bayésienne, MCMC, recherche
aléatoire). On a aussi étudié, comme demandé, la position des capteurs et la
taille du domaine.

1. **Avec les capteurs du projet, on ne peut presque pas faire mieux que ML + BIC.**
   Même un algorithme « parfait », lancé depuis les vraies positions avec le
   vrai nombre de sources, ne réussit que 52,2 % des cas, contre 48,7 % pour
   ML + BIC. Les échecs viennent des **mesures**, pas de l'algorithme.
2. **Les 3 meilleures solutions** (600 tirages de test) :

   | rang | méthode | réussite | temps / tirage |
   |---|---|---|---|
   | 1 | **ML + BIC + recherche par division** | **50,0 %** | 3,8 s |
   | 2 | **réseau de neurones + ML + BIC** | 49,2 % | ≈ 1 s |
   | 3 | **ML + BIC (référence)**, complété par un **MCMC** pour l'incertitude | 48,7 % | 0,3 s (+ 13 s) |

   La n° 1 est la seule amélioration **statistiquement significative** sur la
   référence (8 tirages gagnés, 0 perdu, test de McNemar p = 0,008). Elle
   atteint exactement le score de l'oracle « départ aux vraies sources + BIC » :
   elle trouve l'optimum global de la vraisemblance.
3. **L'apprentissage pur fait moins bien.** Les classifieurs choisissent mieux
   le nombre de sources que le BIC (63 % de K exacts contre 58 %), mais
   donnent **moins** de reconstructions réussies (43 % contre 48 %). La
   régression directe marche avec 1 source (précision proche du ML, 1000 fois
   plus rapide) mais échoue avec 2 sources (≤ 9 % de réussite contre 83 %).
4. **Le vrai levier est la position des capteurs.** Avec les mêmes 64 capteurs
   et la même méthode, la réussite passe de **49 %** (cercle du projet) à
   **69 %** (capteurs placés autour de D par plan d'expérience optimal) et à
   **79 %** (capteurs dans D). La taille du domaine ne compte pas : seules sa
   forme et la position des capteurs par rapport à lui comptent.

![Réussite des principales méthodes](results/synthese_methodes.png)

*La courbe de la référence ML + BIC est cachée sous celles des autres
méthodes, presque identiques. La courbe grise en pointillés est l'oracle.*

## 1. Protocole

**Scénario.** C'est celui du projet (`COMPTE_RENDU_ML.md`) : sources uniformes
dans D = [0,10] × [0,20], intensités uniformes sur le simplexe de somme 100,
source centrale d'intensité 5000 retirée des mesures (son bruit reste),
64 capteurs sur le cercle circonscrit, bruit relatif σ.

**Simulateur rapide** (`simulation.py`). Il reproduit exactement
`experiments_projet.scenario` (mêmes capteurs, m = y − centrale à 10⁻¹⁴ près)
mais 100 fois plus vite, ce qui permet de fabriquer des bases d'apprentissage
de 150 000 tirages. Il accepte n'importe quelle géométrie (classe `Scene`).

**Bases.**

- entraînement : 3000 tirages (K uniforme dans 1..5, σ ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³, 3·10⁻³}) ;
  150 000 pour le réseau de neurones ;
- test : 600 tirages, **les mêmes pour toutes les méthodes** : K = 1..5 ×
  σ ∈ {10⁻⁴, 10⁻³, 3·10⁻³} × 40 tirages, avec une graine différente.

**Réussite** (même définition que dans les comptes rendus précédents) : bon
nombre de sources, et chaque source à moins de 0,25 × min(séparation, 4)
de sa vraie position.

**Précision des chiffres.** Avec 40 tirages par case, un taux de 50 % est
connu à ± 8 points ; sur les 600 tirages, à ± 2 points. Pour comparer deux
méthodes, on utilise le **test de McNemar** : sur les mêmes tirages, on compte
ceux que l'une réussit et pas l'autre.

## 2. L'idée clé : un oracle pour savoir ce qui est atteignable

Le ML + BIC peut échouer pour trois raisons :

1. **mauvais K** : le BIC choisit le mauvais nombre de sources ;
2. **minimum local** : à K fixé, l'optimisation s'arrête sur une mauvaise solution ;
3. **information absente** : même la meilleure solution de la vraisemblance
   est fausse, parce que le bruit rend deux configurations indiscernables.

Seuls les cas 1 et 2 peuvent être corrigés par un meilleur algorithme. Pour
les mesurer, on a calculé deux **oracles** (qui trichent en utilisant la
vérité, uniquement pour le diagnostic) :

| | réussite |
|---|---|
| ML + BIC (référence) | 48,7 % |
| chemin glouton, **vrai K** imposé | 49,2 % |
| ML lancé **depuis les vraies sources**, K choisi par BIC | 50,0 % |
| ML lancé **depuis les vraies sources**, **vrai K** | 52,2 % |

Conclusion : connaître le vrai K rapporte au plus 2 points, éviter les
minima locaux au plus 1,3 point. **Les ~48 % d'échecs restants sont des cas
de type 3**, où aucune méthode ne peut réussir avec ces mesures. Cela confirme
la conclusion de `COMPTE_RENDU_ML.md` (« ses limites sont celles des
données ») avec un argument direct.

## 3. Les méthodes essayées

Récapitulatif sur les 600 tirages de test (par nombre de sources K, tous σ
confondus) :

| méthode | type | réussite | K=1 | K=2 | K=3 | K=4 | K=5 |
|---|---|---|---|---|---|---|---|
| ML + BIC + division | probabiliste (recherche) | **50,0 %** | 97 | 76 | 47 | 21 | 10 |
| réseau + ML + BIC | apprentissage + ML | 49,2 % | 97 | 76 | 46 | 19 | 8 |
| ML + BIC (référence) | probabiliste | 48,7 % | 97 | 75 | 45 | 19 | 7 |
| ML + évidence de Laplace | bayésien | 46,8 % | 95 | 72 | 42 | 18 | 7 |
| ML + K par SVM | apprentissage | 43,8 % | 100 | 71 | 30 | 12 | 7 |
| ML + K par régression logistique | apprentissage | 43,0 % | 97 | 69 | 32 | 10 | 7 |
| ML + AIC | probabiliste | 36,2 % | | | | | |
| réseau de neurones seul | apprentissage | 21,0 % | 99 | 6 | 0 | 0 | 0 |

### 3.1 Apprendre à choisir le nombre de sources (`selection_k.py`)

**Idée.** Le ML glouton produit un « chemin » : la meilleure solution à
K = 0, 1, …, 7 sources et son χ². Le BIC choisit K avec une règle fixe
(une source doit faire baisser χ² d'au moins 3 log M ≈ 12,5). On remplace
cette règle par un classifieur entraîné sur 3000 simulations. Entrée : le
profil du chemin (log des baisses successives de χ², log χ²/M, log σ). Sortie :
le vrai K.

| règle | K exact | réussite |
|---|---|---|
| régression logistique | **63,2 %** | 43,0 % |
| forêt aléatoire (300 arbres) | 62,5 % | 42,8 % |
| SVM (noyau gaussien) | 61,7 % | 43,8 % |
| LDA | 61,2 % | 41,7 % |
| gradient boosting | 59,8 % | 40,5 % |
| évidence de Laplace | 59,0 % | 46,8 % |
| arbre de décision (profondeur 6) | 58,8 % | 40,8 % |
| **BIC** | 57,8 % | **48,3 %** |
| perceptron multicouche (64, 64) | 56,0 % | 36,2 % |
| k plus proches voisins (k = 15) | 55,3 % | 39,3 % |
| AIC | 52,7 % | 36,2 % |

**Résultat surprenant : deviner mieux K donne de moins bons résultats.** Le
classifieur sait qu'il y a souvent 3 sources quand le chemin en montre 2
clairement, et il le dit. Mais la solution à 3 sources du chemin est alors
fausse (deux sources trop proches, ou une trop faible : cas de type 3). Le
BIC, lui, ne garde que les sources que les mesures montrent vraiment. Pour
réussir, il faut trouver **K et les positions**, et quand les positions sont
inaccessibles, un K exact ne sert à rien. Le BIC est significativement
meilleur que le SVM (32 tirages contre 5, p < 0,001).

**Évidence bayésienne de Laplace** (`chemins.py`). C'est la version
« complète » du BIC : on approche la loi a posteriori par une gaussienne
autour du maximum, ce qui donne

$$\log p(m \mid K) \approx -\tfrac{\chi^2}{2} + \log \pi(\hat\theta) + \tfrac{3K}{2}\log 2\pi - \tfrac12 \log\det F + \log K!$$

(F : matrice de Fisher ; a priori : positions uniformes sur D, intensités
log-uniformes sur [0,1 ; 1000]). En théorie plus juste que le BIC, elle fait un
peu moins bien en pratique (46,8 % contre 48,3 %, p = 0,035), car elle dépend
de l'a priori choisi pour les intensités. Le BIC est robuste justement parce
qu'il ignore l'a priori.

### 3.2 Régression directe : le modèle appris donne les positions (`regression_directe.py`)

Le modèle lit les 64 mesures blanchies et renvoie directement les positions,
à K connu, σ = 10⁻³ (20 000 tirages d'entraînement, 300 de test).

| modèle | erreur médiane K = 1 | réussite K = 2 | temps |
|---|---|---|---|
| ACP (20 composantes) + régression linéaire | 0,76 | 1 % | 0,02 ms |
| k plus proches voisins | 0,045 | 5 % | 0,1 à 10 ms |
| régression ridge à noyau (RKHS) | 0,030 | 4 % | 0,2 ms |
| SVR | **0,025** | 8 % | 2 à 17 ms |
| forêt aléatoire | 0,042 | 6 % | 0,3 à 2 ms |
| gradient boosting | 0,082 | 5 % | 1,2 ms |
| perceptron multicouche (256, 256) | 0,044 | 9 % | **0,02 ms** |
| **maximum de vraisemblance** | **0,022** | **83 %** | 26 à 56 ms |
| borne de Cramér-Rao (médiane) | 0,032 | | |

- **Avec 1 source, l'apprentissage marche** : la SVR et le noyau RKHS sont
  presque aussi précis que le ML, pour 100 à 1000 fois moins de calcul. Le
  problème est alors une fonction lisse des mesures, ce que ces modèles
  apprennent bien. La relation n'est pas linéaire : l'ACP + régression
  linéaire se trompe de 0,76.
- **Avec 2 sources, tout échoue.** La fonction « mesures → positions » n'est
  plus simple : il faut décider quelle source est la n° 1 (on les a rangées par
  intensité, mais deux sources d'intensités proches s'échangent brutalement),
  et l'espace à couvrir a 6 dimensions au lieu de 3. 20 000 exemples ne
  suffisent pas, alors que le ML, qui connaît la physique, n'a besoin
  d'aucun exemple.
- **ACP des mesures** : 7 composantes portent 91 % de la variance, 11 en
  portent 97 %. C'est cohérent avec la section 9 de `COMPTE_RENDU_ML.md` :
  les mesures contiennent peu de « directions » indépendantes.

### 3.3 Réseau de neurones + maximum de vraisemblance (`reseau.py`, `hybride.py`)

**Le réseau.** Un perceptron multicouche (3 couches de 512 neurones, ReLU,
PyTorch) lit les mesures et prédit une **carte de présence** des sources sur
une grille de pas 0,5 (une bosse gaussienne par source, comme en microscopie
de localisation), plus le nombre de sources. Il est entraîné sur
150 000 simulations (31 min sur CPU). Les maxima locaux de la carte donnent
des positions candidates.

**Utilisation.** Pour chaque K, on garde la meilleure (plus petit χ²) de trois
solutions : le chemin glouton, le ML lancé depuis les K plus hauts pics du
réseau, et le ML lancé depuis la solution K − 1 plus un pic du réseau. K est
ensuite choisi par BIC.

| | réussite |
|---|---|
| réseau seul (pics, sans ML) | 21,0 % |
| réseau + ML, K donné par le réseau | 33,8 % |
| réseau + ML + BIC | **49,2 %** |
| réseau + ML + K par gradient boosting / forêt / MLP | 41,2 / 43,0 / 37,2 % |

- Le réseau seul n'est utile que pour 1 source. Pour 2 sources et plus, sa
  carte est floue : il voit « où il y a du signal », pas assez précisément.
- Ses propositions sont retenues dans 19 % des cas (6 % directement, 13 %
  combinées avec le glouton). Gain : 3 tirages gagnés, 0 perdu, **non
  significatif** (p = 0,25).
- Le réseau prédit K moins bien (54 %) que les classifieurs de la section 3.1.

### 3.4 Recherche stochastique du maximum de vraisemblance (`recherche.py`)

On s'attaque aux minima locaux (cause 2). À chaque K, on lance le ML depuis
plusieurs points de départ et on garde le meilleur χ² :

- **glouton** : solution K − 1 + une source au maximum de corrélation (comme `greedy_ml`) ;
- **division** : solution K − 1 où une source est coupée en deux sources
  voisines (écart ± 0,7 selon x ou y). Cela sépare deux sources proches que le
  glouton avait prises pour une seule ;
- **a priori** : solution K − 1 + une source tirée au hasard dans D ;
- **réseau** : K positions tirées au hasard dans la carte du réseau, vue comme
  une loi de probabilité.

| variante | réussite | ajustements / tirage | temps |
|---|---|---|---|
| glouton + division | **50,0 %** | 21 | 3,8 s |
| glouton + réseau (8 tirages) | 50,0 % | 40 | 4,0 s |
| tout (division + a priori + réseau) | 50,0 % | 92 | 8,6 s |

Les trois variantes atteignent **exactement le score de l'oracle** « départ
aux vraies sources + BIC » (50,0 %, mêmes tirages réussis) : elles trouvent
l'optimum global de la vraisemblance. La division seule suffit et c'est la
moins chère. Le gain sur la référence se concentre sur 4 et 5 sources
(K = 5, σ = 10⁻⁴ : 17 % → 25 %).

### 3.5 Incertitude par MCMC (`mcmc.py`)

**Idée.** Au lieu d'une seule réponse, on décrit toute la loi a posteriori
des sources : la vraisemblance gaussienne fois l'a priori (positions
uniformes, intensités log-uniformes). On en tire des échantillons avec
l'algorithme de **Metropolis** : on propose un petit déplacement aléatoire
de tous les paramètres, on l'accepte avec probabilité
min(1, π(nouveau)/π(ancien)), et on recommence 40 000 fois. La covariance des
propositions est adaptée pendant la chauffe (Haario et al.) pour viser 23 %
d'acceptation.

**Ce qu'on mesure : la calibration.** Pour chaque source, on calcule
l'ellipse qui contient 90 % des échantillons. Si l'incertitude est juste,
90 % des vraies sources doivent tomber dedans. On compare à l'ellipse de
Cramér-Rao (approximation gaussienne locale). Mesure faite sur 110 tirages de
test où ML + BIC trouve le bon K (248 sources).

| | couverture MCMC (visée : 90 %) | couverture Cramér-Rao |
|---|---|---|
| 1 source | **89 %** | 89 % |
| 2 sources | 83 % | 88 % |
| 3 sources | 71 % | 81 % |
| 4 sources | 54 % | 71 % |
| 5 sources | 60 % | 73 % |
| σ = 10⁻³ | **92 %** | 92 % |
| toutes | 72 % | 81 % |

- Pour 1 à 2 sources et σ = 10⁻³, l'incertitude est **bien calibrée** : on peut
  donner à l'utilisateur une zone de confiance fiable autour de chaque source.
- Pour 3 sources et plus, les deux ellipses sont **trop petites**. La loi a
  posteriori a alors plusieurs modes (plusieurs configurations expliquent
  presque aussi bien les mesures). La chaîne, partie de la solution ML, reste
  dans un seul mode et ne voit pas les autres.
- Le MCMC voit tout de même ce que Cramér-Rao ne voit pas : sur un tirage à
  4 sources, il donne un écart type de 3,0 à une source mal placée (erreur
  6,9), signe clair qu'elle n'est pas fiable.
- Coût : 10 à 17 s par tirage.

**Correction en cours de route.** La première version adaptait mal la
covariance des propositions : la chaîne ne bougeait presque plus
(acceptation de 69 %, couverture de 54 %). La version corrigée règle l'échelle
pendant toute la chauffe. Seuls les chiffres corrigés figurent ici.

## 4. Les 3 meilleures solutions

**1. ML + BIC + recherche par division** (`recherche.global_search(d, split=True, r_prior=0, r_nn=0)`).
La méthode de référence, plus un mouvement : à chaque ajout de source, on
essaie aussi de couper chaque source existante en deux. C'est la meilleure
méthode (50,0 %), la seule amélioration significative, et elle atteint
l'optimum global de la vraisemblance. Coût : 12 fois celui de `greedy_ml`
(3,8 s par tirage). Aucun apprentissage, aucun réglage.

**2. Réseau de neurones + ML + BIC** (`hybride.py`). Un réseau entraîné une
fois sur 150 000 simulations propose des positions de départ, que le ML
corrige. 49,2 % : aussi bon que la référence, légèrement mieux mais sans que
ce soit significatif. C'est la bonne architecture si on veut traiter
beaucoup de tirages : l'apprentissage coûte une fois, la prédiction est
instantanée. Mais il faut réentraîner le réseau si la géométrie change.

**3. ML + BIC (référence) + MCMC pour l'incertitude.** Le ML + BIC reste le
meilleur compromis précision/temps (48,7 %, 0,3 s). Le MCMC (section 3.5) y
ajoute une zone de confiance fiable pour 1 à 2 sources, et signale les
sources mal déterminées.

**À ne pas utiliser** : l'apprentissage pur (régression directe ou réseau
seul) dès 2 sources, et les classifieurs à la place du BIC pour choisir K.

## 5. Position des capteurs et taille du domaine (`geometrie.py`)

C'est ici que se trouvent les vrais gains. Pour chaque géométrie (toujours
64 capteurs et la source centrale), on mesure la borne de Cramér-Rao médiane
(3 sources, σ = 10⁻³ : l'erreur minimale possible, quel que soit
l'algorithme) et la réussite de ML + BIC (K ∈ {2, 3, 5}, σ ∈ {10⁻⁴, 10⁻³},
180 tirages par géométrie).

![Géométries testées](results/geometrie_capteurs.png)

### 5.1 Où mettre les capteurs

| géométrie (64 capteurs) | Cramér-Rao | réussite | K=3, σ=10⁻³ | K=5, σ=10⁻⁴ |
|---|---|---|---|---|
| D-optimal partout | 0,06 | **79 %** | 70 % | 80 % |
| aléatoires dans D | 0,11 | **79 %** | 73 % | 77 % |
| grille 8 × 8 dans D | 0,05 | **79 %** | 73 % | 80 % |
| **D-optimal hors de D** | 0,42 | **69 %** | 57 % | 53 % |
| bord de D, marge 0,5 | 0,42 | 65 % | 60 % | 50 % |
| bord de D, marge 2 | 1,38 | 56 % | 47 % | 23 % |
| ellipse passant par les coins | 1,07 | 55 % | 50 % | 20 % |
| **cercle circonscrit (projet)** | 1,75 | **49 %** | 30 % | 17 % |
| deux cercles (× 1 et × 1,5) | 2,20 | 49 % | 33 % | 13 % |
| cercle × 0,8 (coupe les coins de D) | 0,46 | 46 % | 37 % | 23 % |
| cercle × 1,5 | 18,1 | 32 % | 7 % | 0 % |
| cercle × 2 | 73,9 | 18 % | 0 % | 0 % |

![Synthèse géométrie](results/synthese_geometrie.png)

- **Il faut rapprocher les capteurs des sources.** Le signal décroît en 1/r² :
  un capteur deux fois plus proche reçoit 4 fois plus de signal, et les
  variations de position y sont beaucoup plus visibles. Éloigner le cercle
  (× 1,5, × 2) est catastrophique ; le cercle du projet est loin des
  grands côtés de D (jusqu'à 6 unités).
- **Plan d'expérience bayésien D-optimal.** On choisit les 64 capteurs un par
  un, parmi 2064 positions candidates, pour maximiser en moyenne sur
  200 tirages a priori log det(Fisher) (le volume de l'ellipse d'incertitude
  est inversement proportionnel à √det F). Hors de D, l'algorithme
  **retrouve tout seul** l'idée de coller les capteurs au rectangle : il les
  place sur les grands côtés, au plus près, et évite les coins. Gain :
  réussite de 49 % à 69 % (K = 3, σ = 10⁻³ : 30 % à 57 %).
- **Des capteurs dans le domaine** divisent la borne de Cramér-Rao par 15 à 30
  et portent la réussite à 79 %. Leur disposition exacte compte peu
  (grille, hasard ou D-optimal). Reste à savoir si c'est physiquement possible.
- Le cercle × 0,8 a une borne de Cramér-Rao faible mais une réussite faible :
  des capteurs à l'intérieur des coins de D sont parfois presque sur une
  source, ce qui rend l'optimisation difficile.

### 5.2 Combien de capteurs

| capteurs | 16 | 32 | 64 | 128 | 256 | D-optimal hors D, **32** |
|---|---|---|---|---|---|---|
| Cramér-Rao | 6,80 | 3,25 | 1,75 | 1,34 | 0,76 | **0,63** |
| réussite | 38 % | 46 % | 49 % | 59 % | 59 % | **64 %** |

**32 capteurs bien placés valent mieux que 256 capteurs sur le cercle.**

### 5.3 Taille et forme du domaine

| domaine (capteurs sur le cercle circonscrit) | Cramér-Rao | Cramér-Rao / √aire | réussite |
|---|---|---|---|
| 5 × 10 | 0,88 | 0,124 | 51 % |
| 10 × 20 (projet) | 1,75 | 0,124 | 49 % |
| 20 × 40 | 3,51 | 0,124 | 47 % |
| 40 × 80 | 7,02 | 0,124 | 41 % |
| carré 14,1 × 14,1 (même aire) | 1,42 | 0,100 | 56 % |
| allongé 5 × 40, cercle | 6,36 | 0,450 | 38 % |
| allongé 5 × 40, **ellipse** | 0,25 | 0,018 | 73 % |
| très allongé 2,5 × 80, cercle | 18,4 | 1,30 | 29 % |
| très allongé 2,5 × 80, **ellipse** | 0,13 | 0,009 | 59 % |

- **La taille ne compte pas.** Si on multiplie toutes les longueurs par a
  (domaine et capteurs), toutes les mesures sont divisées par a², source
  centrale comprise. Le bruit étant relatif, les mesures blanchies m/s ne
  changent pas : c'est exactement le même problème, et seules les erreurs
  sont multipliées par a. Le calcul le confirme : la borne de Cramér-Rao
  double exactement quand la taille double (Cramér-Rao / √aire = 0,124
  constant). La baisse de réussite de 51 % à 41 % est un **artefact du
  critère** : sa tolérance est plafonnée à 0,25 × 4 = 1 unité, une longueur
  absolue.
- **La forme compte.** Un domaine allongé entouré d'un cercle est une très
  mauvaise configuration (la plupart des capteurs sont loin). Avec des
  capteurs sur une ellipse qui épouse le domaine, c'est l'inverse : le
  domaine 5 × 40 donne 73 % de réussite.

### 5.4 Les deux leviers ensemble (`combinaison.py`)

Même 180 tirages par géométrie :

| géométrie | ML + BIC | ML + BIC + division |
|---|---|---|
| cercle (projet) | 49,4 % | 51,1 % |
| bord de D, marge 0,5 | 65,0 % | 63,9 % |
| D-optimal hors de D | 69,4 % | **70,0 %** |

Sur une bonne géométrie, la division n'apporte plus rien de mesurable : les
minima locaux deviennent rares quand les mesures sont informatives.
**Changer les capteurs apporte +20 points ; changer d'algorithme, +1 à 2 points.**

## 6. Limites et remarques

- Les chiffres par case (40 tirages) ont une incertitude de ± 8 points. Les
  comparaisons entre méthodes reposent sur les 600 tirages et sur le test de
  McNemar.
- Les méthodes apprises sont entraînées pour **cette** géométrie et **cette**
  loi des sources (K ≤ 5, σ dans une plage donnée). Elles sont avantagées par
  rapport au BIC, qui n'utilise aucune de ces informations, et font pourtant
  moins bien.
- Le MCMC n'explore qu'un mode de la loi a posteriori. Le tempering parallèle
  (plusieurs chaînes à différentes « températures » qui échangent leurs
  états) serait l'étape suivante pour calibrer les cas à 3 sources et plus.
  On n'a pas essayé le MCMC à saut réversible (qui fait aussi varier K), plus
  complexe.
- Le plan D-optimal optimise la borne de Cramér-Rao pour 3 sources ; il n'a pas
  été réoptimisé pour d'autres K.
- Interprétation de la demande : « RKM » a été compris comme les méthodes à
  noyau (RKHS : régression ridge à noyau, SVR) et « SFM » comme les SVM.

## 7. Reproduire

```bash
python -m sparse_sources.apprentissage.donnees            # bases + chemins ML (~9 min)
python -m sparse_sources.apprentissage.selection_k        # 3.1  (~30 s)
python -m sparse_sources.apprentissage.regression_directe # 3.2  (~10 min)
python -m sparse_sources.apprentissage.hybride            # 3.3  (réseau : ~30 min la 1re fois, puis ~10 min)
python -m sparse_sources.apprentissage.recherche          # 3.4  (~25 min)
python -m sparse_sources.apprentissage.mcmc               # 3.5  (~4 min)
python -m sparse_sources.apprentissage.geometrie          # 5    (~12 min)
python -m sparse_sources.apprentissage.combinaison        # 5.4  (~6 min)
python -m sparse_sources.apprentissage.figures            # figures de synthèse
```

Les fichiers `.pkl` (bases, chemins) et `.pt` (réseau) sont des caches
ignorés par git ; les CSV et figures sont dans `results/`.

| fichier | contenu |
|---|---|
| `simulation.py` | simulateur rapide, classe `Scene` (géométrie quelconque) |
| `base.py` | `Fitter` (ML à K fixé, chemin glouton), évaluation, base de test, calcul parallèle |
| `chemins.py` | chemins ML mis en cache, Fisher, évidence de Laplace |
| `selection_k.py` | choix de K : BIC, AIC, Laplace, 8 classifieurs |
| `regression_directe.py` | régression directe (7 modèles), ACP |
| `reseau.py` | perceptron multicouche PyTorch : carte de présence + K |
| `hybride.py` | réseau + ML, oracles |
| `recherche.py` | recherche stochastique (division, a priori, réseau) |
| `mcmc.py` | Metropolis adaptatif, calibration des incertitudes |
| `geometrie.py` | capteurs, nombre, domaine, plan D-optimal |
| `combinaison.py` | meilleure géométrie + meilleure méthode |

## Références

- G. Schwarz, *Estimating the dimension of a model*, Annals of Statistics, 1978 (BIC).
- C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006 :
  §4.4 (approximation de Laplace), ch. 5 (réseaux), 6–7 (noyaux, SVM), 11 (MCMC), 14 (arbres).
- H. Haario, E. Saksman, J. Tamminen, *An adaptive Metropolis algorithm*, Bernoulli, 2001.
- K. Chaloner, I. Verdinelli, *Bayesian experimental design: a review*, Statistical Science, 1995 (plans D-optimaux).
- S. M. Kay, *Fundamentals of Statistical Signal Processing*, vol. I, 1993 (Cramér-Rao).
- E. Nehme et al., *Deep-STORM: super-resolution single-molecule microscopy by deep learning*, Optica, 2018 (cartes de présence apprises).

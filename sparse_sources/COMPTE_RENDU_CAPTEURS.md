---
title: "Compte rendu : où placer les capteurs pour retrouver beaucoup de sources"
subtitle: "K imposé, cercle agrandi, capteurs à l'intérieur du domaine"
lang: fr
---

# Compte rendu : où placer les capteurs

Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/`. Suite de
`COMPTE_RENDU_ML.md` (section 9 : « Et avec beaucoup de sources ? »).

Avec les capteurs du projet, le maximum de vraisemblance + BIC (« ML + BIC »)
ne retrouve que quelques sources sur 50. Ce compte rendu répond à trois
questions :

1. Peut-on faire mieux en **imposant** à la méthode qu'il y a 50 sources ?
2. Peut-on faire mieux en **agrandissant le cercle** de capteurs ?
3. Peut-on faire mieux en **plaçant des capteurs où l'on veut**, en
   particulier à l'intérieur du domaine ?

**Réponses courtes :** non, non, oui. Imposer K et agrandir le cercle
n'ajoutent aucune information, ou en enlèvent. Mettre des capteurs à
l'intérieur du domaine rend le problème soluble, même pour 50 sources : avec
une grille de 576 capteurs, ML + BIC retrouve 45 sources sur 50, sans
connaître leur nombre.

Rappel du cadre : sources dans $D = [0,10]\times[0,20]$, signal en
$1/\lVert x_i - z_k\rVert^2$, source centrale connue d'intensité 5000 au
centre $(5, 10)$, retirée des mesures mais dont le bruit reste, bruit relatif
$\sigma$. Les sources sont tirées comme dans `config/` : positions uniformes
dans $D$, intensités uniformes sur le simplexe de somme 100.

## 1. Imposer K = 50

**L'idée.** Si l'on sait qu'il y a 50 sources, on peut sauter le BIC et
demander directement la configuration de 50 sources la plus vraisemblable
(option `--k-fixe 50` de `tester_ml.py`).

**Le test.** 50 sources, 64 capteurs du projet, $\sigma = 10^{-4}$. On compare
ML + BIC, ML avec K imposé, et ML avec K imposé **en partant des vraies
positions** (le cas le plus favorable possible).

| méthode | sources estimées | $\chi^2/M$ | vraies sources retrouvées à moins de 0,5 |
|---|---|---|---|
| ML + BIC (K libre) | 13 | 0,60 | **9** |
| ML, K = 50 imposé | 49 | 0,26 | 6 |
| ML, K = 50 imposé, départ sur la vérité | 49 | 0,26 | 5 |

(Un seul tirage : chaque reconstruction prend plusieurs minutes. Une des
50 sources est fusionnée avec une voisine pendant l'optimisation, d'où 49.)

**Ce qu'on en tire.**

- **Imposer K fait pire que le BIC.** Les sources en trop ne servent pas à
  retrouver les vraies : elles servent à coller au bruit. On le voit au
  $\chi^2/M$ de 0,26, alors qu'une estimation honnête donne environ
  $(M - 3K)/M$, ici proche de 1 tant que $3K < M$.
- **Même en partant de la vraie solution, l'optimisation s'en éloigne.** Elle
  trouve une configuration fausse qui explique mieux les mesures que la
  vérité. La vraie solution n'est donc pas le maximum de vraisemblance :
  l'algorithme n'est pas en cause, ce sont les données qui ne suffisent pas.
- **La raison** : 150 inconnues ($x$, $y$, $I$ pour chaque source) pour 64
  mesures. Le système n'a pas de solution unique, même sans bruit. Connaître
  K ajoute une seule information.

## 2. Ce que mesure une disposition de capteurs

Deux outils, déjà utilisés dans `COMPTE_RENDU_ML.md`, permettent de juger une
disposition de capteurs **avant** de lancer une reconstruction.

**Les directions mesurables** (sans bruit). Avec K sources, il y a $3K$
paramètres. On compte combien de « directions » de cet espace les capteurs
mesurent vraiment : ce sont les valeurs singulières de la matrice des
dérivées du modèle (colonnes normalisées) qui dépassent $10^{-6}$ fois la
plus grande. S'il en manque, certaines combinaisons de paramètres sont
invisibles, quel que soit le bruit.

**La borne de Cramér–Rao (CRB)**, avec le bruit du projet (source centrale
comprise). C'est l'erreur de position minimale que n'importe quelle méthode
peut atteindre. Une source dont la CRB dépasse sa distance aux autres sources
ne peut pas être localisée de façon fiable.

Dispositions comparées (expérience `geometrie` de `experiments_ml.py`,
`results/ml_geometrie.csv`, 5 tirages par cas) :

| disposition | capteurs $M$ |
|---|---|
| cercle du projet (cercle circonscrit à $D$) | 64 |
| cercle de rayon 2 fois plus grand | 64 |
| cercle du projet, 16 fois plus de capteurs | 1024 |
| cercle + grille 4×8 dans $D$ | 96 |
| cercle + grille 8×16 dans $D$ | 192 |
| cercle + grille 16×32 dans $D$ | 576 |

Les grilles placent un capteur au centre de chaque case d'un découpage de
$D$ : 4×8 donne un capteur tous les 2,5, 8×16 tous les 1,25, 16×32 tous les
0,625. Aucun capteur ne tombe sur la source centrale.

## 3. Agrandir le cercle fait pire

**Directions mesurables** (médiane sur 5 tirages, il en faut $3K$) :

| disposition | K = 3 (9) | K = 10 (30) | K = 20 (60) | K = 50 (150) |
|---|---|---|---|---|
| cercle du projet, 64 | 9 | 29 | 46 | 56 |
| cercle rayon × 2, 64 | 9 | 22 | 27 | 31 |
| cercle, 1024 capteurs | 9 | 29 | 47 | 71 |

**CRB médiane sur la position** ($\sigma = 10^{-3}$) :

| disposition | K = 3 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|
| cercle du projet, 64 | 0,15 | 350 | 7 400 | 14 000 |
| cercle rayon × 2, 64 | 9,3 | 94 000 | 430 000 | 160 000 |
| cercle, 1024 capteurs | 0,037 | 86 | 830 | 2 800 |

(Une CRB de plusieurs centaines signifie que la position est indéterminée :
le domaine ne mesure que 10 × 20.)

- **Agrandir le cercle dégrade tout.** Vu de plus loin, le signal en $1/r^2$
  est encore plus lisse : les sources se confondent. Même pour 3 sources, la
  CRB passe de 0,15 à 9,3. Sur le tirage par défaut de `tester_ml.py`
  (3 sources, $\sigma = 10^{-3}$), ML + BIC trouve 3 sources avec le cercle
  du projet, mais seulement 2, mal placées, avec le cercle agrandi.
- **Ajouter des capteurs sur le cercle aide peu.** 16 fois plus de capteurs
  divisent la CRB par 4 environ (le bruit se moyenne), mais le nombre de
  directions mesurables plafonne à 71 pour 50 sources : des capteurs
  supplémentaires sur le même cercle répètent presque la même information.

## 4. Des capteurs à l'intérieur du domaine résolvent le problème

**Directions mesurables** (médiane, il en faut $3K$) :

| disposition | K = 3 (9) | K = 10 (30) | K = 20 (60) | K = 50 (150) |
|---|---|---|---|---|
| cercle du projet, 64 | 9 | 29 | 46 | 56 |
| cercle + grille 4×8, 96 | 9 | **30** | 58 | 80 |
| cercle + grille 8×16, 192 | 9 | **30** | **60** | 144 |
| cercle + grille 16×32, 576 | 9 | **30** | **60** | **150** |

**CRB médiane sur la position :**

| disposition | $\sigma$ | K = 3 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|---|
| cercle du projet, 64 | $10^{-4}$ | 0,015 | 35 | 740 | 1 400 |
|  | $10^{-3}$ | 0,15 | 350 | 7 400 | 14 000 |
| cercle + grille 4×8, 96 | $10^{-4}$ | 0,0015 | 0,056 | 2,8 | 23 |
|  | $10^{-3}$ | 0,015 | 0,56 | 28 | 230 |
| cercle + grille 8×16, 192 | $10^{-4}$ | 0,0002 | 0,0022 | 0,0063 | 1,5 |
|  | $10^{-3}$ | 0,0024 | 0,022 | 0,063 | 15 |
| cercle + grille 16×32, 576 | $10^{-4}$ | 0,00005 | 0,0002 | 0,0005 | **0,0021** |
|  | $10^{-3}$ | 0,0005 | 0,0022 | 0,0049 | **0,021** |

**Part des sources dont la CRB est inférieure à 0,1** (moyenne sur
5 tirages, $\sigma = 10^{-3}$) :

| disposition | K = 3 | K = 10 | K = 20 | K = 50 |
|---|---|---|---|---|
| cercle du projet, 64 | 40 % | 2 % | 1 % | 0 % |
| cercle, 1024 capteurs | 53 % | 18 % | 5 % | 1 % |
| cercle + grille 4×8, 96 | 93 % | 24 % | 4 % | 0 % |
| cercle + grille 8×16, 192 | 100 % | 74 % | 61 % | 3 % |
| cercle + grille 16×32, 576 | 100 % | 98 % | 99 % | **84 %** |

Ce qu'on en tire :

- **C'est la proximité qui compte, pas le nombre de capteurs.** La grille 4×8
  (96 capteurs) fait mieux que 1024 capteurs sur le cercle : la CRB de
  3 sources est 2,5 fois plus petite, celle de 10 sources 150 fois plus
  petite. Un capteur proche d'une source la voit séparément de ses voisines.
- **Il faut environ un capteur par source, à une distance comparable à
  l'écart entre sources.** Avec 50 sources tirées au hasard, les sources sont
  en moyenne à environ 1,1 de leur plus proche voisine, et la plus petite
  distance est de l'ordre de 0,15. La grille 8×16 (pas de 1,25) suffit
  jusqu'à 20 sources ; pour 50 sources, il faut la grille 16×32 (pas de
  0,625).
- **Avec la grille 16×32, le problème à 50 sources devient bien posé** : les
  150 paramètres sont mesurés et la CRB médiane vaut 0,02 à
  $\sigma = 10^{-3}$, soit 50 fois moins que l'écart moyen entre sources.
- **La source centrale reste le premier facteur de bruit.** Les capteurs
  proches du centre reçoivent beaucoup de son bruit. Avec la grille 16×32 et
  50 sources ($\sigma = 10^{-3}$), la CRB médiane vaut 0,021 avec la source
  centrale et 0,0008 sans, soit 26 fois moins. Elle n'empêche plus la
  localisation, mais la réduire reste le meilleur gain possible après le
  placement des capteurs.

## 5. Ce que ML + BIC retrouve avec des capteurs dans le domaine

Les sections 3 et 4 disent ce que les mesures **permettent** de retrouver.
Voici ce que ML + BIC retrouve **vraiment**, sans connaître K ($\sigma =
10^{-3}$, source centrale, $k_{\max} = K + 15$). Les sources sont tirées
comme dans l'expérience `beaucoup` (graine 0). Il n'y a qu'un tirage par
cas : chaque reconstruction de 50 sources prend plus de 20 minutes.

| cas | sources estimées | $\chi^2/M$ | retrouvées à moins de 0,5 | retrouvées à moins de 0,1 | sources à CRB < 0,1 | temps |
|---|---|---|---|---|---|---|
| 20 sources, cercle du projet (64)¹ | 6 | | 3 | | ≈ 0 | 1 s |
| 20 sources, cercle + grille 8×16 (192) | 18 | 0,84 | **17** | 14 | 13 | 42 s |
| 50 sources, cercle du projet (64)¹ | 7 | | 4 | | ≈ 0 | 1,5 s |
| 50 sources, cercle + grille 16×32 (576) | 46 | 1,19 | **45** | 40 | 42 | 23 min |

¹ Médiane sur 5 tirages, d'après la section 9 de `COMPTE_RENDU_ML.md`.

- **Avec des capteurs dans le domaine, ML + BIC retrouve presque toutes les
  sources.** 45 sources sur 50 à moins de 0,5 près, 40 à moins de 0,1 près,
  contre 4 avec le cercle du projet. Pour 20 sources : 17 sur 20 contre 3.
- **La méthode atteint ce que la CRB prévoit.** Le nombre de sources
  retrouvées à moins de 0,1 près (40 et 14) est très proche du nombre de
  sources dont la CRB est inférieure à 0,1 (42 et 13). Les sources manquées
  sont celles que les mesures ne permettent pas de voir : les plus faibles
  (pour 50 sources, les intensités tirées vont de 0,02 à 7,7) ou celles qui
  sont collées à une voisine.
- **Le BIC reste prudent.** Il estime 46 sources au lieu de 50 et 18 au lieu
  de 20 : il retire les sources que les mesures ne justifient pas, au lieu
  d'en inventer. $\chi^2/M$ vaut 0,84 et 1,19, proche de la valeur attendue
  $(M - 3K)/M$ (0,69 et 0,74) : le modèle explique les mesures, à quelques
  sources faibles près.
- **Le coût augmente vite avec K.** 42 s pour 20 sources et 192 capteurs,
  23 min pour 50 sources et 576 capteurs : chaque ajout fait glisser toutes
  les sources en même temps.

Le même essai se lance sur un tirage de `config/` avec
`python sparse_sources/tester_ml.py --sources 50 --sigma 1e-3 --interieur 16x32 --k-max 65`.

## 6. Un capteur mobile que l'on place où l'on veut

**Le dispositif.** On garde les 64 capteurs du cercle et on ajoute un seul
capteur que l'on peut déplacer dans $D$. Il fait $N$ mesures, une par
position, avec le même bruit que les autres capteurs (bruit relatif
$\sigma$, source centrale comprise). La question est : **où faire ces $N$
mesures ?**

**Trois stratégies**, avec les mêmes positions possibles (centres des cases
de pas 0,25 de $D$) :

- **au hasard** : $N$ positions tirées uniformément ;
- **grille** : une grille régulière de $N$ points fixée à l'avance
  (2×4, 4×8, 6×12, 8×16) ;
- **adaptatif** : on alterne mesure et estimation (module `mobile.py`).
  1. ML + BIC estime les sources à partir des mesures déjà faites.
  2. On choisit les 8 positions suivantes une à une. Chacune est celle qui
     fait le plus baisser la somme des variances de Cramér–Rao des
     positions estimées (plan « A-optimal »). Le bruit prévu à chaque
     position tient compte de la source centrale.
  3. On mesure à ces 8 positions, on relance ML + BIC en partant de
     l'estimation précédente, et on recommence.

**Résultats** (expérience `mobile` de `experiments_ml.py`, `results/ml_mobile.csv`,
`ml_mobile.png`, $\sigma = 10^{-3}$, 5 tirages, moyenne du nombre de vraies
sources retrouvées à moins de 0,1 près) :

| sources | stratégie | $N = 0$ | $N = 8$ | $N = 32$ | $N = 72$ | $N = 128$ |
|---|---|---|---|---|---|---|
| 10 | au hasard | 1,4 | 1,8 | 2,8 | 5,2 | 7,6 |
|  | grille | 1,4 | 1,6 | 4,2 | 5,8 | 6,6 |
|  | **adaptatif** | 1,4 | 2,4 | **6,6** | **8,6** | **9,0** |
| 20 | au hasard | 1,8 | 1,4 | 3,0 | 6,6 | 11,6 |
|  | grille | 1,8 | 2,0 | 2,8 | 5,4 | 11,0 |
|  | **adaptatif** | 1,8 | 2,4 | **5,2** | **12,2** | **17,4** |

($N = 0$ : les 64 capteurs du cercle seuls.)

- **Choisir où mesurer compte autant que le nombre de mesures.** Pour
  20 sources, 72 mesures adaptatives valent mieux que 128 mesures au hasard
  ou en grille (12,2 sources contre 11 à 11,6). Avec 128 mesures, la
  stratégie adaptative retrouve 17,4 sources sur 20 à moins de 0,1 près,
  contre 11 à 11,6 pour les autres.
- **Pour 10 sources, 72 mesures adaptatives suffisent** : 8,6 sources sur
  10 retrouvées à 0,1 près, et 9,0 avec 128 mesures.
- **Le capteur va se placer tout contre les sources.** Sur trois tirages
  vérifiés, la distance médiane entre une position choisie et la vraie
  source la plus proche vaut 0,2 à 0,4, contre environ 2 pour une position
  quelconque de $D$. Il revient aussi souvent au même endroit
  (69 positions distinctes sur 128 mesures pour 10 sources) : répéter une
  mesure là où elle est la plus informative moyenne le bruit.
- **Les sources manquées sont surtout les plus faibles.** Pour 10 sources,
  c'est à chaque fois la plus faible (intensité 0,44 ou 0,75, pour une
  moyenne de 10). Pour 20 sources, 3 des 4 sources manquées ont une
  intensité inférieure à 1,6.
- **Le gain est faible pour les toutes premières mesures** ($N = 8$) : tant
  que ML + BIC n'a vu que quelques sources, le capteur affine celles-là et
  ne découvre pas les autres. Il faut quelques paquets de mesures pour que
  de nouvelles sources apparaissent.
- **Coût** : la stratégie adaptative relance ML + BIC après chaque paquet de
  8 mesures, soit 16 reconstructions pour 128 mesures. En tout, cela prend
  30 s pour 10 sources et 12 min pour 20 sources, contre quelques secondes à
  2,5 min pour une seule reconstruction avec 128 mesures fixées à l'avance.

## 7. Un capteur qui enregistre en continu le long d'un trajet

**Le dispositif.** Le capteur mobile parcourt $D$ en serpentin :
aller-retour le long de lignes parallèles au grand côté, régulièrement
espacées. Il enregistre en continu, ce qu'on modélise par une mesure tous
les 0,25 (ou 0,1) le long du trajet, chacune avec son propre bruit. Les
64 capteurs du cercle sont gardés. Le trajet est fixé à l'avance : ce n'est
pas une stratégie adaptative.

**Ce que les mesures permettent, pour 50 sources** (expérience `continu`,
`results/ml_continu_crb.csv`, $\sigma = 10^{-3}$, 5 tirages) :

| capteur mobile, en plus du cercle | mesures | longueur du trajet | directions (sur 150) | CRB médiane | sources à CRB < 0,1 |
|---|---|---|---|---|---|
| aucun (cercle seul) | 0 | | 56 | 14 000 | 0 % |
| grille fixe 16×32 (rappel, section 4) | 512 | | 150 | 0,021 | 84 % |
| serpentin 4 lignes, une mesure tous les 0,25 | 320 | 87 | 150 | 0,10 | 47 % |
| serpentin 8 lignes, une mesure tous les 0,25 | 640 | 167 | 150 | **0,012** | 82 % |
| serpentin 8 lignes, une mesure tous les 0,1 | 1 600 | 168 | 150 | 0,0047 | **94 %** |
| serpentin 16 lignes, une mesure tous les 0,25 | 1 280 | 325 | 150 | 0,0035 | **94 %** |

- **Un seul capteur qui se déplace en enregistrant suffit à rendre le
  problème bien posé.** Dès 4 lignes, les 150 paramètres de 50 sources sont
  mesurés.
- **8 lignes valent presque la grille fixe de 512 capteurs** : même proportion de
  sources localisables (82 % contre 84 %), CRB médiane plus faible (0,012
  contre 0,021), avec un seul capteur et un trajet de longueur 167.
- **Enregistrer plus souvent sur le même trajet aide autant que doubler le
  nombre de lignes.** Passer d'une mesure tous les 0,25 à une tous les 0,1
  donne 94 %, comme 16 lignes. Les mesures rapprochées ne voient rien de
  nouveau, mais leurs bruits se moyennent. Cela suppose que le bruit de deux
  mesures successives est indépendant, ce qui n'est vrai que si
  l'enregistrement n'est pas trop rapide.

**Ce que ML + BIC retrouve :**

50 sources, $\sigma = 10^{-3}$, graine 0 (même tirage qu'à la section 5),
nombre de sources inconnu (`results/ml_continu_ml.csv`) :

| capteur mobile, en plus du cercle | mesures en tout | sources estimées | $\chi^2/M$ | retrouvées à moins de 0,5 | à moins de 0,1 | sources à CRB < 0,1 | temps |
|---|---|---|---|---|---|---|---|
| aucun (cercle seul)¹ | 64 | 7 | | 4 | | 0 | 1,5 s |
| serpentin 4 lignes | 384 | 38 | 1,08 | 28 | 17 | 28 | 10 min |
| serpentin 8 lignes | 704 | 46 | 0,92 | **42** | **35** | 43 | 93 min² |
| grille fixe 16×32 (section 5) | 576 | 46 | 1,19 | **45** | **40** | 42 | 23 min |

¹ Médiane sur 5 tirages (section 9 de `COMPTE_RENDU_ML.md`).
² Calcul ralenti par d'autres calculs lancés en même temps sur la machine.
Seul, il prendrait sans doute de l'ordre de 25 min.

- **Un seul capteur qui parcourt 8 lignes permet à ML + BIC de retrouver
  42 sources sur 50**, dont 35 à moins de 0,1 près, contre 4 avec le
  cercle seul. C'est presque aussi bien que la grille fixe de 512 capteurs.
- **Avec 4 lignes, on retrouve encore 28 sources**, dont 17 à 0,1 près.
  Les lignes sont espacées de 2,5 : les sources situées entre deux lignes
  sont vues de trop loin pour être séparées de leurs voisines.
- **La méthode reste au niveau de la CRB.** Le nombre de sources retrouvées
  à 0,1 près (17 et 35) suit le nombre de sources que la CRB déclare
  localisables (28 et 43), avec un petit écart dû aux sources faibles que
  le BIC préfère écarter.

## 8. Et le Frank-Wolfe normalisé ?

Le Frank-Wolfe glissant (BLASSO, `sliding_frank_wolfe`, noyau normalisé)
est la méthode de référence du projet. Avec les capteurs du cercle, il
échoue sur beaucoup de sources, comme ML + BIC. On l'a relancé avec les
nouveaux capteurs, sur les **mêmes tirages et le même bruit** que ML + BIC
($\sigma = 10^{-3}$, graine 0, un tirage par cas ; expérience `frank_wolfe`,
`results/ml_fw_beaucoup.csv`, environ 3 min), avec deux réglages :

- **réglage du projet** : $\lambda = 1{,}5\,\sigma\lVert y\rVert$ sur les
  mesures brutes ;
- **blanchi** : chaque capteur est divisé par l'écart type de son bruit
  $s_i$ (comme dans ML), puis $\lambda = 1{,}5\sqrt{M}$, la norme attendue
  du bruit blanchi.

Le nombre de sources n'est jamais donné : c'est $\lambda$ qui le fixe.

| cas | méthode | sources estimées | retrouvées à moins de 0,5 | à moins de 0,1 | temps |
|---|---|---|---|---|---|
| 20 sources, grille 8×16 (192 capteurs) | FW, réglage du projet | 12 | 7 | 1 | 1 s |
|  | FW blanchi | 19 | 13 | 9 | 4 s |
|  | ML + BIC | 18 | **17** | **14** | 42 s |
| 50 sources, cercle seul (64) | FW, les deux réglages | 6 | 3 | 0 | < 1 s |
| 50 sources, grille 16×32 (576) | FW, réglage du projet | 16 | 14 | 1 | 1 s |
|  | FW blanchi | 44 | 33 | 26 | 51 s |
|  | ML + BIC | 46 | **45** | **40** | 23 min |
| 50 sources, serpentin 8 lignes (704) | FW, réglage du projet | 22 | 19 | 9 | 7 s |
|  | FW blanchi | 45 | 32 | 19 | 2 min |
|  | ML + BIC | 46 | **42** | **35** | 93 min |

Les mesures permettent de localiser à moins de 0,1 près 13 sources sur 20
(grille 8×16), 42 sur 50 (grille 16×32) et 43 sur 50 (serpentin).

- **Le Frank-Wolfe marche avec beaucoup de sources, à condition de
  blanchir.** Sans blanchiment, les capteurs proches de la source centrale,
  très bruités, fixent un $\lambda$ trop grand : la pénalité rejette toutes
  les sources faibles et la méthode s'arrête à 12 à 22 sources. Blanchi, il
  en trouve 44 ou 45 sur 50.
- **Il reste moins précis que ML + BIC** : 26 sources sur 50 à 0,1 près
  contre 40. La pénalité $\lambda\sum I_k$ biaise les intensités et les
  positions, et sépare mal les sources proches. C'est la limite déjà
  décrite dans `COMPTE_RENDU_ML.md` : de meilleurs capteurs ne la font pas
  disparaître.
- **Son avantage est la vitesse** : 51 s au lieu de 23 min pour 50 sources.

## 9. Conclusion

| levier | effet pour 50 sources |
|---|---|
| imposer K = 50 | pire que le BIC : 6 sources retrouvées au lieu de 9, les sources en trop collent au bruit |
| agrandir le cercle | pire : signal plus lisse, 31 directions sur 150 |
| 16 fois plus de capteurs sur le cercle | faible : 71 directions sur 150 |
| capteurs dans le domaine, un tous les 0,6 environ | problème bien posé : 150 directions sur 150, CRB 0,02 ; ML + BIC retrouve 45 sources sur 50 |
| un capteur mobile qui enregistre en continu (serpentin, 8 lignes) | presque aussi bon que 512 capteurs fixes : CRB 0,012 ; ML + BIC retrouve 42 sources sur 50 |
| un capteur mobile placé de façon adaptative (20 sources, 128 mesures) | 17,4 sources sur 20 à 0,1 près, contre 11 à 11,6 au hasard ou en grille |
| Frank-Wolfe normalisé au lieu de ML + BIC (grille 16×32) | blanchi : 26 sources à 0,1 près contre 40, mais 30 fois plus rapide ; non blanchi : 1 |

1. **Les limites de la section 9 de `COMPTE_RENDU_ML.md` venaient de la
   disposition des capteurs**, pas de la méthode ni de la connaissance de K.
   Avec de bons capteurs, la même méthode ML + BIC, sans aucun réglage,
   retrouve presque toutes les sources que la CRB déclare visibles.
2. **Pour retrouver beaucoup de sources, il faut des capteurs proches des
   sources**, avec un espacement comparable à la distance entre sources.
   S'éloigner ou densifier le cercle ne suffit pas.
3. **Un seul capteur mobile peut remplacer des centaines de capteurs
   fixes**, soit en enregistrant en continu le long d'un trajet qui couvre
   $D$, soit en choisissant ses positions d'après les sources déjà
   estimées. La stratégie adaptative tire le meilleur parti d'un petit
   nombre de mesures.
4. **ML + BIC reste la méthode à utiliser.** Le Frank-Wolfe normalisé
   fonctionne avec beaucoup de sources s'il est blanchi, mais il est moins
   précis.
5. **Avant de choisir une disposition, on peut l'évaluer en quelques
   secondes** avec les directions mesurables et la CRB (expériences
   `geometrie` et `continu`), sans lancer de reconstruction.

Cette étude suppose qu'on peut placer des capteurs dans le plan des sources
(hauteur nulle). Si les capteurs doivent rester au-dessus du plan, l'option
`--hauteur` de `tester_ml.py` permet de mesurer la perte : plus les capteurs
sont hauts, plus le signal est lisse, comme pour un cercle agrandi.

## 10. Utilisation

```bash
# capteurs du projet (inchangé)
python sparse_sources/tester_ml.py --sources 5 --sigma 1e-4

# cercle agrandi, plus de capteurs sur le cercle
python sparse_sources/tester_ml.py --rayon 2
python sparse_sources/tester_ml.py --nb-capteurs 1024

# grille de capteurs dans D (n×m cases, un capteur au centre de chaque case)
python sparse_sources/tester_ml.py --sources 50 --sigma 1e-3 --interieur 16x32 --k-max 70

# capteurs placés à la main (une ligne « x,y » par capteur), avec ou sans le cercle
python sparse_sources/tester_ml.py --fichier-capteurs mes_capteurs.txt
python sparse_sources/tester_ml.py --sans-cercle --fichier-capteurs mes_capteurs.txt

# capteurs au-dessus du plan des sources
python sparse_sources/tester_ml.py --interieur 8x16 --hauteur 0.5

# ML à K imposé (pas de BIC)
python sparse_sources/tester_ml.py --sources 50 --k-fixe 50

# tableaux des sections 3 et 4 (quelques secondes)
python sparse_sources/experiments_ml.py --seulement geometrie

# capteur mobile adaptatif, section 6 (~1 h 30)
python sparse_sources/experiments_ml.py --seulement mobile

# enregistrement continu, section 7 (CRB : quelques secondes ; ML + BIC : long)
python sparse_sources/experiments_ml.py --seulement continu

# Frank-Wolfe normalisé avec beaucoup de sources, section 8 (~3 min)
python sparse_sources/experiments_ml.py --seulement frank_wolfe
```

Capteur mobile en Python :

```python
from sparse_sources.mobile import adaptive_ml, serpentine

# measure(points) -> (mesures sans la source centrale, signal brut) aux points demandés
etats = adaptive_ml(measure, capteurs_fixes, m0, y0, domaine, sigma, n_moves=128,
                    batch=8, checkpoints=(32, 72))
etats[-1]["z"], etats[-1]["I"], etats[-1]["sensors"]       # sources estimées, positions visitées

trajet = serpentine(domaine, n_lines=8, step=0.25)          # (640, 2), dans l'ordre du parcours
```

Avec d'autres capteurs que ceux du projet, `tester_ml.py` garde les mêmes
sources (même graine) et recalcule les mesures pour ces capteurs. Un capteur
placé exactement sur la source centrale, à hauteur nulle, est refusé (signal
infini). L'option `--comparer` (Prony, Frank-Wolfe) ne fonctionne qu'avec les
capteurs du projet.

En Python :

```python
import numpy as np
from sparse_sources import greedy_ml, cramer_rao, relative_noise_std, build_grid
from sparse_sources.kernel import Setup
from sparse_sources.experiments_projet import SETUP

capteurs = np.vstack([SETUP.sensors, mes_capteurs])        # (M, 2)
S = Setup(capteurs, SETUP.domain, 0.0, normalize=True)
s = relative_noise_std(y_brut, sigma)                      # y_brut : mesures avec la source centrale
r = greedy_ml(m, s, setup=S, grid=build_grid(step=0.25, setup=S), k_max=70)
cramer_rao(r["z"], r["I"], s, S)["pos_std"]                # précision de chaque source
```

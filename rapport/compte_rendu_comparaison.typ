// Compte rendu détaillé : comparaison des méthodes (Prony, ML + BIC, Frank–Wolfe glissant,
// descente de gradient d'Orano) et analyse du cas à deux sources.
// Compiler depuis le dossier rapport/ :  typst compile compte_rendu_comparaison.typ
// Les figures sont lues dans images/ (PNG). Pour refaire les études : main_test.py --preset …
// Typst ≥ 0.14 sait inclure des PDF ; avec une version plus ancienne, mettre ext = "png".

#let ext = "pdf"
#let fig(chemin, largeur: 100%) = image("images/" + chemin.split("/").last() + ".png", width: largeur)

#set document(title: "Comparaison des méthodes et cas à deux sources", author: "Cyril Flury-Hérard")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.3cm), numbering: "1 / 1")
#set text(lang: "fr", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1", supplement: [section])
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => { v(0.6em); it; v(0.25em) }
#show table: set text(size: 9pt)
#show figure.caption: set text(size: 9pt)

#let encadre(titre, corps) = block(
  fill: rgb("#eef3fb"), stroke: (left: 3pt + rgb("#2a5caa")),
  inset: (x: 10pt, y: 8pt), radius: 2pt, width: 100%,
)[*#titre* #h(0.3em) #corps]

#let attention(corps) = block(
  fill: rgb("#fdf3e7"), stroke: (left: 3pt + rgb("#c77a12")),
  inset: (x: 10pt, y: 8pt), radius: 2pt, width: 100%, corps,
)

#align(center)[
  #text(size: 17pt, weight: "bold")[Comparaison des méthodes de localisation]
  #v(0.2em)
  #text(size: 12.5pt)[Prony, ML + BIC, Frank–Wolfe glissant et descente de gradient d'Orano, \ et pourquoi la descente échoue déjà à deux sources]
  #v(0.5em)
  Cyril Flury-Hérard — EA recherche — novembre 2026 \
  #text(size: 9pt)[Branche `travail-sur-le-probleme-inverse`. Code dans `localisation/`, points d'entrée `main.py` et `main_test.py`. Tous les chiffres viennent de `resultats/` ; aucune méthode n'a été modifiée.]
]

#v(0.6em)

#encadre[Résumé.][
+ *Sur le cercle de 64 capteurs, aucune méthode ne dépasse deux sources* dans le cadre du projet ($I_c = 50 S$, $sigma = 10^(-3)$). À $N = 2$, ML + BIC est la meilleure méthode (erreur médiane $E = 0","093$, bon nombre de sources dans 92 % des tirages) ; dès $N = 4$, toutes les méthodes ont une erreur de l'ordre de la taille d'une source manquée.
+ *Avec la grille de 512 capteurs dans le hangar, le problème change de nature* : ML + BIC trouve le bon nombre de sources dans 81 à 99 % des tirages jusqu'à $N = 10$, avec une erreur médiane de $6 dot 10^(-4)$ à $7 dot 10^(-3)$, soit 150 à 800 fois moins que sur le cercle. Le Frank–Wolfe suit, avec une erreur 2 à 10 fois plus grande et un nombre de sources souvent surestimé.
+ *La descente de gradient d'Orano échoue déjà à deux sources*, même sans bruit et avec le bon $N$ : 71 % de réussite à $x = 1$ et 2 % à $x = 300$, contre 100 % pour Gauss–Newton parti du même point. Trois causes : un conditionnement $kappa$ qui croît comme $x^(1","8)$ (le budget de 20 000 itérations ne suffit plus), une projection qui « éteint » une source (son intensité tombe à 0 et le gradient en position s'annule), et, avec la grille, un critère non pondéré plein de minima locaux.
+ *Prony est le plus rapide* (1 ms) mais, avec le seuil de rang du projet, il surestime le nombre de sources dès que la source centrale est présente (2,6 fausses sources en moyenne à $N = 2$).
]

= Matériel, logiciels et durées

== Machine

Tous les calculs ont été faits sur mon ordinateur portable :

#table(
  columns: (auto, 1fr),
  stroke: none, align: (left, left),
  table.hline(),
  [*Machine*], [ordinateur portable HUAWEI, modèle NBLB-WAX9N],
  [*Processeur (puce)*], [Intel Core i5-10210U (génération Comet Lake, 2019) : 4 cœurs physiques, 8 fils d'exécution, 1,6 GHz de base, jusqu'à 4,2 GHz en turbo, enveloppe thermique de 15 W],
  [*Carte graphique*], [Intel UHD Graphics (puce graphique intégrée au processeur, 1 Go de mémoire partagée, pilote 27.20.100.8984)],
  [*Mémoire vive*], [7,8 Go],
  [*Système*], [Windows 11 Famille],
  [*Logiciels*], [Python 3.14.0, NumPy 2.3.3, SciPy 1.16.2, scikit-learn 1.7.2, PyTorch 2.9.0 (version CPU), Matplotlib],
  table.hline(),
)

*La carte graphique n'a pas servi.* Une puce Intel UHD intégrée n'a pas de support CUDA, et PyTorch est installé en version CPU. Surtout, les méthodes comparées sont écrites en NumPy/SciPy (optimisation L-BFGS-B, moindres carrés, décompositions de petites matrices) : ce sont des calculs séquentiels sur des matrices de quelques centaines de lignes, que la carte graphique n'accélérerait pas. *Tout le calcul est sur le processeur*, avec un parallélisme par tirage : chaque processus traite un tirage complet, et les bibliothèques d'algèbre linéaire sont limitées à un fil par processus pour éviter la sur-souscription.

== Durées réelles

#block(breakable: false, table(
  columns: (1fr, auto, auto),
  stroke: none, align: (left, right, left),
  table.hline(),
  [*Calcul*], [*Durée*], [*Processus*],
  table.hline(stroke: 0.5pt),
  [Essai rapide (`--quick`, 10 tirages, 3 points par balayage, avec la descente)], [13,9 min], [6],
  [Comparaison complète, cercle (14 400 tâches)], [15,0 min], [6],
  [Comparaison complète, cercle + grille (14 400 tâches)], [78,4 min], [6],
  [Comparaison complète, total avec figures et Typst], [94,2 min], [6],
  [Cas à deux sources : réussite, causes, conditionnement (1 800 tirages)], [80,4 min], [2],
  [Cas à deux sources : bassins d'attraction (864 départs)], [35,7 min], [2],
  [Cas à deux sources : total avec figures], [116,3 min], [2],
  table.hline(),
))

Les deux calculs ont tourné en même temps : 8 processus sur 4 cœurs physiques. *Les temps de calcul mesurés par tirage sont donc des temps « sous charge »*, plus longs (d'environ un facteur 1,5 à 2) que sur une machine au repos ; les comparaisons entre méthodes restent valables, car toutes ont subi la même charge.

Un premier lancement complet incluait la descente de gradient ; il avançait au rythme d'environ 6 h pour l'ensemble, la descente prenant à elle seule plus de 90 % du temps (jusqu'à 12 s par tirage à $N = 10$, contre 0,6 s pour ML + BIC). Je l'ai arrêté et la descente a été retirée des balayages généraux : elle est étudiée en détail dans le cas à deux sources (@sec:deux), qui est le cas où Orano l'a testée.

= Le problème et les méthodes

== Le problème

On cherche $N$ sources ponctuelles de plutonium, de positions $z_i$ et d'intensités $I_i > 0$, dans le hangar $D = [0, 10] times [0, 20]$. Un capteur placé en $s_k$ mesure
$ Phi_theta (s_k) = sum_(i=1)^N I_i / norm(s_k - z_i)^2 , $
plus le signal d'une source centrale connue, d'intensité $I_c$, placée en $c = (5, 10)$. Le bruit est relatif : le capteur lit $y_k^"brut" (1 + sigma epsilon_k)$ avec $epsilon_k tilde cal(N)(0, 1)$. On retire le signal de la source centrale, mais *son bruit reste* : l'écart type au capteur $k$ vaut $sigma_k = sigma y_k^"brut"$, dominé par la source centrale dès que $I_c$ est grand devant la somme $S = sum I_i = 100$.

Deux dispositions de capteurs sont comparées :
- *cercle* : 64 capteurs régulièrement répartis sur le cercle de centre $c$ passant par les coins de $D$ (rayon $sqrt(125) approx 11","2$) ;
- *cercle + grille* : les mêmes 64 capteurs plus une grille régulière de $16 times 32 = 512$ capteurs dans $D$, soit 576 capteurs.

== Les méthodes, telles qu'elles sont dans le dépôt

Les méthodes sont utilisées *sans modification*, avec les réglages déjà présents dans le dépôt.

- *Prony* (`src/particle_sources/inversion/prony.py`, `prony_circle`). Les mesures du cercle sont vues comme une fonction sur le cercle ; ses coefficients de Fourier forment une matrice de Hankel dont le rang est le nombre de sources, et les sources sont les racines d'un polynôme. Le rang est lu avec le seuil relatif `rank_tol = 0.005` de `config/reconstruction.yaml`. *Prony n'utilise que les capteurs du cercle*, y compris dans la configuration cercle + grille.
- *ML + BIC* (`localisation/ml_bic.py`, `greedy_ml`). Maximum de vraisemblance : moindres carrés pondérés par $1 \/ sigma_k^2$, résolus par L-BFGS-B (et non par Gauss–Newton), en ajoutant les sources une par une à partir d'une recherche exhaustive sur une grille de positions candidates ; le nombre de sources est choisi par le critère BIC ($k_max = 20$).
- *Frank–Wolfe glissant* (`localisation/blasso.py`, `sliding_frank_wolfe`). Problème BLASSO sur les mesures, capteurs blanchis (divisés par $sigma_k$) et noyau normalisé, pénalité $lambda = 1","5 sqrt(M)$ ($M$ = nombre de capteurs), 40 itérations au plus.
- *Descente de gradient d'Orano* (`localisation/descente.py`, `descente_projetee`). Descente de gradient sur $theta = (z_i, I_i)$ normalisés, pas d'Armijo (le pas est doublé à chaque itération puis divisé par 2 jusqu'à décroissance suffisante), déplacement limité à 5 % du domaine, projection des positions sur $D$ et des intensités sur $RR_+$, 20 000 itérations au plus. Le critère est la somme des carrés des résidus *non pondérée*, divisée par $norm(y)^2$. Elle reçoit le vrai $N$ ; les positions de départ sont tirées uniformément dans $D$ et les intensités de départ valent $S \/ N$.
- *Gauss–Newton* (`orano.gauss_newton`, région de confiance de `scipy.optimize.least_squares`) : même critère que la descente, même point de départ, même $N$. Il sert de *témoin* dans le cas à deux sources : il permet de séparer ce qui vient du critère et ce qui vient de l'algorithme.

= Protocole de comparaison

*Situations.* Pour chaque $N in {2, 4, 6, 10}$ et chaque tirage, les positions $z_i$ sont uniformes dans $D$ et les intensités suivent une loi de Dirichlet de somme $S = 100$. Le générateur aléatoire est initialisé par $(N, "tirage")$ seulement : *toutes les méthodes et les deux dispositions de capteurs voient exactement les mêmes sources et le même bruit*. Les différences entre méthodes ne viennent donc pas du hasard des tirages.

*Erreur d'un tirage.* Les sources estimées sont appariées aux vraies par l'algorithme hongrois (appariement qui minimise la somme des distances). Pour chaque vraie source $i$, $d_i = norm(hat(z)_i - z_i)$ et $r_i = abs(hat(I)_i - I_i) \/ I_i$, plafonnés à $d_i <= 5$ et $r_i <= 1$ ; une source manquée compte $(5, 1)$. L'erreur est
$ E = 1/N sum_(i=1)^N (alpha d_i + beta r_i), quad alpha = 0","75, quad beta = 0","25 . $
*Ordre de grandeur :* $E approx 4$ veut dire « tout est faux » ; $E approx 1$ correspond par exemple à une source sur quatre manquée ; $E < 0","01$ veut dire que les sources sont trouvées à quelques centimètres près (le hangar fait 10 × 20). Les fausses sources (sources estimées en trop) ne sont pas dans $E$ : elles sont comptées à part, et la proportion de tirages avec $hat(N) = N$ est tracée sous chaque courbe d'erreur.

*Balayages.* Chaque balayage fait varier un paramètre autour du cadre par défaut $I_c = 50 S$, $sigma = 10^(-3)$ :
(a) bruit $sigma in {10^(-4), dots, 10^(-2)}$, 7 valeurs régulièrement espacées en échelle log ;
(b) source centrale $I_c \/ S in {0, 10, 25, 50, 100, 200}$ ;
(c) nombre de sources $N$ ;
(d) temps de calcul.
Chaque point compte 100 tirages, soit 48 points × 100 tirages × 3 méthodes = 14 400 calculs par disposition.

*Timeout.* Un calcul qui dépasse 60 s est arrêté (le processus est tué et relancé) et compté comme un échec complet ($E$ d'une solution vide).

= Résultats avec les capteurs sur le cercle

#figure(fig("comparaison/bruit_cercle"),
  caption: [Erreur $E$ (médiane, bande entre premier et troisième quartile, échelle log) et proportion de tirages avec $hat(N) = N$, en fonction du bruit relatif $sigma$ ; $I_c = 50 S$, 64 capteurs sur le cercle, 100 tirages par point.]) <fig:cr-bruit-cercle>

*À deux sources, le bruit décide de tout.* À $N = 2$ (@fig:cr-bruit-cercle), l'erreur médiane de ML + BIC passe de 0,009 à $sigma = 10^(-4)$ à 2,05 à $sigma = 10^(-2)$, et celle des autres méthodes suit la même pente : l'erreur est proportionnelle à $sigma$ tant que les deux sources sont détectées, puis sature quand l'une d'elles disparaît dans le bruit. ML + BIC est meilleur que le Frank–Wolfe et que Prony d'un facteur 2 à 5 sur toute la gamme : à $sigma = 10^(-3)$, $E = 0","093$ contre 0,30 (Frank–Wolfe) et 0,45 (Prony).

*Au-delà de deux sources, le cercle ne suffit plus.* À $N = 4$, même avec le bruit le plus faible ($sigma = 10^(-4)$), ML + BIC ne trouve le bon nombre de sources que dans 66 % des tirages, et seulement 23 % à $sigma = 10^(-3)$. À $N = 6$ et $N = 10$, aucune méthode ne descend sous $E approx 1$ : on retrouve le résultat du compte rendu précédent, le cercle ne mesure qu'une dizaine de paramètres à la précision utile.

#figure(fig("comparaison/centrale_cercle"),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction de l'intensité de la source centrale ($I_c \/ S = 0$ : pas de source centrale) ; $sigma = 10^(-3)$, 64 capteurs sur le cercle.]) <fig:cr-centrale-cercle>

*La source centrale est le facteur limitant.* Sans source centrale (@fig:cr-centrale-cercle, $I_c = 0$), ML + BIC atteint $E = 0","003$ à $N = 2$ et retrouve le bon nombre de sources dans 83 % des tirages à $N = 4$. Avec $I_c = 200 S$, l'erreur à $N = 2$ monte à 0,38 et la détection à $N = 4$ tombe à 7 %. C'est logique : le bruit au capteur $k$ vaut $sigma y_k^"brut"$, et $y_k^"brut"$ est dominé par la source centrale ; multiplier $I_c$ par 10 revient à peu près à multiplier $sigma$ par 10.

*Prony a un comportement à part.* Sa proportion de $hat(N) = N$ *augmente* avec $I_c$ à $N = 6$ (de 0 à 23 %) : ce n'est pas une meilleure détection, c'est que Prony renvoie trop de sources (voir @sec:anomalies), et qu'en présence de bruit fort ce nombre trop grand tombe parfois sur 6.

#figure(fig("comparaison/nsources_cercle"),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction du nombre de sources, cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$) ; 64 capteurs sur le cercle.]) <fig:cr-n-cercle>

@fig:cr-n-cercle résume le cadre par défaut : à $N = 2$, ML + BIC 0,093 (92 % de $hat(N) = N$), Frank–Wolfe 0,30 (82 %), Prony 0,45 (16 %) ; à $N = 4$, les trois méthodes sont entre 1,2 et 1,8 ; à $N = 10$, entre 2,6 et 2,8, c'est-à-dire que plus de la moitié des sources sont manquées.

= Résultats avec le cercle et la grille

#figure(fig("comparaison/bruit_grille"),
  caption: [Même balayage en bruit que @fig:cr-bruit-cercle, avec en plus 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).]) <fig:cr-bruit-grille>

*La grille fait gagner deux à trois ordres de grandeur.* Avec la grille (@fig:cr-bruit-grille), l'erreur de ML + BIC reste proportionnelle à $sigma$ pour *tous* les $N$ : à $sigma = 10^(-3)$, $E = 6 dot 10^(-4)$ à $N = 2$ et $6","7 dot 10^(-3)$ à $N = 10$. Le bon nombre de sources est trouvé dans 93 % des tirages à $N = 4$ et 81 % à $N = 10$. *Les capteurs intérieurs voient chaque source de près* : la grille a un pas de 0,625, donc chaque source a un capteur à moins de 0,45 ; elle y produit un signal plusieurs centaines de fois plus fort que sur les capteurs du cercle, situés à plusieurs mètres, alors que le bruit de la source centrale, lui, ne change pas d'ordre de grandeur.

*Le Frank–Wolfe profite aussi de la grille*, mais moins : $E = 10^(-3)$ à $N = 2$ et 0,068 à $N = 10$, avec une détection exacte du nombre de sources dans seulement 21 à 57 % des tirages à $N >= 4$. Le Frank–Wolfe ajoute de petites sources parasites près des vraies (la pénalité $lambda$ ne suffit pas à les supprimer quand il y a 576 capteurs) ; elles ne dégradent presque pas $E$ mais faussent $hat(N)$. À $sigma = 10^(-2)$ et $N >= 4$, son erreur remonte brusquement (0,74 à $N = 4$, 1,1 à $N = 10$) : il manque alors des sources.

*Prony ne change pas* : il n'utilise que les capteurs du cercle, ses courbes sont identiques dans les deux figures.

#figure(fig("comparaison/centrale_grille"),
  caption: [Balayage en source centrale avec cercle + grille ; $sigma = 10^(-3)$.]) <fig:cr-centrale-grille>

Avec la grille, la source centrale reste le facteur limitant (@fig:cr-centrale-grille), mais à un niveau bien plus bas : pour ML + BIC à $N = 6$, $E$ passe de $2","3 dot 10^(-4)$ sans source centrale à 0,012 avec $I_c = 200 S$, et le bon nombre de sources est trouvé dans au moins 90 % des tirages.

#figure(fig("comparaison/synthese_nsources_temps"),
  caption: [Cadre par défaut : erreur $E$ médiane, proportion de $hat(N) = N$ et temps médian en fonction de $N$ ; cercle seul (traits pleins) et cercle + grille (pointillés). Prony n'utilise que le cercle dans les deux cas.]) <fig:cr-synthese>

#figure(fig("comparaison/synthese_bruit"),
  caption: [Balayage en bruit : cercle seul (traits pleins) et cercle + grille (pointillés), mêmes tirages.]) <fig:cr-synthese-bruit>

@fig:cr-synthese et @fig:cr-synthese-bruit superposent les deux dispositions. À $N = 10$ et $sigma = 10^(-3)$, l'erreur médiane de ML + BIC passe de 2,59 (cercle) à 0,0067 (cercle + grille), celle du Frank–Wolfe de 2,76 à 0,068.

= Temps de calcul

#figure(
  grid(columns: 2, gutter: 6pt, fig("comparaison/temps_cercle"), fig("comparaison/temps_grille")),
  caption: [Temps de calcul par tirage (médiane et quartiles), génération des données exclue ; cadre par défaut. À gauche : cercle ; à droite : cercle + grille. Mesuré avec 8 processus sur 4 cœurs.]) <fig:cr-temps>

#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: none, align: (left, right, right, right, right, right),
  table.hline(),
  [*Méthode*], [*Disposition*], [$N = 2$], [$N = 4$], [$N = 6$], [$N = 10$],
  table.hline(stroke: 0.5pt),
  [Prony], [les deux], [1,3 ms], [1,3 ms], [1,3 ms], [1,4 ms],
  [ML + BIC], [cercle], [0,21 s], [0,36 s], [0,46 s], [0,61 s],
  [ML + BIC], [cercle + grille], [0,55 s], [1,1 s], [1,8 s], [6,9 s],
  [Frank–Wolfe], [cercle], [0,067 s], [0,10 s], [0,14 s], [0,16 s],
  [Frank–Wolfe], [cercle + grille], [0,35 s], [0,61 s], [1,1 s], [3,6 s],
  [Descente (Orano)], [cercle, $N = 2$], [7,3 s], [—], [—], [—],
  table.hline(),
)

*Prony est mille fois plus rapide que les autres* (une décomposition de matrice de Hankel de taille fixe : son temps ne dépend pas de $N$). ML + BIC et le Frank–Wolfe ont un coût proportionnel au nombre de capteurs et augmentent avec $N$ ; avec 576 capteurs et $N = 10$, ML + BIC prend 7 s en médiane. *La descente d'Orano est la plus lente* : 7,3 s pour deux sources seulement, parce qu'elle fait presque toujours ses 20 000 itérations (@sec:deux).

*Timeouts (60 s).* Sur 28 800 calculs :
- cercle : 3 timeouts, tous pour ML + BIC (2 à $N = 6$ et 1 à $N = 4$, sans source centrale) ; 0 pour Prony et le Frank–Wolfe ;
- cercle + grille : 88 timeouts pour ML + BIC (dont 78 à $N = 10$, surtout à bruit faible : 16 à $sigma = 10^(-4)$), 5 pour le Frank–Wolfe (4 à $N = 10$), 0 pour Prony.
Le BIC ajoute des sources tant que la vraisemblance s'améliore assez ; à bruit faible et avec 576 capteurs, chaque ajout est accepté longtemps et chaque optimisation coûte cher. Ces 93 timeouts sur 14 400 calculs (0,6 %) sont comptés comme des échecs complets ; ils pèsent peu sur les médianes. Une partie vient sans doute de la charge de la machine : les temps maximaux observés hors timeout (54 s à $N = 10$ sur le cercle) montrent que certains calculs sont proches de la limite.

= Le cas à deux sources : pourquoi la descente échoue <sec:deux>

== Dispositif

Orano n'a testé la descente qu'à *deux sources*, en faisant varier le rapport d'intensité $x = I_1 \/ I_2$. J'ai refait cette expérience avec 100 tirages par valeur de $x in {1, 3, 10, 30, 100, 300}$ : les positions sont celles des tirages à $N = 2$ de la comparaison, et les intensités valent $I_1 = S x \/ (x + 1)$, $I_2 = S \/ (x + 1)$. Trois cadres :
- *cas test d'Orano* : cercle, sans bruit, sans source centrale ($sigma = 0$, $I_c = 0$) ;
- *projet, cercle* : $sigma = 10^(-3)$, $I_c = 50 S$ ;
- *projet, cercle + grille* : idem avec les 576 capteurs.

Un tirage est une *réussite* si la méthode renvoie exactement 2 sources et que chacune est à moins de 0,1 de la vraie (0,5 % de la largeur du hangar). La descente et Gauss–Newton reçoivent $N = 2$ et partent *du même point*.

Pour chaque échec de la descente, j'ai cherché la cause, dans cet ordre :
+ *source éteinte* : l'intensité d'une source est tombée à 0 (projection sur $RR_+$) ;
+ *budget épuisé* : les 20 000 itérations sont faites sans atteindre la précision ;
+ *limite statistique* : la perte finale est au plus celle du minimum voisin de la vérité (obtenu par Gauss–Newton parti des vraies sources) : la descente a trouvé le bon minimum, mais avec le bruit ce minimum est à plus de 0,1 des vraies sources, et aucune méthode ne peut faire mieux ;
+ *bloquée au bord* : une source est collée au bord de $D$ ;
+ *minimum local* : tout le reste.

== Taux de réussite

#figure(fig("deux_sources/ds_reussite_rapport"),
  caption: [Réussite à deux sources ($hat(N) = 2$ et chaque source à moins de 0,1) en fonction du rapport d'intensité $x$, 100 tirages par point. Gauss–Newton et la descente reçoivent $N = 2$ et partent du même point. Croix : valeurs de la synthèse Orano.]) <fig:cr-ds-reussite>

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  stroke: none, align: (left, left) + (right,) * 6,
  table.hline(),
  [*Cadre*], [*Méthode*], [$x = 1$], [$3$], [$10$], [$30$], [$100$], [$300$],
  table.hline(stroke: 0.5pt),
  [Orano (synthèse)], [descente], [95 %], [85 %], [55 %], [20 %], [5 %], [—],
  table.hline(stroke: 0.3pt),
  table.cell(rowspan: 5)[cas test d'Orano], [descente], [71 %], [68 %], [52 %], [30 %], [13 %], [2 %],
  [Gauss–Newton], [100 %], [100 %], [100 %], [100 %], [100 %], [99 %],
  [ML + BIC], [100 %], [100 %], [100 %], [100 %], [100 %], [100 %],
  [Frank–Wolfe], [100 %], [100 %], [100 %], [100 %], [100 %], [100 %],
  [Prony], [97 %], [95 %], [89 %], [69 %], [39 %], [22 %],
  table.hline(stroke: 0.3pt),
  table.cell(rowspan: 5)[projet, cercle], [descente], [30 %], [30 %], [22 %], [13 %], [3 %], [1 %],
  [Gauss–Newton], [43 %], [40 %], [27 %], [16 %], [7 %], [2 %],
  [ML + BIC], [43 %], [38 %], [25 %], [15 %], [7 %], [2 %],
  [Frank–Wolfe], [13 %], [13 %], [10 %], [9 %], [6 %], [3 %],
  [Prony], [11 %], [10 %], [4 %], [4 %], [1 %], [0 %],
  table.hline(stroke: 0.3pt),
  table.cell(rowspan: 5)[projet, cercle + grille], [descente], [0 %], [0 %], [0 %], [0 %], [0 %], [0 %],
  [Gauss–Newton], [0 %], [0 %], [0 %], [0 %], [0 %], [0 %],
  [ML + BIC], [100 %], [100 %], [99 %], [98 %], [95 %], [85 %],
  [Frank–Wolfe], [91 %], [88 %], [83 %], [79 %], [66 %], [36 %],
  [Prony], [11 %], [10 %], [4 %], [4 %], [1 %], [0 %],
  table.hline(),
)

*Dans le cas test d'Orano, on retrouve la même tendance qu'Orano* : la réussite de la descente baisse fortement avec $x$ (71 % → 2 %), alors que Gauss–Newton, ML + BIC et le Frank–Wolfe réussissent presque toujours. Mes chiffres sont plus bas qu'Orano à $x$ petit (71 % contre 95 % à $x = 1$) et un peu plus hauts à $x$ grand (30 % contre 20 % à $x = 30$). Les tirages ne sont pas les mêmes (Orano ne précise ni ses positions ni son initialisation) ; ici les positions de départ sont uniformes dans tout le hangar, ce qui est exigeant. *L'ordre de grandeur et la pente sont les mêmes.*

*Le point essentiel est Gauss–Newton* : avec le même critère, le même point de départ et le même $N$, il réussit 100 % des tirages sans bruit jusqu'à $x = 100$. *Le problème de la descente n'est donc ni le critère ni l'initialisation, c'est l'algorithme* : un pas de gradient ne tient pas compte de la courbure, et la courbure est très inégale selon les directions (section suivante).

*Dans le cadre du projet sur le cercle*, toutes les méthodes s'effondrent, mais pour une autre raison : le bruit. La part des tirages où Gauss–Newton *parti des vraies sources* finit à moins de 0,1 vaut 43/40/27/16/7/2 % : c'est exactement la réussite de Gauss–Newton, et à un ou deux points près celle de ML + BIC (43/38/25/15/7/2 %). *Ces deux méthodes sont à la limite statistique* : aucune méthode ne peut faire mieux avec ces données. La descente (30/30/22/13/3/1 %) reste en dessous.

*Avec la grille, la descente et Gauss–Newton échouent à 100 %, alors que ML + BIC réussit 85 à 100 %.* Pourtant Gauss–Newton parti des vraies sources finit à moins de 0,1 dans 87 à 100 % des tirages : le bon minimum existe et il est précis. C'est donc *l'initialisation* qui échoue : avec le critère non pondéré, la perte est dominée par les quelques capteurs de la grille les plus proches des sources ; elle est hérissée de pics sur les capteurs avec des plateaux entre eux, et la descente s'y arrête loin des vraies sources (@fig:cr-ds-paysage). ML + BIC évite ce piège parce qu'il place chaque source par une recherche exhaustive sur une grille avant d'optimiser, et parce qu'il pondère les capteurs par leur bruit.

== Causes des échecs

#figure(fig("deux_sources/ds_causes"),
  caption: [Issue de chaque descente, par cadre et par rapport d'intensité $x$ (100 tirages par barre).]) <fig:cr-ds-causes>

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto),
  stroke: none, align: (left,) + (right,) * 6,
  table.hline(),
  [*Cadre (600 descentes)*], [réussite], [budget épuisé], [source éteinte], [limite stat.], [bord], [min. local],
  table.hline(stroke: 0.5pt),
  [cas test d'Orano], [236], [329], [35], [0], [0], [0],
  [projet, cercle], [99], [392], [30], [79], [0], [0],
  [projet, cercle + grille], [0], [36], [1], [0], [177], [386],
  table.hline(),
)

@fig:cr-ds-causes montre trois mécanismes différents :

- *Budget épuisé* (sur le cercle, 55 % des descentes du cas test d'Orano). La descente avance, mais trop lentement : à $x = 1$, 26 descentes sur 100 n'ont pas fini en 20 000 itérations ; à $x = 300$, 89 sur 100. Pour ces descentes, l'erreur de position médiane est encore de 3,2 à la fin. Même les descentes réussies prennent 10 900 itérations en médiane.
- *Source éteinte* (6 % des descentes sur le cercle, de 3 % à $x = 1$ à 9 % à $x = 300$). Quand une source est loin de sa cible, la descente fait baisser son intensité ; la projection sur $RR_+$ la met à 0. Or le gradient par rapport à la position d'une source est proportionnel à son intensité ($partial Phi \/ partial z_i = I_i nabla_z G$) : *une source d'intensité nulle ne bouge plus jamais*. C'est un piège propre à la paramétrisation $(z_i, I_i)$ avec projection.
- *Bloquée au bord et minimum local* (grille, 94 % des descentes). Avec la grille, la descente s'arrête vite (774 itérations en médiane) parce que le pas d'Armijo devient négligeable : elle est dans un minimum local dont la perte est en médiane 1 400 fois celle du bon minimum. Dans 30 % des cas, une source est même collée au bord de $D$.

== Conditionnement

#figure(fig("deux_sources/ds_conditionnement"),
  caption: [À gauche : conditionnement $kappa$ de $J^top J$ (critère d'Orano, paramètres normalisés) aux vraies sources, en fonction de $x$ ; droite de pente 2 en référence. À droite : itérations de la descente en fonction de $kappa$, cas test d'Orano.]) <fig:cr-ds-kappa>

Le nombre d'itérations d'une descente de gradient sur une cuvette quadratique est proportionnel au conditionnement $kappa = lambda_max \/ lambda_min$ de la hessienne (ici $J^top J$, $J$ jacobienne du résidu). @fig:cr-ds-kappa montre la médiane de $kappa$ aux vraies sources : $7 dot 10^3$ à $x = 1$, $8 dot 10^3$ à $x = 10$, $4","7 dot 10^4$ à $x = 30$, $4 dot 10^5$ à $x = 100$ et $3","6 dot 10^6$ à $x = 300$. *Pour $x >= 10$, la pente en échelle log-log vaut 1,8*, proche du $kappa >= c x^2$ annoncé par Orano. L'explication est simple : la position de la source faible n'agit sur les mesures qu'à travers $I_2 nabla G$, donc la colonne correspondante de $J$ est $x$ fois plus petite que celle de la source forte, et la valeur propre associée de $J^top J$ est $x^2$ fois plus petite.

Le panneau de droite de @fig:cr-ds-kappa montre le lien directement : *le nombre d'itérations d'une descente réussie croît avec $kappa$*, de 800 itérations pour $kappa approx 10^2$ à 20 000 pour $kappa approx 3 dot 10^3$ ; 90 % des descentes réussies ont $kappa < 1","1 dot 10^4$, et au-delà presque toutes les descentes épuisent leur budget. Les sources éteintes, elles, apparaissent pour tous les $kappa$ : c'est un autre mécanisme.

Avec $kappa approx 4 dot 10^5$ à $x = 100$, une descente de gradient demande de l'ordre de $kappa$ itérations pour réduire l'erreur d'un facteur $e$ dans la direction la plus plate : *20 000 itérations ne peuvent pas suffire*. Gauss–Newton, qui résout le système $J^top J delta = -J^top r$ à chaque pas, n'est pas sensible à $kappa$ de la même manière : il converge en quelques dizaines d'itérations (11 à 34 ms).

Avec la grille, $kappa$ est encore plus grand ($1","4 dot 10^5$ à $x = 1$, $10^7$ à $x = 300$) : les capteurs très proches d'une source rendent le critère extrêmement raide dans certaines directions.

== Paysage de la perte

#figure(fig("deux_sources/ds_paysage"),
  caption: [Perte d'Orano en fonction de la position de la source faible (l'autre source et les intensités à leur vraie valeur) et trajectoire de la descente, sur le cercle et sur cercle + grille ; cas test d'Orano, $x = 30$, tirage 1. À droite, distance de chaque source estimée à la vraie source la plus proche au fil des itérations.]) <fig:cr-ds-paysage>

@fig:cr-ds-paysage rend visibles les deux mécanismes. *Sur le cercle*, la perte vue depuis la source faible est une vallée large et très plate : la descente y entre vite, puis avance à petits pas, et la distance à la vraie source ne décroît que lentement au fil des itérations. Dans l'exemple tracé ($x = 30$, sources à 1,2 l'une de l'autre), la source forte arrive à 0,15 de sa cible, mais la source faible reste à 1 de la sienne après 20 000 itérations. *Avec la grille*, la carte est hérissée d'un pic sur chacun des 512 capteurs (la perte explose quand la source déplacée passe sur un capteur, à cause de la singularité en $1\/r^2$), avec un plateau entre les pics. Le puits de la source faible est minuscule et invisible à cette échelle. La descente s'arrête au bout de 572 itérations, ses deux sources toujours à environ 9 de leurs cibles et l'une collée au bord droit : le pas d'Armijo est devenu négligeable sans que le gradient indique la direction des vraies sources.

== Bassins d'attraction

#figure(fig("deux_sources/ds_bassins"),
  caption: [Bassins d'attraction : issue de la descente selon la position de départ de la source 1 (la source 2 part toujours du même point), sur une grille de $12 times 24$ départs ; croix : départs depuis lesquels Gauss–Newton échoue.]) <fig:cr-ds-bassins>

Pour trois situations, j'ai relancé la descente depuis $12 times 24 = 288$ positions de départ de la source 1, la source 2 partant toujours du même point (@fig:cr-ds-bassins) :
- *cas test d'Orano, $x = 1$, tirage 1* (deux sources à 1,2 l'une de l'autre, $kappa = 8","5 dot 10^6$) : la descente réussit depuis 32 départs sur 288 (11 %), toujours dans une bande étroite ; elle épuise son budget depuis 240 départs et éteint une source depuis 16 départs, tous près du bas du hangar ;
- *même tirage, $x = 30$* : la descente ne réussit depuis *aucun* départ (273 budgets épuisés, 15 sources éteintes) ;
- *cadre du projet, $x = 1$, tirage 7* (sources éloignées de 14,8, $kappa = 2 dot 10^4$) : la descente réussit depuis 92 départs (32 %), tous dans la moitié basse du hangar, du côté de la source la plus proche de son départ fixe.

*Gauss–Newton réussit depuis les 288 départs dans les trois situations.* Il n'y a donc pas de mauvais minimum local dans ces cas : *depuis n'importe quel départ, il existe un chemin vers la bonne solution, et c'est la descente qui ne le parcourt pas assez vite*. Les zones « source éteinte » sont celles où une source part loin de toute vraie source et près du bord : la descente fait d'abord baisser son intensité pour réduire le résidu, la projection la met à 0, et elle ne bouge plus.

== Distance entre les sources

#figure(fig("deux_sources/ds_separation"),
  caption: [Réussite en fonction de la distance entre les deux sources, cas test d'Orano, $x = 1$ et $x = 30$.]) <fig:cr-ds-separation>

Deux sources proches sont plus difficiles à séparer (@fig:cr-ds-separation). Dans le cas test d'Orano à $x = 1$, la descente réussit dans 1 cas sur 7 quand les sources sont à moins de 2 l'une de l'autre, 43 % entre 2 et 4, et environ 80 % au-delà de 4. À $x = 30$, elle ne réussit jamais quand les sources sont à moins de 4 l'une de l'autre, et dans 27 à 48 % des cas au-delà. Une source faible proche d'une source forte est « cachée » dans son signal : le gradient par rapport à sa position est minuscule devant celui de la source forte.

== Temps

#figure(fig("deux_sources/ds_temps"),
  caption: [Temps de calcul médian par tirage en fonction de $x$.]) <fig:cr-ds-temps>

Dans le cas test d'Orano, les temps médians sont de 7,3 à 8,8 s pour la descente, 11 à 34 ms pour Gauss–Newton, 0,13 à 0,22 s pour ML + BIC, 73 à 106 ms pour le Frank–Wolfe et 1 ms pour Prony (@fig:cr-ds-temps). Gauss–Newton est donc 200 à 700 fois plus rapide que la descente, et réussit beaucoup plus souvent. Orano annonçait 59 ms pour Gauss–Newton et 10 à 25 ms pour « Prony + ML » : nos ordres de grandeur sont les mêmes.

== Ce qu'il faut retenir du cas à deux sources

#encadre[Bilan.][
La descente de gradient d'Orano échoue pour deux raisons qui s'additionnent :
+ *une raison d'algorithme* : le pas de gradient ignore la courbure, alors que $kappa$ croît comme $x^(1","8)$ ; le budget d'itérations est épuisé avant la convergence. Gauss–Newton, avec le même critère et le même départ, règle ce problème (100 % sans bruit jusqu'à $x = 100$) ;
+ *une raison de paramétrisation* : la projection des intensités sur $RR_+$ peut éteindre une source, qui ne bouge alors plus jamais.

Avec la grille, s'y ajoute *une raison de critère* : la perte non pondérée a des minima locaux partout, que ni la descente ni Gauss–Newton ne peuvent éviter à partir d'un départ aléatoire. Il faut une bonne initialisation, ce que fait ML + BIC avec sa recherche sur grille.

Dans le cadre bruité du projet sur le cercle, la limite n'est plus l'algorithme mais l'information : Gauss–Newton et ML + BIC sont à la limite statistique, et seule une meilleure géométrie des capteurs permet de faire mieux.
]

= Anomalies et limites <sec:anomalies>

- *Prony surestime le nombre de sources.* Avec le seuil de rang du projet (`rank_tol = 0.005`), Prony renvoie en moyenne 2,6 fausses sources à $N = 2$ et 1,4 à $N = 4$ dans le cadre par défaut, et 4,3 à 4,8 sources au lieu de 2 dans le cadre bruité à deux sources. Sans bruit (cas test d'Orano), c'est l'inverse : à $x$ grand, il ne voit souvent qu'une source (1,24 source en moyenne à $x = 300$), parce que la source faible passe sous le seuil. Le seuil fixe n'est adapté ni au bruit du projet ni aux grands rapports d'intensité ; je ne l'ai pas modifié, conformément à la consigne.
- *La descente a été retirée des balayages généraux*, à cause de sa lenteur (environ 6 h de calcul prévues, dont plus de 90 % pour elle). Elle n'apparaît que dans le cas à deux sources. Dans l'essai rapide (10 tirages), elle donnait déjà $E approx 3","8$ avec la grille : elle s'y arrête dans un minimum local au bout de quelques centaines d'itérations.
- *ML + BIC utilise L-BFGS-B*, pas Gauss–Newton : la synthèse d'Orano parle de « Prony + ML » optimisé par Gauss–Newton ; le code du dépôt optimise la vraisemblance par L-BFGS-B. Les conclusions ne changent pas (même minimum), seuls les temps diffèrent.
- *Temps sous charge* : 8 processus sur 4 cœurs physiques (voir les durées). Les temps absolus sont surestimés d'un facteur 1,5 à 2 environ ; les rapports entre méthodes sont fiables.
- *Timeouts* : 3 (cercle) et 93 (cercle + grille, dont 88 pour ML + BIC) sur 14 400 calculs par disposition, comptés comme des échecs complets.
- *Échelle des figures* : l'axe des erreurs est en échelle log avec un plancher à $10^(-4)$ ; à $N = 2$ avec la grille et $sigma = 10^(-4)$, la médiane de ML + BIC ($5","9 dot 10^(-5)$) est sous ce plancher et apparaît à $10^(-4)$.
- *Aucun bug évident* n'a été trouvé dans les méthodes : en particulier, le blanchiment du noyau du Frank–Wolfe (`localisation/noyau.py`) est cohérent avec la conversion des intensités en unités physiques.
- *Compilation Typst* : ce document et le squelette `rapport/comparaison.typ` compilent sans erreur ni avertissement avec Typst 0.15.1.

= Reproduire les résultats

```
python main_test.py --preset nombre_sources       # relit resultats/sources/ : CSV en quelques secondes
python main_test.py --preset bruit
python main_test.py --preset source_centrale
python main_test.py --preset deux_sources
python main_test.py --preset deux_sources_orano
python main_test.py --N 2 4 6 10 --tirages 100 --nom refaire   # tout recalculer (≈ 1 h 30 par disposition)
python main.py --N 6 --tirage 21                      # une simulation et son image
```

*Fichiers.* Le code est dans `localisation/` (un fichier par méthode : `prony.py`, `ml_bic.py`, `blasso.py`, `descente.py`) ; le mode d'emploi de `main.py` et `main_test.py` est dans le `README.md` du dépôt.
- `resultats/sources/comparaison.jsonl`, `resultats/sources/deux_sources.jsonl` : vraies sources et sources estimées de chaque tirage ;
- `resultats/etudes/<étude>/tirages.csv` : une ligne par tirage ; `resume.csv` : médiane de $E$ avec IC 95 %, moyenne, détection, précision, masse parasite, temps ;
- `rapport/images/` : toutes les images du rapport ;
- `rapport/synthese_resultats.typ` (aperçu : `rapport/apercu_synthese.typ`), `rapport/annexe_donnees.typ`, et ce fichier.

Les calculs reprennent où ils s'étaient arrêtés : les sources estimées sont écrites au fil de l'eau. Les fichiers de l'ancienne organisation (`compare/`, `sparse_sources/`, `results/`) restent dans l'historique git (commit `e384cec`).

// Annexe : valeurs numériques de toutes les figures (générée par l'ancien script compare/annexe.py)
#pagebreak()
#[
  #set heading(numbering: none)
  #include "annexe_donnees.typ"
]

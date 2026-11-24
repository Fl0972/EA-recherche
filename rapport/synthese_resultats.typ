// Synthèse des résultats (≈ 6 pages). À inclure sous « = Résultats » :
//   #include "synthese_resultats.typ"
// Les figures sont dans images/ (fig_deux_sources, fig_reconstruction, fig_nsources, fig_balayages, fig_detection).
// Valeurs : médiane de E (IC 95 % bootstrap), taux de détection (d < 0,1) et précision sur les sources détectées,
// métrique avec fusion à δ = 0,1 ; tirées de annexe_donnees.typ.

== Le cas test d'Orano : la descente de gradient échoue dès deux sources

*Le cas test.* Orano a retenu l'approche la plus simple : une descente de gradient projetée sur les moindres carrés non pondérés, à laquelle on donne le vrai nombre de sources, testée uniquement sur *deux sources* dont l'une est $x$ fois plus intense que l'autre ($I_1 = x I_2$). Nous avons refait ce test avec 100 tirages par valeur de $x in {1, 3, 10, 30, 100, 300}$, dans le cas test d'Orano (sans bruit, sans source centrale) puis dans le cadre du projet ($sigma = 10^(-3)$, $I_c = 50 S$). Un tirage est réussi si chacune des deux vraies sources est retrouvée à moins de 0,1, après fusion des sources estimées distantes de moins de $delta = 0","1$ (voir plus bas) : une méthode qui coupe une source en deux morceaux voisins n'est donc pas pénalisée. La descente et Gauss–Newton partent du même point.

#figure(
  image("images/fig_deux_sources.png", width: 100%),
  caption: [Cas à deux sources, 100 tirages par point, réussite après fusion ($delta = 0","1$). (a) Cas test d'Orano : Gauss–Newton, ML + BIC et le Frank–Wolfe sont confondus à 100 % ; croix : résultats annoncés par Orano. (b) Cadre du projet, sur le cercle (traits pleins) et avec la grille (pointillés). (c) Conditionnement $kappa$ de $J^top J$ aux vraies sources (médiane et quartiles).],
) <fig:deux-sources>

*Même dans le cas le plus favorable, la descente échoue.* Sans bruit et avec le bon $N$, sa réussite passe de 71 % à $x = 1$ à 13 % à $x = 100$ et 2 % à $x = 300$ (@fig:deux-sources a). On retrouve la tendance observée par Orano (95 % à 5 %) ; l'écart à $x = 1$ vient de nos positions de départ, tirées dans tout le hangar. Dans les mêmes conditions, Gauss–Newton, ML + BIC et le Frank–Wolfe réussissent dans 99 à 100 % des tirages ; Prony, lui, descend de 97 % à 22 % quand $x$ croît.

*La cause est l'algorithme, pas le problème.* Gauss–Newton utilise le même critère, le même départ et le même $N$ que la descente, et réussit presque toujours : l'échec ne vient donc ni du critère ni de l'initialisation. Il a trois causes :
- *un conditionnement qui explose* : la position de la source faible n'agit sur les mesures qu'à travers $I_2$, donc la perte est $x^2$ fois plus plate dans ces directions. Le conditionnement $kappa$ croît comme $x^(1","8)$ et atteint $4 dot 10^5$ à $x = 100$ (@fig:deux-sources c) ; une descente de gradient demande de l'ordre de $kappa$ itérations, et 55 % des descentes épuisent leur budget de 20 000 itérations. Gauss–Newton, qui corrige la courbure, converge en quelques dizaines d'itérations (11 à 34 ms, contre 7 à 9 s pour la descente) ;
- *des sources éteintes* (6 % des descentes) : la projection met l'intensité d'une source à 0, et comme le gradient en position est proportionnel à $I_i$, cette source ne bouge plus jamais ;
- *des minima locaux avec la grille* : le critère non pondéré présente un pic sur chaque capteur intérieur, et la descente (comme Gauss–Newton parti du même point) s'arrête loin des sources dans 100 % des tirages.

*Dans le cadre du projet, la descente n'a plus aucun intérêt* (@fig:deux-sources b). Sur le cercle, ML + BIC atteint la limite statistique : il réussit pratiquement aussi souvent que Gauss–Newton parti des vraies sources (43 % contre 43 % à $x = 1$, 7 % contre 7 % à $x = 100$), alors que la descente reste en dessous (30 % à $x = 1$) et le Frank–Wolfe aussi (14 %). Avec la grille, la descente échoue toujours, quand ML + BIC réussit dans 86 à 100 % des tirages, au niveau de Gauss–Newton parti des vraies sources (87 à 100 %). Le Frank–Wolfe suit jusqu'à $x = 30$ (95 %) puis décroche (50 % à $x = 300$).

#block(fill: luma(242), inset: 8pt, radius: 4pt, width: 100%)[
  *Conclusion.* La méthode d'Orano ne fonctionne pas dès deux sources, même sans bruit et en connaissant leur nombre. Nous l'écartons pour la suite et comparons nos trois méthodes, qui estiment elles-mêmes $N$, sur des situations à plus de deux sources.
]

== Comparaison de nos trois méthodes au-delà de deux sources

*Protocole.* Pour $N in {2, 4, 6, 10}$, les positions sont uniformes dans $D$ et les intensités aléatoires de somme $S = 100$. Le cadre par défaut est celui du projet : $sigma = 10^(-3)$ et une source centrale $I_c = 50 S$, soit 50 fois la somme des sources recherchées. On fait varier un paramètre à la fois (bruit, source centrale, nombre de sources), avec 100 tirages par point. Toutes les méthodes voient exactement les mêmes sources et le même bruit. On compare deux dispositions : 64 capteurs sur le cercle, et ces 64 capteurs plus une grille de $16 times 32$ capteurs dans le hangar. Prony n'utilise que les capteurs du cercle.

*Mesure de l'erreur.* Une méthode peut représenter une source par deux ou trois sources très proches, ce qui n'est pas une vraie erreur. On fusionne donc d'abord les sources estimées distantes de moins de $delta = 0","1$ (regroupement hiérarchique à lien simple ; intensité = somme, position = barycentre pondéré par les intensités), et de même les vraies sources. Les sources fusionnées sont appariées aux vraies par l'algorithme hongrois. On rapporte alors trois indicateurs, chacun répondant à une seule question :
- *le taux de détection* : la part des vraies sources retrouvées à moins de 0,1 ;
- *la précision* : l'erreur de position $d_i = norm(hat(z)_i - z_i)$ et l'erreur d'intensité relative $r_i = abs(hat(I)_i - I_i) slash I_i$, sur les seules sources détectées ;
- *la masse parasite* : la somme des intensités des sources estimées en trop, rapportée à $S$.

Pour classer les méthodes d'un seul nombre, on garde aussi le score global
$ E = 1/N' sum_(i=1)^(N') (0","75 thin d_i + 0","25 thin r_i), $ <eq:erreur>
sur les $N'$ vraies sources après fusion, avec $d_i <= 5$ et $r_i <= 1$, une source manquée comptant $(5, 1)$. Une source manquée ajoute donc $4 slash N$ à $E$. On en rapporte la *médiane* sur les 100 tirages, avec un intervalle de confiance à 95 % par bootstrap : elle décrit le tirage typique et n'est pas dominée par les quelques tirages où une source est manquée, ce qui est acceptable quand $N$ est grand. Ces échecs restent visibles dans le taux de détection.

#figure(
  image("images/fig_reconstruction.png", width: 100%),
  caption: [Exemple de reconstruction à $N = 6$ (même tirage dans les deux panneaux). Cercles noirs : vraies sources, de surface proportionnelle à l'intensité (nombres : intensités, $S = 100$) ; carrés : ML + BIC ; triangles : Frank–Wolfe ; croix : Prony ; étoile : source centrale. (a) Cercle seul, sans source centrale : ML + BIC retrouve les 6 sources (erreur de position maximale 0,06), le Frank–Wolfe 2 sur 6, Prony 2 sur 6. (b) Cercle + grille, cadre du projet : ML + BIC et le Frank–Wolfe retrouvent les 6 sources à $10^(-3)$ et $4 dot 10^(-3)$ près ; Prony, qui ne voit que le cercle, n'en retrouve aucune.],
) <fig:reconstruction>

*Un exemple.* La @fig:reconstruction montre un tirage à six sources, d'intensités 5 à 45. Sur le cercle seul et sans source centrale (a), ML + BIC retrouve les six sources, alors que le Frank–Wolfe déplace la source d'intensité 9 de 1,5 et coupe la plus forte en deux. Ce tirage est favorable : sur le cercle, ML + BIC ne retrouve toutes les sources que dans 4 tirages sur 100 à $N = 6$ sans source centrale, et jamais avec. Avec la grille et dans le cadre du projet (b), la reconstruction est quasi parfaite pour ML + BIC comme pour le Frank–Wolfe, et c'est le cas typique.

#figure(
  image("images/fig_nsources.png", width: 100%),
  caption: [Cadre par défaut ($sigma = 10^(-3)$, $I_c = 50 S$), en fonction du nombre de sources. (a) Erreur $E$ médiane ; (b) taux de détection (vraie source retrouvée à moins de 0,1) ; (c) précision : erreur de position médiane sur les sources détectées ; (d) masse parasite moyenne, rapportée à $S$ ; (e) temps de calcul médian. Bandes : intervalles de confiance à 95 %. Traits pleins : cercle ; pointillés : cercle + grille (Prony n'utilise que le cercle).],
) <fig:nsources>

*Sur le cercle, aucune méthode ne dépasse deux sources* (@fig:nsources). À $N = 2$, ML + BIC est le meilleur ($E = 0","093$ [0,058 ; 0,15], 55 % des sources détectées, bon $N$ dans 92 % des tirages), devant le Frank–Wolfe ($E = 0","30$ ; 33 %) et Prony ($0","45$ ; 29 %). Dès $N = 4$, les trois méthodes ont $E > 1$ et ne détectent plus qu'une source sur quatre à cinq (26 % pour ML + BIC), puis une sur dix à $N = 10$. Les quelques sources détectées le sont à 0,02 à 0,05 près, mais ce chiffre est biaisé : seules comptent celles qui sont déjà à moins de 0,1. Le bruit de la source centrale noie l'information : vu depuis le cercle, le signal des sources recherchées est trop faible et trop lisse pour séparer plus de deux sources.

*Avec la grille, le problème change de nature.* ML + BIC détecte 100 % des sources à $N = 2$ et encore 97 % à $N = 10$, avec une erreur de position médiane de $5 dot 10^(-4)$ à $3","6 dot 10^(-3)$, et $E$ médiane passe de $5","9 dot 10^(-4)$ à $6","5 dot 10^(-3)$ (150 fois moins que sur le cercle à $N = 2$). Chaque source a un capteur à moins de 0,45, qui la voit des centaines de fois plus fort que les capteurs du cercle, alors que le bruit de la source centrale ne change pas d'ordre de grandeur. Le Frank–Wolfe profite aussi de la grille : il détecte 98 % puis 89 % des sources, mais il est 3 fois moins précis à $N = 10$ ($1","2 dot 10^(-2)$) et son erreur $E$ est 1,6 fois plus grande à $N = 2$ et 10 fois à $N = 10$. Il surestime aussi le nombre de sources (bon $N$ dans 21 % des tirages à $N = 10$, 1,3 source en trop en moyenne) : la pénalité $lambda$ laisse passer de petites sources parasites, qui ne pèsent toutefois que 2 % de $S$ (0,07 % pour ML + BIC).

*Prony est mille fois plus rapide*, en 1,5 ms quel que soit $N$, mais il surestime le nombre de sources dès que la source centrale est présente (2,6 sources en trop en moyenne à $N = 2$). Son intérêt est de fournir un point de départ instantané, pas une reconstruction. ML + BIC est le plus lent : de 0,2 à 0,5 s sur le cercle, et 8,6 s en médiane avec la grille à $N = 10$ (4,6 s pour le Frank–Wolfe).

#figure(
  image("images/fig_balayages.png", width: 100%),
  caption: [Erreur $E$ médiane et son intervalle de confiance à 95 %, pour $N = 4$, 6 et 10. En haut : en fonction du bruit relatif $sigma$, avec $I_c = 50 S$. En bas : en fonction de l'intensité de la source centrale rapportée à la somme des autres, avec $sigma = 10^(-3)$. Traits pleins : cercle ; pointillés : cercle + grille ; trait mixte : cadre par défaut.],
) <fig:balayages>

#figure(
  image("images/fig_detection.png", width: 100%),
  caption: [Taux de détection (part des vraies sources retrouvées à moins de 0,1), mêmes balayages et mêmes conventions que la @fig:balayages.],
) <fig:detection>

*Bruit.* Avec la grille, l'erreur médiane de ML + BIC est proportionnelle à $sigma$ pour tous les $N$ (@fig:balayages, en haut) : à $N = 4$, de $1","5 dot 10^(-4)$ à $1","5 dot 10^(-2)$ quand $sigma$ va de $10^(-4)$ à $10^(-2)$. C'est le comportement attendu d'un estimateur du maximum de vraisemblance, dont l'erreur suit le bruit. Son taux de détection reste au-dessus de 84 % même à $sigma = 10^(-2)$ et $N = 10$ (@fig:detection). Le Frank–Wolfe a une erreur médiane 4 à 15 fois plus grande, puis décroche au-delà de $sigma approx 5 dot 10^(-3)$ : à $sigma = 10^(-2)$, il ne détecte plus que 78 % des sources à $N = 4$ et 55 % à $N = 10$. Sur le cercle, seul ML + BIC à $N = 4$ et $sigma = 10^(-4)$ passe sous $E = 1$ (médiane 0,31, 55 % des sources détectées).

*Source centrale.* Elle agit exactement comme le bruit (@fig:balayages et @fig:detection, en bas). L'écart type au capteur $k$ vaut $sigma_k = sigma y_k^"brut"$, et $y_k^"brut"$ est dominé par la source centrale : multiplier $I_c$ par 10 revient à peu près à multiplier $sigma$ par 10. Sans source centrale, ML + BIC retrouve 4 sources sur le cercle ($E = 0","054$, 73 % des sources détectées, bon $N$ dans 84 % des tirages) ; avec $I_c = 10 S$ seulement, il n'y arrive plus ($E = 1","02$, 42 %). Avec la grille, ML + BIC reste à $E < 3 dot 10^(-2)$ et détecte au moins 93 % des sources jusqu'à $I_c = 200 S$ et $N = 10$, quand le Frank–Wolfe tombe à $E = 0","51$ et 74 %.

*Sensibilité au rayon de fusion.* La fusion ne change rien pour ML + BIC et Prony, qui ne renvoient presque jamais deux sources à moins de 0,1 l'une de l'autre. Elle profite au seul Frank–Wolfe avec la grille, et modestement : son erreur médiane à $N = 4$ passe de $8","0 dot 10^(-3)$ sans fusion à $6","2 dot 10^(-3)$ avec $delta = 0","1$, et sa masse parasite de 1,2 % à 0,5 % de $S$. Entre $delta = 0","05$ et $delta = 0","2$, aucune conclusion ne change.

#[
#set text(size: 9pt)
#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center, center, center, center),
    stroke: none, inset: 4pt,
    table.hline(),
    table.header([*Méthode*], [*$E$, cercle, $N = 2$*], [*$E$, grille, $N = 4$*], [*$E$, grille, $N = 10$*], [*Détection, grille, $N = 10$*], [*Précision, grille, $N = 10$*], [*Masse parasite, grille, $N = 10$*], [*Temps, grille, $N = 10$*]),
    table.hline(stroke: 0.5pt),
    [Descente (Orano)], table.cell(colspan: 7)[échoue dès 2 sources : 30 % de réussite sur le cercle, 0 % avec la grille ($x = 1$), même en connaissant $N$],
    [Prony], [0,45], [1,8], [2,7], [4 %], [0,05], [0 %], [1,5 ms],
    [Frank–Wolfe], [0,30], [$6","2 dot 10^(-3)$], [$6","8 dot 10^(-2)$], [89 %], [$1","2 dot 10^(-2)$], [2,2 %], [4,6 s],
    [*ML + BIC*], [*0,093*], [*$1","5 dot 10^(-3)$*], [*$6","5 dot 10^(-3)$*], [*97 %*], [*$3","6 dot 10^(-3)$*], [*0,07 %*], [8,6 s],
    table.hline(),
  ),
  caption: [Synthèse dans le cadre par défaut ($sigma = 10^(-3)$, $I_c = 50 S$), 100 tirages par point. $E$ : erreur médiane (@eq:erreur) ; détection : part des vraies sources retrouvées à moins de 0,1 ; précision : erreur de position médiane sur les sources détectées, dans les unités du hangar ($10 times 20$) ; masse parasite en pourcentage de $S$ ; temps médian. Prony utilise uniquement le cercle.],
) <tab:synthese>
]

#block(fill: luma(242), inset: 8pt, radius: 4pt, width: 100%)[
  *À retenir.*
  + La descente de gradient d'Orano échoue dès deux sources ; Gauss–Newton, avec le même critère et le même départ, montre que l'échec vient de l'algorithme.
  + *ML + BIC est la meilleure méthode dans tous les cas*, sans réglage : il détecte le plus de sources, estime correctement $N$, ne crée presque pas de sources parasites et localise les sources avec une erreur proportionnelle au bruit.
  + *Le facteur limitant est la disposition des capteurs, pas la méthode.* Sur le cercle seul, avec la source centrale du projet, aucune méthode ne dépasse deux sources ; avec des capteurs dans le hangar, ML + BIC détecte 97 % de 10 sources et les localise à $4 dot 10^(-3)$ près.
  + Le Frank–Wolfe est un bon second, deux fois plus rapide, mais moins précis et il surestime $N$ ; Prony est instantané mais ne sert que d'initialisation.
]

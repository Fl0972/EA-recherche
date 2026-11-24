// Synthèse des résultats (≈ 4 pages). À inclure sous « = Résultats » :
//   #include "synthese_resultats.typ"
// Figures : images/fig_deux_sources (partie 1), images/cr_exemple, cr_bruit, cr_defaut (partie 2),
// produites par figures_compte_rendu.py à partir de results/sources/comparaison_*.jsonl.

== Conditions des calculs

*Rappel du cadre* (voir @sec:cadre). Le hangar est le domaine $D = [0, 10] times [0, 20]$ ; on y cherche $N$ sources ponctuelles de positions $z_i$ et d'intensités $I_i$, de somme $S = sum_i I_i = 100$. Chaque capteur mesure la somme des intensités divisées par le carré des distances (modèle @eq:modele), plus le signal d'une source centrale connue d'intensité $I_c$, placée en $c = (5, 10)$. Le bruit est relatif : le capteur $k$ lit $y_k^"brut" (1 + sigma epsilon_k)$, si bien que la source centrale, que l'on sait retirer du signal, laisse son bruit. Sauf mention contraire, on se place dans le *cadre du projet* : $sigma = 10^(-3)$ et $I_c = 50 S$, une source centrale 50 fois plus intense que toutes les autres réunies. Deux dispositions de capteurs sont comparées : 64 capteurs régulièrement répartis sur le cercle de centre $c$ passant par les coins du hangar, et ces 64 capteurs complétés par une grille de $16 times 32$ capteurs à l'intérieur du hangar (576 capteurs).

*Machine.* Tous les calculs ont été faits sur un ordinateur portable (Intel Core i5-10210U, 4 cœurs à 1,6 GHz, 7,8 Go de mémoire vive, Windows 11), en Python 3.14 avec NumPy et SciPy. La carte graphique, intégrée au processeur, n'a pas servi : les méthodes reposent sur de petites optimisations et décompositions de matrices, que l'on a parallélisées en traitant plusieurs tirages à la fois sur les 4 cœurs. La comparaison générale a demandé environ 1 h 30 de calcul (15 min pour le cercle, 78 min avec la grille) et le cas à deux sources environ 2 h. Comme plusieurs calculs tournaient en même temps, les temps mesurés par tirage sont surestimés d'un facteur 1,5 à 2 ; les comparaisons entre méthodes restent valables, car toutes ont été mesurées dans les mêmes conditions.

*Méthodologie.* Chaque cas (un nombre de sources, un niveau de bruit, une source centrale, une disposition de capteurs) est testé sur *100 tirages* : les positions des sources sont tirées uniformément dans le hangar et leurs intensités au hasard avec une somme fixée. Toutes les méthodes reçoivent exactement les mêmes sources et le même bruit, de sorte que leurs écarts ne viennent pas du hasard des tirages. Deux sources estimées à moins de 0,1 l'une de l'autre sont regroupées en une seule (intensités additionnées, position au barycentre pondéré par les intensités), puis appariées aux vraies sources. On résume ensuite les 100 tirages ainsi :
- pour les *taux* (réussite, détection, bon nombre de sources) : la proportion de tirages ou de sources concernés ;
- pour les *erreurs de position* et les *temps de calcul* : la *médiane*, qui décrit le tirage typique sans être faussée par quelques tirages extrêmes ;
- dans les deux cas, un *intervalle de confiance à 95 %*, obtenu par bootstrap sur les tirages (2 000 rééchantillonnages) ou par la formule de Wilson pour les taux de réussite.
Un calcul qui dépasse 60 s est arrêté et relancé avec une limite de 300 s ; il en reste 2 sur 28 800, comptés comme des échecs complets.

== Deux sources : la méthode d'Orano ne fonctionne pas

*Le test.* Orano a retenu l'approche la plus simple : une descente de gradient sur l'écart quadratique non pondéré, à laquelle on donne le vrai nombre de sources, testée sur *deux sources* dont l'une est $x$ fois plus intense que l'autre ($I_1 = x I_2$). Nous avons refait ce test avec 100 tirages pour chaque $x in {1, 3, 10, 30, 100, 300}$, d'abord dans les conditions d'Orano (sans bruit, sans source centrale), puis dans le cadre du projet ($sigma = 10^(-3)$, $I_c = 50 S$). Un tirage est *réussi* si les deux sources sont retrouvées à moins de 0,1. Au total, 1 800 tirages par méthode (6 valeurs de $x$, 3 cadres, 100 tirages) ; l'intervalle de confiance à 95 % d'un taux de réussite est d'au plus $plus.minus 10$ points, et d'environ $plus.minus 3$ points près de 0 % ou de 100 %.

#figure(
  image("images/fig_deux_sources.png", width: 100%),
  caption: [Cas à deux sources, 100 tirages par point. (a) Conditions d'Orano : ML + BIC, le Frank–Wolfe et Gauss–Newton sont confondus à 100 % ; croix : résultats d'Orano. (b) Cadre du projet, cercle seul (traits pleins) et cercle + grille (pointillés). (c) Conditionnement $kappa$ de la perte à la solution (médiane et quartiles).],
) <fig:deux-sources>

*La descente échoue dès deux sources, nos méthodes réussissent.* Même sans bruit et en connaissant $N$, la réussite de la descente passe de 71 % à $x = 1$ à 13 % à $x = 100$ et 2 % à $x = 300$ (@fig:deux-sources a), ce qui reproduit la tendance observée par Orano. Dans les mêmes conditions, ML + BIC et le Frank–Wolfe réussissent dans 100 % des tirages.

*La cause est le conditionnement.* La position de la source faible n'agit sur les mesures qu'à travers son intensité $I_2$ : la perte est donc environ $x^2$ fois plus plate dans ces directions que dans celles de la source forte. Le conditionnement $kappa = lambda_max \/ lambda_min$ de la perte croît comme $x^(1","8)$ et atteint $4 dot 10^5$ à $x = 100$ (@fig:deux-sources c). Or une descente de gradient demande de l'ordre de $kappa$ itérations : 55 % des descentes s'arrêtent sur leur budget de 20 000 itérations sans avoir convergé, et 6 % « éteignent » une source (son intensité tombe à 0 et elle ne bouge plus). Gauss–Newton, qui part du même point avec le même critère mais corrige la courbure, réussit dans 100 % des tirages : l'échec vient bien de l'algorithme de descente, pas du problème.

*Dans le cadre du projet, l'écart se creuse* (@fig:deux-sources b). Avec la grille de capteurs, la descente échoue dans 100 % des tirages : le critère non pondéré présente un pic sur chaque capteur et elle s'arrête dans un minimum local. ML + BIC réussit dans 86 à 100 % des tirages. Sur le cercle seul, ML + BIC réussit aussi souvent que Gauss–Newton parti des vraies sources (43 % à $x = 1$) : il atteint la limite de ce que les données permettent.

#figure(
  image("images/fig_deux_sources_05.png", width: 100%),
  caption: [Même expérience avec un critère moins exigeant : un tirage est réussi si les deux sources sont retrouvées à moins de 0,5. Bandes : intervalles de confiance à 95 % (Wilson, 100 tirages par point). (a) Conditions d'Orano ; croix : résultats d'Orano (à 0,1 près). (b) Cadre du projet, cercle seul (traits pleins) et cercle + grille (pointillés).],
) <fig:deux-sources-05>

*Relâcher la tolérance ne sauve pas la descente* (@fig:deux-sources-05). À 0,5 près, sa réussite dans les conditions d'Orano passe de 71 % à 82 % à $x = 1$, mais reste à 13 % à $x = 100$ et 2 % à $x = 300$ : ses échecs ne sont pas des sources légèrement mal placées, mais des sources restées loin de leur cible. Dans le cadre du projet, sur le cercle, ML + BIC retrouve les deux sources à 0,5 près dans 88 % des tirages à $x = 1$ (contre 78 % pour la descente, qui connaît pourtant $N$), et l'écart se maintient jusqu'à $x = 100$ (18 % contre 11 %). Avec la grille, ML + BIC réussit dans 96 à 100 % des tirages, la descente dans aucun.

#block(fill: luma(242), inset: 8pt, radius: 4pt, width: 100%)[
  *Conclusion.* La méthode d'Orano ne fonctionne pas dès deux sources, même sans bruit et en connaissant leur nombre, à cause du conditionnement du problème. Nos méthodes ne souffrent pas de ce défaut. Nous comparons maintenant nos trois méthodes, qui estiment elles-mêmes le nombre de sources, au-delà de deux sources.
]

== Plus de deux sources : comparaison de nos trois méthodes

*Protocole.* On teste $N in {2, 4, 6, 10}$ sources, en faisant varier un paramètre à la fois autour du cadre du projet (bruit, intensité de la source centrale, nombre de sources) : 48 cas de 100 tirages, soit 4 800 tirages et 14 400 reconstructions par disposition de capteurs. Prony n'utilise que les capteurs du cercle. Après regroupement des sources proches, une vraie source est *détectée* si une source estimée est à moins de 0,5 ; pour les sources détectées, on mesure l'*erreur de position*. Les intervalles de confiance à 95 % des taux de détection valent environ $plus.minus 5$ points sur le cercle et $plus.minus 1$ point avec la grille, ce qui est petit devant les écarts commentés ci-dessous.

#figure(
  image("images/cr_exemple.png", width: 88%),
  caption: [Un tirage à $N = 6$ dans le cadre du projet. Cercles noirs : vraies sources (nombres : intensités) ; pointillés : zone de détection de rayon 0,5 ; symboles colorés : sources estimées après fusion, de taille proportionnelle à l'intensité.],
) <fig:exemple>

*Ce que « détecter » veut dire* (@fig:exemple). Sur le cercle seul, les trois méthodes retrouvent les trois sources proches des coins, mais manquent les trois autres : les deux sources les plus proches de la source centrale (intensités 45 et 23) sont confondues en une seule, et la source d'intensité 9 est placée trop loin. Avec la grille, ML + BIC et le Frank–Wolfe retrouvent les 6 sources ; Prony, qui n'utilise que le cercle, reste à 3.

#figure(
  image("images/cr_bruit.png", width: 100%),
  caption: [En fonction du bruit relatif $sigma$ ($I_c = 50 S$), pour $N = 4$, 6 et 10. En haut : part des sources détectées (à moins de 0,5). En bas : erreur de position médiane des sources détectées. Traits pleins : cercle ; pointillés : cercle + grille ; trait mixte : cadre par défaut. 100 tirages par point.],
) <fig:bruit>

*Sur le cercle, on ne détecte pas plus de la moitié des sources* (@fig:bruit, en haut). Dans le cadre par défaut, ML + BIC détecte 50 % des sources à $N = 4$ et 22 % à $N = 10$ ; même avec dix fois moins de bruit ($sigma = 10^(-4)$), il plafonne à 76 % et 38 %. Le Frank–Wolfe et Prony font nettement moins bien (33 % et 31 % à $N = 4$). Les sources détectées le sont à environ 0,1 près (de 0,03 à 0,2), quel que soit le bruit (en bas) : le cercle situe certaines sources à peu près, mais en manque beaucoup.

*Avec la grille, ML + BIC détecte presque tout, avec une erreur proportionnelle au bruit.* Il détecte au moins 99 % des sources dans le cadre par défaut et encore 94 % à $sigma = 10^(-2)$ et $N = 10$. Son erreur de position est proportionnelle à $sigma$ : de $6 dot 10^(-5)$ à $6 dot 10^(-3)$ à $N = 4$ quand $sigma$ passe de $10^(-4)$ à $10^(-2)$, comportement attendu d'un estimateur du maximum de vraisemblance. Le Frank–Wolfe suit de près à faible bruit, avec une erreur 2 à 4 fois plus grande, puis décroche au-delà de $sigma approx 2 dot 10^(-3)$ : à $sigma = 10^(-2)$, il ne détecte plus que 84 % des sources à $N = 4$ et 69 % à $N = 10$.

*La source centrale agit comme le bruit.* Le bruit au capteur $k$ vaut $sigma y_k^"brut"$, et $y_k^"brut"$ est dominé par la source centrale : multiplier $I_c$ par 10 revient à peu près à multiplier $sigma$ par 10. Sans source centrale, ML + BIC détecte 88 % des sources à $N = 4$ sur le cercle, contre 50 % avec $I_c = 50 S$. Avec la grille, il en détecte encore 97 % à $N = 10$ et $I_c = 200 S$.

#figure(
  image("images/cr_defaut.png", width: 100%),
  caption: [Cadre par défaut ($sigma = 10^(-3)$, $I_c = 50 S$). (a) Part des sources détectées selon $N$. (b) Part des sources détectées selon leur intensité rapportée à la somme $S$, pour $N = 4$ à 10. (c) Temps de calcul médian par tirage. 100 tirages par valeur de $N$ ; en (b), 300 tirages ($N = 4$, 6 et 10), soit 2 000 sources.],
) <fig:defaut>

*Ce sont les sources faibles qu'on rate* (@fig:defaut b). Sur le cercle, ML + BIC détecte 55 % des sources qui portent plus de 20 % de l'intensité totale, mais seulement 9 % de celles qui en portent moins de 2 %. Avec la grille, il détecte 93 % de ces sources très faibles, pourtant 2 500 fois moins intenses que la source centrale, et 100 % des autres ; le Frank–Wolfe en détecte 78 %.

*Temps de calcul* (@fig:defaut c). Prony répond en 1 ms quel que soit $N$. Sur le cercle, le Frank–Wolfe prend 0,06 à 0,14 s et ML + BIC 0,2 à 0,5 s. Avec la grille, les deux méthodes deviennent plus coûteuses : 0,3 à 4,6 s pour le Frank–Wolfe et 0,5 à 8,6 s pour ML + BIC à $N = 10$, ce qui reste négligeable devant la durée d'une campagne de mesures.

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr, 1.25fr, 1fr, 1fr, 1fr),
    align: (left, center) + (center,) * 6, stroke: none, inset: 3.5pt,
    table.hline(),
    table.header(
      table.cell(rowspan: 2, align: horizon)[*Méthode*], table.cell(rowspan: 2, align: horizon)[*$N$*],
      [*Cercle*], table.cell(colspan: 5)[*Cercle + grille*],
      [détection], [détection], [erreur de position], [bon nombre de sources], [masse parasite], [temps],
    ),
    table.hline(stroke: 0.5pt),
    table.cell(rowspan: 2, align: horizon)[Prony],
      [4], [31 % [26 ; 36]], table.cell(rowspan: 2, colspan: 4, align: horizon)[n'utilise que le cercle], [1,3 ms],
      [10], [17 % [15 ; 19]], [1,5 ms],
    table.hline(stroke: 0.3pt),
    table.cell(rowspan: 2, align: horizon)[Frank–Wolfe],
      [4], [33 % [29 ; 38]], [99 % [97 ; 100]], [$1","4 dot 10^(-3)$ [1,1 ; 1,7]], [57 % [47 ; 67]], [0,5 % [0,2 ; 0,7]], [0,55 s],
      [10], [17 % [15 ; 19]], [96 % [95 ; 97]], [$5","1 dot 10^(-3)$ [4,0 ; 6,3]], [21 % [14 ; 29]], [2,2 % [1,3 ; 3,3]], [4,6 s],
    table.hline(stroke: 0.3pt),
    table.cell(rowspan: 2, align: horizon)[*ML + BIC*],
      [4], [*50 % [45 ; 56]*], [*99 % [98 ; 100]*], [*$6","2 dot 10^(-4)$ [5,4 ; 6,9]*], [*93 % [88 ; 97]*], [*0,01 % [0 ; 0,03]*], [0,98 s],
      [10], [*22 % [20 ; 24]*], [*99 % [98 ; 99]*], [*$1","4 dot 10^(-3)$ [1,2 ; 1,6]*], [*85 % [78 ; 91]*], [*0,07 % [0 ; 0,18]*], [8,6 s],
    table.hline(),
  ),
  caption: [Synthèse dans le cadre par défaut ($sigma = 10^(-3)$, $I_c = 50 S$), 100 tirages par ligne ; entre crochets, intervalle de confiance à 95 % (bootstrap sur les tirages ; pour l'erreur de position, en unités de la puissance de 10 affichée). Détection : part des sources retrouvées à moins de 0,5. Erreur de position : médiane sur les sources détectées, dans les unités du hangar ($10 times 20$). Bon nombre de sources : part des tirages où le nombre de sources estimé (après fusion) est exact. Masse parasite : intensité des sources estimées en trop, en pourcentage de $S$. Temps : médiane par tirage.],
) <tab:synthese>

#block(fill: luma(242), inset: 8pt, radius: 4pt, width: 100%)[
  *À retenir.*
  + *ML + BIC est la meilleure méthode dans tous les cas*, sans réglage : il détecte le plus de sources, en particulier les plus faibles, estime correctement leur nombre et les localise avec une erreur proportionnelle au bruit.
  + *Le facteur limitant est la disposition des capteurs, pas la méthode.* Sur le cercle seul, la source centrale masque la moitié des sources dès $N = 4$ ; avec des capteurs dans le hangar, ML + BIC détecte 99 % de 10 sources à environ $10^(-3)$ près.
  + Le Frank–Wolfe est un bon second, deux fois plus rapide, mais moins précis, plus fragile au bruit, et il ajoute de petites sources parasites ; Prony est instantané mais ne sert que de point de départ.
]

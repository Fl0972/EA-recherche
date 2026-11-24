// Partie « Comparaison » du rapport — squelette généré par l'ancien script compare/typst_out.py
// (historique git). Les chiffres correspondent à resultats/etudes/*/resume.csv.
// Les figures sont en PDF (Typst ≥ 0.14 sait inclure des PDF ; sinon remplacer .pdf par .png).

== Le cas test d'Orano

#figure(
  rect(width: 100%, height: 6cm, fill: luma(230), stroke: none),
  caption: [Cas test d'Orano à deux sources (figure à fournir).],
) <fig:orano>
// Résultats existants (synthèse Orano) :
//   - Orano a retenu l'approche la plus simple, une descente de gradient sur θ = (z_i, I_i),
//     testée uniquement sur 2 sources.
//   - Réussite 95 %, 85 %, 55 %, 20 %, 5 % pour un rapport d'intensité x = 1, 3, 10, 30, 100.
//   - Ralentissement dû au conditionnement : κ ≥ c x².
//   - Gauss–Newton : 90 % à x = 100, en 59 ms.
//   - Prony + ML : 100 % jusqu'à x = 300, en 10 à 25 ms.

== Le cas à deux sources

#figure(
  image("images/ds_reussite_rapport.png", width: 100%),
  caption: [Réussite à 2 sources ($hat(N) = 2$ et chaque source à moins de 0,1) en fonction du rapport d'intensité $x = I_1 \/ I_2$, 100 tirages par point. Gauss–Newton et la descente reçoivent $N = 2$ et partent du même point. Croix : valeurs de la synthèse Orano.],
) <fig:ds-reussite>
// À rédiger :
//   - cas test d'Orano : réussite pour x = 1, 3, 10, 30, 100, 300 — descente 71 %/68 %/52 %/30 %/13 %/2 % ; Gauss–Newton 100 %/100 %/100 %/100 %/100 %/99 % ; ML + BIC 100 %/100 %/100 %/100 %/100 %/100 % ; Frank–Wolfe 100 %/100 %/100 %/100 %/100 %/100 % ; Prony 97 %/95 %/89 %/69 %/39 %/22 %
//   - projet, cercle : réussite pour x = 1, 3, 10, 30, 100, 300 — descente 30 %/30 %/22 %/13 %/3 %/1 % ; Gauss–Newton 43 %/40 %/27 %/16 %/7 %/2 % ; ML + BIC 43 %/38 %/25 %/15 %/7 %/2 % ; Frank–Wolfe 13 %/13 %/10 %/9 %/6 %/3 % ; Prony 11 %/10 %/4 %/4 %/1 %/0 %
//   - projet, cercle + grille : réussite pour x = 1, 3, 10, 30, 100, 300 — descente 0 %/0 %/0 %/0 %/0 %/0 % ; Gauss–Newton 0 %/0 %/0 %/0 %/0 %/0 % ; ML + BIC 100 %/100 %/99 %/98 %/95 %/85 % ; Frank–Wolfe 91 %/88 %/83 %/79 %/66 %/36 % ; Prony 11 %/10 %/4 %/4 %/1 %/0 %
//   - synthèse Orano (descente) : 95/85/55/20/5 % pour x = 1, 3, 10, 30, 100

#figure(
  image("images/ds_causes.png", width: 100%),
  caption: [Issue de chaque descente : réussite, source éteinte (intensité projetée à 0), budget de 20 000 itérations épuisé, limite statistique (perte finale au plus égale à celle du minimum voisin de la vérité), source bloquée au bord de $D$, minimum local.],
) <fig:ds-causes>
// À rédiger :
//   - cas test d'Orano, tous x confondus : réussite 39 %, source éteinte 6 %, budget épuisé 55 %
//   - projet, cercle, tous x confondus : réussite 16 %, source éteinte 5 %, budget épuisé 65 %, limite statistique 13 %
//   - projet, cercle + grille, tous x confondus : source éteinte 0 %, budget épuisé 6 %, bloquée au bord 30 %, minimum local 64 %
//   - cas test d'Orano, itérations de la descente (médiane) : 20000 (x = 1), 20000 (x = 3), 20000 (x = 10), 20000 (x = 30), 20000 (x = 100), 20000 (x = 300)

#figure(
  image("images/ds_conditionnement.png", width: 100%),
  caption: [À gauche : conditionnement $kappa$ de $J^top J$ (perte d'Orano, paramètres normalisés) aux vraies sources, en fonction de $x$. À droite : itérations de la descente en fonction de $kappa$, cas test d'Orano.],
) <fig:ds-conditionnement>
// À rédiger :
//   - κ médian à la vérité : 7126 (x = 1), 5404 (x = 3), 8458 (x = 10), 46916 (x = 30), 4,03 × 10^5 (x = 100), 3,62 × 10^6 (x = 300)
//   - pente log κ / log x pour x ≥ 10 : 1,78 (Orano : κ ≥ c x², pente 2)
//   - descente réussie : itérations médianes 10890 ; budget de 20 000 atteint dans 69 % des tirages

#figure(
  image("images/ds_paysage.png", width: 100%),
  caption: [Perte d'Orano en fonction de la position de la source faible (l'autre source et les intensités à leur vraie valeur) et trajectoire de la descente, sur le cercle et sur cercle + grille ; à droite, distance de chaque source estimée à la vraie source la plus proche au fil des itérations.],
) <fig:ds-paysage>
// À rédiger :
//   - carte : perte d'Orano quand la source faible est déplacée (le reste à la vérité) ; sur le cercle la perte varie lentement (vallée large et plate), la descente s'approche vite puis rampe (budget de 20 000 itérations atteint) ; sur cercle + grille la perte est hérissée d'un pic sur chaque capteur de la grille (singularité en 1/r²) avec un plateau entre les pics : la descente s'arrête au bout de quelques centaines d'itérations, loin des vraies sources

#figure(
  image("images/ds_bassins.png", width: 100%),
  caption: [Bassins d'attraction : issue de la descente selon la position de départ de la source 1 (la source 2 part toujours du même point) ; croix : départs depuis lesquels Gauss–Newton échoue.],
) <fig:ds-bassins>
// À rédiger :
//   - orano, x = 1, tirage 1 : la descente réussit depuis 11 % des départs, Gauss–Newton depuis 100 %
//   - orano, x = 30, tirage 1 : la descente réussit depuis 0 % des départs, Gauss–Newton depuis 100 %
//   - projet, x = 1, tirage 7 : la descente réussit depuis 32 % des départs, Gauss–Newton depuis 100 %

#figure(
  image("images/ds_separation.png", width: 100%),
  caption: [Réussite en fonction de la distance entre les deux sources, cas test d'Orano, $x = 1$ et $x = 30$.],
) <fig:ds-separation>
// À rédiger :
//   - x = 1, descente : réussite 14 % (distance [0, 2)), 43 % (distance [2, 4)), 84 % (distance [4, 8)), 79 % (distance [8, 23))
//   - x = 30, descente : réussite 0 % (distance [0, 2)), 0 % (distance [2, 4)), 27 % (distance [4, 8)), 48 % (distance [8, 23))

#figure(
  image("images/ds_temps.png", width: 100%),
  caption: [Temps de calcul médian par tirage en fonction de $x$.],
) <fig:ds-temps>
// À rédiger :
//   - projet, cercle, temps médian à x = 1 : descente 6,4 s, Gauss–Newton 0,013 s, ML + BIC 0,17 s, Frank–Wolfe 0,059 s, Prony 0,0013 s

== Protocole de comparaison

- Situations : $N in {2, 4, 6, 10}$ sources $z_i$ uniformes dans $D$, intensités $I_i$ aléatoires
  (Dirichlet) de somme $S = 100$ ; mêmes situations, mêmes bruits et mêmes graines pour toutes les
  méthodes et les deux dispositions de capteurs.
- Mesures : $y_k^"brut" = Phi_theta (s_k) + I_c \/ norm(s_k - c)^2$, bruit relatif
  $y_k^"brut" (1 + sigma epsilon_k)$, source centrale retirée ; $sigma_k = sigma y_k^"brut"$.
- Cadre par défaut : $I_c = 50 S$, $sigma = 10^(-3)$.
- Appariement des sources estimées et vraies par l'algorithme hongrois sur la distance euclidienne.
- Erreur d'un tirage : $E = 1/N sum_i (alpha d_i + beta r_i)$, $alpha = 0","75$, $beta = 0","25$,
  $d_i = norm(hat(z)_i - z_i)$, $r_i = abs(hat(I)_i - I_i) \/ I_i$ ; source manquée : $d_i = 5$,
  $r_i = 1$ (plafonds) ; les fausses sources sont comptées à part.
- Prony, ML + BIC et Frank–Wolfe estiment $N$. La descente de gradient d'Orano (20 000 itérations, $N$ connu) est trop lente pour ces balayages ; elle est étudiée à part dans le cas à 2 sources (@fig:ds-reussite).
- 100 tirages par point ; timeout de 60 s par tirage et par méthode (compté comme échec).

== Capteurs sur le cercle

#figure(
  image("images/bruit_cercle.png", width: 100%),
  caption: [Erreur $E$ (médiane et quartiles, échelle log) et proportion de tirages où $hat(N) = N$ en fonction du bruit relatif $sigma$, pour $I_c = 50 S$ ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$. 100 tirages par point.],
) <fig:bruit-cercle>
// À rédiger :
//   - N = 2, σ = 10^-3 : meilleure E médiane = 0,0932 (ML + BIC) ; Prony 0,453, Frank–Wolfe glissant 0,302
//   - N = 10, σ = 10^-3 : meilleure E médiane = 2,59 (ML + BIC) ; Prony 2,71, Frank–Wolfe glissant 2,76
//   - ML + BIC, N = 4 : proportion N̂ = N de 66 % → 52 % → 38 % → 23 % → 13 % → 3 % → 0 % quand σ va de 0,0001 à 0,01 (sous 50 % dès σ = 0,000464)
//   - N = 2, E médiane à σ = 0,0001 et σ = 0,01 : Prony 0,017 → 2,14 ; ML + BIC 0,00927 → 2,05 ; Frank–Wolfe glissant 0,0301 → 2,19

#figure(
  image("images/centrale_cercle.png", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction de l'intensité de la source centrale $I_c \/ S$ (0 : pas de source centrale), $sigma = 10^(-3)$ ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$.],
) <fig:centrale-cercle>
// À rédiger :
//   - N = 2, ML + BIC : E médiane 0,00307 sans source centrale (I_c/S = 0) et 0,375 pour I_c/S = 200
//   - N = 6, ML + BIC : E médiane 0,764 sans source centrale (I_c/S = 0) et 2,48 pour I_c/S = 200
//   - N = 4, proportion N̂ = N pour I_c/S = 0, 10, 25, 50, 100, 200 : ML + BIC 84 %/51 %/37 %/23 %/13 %/7 % ; Frank–Wolfe glissant 33 %/28 %/23 %/17 %/9 %/3 % ; Prony 17 %/28 %/20 %/19 %/14 %/11 %

#figure(
  image("images/nsources_cercle.png", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction du nombre de sources $N$, cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$) ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$.],
) <fig:nsources-cercle>
// À rédiger :
//   - Prony : E médiane 0,453 (N = 2), 1,84 (N = 4), 2,36 (N = 6), 2,71 (N = 10) ; N̂ = N : 16 %, 19 %, 20 %, 2 %
//   - ML + BIC : E médiane 0,0932 (N = 2), 1,24 (N = 4), 1,85 (N = 6), 2,59 (N = 10) ; N̂ = N : 92 %, 23 %, 0 %, 0 %
//   - Frank–Wolfe glissant : E médiane 0,302 (N = 2), 1,75 (N = 4), 2,22 (N = 6), 2,76 (N = 10) ; N̂ = N : 82 %, 17 %, 1 %, 0 %
//   - Prony : nombre moyen de fausses sources 2,6 (N = 2), 1,4 (N = 4), 0,66 (N = 6), 0 (N = 10)

#figure(
  image("images/temps_cercle.png", width: 100%),
  caption: [Temps de calcul par tirage (médiane et quartiles, génération des données exclue) en fonction de $N$, cadre par défaut ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$.],
) <fig:temps-cercle>
// À rédiger :
//   - Prony : temps médian 0,0013 s (N = 2), 0,0012 s (N = 4), 0,0013 s (N = 6), 0,0013 s (N = 10)
//   - ML + BIC : temps médian 0,18 s (N = 2), 0,3 s (N = 4), 0,44 s (N = 6), 0,53 s (N = 10)
//   - Frank–Wolfe glissant : temps médian 0,061 s (N = 2), 0,089 s (N = 4), 0,11 s (N = 6), 0,14 s (N = 10)
//   - tirages en timeout (cadre par défaut) : Prony 0, ML + BIC 0, Frank–Wolfe glissant 0

== Cercle et grille de capteurs

#figure(
  image("images/bruit_grille.png", width: 100%),
  caption: [Erreur $E$ (médiane et quartiles, échelle log) et proportion de tirages où $hat(N) = N$ en fonction du bruit relatif $sigma$, pour $I_c = 50 S$ ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle). 100 tirages par point.],
) <fig:bruit-grille>
// À rédiger :
//   - N = 2, σ = 10^-3 : meilleure E médiane = 0,00059 (ML + BIC) ; Prony 0,453, Frank–Wolfe glissant 0,000938
//   - N = 10, σ = 10^-3 : meilleure E médiane = 0,00654 (ML + BIC) ; Prony 2,71, Frank–Wolfe glissant 0,0678
//   - ML + BIC, N = 4 : proportion N̂ = N de 95 % → 94 % → 94 % → 93 % → 94 % → 91 % → 87 % quand σ va de 0,0001 à 0,01
//   - N = 2, E médiane à σ = 0,0001 et σ = 0,01 : Prony 0,017 → 2,14 ; ML + BIC 5,92 × 10^-5 → 0,00595 ; Frank–Wolfe glissant 9,54 × 10^-5 → 0,00957

#figure(
  image("images/centrale_grille.png", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction de l'intensité de la source centrale $I_c \/ S$ (0 : pas de source centrale), $sigma = 10^(-3)$ ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).],
) <fig:centrale-grille>
// À rédiger :
//   - N = 2, ML + BIC : E médiane 0,00012 sans source centrale (I_c/S = 0) et 0,0017 pour I_c/S = 200
//   - N = 6, ML + BIC : E médiane 0,000231 sans source centrale (I_c/S = 0) et 0,0117 pour I_c/S = 200
//   - N = 4, proportion N̂ = N pour I_c/S = 0, 10, 25, 50, 100, 200 : ML + BIC 98 %/94 %/94 %/93 %/93 %/90 % ; Frank–Wolfe glissant 79 %/57 %/59 %/57 %/46 %/45 % ; Prony 17 %/28 %/20 %/19 %/14 %/11 %

#figure(
  image("images/nsources_grille.png", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction du nombre de sources $N$, cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$) ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).],
) <fig:nsources-grille>
// À rédiger :
//   - Prony : E médiane 0,453 (N = 2), 1,84 (N = 4), 2,36 (N = 6), 2,71 (N = 10) ; N̂ = N : 16 %, 19 %, 20 %, 2 %
//   - ML + BIC : E médiane 0,00059 (N = 2), 0,00153 (N = 4), 0,00328 (N = 6), 0,00654 (N = 10) ; N̂ = N : 100 %, 93 %, 93 %, 85 %
//   - Frank–Wolfe glissant : E médiane 0,000938 (N = 2), 0,00624 (N = 4), 0,0253 (N = 6), 0,0678 (N = 10) ; N̂ = N : 89 %, 57 %, 27 %, 21 %
//   - Prony : nombre moyen de fausses sources 2,6 (N = 2), 1,4 (N = 4), 0,66 (N = 6), 0 (N = 10)

#figure(
  image("images/temps_grille.png", width: 100%),
  caption: [Temps de calcul par tirage (médiane et quartiles, génération des données exclue) en fonction de $N$, cadre par défaut ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).],
) <fig:temps-grille>
// À rédiger :
//   - Prony : temps médian 0,0015 s (N = 2), 0,0013 s (N = 4), 0,0014 s (N = 6), 0,0015 s (N = 10)
//   - ML + BIC : temps médian 0,52 s (N = 2), 0,98 s (N = 4), 1,7 s (N = 6), 8,6 s (N = 10)
//   - Frank–Wolfe glissant : temps médian 0,32 s (N = 2), 0,55 s (N = 4), 1 s (N = 6), 4,6 s (N = 10)
//   - tirages en timeout (cadre par défaut) : Prony 0, ML + BIC 0, Frank–Wolfe glissant 0

=== Cercle contre cercle et grille
#figure(
  image("images/synthese_bruit.png", width: 100%),
  caption: [Balayage en bruit : cercle seul (traits pleins) et cercle + grille (pointillés), mêmes tirages.],
) <fig:synthese-bruit>
// À rédiger :
//   - ML + BIC, N = 2, σ = 0,001 : E médiane 0,0932 (cercle) → 0,00059 (cercle + grille)
//   - ML + BIC, N = 10, σ = 0,001 : E médiane 2,59 (cercle) → 0,00654 (cercle + grille)
//   - Frank–Wolfe glissant, N = 2, σ = 0,001 : E médiane 0,302 (cercle) → 0,000938 (cercle + grille)
//   - Frank–Wolfe glissant, N = 10, σ = 0,001 : E médiane 2,76 (cercle) → 0,0678 (cercle + grille)
#figure(
  image("images/synthese_centrale.png", width: 100%),
  caption: [Balayage en source centrale : cercle seul (traits pleins) et cercle + grille (pointillés).],
) <fig:synthese-centrale>
// À rédiger :
//   - ML + BIC, N = 2, I_c/S = 0 : E médiane 0,00307 (cercle) → 0,00012 (cercle + grille)
//   - ML + BIC, N = 10, I_c/S = 0 : E médiane 1,55 (cercle) → 0,000316 (cercle + grille)
//   - Frank–Wolfe glissant, N = 2, I_c/S = 0 : E médiane 0,00671 (cercle) → 0,00027 (cercle + grille)
//   - Frank–Wolfe glissant, N = 10, I_c/S = 0 : E médiane 1,88 (cercle) → 0,003 (cercle + grille)
#figure(
  image("images/synthese_nsources_temps.png", width: 100%),
  caption: [Erreur $E$, proportion de $hat(N) = N$ et temps médian en fonction de $N$, cadre par défaut : cercle seul (traits pleins) et cercle + grille (pointillés).],
) <fig:synthese-nsources>
// À rédiger :
//   - Prony : E médiane à N = 10 2,71 (cercle) → 2,71 (grille) ; temps médian 0,0013 s → 0,0015 s
//   - ML + BIC : E médiane à N = 10 2,59 (cercle) → 0,00654 (grille) ; temps médian 0,53 s → 8,6 s
//   - Frank–Wolfe glissant : E médiane à N = 10 2,76 (cercle) → 0,0678 (grille) ; temps médian 0,14 s → 4,6 s

== Synthèse

#include "tableau_synthese.typ"

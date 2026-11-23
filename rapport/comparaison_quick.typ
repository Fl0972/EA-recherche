// Partie « Comparaison » du rapport — squelette généré par python -m compare.run
// (compare/typst_out.py). Les observations « À rédiger » sont calculées depuis results/*.csv.
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

== Protocole de comparaison

- Situations : $N in {2, 4, 6, 10}$ sources $z_i$ uniformes dans $D$, intensités $I_i$ aléatoires
  (Dirichlet) de somme $S = 100$ ; mêmes situations, mêmes bruits et mêmes graines pour toutes les
  méthodes et les deux dispositions de capteurs.
- Mesures : $y_k^"brut" = Phi_theta (s_k) + I_c \/ norm(s_k - c)^2$, bruit relatif
  $y_k^"brut" (1 + sigma epsilon_k)$, source centrale retirée ; $sigma_k = sigma y_k^"brut"$.
- Cadre par défaut : $I_c = 50 S$, $sigma = 10^(-3)$.
- Appariement des sources estimées et vraies par l'algorithme hongrois sur la distance euclidienne.
- Erreur d'un tirage : $E = 1/N sum_i (alpha d_i + beta r_i)$, $alpha = 0,75$, $beta = 0,25$,
  $d_i = norm(hat(z)_i - z_i)$, $r_i = abs(hat(I)_i - I_i) \/ I_i$ ; source manquée : $d_i = 5$,
  $r_i = 1$ (plafonds) ; les fausses sources sont comptées à part.
- Prony, ML + BIC et Frank–Wolfe estiment $N$. La descente de gradient d'Orano (20 000 itérations, $N$ connu) est trop lente pour ces balayages ; elle est étudiée à part dans le cas à 2 sources (@fig:ds-reussite).
- 10 tirages par point ; timeout de 60 s par tirage et par méthode (compté comme échec).

== Capteurs sur le cercle

#figure(
  image("../figures/comparaison/quick/bruit_cercle.pdf", width: 100%),
  caption: [Erreur $E$ (médiane et quartiles, échelle log) et proportion de tirages où $hat(N) = N$ en fonction du bruit relatif $sigma$, pour $I_c = 50 S$ ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$. 10 tirages par point.],
) <fig:bruit-cercle>
// À rédiger :
//   - N = 2, σ = 10^-3 : meilleure E médiane = 0,185 (ML + BIC) ; Prony 1,29, Frank–Wolfe glissant 0,43
//   - N = 10, σ = 10^-3 : meilleure E médiane = 2,53 (ML + BIC) ; Prony 2,6, Frank–Wolfe glissant 2,79
//   - ML + BIC, N = 4 : proportion N̂ = N de 70 % → 30 % → 0 % quand σ va de 0,0001 à 0,01 (sous 50 % dès σ = 0,001)
//   - N = 2, E médiane à σ = 0,0001 et σ = 0,01 : Prony 0,0194 → 2,76 ; ML + BIC 0,0158 → 2,21 ; Frank–Wolfe glissant 0,037 → 2,23

#figure(
  image("../figures/comparaison/quick/centrale_cercle.pdf", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction de l'intensité de la source centrale $I_c \/ S$ (0 : pas de source centrale), $sigma = 10^(-3)$ ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$.],
) <fig:centrale-cercle>
// À rédiger :
//   - N = 2, ML + BIC : E médiane 0,00748 sans source centrale (I_c/S = 0) et 1,18 pour I_c/S = 200
//   - N = 6, ML + BIC : E médiane 0,723 sans source centrale (I_c/S = 0) et 2,48 pour I_c/S = 200
//   - N = 4, proportion N̂ = N pour I_c/S = 0, 50, 200 : ML + BIC 90 %/30 %/10 % ; Frank–Wolfe glissant 30 %/30 %/0 % ; Prony 30 %/10 %/0 %

#figure(
  image("../figures/comparaison/quick/nsources_cercle.pdf", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction du nombre de sources $N$, cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$) ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$.],
) <fig:nsources-cercle>
// À rédiger :
//   - Prony : E médiane 1,29 (N = 2), 1,34 (N = 4), 2,34 (N = 6), 2,6 (N = 10) ; N̂ = N : 10 %, 10 %, 20 %, 20 %
//   - ML + BIC : E médiane 0,185 (N = 2), 1,23 (N = 4), 1,69 (N = 6), 2,53 (N = 10) ; N̂ = N : 80 %, 30 %, 0 %, 0 %
//   - Frank–Wolfe glissant : E médiane 0,43 (N = 2), 1,79 (N = 4), 2,34 (N = 6), 2,79 (N = 10) ; N̂ = N : 70 %, 30 %, 0 %, 0 %
//   - Prony : nombre moyen de fausses sources 3,5 (N = 2), 1,8 (N = 4), 0,8 (N = 6), 0 (N = 10)

#figure(
  image("../figures/comparaison/quick/temps_cercle.pdf", width: 100%),
  caption: [Temps de calcul par tirage (médiane et quartiles, génération des données exclue) en fonction de $N$, cadre par défaut ; 64 capteurs $s_k$ sur le cercle de centre $c$ passant par les coins de $D$.],
) <fig:temps-cercle>
// À rédiger :
//   - Prony : temps médian 0,0014 s (N = 2), 0,0012 s (N = 4), 0,0013 s (N = 6), 0,0013 s (N = 10)
//   - ML + BIC : temps médian 0,18 s (N = 2), 0,43 s (N = 4), 0,48 s (N = 6), 0,83 s (N = 10)
//   - Frank–Wolfe glissant : temps médian 0,06 s (N = 2), 0,12 s (N = 4), 0,12 s (N = 6), 0,16 s (N = 10)
//   - tirages en timeout (cadre par défaut) : Prony 0, ML + BIC 0, Frank–Wolfe glissant 0

== Cercle et grille de capteurs

#figure(
  image("../figures/comparaison/quick/bruit_grille.pdf", width: 100%),
  caption: [Erreur $E$ (médiane et quartiles, échelle log) et proportion de tirages où $hat(N) = N$ en fonction du bruit relatif $sigma$, pour $I_c = 50 S$ ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle). 10 tirages par point.],
) <fig:bruit-grille>
// À rédiger :
//   - N = 2, σ = 10^-3 : meilleure E médiane = 0,000404 (ML + BIC) ; Prony 1,29, Frank–Wolfe glissant 0,000713
//   - N = 10, σ = 10^-3 : meilleure E médiane = 0,0132 (ML + BIC) ; Prony 2,6, Frank–Wolfe glissant 0,0963
//   - ML + BIC, N = 4 : proportion N̂ = N de 80 % → 80 % → 70 % quand σ va de 0,0001 à 0,01
//   - N = 2, E médiane à σ = 0,0001 et σ = 0,01 : Prony 0,0194 → 2,76 ; ML + BIC 4,06 × 10^-5 → 0,00385 ; Frank–Wolfe glissant 7,17 × 10^-5 → 0,00678

#figure(
  image("../figures/comparaison/quick/centrale_grille.pdf", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction de l'intensité de la source centrale $I_c \/ S$ (0 : pas de source centrale), $sigma = 10^(-3)$ ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).],
) <fig:centrale-grille>
// À rédiger :
//   - N = 2, ML + BIC : E médiane 0,000101 sans source centrale (I_c/S = 0) et 0,00121 pour I_c/S = 200
//   - N = 6, ML + BIC : E médiane 0,000315 sans source centrale (I_c/S = 0) et 0,0111 pour I_c/S = 200
//   - N = 4, proportion N̂ = N pour I_c/S = 0, 50, 200 : ML + BIC 80 %/80 %/80 % ; Frank–Wolfe glissant 80 %/80 %/60 % ; Prony 30 %/10 %/0 %

#figure(
  image("../figures/comparaison/quick/nsources_grille.pdf", width: 100%),
  caption: [Erreur $E$ et proportion de $hat(N) = N$ en fonction du nombre de sources $N$, cadre par défaut ($I_c = 50 S$, $sigma = 10^(-3)$) ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).],
) <fig:nsources-grille>
// À rédiger :
//   - Prony : E médiane 1,29 (N = 2), 1,34 (N = 4), 2,34 (N = 6), 2,6 (N = 10) ; N̂ = N : 10 %, 10 %, 20 %, 20 %
//   - ML + BIC : E médiane 0,000404 (N = 2), 0,00104 (N = 4), 0,0031 (N = 6), 0,0132 (N = 10) ; N̂ = N : 100 %, 80 %, 90 %, 70 %
//   - Frank–Wolfe glissant : E médiane 0,000713 (N = 2), 0,00521 (N = 4), 0,0329 (N = 6), 0,0963 (N = 10) ; N̂ = N : 90 %, 80 %, 20 %, 30 %
//   - Prony : nombre moyen de fausses sources 3,5 (N = 2), 1,8 (N = 4), 0,8 (N = 6), 0 (N = 10)

#figure(
  image("../figures/comparaison/quick/temps_grille.pdf", width: 100%),
  caption: [Temps de calcul par tirage (médiane et quartiles, génération des données exclue) en fonction de $N$, cadre par défaut ; 64 capteurs sur le cercle et 512 capteurs en grille $16 times 32$ dans $D$ (Prony n'utilise que les capteurs du cercle).],
) <fig:temps-grille>
// À rédiger :
//   - Prony : temps médian 0,0015 s (N = 2), 0,0013 s (N = 4), 0,0014 s (N = 6), 0,0013 s (N = 10)
//   - ML + BIC : temps médian 0,54 s (N = 2), 0,98 s (N = 4), 1,8 s (N = 6), 5,6 s (N = 10)
//   - Frank–Wolfe glissant : temps médian 0,37 s (N = 2), 0,58 s (N = 4), 1,2 s (N = 6), 3 s (N = 10)
//   - tirages en timeout (cadre par défaut) : Prony 0, ML + BIC 0, Frank–Wolfe glissant 0

=== Cercle contre cercle et grille
#figure(
  image("../figures/comparaison/quick/synthese_bruit.pdf", width: 100%),
  caption: [Balayage en bruit : cercle seul (traits pleins) et cercle + grille (pointillés), mêmes tirages.],
) <fig:synthese-bruit>
// À rédiger :
//   - ML + BIC, N = 2 : E médiane 0,185 (cercle) → 0,000404 (cercle + grille)
//   - ML + BIC, N = 10 : E médiane 2,53 (cercle) → 0,0132 (cercle + grille)
//   - Frank–Wolfe glissant, N = 2 : E médiane 0,43 (cercle) → 0,000713 (cercle + grille)
//   - Frank–Wolfe glissant, N = 10 : E médiane 2,79 (cercle) → 0,0963 (cercle + grille)
#figure(
  image("../figures/comparaison/quick/synthese_centrale.pdf", width: 100%),
  caption: [Balayage en source centrale : cercle seul (traits pleins) et cercle + grille (pointillés).],
) <fig:synthese-centrale>
// À rédiger :
//   - ML + BIC, N = 2 : E médiane 0,00748 (cercle) → 0,000101 (cercle + grille)
//   - ML + BIC, N = 10 : E médiane 1,53 (cercle) → 0,000301 (cercle + grille)
//   - Frank–Wolfe glissant, N = 2 : E médiane 0,0154 (cercle) → 0,000481 (cercle + grille)
//   - Frank–Wolfe glissant, N = 10 : E médiane 1,94 (cercle) → 0,00271 (cercle + grille)
#figure(
  image("../figures/comparaison/quick/synthese_nsources_temps.pdf", width: 100%),
  caption: [Erreur $E$, proportion de $hat(N) = N$ et temps médian en fonction de $N$, cadre par défaut : cercle seul (traits pleins) et cercle + grille (pointillés).],
) <fig:synthese-nsources>
// À rédiger :
//   - Prony : E médiane à N = 10 2,6 (cercle) → 2,6 (grille) ; temps médian 0,0013 s → 0,0013 s
//   - ML + BIC : E médiane à N = 10 2,53 (cercle) → 0,0132 (grille) ; temps médian 0,83 s → 5,6 s
//   - Frank–Wolfe glissant : E médiane à N = 10 2,79 (cercle) → 0,0963 (grille) ; temps médian 0,16 s → 3 s

== Synthèse

#include "../figures/comparaison/quick/synthese.typ"

// Compte rendu : pourquoi le Frank-Wolfe échoue, comment ML + BIC atteint la borne de Cramér–Rao,
// et pourquoi il faut des capteurs à l'intérieur du domaine.
// Compiler depuis le dossier sparse_sources/ :  typst compile COMPTE_RENDU_METHODES.typ
// (les figures sont lues dans results/).

#set document(title: "Localisation de sources : Frank-Wolfe, ML + BIC et placement des capteurs", author: "Cyril Flury-Hérard")
#set page(paper: "a4", margin: (x: 2.2cm, y: 2.3cm), numbering: "1 / 1")
#set text(lang: "fr", size: 10.5pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
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
  #text(size: 17pt, weight: "bold")[Localisation de sources ponctuelles]
  #v(0.2em)
  #text(size: 12.5pt)[Pourquoi le Frank-Wolfe échoue, pourquoi ML + BIC atteint la borne de Cramér–Rao, \ et pourquoi il faut placer des capteurs à l'intérieur du domaine]
  #v(0.5em)
  Cyril Flury-Hérard — EA recherche — novembre 2026 \
  #text(size: 9pt)[Branche `travail-sur-le-probleme-inverse`, dossier `sparse_sources/`. Tous les chiffres viennent des CSV de `sparse_sources/results/`.]
]

#v(0.6em)

#encadre[Résumé.][
+ *Le Frank-Wolfe (BLASSO)* résout un problème convexe sur les mesures, mais il a trois défauts sur notre problème : la pénalité $lambda sum I_k$ biaise les positions, la condition de Duval–Peyré qui garantit son succès est fausse dès 3 sources, et il ne pondère pas les capteurs par leur bruit.
+ *Au-delà de ces défauts, aucune méthode ne peut retrouver beaucoup de sources avec le cercle de capteurs du projet* : sur les $3K$ paramètres, les capteurs n'en mesurent qu'environ 50 (sans bruit) et une douzaine à la précision 0,1 (avec le bruit du projet). Chercher un minimum, c'est résoudre un système linéaire (les équations normales) ; une direction non mesurée est une direction où le critère est plat, donc où le minimum est n'importe où (section 3).
+ *ML + BIC* sépare les deux questions « où » (maximum de vraisemblance, sans pénalité, capteurs pondérés par leur bruit) et « combien » (critère BIC, sans réglage). C'est la meilleure méthode dans tous les tests, et son erreur est au niveau de la borne de Cramér–Rao : on ne peut pas faire mieux en changeant d'algorithme (sections 4 et 5).
+ *Le levier est la géométrie des capteurs.* Vu du cercle, le signal en $1\/r^2$ est un noyau de Poisson qui écrase les détails fins d'un facteur $(rho\/R)^n$. Avec une grille de capteurs dans le domaine, ML + BIC retrouve *45 sources sur 50* au lieu de 4 ; un seul capteur mobile en serpentin en retrouve 42 (section 7).
]

= Le problème

On cherche $K$ sources ponctuelles (positions $z_k$, intensités $I_k > 0$, $K$ inconnu) dans $D = [0,10] times [0,20]$. Les $M = 64$ capteurs sont sur le cercle circonscrit à $D$, de centre $c = (5, 10)$ et de rayon $R = sqrt(125) approx 11","18$. Avec la signature d'une source unité
$ phi(z) = (G(x_1, z), dots, G(x_M, z)) in RR^M, quad G(x, z) = 1 / norm(x - z)^2, $
le modèle est
$ m = f(theta) + w, quad f(theta) = sum_(k=1)^K I_k thin phi(z_k), quad theta = (z_1, I_1, dots, z_K, I_K) in RR^(3K). $

*Le bruit et la source centrale.* Le capteur $i$ lit $y_i (1 + sigma epsilon_i)$ avec $epsilon_i tilde cal(N)(0,1)$ et $sigma in [10^(-4), 10^(-2)]$. Les mesures brutes contiennent une source centrale connue d'intensité $I_c = 5000$ (contre 100 pour toutes les autres sources réunies), que l'on retire. Son signal part, son bruit reste :
$ s_i = sigma thin y_i approx sigma (I_c / R^2 + f_i) = sigma (40 + f_i), $
alors que le signal utile vaut environ 7 en moyenne quadratique. *Le bruit est dominé par la source centrale* : c'est la difficulté principale du projet.

*Notation blanchie.* On divise chaque capteur par l'écart type de son bruit : $m_w = m\/s$, $f_w = f\/s$ (composante par composante). Le bruit blanchi $w_w = w\/s$ suit alors $cal(N)(0, I_M)$. On note $J = partial f_w \/ partial theta in RR^(M times 3K)$ la jacobienne blanchie :
$ J = op("diag")(1\/s_i) [ phi(z_1), dots, phi(z_K) | I_1 nabla phi(z_1), dots, I_K nabla phi(z_K) ], quad nabla_z G(x, z) = (2 (x - z)) / norm(x - z)^4 . $

= Le Frank-Wolfe glissant (BLASSO) : comment il marche

== Le problème résolu

Le BLASSO cherche une mesure positive $mu = sum_k I_k delta_(z_k)$ qui minimise
$ min_(mu >= 0) thin 1/2 norm(m - A mu)^2 + lambda norm(mu)_"TV", quad A mu = sum_k I_k phi(z_k), quad norm(mu)_"TV" = sum_k I_k . $ <eq:blasso>
L'idée : on « relâche » le problème non convexe en $(z_k, I_k)$ en un problème *convexe* sur l'espace des mesures. La norme TV joue le rôle de la norme $ell^1$ du LASSO : elle favorise les solutions faites de peu de Diracs. Le nombre de sources n'est pas une inconnue : c'est $lambda$ qui le fixe.

== Certificat dual et condition d'optimalité

Pour le résidu $r = m - A mu$, on définit le certificat
$ eta(z) = 1/lambda ⟨phi(z), r⟩ . $
*$mu$ est solution de @eq:blasso si et seulement si $eta(z) <= 1$ sur $D$ et $eta(z_k) = 1$ en chaque Dirac.* C'est l'analogue des conditions KKT du LASSO : $eta(z)$ mesure combien on gagnerait à ajouter une petite source en $z$ ; tant qu'un point dépasse 1, ajouter une source y fait baisser l'objectif plus que ce qu'elle coûte en pénalité.

== L'algorithme (Denoyelle, Duval, Peyré, Soubies 2019)

Frank-Wolfe classique : pour minimiser une fonction convexe $F$ sur un convexe $C$, on se déplace vers le point de $C$ qui minimise le gradient linéarisé, $s_t = op("arg min")_(s in C) ⟨nabla F(x_t), s⟩$. Sur une boule de mesures positives de norme TV bornée, les points extrêmes sont des Diracs $tau delta_z$, et le gradient de $1/2 norm(m - A mu)^2$ vaut $-A^* r$ : *la direction de descente est un Dirac placé au maximum de $eta$*. L'algorithme (`blasso.py`, `sliding_frank_wolfe`) répète :

+ résidu $r = m - A mu$, puis $z^star = op("arg max")_D thin eta$ (grille fine, puis L-BFGS-B) ;
+ si $eta(z^star) <= 1 + "tol"$ : arrêt, $mu$ est *certifiée optimale* ;
+ ajout d'un Dirac en $z^star$, puis intensités par LASSO positif à positions fixées ;
+ *glissement* : descente conjointe de toutes les positions et intensités sur @eq:blasso (non convexe, mais local) ;
+ suppression des Diracs nuls, fusion des Diracs confondus ; à la fin, débiaisage des intensités (moindres carrés positifs à positions fixées).

Le noyau brut ne marche pas du tout (voir ci-dessous) : le projet utilise le noyau *normalisé* $psi(z) = phi(z) \/ norm(phi(z))$, ce qui revient à la pénalité $lambda sum_k I_k norm(phi(z_k))$, avec $lambda = 1","5 thin sigma norm(y)$.

== Le théorème qui garantit le succès

Duval et Peyré (2015) montrent que le BLASSO retrouve exactement $K$ sources, à distance $O(norm(w))$ des vraies, *si* trois hypothèses sont vraies. Avec
$ Gamma = [ phi(z_1), dots, phi(z_K) | partial_1 phi(z_1), dots | partial_2 phi(z_1), dots ] in RR^(M times 3K), quad eta_V = ⟨phi(dot), (Gamma^top)^+ (bb(1)_K, 0_(2K))⟩, $
- *(H1) $op("rang") Gamma = 3K$ — ce qui exige au moins $M >= 3K$* ;
- *(H2)* $eta_V(z) < 1$ hors des sources ; *(H3)* $nabla^2 eta_V (z_k) prec 0$.

Point important pour la suite : *$Gamma$ a les mêmes colonnes que la jacobienne $J$* (à un facteur $I_k$ et au blanchiment près). (H1) est donc exactement « toutes les directions de l'espace des paramètres sont mesurées ».

= Pourquoi le Frank-Wolfe échoue

== Quatre causes, de la plus spécifique à la plus fondamentale

*1. Le noyau brut est sous-harmonique.* En 2D, $Delta_z (1\/norm(x - z)^2) = 4\/norm(x - z)^4 > 0$. Le pré-certificat $eta_V$ ne peut donc pas avoir de maximum strict à l'intérieur de $D$ : (H2) et (H3) sont fausses dans *100 % des tirages*, et le BLASSO brut renvoie 20 à 40 sources parasites collées aux coins. De plus $norm(phi(z))$ varie d'un facteur 215 sur $D$ : une source près d'un capteur produit 215 fois plus de signal par unité de pénalité. La normalisation corrige ce défaut (une source seule est alors toujours identifiée, par Cauchy–Schwarz).

*2. Même normalisé, (H2) devient fausse dès 3 sources.* Taux de tirages où la condition est vraie, avec $alpha = 1$ : 100 % pour 1 et 2 sources, 20 % pour 3 et 4, 0 % pour 5 (`r2_alpha.csv`). Elle échoue quand deux sources sont proches, ou quand une paire est à moins de 4 à 5 d'un capteur — ce qui arrive souvent car le cercle passe par les coins de $D$. Quand la condition est fausse, le Frank-Wolfe ne réussit que dans 2 à 6 % des cas.

*3. Un seul paramètre $lambda$ pour deux rôles contradictoires.* Il faut $lambda$ de l'ordre de $norm(w)$ pour rejeter les fausses sources, mais la pénalité biaise alors positions et intensités d'un $O(lambda)$ (section 5.3). Un $lambda$ petit laisse passer des sources parasites, un $lambda$ grand fusionne et décale les vraies.

*4. Fondamentalement : les capteurs ne mesurent pas assez de directions.* C'est la cause qui domine dès que $K$ dépasse 5, et elle touche *toutes* les méthodes, ML + BIC comprise. La section suivante explique pourquoi elle compte même si on « cherche un minimum ».

== Chercher un minimum, c'est résoudre un système

L'objection naturelle : on ne résout pas $f(theta) = m$, on minimise un critère ; un minimum existe toujours, alors pourquoi le nombre de directions mesurées compterait-il ? Réponse : *un minimum existe, mais il n'est ni unique ni proche de la vérité*. Voici pourquoi.

*Linéarisation.* Près de la vraie valeur $theta^star$, on écrit $theta = theta^star + delta$. Puisque $m_w = f_w (theta^star) + w_w$,
$ chi^2(theta^star + delta) = norm(m_w - f_w (theta^star + delta))^2 approx norm(w_w - J delta)^2 . $
*Minimiser $chi^2$ revient donc à résoudre au sens des moindres carrés le système linéaire $J delta = w_w$* ($M$ équations, $3K$ inconnues). La condition de minimum $nabla chi^2 = 0$ s'écrit
$ J^top J thin delta = J^top w_w $ <eq:normales>
(les équations normales) : *3K équations pour 3K inconnues*. Chercher le minimum, c'est résoudre ce système carré. Le critère pénalisé du BLASSO conduit au même système, avec un terme en $lambda$ en plus au second membre.

*Décomposition en valeurs singulières.* Avec $J = sum_j s_j u_j v_j^top$ ($s_1 >= s_2 >= dots >= 0$), les solutions de @eq:normales sont
$ hat(delta) = sum_(j : s_j > 0) (u_j^top w_w) / s_j thin v_j thin + thin ("n'importe quoi dans" ker J). $ <eq:svd>
Chaque $v_j$ est une « direction » de l'espace des paramètres (une combinaison de positions et d'intensités),*il est dans l'espace 3K  c'est les paramètres , les u sont les mesures* et $s_j$ dit combien les mesures changent quand on se déplace dans cette direction. Deux cas :

- *$s_j = 0$ : direction non mesurée.* Se déplacer le long de $v_j$ ne change pas les mesures, donc ne change pas $chi^2$. Le critère a une *vallée plate* : une infinité de minima de même valeur, et l'algorithme s'arrête n'importe où dans la vallée selon son point de départ (ou, pour le BLASSO, au point de la vallée où $sum I_k$ est le plus petit, qui n'a aucune raison d'être la vérité). Avec $K = 50$ et $M = 64$, $op("rang") J <= 64 < 150$ : *au moins 86 directions plates*, quel que soit l'algorithme.
- *$s_j$ petit : direction mal mesurée.* Comme $u_j^top w_w tilde cal(N)(0,1)$, le minimum est déplacé le long de $v_j$ d'une quantité d'écart type $1\/s_j$. Le minimum existe et l'algorithme le trouve, mais *le minimum lui-même est loin de la vérité* : c'est le bruit qui le place. Ce n'est pas un défaut d'optimisation.

*Le lien avec la borne de Cramér–Rao.* D'après @eq:svd, la covariance de l'erreur vaut
$ op("Cov")(hat(delta)) = sum_j (v_j v_j^top) / s_j^2 = (J^top J)^(-1) = F^(-1), $
qui est exactement la borne de Cramér–Rao (section 5). « Compter les directions mesurables » et « calculer la CRB » sont deux façons de lire la même décomposition. Le compte « sans bruit » (seuil $10^(-6) s_1$) dit quelles directions sont visibles en principe ; avec le bruit, il faut que $1\/s_j$ soit plus petit que la précision voulue.

#attention[
*Vérification expérimentale : imposer $K = 50$* (`COMPTE_RENDU_CAPTEURS.md`, §1, un tirage, $sigma = 10^(-4)$). Le ML à $K = 50$ imposé trouve une solution avec $chi^2\/M = 0","26$, alors que la vérité donne $chi^2\/M approx 1$ : *la vérité n'est pas le minimum*. Même en partant des vraies positions, l'optimisation s'en éloigne pour descendre dans la vallée : 5 sources retrouvées à 0,5 près, contre 9 pour ML + BIC qui ne garde que les directions justifiées.
]

== Combien de directions mesure le cercle ?

#figure(
  table(
    columns: 6, align: center, stroke: 0.4pt + gray,
    table.header[*sources $K$*][*paramètres $3K$*][*sans bruit, 64 capteurs*][*sans bruit, 1024 capteurs*][*à 0,1 près, $sigma = 10^(-3)$, 64 capteurs*][*plus petite distance entre sources*],
    [3], [9], [9], [9], [4], [4,3],
    [10], [30], [29], [29], [7], [0,74],
    [20], [60], [46], [47], [7], [0,62],
    [50], [150], [56], [71], [12], [0,15],
  ),
  caption: [Directions mesurées par le cercle (médiane sur 5 tirages). Colonnes « sans bruit » : `ml_beaucoup_directions.csv` (valeurs singulières de la jacobienne à colonnes normalisées au-dessus de $10^(-6)$ fois la plus grande). Colonne « à 0,1 près » : dans combien de directions l’erreur minimale possible est-elle inférieure à 0,1, ie nombre de directions où $1\/s_j < 0","1$ avec la jacobienne blanchie, source centrale comprise.],
) <tab:directions>

Sans bruit, le nombre de directions plafonne autour de 50 à 70, même avec 16 fois plus de capteurs. Avec le bruit réaliste du projet, *seule une douzaine de directions sont connues à 0,1 près pour 50 sources* : il y a 150 inconnues, et les données en fixent une douzaine. La section 7 explique d'où vient ce plafond.

== Ce que donnent le Frank-Wolfe et ML + BIC

#figure(
  table(
    columns: 5, align: center, stroke: 0.4pt + gray,
    table.header[*sources*][*$sigma$*][*Prony*][*Frank-Wolfe normalisé*][*ML + BIC*],
    [1], [$10^(-3)$], [15 %], [100 %], [*100 %*],
    [1], [$10^(-2)$], [0 %], [100 %], [*100 %*],
    [2], [$10^(-4)$], [85 %], [90 %], [*95 %*],
    [2], [$10^(-3)$], [15 %], [70 %], [*75 %*],
    [2], [$3 times 10^(-3)$], [0 %], [40 %], [*75 %*],
    [3], [$10^(-4)$], [55 %], [40 %], [*80 %*],
    [3], [$10^(-3)$], [10 %], [5 %], [*40 %*],
    [5], [$10^(-4)$], [5 %], [0 %], [*50 %*],
  ),
  caption: [Taux de réussite sur le scénario du projet (20 tirages par case, mêmes mesures pour toutes les méthodes ; réussite = bon $K$ et chaque source à moins d'un quart de la plus petite distance entre sources). `projet_methodes_resume.csv`.],
)

Pour 1 ou 2 sources, (H1)–(H3) sont vraies et le Frank-Wolfe marche presque aussi bien que ML + BIC. Dès 3 sources, (H2) casse et le biais de $lambda$ pèse : le Frank-Wolfe tombe à 5 % là où ML + BIC garde 40 %. Pour 50 sources sur le cercle, *les deux méthodes échouent de la même façon* (Frank-Wolfe : 3 sources retrouvées à 0,5 près ; ML + BIC : 4) : c'est le manque de directions, commun à toutes les méthodes.

= ML + BIC : comment il marche

La méthode répond séparément aux deux questions que le BLASSO mélange dans $lambda$ : *où* sont les sources (maximum de vraisemblance), et *combien* il y en a (BIC).

== Vraisemblance et blanchiment

Le bruit est gaussien, indépendant, de variance $s_i^2$ connue. Donc
$ -2 log p(m | theta) = sum_(i=1)^M ((m_i - f_i (theta)) / s_i)^2 + "cste" = chi^2(theta) + "cste" . $
*Maximiser la vraisemblance, c'est minimiser des moindres carrés pondérés par $1\/s_i^2$* : on fait moins confiance aux capteurs les plus bruités. Il n'y a *aucune pénalité* : rien ne tire les positions ou les intensités loin de leur valeur. $s_i = sigma y_i$ est calculé sur les mesures brutes, source centrale comprise (`relative_noise_std`).

À $K$ fixé, l'estimateur est
$ hat(theta)_K = op("arg min")_(z_k in D, thin I_k >= 0) chi^2(theta), $
un problème de moindres carrés non linéaires : linéaire en $I$, non linéaire en $z$. Il est résolu par L-BFGS-B avec gradient exact, $nabla chi^2 = -2 J^top r_w$ où $r_w = m_w - f_w (theta)$.

== L'algorithme glouton

Le problème n'est pas convexe : il faut un bon point de départ. On construit la solution source par source (`ml.py`, `greedy_ml`).

*Ajout d'une source : le filtre adapté.* Soit $r_w$ le résidu blanchi actuel et $g(z) = phi_w (z)$. Ajouter une source d'intensité $I$ en $z$ donne $chi^2(I) = norm(r_w - I g(z))^2$, une parabole en $I$ minimale en $I^star = ⟨r_w, g(z)⟩ \/ norm(g(z))^2$. La baisse obtenue vaut
$ Delta(z) = ⟨r_w, g(z)⟩^2 / norm(g(z))^2 = norm(r_w)^2 cos^2 (r_w, g(z)) . $
On place donc la nouvelle source là où *le résidu ressemble le plus à la signature d'une source* : c'est un filtre adapté (corrélation normalisée), calculé sur une grille de $D$ puis affiné. C'est la même étape que le Frank-Wolfe, mais sur les données blanchies.

*Puis :*
+ intensités par moindres carrés positifs, puis *glissement* : réajustement conjoint de toutes les positions et intensités, *sans pénalité*. Ce réajustement est indispensable : face à deux sources proches, la première source placée se met entre elles et ne glisse vers sa vraie place qu'après l'ajout de la seconde ;
+ calcul du critère $C(K)$ (ci-dessous) ; on recommence tant que $C$ baisse, et on s'arrête quand il n'a pas baissé depuis 2 ajouts ;
+ *étape arrière* : on essaie de retirer chaque source, et on la retire si $C$ baisse ;
+ on renvoie la solution de plus petit $C(K)$.

Exemple réel (3 sources, $sigma = 10^(-3)$, source centrale) : $C(K) = 735 thin 305$ ; 238 756 ; 997 ; *89,8* ; 97,7 ; 102,9 pour $K = 0, dots, 5$. Le critère chute jusqu'à $K = 3$ puis remonte. Durée : 0,1 à 0,5 s pour 1 à 5 sources.

== Le critère BIC : combien de sources ?

$chi^2(hat(theta)_K)$ baisse toujours quand $K$ augmente : une source de trop « explique » toujours un peu de bruit. Le BIC (Schwarz, 1978) choisit le $K$ le plus probable a posteriori. La vraisemblance marginale d'un modèle à $p = 3K$ paramètres s'approche par la méthode de Laplace :
$ p(m | K) = integral p(m | theta) pi(theta) dif theta approx p(m | hat(theta)_K) thin pi(hat(theta)_K) thin (2 pi)^(p\/2) thin abs(H)^(-1\/2), $
où $H$, la matrice d'information, grandit comme $M$ : $log abs(H) approx p log M + O(1)$. En passant à $-2 log$ :
$ C(K) = chi^2(hat(theta)_K) + 3K log M . $
*Intuition* : $abs(H)^(-1\/2)$ est le volume des paramètres compatibles avec les données. Plus un modèle a de paramètres, plus ce volume est petit comparé au prior : le modèle « paie » sa complexité. Une source est gardée si elle fait baisser $chi^2$ de plus de $3 log 64 approx 12","5$. *Aucun réglage* : le seuil ne dépend que de $M$.

*Une vraie source est gardée si son SNR dépasse environ 3.* Pour une source de rapport signal sur bruit $"SNR" = I norm(phi_w (z))$, la baisse $Delta$ suit approximativement un $chi^2_3$ décentré de paramètre $"SNR"^2$, de moyenne $"SNR"^2 + 3$. Elle passe le seuil si $"SNR"^2 + 3 > 12","5$, soit $"SNR" gt.tilde 3$. C'est vérifié : la probabilité de détection d'une source seule suit cette prédiction (25 % pour $I = 0","5$, 90 % pour $I = 2$, 100 % dès $I = 5$ ; `ml_detection.csv`).

*Une fausse source est presque toujours rejetée.* Sur du bruit seul, la méthode invente une source dans *2 %* des cas. L'argument usuel (« une fausse source fait baisser $chi^2$ de 3 en moyenne, loi $chi^2_3$ ») donnerait $P(chi^2_3 > 12","5) approx 0","6 %$. L'écart vient de ce que la position est *cherchée* : on prend le maximum de $Delta(z)$ sur tout $D$. Pour $z$ fixé, $Delta(z)$ suit un $chi^2_1$, et le maximum sur environ $N_"eff"$ positions indépendantes dépasse le seuil avec probabilité $N_"eff" thin P(chi^2_1 > 12","5) approx N_"eff" times 4 dot 10^(-4)$. Avec $N_"eff" approx 50$ (de l'ordre du nombre de directions mesurées), on retrouve les 2 % observés.

*Contrôle a posteriori.* Si le modèle explique les mesures, $chi^2$ au minimum suit une loi $chi^2$ à $M - 3K$ degrés de liberté : $chi^2\/M approx (M - 3K)\/M$ (0,86 pour 3 sources). Une valeur nettement plus grande signale une source manquée ou un bruit sous-estimé.

= Pourquoi ML + BIC atteint la borne de Cramér–Rao

== La borne

La matrice de Fisher d'un modèle gaussien de covariance $op("diag")(s_i^2)$ est
$ F = J^top J = D^top op("diag")(s_i^(-2)) D, $
et pour tout estimateur sans biais, $op("Cov")(hat(theta)) succ.eq F^(-1)$. La CRB d'une source est $"CRB"_k = sqrt((F^(-1))_(x_k x_k) + (F^(-1))_(y_k y_k))$ (`cramer_rao`). *C'est une limite physique : elle ne dépend que des capteurs, du bruit et des sources, pas de l'algorithme.* Elle permet de distinguer « la méthode est mauvaise » de « le problème est impossible ».

== Pourquoi le maximum de vraisemblance l'atteint

Reprenons la linéarisation de la section 3.2. Le minimum de $chi^2$ vérifie les équations normales @eq:normales, donc
$ hat(theta) - theta^star approx (J^top J)^(-1) J^top w_w = F^(-1) J^top w_w . $
Comme $w_w tilde cal(N)(0, I_M)$ :
- *espérance nulle* : pas de biais au premier ordre ;
- *covariance* $F^(-1) J^top J F^(-1) = F^(-1)$ : *exactement la borne*.

Autrement dit, pour un bruit gaussien et un modèle localement linéaire, le maximum de vraisemblance est l'estimateur linéaire sans biais de variance minimale (théorème de Gauss–Markov dans les coordonnées blanchies), et il est efficace. Cela tient tant que le bruit est petit devant la courbure du modèle, et à condition que le BIC ait trouvé le bon $K$ et que le glouton ait trouvé le bon bassin.

== Pourquoi le Frank-Wolfe ne l'atteint pas

Deux écarts, chacun visible dans la même linéarisation :

*1. La pénalité crée un biais.* Le glissement minimise $1/2 norm(m - f)^2 + lambda sum I_k$. Les équations normales deviennent $J_u^top J_u delta = J_u^top w - lambda e$, où $J_u$ est la jacobienne non blanchie et $e$ vaut 1 sur les coordonnées d'intensité, 0 sur les positions. D'où un biais
$ EE[hat(delta)] = -lambda (J_u^top J_u)^(-1) e, $
qui touche les intensités *et les positions* par les termes croisés de $(J_u^top J_u)^(-1)$. Le débiaisage final corrige les intensités à positions fixées, pas les positions. Et $lambda$ ne peut pas être petit, puisque c'est lui qui rejette les fausses sources.

*2. Pas de blanchiment.* Les moindres carrés non pondérés ont une covariance $(J_u^top J_u)^(-1) J_u^top Sigma J_u (J_u^top J_u)^(-1) succ.eq F^(-1)$ (Gauss–Markov), avec égalité seulement si le bruit est le même partout. Sur le cercle du projet, la source centrale donne presque le même bruit à tous les capteurs, et cet écart est faible. Avec des capteurs dans le domaine, il devient énorme (section 7.4).

== Ce que montrent les expériences

#figure(
  table(
    columns: 6, align: center, stroke: 0.4pt + gray,
    table.header[*sources*][*$sigma$*][*Prony*][*Frank-Wolfe*][*ML + BIC*][*CRB médiane*],
    [1], [$10^(-3)$], [0,059], [0,023], [*0,022*], [0,025],
    [1], [$10^(-2)$], [0,84], [0,23], [*0,23*], [0,25],
    [2], [$10^(-4)$], [0,014], [0,035], [*0,014*], [0,016],
    [2], [$10^(-3)$], [0,32], [0,29], [*0,13*], [0,16],
    [2], [$3 times 10^(-3)$], [1,6], [0,66], [*0,36*], [0,49],
    [3], [$10^(-4)$], [0,14], [0,72], [*0,15*], [0,22],
    [5], [$10^(-4)$], [0,88], [1,6], [*0,60*], [1,04],
  ),
  caption: [Erreur de position médiane comparée à la CRB (`projet_methodes_resume.csv`). La CRB des tableaux est celle de la source la moins bien déterminée du tirage, $max_k "CRB"_k$.],
)

- *ML + BIC est au niveau de la CRB* : son erreur médiane vaut 0,7 à 0,9 fois la CRB pour 1 et 2 sources. Le rapport est un peu inférieur à 1 parce qu'on compare une erreur *médiane* à la CRB de la *pire* source, qui est plus grande ; ce n'est pas un estimateur qui battrait la borne. Le test `test_cramer_rao_scaling_and_efficiency` vérifie l'efficacité pour une source.
- *Ses échecs sont des cas impossibles* : sur les tirages où la CRB permet de réussir, ML + BIC réussit dans *90 à 100 %* des cas, dans toutes les cases.
- *Le Frank-Wolfe est 2 à 5 fois au-dessus de la CRB* dès 2 sources (biais de $lambda$).
- Quand la CRB est très grande (3 sources, $sigma = 10^(-3)$ : CRB 2,2, erreur 0,42), le rapport n'a plus de sens : la pire source n'est pas déterminée par les données, et l'estimateur est alors biaisé.

*Conséquence pratique : on ne gagnera presque rien à changer d'algorithme.* Pour aller plus loin, il faut changer les données.

#attention[
*La source centrale est le premier frein.* Pour 3 sources à $sigma = 10^(-3)$, ML + BIC réussit dans *100 %* des cas sans source centrale et *40 %* avec ; la CRB médiane passe de 0,052 à 2,2, un facteur 42 (`projet_centrale_resume.csv`). Retirer la source centrale enlève son signal, pas son bruit.
]

= Beaucoup de sources : comparaison des méthodes

Sources tirées comme dans `config/` : positions uniformes dans $D$, intensités uniformes sur le simplexe de somme 100.

#figure(
  table(
    columns: 6, align: center, stroke: 0.4pt + gray,
    table.header[*cas*][*méthode*][*sources estimées*][*retrouvées à 0,5 près*][*à 0,1 près*][*temps*],
    table.cell(rowspan: 3)[50 sources, \ cercle (64 capteurs)], [Frank-Wolfe (les deux réglages)], [6], [3], [0], [< 1 s],
    [ML à $K = 50$ imposé ($sigma = 10^(-4)$, un tirage)], [49], [6 (ML + BIC sur ce tirage : 9)], [—], [plusieurs min],
    [*ML + BIC*], [7], [4], [—], [1,5 s],
    table.cell(rowspan: 3)[20 sources, \ cercle + grille 8×16 \ (192 capteurs)], [Frank-Wolfe, réglage du projet], [12], [7], [1], [1 s],
    [Frank-Wolfe blanchi], [19], [13], [9], [4 s],
    [*ML + BIC*], [18], [*17*], [*14*], [42 s],
    table.cell(rowspan: 3)[50 sources, \ cercle + grille 16×32 \ (576 capteurs)], [Frank-Wolfe, réglage du projet], [16], [14], [1], [1 s],
    [Frank-Wolfe blanchi], [44], [33], [26], [51 s],
    [*ML + BIC*], [46], [*45*], [*40*], [23 min],
  ),
  caption: [$sigma = 10^(-3)$ sauf mention contraire, source centrale, $K$ inconnu (sauf « imposé »). Grilles : un tirage (graine 0) ; cercle seul : médiane sur 5 tirages. `ml_fw_beaucoup.csv`, `ml_beaucoup_ml.csv`, `COMPTE_RENDU_CAPTEURS.md`. Les mesures permettent de localiser à 0,1 près 13 sources sur 20 (grille 8×16) et 42 sur 50 (grille 16×32), d'après la CRB.],
)

- *Avec le cercle, toutes les méthodes échouent* : 3 ou 4 sources sur 50. ML + BIC ne s'effondre pas : il garde les quelques sources que les données permettent de voir (les plus intenses et les plus isolées) et n'invente pas les autres. Imposer $K = 50$ fait pire.
- *Avec des capteurs dans le domaine, ML + BIC retrouve 45 sources sur 50* (40 à 0,1 près, pour 42 que la CRB déclare localisables). Il atteint donc ce que les données permettent.
- *Le Frank-Wolfe ne marche qu'à condition de blanchir* : sans blanchiment, les capteurs proches de la source centrale, très bruités, imposent un $lambda$ trop grand qui rejette toutes les sources faibles (16 sources estimées sur 50). Même blanchi, il reste moins précis (26 sources à 0,1 près contre 40) : c'est le biais de la pénalité. Son avantage est la vitesse (51 s contre 23 min).

= La solution : des capteurs à l'intérieur du domaine

== D'où vient le plafond de directions : le noyau de Poisson

Plaçons un capteur sur le cercle, $x = c + R e^(i phi)$, et une source en $z = c + rho e^(i psi)$ avec $rho < R$. Alors $norm(x - z)^2 = R^2 - 2 R rho cos(phi - psi) + rho^2$, et l'on reconnaît le *noyau de Poisson* :
$ 1 / norm(x - z)^2 = 1 / (R^2 - rho^2) [ 1 + 2 sum_(n >= 1) (rho / R)^n cos n (phi - psi) ] . $ <eq:poisson>
Le cercle de capteurs voit donc les *harmoniques circulaires* de la distribution des sources, et *l'harmonique $n$ est atténuée par $(rho\/R)^n$*. (La section 8 démontre ce développement et en tire un ordre de grandeur.) Nous l'avons vérifié numériquement : pour une source en $(7, 15)$, le rapport entre deux coefficients de Fourier successifs du signal sur les 64 capteurs vaut exactement $rho\/R = 0","4817$.

Conséquences :
- $n = 0$ donne l'intensité totale, $n = 1$ le barycentre, $n = 2$ l'étalement : ce sont les « grandes structures » que les capteurs voient bien.
- Le nombre d'harmoniques au-dessus du bruit est $n_"max" approx log("SNR") \/ log(R\/rho)$, et chaque harmonique apporte 2 nombres (cosinus, sinus) : environ $2 n_"max" + 1$ directions mesurables. Avec $rho\/R approx 0","5$ et un seuil de $10^(-6)$, $n_"max" approx 20$, soit environ 41 directions : c'est l'ordre de grandeur du plateau observé (46 à 56). Le bruit réduit encore ce nombre (une douzaine de directions à 0,1 près).
- *Ajouter des capteurs sur le cercle n'aide pas* : 64 capteurs échantillonnent déjà les harmoniques jusqu'à $n = 32$, qui sont de toute façon écrasées. 16 fois plus de capteurs moyennent le bruit (CRB divisée par environ 4) sans révéler de nouvelles harmoniques.
- *Agrandir le cercle fait pire* : $rho\/R$ diminue, toutes les harmoniques s'écrasent plus vite (31 directions sur 150, CRB de 3 sources multipliée par 60).
- *Rapprocher les capteurs des sources* fait tendre $rho\/R$ vers 1 : c'est le seul moyen de faire apparaître les détails fins. À l'intérieur du domaine, le développement @eq:poisson ne s'applique plus : un capteur proche d'une source la voit séparément de ses voisines.

== Ce que mesurent des capteurs dans le domaine

On garde le cercle et on ajoute une grille de capteurs dans $D$ (un capteur au centre de chaque case ; aucun ne tombe sur la source centrale).

#figure(
  table(
    columns: 5, align: center, stroke: 0.4pt + gray,
    table.header[*disposition*][*K = 3*][*K = 10*][*K = 20*][*K = 50*],
    [cercle du projet, 64], [40 %], [2 %], [1 %], [0 %],
    [cercle, 1024 capteurs], [53 %], [18 %], [5 %], [1 %],
    [cercle + grille 4×8, 96], [93 %], [24 %], [4 %], [0 %],
    [cercle + grille 8×16, 192], [100 %], [74 %], [61 %], [3 %],
    [cercle + grille 16×32, 576], [100 %], [98 %], [99 %], [*84 %*],
  ),
  caption: [Part des sources dont la CRB est inférieure à 0,1 ($sigma = 10^(-3)$, source centrale, moyenne sur 5 tirages).],
)


*À retenir :*
- *C'est la proximité qui compte, pas le nombre de capteurs* : la grille 4×8 (96 capteurs) fait mieux que 1024 capteurs sur le cercle.
- *Il faut environ un capteur par source, à une distance comparable à l'écart entre sources* : la grille 8×16 (pas de 1,25) suffit jusqu'à 20 sources ; pour 50 sources, il faut la grille 16×32 (pas de 0,625). Avec elle, la CRB médiane vaut 0,02, soit 50 fois moins que l'écart moyen entre sources : *le problème à 50 sources devient bien posé*.
- *ML + BIC atteint alors ce que la CRB prévoit* : 40 sources retrouvées à 0,1 près pour 42 déclarées localisables (50 sources) ; 14 pour 13 (20 sources). Les sources manquées sont les plus faibles (les intensités tirées vont de 0,02 à 7,7) ou collées à une voisine.

== Un seul capteur mobile peut remplacer la grille

Si l'on ne peut pas poser des centaines de capteurs, un seul capteur qui se déplace dans $D$ en enregistrant suffit (`mobile.py`, `ml_continu_crb.csv`, `ml_continu_ml.csv`, 50 sources, $sigma = 10^(-3)$) :

#figure(
  table(
    columns: 5, align: center, stroke: 0.4pt + gray,
    table.header[*en plus du cercle*][*mesures*][*directions (sur 150)*][*CRB médiane*][*ML + BIC : retrouvées à 0,5 / 0,1 près*],
    [rien], [64], [56], [14 000], [4 / —],
    [serpentin 4 lignes, une mesure tous les 0,25], [384], [150], [0,10], [28 / 17],
    [serpentin 8 lignes, une mesure tous les 0,25], [704], [150], [*0,012*], [*42 / 35*],
    [grille fixe 16×32], [576], [150], [0,021], [45 / 40],
  ),
  caption: [Un capteur mobile en serpentin (aller-retour le long de lignes parallèles au grand côté). Reconstruction ML + BIC : un tirage, $K$ inconnu.],
)

Un capteur qui parcourt 8 lignes vaut presque la grille de 512 capteurs. Si l'on peut choisir les positions une à une, la *stratégie adaptative* (placer chaque mesure là où elle fait le plus baisser la CRB des sources déjà estimées) fait encore mieux avec peu de mesures : pour 20 sources, 128 mesures adaptatives retrouvent 17,4 sources sur 20 à 0,1 près, contre 11 à 11,6 en grille ou au hasard (`ml_mobile.csv`, 5 tirages). Le capteur va se placer tout contre les sources (à 0,2–0,4 d'une vraie source en médiane).

== Le blanchiment devient indispensable

Avec des capteurs dans le domaine, le bruit n'est plus du tout uniforme : les capteurs proches du centre reçoivent l'essentiel du bruit de la source centrale. C'est pourquoi le Frank-Wolfe non blanchi s'effondre sur la grille (16 sources estimées, 1 seule à 0,1 près) alors que sa version blanchie en trouve 44. ML + BIC blanchit par construction. La source centrale reste le premier facteur de bruit : avec la grille 16×32, la CRB médiane vaut 0,021 avec elle et 0,0008 sans, soit 26 fois moins.

= Le noyau de Poisson : ce que voit un cercle de capteurs

La section 7.1 a donné le résultat. On le démontre ici, puis on en tire un ordre de grandeur : combien de sources un cercle de capteurs peut séparer.

== Le calcul

On identifie le plan à $CC$. Un capteur est en $x = c + R e^(i phi)$ et une source en $z = c + rho e^(i psi)$, avec $rho < R$ (la source est dans le disque). On pose
$ q = rho / R thin e^(i (psi - phi)), quad abs(q) = r = rho / R < 1 . $
Alors $x - z = R e^(i phi) (1 - q)$, donc $norm(x - z)^2 = R^2 abs(1 - q)^2$. Pour $abs(q) < 1$, la série géométrique donne
$ (1 + q) / (1 - q) = (1 + q) sum_(n >= 0) q^n = 1 + 2 sum_(n >= 1) q^n . $
On prend la partie réelle des deux côtés. À gauche, $(1 + q)(1 - overline(q)) = 1 - abs(q)^2 + (q - overline(q))$ et $q - overline(q)$ est imaginaire pur, donc $op("Re") (1+q)/(1-q) = (1 - r^2) \/ abs(1 - q)^2$. À droite, $op("Re") q^n = r^n cos n(psi - phi)$. D'où
$ 1 / abs(1 - q)^2 = (1 + 2 sum_(n >= 1) r^n cos n (phi - psi)) / (1 - r^2), $
et, en divisant par $R^2$, exactement le développement @eq:poisson. C'est le noyau de Poisson du disque : la fonction qui donne, sur le bord, la « trace » d'un point intérieur.

== Les capteurs mesurent des moments

On somme sur les sources. Avec le poids $w_k = I_k \/ (R^2 - rho_k^2)$ et la position complexe réduite $zeta_k = (z_k - c) \/ R$ ($abs(zeta_k) < 1$), les coefficients de Fourier du signal sur le cercle valent
$ mu_n = 1/M sum_(i=1)^M y_i thin e^(i n phi_i) = sum_(k=1)^K w_k thin zeta_k^n, quad n = 0, 1, dots, M\/2 . $ <eq:moments>
*Les 64 mesures sont donc équivalentes (par transformée de Fourier discrète) aux moments complexes $mu_0, dots, mu_32$ de la distribution des sources dans le disque.* Nous l'avons vérifié numériquement sur un tirage de 3 sources : @eq:moments est exacte à $10^(-4)$ près (l'écart vient du repliement des harmoniques d'ordre $>= M$).

- $mu_0 = sum w_k$ : l'intensité totale (pondérée) ;
- $mu_1 \/ mu_0$ : le barycentre ;
- $mu_2$ : l'étalement et son orientation ; et ainsi de suite.

C'est exactement le problème de Prony : retrouver $K$ points et $K$ poids à partir de leurs moments. La méthode de Prony du projet inverse directement ces moments, ce qui explique sa fragilité au bruit (section 3.4).

== Combien de moments faut-il, combien en mesure-t-on ?

*Ce qu'il faut.* $K$ sources font $3K$ inconnues réelles. $mu_0$ est réel et chaque $mu_n$ ($n >= 1$) apporte 2 réels : il faut au moins les moments jusqu'à l'ordre
$ N >= (3K - 1) / 2 : quad N = 4 "pour" K = 3, quad N = 15 "pour" K = 10, quad N = 75 "pour" K = 50 . $

*Première limite : l'échantillonnage.* 64 capteurs ne distinguent les harmoniques que jusqu'à l'ordre 32 ; au-delà, elles se replient sur les premières. Pour 50 sources, il faudrait l'ordre 75 : *c'est impossible même sans bruit*, ce que traduit $op("rang") J <= 64 < 150$.

*Deuxième limite : le bruit.* $abs(mu_n) <= sum_k w_k abs(zeta_k)^n$ décroît géométriquement avec $n$, alors que le bruit sur chaque $mu_n$ (transformée de Fourier d'un bruit blanc) vaut environ $overline(s) \/ sqrt(M)$, avec $overline(s) approx 47 sigma$ (source centrale comprise). Le moment d'ordre $n$ est donc utilisable tant que $W r^n > overline(s) \/ sqrt(M)$, avec $W = sum_k w_k$ :
$ n_"max" approx ln(W sqrt(M) \/ overline(s)) / ln(1 \/ r) . $

*Ordre de grandeur pour le projet* ($sigma = 10^(-3)$, 64 capteurs) : $overline(s) \/ sqrt(M) approx 0","047 \/ 8 approx 0","006$, $W approx 100 \/ (R^2 - rho^2) approx 1$ et $r approx 0","5$ en moyenne. On trouve $n_"max" approx ln(170) \/ ln 2 approx 7$, soit $1 + 2 times 7 = 15$ nombres réels mesurés. C'est l'ordre de grandeur des 12 directions connues à 0,1 près (tableau de la section 3.3) : *le développement de Poisson explique le chiffre de la SVD.*

- *3 sources* : il faut $N = 4$, et $W r^4 approx 0","06$, soit 10 fois le bruit. C'est faisable mais juste, d'où 40 % de réussite.
- *50 sources* : il faudrait $N = 75$ ; on en mesure 7. Une source isolée d'intensité 2 ($w approx 0","02$) ne dépasse le bruit que jusqu'à l'ordre 1 ou 2 : elle contribue à l'intensité totale et au barycentre, mais elle ne peut pas être séparée de ses voisines.
- Les sources proches du cercle (près des coins de $D$, $r -> 1$) sont mieux vues : leurs moments décroissent lentement.

== Conséquences pour placer les capteurs

- *Plus de capteurs sur le même cercle* : le bruit sur chaque moment baisse comme $1 \/ sqrt(M)$, donc $n_"max"$ ne gagne que $ln(M'\/M) \/ (2 ln(1\/r))$. Avec 16 fois plus de capteurs, cela fait *2 moments de plus* : d'où le gain lent observé (40 % à 70 % de réussite pour 3 sources avec 1024 capteurs).
- *Un cercle plus grand* : $r = rho \/ R$ diminue, chaque ordre perd davantage, donc moins de moments utilisables (31 directions sur 150).
- *Des capteurs dans le domaine* : le développement ne s'applique plus. Un capteur à distance $d$ d'une source la voit directement, en $I \/ d^2$, indépendamment des sources lointaines. L'information devient *locale* au lieu de passer par des moments d'ordre élevé, et l'espacement des capteurs fixe la résolution : il doit être de l'ordre de la distance entre sources (section 7.2).

#attention[
*En une phrase :* un cercle de capteurs ne mesure que les premiers moments $sum_k w_k zeta_k^n$ de la distribution des sources, atténués d'un facteur $(rho\/R)^n$ ; avec le bruit du projet, environ 7 moments, soit une quinzaine de nombres, alors qu'il en faut $3K$. Seuls des capteurs proches des sources lèvent cette limite.
]

= Conclusion

#figure(
  table(
    columns: 2, align: (left, left), stroke: 0.4pt + gray,
    table.header[*situation (2D, source centrale du projet)*][*ce que ML + BIC retrouve*],
    [cercle, $sigma = 10^(-4)$], [1 à 3 sources de façon fiable, 5 sources une fois sur deux],
    [cercle, $sigma = 10^(-3)$], [1 à 2 sources, 3 sources dans 40 % des cas],
    [cercle, 20 à 50 sources], [seulement les quelques sources les plus intenses et isolées (aucune méthode ne fait mieux)],
    [cercle + grille 16×32 dans $D$, 50 sources], [45 sources sur 50, dont 40 à 0,1 près],
    [cercle + un capteur mobile, 8 lignes, 50 sources], [42 sources sur 50, dont 35 à 0,1 près],
  ),
)

+ *Le Frank-Wolfe échoue pour des raisons qui lui sont propres* (pénalité qui biaise, condition de Duval–Peyré fausse dès 3 sources, pas de blanchiment) *et pour une raison commune à toutes les méthodes* : le cercle ne mesure pas assez de directions. Minimiser un critère revient à résoudre les équations normales $J^top J delta = J^top w$ ; dans une direction non mesurée le critère est plat, et le minimum est n'importe où.
+ *ML + BIC est la méthode à utiliser* : meilleure dans tous les tests, sans paramètre à régler, et au niveau de la borne de Cramér–Rao. Ses échecs sont ceux des données.
+ *Le levier est la géométrie des capteurs.* Le cercle voit un noyau de Poisson qui écrase les détails d'un facteur $(rho\/R)^n$ ; des capteurs dans le domaine, espacés comme les sources, rendent le problème bien posé même pour 50 sources. Un capteur mobile suffit.
+ *La CRB permet d'évaluer une disposition de capteurs en quelques secondes*, avant toute reconstruction : c'est l'outil pour décider d'une expérience.

*Questions ouvertes pour la suite :*
- Est-il physiquement possible de placer des capteurs dans le domaine, ou un capteur mobile ? Sinon, à quelle hauteur au-dessus du plan ? (Plus les capteurs sont hauts, plus le signal est lisse, comme pour un cercle agrandi : option `--hauteur` de `tester_ml.py`.)
- Peut-on réduire le bruit de la source centrale (mesure différentielle, capteurs blindés vers le centre, source centrale moins intense) ? C'est le deuxième levier, après la géométrie.
- Le coût de ML + BIC grandit vite avec $K$ (23 min pour 50 sources, 576 capteurs) : un départ depuis la solution du Frank-Wolfe blanchi (51 s) pourrait l'accélérer.

#v(0.8em)
#text(size: 9pt)[
*Références.*
G. Schwarz, _Estimating the dimension of a model_, Annals of Statistics, 1978. —
S. M. Kay, _Fundamentals of Statistical Signal Processing_, vol. I et II, Prentice Hall, 1993 et 1998. —
V. Duval, G. Peyré, _Exact support recovery for sparse spikes deconvolution_, Foundations of Computational Mathematics, 2015. —
Q. Denoyelle, V. Duval, G. Peyré, E. Soubies, _The sliding Frank-Wolfe algorithm and its application to super-resolution microscopy_, Inverse Problems, 2019.

*Reproduire.* `pytest sparse_sources/tests` ; `python sparse_sources/experiments_projet.py` (comparaison des méthodes) ; `python sparse_sources/experiments_ml.py` (détection, beaucoup de sources, géométrie, capteur mobile, Frank-Wolfe) ; `python sparse_sources/figure_spectre_directions.py` (figure du spectre).
]

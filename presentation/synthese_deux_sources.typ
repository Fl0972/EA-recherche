// Synthèse : cas test à deux sources, descente de gradient et rapport d'intensité
// Figures : python presentation/figures_two_sources.py
// Compiler : typst compile presentation/synthese_deux_sources.typ

#set document(title: "Cas test à deux sources : descente de gradient et rapport d'intensité",
  author: "Florian Guillaumey")
#set page(paper: "a4", margin: (x: 2.2cm, y: 1.5cm), numbering: "1")
#set text(lang: "fr", size: 11pt)
#set par(justify: true)
#set enum(indent: 0.5em)
#set list(indent: 0.5em)
#set math.equation(numbering: none)
#show heading.where(level: 2): set block(above: 1.6em, below: 0.8em)

#let encadre(body) = block(fill: rgb("#EEF3F9"), stroke: (left: 3pt + rgb("#1F3A5F")),
  inset: 10pt, radius: 2pt, width: 100%, body)
#let prop(titre, body) = block(stroke: 0.6pt + rgb("#9AA5B1"), inset: 10pt, radius: 3pt,
  width: 100%, breakable: false)[*#titre.* #body]

= Cas test à deux sources : descente de gradient et rapport d'intensité

Novembre 2026 · Florian Guillaumey

L'équipe industrielle a proposé un cas test simple et parlant : deux sources, l'une $x$ fois plus intense que l'autre, retrouvées par descente de gradient. Leur constat : la méthode fonctionne pour $x = 10$, plus pour $x = 100$. Nous avons reproduit ce cas test sur notre simulateur pour disposer d'un point de comparaison commun. Le constat se retrouve à l'identique, et l'analyse montre qu'il tient à une propriété mathématique du problème, son conditionnement, et non à la mise en œuvre.

#encadre[*Cadre de l'étude : mesures sans bruit.* Les sections 1 à 4 utilisent des mesures exactes ; la section 5 donne un bref aperçu avec bruit. Les difficultés observées viennent donc uniquement de l'optimisation, pas d'un manque d'information dans les mesures. Pas de source centrale non plus, comme dans le cas test d'origine.]

#figure(
  image("figures_deux_sources/deux_sources_schema.png", width: 40%),
  caption: [Le cas test : 64 capteurs sur le cercle circonscrit au domaine $10 times 20$ (pointillés), une source forte d'intensité $x I$ en $(3, 6)$, une source faible d'intensité $I$ en $(6,5 ; 15)$, intensité totale fixée. Les croix marquent l'initialisation commune à toutes les méthodes.],
)

== 1. Le problème d'optimisation

*Modèle direct.* Un capteur en $s$ mesure la somme des intensités divisées par le carré des distances. Avec $theta = (p_1, p_2, I_1, I_2)$ les positions et intensités inconnues :
$ Phi_theta (s) = sum_(i=1)^2 I_i / norm(s - p_i)^2 . $

*Fonction de perte.* On cherche les sources qui reproduisent les $K$ mesures $y_k$ au sens des moindres carrés (normalisés par l'énergie du signal) :
$ L(theta) = 1/2 (sum_(k=1)^K (Phi_theta (s_k) - y_k)^2) / norm(y)^2 . $
Sans bruit, les vraies sources $theta^*$ donnent $L(theta^*) = 0$ : c'est le minimum global. La perte n'est pas convexe en les positions.

*Gradient.* Avec le résidu $r_k = Phi_theta (s_k) - y_k$ et la jacobienne $J_(k j) = partial Phi_theta (s_k) \/ partial theta_j$ :
$ nabla L(theta) = (J^T r) / norm(y)^2, quad (partial Phi_theta (s_k)) / (partial I_i) = 1 / norm(s_k - p_i)^2, quad (partial Phi_theta (s_k)) / (partial p_i) = I_i dot (2 (s_k - p_i)) / norm(s_k - p_i)^4 . $
Le facteur $I_i$ dans la dérivée par rapport à la position est le point clé de la suite.

*Descente de gradient projetée.* Les positions sont ramenées à $[0, 1]$ dans le domaine et les intensités divisées par l'intensité totale, pour que les paramètres aient des échelles comparables. Puis
$ theta_(n+1) = P_C (theta_n - eta_n nabla L(theta_n)), quad C = D^2 times RR_+^2 , $
où $P_C$ projette sur les contraintes (sources dans le domaine $D$, intensités positives). Le pas $eta_n$ est choisi par la règle d'Armijo : on part du double du pas précédent et on le divise par deux jusqu'à une diminution suffisante,
$ L(theta_(n+1)) <= L(theta_n) - c thin nabla L(theta_n) dot (theta_n - theta_(n+1)), quad c = 10^(-4) , $
avec un déplacement limité à 5 % du domaine par itération. Budget : 20 000 itérations.

== 2. Reproduction du cas test

Le constat de l'équipe industrielle se retrouve : pour $x = 10$, la descente retrouve les deux sources (erreur de position $< 10^(-7)$) ; pour $x = 100$, la source forte est retrouvée à 0,4 près, mais la source faible reste collée à la forte, à 8,8 de sa vraie position.

#figure(
  image("figures_deux_sources/deux_sources_trajectoires.png", width: 82%),
  caption: [Trajectoires de la descente de gradient sur la carte de l'intensité, depuis la même initialisation (croix). Blanc : source forte ; orange : source faible ; étoiles : vraies sources ; cercles : fin de la descente.],
)

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt + rgb("#B0B7C0"),
    table.header([*Rapport $x$*], [1], [3], [10], [30], [100]),
    [Itérations pour converger], [1 798], [2 702], [11 059], [92 475#super[\*]], [$> 400 000$#super[\*]],
    [Réussite, 20 situations aléatoires], [95 %], [85 %], [55 %], [20 %], [5 %],
  ),
  caption: [Descente de gradient. Ligne 1 : cas fixe ci-dessus (\* : budget prolongé au-delà de 20 000 itérations pour cette mesure). Ligne 2 : positions et initialisation tirées au hasard ; réussite si chaque source est retrouvée à moins de 0,1.],
)

À budget prolongé, la descente finit par converger pour $x = 30$ ; pour $x = 100$, la source faible est encore à 4,3 de sa position après 400 000 itérations. L'échec à $x = 100$ est donc une lenteur extrême plutôt qu'un blocage définitif, ce qu'explique la section suivante.

== 3. Analyse : pourquoi la descente ralentit

Près de la solution, la perte se comporte comme une forme quadratique. Comme le résidu y est nul, sa hessienne se réduit au terme de Gauss-Newton :
$ L(theta) approx 1/2 (theta - theta^*)^T H (theta - theta^*), quad H = (J^T J) / norm(y)^2 . $

#prop[Proposition 1 (vitesse de la descente)][
Sur une forme quadratique de valeurs propres $lambda_min <= dots <= lambda_max$, la descente de gradient à pas optimal vérifie
$ norm(theta_n - theta^*) <= ((kappa - 1) / (kappa + 1))^n norm(theta_0 - theta^*), quad kappa = lambda_max / lambda_min . $
Il faut donc de l'ordre de $kappa / 2 dot ln(1 \/ epsilon)$ itérations pour gagner un facteur $epsilon$ (résultat classique, admis). La direction la plus plate, de courbure $lambda_min$, impose son rythme à toute la descente.
]

#prop[Proposition 2 (le conditionnement croît comme $x^2$)][
Avec $I_1 = x I_2$, il existe une constante $c > 0$ ne dépendant que de la géométrie (capteurs et positions des sources) telle que $kappa >= c x^2$.

_Idée de la preuve._ D'après le gradient ci-dessus, les colonnes de $J$ associées à la position $p_i$ s'écrivent $I_i G_i$, où $G_i$ ne dépend que de la géométrie. Pour un vecteur unitaire $e$ portant uniquement sur $p_2$ (source faible), le quotient de Rayleigh donne $lambda_min <= e^T H e = I_2^2 norm(G_2 e)^2 \/ norm(y)^2$ ; de même, en portant sur $p_1$, $lambda_max >= I_1^2 norm(G_1 e')^2 \/ norm(y)^2$. Le rapport fait apparaître $(I_1 \/ I_2)^2 = x^2$, le reste ne dépendant que de la géométrie.
]

Autrement dit, la perte est presque plate dans les directions qui déplacent la source faible : la déplacer ne modifie les mesures que d'un facteur $1 \/ x$, donc la perte d'un facteur $1 \/ x^2$. Le gradient ne la guide presque plus.

#figure(
  image("figures_deux_sources/deux_sources_paysage.png", width: 55%),
  caption: [Perte en fonction de la position supposée de la source faible, la source forte étant fixée à sa vraie place (étoile blanche). Pour $x = 100$, la vallée autour de la vraie position (étoile orange) est environ 100 fois moins profonde : partout dans le domaine, la perte reste faible.],
)

*Vérification numérique.* Le conditionnement mesuré suit la pente $x^2$ dès $x >= 10$ (figure ci-dessous) : $1,1 dot 10^3$ pour $x = 10$, $7,7 dot 10^3$ pour $x = 30$, $8,3 dot 10^4$ pour $x = 100$. D'après la proposition 1, le nombre d'itérations doit suivre :
- de $x = 10$ à $x = 30$ : $kappa$ multiplié par 7, itérations multipliées par 8,4 (11 059 puis 92 475) ;
- de $x = 10$ à $x = 100$ : $kappa$ multiplié par 76, soit environ 800 000 itérations attendues, cohérent avec la non-convergence observée à 400 000.

#figure(
  image("figures_deux_sources/deux_sources_reussite.png", width: 82%),
  caption: [À gauche : taux de réussite sur 20 situations aléatoires par rapport $x$ (mêmes situations et mêmes initialisations pour toutes les méthodes). À droite : conditionnement $kappa$ de la perte à la solution, cas fixe ; pointillés : pente $x^2$.],
)

== 4. Comparaison avec des méthodes du second ordre

*Gauss-Newton.* Même perte, mais chaque pas résout un problème linéarisé :
$ theta_(n+1) = theta_n - (J^T J)^(-1) J^T r . $
#prop[Proposition 3 (invariance par changement d'échelle)][
Si l'on reparamètre $theta = M tilde(theta)$ avec $M$ inversible, les itérés de Gauss-Newton sont les mêmes. En effet $tilde(J) = J M$ et le pas devient $-M^(-1) (J^T J)^(-1) J^T r$, qui redonne le même pas en $theta$. En prenant $M$ qui multiplie la position de chaque source par $1 \/ I_i$, on fait disparaître le facteur $I_i$ des colonnes de $J$ : le facteur $x^2$ de la proposition 2 n'affecte pas Gauss-Newton.
]
La descente de gradient, elle, n'a pas cette invariance : son pas se transforme par $M M^T$. Résultat, depuis la même initialisation, Gauss-Newton retrouve les deux sources pour $x = 100$ en 59 ms, et dans 90 % des situations aléatoires.

*Prony + maximum de vraisemblance.* Notre méthode précédente obtient un point de départ par une formule directe (méthode de Prony), sans initialisation, puis l'affine par Gauss-Newton. Elle retrouve les deux sources dans 100 % des situations testées, jusqu'à $x = 300$, en 10 à 25 ms.

*Limite commune.* La perte n'étant pas convexe, toute méthode locale peut s'arrêter sur un point stationnaire, typiquement les deux sources fusionnées en une seule. C'est ce qui arrive à Gauss-Newton pour $x = 300$ dans le cas fixe, et dans 25 % des situations aléatoires.

#pagebreak()
== 5. Aperçu avec bruit

Même cas test, avec un bruit gaussien relatif $y_k (1 + sigma epsilon_k)$, $epsilon_k tilde cal(N)(0, 1)$, pour $x$ autour de 100 (50 tirages de bruit par point).

- La descente de gradient reste bloquée comme sans bruit : à $x = 100$, la source faible finit à environ 8,8 de sa position, quel que soit $sigma$.
- Gauss-Newton et Prony + maximum de vraisemblance atteignent la borne de Cramér-Rao : la limite devient l'information contenue dans les mesures.
- Cette borne, pour la source faible, vaut environ 0,77 $x sigma$ dans cette géométrie (contre 1,8 $sigma$ pour la source forte). À $x = 100$, la source faible est localisée à 0,1 près dans 84 % des cas pour $sigma$ = 0,1 %, et 2 % pour $sigma$ = 1 %.

#figure(
  image("figures_deux_sources/deux_sources_bruit.png", width: 62%),
  caption: [Erreur de position médiane sur la source faible (Prony + maximum de vraisemblance) selon le bruit, pour $x = 50, 100, 200$. Pointillés : borne de Cramér-Rao ; croix : descente de gradient à $x = 100$.],
)

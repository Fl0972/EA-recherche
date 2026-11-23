// Fiche synthèse des méthodes de reconstruction
// Compiler : typst compile presentation/fiche_synthese_methodes.typ

#set document(title: "Fiche synthèse des méthodes de reconstruction", author: "Florian Guillaumey")
#set page(paper: "a4", margin: (x: 2.2cm, y: 1.7cm), numbering: "1")
#set text(lang: "fr", size: 11pt)
#set par(justify: true)
#set enum(indent: 0.5em)
#set list(indent: 0.5em)
#show heading.where(level: 2): set block(above: 1.6em, below: 0.8em)

= Fiche synthèse des méthodes de reconstruction

Novembre 2026 · Florian Guillaumey

Un outil de simulation, puis deux méthodes pour retrouver des sources ponctuelles à partir de capteurs placés sur un cercle : une formule exacte (Prony), puis son affinage statistique (maximum de vraisemblance).

== Outil de simulation

Un simulateur Python génère des situations aléatoires, simule les mesures des capteurs et teste les méthodes de reconstruction dessus. Tout se règle par des fichiers de configuration, et chaque résultat se reproduit à l'identique à partir d'une graine.

- *Génération* : nombre de sources $N$ selon une loi de Poisson, positions uniformes dans le domaine, intensités de somme fixée, source centrale fixe, bruit relatif gaussien ou de comptage de Poisson.
- *Reconstruction* : chaque méthode reçoit les mesures et renvoie les sources estimées, comparées ensuite aux vraies.
- *Banc d'essai* : compare les méthodes en faisant varier un paramètre (bruit, nombre de capteurs, nombre moyen de sources), avec la borne de Cramér-Rao comme référence.
- *Jeux de données* pour entraîner des modèles d'apprentissage : 100 000 situations en 14 s.

Le modèle direct accepte des capteurs placés n'importe où ; seule la disposition en cercle est codée pour l'instant.

#figure(
  image("figures/situation.png", width: 82%),
  caption: [Une situation simulée : 4 sources (points rouges), la source centrale (étoile), 64 capteurs sur le cercle (orange). À droite, les mesures des capteurs selon leur angle.],
)

== 1. Méthode de Prony

*Idée.* Vues en fonction de l'angle du capteur, les mesures forment une série de Fourier dont les coefficients sont des puissances des positions des sources : on lit les sources directement dans ces coefficients.

*Esquisse de la preuve* (calculs sautés).

+ En complexe, un capteur est en $R e^(i t)$, une source en $z = rho e^(i theta)$, avec $rho < R$.
+ Le noyau $1 \/ r^2$ sur un cercle est le noyau de Poisson ; il se développe en série de Fourier (admis) :
  $ 1 / abs(R e^(i t) - z)^2 = 1 / (R^2 - rho^2) sum_(n in ZZ) (rho / R)^abs(n) e^(i n (t - theta)) $
+ En sommant sur les sources, les coefficients de Fourier des mesures valent :
  $ c_n = sum_(i=1)^N w_i u_i^n, quad u_i = overline(z_i) / R, quad w_i = I_i / (R^2 - rho_i^2) $
+ Une somme de $N$ exponentielles vérifie une récurrence linéaire d'ordre $N$, dont le polynôme caractéristique a pour racines les $u_i$. Conséquence : la matrice de Hankel $[c_(j+k)]$ est de rang $N$, et ses valeurs propres généralisées donnent les $u_i$ (donc les positions).
+ Les $w_i$ (donc les intensités) s'obtiennent ensuite par un système linéaire.

*Résultat.* Sans bruit, reconstruction exacte du nombre de sources, des positions et des intensités, dès $K >= 4N - 1$ capteurs équirépartis. Calcul direct, environ 1 ms.

#figure(
  image("figures/prony_bruit.png", width: 72%),
  caption: [Prony sur une même situation simulée : exact sans bruit (à gauche), une source perdue avec 0,1 % de bruit (à droite). Points rouges : vraies sources ; carrés : sources retrouvées.],
)

*Limites.*
- Capteurs sur un cercle, équirépartis, et sources ponctuelles : la preuve repose sur ces deux points.
- Très sensible au bruit : l'information sur une source décroît comme $(rho \/ R)^n$ avec $n$, et disparaît vite sous le bruit. Dès 0,1 % de bruit, une source faible proche du centre est perdue.

#figure(
  image("figures/fourier_decroissance.png", width: 54%),
  caption: [Contribution de chaque source aux coefficients $c_n$ : elle décroît comme $(rho \/ R)^n$. Les lignes noires marquent le niveau du bruit ; au-dessous, l'information est perdue. Les sources proches du centre ($rho \/ R = 0,42$) ne sont visibles que dans les premiers coefficients.],
)

== 2. Prony + maximum de vraisemblance

*Idée.* Prony donne un bon point de départ ; on l'ajuste ensuite pour trouver les sources qui expliquent le mieux les mesures, compte tenu du bruit et de ce qu'on sait déjà.

#block(sticky: true)[*Principe.*]

+ #block(breakable: false)[Pour un nombre de sources $N$ fixé, on minimise l'écart entre mesures et modèle, chaque capteur étant pondéré par son niveau de bruit $sigma_k$ :
  $ chi^2 (theta) = sum_(k=1)^K ((y_k - Phi_theta (s_k)) / sigma_k)^2 $]
+ Contraintes : sources dans le domaine, intensités positives, somme des intensités connue.
+ Minimisation itérative (Gauss-Newton avec région de confiance), en partant de la solution de Prony : une dizaine de pas.
+ On recommence pour $N = 1$ à $6$ et on garde le $N$ qui minimise $chi^2 + 3 N log K$ (critère BIC, qui pénalise les sources en trop).

*Résultat.*
- Quand $N$ est bien estimé, la précision est optimale : elle atteint la borne de Cramér-Rao, meilleure précision possible d'un estimateur non biaisé.
- Le comptage reste le point faible : avec 3 sources en moyenne, $N$ est correct dans 59 % des cas à 0,1 % de bruit, et 34 % à 1 % (contre 36 % et 22 % pour Prony seul).
- Environ 100 ms par situation.

#figure(
  image("figures/n_correct.png", width: 75%),
  caption: [Banc d'essai : proportion de situations où le nombre de sources est bien estimé, selon le bruit ($lambda$ = nombre moyen de sources, 200 situations par point).],
)

*Limites.*
- Dépend du point de départ : si Prony se trompe beaucoup, l'optimisation peut s'arrêter sur une mauvaise solution.
- L'estimation de $N$ par le BIC rate surtout des sources faibles ou proches d'une autre.

# Localisation de sources ponctuelles

On cherche des sources ponctuelles (plutonium) dans un hangar $D = [0, 10] \times [0, 20]$, à partir
de capteurs qui mesurent chacun

$$\Phi(s_k) = \sum_i \frac{I_i}{\lVert s_k - z_i \rVert^2} + \frac{I_c}{\lVert s_k - c \rVert^2},$$

où la seconde partie vient d'une **source centrale connue** d'intensité $I_c$, au centre $c = (5, 10)$.
Le bruit est relatif ($\sigma$) : on retire le signal de la source centrale, mais son bruit reste.

Quatre méthodes sont comparées :

| Méthode | Fichier | Nombre de sources | Capteurs utilisés |
|---|---|---|---|
| **Prony** | `localisation/prony.py` | estimé | cercle seulement |
| **ML + BIC** (maximum de vraisemblance) | `localisation/ml_bic.py` | estimé (BIC) | tous |
| **BLASSO** (Frank–Wolfe glissant) | `localisation/blasso.py` | estimé | tous |
| **Descente de gradient** (méthode d'Orano) | `localisation/descente.py` | **donné** | tous |

`descente.py` contient aussi **Gauss–Newton** (même perte, même départ que la descente). Il sert de témoin : il montre que l'échec de la descente vient de l'algorithme.

## Installation

```bash
pip install numpy scipy matplotlib      # et pytest pour les tests
```

Aucune installation du dépôt n'est nécessaire : on lance les deux scripts depuis la racine.

---

## `main.py` : une simulation, une image

`main.py` fait quatre choses :
1. il tire des sources (ou prend celles que tu donnes) ;
2. il simule les mesures des capteurs ;
3. il lance les méthodes choisies ;
4. il enregistre **une image avec les vraies sources et les sources retrouvées**, un panneau par méthode, et écrit le détail dans le terminal.

```bash
python main.py
```
Ça lance le cadre du projet : 4 sources au hasard, σ = 10⁻³, I_c = 50 S, capteurs cercle + grille, et les 4 méthodes.

### Les options

| Option | Rôle | Défaut |
|---|---|---|
| `--N 6` | nombre de sources tirées au hasard | 4 |
| `--tirage 21` | numéro du tirage : même numéro = mêmes sources et même bruit | 0 |
| `--sources "2,5,60  8,15,30"` | sources imposées, au format `x,y,intensité`, séparées par des espaces | — |
| `--x 30` | cas à deux sources : rapport d'intensité $I_1/I_2$ (avec `--N 2`) | — |
| `--sigma 1e-3` | bruit relatif (0 : sans bruit) | 1e-3 |
| `--ic 50` | source centrale, en multiples de $S = 100$ (0 : aucune) | 50 |
| `--capteurs grille` | `cercle` (64 capteurs) ou `grille` (cercle + grille 16 × 32 = 576 capteurs) | grille |
| `--methodes ml_bic blasso` | parmi `prony ml_bic blasso descente gauss_newton` | `prony ml_bic blasso descente` |
| `--seuil 0.1` | rayon de tolérance : une source est « détectée » si elle est retrouvée à moins de ce rayon | 0,1 |
| `--sortie image.png` | où enregistrer l'image | `resultats/simulations/…` |
| `--afficher` | ouvre l'image à la fin | — |

### Exemples
```bash
python main.py --N 6 --capteurs cercle --ic 0 --methodes ml_bic blasso --tirage 21
python main.py --sources "2,5,60  8,15,30  5,3,10" --methodes ml_bic --afficher
python main.py --N 2 --x 30 --sigma 0 --ic 0 --capteurs cercle --methodes descente gauss_newton
```
Le dernier exemple refait le cas test d'Orano.

### Ce qu'on obtient
**Dans le terminal**, pour chaque méthode :
- les sources estimées, chacune avec sa distance à la vraie source la plus proche ;
- l'erreur E ;
- le nombre de sources détectées ;
- la précision ;
- le temps de calcul.

**L'image**, un panneau par méthode :
- les vraies sources en cercles noirs, d'aire proportionnelle à l'intensité, avec l'intensité écrite à côté ;
- les sources estimées en couleur ;
- les capteurs et la source centrale ;
- sous chaque panneau, le résumé.

---

## `main_test.py` : comparer les méthodes sur beaucoup de tirages

On choisit les méthodes et les valeurs des paramètres. Le programme lance **toutes les combinaisons** : méthodes × capteurs × N × σ × I_c × x × tirages. Il écrit ensuite les résultats et une figure.

```bash
python main_test.py --N 2 4 --tirages 10 --nom essai
```

Chaque paramètre accepte une ou plusieurs valeurs :

| Option | Rôle | Défaut |
|---|---|---|
| `--methodes` | méthodes comparées | `prony ml_bic blasso` |
| `--N` | nombres de sources | `2 4 6 10` |
| `--sigma` | bruits relatifs | `1e-3` |
| `--ic` | sources centrales ($\times S$) | `50` |
| `--capteurs` | `cercle`, `grille` | les deux |
| `--x` | rapports d'intensité (avec `--N 2`) | — |
| `--tirages` | tirages par combinaison | 100 |
| `--axe` | paramètre en abscisse de la figure (`N`, `sigma`, `ic`, `x`) | le premier qui varie |
| `--nom` | nom du test → dossier `resultats/tests/<nom>/` | `test` |
| `--processus` | calculs en parallèle | nombre de cœurs − 1 |
| `--timeout` | secondes au plus par calcul. Au-delà, le calcul compte comme un échec complet. | 300 |
| `--relancer-timeouts` | relance les calculs qui avaient dépassé le temps | — |
| `--seuil`, `--delta` | rayon de tolérance de la détection, rayon de fusion | 0,1 et 0,1 |

### Exemples
**Effet du bruit sur ML + BIC et le BLASSO, avec la grille :**
```bash
python main_test.py --methodes ml_bic blasso --N 4 --sigma 1e-4 1e-3 1e-2 --capteurs grille --nom bruit_grille
```

**Cas à deux sources, cas test d'Orano :**
```bash
python main_test.py --methodes descente gauss_newton ml_bic --N 2 --x 1 10 100 --sigma 0 --ic 0 --capteurs cercle --tirages 20 --nom orano
```

### Les études du rapport, sans rien recalculer
Les sources estimées de toutes les études du rapport sont déjà enregistrées dans `resultats/sources/`. Avec `--preset`, `main_test.py` les relit : les CSV sortent en quelques secondes, dans `resultats/etudes/<étude>/`.

```bash
python main_test.py --preset nombre_sources
```

| Étude (`--preset`) | Contenu |
|---|---|
| `nombre_sources` | N = 2, 4, 6, 10 ; cadre du projet (σ = 10⁻³, I_c = 50 S) |
| `bruit` | σ de 10⁻⁴ à 10⁻² (7 valeurs), N = 2, 4, 6, 10 |
| `source_centrale` | I_c / S = 0, 10, 25, 50, 100, 200, N = 2, 4, 6, 10 |
| `deux_sources` | N = 2, x = 1 à 300 : descente, Gauss–Newton et les 3 autres méthodes |
| `deux_sources_orano` | idem dans le cas test d'Orano (σ = 0, sans source centrale, cercle) |

### Ce qu'on obtient (dans `resultats/tests/<nom>/`)
- **`sources.jsonl`** : vraies sources et sources estimées de chaque calcul. Un calcul déjà présent n'est **jamais refait**. Si on arrête le programme, il reprend où il en était, et on peut changer `--seuil` ou `--delta` sans rien relancer.
- **`tirages.csv`** : une ligne par calcul.
- **`resume.csv`** : une ligne par combinaison. Elle donne :
  - la médiane de E et son intervalle de confiance à 95 % ;
  - la moyenne de E ;
  - le taux de détection ;
  - la part des tirages où toutes les sources sont détectées ;
  - la précision ;
  - la masse parasite ;
  - la part des tirages où N est bien estimé ;
  - le temps médian et le nombre de timeouts.
- **`figure.png`** : quatre panneaux (erreur E médiane, détection, précision et temps), en fonction du paramètre choisi par `--axe`.
- Le terminal affiche aussi un tableau résumé.

---

## Comment l'erreur est mesurée (`localisation/metrique.py`)

1. **Fusion.** Les sources estimées distantes de moins de δ = 0,1 sont fusionnées, et de même pour les vraies sources. Une source coupée en deux morceaux voisins n'est donc pas une erreur.
2. **Appariement.** Les sources estimées sont appariées aux vraies par l'algorithme hongrois.
3. **Indicateurs :**
   - **détection** : la part des vraies sources retrouvées à moins du **rayon de tolérance** (0,1) ;
   - **précision** : la distance entre source estimée et vraie source, calculée sur les sources détectées seulement ;
   - **masse parasite** : l'intensité des sources estimées en trop, en fraction de S ;
   - **E** $= \frac{1}{N} \sum_i (0{,}75\, d_i + 0{,}25\, r_i)$, avec $d_i \le 5$, $r_i \le 1$. Une source manquée compte $(5, 1)$, donc ajoute $4/N$ à E.

## Organisation

```
main.py                 une simulation → une image
main_test.py            comparaison des méthodes → CSV + figure
localisation/
    probleme.py         hangar, capteurs (cercle, grille), source centrale, bruit, tirage d'une situation
    noyau.py            noyau 1/r² et dérivées, Setup (blanchiment, normalisation), grille de recherche
    optimisation.py     briques communes à ML + BIC et au BLASSO (certificat, glissement, NNLS, LASSO)
    prony.py            méthode de Prony
    ml_bic.py           maximum de vraisemblance + BIC
    blasso.py           BLASSO par Frank–Wolfe glissant
    descente.py         descente de gradient d'Orano + Gauss–Newton
    methodes.py         registre des méthodes : lancer(nom, situation)
    metrique.py         fusion, appariement, détection, précision, E
    comparaison.py      tâches, stockage des sources, agrégation (bootstrap)
    figures.py          image de reconstruction, courbes de comparaison
    pool.py             calcul parallèle avec timeout par calcul
tests/                  python -m pytest
resultats/
    sources/            sources estimées de toutes les études du rapport (JSONL)
    etudes/<étude>/     resume.csv, tirages.csv, parametres.json (main_test.py --preset <étude>)
    images/             3 figures : reconstruction, nombre de sources, cas à deux sources
rapport/                compte rendu (Typst, PDF, Markdown) et ses images
```

**Reproductibilité.** Les sources, le bruit et le départ de la descente ne dépendent que de (N, numéro de tirage). Pour un même tirage, toutes les méthodes, les deux dispositions de capteurs et toutes les valeurs de σ et I_c voient donc les mêmes sources et le même bruit.

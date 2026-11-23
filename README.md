# Localisation de sources de particules

Des sources ponctuelles apparaissent aléatoirement dans un domaine et émettent
des particules. Un capteur à distance r d'une source d'intensité I mesure I / r².
Le but est de retrouver le nombre, la position et l'intensité des sources à
partir des mesures de capteurs placés sur le cercle circonscrit au domaine.

## Installation

```bash
pip install -e ".[dev]"
```

L'installation en mode éditable (`-e`) rend le paquet `particle_sources`
importable depuis n'importe où, et les modifications du code sont prises en
compte sans réinstaller.

## Utilisation

```bash
python scripts/simulate.py       # génération + affichage
python scripts/reconstruct.py    # + bruit + reconstruction
pytest                           # tests
```

## Jeux de données pour l'apprentissage

```bash
python scripts/generate_dataset.py --set dataset.n_samples=100000
```

Paramètres dans `config/dataset.yaml` ; les sources, capteurs et bruit
suivent les autres fichiers de config. Le dossier de sortie (ignoré par git)
contient `config.yaml` (config utilisée), `manifest.json` et des fichiers
`shard_XXXXX.npz`. Lecture :

```python
from particle_sources.dataset import load_dataset
data = load_dataset("data/dataset")
```

| Clé | Forme | Contenu |
|---|---|---|
| `measurements` | (M, K) | mesures bruitées (entrée du modèle) |
| `measurements_clean` | (M, K) | mesures sans bruit (si `store_clean`) |
| `n_sources` | (M,) | nombre de sources tirées |
| `positions` | (M, N_max, 2) | positions, triées par intensité décroissante, 0 au-delà de `n_sources` |
| `intensities` | (M, N_max) | intensités, même ordre, 0 au-delà |
| `mask` | (M, N_max) | True pour les vraies sources |
| `sensor_positions` | (K, 2) | positions des capteurs |

Les sources fixes (source centrale) sont dans les mesures mais pas dans les
labels ; elles sont décrites dans `manifest.json`.

## Banc d'essai des méthodes

```bash
python scripts/benchmark.py
```

Paramètres dans `config/benchmark.yaml` : un paramètre de config est balayé
(niveau de bruit, nombre de capteurs, ...) et chaque méthode est évaluée sur
les mêmes situations. Sorties dans `outputs/benchmark/` : `summary.csv`,
`benchmark.png`, `config.yaml`.

Métriques : proportion de N correct, distance OSPA, erreurs médianes de
position / intensité (situations à N correct), et rapport erreur / borne de
Cramér-Rao source par source (≈ 0,7-0,8 : la méthode atteint la meilleure
précision possible à N connu).

Une nouvelle méthode (par ex. un réseau entraîné) s'écrit comme une fonction
`(y, problem: InverseProblem) -> Sources`, s'enregistre dans
`config/registry.py`, et s'ajoute à la liste `methods` du banc d'essai.

## Cas test à deux sources (comparaison avec l'industriel)

```bash
python scripts/two_sources.py
python scripts/two_sources.py --set two_sources.random_trials.n=0   # cas fixe seul (~1 min)
```

Deux sources, l'une x fois plus intense que l'autre, reconstruites par
descente de gradient (méthode de l'industriel : réussite annoncée pour
x = 10, échec pour x = 100). Comparaison avec Gauss-Newton depuis la même
initialisation et avec Prony + maximum de vraisemblance. Paramètres dans
`config/two_sources.yaml` ; sorties dans `outputs/two_sources/` :
`fixed_case.csv`, `random_trials.csv`, `two_sources.png`, `config.yaml`.

Même cas test avec bruit gaussien relatif, x autour de 100, comparé à la
borne de Cramér-Rao (paramètres dans `config/two_sources_noise.yaml`, sorties
dans `outputs/two_sources_noise/`) :

```bash
python scripts/two_sources_noise.py
```

## Configuration

Tous les paramètres sont dans le dossier `config/`, un fichier par étape :

| Fichier | Contenu |
|---|---|
| `simulation.yaml` | graine, domaine, lois du nombre / des intensités / des positions des sources |
| `sensors.yaml` | géométrie des capteurs, bruit de mesure |
| `reconstruction.yaml` | méthode d'inversion et ses paramètres |
| `visualization.yaml` | options d'affichage, chemin d'enregistrement de la figure |
| `two_sources.yaml` | cas test à deux sources (scripts/two_sources.py) |
| `two_sources_noise.yaml` | même cas test avec bruit (scripts/two_sources_noise.py) |

Chaque loi ou méthode s'écrit `{name: ..., params: {...}}` ; les noms
disponibles sont listés en commentaire dans chaque fichier (et dans
`src/particle_sources/config/registry.py`).

Pour une expérience ponctuelle, on surcharge un paramètre sans toucher aux
fichiers :

```bash
python scripts/reconstruct.py --set seed=3 --set sensors.noise.params.sigma=0.01
```

Pour une expérience à conserver, copier le dossier (par ex.
`config/experiments/bruit_fort/`) et le passer avec `--config`. Quand
`save_path` est renseigné, la config effectivement utilisée (surcharges
comprises) est enregistrée à côté de la figure, en `.yaml`.

## Organisation

```
config/                Paramètres des expériences (YAML)
src/particle_sources/
├── config/            Lecture des YAML et construction des objets
│   ├── loader.py        load_config -> ExperimentConfig
│   ├── registry.py      noms YAML -> fonctions du code
│   └── cli.py           options --config / --set communes aux scripts
├── model/             Objets physiques (aucun aléatoire, aucun affichage)
│   ├── domain.py        Box : domaine rectangulaire, cercle circonscrit
│   ├── sources.py       Sources : positions + intensités
│   └── field.py         intensity_at : modèle direct Phi(x) = sum I_i / r_i²
├── generation/        Génération de données synthétiques
│   ├── count_laws.py    lois du nombre de sources (Poisson, binomiale, ...)
│   ├── intensity_laws.py lois des intensités (uniforme sur le simplexe, Dirichlet)
│   ├── position_laws.py lois des positions (uniforme dans le domaine)
│   └── sampler.py       sample_sources : assemble les trois lois
├── sensors/           Capteurs
│   ├── layouts.py       géométries de capteurs (CircleSensors)
│   ├── noise.py         modèles de bruit (gaussien relatif, comptage Poisson)
│   └── measurement.py   measure : simule les mesures
├── inversion/         Problème inverse : méthodes (y, InverseProblem) -> Sources
│   ├── problem.py       InverseProblem : ce que l'inversion sait (capteurs, bruit, ...)
│   ├── prony.py         méthode de Prony (formule directe)
│   ├── likelihood.py    affinage par maximum de vraisemblance, choix de N (BIC)
│   └── gradient_descent.py descente de gradient (méthode de l'industriel)
├── benchmark/         Évaluation des méthodes
│   ├── metrics.py       association des sources, OSPA, erreurs
│   ├── crb.py           borne de Cramér-Rao
│   ├── runner.py        balayage d'un paramètre, comparaison des méthodes
│   ├── intensity_ratio.py cas test à deux sources de rapport d'intensité x
│   └── two_sources_noise.py même cas test avec bruit (borne de Cramér-Rao)
├── dataset/           Jeux de données pour l'apprentissage
└── visualization/     Affichage uniquement (matplotlib)
    ├── field_map.py     carte de Phi, sources, domaine
    ├── sensors_plot.py  capteurs et profil sur le cercle
    ├── figures.py       figures complètes
    ├── intensity_ratio_plot.py figures du cas test à deux sources
    └── two_sources_noise_plot.py figures du cas test à deux sources avec bruit
scripts/               Points d'entrée : expériences reproductibles
tests/                 Tests unitaires (pytest)
```

Dépendances entre sous-paquets (une flèche = « importe ») :

```
model  ←  generation
model  ←  sensors  ←  inversion
model, sensors  ←  visualization
model, generation, sensors, inversion  ←  config
tout  ←  scripts
```

`model` ne dépend de rien. Rien n'importe `visualization` ni `scripts`.

## Conventions

- **Aléatoire** : aucune graine globale. Chaque fonction aléatoire reçoit un
  `np.random.Generator` ; la graine est fixée une seule fois, dans le script.
- **Lois interchangeables** : une loi est une simple fonction (par ex.
  `rng -> int` pour le nombre de sources). Pour en ajouter une, écrire une
  fabrique dans le fichier `*_laws.py` correspondant, puis l'enregistrer
  sous un nom dans `config/registry.py`.
- **Nouvelle méthode d'inversion** : un nouveau fichier dans `inversion/`
  qui renvoie un objet `Sources`, son nom dans `registry.py`, plus un test.
- **Nouvelle géométrie de capteurs** : une classe dans `sensors/layouts.py`
  avec un attribut `positions` de forme (K, d), et son nom dans `registry.py`.
- **Aucun paramètre en dur dans les scripts** : tout passe par `config/`.

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
python scripts/simulate.py            # génération standard + affichage
python scripts/simulate_cluster.py    # simulation avec gros cluster central émetteur
python scripts/reconstruct.py         # + bruit + reconstruction
pytest                                # tests
```

### Simulation avec cluster central

Une configuration prédéfinie est disponible dans `config/experiments/cluster/`.
Elle place au centre du domaine un cluster dense émettant la grande majorité de la radiation (par défaut 90% de l'intensité sur 80% des sources dans un rayon de 1.5).

```bash
# Lancer la simulation cluster
python scripts/simulate_cluster.py

# Modifier la taille spatiale du cluster (rayon r = 0.5 pour un cluster très compact, r = 3.0 pour plus étendu)
python scripts/simulate_cluster.py --set sources.position_law.params.radius=0.5

# Modifier l'intensité totale ou la part émise par le cluster
python scripts/simulate_cluster.py --set sources.total_intensity=1000.0
python scripts/simulate_cluster.py --set sources.intensity_law.params.cluster_share=0.95

# Ou lancer avec simulate.py / reconstruct.py via --config
python scripts/simulate.py --config config/experiments/cluster
python scripts/reconstruct.py --config config/experiments/cluster
```

## Configuration

Tous les paramètres sont dans le dossier `config/`, un fichier par étape :

| Fichier | Contenu |
|---|---|
| `simulation.yaml` | graine, domaine, lois du nombre / des intensités / des positions des sources |
| `sensors.yaml` | géométrie des capteurs, bruit de mesure |
| `reconstruction.yaml` | méthode d'inversion et ses paramètres |
| `visualization.yaml` | options d'affichage, chemin d'enregistrement de la figure |

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
├── inversion/         Problème inverse
│   └── prony.py         reconstruction par la méthode de Prony
└── visualization/     Affichage uniquement (matplotlib)
    ├── field_map.py     carte de Phi, sources, domaine
    ├── sensors_plot.py  capteurs et profil sur le cercle
    └── figures.py       figures complètes
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

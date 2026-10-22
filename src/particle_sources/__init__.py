"""
Localisation de sources de particules à partir de mesures d'intensité.

Des sources ponctuelles (positions x_i, intensités I_i) sont placées dans un
domaine rectangulaire ; une grosse source fixe peut s'y ajouter au centre. Des
capteurs disposés sur un cercle autour du domaine mesurent

    Phi(s) = sum_i I_i / ||s - x_i||^2

et l'on cherche à retrouver le nombre, la position et l'intensité des sources
à partir de ces seules mesures (problème inverse).

Sous-paquets
------------
model          objets physiques (domaine, sources) et modèle direct Phi
generation     tirage aléatoire de configurations de sources
sensors        géométrie des capteurs, bruit, simulation des mesures
inversion      reconstruction des sources à partir des mesures
config         lecture des fichiers YAML et construction des objets
visualization  figures matplotlib
"""

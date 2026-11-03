"""Ce que la méthode d'inversion sait du problème (en plus des mesures)."""

from dataclasses import dataclass

import numpy as np

from ..model import Box, Sources, intensity_at
from ..sensors.noise import NoiseModel


@dataclass(frozen=True)
class InverseProblem:
    sensors: object            # géométrie des capteurs (.positions)
    box: Box                   # domaine où se trouvent les sources inconnues
    noise: NoiseModel          # modèle de bruit (pour pondérer / la vraisemblance)
    known: Sources             # sources connues (ex. source centrale), sinon vide
    total_intensity: float     # somme des intensités des sources inconnues (si N >= 1)
    eps: float = 0.0

    def known_contribution(self) -> np.ndarray:
        return intensity_at(self.known, self.sensors.positions, self.eps)

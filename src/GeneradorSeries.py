import random
import numpy as np


class GeneradorSeries:
    def __init__(self, semilla=None):
        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

    """
    Genera una serie temporal aleatoria de 6 numeros unicos ordenados
    """
    def generar_serie_aleatoria(self, cantidad=6, minimo=1, maximo=49):
        serie = random.sample(range(minimo, maximo + 1), cantidad)
        serie.sort()
        return serie

import random
import numpy as np


class GeneradorSeries:
    def __init__(self, semilla=None):
        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

    def generar_serie_aleatoria(self, cantidad=6, minimo=1, maximo=49):
        """
        Genera una serie temporal aleatoria de 6 numeros unicos ordenados
        """
        serie = random.sample(range(minimo, maximo + 1), cantidad)
        serie.sort()
        return serie
    
    def generar_series(self, cantidad, numeros_por_serie=6, minimo=1, maximo=49):
        """
        Genera múltiples series temporales aleatorias
        
        Args:
            cantidad: número de series a generar
            numeros_por_serie: cantidad de números por serie (default: 6)
            minimo: valor mínimo de los números (default: 1)
            maximo: valor máximo de los números (default: 49)
        
        Returns:
            Lista de listas con las series generadas
        """
        series = []
        for _ in range(cantidad):
            serie = self.generar_serie_aleatoria(numeros_por_serie, minimo, maximo)
            series.append(serie)
        return series

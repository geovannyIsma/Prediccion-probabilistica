"""
Clases encargado de crear los datos de entrenamiento
crear un dataframe con la 1000 combinaciones aleatorias de loteria con un columnas 
"""

import pandas as pd
import numpy as np
from GeneradorSeries import GeneradorSeries

class DatosLoteria:
    def __init__(self, cantidad_combinaciones=1000, semilla=None):
        self.cantidad_combinaciones = cantidad_combinaciones
        self.generador = GeneradorSeries(semilla=semilla)
        self.datos = self.generar_datos()
    
    def generar_datos(self):
        """
        Genera un DataFrame con combinaciones aleatorias
        """
        series = self.generador.generar_series(self.cantidad_combinaciones)
        df = pd.DataFrame(series, columns=[f'Num{i+1}' for i in range(6)])
        return df
    
    def generar_datos_entrenamiento(self, cantidad=1000, porcentaje_exito=0.1):
        """
        Genera un DataFrame con combinaciones y columna 'Exito'
        
        Args:
            cantidad: número de combinaciones a generar
            porcentaje_exito: porcentaje de combinaciones ganadoras (default: 0.1)
        
        Returns:
            DataFrame con combinaciones y columna 'Exito'
        """
        series = self.generador.generar_series(cantidad)
        df = pd.DataFrame(series, columns=[f'Num{i+1}' for i in range(6)])
        
        # Generar columna Exito: 10% con valor 1, resto con 0
        cantidad_exitos = int(cantidad * porcentaje_exito)
        exitos = np.concatenate([
            np.ones(cantidad_exitos, dtype=int),
            np.zeros(cantidad - cantidad_exitos, dtype=int)
        ])
        np.random.shuffle(exitos)
        df['Exito'] = exitos
        
        return df
    
    def generar_datos_evaluacion(self, cantidad=100):
        """
        Genera un DataFrame con combinaciones para evaluar (sin columna 'Exito')
        
        Args:
            cantidad: número de combinaciones a generar
        
        Returns:
            DataFrame con combinaciones para evaluar
        """
        series = self.generador.generar_series(cantidad)
        df = pd.DataFrame(series, columns=[f'Num{i+1}' for i in range(6)])
        return df

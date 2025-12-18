from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


class ModeloLoteria:
    def __init__(self, n_estimators=100, random_state=42):
        """
        Inicializa el modelo de predicción de lotería
        
        Args:
            n_estimators: número de árboles en el bosque
            random_state: semilla para reproducibilidad
        """
        self.modelo = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.entrenado = False
    
    def entrenar(self, X, y):
        """
        Entrena el modelo con los datos escalados
        
        Args:
            X: características (combinaciones numéricas)
            y: etiquetas (Exito: 1 o 0)
        """
        # Escalar los datos
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar el modelo
        self.modelo.fit(X_scaled, y)
        self.entrenado = True
        
        return self
    
    def predecir_probabilidades(self, X):
        """
        Devuelve las probabilidades de éxito para cada combinación
        
        Args:
            X: características (combinaciones numéricas)
        
        Returns:
            Array con probabilidades de éxito (clase 1)
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        # Escalar los datos
        X_scaled = self.scaler.transform(X)
        
        # Obtener probabilidades (columna 1 = probabilidad de éxito)
        probabilidades = self.modelo.predict_proba(X_scaled)[:, 1]
        
        return probabilidades
    
    def predecir(self, X):
        """
        Predice la clase (éxito o fracaso) para cada combinación
        
        Args:
            X: características (combinaciones numéricas)
        
        Returns:
            Array con predicciones (1 o 0)
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado antes de predecir")
        
        X_scaled = self.scaler.transform(X)
        return self.modelo.predict(X_scaled)

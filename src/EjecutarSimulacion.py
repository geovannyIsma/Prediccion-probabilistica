from DatosLoteria import DatosLoteria
from ModeloLoteria import ModeloLoteria
from GeneradorSeries import GeneradorSeries
from VisualizadorResultados import VisualizadorResultados
import pandas as pd


class EjecutarSimulacion:
    """
    Clase principal que ejecuta todo el flujo del proyecto
    """
    
    def __init__(self, cantidad_entrenamiento=1000, cantidad_evaluacion=100):
        """
        Inicializa la simulación
        
        Args:
            cantidad_entrenamiento: Número de combinaciones para entrenar
            cantidad_evaluacion: Número de combinaciones para evaluar
        """
        self.cantidad_entrenamiento = cantidad_entrenamiento
        self.cantidad_evaluacion = cantidad_evaluacion
        self.datos_loteria = None
        self.modelo = None
        self.visualizador = VisualizadorResultados()
        self.generador = GeneradorSeries(semilla=123)
        self.df_nuevas = None
        self.probabilidades = None
    
    def ejecutar(self):
        """
        Ejecuta el flujo completo del proyecto:
        1. Genera los datos simulados
        2. Entrena el modelo
        3. Genera nuevas combinaciones a evaluar
        4. Predice la probabilidad de éxito
        5. Muestra por pantalla la mejor combinación
        6. Muestra un gráfico con las 10 más prometedoras
        """
        self._mostrar_encabezado()
        
        # Paso 1: Generar datos simulados
        self._generar_datos()
        
        # Paso 2: Entrenar el modelo
        self._entrenar_modelo()
        
        # Paso 3: Generar nuevas combinaciones
        self._generar_combinaciones_evaluar()
        
        # Paso 4: Predecir probabilidades
        self._predecir_probabilidades()
        
        # Paso 5: Mostrar mejor combinación
        self._mostrar_resultados()
        
        # Paso 6: Mostrar gráfico con top 10
        self._visualizar_resultados()
    
    def _mostrar_encabezado(self):
        """Muestra el encabezado del sistema"""
        print("=" * 60)
        print("Sistema de Predicción Probabilística de Lotería")
        print("Implementación con Programación Orientada a Objetos")
        print("=" * 60)
    
    def _generar_datos(self):
        """Genera los datos simulados para entrenamiento"""
        print("\n[1] Generando datos simulados para entrenamiento...")
        self.datos_loteria = DatosLoteria(cantidad_combinaciones=self.cantidad_entrenamiento)
        self.df_entrenamiento = self.datos_loteria.generar_datos_entrenamiento(
            cantidad=self.cantidad_entrenamiento
        )
        
        print(f"   ✓ Datos generados: {len(self.df_entrenamiento)} combinaciones")
        print(f"   ✓ Combinaciones exitosas: {self.df_entrenamiento['Exito'].sum()}")
        print(f"   ✓ Tasa de éxito: {self.df_entrenamiento['Exito'].mean()*100:.1f}%")
    
    def _entrenar_modelo(self):
        """Entrena el modelo de predicción"""
        print("\n[2] Entrenando modelo RandomForestClassifier...")
        
        # Preparar datos
        X = self.df_entrenamiento.drop('Exito', axis=1)
        y = self.df_entrenamiento['Exito']
        
        # Entrenar modelo
        self.modelo = ModeloLoteria(n_estimators=100, random_state=42)
        self.modelo.entrenar(X, y)
        
        print("   ✓ Modelo entrenado exitosamente")
        print("   ✓ Características escaladas con StandardScaler")
    
    def _generar_combinaciones_evaluar(self):
        """Genera nuevas combinaciones para evaluar"""
        print(f"\n[3] Generando {self.cantidad_evaluacion} combinaciones para evaluar...")
        
        nuevas_series = self.generador.generar_series(self.cantidad_evaluacion)
        self.df_nuevas = pd.DataFrame(nuevas_series, columns=[f'Num{i+1}' for i in range(6)])
        
        print(f"   ✓ {len(self.df_nuevas)} combinaciones generadas")
    
    def _predecir_probabilidades(self):
        """Predice la probabilidad de éxito para las nuevas combinaciones"""
        print("\n[4] Prediciendo probabilidades de éxito...")
        
        self.probabilidades = self.modelo.predecir_probabilidades(self.df_nuevas)
        
        print(f"   ✓ Probabilidades calculadas para {len(self.probabilidades)} combinaciones")
    
    def _mostrar_resultados(self):
        """Muestra por pantalla la mejor combinación y estadísticas"""
        print("\n[5] Mostrando resultados...")
        
        # Mostrar mejor combinación
        self.visualizador.mostrar_mejor_combinacion(self.df_nuevas, self.probabilidades)
        
        # Mostrar estadísticas
        self.visualizador.mostrar_estadisticas(self.probabilidades)
    
    def _visualizar_resultados(self):
        """Muestra un gráfico con las 10 combinaciones más prometedoras"""
        print("\n[6] Generando visualización...")
        print("   ✓ Mostrando gráfico de Top 10 combinaciones...")
        
        self.visualizador.graficar_top_combinaciones(
            self.df_nuevas, 
            self.probabilidades, 
            top_n=10
        )


def main():
    """Función principal para ejecutar la simulación"""
    simulacion = EjecutarSimulacion(
        cantidad_entrenamiento=1000,
        cantidad_evaluacion=100
    )
    simulacion.ejecutar()


if __name__ == "__main__":
    main()

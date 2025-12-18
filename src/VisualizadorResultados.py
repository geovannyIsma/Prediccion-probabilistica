import matplotlib.pyplot as plt
import pandas as pd


class VisualizadorResultados:
    def __init__(self):
        """
        Inicializa el visualizador de resultados
        """
        self.configurar_estilo()
    
    def configurar_estilo(self):
        """
        Configura el estilo de los gráficos
        """
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def graficar_top_combinaciones(self, df_series, probabilidades, top_n=10):
        """
        Muestra un gráfico de barras horizontal con las combinaciones más prometedoras
        
        Args:
            df_series: DataFrame con las combinaciones (columnas Num1-Num6)
            probabilidades: Array con las probabilidades de éxito
            top_n: Número de combinaciones a mostrar (default: 10)
        """
        # Crear DataFrame con resultados
        df_resultados = df_series.copy()
        df_resultados['Probabilidad'] = probabilidades
        
        # Ordenar por probabilidad descendente y tomar top_n
        df_top = df_resultados.nlargest(top_n, 'Probabilidad')
        
        # Crear etiquetas para las combinaciones
        etiquetas = []
        for idx, row in df_top.iterrows():
            combinacion = [int(row[f'Num{i+1}']) for i in range(6)]
            etiqueta = str(combinacion)
            etiquetas.append(etiqueta)
        
        # Crear el gráfico
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Gráfico de barras horizontal
        y_pos = range(len(etiquetas))
        probabilidades_top = df_top['Probabilidad'].values
        
        barras = ax.barh(y_pos, probabilidades_top, color='steelblue', alpha=0.8)
        
        # Agregar valores en las barras
        for i, (barra, prob) in enumerate(zip(barras, probabilidades_top)):
            ax.text(prob + 0.001, i, f'{prob:.4f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        # Configurar ejes y etiquetas
        ax.set_yticks(y_pos)
        ax.set_yticklabels(etiquetas, fontsize=9)
        ax.set_xlabel('Probabilidad de Éxito', fontsize=12, fontweight='bold')
        ax.set_ylabel('Combinaciones', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Combinaciones con Mayor Probabilidad de Éxito',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Agregar grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Ajustar layout
        plt.tight_layout()
        plt.show()
    
    def mostrar_estadisticas(self, probabilidades):
        """
        Muestra estadísticas de las probabilidades calculadas
        
        Args:
            probabilidades: Array con las probabilidades de éxito
        """
        print("\n" + "=" * 60)
        print("Estadísticas de Probabilidades")
        print("=" * 60)
        print(f"Probabilidad máxima:    {probabilidades.max():.6f}")
        print(f"Probabilidad mínima:    {probabilidades.min():.6f}")
        print(f"Probabilidad promedio:  {probabilidades.mean():.6f}")
        print(f"Desviación estándar:    {probabilidades.std():.6f}")
        print(f"Mediana:                {pd.Series(probabilidades).median():.6f}")
        print("=" * 60)
    
    def mostrar_mejor_combinacion(self, df_series, probabilidades):
        """
        Muestra la mejor combinación encontrada
        
        Args:
            df_series: DataFrame con las combinaciones
            probabilidades: Array con las probabilidades de éxito
        """
        # Encontrar el índice de la máxima probabilidad
        idx_max = probabilidades.argmax()
        prob_max = probabilidades[idx_max]
        
        # Obtener la combinación
        mejor_combinacion = [int(df_series.iloc[idx_max][f'Num{i+1}']) for i in range(6)]
        
        print("\n" + "🎯" * 30)
        print("MEJOR COMBINACIÓN ENCONTRADA")
        print("🎯" * 30)
        print(f"\nCombinación: {mejor_combinacion}")
        print(f"Probabilidad de éxito: {prob_max:.6f} ({prob_max*100:.4f}%)")
        print("=" * 60)

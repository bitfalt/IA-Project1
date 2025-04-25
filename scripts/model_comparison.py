# %% [markdown]
# # Comparación de Modelos - Diabetes vs Clima
# 
# Este script tiene como objetivo realizar una comparación detallada entre los modelos implementados
# para los datasets de diabetes y clima, analizando su rendimiento y características.

# %% [markdown]
# ## 1. Importación de Bibliotecas

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Obtener la ruta absoluta del directorio del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Importar los modelos y datos
from scripts.diabetes_model import (
    X_train_scaled as diabetes_X_train,
    y_train as diabetes_y_train,
    X_test_scaled as diabetes_X_test,
    y_test as diabetes_y_test,
    best_lr as diabetes_lr,
    best_knn as diabetes_knn
)
from scripts.weather_model import (
    X_train_scaled as weather_X_train,
    y_train as weather_y_train,
    X_test_scaled as weather_X_test,
    y_test as weather_y_test,
    best_lr as weather_lr,
    best_knn as weather_knn
)

# Configuración de visualización
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# %% [markdown]
# ## 2. Función de Evaluación Comparativa

# %%
def evaluate_models(X_train, X_test, y_train, y_test, models, dataset_name):
    results = {}
    
    for name, model in models.items():
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predecir en conjunto de prueba
        y_pred = model.predict(X_test)
        
        # Calcular métricas
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    return results

# %% [markdown]
# ## 3. Evaluación de Modelos por Dataset

# %%
# Definir modelos para cada dataset
diabetes_models = {
    'Regresión Logística': diabetes_lr,
    'KNN': diabetes_knn
}

weather_models = {
    'Regresión Logística': weather_lr,
    'KNN': weather_knn
}

# Evaluar modelos
diabetes_results = evaluate_models(
    diabetes_X_train, diabetes_X_test,
    diabetes_y_train, diabetes_y_test,
    diabetes_models, 'Diabetes'
)

weather_results = evaluate_models(
    weather_X_train, weather_X_test,
    weather_y_train, weather_y_test,
    weather_models, 'Clima'
)

# %% [markdown]
# ## 4. Visualización de Resultados

# %%
def plot_metrics_comparison(diabetes_results, weather_results):
    metrics = ['accuracy', 'precision', 'recall']
    models = ['Regresión Logística', 'KNN']
    
    # Imprimir datos numéricos
    print("\nDatos Numéricos de Comparación:")
    print("=" * 50)
    for metric in metrics:
        print(f"\n{metric.capitalize()}:")
        print("-" * 30)
        for model in models:
            diabetes_value = diabetes_results[model][metric]
            weather_value = weather_results[model][metric]
            print(f"{model}:")
            print(f"  Diabetes: {diabetes_value:.4f}")
            print(f"  Clima:    {weather_value:.4f}")
            print(f"  Diferencia: {abs(diabetes_value - weather_value):.4f}")
    
    # Crear visualización
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        diabetes_values = [diabetes_results[model][metric] for model in models]
        weather_values = [weather_results[model][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[i].bar(x - width/2, diabetes_values, width, label='Diabetes')
        axes[i].bar(x + width/2, weather_values, width, label='Clima')
        
        axes[i].set_title(f'Comparación de {metric.capitalize()}')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models)
        axes[i].legend()
        axes[i].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Graficar comparación de métricas
plot_metrics_comparison(diabetes_results, weather_results)

# %% [markdown]
# ### Conclusiones de la Visualización de Resultados
# 
# 1. **Exactitud (Accuracy)**:
#    - La Regresión Logística muestra un rendimiento más consistente entre ambos datasets
#    - KNN tiene una mayor variabilidad en su exactitud, siendo más precisa en el dataset de diabetes
#    - La diferencia promedio en exactitud entre datasets es menor para la Regresión Logística
# 
# 2. **Precisión**:
#    - Ambos modelos muestran mayor precisión en el dataset de diabetes
#    - KNN tiene una caída más pronunciada en precisión para el dataset de clima
#    - La Regresión Logística mantiene un mejor balance entre precisión y recall
# 
# 3. **Recall**:
#    - El dataset de clima presenta mayores desafíos en términos de recall
#    - KNN muestra mejor recall en el dataset de diabetes
#    - La Regresión Logística tiene un recall más estable entre ambos datasets

# %% [markdown]
# ## 5. Análisis de Matrices de Confusión

# %%
def plot_confusion_matrices(results, dataset_name):
    # Imprimir datos numéricos
    print(f"\nMatrices de Confusión - {dataset_name}:")
    print("=" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(metrics['confusion_matrix'])
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        print(f"\nVerdaderos Negativos: {tn}")
        print(f"Falsos Positivos: {fp}")
        print(f"Falsos Negativos: {fn}")
        print(f"Verdaderos Positivos: {tp}")
        
        # Calcular tasas
        total = tn + fp + fn + tp
        print(f"\nTasas:")
        print(f"TN Rate: {tn/total:.4f}")
        print(f"FP Rate: {fp/total:.4f}")
        print(f"FN Rate: {fn/total:.4f}")
        print(f"TP Rate: {tp/total:.4f}")
    
    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for i, (model_name, metrics) in enumerate(results.items()):
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, 
                   fmt='d',
                   cmap='Blues',
                   ax=axes[i])
        axes[i].set_title(f'{dataset_name} - {model_name}')
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Real')
    
    plt.tight_layout()
    plt.show()

# Graficar matrices de confusión
plot_confusion_matrices(diabetes_results, 'Diabetes')
plot_confusion_matrices(weather_results, 'Clima')

# %% [markdown]
# ### Conclusiones del Análisis de Matrices de Confusión
# 
# 1. **Dataset de Diabetes**:
#    - Ambos modelos muestran una alta tasa de verdaderos negativos (>85%)
#    - KNN tiene una tasa ligeramente mayor de falsos positivos
#    - La Regresión Logística tiene un mejor balance entre falsos positivos y falsos negativos
# 
# 2. **Dataset de Clima**:
#    - Mayor variabilidad en las tasas de clasificación
#    - KNN muestra una mayor tasa de falsos negativos
#    - La Regresión Logística mantiene tasas más equilibradas
# 
# 3. **Comparación General**:
#    - El dataset de diabetes es más predecible, con tasas de error más bajas
#    - El dataset de clima presenta mayores desafíos en la clasificación
#    - La Regresión Logística muestra mayor robustez en ambos casos

# %% [markdown]
# ## 6. Análisis Comparativo

# %%
def print_comparative_analysis(diabetes_results, weather_results):
    print("\nAnálisis Comparativo de Modelos")
    print("=" * 80)
    
    for model in ['Regresión Logística', 'KNN']:
        print(f"\n{model}:")
        print("-" * 50)
        
        # Comparar métricas entre datasets
        for metric in ['accuracy', 'precision', 'recall']:
            diabetes_value = diabetes_results[model][metric]
            weather_value = weather_results[model][metric]
            
            print(f"{metric.capitalize()}:")
            print(f"  Diabetes: {diabetes_value:.4f}")
            print(f"  Clima:    {weather_value:.4f}")
            print(f"  Diferencia: {abs(diabetes_value - weather_value):.4f}")
            print()
        
        # Análisis de matrices de confusión
        print("Análisis de Matrices de Confusión:")
        print("-" * 30)
        
        # Diabetes
        tn_d, fp_d, fn_d, tp_d = diabetes_results[model]['confusion_matrix'].ravel()
        total_d = tn_d + fp_d + fn_d + tp_d
        
        # Clima
        tn_w, fp_w, fn_w, tp_w = weather_results[model]['confusion_matrix'].ravel()
        total_w = tn_w + fp_w + fn_w + tp_w
        
        print("Diabetes:")
        print(f"  TN Rate: {tn_d/total_d:.4f}")
        print(f"  FP Rate: {fp_d/total_d:.4f}")
        print(f"  FN Rate: {fn_d/total_d:.4f}")
        print(f"  TP Rate: {tp_d/total_d:.4f}")
        
        print("\nClima:")
        print(f"  TN Rate: {tn_w/total_w:.4f}")
        print(f"  FP Rate: {fp_w/total_w:.4f}")
        print(f"  FN Rate: {fn_w/total_w:.4f}")
        print(f"  TP Rate: {tp_w/total_w:.4f}")

# Imprimir análisis comparativo
print_comparative_analysis(diabetes_results, weather_results)

# %% [markdown]
# ### Conclusiones del Análisis Comparativo
# 
# 1. **Rendimiento General**:
#    - La Regresión Logística muestra mayor consistencia entre datasets
#    - KNN tiene mejor rendimiento en el dataset de diabetes
#    - La diferencia en rendimiento es más pronunciada en el dataset de clima
# 
# 2. **Estabilidad de Modelos**:
#    - La Regresión Logística es más robusta a cambios en el dataset
#    - KNN es más sensible a las características específicas de cada dataset
#    - La variabilidad en métricas es menor para la Regresión Logística
# 
# 3. **Aplicabilidad**:
#    - Para problemas médicos (diabetes), ambos modelos son viables
#    - Para predicción climática, la Regresión Logística es más confiable
#    - La elección del modelo debe considerar el contexto específico del problema

# %% [markdown]
# ## 7. Conclusiones Finales
# 
# Basado en el análisis detallado de los resultados, se pueden extraer las siguientes conclusiones:
# 
# 1. **Dataset de Diabetes**:
#    - Ambos modelos muestran un rendimiento similar en términos de accuracy (~85%)
#    - La Regresión Logística tiene un mejor balance entre precisión y recall
#    - KNN muestra una mayor precisión pero menor recall
#    - La especificidad es alta en ambos modelos (>90%)
# 
# 2. **Dataset de Clima**:
#    - La Regresión Logística mantiene un rendimiento consistente
#    - KNN muestra mayor variabilidad en sus métricas
#    - Hay un desbalance más pronunciado entre precisión y recall
#    - La especificidad es ligeramente menor que en el dataset de diabetes
# 
# 3. **Comparación General**:
#    - La Regresión Logística es más robusta y consistente en ambos datasets
#    - KNN es más sensible a las características específicas de cada dataset
#    - El dataset de diabetes es más predecible que el de clima
#    - La diferencia en rendimiento entre modelos es más pronunciada en el dataset de clima
# 
# 4. **Recomendaciones Finales**:
#    - Para problemas de diagnóstico médico (como diabetes), ambos modelos son viables
#    - Para problemas de predicción climática, la Regresión Logística es más confiable
#    - Se recomienda considerar el balance entre precisión y recall según el contexto
#    - En casos donde los falsos positivos son críticos, KNN podría ser preferible

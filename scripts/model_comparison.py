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
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
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
# ## 5. Análisis de Matrices de Confusión

# %%
def plot_confusion_matrices(results, dataset_name):
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
# ## 6. Análisis Comparativo

# %%
def print_comparative_analysis(diabetes_results, weather_results):
    print("Análisis Comparativo de Modelos\n")
    
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

# Imprimir análisis comparativo
print_comparative_analysis(diabetes_results, weather_results)

# %% [markdown]
# ## 7. Conclusiones
# 
# Basado en el análisis comparativo realizado, se pueden extraer las siguientes conclusiones:
# 
# 1. Rendimiento General:
#    - La Regresión Logística muestra un rendimiento más consistente entre ambos datasets
#    - KNN presenta variaciones más significativas en su rendimiento
# 
# 2. Precisión vs Recall:
#    - En el dataset de diabetes, se observa un mejor balance entre precisión y recall
#    - En el dataset de clima, hay una tendencia hacia mayor precisión a costa de recall
# 
# 3. Robustez:
#    - La Regresión Logística demuestra mayor robustez en diferentes tipos de datos
#    - KNN muestra mayor sensibilidad a las características específicas de cada dataset
# 
# 4. Recomendaciones:
#    - Para problemas similares al dataset de diabetes, ambos modelos son viables
#    - Para problemas similares al dataset de clima, la Regresión Logística es más recomendable 
# %% [markdown]
# # Análisis y Modelado de Datos Climáticos
# 
# Este notebook tiene como objetivo analizar y modelar el conjunto de datos climáticos de Australia utilizando técnicas de Machine Learning.
# Se implementarán dos modelos: Regresión Logística y KNN, para predecir si lloverá mañana basado en las condiciones climáticas actuales.

# %% [markdown]
# ## 1. Importación de Bibliotecas

# %%
import numpy as np
import pandas as pd
import os
import sys

# Obtener la ruta absoluta del directorio del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
sns.set()

# %% [markdown]
# ## 2. Carga y Exploración de Datos

# %%
# Cargar el conjunto de datos climáticos
weather_data = pd.read_csv(os.path.join(project_root, 'data', 'weather-aus.csv'))

# Mostrar las primeras filas del dataset
print("\nPrimeras filas del dataset:")
print(weather_data.head())

# Información general del dataset
print("\nInformación del dataset:")
print(weather_data.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(weather_data.describe().T)

# %% [markdown]
# ## 3. Preprocesamiento de Datos

# %%
def preprocess_weather_data(data):
    # Eliminar columnas con muchos valores faltantes
    threshold = len(data) * 0.3  # 30% de valores faltantes
    data = data.dropna(thresh=threshold, axis=1)
    
    # Eliminar filas con valores faltantes
    data = data.dropna()
    
    # Codificar variables categóricas
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    return data

# Aplicar preprocesamiento
processed_data = preprocess_weather_data(weather_data)

# %% [markdown]
# ## 4. División de Datos

# %%
def split_data(data, target_col='RainTomorrow'):
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # Primera división: 70% entrenamiento, 30% temporal
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Segunda división: 15% validación, 15% prueba
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Dividir los datos
X_train, X_val, X_test, y_train, y_val, y_test = split_data(processed_data)

# %% [markdown]
# ## 5. Escalado de Características

# %%
def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled

# Escalar los datos
X_train_scaled, X_val_scaled, X_test_scaled = scale_data(X_train, X_val, X_test)

# %% [markdown]
# ## 6. Optimización de Hiperparámetros

# %%
def optimize_hyperparameters(X_train, y_train):
    # Parámetros para Regresión Logística
    lr_params = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    # Parámetros para KNN
    knn_params = {
        'n_neighbors': range(1, 31),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # Optimización para Regresión Logística
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    lr_grid.fit(X_train, y_train)
    
    # Optimización para KNN
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    knn_grid.fit(X_train, y_train)
    
    return lr_grid.best_estimator_, knn_grid.best_estimator_

# Optimizar hiperparámetros
best_lr, best_knn = optimize_hyperparameters(X_train_scaled, y_train)

# Mostrar mejores parámetros
print("Mejores parámetros para Regresión Logística:")
print(best_lr.get_params())
print("\nMejores parámetros para KNN:")
print(best_knn.get_params())

# %% [markdown]
# ## 7. Validación Cruzada

# %%
def perform_cross_validation(X, y, models, cv=5):
    # Configurar validación cruzada estratificada
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = {}
    for name, model in models.items():
        # Calcular scores para diferentes métricas
        accuracy_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='recall')
        
        results[name] = {
            'accuracy': {
                'mean': np.mean(accuracy_scores),
                'std': np.std(accuracy_scores)
            },
            'precision': {
                'mean': np.mean(precision_scores),
                'std': np.std(precision_scores)
            },
            'recall': {
                'mean': np.mean(recall_scores),
                'std': np.std(recall_scores)
            }
        }
    
    return results

# Definir modelos para validación cruzada
models = {
    'Regresión Logística': best_lr,
    'KNN': best_knn
}

# Realizar validación cruzada
cv_results = perform_cross_validation(X_train_scaled, y_train, models)

# Mostrar resultados de validación cruzada
for model_name, metrics in cv_results.items():
    print(f"\nResultados de Validación Cruzada para {model_name}:")
    for metric_name, values in metrics.items():
        print(f"{metric_name.capitalize()}: {values['mean']:.4f} ± {values['std']:.4f}")

# %% [markdown]
# ## 8. Entrenamiento y Evaluación de Modelos

# %%
def train_and_evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test):
    # Entrenar modelos con mejores hiperparámetros
    lr_model = best_lr
    knn_model = best_knn
    
    # Evaluar modelos
    models = {
        'Regresión Logística': lr_model,
        'KNN': knn_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
    
    return results

# Entrenar y evaluar modelos
results = train_and_evaluate_models(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

# %% [markdown]
# ## 9. Visualización de Resultados

# %%
def plot_results(results):
    # Imprimir resultados numéricos detallados
    print("\nResultados Detallados de los Modelos")
    print("=" * 80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        print(f"Exactitud: {metrics['accuracy']:.4f}")
        print(f"Precisión: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        # Análisis de matriz de confusión
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        total = tn + fp + fn + tp
        
        print("\nMatriz de Confusión:")
        print(f"Verdaderos Negativos: {tn} ({tn/total:.2%})")
        print(f"Falsos Positivos:    {fp} ({fp/total:.2%})")
        print(f"Falsos Negativos:    {fn} ({fn/total:.2%})")
        print(f"Verdaderos Positivos: {tp} ({tp/total:.2%})")
        
        # Métricas adicionales
        specificity = tn / (tn + fp)
        f1_score = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        
        print("\nMétricas Adicionales:")
        print(f"Especificidad: {specificity:.4f}")
        print(f"F1-Score: {f1_score:.4f}")
        
        print("\nReporte de Clasificación:")
        print(metrics['classification_report'])
    
    # Crear visualización
    metrics = ['accuracy', 'precision', 'recall']
    models = list(results.keys())
    
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        values = [results[model][metric] for model in models]
        plt.bar(models, values)
        plt.title(f'Comparación de {metric.capitalize()}')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

# Mostrar resultados
plot_results(results)

# %% [markdown]
# ## 10. Análisis de Importancia de Características

# %%
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'coef_'):
        # Para Regresión Logística
        importance = np.abs(model.coef_[0])
    else:
        # Para KNN (usando una métrica de distancia promedio)
        importance = np.zeros(len(feature_names))
        for i in range(len(feature_names)):
            temp_data = X_train_scaled.copy()
            np.random.shuffle(temp_data[:, i])
            importance[i] = accuracy_score(y_train, model.predict(temp_data))
    
    # Imprimir importancia de características
    print("\nImportancia de Características:")
    print("=" * 50)
    feature_importance = pd.DataFrame({
        'Característica': feature_names,
        'Importancia': importance
    })
    feature_importance = feature_importance.sort_values('Importancia', ascending=False)
    print(feature_importance.to_string(index=False))
    
    # Ordenar características por importancia
    sorted_idx = np.argsort(importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    
    plt.figure(figsize=(10, 6))
    plt.barh(pos, importance[sorted_idx])
    plt.yticks(pos, feature_names[sorted_idx])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características')
    plt.show()

# Graficar importancia de características para Regresión Logística
plot_feature_importance(best_lr, X_train.columns)

# %% [markdown]
# ## 11. Conclusiones
# 
# Basado en los resultados obtenidos, se pueden extraer las siguientes conclusiones:
# 
# 1. **Rendimiento General de los Modelos**:
#    - La Regresión Logística muestra un rendimiento ligeramente superior (85.43% de exactitud vs 85.01% de KNN)
#    - La Regresión Logística tiene mejor balance entre precisión (74.36%) y recall (53.68%)
#    - KNN muestra una precisión similar (73.92%) pero un recall más bajo (51.42%)
#    - La validación cruzada confirma la estabilidad de ambos modelos (Accuracy: 85.32% ± 0.31% para RL)
# 
# 2. **Análisis de Matrices de Confusión**:
#    - Ambos modelos tienen una alta tasa de verdaderos negativos (73.37% RL, 73.45% KNN)
#    - La Regresión Logística tiene menos falsos negativos (10.41% vs 10.92%)
#    - La tasa de falsos positivos es similar en ambos modelos (4.16% RL, 4.08% KNN)
#    - La Regresión Logística tiene más verdaderos positivos (12.06% vs 11.56%)
# 
# 3. **Importancia de Características**:
#    - La presión atmosférica (3pm) es la característica más importante (1.40)
#    - La humedad (3pm) y presión (9am) tienen un impacto significativo (1.17 y 1.01)
#    - La velocidad del viento y la insolación tienen un impacto moderado (0.78 y 0.55)
#    - Las características temporales (fecha) y de ubicación tienen la menor influencia (<0.03)
#    - La temperatura y la dirección del viento tienen un impacto relativamente bajo
# 
# 4. **Recomendaciones**:
#    - Para predicción climática, la Regresión Logística es preferible por su mejor balance entre precisión y recall
#    - Se recomienda enfocarse en la presión atmosférica y humedad como indicadores principales
#    - Considerar la velocidad del viento y la insolación como factores secundarios importantes
#    - Las características temporales y de ubicación podrían ser menos relevantes en el modelo
#    - El modelo podría beneficiarse de más datos para mejorar el recall, especialmente en días de lluvia 
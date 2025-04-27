# %% [markdown]
# # Análisis y Modelado de Datos de Diabetes
# 
# Este notebook tiene como objetivo analizar y modelar el conjunto de datos de diabetes utilizando técnicas de Machine Learning.
# Se implementarán dos modelos: Regresión Logística y KNN, para predecir si un paciente tiene diabetes basado en sus características médicas.

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

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
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
# Cargar el conjunto de datos de diabetes
diabetes_data = pd.read_csv(os.path.join(project_root, 'data', 'diabetes.csv'))

# Mostrar las primeras filas del dataset
print("\nPrimeras filas del dataset:")
print(diabetes_data.head())

# Información general del dataset
print("\nInformación del dataset:")
print(diabetes_data.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(diabetes_data.describe().T)

# %% [markdown]
# ## 3. Preprocesamiento de Datos

# %%
def preprocess_diabetes_data(data):
    # Reemplazar ceros en columnas donde no tiene sentido
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in columns_to_replace:
        data[col] = data[col].replace(0, data[col].mean())
    
    return data

# Aplicar preprocesamiento
processed_data = preprocess_diabetes_data(diabetes_data)

# %% [markdown]
# ## 4. División de Datos

# %%
def split_data(data, target_col='Outcome'):
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    # División en training (80%) y testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Dividir los datos
X_train, X_test, y_train, y_test = split_data(processed_data)

# %% [markdown]
# ## 5. Escalado de Características

# %%
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Escalar los datos
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

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
        n_jobs=-1,
        return_train_score=True,  # Para obtener las puntuaciones de entrenamiento
        verbose=0
    )
    lr_grid.fit(X_train, y_train)
    
    # Optimización para KNN
    knn_grid = GridSearchCV(
        KNeighborsClassifier(),
        knn_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        return_train_score=True,  # Para obtener las puntuaciones de entrenamiento
        verbose=0
    )
    knn_grid.fit(X_train, y_train)
    
    # Registrar hiperparámetros descartados (selección de N combinaciones)
    log_hyperparameter_selection(lr_grid, "Regresión Logística", sample_size=5)
    log_hyperparameter_selection(knn_grid, "KNN", sample_size=8)
    
    return lr_grid.best_estimator_, knn_grid.best_estimator_

def log_hyperparameter_selection(grid_search, model_name, sample_size=5):
    """
    Registra una muestra de los hiperparámetros probados y descartados durante la búsqueda de GridSearch.
    
    Args:
        grid_search: Objeto GridSearchCV después de ajustar
        model_name: Nombre del modelo para el registro
        sample_size: Número de combinaciones a mostrar (además del mejor)
    """
    # Convertir resultados a DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Ordenar por rendimiento (de peor a mejor)
    results = results.sort_values('mean_test_score')
    
    # Mejor conjunto de parámetros
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Calcular el número total de combinaciones
    total_combinations = len(results)
    
    print(f"\n{'='*20} Registro de Selección de Hiperparámetros - {model_name} {'='*20}")
    print(f"Total de combinaciones evaluadas: {total_combinations}")
    print(f"Mejor combinación: {best_params} (score: {best_score:.4f})")
    
    # Seleccionar algunas combinaciones distribuidas (incluyendo la peor)
    if total_combinations <= sample_size:
        sample_indices = list(range(total_combinations))
    else:
        # Siempre incluir el peor conjunto
        sample_indices = [0]
        
        # Agregar algunos conjuntos intermedios
        step = (total_combinations - 2) // (sample_size - 1)
        sample_indices.extend(range(1, total_combinations-1, step))
    
    # Registrar las combinaciones seleccionadas
    print("\nMuestra de combinaciones descartadas:")
    print(f"{'Params':<50} | {'Test Score':<10} | {'Train Score':<10} | {'Diferencia':<10} | {'Razón de descarte'}")
    print("-" * 100)
    
    for idx in sample_indices:
        row = results.iloc[idx]
        params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_') and not pd.isna(v)}
        test_score = row['mean_test_score']
        train_score = row['mean_train_score']
        diff = train_score - test_score
        
        # Determinar la razón del descarte
        if diff > 0.1:
            reason = "Posible sobreajuste"
        elif test_score < best_score - 0.1:
            reason = "Rendimiento bajo"
        elif test_score < best_score - 0.05:
            reason = "Rendimiento moderado"
        else:
            reason = "Subóptimo"
        
        params_str = str(params)
        if len(params_str) > 48:
            params_str = params_str[:45] + "..."
        
        print(f"{params_str:<50} | {test_score:.4f}    | {train_score:.4f}    | {diff:.4f}     | {reason}")
    
    # Guardar los resultados en un archivo CSV para referencia futura
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hyperparameter_selection_{model_name.replace(' ', '_')}_{timestamp}.csv"
    filepath = os.path.join(project_root, 'logs', filename)
    
    # Asegurar que el directorio logs exista
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    
    # Guardar todos los resultados (no solo la muestra)
    results.to_csv(filepath, index=False)
    print(f"\nResultados completos guardados en: {filepath}")

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
def train_and_evaluate_models(X_train, X_test, y_train, y_test):
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
results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)

# %% [markdown]
# ## 9. Curvas de Aprendizaje y Comportamiento

# %%
def plot_learning_curves(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.title(f'Curva de Aprendizaje - {title}')
    plt.xlabel('Tamaño del Conjunto de Entrenamiento')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    # Imprimir resultados numéricos
    print(f"\nResultados de Curva de Aprendizaje - {title}:")
    print("=" * 50)
    for i, size in enumerate(train_sizes):
        print(f"\nTamaño de Entrenamiento: {int(size)} muestras")
        print(f"Training Score: {train_mean[i]:.4f} ± {train_std[i]:.4f}")
        print(f"CV Score: {test_mean[i]:.4f} ± {test_std[i]:.4f}")

# Graficar curvas de aprendizaje para ambos modelos
plot_learning_curves(best_lr, X_train_scaled, y_train, 'Regresión Logística')
plot_learning_curves(best_knn, X_train_scaled, y_train, 'KNN')

# %% [markdown]
# ## 10. Visualización de Resultados

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
# ## 11. Análisis de Importancia de Características

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
# ## 12. Conclusiones
# 
# Basado en los resultados obtenidos, se pueden extraer las siguientes conclusiones:
# 
# 1. **Rendimiento General de los Modelos**:
#    - La Regresión Logística muestra un rendimiento ligeramente superior (75% de exactitud vs 72.41% de KNN)
#    - La Regresión Logística tiene mejor balance entre precisión (68.42%) y recall (60.47%)
#    - KNN muestra una precisión similar (65.71%) pero un recall más bajo (53.49%)
#    - La validación cruzada confirma la estabilidad de ambos modelos (Accuracy: 76.90% ± 4.20% para RL)
# 
# 2. **Análisis de Matrices de Confusión**:
#    - Ambos modelos tienen una alta tasa de verdaderos negativos (52.59%)
#    - La Regresión Logística tiene menos falsos negativos (14.66% vs 17.24%)
#    - La tasa de falsos positivos es igual en ambos modelos (10.34%)
#    - La Regresión Logística tiene más verdaderos positivos (22.41% vs 19.83%)
# 
# 3. **Importancia de Características**:
#    - La glucosa es la característica más importante (1.22), seguida del BMI (0.80)
#    - La edad tiene un impacto moderado (0.39)
#    - El número de embarazos tiene una influencia baja (0.20)
#    - La presión sanguínea y la función de pedigrí tienen un impacto similar (0.13 y 0.12)
#    - La insulina y el grosor de la piel tienen la menor influencia (0.10 y 0.03)
# 
# 4. **Recomendaciones**:
#    - Para diagnóstico médico, la Regresión Logística es preferible por su mejor balance entre precisión y recall
#    - Se recomienda enfocarse en los niveles de glucosa y BMI como indicadores principales
#    - Considerar la edad como un factor de riesgo moderado
#    - La insulina y el grosor de la piel podrían ser menos relevantes en el diagnóstico
#    - El modelo podría beneficiarse de más datos para mejorar el recall, especialmente en casos positivos 
# %%

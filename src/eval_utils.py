'''
src/eval_utils.py

Funciones auxiliares para el notebook de Evaluacion de Modelos
'''
import numpy as np
import pandas as pd
import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.textinput=false"
import matplotlib.pyplot as plt
from typing import Any
import shap

from sklearn.model_selection import (
    GridSearchCV, TimeSeriesSplit, cross_validate,
    validation_curve
    )

from src.modeling import (
    preprocesar_datos,
    construir_regression_pipe
)

# Optimizar hiper-parametros

# Grilla de parametros por modelo
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "rf": {
        "model__n_estimators": [400],
        "model__max_depth": [4],
        "model__min_samples_leaf": [15, 20, 25],
        "model__min_samples_split": [45, 50, 55],  # >= 2 * min_samples_leaf
        "model__max_samples": [0.4],
        "model__max_features": [0.45, 0.5],
    },
    "et": {
        "model__n_estimators": [400],
        "model__max_depth": [4, 6, 8],
        "model__min_samples_leaf": [20, 25, 30],
        "model__min_samples_split": [50, 60, 70],
        "model__bootstrap": [True],  # Requerido en ExtraTrees si se usa max_samples
        "model__max_samples": [0.4, 0.5, 0.6],
        "model__max_features": [0.4, 0.5, 0.6],
    },
    "gbr": {
        "model__n_estimators": [100, 200, 300],
        "model__learning_rate": [0.01, 0.03, 0.05],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [20, 30],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__max_features": [0.4, 0.6, 1.0],
    },
    "ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        "model__solver": ["auto", "svd", "lsqr"],
    },
    "elasticnet": {
        "model__alpha": np.logspace(-3, 0, 10).tolist(),  # 0.001 a 1.0
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "model__max_iter": [5000],
    },
}

def optimizar(
        modelo:str = "rf",
        n_splits:int=5
):
    if modelo not in PARAM_GRIDS:
        valid = ", ".join(PARAM_GRIDS.keys())
        raise ValueError(f"No hay param_grid configurado para '{modelo}'. Opciones: {valid}")
    
    X_train, y_train, _, _, _ = preprocesar_datos()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    print("Configurando GridSearchCV")
    modelo_base = construir_regression_pipe(
        X_sample=X_train,
        model_name=modelo
        )

    param_grid = PARAM_GRIDS[modelo]

    # Configurar el GridSearchCV
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        verbose=0,
        refit=False # No es necesario reentrenar con el mejor modelo
    )

    # Ajusta probando hiperparámetros entrenando en train y testeando en test
    grid_search.fit(X_train, y_train)

    # Resultados
    clean_params = {
        k.replace('model__', ''): v
        for k, v in grid_search.best_params_.items()
    }
    print("\n--- Mejores hiper-parámetros ---")
    for k, v in clean_params.items():
        print(f"    '{k}': {repr(v)},")

    # Mostrar Top 5 combinaciones para evaluar estabilidad entre folds
    results_df = (
        pd.DataFrame(grid_search.cv_results_)
        .sort_values('rank_test_score')
        [['rank_test_score', 'mean_test_score', 'std_test_score', 'params']]
        .head(5)
    )
    print("\n--- Top 5 Combinaciones ---")
    print(results_df.to_string(index=False))


# Curva de complejidad

# Configuración de grilla por modelo
COMPLEXITY_CONFIG: dict[str, dict] = {
    "rf": {
        "complexity_var": "max_depth",
        "complexity_range": np.arange(1, 15),
        "xlabel": "max_depth (mayor profundidad = mayor complejidad)",
        "log_scale": False,
        "invert_xaxis": False,
    },
    "et": {
        "complexity_var": "max_depth",
        "complexity_range": np.arange(1, 15),
        "xlabel": "max_depth (mayor profundidad = mayor complejidad)",
        "log_scale": False,
        "invert_xaxis": False,
    },
    "gbr": {
        "complexity_var": "n_estimators",
        "complexity_range": np.arange(20, 420, 20),
        "xlabel": "n_estimators (más árboles = mayor complejidad)",
        "log_scale": False,
        "invert_xaxis": False,
    },
    "ridge": {
        "complexity_var": "alpha",
        "complexity_range": np.logspace(-3, 3.8, 35), # desde 0.001 hasta ~6300
        "xlabel": "alpha (escala log — menor alpha = mayor complejidad)",
        "log_scale": True,
        "invert_xaxis": True,
    },
    "elasticnet": {
        "complexity_var": "alpha",
        "complexity_range": np.logspace(-4, 1, 20),
        "xlabel": "alpha (escala log — menor alpha = mayor complejidad)",
        "log_scale": True,
        "invert_xaxis": True,
    },
}

def curva(
        modelo:str = "rf",
        n_splits:int=5
):
        '''
        Grafica la evolución del desempeño (-RMSE) del modelo al aumentar la complejidad.
        Identifica el trade-off entre Sesgo y Varianza (valores más altos / cercanos a 0 = mejor):
        - Alto sesgo (underfitting): Métricas de train y test bajas (muy negativas) y cercanas entre sí.
        - Zona de equilibrio: Punto donde la métrica de test alcanza su máximo antes de degradarse.
        - Alta varianza (overfitting): La métrica de train sigue subiendo hacia 0, mientras que la de
        test comienza a caer y se abre una brecha (generalization gap).
        '''
        if modelo not in COMPLEXITY_CONFIG:
            raise ValueError(f"No hay configuración de curva de complejidad para '{modelo}'.")
        
        cfg = COMPLEXITY_CONFIG[modelo]
        complexity_var = cfg["complexity_var"]
        complexity_range = cfg["complexity_range"]
        
        X_train, y_train, _, _, _ = preprocesar_datos()

        train_scores, test_scores = [], []

        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Instanciar el pipeline
        pipe = construir_regression_pipe(
            X_sample=X_train, 
            model_name=modelo
        )

        # Detectar el nombre del parametro dentro del pipeline
        if hasattr(pipe, "steps"):
            last_step_name = pipe.steps[-1][0]
            param_name = f"{last_step_name}__{complexity_var}"

            # Evitar oversubscription de CPU si el estimador ya tiene n_jobs=-1
            estimator = pipe.steps[-1][0]
            if hasattr(estimator, "n_jobs") and estimator.n_jobs == -1:
                pipe.set_params(**{f"{last_step_name}__n_jobs": 1})

        else:
            param_name = complexity_var
            if hasattr(pipe, "n_jobs") and pipe.n_jobs == -1:
                pipe.set_params(n_jobs=1)

        # Ejecutar la curva
        train_scores, test_scores = validation_curve(
            estimator=pipe,
            X=X_train,
            y=y_train,
            param_name=param_name,
            param_range=complexity_range,
            cv=tscv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)

        # Graficar con Matplotlib
        plt.figure(figsize=(8, 4.5))
        plt.plot(complexity_range, train_mean, marker="o", label="Train -RMSE (Aproximación al sesgo)")
        plt.plot(complexity_range, test_mean, marker="s", label="Test CV -RMSE (Generalización)")

        if cfg.get("log_scale"):
            plt.xscale("log")
        if cfg.get("invert_xaxis"):
            plt.gca().invert_xaxis()

        plt.xlabel(cfg["xlabel"])
        plt.ylabel("-RMSE (mas alto = menor error)")
        plt.title(f"Curva de Complejidad modelo {modelo} ({complexity_var})")
        plt.legend()
        plt.grid(True)
        plt.show()


# Explicar predicciones con SHAP

def explicar(
        moneda:str="USD",
        modelo:str = "rf",
        max_display:int=10,
        target_class: int = 1 # Relevante solo si el modelo fuese un clasificador
):
    # Pre-procesar datos y entrenar modelo
    X_train, y_train, X_live, _, ultima_fecha = preprocesar_datos()
    pipe = construir_regression_pipe(
        X_sample=X_train,
        model_name=modelo
    )
    pipe.fit(X_train, y_train)

    # Extraer los componentes del pipeline
    preprocessor = pipe.named_steps['pre']
    model = pipe.named_steps['model']

    # Transformar los datos test usando el preprocesador
    X_train_trans = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_live)
    feature_names = preprocessor.get_feature_names_out()

    # Crear dataframe con los datos transformados para SHAP
    X_test_shap = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_live.index)

    # Obtener el indice de la moneda y extraer su fila
    mask = X_test_shap.index.get_level_values("Currency") == moneda
    if not mask.any():
        raise ValueError(f"La moneda '{moneda}' no se encuentra en X_live.")
    X_ticker_eval = X_test_shap.loc[mask].iloc[[-1]]

    # Inicializar el explicador y calcular los valores SHAP

    # Usar todo el background sin submuestrear
    masker = shap.maskers.Independent(
        data=X_train_trans,
        max_samples=len(X_train_trans),
    )

    explainer = shap.Explainer(
        model, 
        masker=masker, 
        feature_names=feature_names)
    shap_values = explainer(X_ticker_eval)

    # Manejar salida 3D (clasificación) vs 2D (regresión)
    explanation = shap_values[0, :, target_class] if shap_values.values.ndim == 3 else shap_values[0]

    # Crear la figura y visualizar:
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Explicación del modelo {modelo} para la moneda {moneda} (Fecha: {ultima_fecha}):")
    fig.set_size_inches(11, 6)
    fig.subplots_adjust(left=0.40)  # Reserva el 35% izquierdo para los nombres
    plt.show()
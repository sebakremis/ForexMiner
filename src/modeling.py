'''
src/cat_modeling.py

Entrenamiento de modelo de Clasificacion
'''
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from typing import (
   Optional, Any,
   Dict, List,
   Sequence
   )
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score
)
from sklearn.model_selection import (
    GridSearchCV, TimeSeriesSplit, cross_validate
    )
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import CARPETA_DATOS, INTERVALO_DEFECTO, MAP_DIVISAS, DIVISAS


# Funciones de pre-procesamiento

def cargar_datos_clean()->pd.DataFrame:
    fichero_clean = Path(CARPETA_DATOS) / f"clean_data/clean.parquet"
    try:
        df = pd.read_parquet(fichero_clean)
        return df

    except FileNotFoundError:
        print("Error: No se encontro fichero.")
        return pd.DataFrame()   
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame()


def cargar_cols_filtrar()->dict:
    fichero = Path(CARPETA_DATOS) / "cols_filtrar.json"
    try:
        with open(fichero, "r", encoding="utf-8") as file:
            excluir_dict = json.load(file)
            return excluir_dict if isinstance(excluir_dict,dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def wide_to_long_panel(
          df_wide:pd.DataFrame,
          currencies: Optional[list[str]] = None
)->pd.DataFrame:
    """Transforma el DataFrame semanal ancho en un panel largo
    con una fila por (fecha, divisa), alineando features propios de cada divisa,
    spreads de tasas direccionales y variables macro globales.
    """

    if currencies is None:
        currencies = DIVISAS

    # Extraer sufijos base desde "EUR_"
    excluded_suffixes = {"Ret_rank"}
    feature_suffixes = [
         col[4:] # len("EUR_") ==4
         for col in df_wide.columns
         if col.startswith("EUR_") and col [4:] not in excluded_suffixes
    ]

    panel_frames = []

    for ccy in currencies:
        # Target y features
        data = {
              "Currency": ccy,
              "Target_rank": df_wide[f"{ccy}_Ret_rank"],
              **{
                   suf:df_wide.get(f"{ccy}_{suf}", np.nan)
                   for suf in feature_suffixes
              }
         }


        panel_frames.append(pd.DataFrame(data, index=df_wide.index))

    # Apilar y ordenar por Date, Currency
    df_long = pd.concat(panel_frames, axis=0)
    df_long.index.name = "date"
    df_long = (
        df_long.reset_index()
        .sort_values(by=["date", "Currency"])
        .reset_index(drop=True)
    )
    df_long["Currency"] = df_long["Currency"].astype("category")

    return df_long
            

def separar_train_test(
        df:pd.DataFrame,
        fecha:datetime
    )->tuple[pd.DataFrame, pd.DataFrame]:
        '''
        Divide el DataFrame en conjuntos de entrenamiento y prueba basándose en la fecha de corte.
        '''
        df_train = df[df.index.get_level_values("date") < fecha]
        df_test = df[df.index.get_level_values("date") >= fecha].copy()

        return df_train, df_test
    

def separar_predictores_y_objetivo(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_target:Optional[str]=None 
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
        '''

        '''
        if col_target is None:
            col_target = "Target_rank"

        # Predictores a descartar: variables excluidas y labels
        cols_excluded = cargar_cols_filtrar().get(col_target,[])
        columnas_a_descartar = set(cols_excluded+[col_target])
        cols_drop = [c for c in columnas_a_descartar if c in df_train.columns]

        X_train = df_train.drop(columns=cols_drop)
        y_train = df_train[col_target]

        X_test = df_test.drop(columns=cols_drop)
        y_test = df_test[col_target]

        return X_train, y_train, X_test, y_test


def preprocesar_datos(
)->tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, datetime]:
            # Cargar fichero de precios
            df_clean = cargar_datos_clean()

            # Convertir a panel long
            df_panel = wide_to_long_panel(df_clean).set_index(['date', 'Currency'])

            # Separar datos de entrenamiento y datos live
            ultima_fecha = df_panel.index.get_level_values("date").max()
            df_train, df_live = separar_train_test(df=df_panel, fecha=ultima_fecha)

            X_train, y_train, X_live, y_live = separar_predictores_y_objetivo(
                df_train=df_train, 
                df_test=df_live
                )

            return X_train, y_train, X_live, y_live, ultima_fecha.strftime('%Y-%m-%d')     


# Construccion de pipeline

def construir_pipe_rf(
        X_sample: pd.DataFrame,
        cat_cols: list[str]=[],
        **rf_params
)-> Pipeline:
    '''
    Construye un pipeline no entrenado de regresion RandomForest, junto con su preprocesado.
    '''
    num_cols = [
        col for col in X_sample.select_dtypes(include=np.number).columns
        if col not in cat_cols
    ]
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Parametros por defecto
    default_params = {
            'random_state': 42,
            'n_estimators': 400,
            'max_depth': 4,
            'min_samples_leaf': 20,
            'min_samples_split': 50,
            'max_samples': 0.4,
            'max_features': 0.45,
        } 

    # Sobre escribir con los parametros pasados a la funcion
    model_params = {**default_params, **rf_params}
    return Pipeline([
        ('pre', preprocessor),
        ('model', RandomForestRegressor(**model_params))
    ])   


# Funciones de evaluacion

def evaluar_modelo(
        model,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        n_splits:int = 5,
        top_n: int=10
    ) -> Dict[str, Any]:
    """
    Evalúa un modelo regresor con validacion cruzada.
    """
    feature_names = (
        X_train.columns.tolist()
        if hasattr(X_train, "columns")
        else [f"feature_{i}" for i in range(X_train.shape[1])]
    )
    # Predicciones sobre el dataset de entrenamientos completo
    y_pred_train = model.predict(X_train)

    n_samples, n_features = X_train.shape
    r2_train = r2_score(y_train, y_pred_train)
    adj_r2_train = (
        1 - (1 - r2_train) * (n_samples - 1) / (n_samples - n_features - 1)
        if n_samples > n_features+1
        else np.nan
    )
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    # Test de validacion cruzada
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error"
    }

    cv_results = cross_validate(
        estimator=model, 
        X=X_train, 
        y=y_train, 
        cv=tscv,         
        scoring=scoring,
        n_jobs=-1        
    )
    cv_r2_mean, cv_r2_std = cv_results["test_r2"].mean(), cv_results["test_r2"].std()
    cv_rmse_mean, cv_rmse_std = -cv_results["test_rmse"].mean(), cv_results["test_rmse"].std()
    cv_mae_mean, cv_mae_std = -cv_results["test_mae"].mean(), cv_results["test_mae"].std()

    # Brecha Train vs CV
    r2_gap = r2_train - cv_r2_mean
    rmse_ratio = cv_rmse_mean / rmse_train if rmse_train > 0 else np.nan

    # Importancia de variables
    if hasattr(model, "steps"):
        rf = model[-1]
        transformed_names = model[:-1].get_feature_names_out(feature_names)
    else:
        rf = model
        transformed_names = feature_names

    importances = pd.Series(rf.feature_importances_, index=transformed_names).sort_values(ascending=False)

    # Imprimir resumen y top features
    summary = pd.DataFrame(
        {
            "Full Train": [
                f"{r2_train:.4f}",
                f"{adj_r2_train:.4f}",
                f"{rmse_train:.4f}",
                f"{mae_train:.4f}",
            ],
            f"CV ({n_splits} folds)": [
                f"{cv_r2_mean:.4f} ± {cv_r2_std:.4f}",
                "N/A",
                f"{cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}",
                f"{cv_mae_mean:.4f} ± {cv_mae_std:.4f}",
            ],
        },
        index=["R²", "Adjusted R²", "RMSE", "MAE"],
    )
    print(summary.to_string())
    print(f"\nOverfitting Check -> Δ R² (Train - CV): {r2_gap:+.4f} | Ratio RMSE (CV/Train): {rmse_ratio:.2f}x")
    print(f"\nTop {top_n} Features:\n{importances.head(top_n).to_string()}")

    return {
        "train": {
            "r2": r2_train,
            "adj_r2": adj_r2_train,
            "rmse": rmse_train,
            "mae": mae_train,
        },
        "cv": {
            "r2_mean": cv_r2_mean,
            "r2_std": cv_r2_std,
            "rmse_mean": cv_rmse_mean,
            "rmse_std": cv_rmse_std,
            "mae_mean": cv_mae_mean,
            "mae_std": cv_mae_std,
        },
        "overfitting": {
            "r2_gap": r2_gap,
            "rmse_ratio": rmse_ratio,
        },
        "feature_importances": importances,
    }


def registrar_evaluacion_excel(
    dict_evaluacion: Dict[str, Any],
    ultima_fecha: Any,
    modelo: str,
    y_pred: Any,
    filepath: str = "historico_modelos_reg.xlsx",
) -> None:
    """
    Guarda o anexa los resultados de evaluar_modelo (regresión) y sus metadatos
    en un libro de Excel estructurado.
    """
    path = Path(CARPETA_DATOS) / f"evaluacion_modelos/{filepath}"
    run_id = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extraer sub-diccionarios de evaluar_modelo
    train_metrics = dict_evaluacion.get("train", {})
    cv_metrics = dict_evaluacion.get("cv", {})
    overfitting = dict_evaluacion.get("overfitting", {})

    fila_resumen = {
        "run_id": run_id,
        "timestamp_registro": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intervalo": INTERVALO_DEFECTO,
        "modelo": str(modelo),
        "fecha_pred": str(ultima_fecha),
        "y_pred": str(y_pred),
        # Métricas Full Train
        "train_r2": train_metrics.get("r2", np.nan),
        "train_adj_r2": train_metrics.get("adj_r2", np.nan),
        "train_rmse": train_metrics.get("rmse", np.nan),
        "train_mae": train_metrics.get("mae", np.nan),
        # Métricas TimeSeries CV
        "cv_r2_mean": cv_metrics.get("r2_mean", np.nan),
        "cv_r2_std": cv_metrics.get("r2_std", np.nan),
        "cv_rmse_mean": cv_metrics.get("rmse_mean", np.nan),
        "cv_rmse_std": cv_metrics.get("rmse_std", np.nan),
        "cv_mae_mean": cv_metrics.get("mae_mean", np.nan),
        "cv_mae_std": cv_metrics.get("mae_std", np.nan),
        # Diagnóstico de Overfitting
        "r2_gap": overfitting.get("r2_gap", np.nan),
        "rmse_ratio": overfitting.get("rmse_ratio", np.nan),
    }
    df_nueva_fila = pd.DataFrame([fila_resumen])

    # 2. Preparar el DataFrame de Importancia de Features (viene como pd.Series)
    imp_series = dict_evaluacion.get("feature_importances")
    df_feat = None
    if imp_series is not None and len(imp_series) > 0:
        df_feat = imp_series.rename_axis("feature").reset_index(name="importance")
        df_feat.insert(0, "run_id", run_id)
        df_feat["rank"] = range(1, len(df_feat) + 1)

    # 3. Escritura / Concatenación en Excel
    if not path.exists():
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df_nueva_fila.to_excel(
                writer, sheet_name="Registro_Modelos", index=False
            )
            if df_feat is not None:
                df_feat.to_excel(
                    writer, sheet_name="Log_Features", index=False
                )
    else:
        with pd.ExcelWriter(
            path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
        ) as writer:
            # Anexar a Hoja Principal
            try:
                start_row = writer.sheets["Registro_Modelos"].max_row
                df_nueva_fila.to_excel(
                    writer,
                    sheet_name="Registro_Modelos",
                    index=False,
                    header=False,
                    startrow=start_row,
                )
            except KeyError:
                df_nueva_fila.to_excel(
                    writer, sheet_name="Registro_Modelos", index=False
                )

            # Anexar a Log_Features
            if df_feat is not None:
                try:
                    start_row_feat = writer.sheets["Log_Features"].max_row
                    df_feat.to_excel(
                        writer,
                        sheet_name="Log_Features",
                        index=False,
                        header=False,
                        startrow=start_row_feat,
                    )
                except KeyError:
                    df_feat.to_excel(
                        writer, sheet_name="Log_Features", index=False
                    )

    print(f"Registro exitoso para run_id: {run_id} en '{path.name}'")


# Funciones complementarias de evaluacion

def optimizar_rf(
        n_splits:int=5
):
    X_train, y_train, _, _, _ = preprocesar_datos()

    tscv = TimeSeriesSplit(n_splits=n_splits)

    print("Configurando GridSearchCV")
    modelo_base = construir_pipe_rf(X_sample=X_train)

    param_grid = {
        'model__n_estimators': [400],
        'model__max_depth': [4],
        'model__min_samples_leaf': [20,25,30],
        'model__min_samples_split': [50,60,70], # >= 2* min_samples_leaf
        'model__max_samples': [0.35,0.4,0.45],
        'model__max_features': [0.35,0.4,0.45],
    }

    # Configurar el GridSearchCV
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        n_jobs=-1,
        verbose=2,
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


def curva_rf(n_splits:int=5):
        '''
        Grafica la evolución del desempeño (-RMSE) del modelo al aumentar la complejidad.
        Identifica el trade-off entre Sesgo y Varianza (valores más altos / cercanos a 0 = mejor):
        - Alto sesgo (underfitting): Métricas de train y test bajas (muy negativas) y cercanas entre sí.
        - Zona de equilibrio: Punto donde la métrica de test alcanza su máximo antes de degradarse.
        - Alta varianza (overfitting): La métrica de train sigue subiendo hacia 0, mientras que la de
        test comienza a caer y se abre una brecha (generalization gap).
        '''
        X_train, y_train, _, _, _ = preprocesar_datos()

        complexity_var = 'max_depth'
        complexity_range = list(range(1, 10, 1))
        train_scores, test_scores = [], []

        tscv = TimeSeriesSplit(n_splits=n_splits)

        for value in complexity_range:
            # Modelo completo:
            pipe_rf = construir_pipe_rf(X_sample=X_train, max_depth=value)

            # Test de validacion cruzada
            cv_results = cross_validate(
                estimator = pipe_rf,
                X=X_train,
                y=y_train,
                cv=tscv,
                scoring='neg_root_mean_squared_error',
                return_train_score=True,
                n_jobs=-1
            )
            
            # Metricas sobre entrenamiento y test
            train_scores.append(cv_results['train_score'].mean())
            test_scores.append(cv_results['test_score'].mean())


        plt.figure(figsize=(8, 4.5))
        plt.plot(complexity_range, train_scores, marker='o', label="Train -RMSE (Aproximación al sesgo)")
        plt.plot(complexity_range, test_scores, marker='s', label="Test CV -RMSE (Generalización)")
        
        # Invertir la dirección del eje X (para Min Samples leaf):
        #plt.gca().invert_xaxis()
        
        plt.xlabel(complexity_var)
        plt.ylabel("-RMSE (mas alto = menor error)")
        plt.title(f"Curva de Complejidad ({complexity_var})")
        plt.legend()
        plt.grid(True)
        plt.show()


'''
Bloque principal

Ejecucion desde el terminal:

# Obtener predicciones sin evaluar el modelo:
python -m src.modeling

# Obtener predicciones y realizar evaluacion:
python -m src.modeling --evaluar
'''
def main(
        nombre_modelo:str | None = None,
        evaluar:bool=False
):
    # Preprocesar datos clean, separando variables y datos "live"
    X_train, y_train, X_live, _, ultima_fecha = preprocesar_datos()

    # Inicializar modelo y entrenar
    if nombre_modelo is None:
        nombre_modelo = "rf"

    if nombre_modelo == "rf":
        pipe = construir_pipe_rf(X_sample=X_train)
        pipe.fit(X_train, y_train)

    # Realizar predicciones
    y_live_pred = pipe.predict(X_live)
    resultados = pd.DataFrame({
        'Currency': X_live.index.get_level_values("Currency").values,
        'Prediction': y_live_pred
    })
    resultados = resultados.sort_values(by='Prediction', ascending=False).to_string(index=False)

    print(f"\nEntrenamiento del modelo {nombre_modelo} finalizado.")
    print(f"Dimensiones del dataset de entrenamiento:", X_train.shape)
    print(f"\nPredicciones para la semana del {ultima_fecha}:")
    print(resultados)

    # Evaluar modelo
    if evaluar:
        print(f"\nEvaluación del modelo {nombre_modelo}:")
        dict_evaluacion = evaluar_modelo(
            model=pipe, 
            X_train=X_train, 
            y_train=y_train
            )

        registrar_evaluacion_excel(
            dict_evaluacion=dict_evaluacion,
            ultima_fecha=ultima_fecha,
            modelo = nombre_modelo,
            y_pred=resultados # prediccion del modelo completo sobre la ultima fecha
        )


def entrypoint_modeling(argv: Sequence[str] | None = None):
    '''
    Punto de entrada de la linea de comandos.
    '''
    parser = argparse.ArgumentParser(description="Módulo de Modelado")
    parser.add_argument(
         "--modelo",
         type=str,
         help="Nombre del modelo"
    )
    parser.add_argument(
        "--evaluar",
        action="store_true",
        help="Ejecuta la evaluacion del modelo tras el entrenamiento"
    )
    args = parser.parse_args(argv)

    return main(nombre_modelo=args.modelo, evaluar=args.evaluar)


if __name__ == '__main__':
    entrypoint_modeling()

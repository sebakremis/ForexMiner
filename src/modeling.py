'''
src/cat_modeling.py

Entrenamiento de modelo de Clasificacion
'''

# Manejo de datos
import pandas as pd
import numpy as np
import datetime as dt
from pathlib import Path
from typing import (
   Optional, Any,
   Dict, Sequence
)
import argparse
from datetime import datetime
import json
from src.config import CARPETA_DATOS, INTERVALO_DEFECTO, DIVISAS

# Modelos
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,  
)
from sklearn.linear_model import (
    ElasticNet,
    Ridge
)

# Librerias comunes de sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score
)
from sklearn.model_selection import (
    GridSearchCV, TimeSeriesSplit, cross_validate
    )
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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

# Registro de parametros por modelo: model_key -> (EstimatorClass, default_params, requires_scaling)
MODEL_REGISTRY: dict[str, tuple[type, dict[str, Any], bool]] = {
    "rf": (
        RandomForestRegressor,
        {
            "random_state": 42,
            "n_estimators": 400,
            "max_depth": 4,
            "min_samples_leaf": 15,
            "min_samples_split": 45,
            "max_samples": 0.4,
            "max_features": 0.5,
        },
        False,
    ),
    "et": (
        ExtraTreesRegressor,
        {
            "random_state": 42,
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 20,
            "max_features": 0.5,
        },
        False,
    ),
    "gbr": (
        GradientBoostingRegressor,
        {
            "random_state": 42,
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 1,
            "subsample": 0.8,
        },
        False,
    ),
    #Modelos lineales: requieren escalado, pero las features ya fueron estandarizadas
    #mediante los z-score de ventanas moviles, evitando el Lookahead Bias que introduciria
    #el uso de StandardScaler en series temporales. Por esto los seteo en "False", aunque 
    #dejo la posibilidad de usarlo.
    "ridge": (
        Ridge,
        {
            "alpha": 10.0,
            "random_state": 42,
            "solver": "auto",
        },
        False,
    ),
    "elasticnet": (
        ElasticNet,
        {
            "alpha": 0.1,
            "l1_ratio": 0.5,
            "random_state": 42,
        },
        False,
    ),
}


def construir_regression_pipe(
        X_sample: pd.DataFrame,
        model_name:str = "rf",
        cat_cols: list[str] | None = None,
        scale_num: bool | None = None,
        **model_params: Any
)-> Pipeline:
    '''
    Construye un pipeline no entrenado de regresion, con preprocesado
    comun y parametros especificos para cada modelo.
    '''
    # Validacion
    if model_name not in MODEL_REGISTRY:
        valid = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Modelo desconocido: {model_name}. Opciones disponibles: {valid}")

    # Obtener parametros y separar variables
    estimator_cls, default_params, default_scale = MODEL_REGISTRY[model_name]
    use_scaler = default_scale if scale_num is None else scale_num
    cat_cols = cat_cols or []
    num_cols = [
        col for col in X_sample.select_dtypes(include=np.number).columns
        if col not in cat_cols
    ]   
    num_transformer = StandardScaler() if use_scaler else "passthrough"

    # Sobre-escribir parametros por defecto
    final_params = {**default_params, **model_params}

    # Preprocesado
    preprocessor = ColumnTransformer([
                        ('num', num_transformer, num_cols),
                        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
                    ],remainder="drop",
    )

    # Devolver el pipe sin entrenar
    return Pipeline([
        ('pre', preprocessor),
        ('model', estimator_cls(**final_params))
    ])   


# Funciones de evaluacion

def _extract_feature_importance(
    estimator: Any,
    feature_names: list[str] | np.ndarray,
) -> pd.Series | None:
    """
    Extrae importancia de variables de forma agnóstica al tipo de estimador:
    - Modelos basados en árboles: `feature_importances_`
    - Modelos lineales (Ridge, ElasticNet, Lasso): `|coef_|` (promediado si es multi-output)
    - Wrappers MultiOutputRegressor: promedio sobre `estimators_`
    """
    if hasattr(estimator, "feature_importances_"):
        raw_imp = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        # Si es multi-output (n_targets, n_features), promediar magnitud absoluta por feature
        raw_imp = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
    elif hasattr(estimator, "estimators_"):
        sub_imps = []
        for sub_est in estimator.estimators_:
            if hasattr(sub_est, "feature_importances_"):
                sub_imps.append(np.asarray(sub_est.feature_importances_))
            elif hasattr(sub_est, "coef_"):
                sub_imps.append(np.abs(np.asarray(sub_est.coef_)).ravel())
        if not sub_imps:
            return None
        raw_imp = np.mean(sub_imps, axis=0)
    else:
        return None

    return pd.Series(raw_imp, index=feature_names).sort_values(ascending=False)


def evaluar_modelo(
        model:Any,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame | pd.Series,
        n_splits:int = 5,
        top_n: int=10
    ) -> Dict[str, Any]:
    """
    Evalúa un modelo regresor (árboles o lineal) con validación cruzada temporal.
    """
    feature_names = (
        X_train.columns.tolist()
        if hasattr(X_train, "columns")
        else [f"feature_{i}" for i in range(X_train.shape[1])]
    )
    # Extraer estimador final y nombres post-preprocesamiento
    if hasattr(model, "steps"):
        estimator = model[-1]
        transformed_names = model[:-1].get_feature_names_out(feature_names)
        n_features = len(transformed_names)
    else:
        estimator = model
        transformed_names = feature_names
        n_features = X_train.shape[1]

    # Predicciones sobre el dataset de entrenamientos completo
    y_pred_train = model.predict(X_train)

    # Metricas de entrenamiento
    n_samples = X_train.shape[0]
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

    # Importancia de variables (soporta árboles y coeficientes lineales)
    importances = _extract_feature_importance(estimator, transformed_names)
    imp_label = (
        "|Coeficientes|" if hasattr(estimator, "coef_") else "Feature Importances"
    )

    # Imprimir resumen 
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

    # Imprimir top features
    if importances is not None:
        print(f"\nTop {top_n} Features ({imp_label}):\n{importances.head(top_n).to_string()}")
    else:
        print(f"\nTop {top_n} Features: no disponible nativamente para {type(estimator).__name__}.")

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


'''
Bloque principal

Ejecucion desde el terminal:

# Obtener predicciones sin evaluar el modelo (RandomForest por defecto):
python -m src.modeling

# Obtener predicciones y realizar evaluacion:
python -m src.modeling --evaluar
'''
def main(
        nombre_modelo:str,
        evaluar:bool=False,             
):
    # Preprocesar datos clean, separando variables y datos "live"
    X_train, y_train, X_live, _, ultima_fecha = preprocesar_datos()

    # Inicializar modelo y entrenar
    pipe = construir_regression_pipe(
        X_sample=X_train,
        model_name=nombre_modelo
    )
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
         default="rf",
         choices=list(MODEL_REGISTRY.keys()),
         help="Modelo a entrenar (por defecto: rf)"
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

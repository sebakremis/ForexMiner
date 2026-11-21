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
   Dict, List
   )
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
    df_wide: pd.DataFrame,
    currencies: Optional[List[str]] = None,
    include_macro: bool = True,
) -> pd.DataFrame:
  """Transforma el DataFrame semanal ancho en un panel largo (6370 filas)

  con una fila por (fecha, divisa), alineando features propios de cada divisa,
  spreads de tasas direccionales y variables macro globales.
  """
  if currencies is None:
    currencies = DIVISAS

  # Excluir prefijos de divisas y pares con spread (mantiene DXY_vel_50d_lag1 en macro_cols)
  prefixes_to_exclude = tuple(
      [f"{c}_" for c in currencies]
      + [f"{p}_" for p, _ in MAP_DIVISAS.values()]
  )
  macro_cols = [
      col for col in df_wide.columns if not col.startswith(prefixes_to_exclude)
  ]

  ccy_feature_suffixes = [
      "return_rank_lag1",
      "return_skew_lag1",
      "return_vol_dir_lag1_log",
      "return_sma52_zscore_lag1",
      "return_vol_ratio_lag1",
      "return_sma26_delta_sma52_zscore_lag1",
      "return_std_lag1_log",
      "SPY_return_zscore_exposure_lag1",
      "Oil_return_zscore_exposure_lag1",
      "Gold_return_zscore_exposure_lag1",
      "VIX_pct_diff_lag1_zscore_exposure",
      "Real_yield_delta_lag1_zscore_exposure"
  ]

  spread_suffixes = [
      "yield_spread_zscore_lag1"
  ]

  panel_frames = []

  for ccy in currencies:
    df_ccy = pd.DataFrame(index=df_wide.index)
    df_ccy["Currency"] = ccy

    # 1. Target de la semana actual
    df_ccy["target_rank"] = df_wide[f"{ccy}_return_rank"]

    # 2. Features propios de la divisa (si no existe para USD, rellena con NaN)
    for suf in ccy_feature_suffixes:
      col_name = f"{ccy}_{suf}"
      df_ccy[suf] = df_wide[col_name] if col_name in df_wide.columns else np.nan

    # 3. Yield spreads alineados direccionalmente (positivo = diferencial a favor de 'ccy')
    if ccy in MAP_DIVISAS:
      pair_name, sign = MAP_DIVISAS[ccy]
      for suf in spread_suffixes:
        df_ccy[suf] = sign * df_wide[f"{pair_name}_{suf}"]
    else:
      # Para USD: promedio ponderado inverso de los 6 pares contra USD (o 0.0 si prefieres neutral)
      for suf in spread_suffixes:
        usd_spreads = [
            -sign * df_wide[f"{pair}_{suf}"]
            for pair, sign in MAP_DIVISAS.values()
        ]
        df_ccy[suf] = pd.concat(usd_spreads, axis=1).mean(axis=1)

    # 4. Variables macro globales (incluye DXY_vel_50d_lag1)
    if include_macro:
      df_ccy = pd.concat([df_ccy, df_wide[macro_cols]], axis=1)

    panel_frames.append(df_ccy)

  # Apilar las 7 divisas y ordenar cronológicamente por (date, Currency)
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
        df_train = df[df.index < fecha]
        df_test = df[df.index >= fecha].copy()

        return df_train, df_test
    

def separar_predictores_y_objetivo(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_target:Optional[str]=None 
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
        '''

        '''
        if col_target is None:
            col_target = "target_rank"

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
            df_panel = wide_to_long_panel(df_clean).set_index('date')

            # Separar datos de entrenamiento y datos live
            ultima_fecha = df_panel.index.max()
            df_train, df_live = separar_train_test(df=df_panel, fecha=ultima_fecha)

            X_train, y_train, X_live, y_live = separar_predictores_y_objetivo(
                df_train=df_train, 
                df_test=df_live
                )

            return X_train, y_train, X_live, y_live, ultima_fecha     

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
        "ultima_fecha": str(ultima_fecha),
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

def optimizar(
        simbolo:str='EURUSD'
):
    # Predictores a eliminar de los datasets train y live
    predictores_a_excluir = []
    X_train, y_train, X_test, y_test, _ = preprocesar_evaluacion(par=simbolo, variables_a_excluir=predictores_a_excluir)

    # 1. Concatenar train y test
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.concatenate([y_train, y_test])

    # 2. Definir índices: -1 entrena, 0 evalúa
    split_indices = np.full(len(X_combined), -1)
    split_indices[len(X_train):] = 0

    pds = PredefinedSplit(test_fold=split_indices)

    print("Configurando GridSearchCV para {simbolo}")
    modelo_base = Pipeline(steps=[
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [400],
        'rf__max_depth': [4],
        'rf__min_samples_leaf': [33],
        'rf__min_samples_split': [4],
        'rf__max_samples': [0.45],
        'rf__max_features': [0.3],
        'rf__class_weight': ['balanced_subsample']
    }

    # Configurar el GridSearchCV
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=pds,
        n_jobs=-1,
        verbose=2,
        refit=True # Al terminar, entrena el mejor modelo automáticamente
    )

    # Ajusta probando hiperparámetros entrenando en train y testeando en test
    grid_search.fit(X_combined, y_combined)

    # Resultados
    best_model = grid_search.best_estimator_
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds))
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)


def curva_desempenio(
        simbolo: str = "EURUSD",
        test_size: float=0.2
    ):
        '''
        Grafica la evolucion del desempeno del modelo al aumentar la complejidad.
        Identifica el trade-off entre Sesgo y Varianza:
        - Alto sesgo (underfitting): Metricas de train y test bajas y cercanas entre si.
        - Zona de equilibrio: punto donde la metrica test alcanza su valor optimo antes de degradarse.
        - Alta varianza (overfitting): la metrica de train sigue subiendo, mientras que la de
        test comienza a caer o se abre una brecha (generalization gap).
        '''
        predictores_a_excluir = []
        X_train, y_train, X_test, y_test, _ = preprocesar_evaluacion(
            par=simbolo, 
            variables_a_excluir=predictores_a_excluir,
            test_size=test_size
        )

        complexity_var = 'max_depth'
        complexity_range = list(range(1, 10, 1))
        train_scores, test_scores = [], []

        for value in complexity_range:
            rf = RandomForestClassifier(
                random_state=42,
                n_estimators=400, 
                max_depth=value,
                min_samples_leaf=28,
                min_samples_split=4,
                max_samples=0.45,
                max_features=0.3,
                class_weight='balanced_subsample',              
                n_jobs=1
            )
            rf.fit(X_train, y_train)

            # Probabilidades en train y test
            y_train_prob = rf.predict_proba(X_train)
            y_test_prob = rf.predict_proba(X_test)

            # Metricas roc_auc
            train_auc = roc_auc_score(y_train, y_train_prob, multi_class='ovr', average='weighted')
            test_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted')

            train_scores.append(train_auc)
            test_scores.append(test_auc)

        plt.figure(figsize=(8, 4.5))
        plt.plot(complexity_range, train_scores, marker='o', label="Train AUC (Aproximación al sesgo)")
        plt.plot(complexity_range, test_scores, marker='s', label="Test AUC (Generalización)")
        
        # Invertir la dirección del eje X (para Min Samples leaf):
        #plt.gca().invert_xaxis()
        
        plt.xlabel(complexity_var)
        plt.ylabel("ROC AUC")
        plt.title(f"Curva de Complejidad para el par {simbolo}")
        plt.legend()
        plt.grid(True)
        plt.show()


'''
Bloque principal

Ejecucion desde el terminal:
python -m src.modeling
'''
def main():

    X_train, y_train, X_live, _, ultima_fecha = preprocesar_datos()

    # Columnas numericas y categoricas
    cols_cat = ['Currency']
    cols_numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    # Inicializar modelo y entrenar
    #nombre_modelo = 'RandomForestRegressor'
    preprocessor = ColumnTransformer([
        ('num', 'passthrough', cols_numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cols_cat)
    ])
    pipe_rf = Pipeline([
        ('pre', preprocessor),
        ('model', RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=30,
            min_samples_split=50,
            max_samples=0.3,
            max_features=0.25
        ))
    ])

    pipe_rf.fit(X_train, y_train)

    # Realizar predicciones
    y_live_pred = pipe_rf.predict(X_live)
    resultados = pd.DataFrame({
        'Currency': X_live['Currency'].values,
        'Prediction': y_live_pred
    })

    print("Entrenamiento finalizado.")
    print(f"Dimensiones del dataset de entrenamiento:", X_train.shape)
    print(f"\nPredicciones para el {ultima_fecha}:")
    print(resultados.sort_values(by='Prediction', ascending=False).to_string(index=False))

    # Evaluar modelo
    evaluar = True
    if evaluar:
        print("\nEvaluación del modelo:")
        dict_evaluacion = evaluar_modelo(
            model=pipe_rf, 
            X_train=X_train, 
            y_train=y_train
            )


        registrar_evaluacion_excel(
            dict_evaluacion=dict_evaluacion,
            ultima_fecha=ultima_fecha,
            y_pred=resultados # prediccion del modelo completo sobre la ultima fecha
        )



if __name__ == '__main__':
    main()

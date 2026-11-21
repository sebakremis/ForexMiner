'''
src/cat_modeling.py

Entrenamiento de modelo de Clasificacion
'''
import pandas as pd
import numpy as np
import datetime as dt
import argparse
from pathlib import Path
from typing import (
   Sequence, Optional, Any,
   Dict, List
   )
from datetime import datetime
import matplotlib.pyplot as plt
import json

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit, TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import CARPETA_DATOS, INTERVALO_DEFECTO, MAP_DIVISAS, DIVISAS


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


# Funciones de pre-procesamiento

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
    col_target:Optional[str]=None,
    flag_evaluacion:bool=False    
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
        Y_train = df_train[col_target]

        if flag_evaluacion:
            Y_test = df_test[col_target]
            
        else:
            Y_test=pd.DataFrame() # dataframe vacio
            # se asigna a nan para evitar error al eliminar columnas
            df_test.loc[:, col_target] = np.nan 
        
        X_test = df_test.drop(columns=cols_drop)

        return X_train, Y_train, X_test, Y_test


def preprocesar_entrenamiento(
)->tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, datetime]:
            # Cargar fichero de precios
            df_clean = cargar_datos_clean()

            # Convertir a panel long
            df_panel = wide_to_long_panel(df_clean).set_index('date')

            # Separar datos de entrenamiento y datos live
            ultima_fecha = df_panel.index.max()
            df_train, df_live = separar_train_test(df=df_panel, fecha=ultima_fecha)

            X_train, y_train, X_live, _ = separar_predictores_y_objetivo(
                df_train=df_train, 
                df_test=df_live, 
                flag_evaluacion=False
                )

            return X_train, y_train, X_live, pd.Series(), ultima_fecha     

# Funciones de evaluacion

def categorizar_con_bins(
    df: pd.DataFrame,
    columna_target: str,
    df_bins: pd.DataFrame | None = None,
    extender_extremos: bool = True,
    nombre_col_salida: str | None = None,
) -> pd.DataFrame:
    """Categoriza una columna en un DataFrame utilizando los límites precalculados en df_bins.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a procesar.
    columna_target : str
        Nombre de la columna numérica a categorizar.
    df_bins : pd.DataFrame
        DataFrame con columnas ['categoria', 'lim_inferior', 'lim_superior'].
    extender_extremos : bool, default True
        Si es True, reemplaza el límite inferior mínimo por -inf y el superior
        máximo por inf, evitando que valores fuera del rango original queden
        como NaN.
    nombre_col_salida : str, optional
        Nombre de la nueva columna con las categorías. Por defecto será
        '{columna_target}_cat'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con la nueva columna categorizada.
    """
    if columna_target not in df.columns:
        raise KeyError(
            f"La columna '{columna_target}' no está presente en el DataFrame."
        )

    if df_bins is None:
        raise KeyError(
                    "Falta la tabla de bins."
                )

    # Validar que df_bins tenga la estructura requerida y no esté vacío
    columnas_requeridas = {"categoria", "lim_inferior", "lim_superior"}
    if not columnas_requeridas.issubset(df_bins.columns):
        raise ValueError(
            f"df_bins debe contener las columnas: {columnas_requeridas}"
        )

    df_out = df.copy()
    col_out = nombre_col_salida or f"{columna_target}_cat"

    if df_bins.empty:
        df_out[col_out] = np.nan
        return df_out

    # Ordenar por categoría para garantizar la secuencia monótona de cortes
    bins_sorted = df_bins.sort_values("categoria").reset_index(drop=True)

    # Reconstruir el array continuo de bordes: [lim_inf_0, lim_sup_0, lim_sup_1, ...]
    cortes = np.concatenate(
        ([bins_sorted["lim_inferior"].iloc[0]], bins_sorted["lim_superior"].values)
    )
    etiquetas = bins_sorted["categoria"].tolist()

    if extender_extremos:
        cortes[0] = -np.inf
        cortes[-1] = np.inf

    # pd.cut asigna según los intervalos predefinidos
    # include_lowest=True asegura que el límite inferior del primer bin sea cerrado (análogo a qcut)
    df_out[col_out] = pd.cut(
        df_out[columna_target],
        bins=cortes,
        labels=etiquetas,
        include_lowest=True,
    )

    return df_out


def preprocesar_evaluacion(
        test_size:float=0.2
        )->tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, datetime]:
            # Cargar fichero de precios
            df_clean = cargar_datos_clean()

            # Convertir a panel long
            df_panel = wide_to_long_panel(df_clean).set_index('date')

            # Primero se remueven los datos live (ultima fecha contemporanea)
            ultima_fecha = df_panel.index.max()
            df_train, _ = separar_train_test(df=df_panel, fecha=ultima_fecha)

            # Luego de remover los datos live, se separan los datos de train y validacion
            split_idx = int(len(df_train) * (1 - test_size))
            fecha_corte = df_train.index[split_idx]

            df_train, df_test = separar_train_test(df=df_train, fecha=fecha_corte)

            X_train, Y_train, X_test, Y_test = separar_predictores_y_objetivo(
                    df_train=df_train, 
                    df_test=df_test, 
                    flag_evaluacion=True
                    )
            
            return X_train, Y_train, X_test, Y_test, fecha_corte


def _extract_estimator(model):
  """Extrae el estimador final si 'model' es un sklearn Pipeline."""
  if isinstance(model, Pipeline):
    return model.steps[-1][1]
  return model


def _compute_weekly_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
  """Calcula Spearman Rank IC semana a semana (axis=1) y métricas de extremos."""
  ics = []
  top_beats_bottom = []

  for t in range(y_true.shape[0]):
    row_true = y_true[t]
    row_pred = y_pred[t]

    # Evitar NaNs o filas constantes
    if np.std(row_pred) < 1e-12 or np.std(row_true) < 1e-12:
      ics.append(0.0)
      top_beats_bottom.append(0.5)
      continue

    ic, _ = spearmanr(row_true, row_pred)
    ics.append(0.0 if np.isnan(ic) else float(ic))

    # ¿La divisa predicha más fuerte (argmax) superó realmente a la más débil (argmin)?
    idx_long = int(np.argmax(row_pred))
    idx_short = int(np.argmin(row_pred))
    if row_true[idx_long] > row_true[idx_short]:
      top_beats_bottom.append(1.0)
    elif row_true[idx_long] < row_true[idx_short]:
      top_beats_bottom.append(0.0)
    else:
      top_beats_bottom.append(0.5)

  ics = np.array(ics)
  ic_mean = float(np.mean(ics))
  ic_std = float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan
  ic_ir = float(ic_mean / ic_std) if ic_std and ic_std > 0 else np.nan

  return {
      "ic_series": ics,
      "ic_mean": ic_mean,
      "ic_std": ic_std,
      "ic_ir": ic_ir,
      "ic_positive_rate": float(np.mean(ics > 0)),
      "top_vs_bottom_hit_rate": float(np.mean(top_beats_bottom)),
  }


def _rank_ic_scorer(estimator, X, y):
  """Scorer custom para permutation_importance basado en Mean Weekly Rank IC."""
  y_arr = y.values if isinstance(y, pd.DataFrame) else np.asarray(y)
  y_pred = estimator.predict(X)
  return _compute_weekly_rank_ic(y_arr, y_pred)["ic_mean"]


def evaluar_modelo_multitask_rank(
    model,
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    random_state: int = 42,
    run_permutation_importance: bool = False,
    currency_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
  """Evalúa un modelo MultiTaskElasticNetCV (o Pipeline) sobre rankings cross-sectional

  de divisas con partición temporal.
  """
  feature_names = (
      X_train.columns.tolist()
      if hasattr(X_train, "columns")
      else [f"feature_{i}" for i in range(X_train.shape[1])]
  )

  if currency_names is None:
    currency_names = (
        Y_train.columns.tolist()
        if hasattr(Y_train, "columns")
        else [f"currency_{i}" for i in range(Y_train.shape[1])]
    )

  # Entrenamiento del estimador / pipeline
  model.fit(X_train, Y_train)
  est = _extract_estimator(model)

  # Predicciones
  Y_pred_train = model.predict(X_train)
  Y_pred_val = model.predict(X_val)

  Y_train_arr = (
      Y_train.values if isinstance(Y_train, pd.DataFrame) else np.asarray(Y_train)
  )
  Y_val_arr = (
      Y_val.values if isinstance(Y_val, pd.DataFrame) else np.asarray(Y_val)
  )

  # Métricas de regresión y Rank IC cross-sectional
  train_rmse = float(np.sqrt(mean_squared_error(Y_train_arr, Y_pred_train)))
  val_rmse = float(np.sqrt(mean_squared_error(Y_val_arr, Y_pred_val)))
  val_mae = float(mean_absolute_error(Y_val_arr, Y_pred_val))
  val_r2 = float(r2_score(Y_val_arr, Y_pred_val, multioutput="uniform_average"))

  train_ic_stats = _compute_weekly_rank_ic(Y_train_arr, Y_pred_train)
  val_ic_stats = _compute_weekly_rank_ic(Y_val_arr, Y_pred_val)

  # Coeficientes y esparsidad del MultiTaskElasticNetCV
  # coef_ tiene shape (n_targets, n_features)
  coef_matrix = est.coef_
  l21_norms = np.linalg.norm(coef_matrix, axis=0)
  n_selected = int(np.sum(l21_norms > 1e-8))
  total_features = len(feature_names)

  alpha_val = float(getattr(est, "alpha_", np.nan))
  l1_ratio_val = float(getattr(est, "l1_ratio_", np.nan))

  validation_summary = {
      "Val Mean Weekly Rank IC": val_ic_stats["ic_mean"],
      "Val Rank IC IR (Mean/Std)": val_ic_stats["ic_ir"],
      "Val IC > 0 (%)": val_ic_stats["ic_positive_rate"],
      "Val Top-vs-Bottom Hit Rate": val_ic_stats["top_vs_bottom_hit_rate"],
      "Val RMSE": val_rmse,
      "Val MAE": val_mae,
      "Val R2 (Uniform Avg)": val_r2,
      "Train vs Val Rank IC Gap": (
          train_ic_stats["ic_mean"] - val_ic_stats["ic_mean"]
      ),
      "Selected Alpha": alpha_val,
      "Selected L1 Ratio": l1_ratio_val,
      "Active Features": f"{n_selected} / {total_features}",
  }

  raw_scores = {
      "train_rmse": train_rmse,
      "val_rmse": val_rmse,
      "val_mae": val_mae,
      "val_r2": val_r2,
      "train_rank_ic_mean": train_ic_stats["ic_mean"],
      "val_rank_ic_mean": val_ic_stats["ic_mean"],
      "val_rank_ic_std": val_ic_stats["ic_std"],
      "val_rank_ic_ir": val_ic_stats["ic_ir"],
      "val_ic_positive_rate": val_ic_stats["ic_positive_rate"],
      "val_top_vs_bottom_hit_rate": val_ic_stats["top_vs_bottom_hit_rate"],
      "alpha": alpha_val,
      "l1_ratio": l1_ratio_val,
      "n_selected_features": n_selected,
      "n_total_features": total_features,
  }

  # Reporte detallado por divisa
  currency_rows = []
  for idx, ccy in enumerate(currency_names):
    yt_c = Y_val_arr[:, idx]
    yp_c = Y_pred_val[:, idx]
    corr_c, _ = spearmanr(yt_c, yp_c)
    currency_rows.append({
        "Currency": ccy,
        "RMSE": np.sqrt(mean_squared_error(yt_c, yp_c)),
        "MAE": mean_absolute_error(yt_c, yp_c),
        "R2": r2_score(yt_c, yp_c),
        "Temporal Spearman": 0.0 if np.isnan(corr_c) else corr_c,
        "Pred Std": np.std(yp_c),
    })
  df_currency_report = pd.DataFrame(currency_rows).set_index("Currency")

  # 5. Matriz de transición de rangos discretos 1..N (equivalente a confusion_matrix)
  n_ccy = len(currency_names)
  true_ranks_int = (
      pd.DataFrame(Y_val_arr).rank(axis=1, method="first").values.astype(int)
  )
  pred_ranks_int = (
      pd.DataFrame(Y_pred_val).rank(axis=1, method="first").values.astype(int)
  )

  rank_cm_raw = np.zeros((n_ccy, n_ccy), dtype=int)
  for t_r, p_r in zip(true_ranks_int.ravel(), pred_ranks_int.ravel()):
    rank_cm_raw[t_r - 1, p_r - 1] += 1

  row_sums = rank_cm_raw.sum(axis=1, keepdims=True)
  rank_cm_norm = np.divide(
      rank_cm_raw, row_sums, out=np.zeros_like(rank_cm_raw, dtype=float), where=row_sums != 0
  )
  rank_labels = [f"Rank {r}" for r in range(1, n_ccy + 1)]
  df_rank_cm = pd.DataFrame(
      rank_cm_norm,
      index=[f"True: {lbl}" for lbl in rank_labels],
      columns=[f"Pred: {lbl}" for lbl in rank_labels],
  )

  # 6. Importancia de variables por norma L2,1 conjunta + pesos por divisa
  df_coef_per_ccy = pd.DataFrame(
      coef_matrix.T, index=feature_names, columns=currency_names
  )
  df_importance = pd.DataFrame({
      "feature": feature_names,
      "importance_l21_norm": l21_norms,
      "active": l21_norms > 1e-8,
  })
  df_importance = pd.concat(
      [df_importance, df_coef_per_ccy.reset_index(drop=True)], axis=1
  ).sort_values(by="importance_l21_norm", ascending=False)

  # 7. Permutation Feature Importance sobre el conjunto de validación (usando Rank IC)
  df_perm_importance = None
  if run_permutation_importance:
    perm_result = permutation_importance(
        model,
        X_val,
        Y_val,
        scoring=_rank_ic_scorer,
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )
    df_perm_importance = pd.DataFrame({
        "feature": feature_names,
        "perm_ic_drop_mean": perm_result.importances_mean,
        "perm_ic_drop_std": perm_result.importances_std,
    }).sort_values(by="perm_ic_drop_mean", ascending=False)

  # Impresión de resultados
  print("=" * 70)
  print("RESUMEN DE VALIDACIÓN (MULTI-TASK RANKING)")
  print("=" * 70)
  for metric, val in validation_summary.items():
    if isinstance(val, str):
      display_val = val
    elif isinstance(val, float) and not np.isnan(val):
      display_val = f"{val:.4f}"
    else:
      display_val = "N/A"
    print(f"{metric:<32}: {display_val}")

  print("\n" + "=" * 70)
  print("REPORTE POR DIVISA (SOBRE VALIDACIÓN)")
  print("=" * 70)
  print(df_currency_report.round(4))

  print("\n" + "=" * 70)
  print("MATRIZ DE TRANSICIÓN DE RANGOS (1=MÁS DÉBIL, 7=MÁS FUERTE)")
  print("=" * 70)
  print(df_rank_cm.round(4))

  print("\n" + "=" * 70)
  print("RANKING DE VARIABLES - NORMA CONJUNTA L2,1")
  print("=" * 70)
  cols_to_print = ["feature", "importance_l21_norm"] + currency_names
  print(df_importance[cols_to_print].head(30).to_string(index=False))

  if df_perm_importance is not None:
    print("\n" + "=" * 70)
    print("RANKING DE VARIABLES - IMPORTANCIA POR PERMUTACIÓN (RANK IC)")
    print("=" * 70)
    print(df_perm_importance.head(20).to_string(index=False))

  # DataFrame de predicciones en validación para análisis posterior
  val_index = Y_val.index if isinstance(Y_val, pd.DataFrame) else None
  df_val_preds = pd.DataFrame(
      Y_pred_val, index=val_index, columns=currency_names
  )

  return {
      "validation_summary": validation_summary,
      "validation_raw_results": raw_scores,
      "weekly_ic_series": val_ic_stats["ic_series"],
      "currency_report": df_currency_report,
      "rank_matrix_raw": rank_cm_raw,
      "rank_matrix_normalized": df_rank_cm,
      "feature_importance_l21": df_importance,
      "feature_importance_permutation": df_perm_importance,
      "val_predictions": df_val_preds,
      "fitted_model": model,
      "partitions": {
          "X_train": X_train,
          "X_val": X_val,
          "y_train": Y_train,
          "y_val": Y_val,
      },
  }


def registrar_evaluacion_excel(
    dict_evaluacion,
    simbolo,
    ultima_fecha,
    y_pred,
    filepath="historico_modelos_cat.xlsx",
    ):
        """Guarda o anexa los resultados de evaluar_modelo_cat y sus metadatos

        en un libro de Excel estructurado.
        """
        path = Path(CARPETA_DATOS) / f"evaluacion_modelos/{filepath}"
        run_id = f"{simbolo}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Asegurar que el directorio exista
        path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Aplanar métricas de validación cruzada y metadatos (Fila única)
        val_summary = dict_evaluacion.get("validation_summary", {})
        fila_resumen = {
            "run_id": run_id,
            "timestamp_registro": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "simbolo": simbolo,
            "intervalo": INTERVALO_DEFECTO,
            "ultima_fecha": str(ultima_fecha),
            "y_pred": str(y_pred),
            # Métricas escalares agregadas
            "val_accuracy": val_summary.get("Val Accuracy", np.nan),
            "val_balanced_acc": val_summary.get("Val Balanced Accuracy", np.nan),
            "val_f1_macro": val_summary.get("Val F1 (Macro)", np.nan),
            "val_f1_weighted": val_summary.get("Val F1 (Weighted)", np.nan),
            "val_log_loss": val_summary.get("Val Log Loss", np.nan),
            "train_test_gap": val_summary.get("Train vs Val Accuracy Gap", np.nan),
        }
        df_nueva_fila = pd.DataFrame([fila_resumen])

        # 2. Preparar el DataFrame de Importancia de Features para anexar
        df_feat = dict_evaluacion.get("feature_importance_mdi")
        if df_feat is not None:
            df_feat = df_feat.copy()
            df_feat.insert(0, "run_id", run_id)
            df_feat.insert(1, "simbolo", simbolo)

        # 3. Preparar el Classification Report para anexar
        df_report = dict_evaluacion.get("classification_report")
        if df_report is not None:
            df_report = df_report.copy().reset_index()
            df_report = df_report.rename(columns={"index": "clase_o_metrica"})
            df_report.insert(0, "run_id", run_id)
            df_report.insert(1, "simbolo", simbolo)

        # 4. Escritura / Concatenación en Excel
        if not path.exists():
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df_nueva_fila.to_excel(
                    writer, sheet_name="Registro_Modelos", index=False
                )
                if df_feat is not None:
                    df_feat.to_excel(
                        writer, sheet_name="Log_Features", index=False
                    )
                if df_report is not None:
                    df_report.to_excel(
                        writer, sheet_name="Log_Clases", index=False
                    )
        else:
            # Modo append conservando las hojas existentes
            with pd.ExcelWriter(
                path, engine="openpyxl", mode="a", if_sheet_exists="overlay"
            ) as writer:
                # Anexar a Hoja Principal
                try:
                    hoja_existente = writer.sheets["Registro_Modelos"]
                    start_row = hoja_existente.max_row
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
                        hoja_feat = writer.sheets["Log_Features"]
                        start_row_feat = hoja_feat.max_row
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

                # Anexar a Log_Clases
                if df_report is not None:
                    try:
                        hoja_report = writer.sheets["Log_Clases"]
                        start_row_rep = hoja_report.max_row
                        df_report.to_excel(
                            writer,
                            sheet_name="Log_Clases",
                            index=False,
                            header=False,
                            startrow=start_row_rep,
                        )
                    except KeyError:
                        df_report.to_excel(
                            writer, sheet_name="Log_Clases", index=False
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


# Predictores a excluir segun el par forex
def cargar_cols_filtrar()->dict:
    fichero = Path(CARPETA_DATOS) / "cols_filtrar.json"
    try:
        with open(fichero, "r", encoding="utf-8") as file:
            excluir_dict = json.load(file)
            return excluir_dict if isinstance(excluir_dict,dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}



'''
Bloque principal

Ejecucion desde el terminal:
python -m src.modeling
'''
def main():

    X_train, Y_train, X_live, _, ultima_fecha = preprocesar_entrenamiento()

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

    pipe_rf.fit(X_train, Y_train)

    # Realizar predicciones
    Y_live_pred = pipe_rf.predict(X_live)

    print("Entrenamiento finalizado.")
    print(f"Dimensiones del dataset de entrenamiento:", X_train.shape)
    print(f"Predicciones para el {ultima_fecha.strftime("%Y-%m-%d")}:\n{Y_live_pred}")
    #print("Probabilidades:")
    #print(df_probabilidades.head().to_string(index=False))

    # Evaluar modelo
    evaluar = False
    if evaluar:
        model = pipe_rf.named_steps['model']

        X_train, Y_train, X_test, Y_test, fecha_corte = preprocesar_evaluacion()
        print("Evaluando modelo...")
        print("Fecha de corte:", fecha_corte)

        dict_evaluacion = evaluar_modelo_multitask_rank(
            model=model, 
            X_train=X_train, 
            Y_train=Y_train,
            X_val=X_test,
            Y_val=Y_test
            )
        top_feats = dict_evaluacion["feature_importance_l21"].head(5)
        print(top_feats[["feature", "importance_l21_norm"]])

        # Verificar si esa variable tiene una escala numérica mucho mayor que el resto
        print("\nDesviación estándar en X_train de esas variables:")
        print(X_train[top_feats["feature"]].std())
'''
        registrar_evaluacion_excel(
            dict_evaluacion=dict_evaluacion,
            ultima_fecha=ultima_fecha,
            y_pred=Y_live_pred # prediccion del modelo completo sobre la ultima fecha
        )
'''


if __name__ == '__main__':
    main()

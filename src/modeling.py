'''
src/cat_modeling.py

Entrenamiento de modelo de Clasificacion
'''
import pandas as pd
import numpy as np
import datetime as dt
import argparse
from pathlib import Path
from typing import Tuple, Optional, Sequence
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.pipeline import Pipeline

from src.transform import encontrar_columnas
from src.config import carpeta_datos, intervalo_defecto, forex_cols


def cargar_datos_clean()->pd.DataFrame:
    fichero_clean = Path(carpeta_datos) / "clean_data/datos_depurados.parquet"
    try:
        df = pd.read_parquet(fichero_clean)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{fichero_clean}' no existe en la ruta especificada.")
        return pd.DataFrame()   
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame()


def filtrar_pares_restantes(
        df:pd.DataFrame,
        target:str
)->pd.DataFrame:
    columnas_clean = df.columns.tolist()
   
    simbolos_restantes = [col for col in forex_cols if col != target]

    features_a_mantener = [
        '_return_zscore_lag1',
        '_std_lag1_log',
        '_return_skew_lag1',
        '_gap_skew_lag1',

    ]
    cols_permitidas = {
        f"{s}{feat}"
        for s in simbolos_restantes
        for feat in features_a_mantener
    }

    # Se eliminan las demas columnas
    cols_drop = [
        col
        for s in simbolos_restantes
        for col in encontrar_columnas(columnas_clean, s)
        if col not in cols_permitidas
    ]

    return df.drop(columns=cols_drop, errors='ignore')

# Funciones de pre-procesamiento

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
    

def categorizar_en_cuantiles(
    df_train: pd.DataFrame,
    columna: str,
    num_cuantiles: int = 3,
    quitar_original: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Transforma una columna continua en cuantiles globales.

    Parámetros:
    - df: DataFrame con los datos.
    - columna: Nombre de la columna continua a transformar.
    - num_cuantiles: Cantidad de cuantiles solicitados.
    - quitar_original: Si se elimina la columna original tras crear los cuantiles.

    Retorna:
    - (df_out, df_bins): Tupla con el nuevo DataFrame y un DataFrame con los
      límites inferiores y superiores de cada cuantil generado. Si no fue posible
      calcular los cuantiles, df_bins será None.
    """
    df_out = df_train.copy()
    nueva_columna = f"{columna}_cat"

    serie = df_out[columna]
    valid_count = serie.count()
    df_bins = None

    # Verificación de datos suficientes y varianza
    if valid_count < num_cuantiles or serie.nunique() < 2:
        df_out[nueva_columna] = np.nan
    else:
        try:
            # retbins=True devuelve la serie categórica y el array con los bordes
            cuantiles, bins = pd.qcut(
                serie,
                q=num_cuantiles,
                labels=False,
                duplicates="drop",
                retbins=True,
            )

            # Sumar 1 para etiquetar de 1 a N
            df_out[nueva_columna] = cuantiles + 1

            # Construir tabla de límites por categoría
            # len(bins) - 1 corresponde al número real de cuantiles formados
            num_categorias_creadas = len(bins) - 1
            df_bins = pd.DataFrame(
                {
                    "categoria": range(1, num_categorias_creadas + 1),
                    "lim_inferior": bins[:-1],
                    "lim_superior": bins[1:],
                }
            )

        except ValueError:
            df_out[nueva_columna] = np.nan

    # Convertir a tipo 'category'
    df_out[nueva_columna] = df_out[nueva_columna].astype("category")

    # Eliminar columna original si corresponde
    if quitar_original:
        df_out = df_out.drop(columns=[columna])

    return df_out, df_bins


def separar_predictores_y_objetivo(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    col_target:str, 
    cols_excluded: list[str],
    flag_evaluacion:bool
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    '''

    '''
    # Columnas a descartar de predictores
    cols_excluded.extend([col_target])
    columnas_a_descartar = set(cols_excluded)
    cols_drop = [c for c in columnas_a_descartar if c in df_train.columns]

    X_train = df_train.drop(columns=cols_drop)
    y_train = df_train[col_target]

    if flag_evaluacion:
        y_test = df_test[col_target]
        
    else:
        y_test=pd.Series() # serie vacia
        # se asigna a nan para evitar error al eliminar columnas
        df_test.loc[:, col_target] = np.nan 
    
    X_test = df_test.drop(columns=cols_drop)

    return X_train, y_train, X_test, y_test


def preprocesar_entrenamiento(
        par:str,
        variables_a_excluir:list[str]
)->tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, datetime]:
    # Cargar fichero de precios
    df = cargar_datos_clean()
    df = filtrar_pares_restantes(df,par)

    # Separar datos de entrenamiento y datos live
    ultima_fecha = df.index.max()
    df_train, df_live = separar_train_test(df=df, fecha=ultima_fecha)

    # Se crea la variable categorica a partir del dataset de entrenamiento
    # Variable objetivo: se crea variable categorica por cuantiles
    objetivo = f"{par}_intra_weekly"
    objetivo_cat = f"{objetivo}_cat"

    df_train, tabla_bins = categorizar_en_cuantiles(
        df_train=df_train, 
        columna=objetivo, 
        num_cuantiles=3, 
        quitar_original=False
        )
   
    print(f"\n--- Rangos por Categoría para el ticker {par} ---")
    print(tabla_bins)

    # Separar predictores y variable objetivo
    variables_a_excluir.extend([objetivo])


    X_train, y_train, X_live, _ = separar_predictores_y_objetivo(
        df_train=df_train, 
        df_test=df_live, 
        col_target=objetivo_cat,
        cols_excluded=variables_a_excluir,
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
        par:str,
        variables_a_excluir:list[str],
        test_size:float=0.2
)->tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, datetime]:
    # Cargar fichero de precios
    df = cargar_datos_clean()
    df = filtrar_pares_restantes(df,par)

    # Primero se remueven los datos live (ultima fecha contemporanea)
    ultima_fecha = df.index.max()
    df_train, _ = separar_train_test(df=df, fecha=ultima_fecha)

    # Luego de remover los datos live, se separan los datos de train y validacion
    split_idx = int(len(df_train) * (1 - test_size))
    fecha_corte = df_train.index[split_idx]

    df_train, df_test = separar_train_test(df=df_train, fecha=fecha_corte)
    print("Generando los datos para la evaluación.")

    # Se crea la variable categorica a partir del dataset de entrenamiento
    # Variable objetivo: se crea variable categorica por cuantiles
    objetivo = f"{par}_intra_weekly"
    objetivo_cat = f"{objetivo}_cat"

    df_train, tabla_bins = categorizar_en_cuantiles(
        df_train, 
        columna=objetivo, 
        num_cuantiles=3, 
        quitar_original=False
        )
    print(f"\n--- Rangos por Categoría para el ticker {par} ---")
    print(tabla_bins)

    # Categorizar la variable objetivo en df_test con los mismos limites (extendidos a -inf / +inf)
    df_test = categorizar_con_bins(
        df=df_test,
        columna_target=objetivo,
        df_bins=tabla_bins
    )
    
    # Separar predictores y variable objetivo
    variables_a_excluir.extend([objetivo])

    X_train, y_train, X_test, y_test = separar_predictores_y_objetivo(
            df_train=df_train, 
            df_test=df_test, 
            col_target=objetivo_cat,
            cols_excluded=variables_a_excluir,
            flag_evaluacion=True
            )
    
    return X_train, y_train, X_test, y_test, fecha_corte


def evaluar_modelo_cat(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    random_state=42,
    run_permutation_importance=False,
    class_names=None,
):
    """Evalúa un RandomForest multiclase a partir de datos con particion temporal.
    Reporte por clase, matrices de confusión y análisis de importancia de variables.
    """
    feature_names = (
        X_train.columns.tolist()
        if hasattr(X_train, "columns")
        else [f"feature_{i}" for i in range(X_train.shape[1])]
    )

    # Entrenamiento del estimador
    model.fit(X_train, y_train)

    if class_names is None:
        class_names = [str(cls) for cls in model.classes_]

    # Predicciones de validación y probabilidades
    y_pred_val = model.predict(X_val)
    y_pred_train = model.predict(X_train)

    # Manejo de probabilidades para Log Loss
    if hasattr(model, "predict_proba"):
        y_proba_val = model.predict_proba(X_val)
        val_log_loss = log_loss(y_val, y_proba_val, labels=model.classes_)
    else:
        val_log_loss = np.nan

    # Métricas equivalentes calculadas sobre el set de validación
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_balanced_acc = balanced_accuracy_score(y_val, y_pred_val)
    val_f1_macro = f1_score(y_val, y_pred_val, average="macro")
    val_f1_weighted = f1_score(y_val, y_pred_val, average="weighted")

    validation_summary = {
        "Val Accuracy": f"{val_acc:.4f}",
        "Val Balanced Accuracy": f"{val_balanced_acc:.4f}",
        "Val F1 (Macro)": f"{val_f1_macro:.4f}",
        "Val F1 (Weighted)": f"{val_f1_weighted:.4f}",
        "Val Log Loss": f"{val_log_loss:.4f}"
        if not np.isnan(val_log_loss)
        else "N/A",
        "Train vs Val Accuracy Gap": f"{(train_acc - val_acc):.4f}",
    }

    raw_scores = {
        "train_accuracy": train_acc,
        "test_accuracy": val_acc,
        "test_balanced_accuracy": val_balanced_acc,
        "test_f1_macro": val_f1_macro,
        "test_f1_weighted": val_f1_weighted,
        "test_log_loss": val_log_loss,
    }

    # Reporte por clase y matriz de confusión (evaluados sobre validación)
    cls_report_dict = classification_report(
        y_val,
        y_pred_val,
        target_names=class_names,
        labels=model.classes_,
        output_dict=True,
    )
    df_cls_report = pd.DataFrame(cls_report_dict).transpose()

    cm_raw = confusion_matrix(y_val, y_pred_val, labels=model.classes_)
    cm_norm = confusion_matrix(
        y_val, y_pred_val, labels=model.classes_, normalize="true"
    )
    df_cm = pd.DataFrame(
        cm_norm,
        index=[f"True: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names],
    )

    # Importancia de Variables (MDI)
    mdi_importances = model.feature_importances_
    df_importance = pd.DataFrame(
        {"feature": feature_names, "importance_mdi": mdi_importances}
    ).sort_values(by="importance_mdi", ascending=False)

    # 6. Permutation Feature Importance sobre el conjunto de validación
    df_perm_importance = None
    if run_permutation_importance:
        perm_result = permutation_importance(
            model,
            X_val,
            y_val,
            scoring="f1_macro",
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        )
        df_perm_importance = pd.DataFrame(
            {
                "feature": feature_names,
                "perm_importance_mean": perm_result.importances_mean,
                "perm_importance_std": perm_result.importances_std,
            }
        ).sort_values(by="perm_importance_mean", ascending=False)

    # Impresión de resultados
    print("=" * 60)
    print("RESUMEN DE VALIDACIÓN")
    print("=" * 60)
    for metric, val in validation_summary.items():
        print(f"{metric:<30}: {val}")

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN DETALLADO (SOBRE VALIDACIÓN)")
    print("=" * 60)
    print(df_cls_report.round(4))

    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSIÓN (NORMALIZADA POR SOPORTE REAL)")
    print("=" * 60)
    print(df_cm.round(4))

    print("\n" + "=" * 60)
    print("RANKING DE VARIABLES - IMPORTANCIA POR IMPUREZA (MDI)")
    print("=" * 60)
    print(df_importance.head(20).to_string(index=False))

    if df_perm_importance is not None:
        print("\n" + "=" * 60)
        print("RANKING DE VARIABLES - IMPORTANCIA POR PERMUTACIÓN (F1-Macro)")
        print("=" * 60)
        print(df_perm_importance.head(20).to_string(index=False))

    return {
        "validation_summary": validation_summary,
        "validation_raw_results": raw_scores,
        "classification_report": df_cls_report,
        "confusion_matrix_raw": cm_raw,
        "confusion_matrix_normalized": df_cm,
        "feature_importance_mdi": df_importance,
        "feature_importance_permutation": df_perm_importance,
        "fitted_model": model,
        "partitions": {
            "X_train": X_train,
            "X_val": X_val,
            "y_train": y_train,
            "y_val": y_val,
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
    path = Path(carpeta_datos) / f"evaluacion_modelos/{filepath}"
    run_id = f"{simbolo}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # Asegurar que el directorio exista
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Aplanar métricas de validación cruzada y metadatos (Fila única)
    val_summary = dict_evaluacion.get("validation_summary", {})
    fila_resumen = {
        "run_id": run_id,
        "timestamp_registro": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "simbolo": simbolo,
        "intervalo": intervalo_defecto,
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
        df_report.rename(columns={"index": "clase_o_metrica"}, inplace=True)
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


# Predictores a excluir segun el par forex
excluir_dict = {
    'AUDUSD' : ['SPY_Volume_zscore_lag1_yeo', 'Oil_Volume_zscore_lag1_yeo', 'Gold_Volume_zscore_lag1_yeo'],  
    'EURUSD' : ['EURUSD_accel_diff_zscore_lag1', 'Gold_acceleration_zscore_lag1', 'Gold_vol_ratio_lag1', 'Oil_vol_ratio_lag1', 'Gold_accel_diff_zscore_lag1'], 
    'GBPUSD' : [], 
    'USDCAD' : ['USDCAD_accel_diff_zscore_lag1','USDCAD_vol_ratio_lag1'], 
    'USDCHF' : ['USDCHF_accel_diff_zscore_lag1', 'Gold_acceleration_zscore_lag1'],
    'USDJPY' : ['USDJPY_accel_diff_zscore_lag1']  
}


'''
Bloque principal

Ejecucion desde el terminal:
python -m src.modeling
'''
def main(
        simbolo:str,
        evaluar:bool=False
):
    # Predictores a eliminar de los datasets train y live
    predictores_a_excluir = excluir_dict.get(simbolo,[])

    X_train, y_train, X_live, _, ultima_fecha = preprocesar_entrenamiento(par=simbolo, variables_a_excluir=predictores_a_excluir)

    # Inicializar modelo y entrenar
    pipe = Pipeline([
        ('model', RandomForestClassifier(
            random_state=42,
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=33,
            min_samples_split= 4,
            max_samples= 0.45,
            max_features= 0.3,
            class_weight='balanced_subsample'            
        ))

    ])

    pipe.fit(X_train, y_train)

    # Realizar predicciones
    y_live_pred = pipe.predict(X_live)
    y_live_pred_proba = pipe.predict_proba(X_live)
    df_probabilidades = pd.DataFrame(y_live_pred_proba, columns=pipe.classes_)

    print(f"Entrenamiento modelo {simbolo} finalizado.")
    print(f"Dimensiones del dataset de entrenamiento:", X_train.shape)
    print(f"Categoria predicha para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_live_pred[0]}")
    print("Probabilidades:")
    print(df_probabilidades.head().to_string(index=False))

    # Evaluar modelo
    if evaluar:
        rf_model = pipe.named_steps['model']

        X_train, y_train, X_test, y_test, fecha_corte = preprocesar_evaluacion(
            par=simbolo, 
            variables_a_excluir=predictores_a_excluir
            )
        print("Evaluando modelo...")
        print("Fecha de corte:", fecha_corte)

        dict_evaluacion = evaluar_modelo_cat(
            model=rf_model, 
            X_train=X_train, 
            y_train=y_train,
            X_val=X_test,
            y_val=y_test
            )

        registrar_evaluacion_excel(
            dict_evaluacion=dict_evaluacion,
            simbolo=simbolo,
            ultima_fecha=ultima_fecha,
            y_pred=y_live_pred # prediccion del modelo completo sobre la ultima fecha
        )

def entrypoint_modeling(argv: Sequence[str] | None = None):
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description="Módulo de Modelado Categórico")

    parser.add_argument(
        "--simbolo", type=str, default="EURUSD", help="Símbolo del activo"
    )
    parser.add_argument(
            "--evaluar",
            action="store_true",
            help="Ejecuta la evaluación del modelo tras el entrenamiento",
        )
    args = parser.parse_args(argv)
    
    # Ejecuta la lógica central con los argumentos parseados
    return main(simbolo=args.simbolo, evaluar=args.evaluar)

if __name__ == '__main__':
    entrypoint_modeling()

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
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
        '_variance_lag1_log',
        '_return_skew_lag1'
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

def separar_datos_live(
        df:pd.DataFrame,
        fecha:datetime
)->tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Divide el DataFrame en conjuntos de entrenamiento y prueba basándose en la fecha más reciente.
    '''
    df_train = df[df.index < fecha]
    df_live = df[df.index == fecha].copy()

    return df_train, df_live
    

def categorizar_en_cuantiles(
    df: pd.DataFrame,
    columna: str,
    num_cuantiles: int = 5,
    quitar_original: bool = True,
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
    df_out = df.copy()
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


def split_train_live(
    df_train: pd.DataFrame,
    df_live: pd.DataFrame,
    col_target:str, 
    cols_excluded: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    '''
    
    Tramifica la variable continua objetivo en clases. Es necesario hacerlo luego de separar los 
    datos de la ultima fecha para evitar data leakage al calcular las clases.
    '''
    df_live.loc[:, col_target] = np.nan # se asigna a nan para evitar error al eliminar columnas

    # Columnas a descartar de predictores
    cols_excluded.extend([col_target])
    columnas_a_descartar = set(cols_excluded)
    cols_drop = [c for c in columnas_a_descartar if c in df_train.columns]

    X_train = df_train.drop(columns=cols_drop)
    y_train = df_train[col_target]
    
    X_live = df_live.drop(columns=cols_drop)

    return X_train, y_train, X_live


def preprocesar_forex(
        par:str,
        variables_a_excluir:list[str]
)->tuple[pd.DataFrame, pd.Series, pd.DataFrame, datetime]:
    # Cargar fichero de precios
    df = cargar_datos_clean()
    df = filtrar_pares_restantes(df,par)

    # Separar datos de entrenamiento y test  
    ultima_fecha = df.index.max()
    df_train, df_live = separar_datos_live(df=df, fecha=ultima_fecha)

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

    # Separar predictores y variable objetivo
    variables_a_excluir.extend([objetivo])

    X_train, y_train, X_live = split_train_live(df_train=df_train, df_live=df_live, col_target=objetivo_cat,cols_excluded=variables_a_excluir)

    return X_train, y_train, X_live, ultima_fecha     

# Funciones de evaluacion

def evaluar_modelo_cat(
    model,
    X,
    y,
    test_size=0.20,
    random_state=42,
    run_permutation_importance=False,
    class_names=None,
):
    """Evalúa exhaustivamente un RandomForest multiclase mediante partición temporal

    o secuencial (80% entrenamiento, 20% validación final), reporte por clase,
    matrices de confusión y análisis de importancia de variables.
    """
    feature_names = (
        X.columns.tolist()
        if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    # 1. Partición secuencial (80% train / 20% val preservando el orden temporal)
    split_idx = int(len(X) * (1 - test_size))

    if hasattr(X, "iloc"):
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

    # 2. Entrenamiento del estimador sobre la partición del 80%
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

    # 3. Métricas equivalentes calculadas sobre el set de validación
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

    # 4. Reporte por clase y matriz de confusión (evaluados sobre validación)
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

    # 5. Importancia de Variables (MDI)
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
    print("RESUMEN DE VALIDACIÓN (80% TRAIN / 20% VAL)")
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


def optimizar_parametros(
        simbolo:str='EURUSD'
):
    tscv = TimeSeriesSplit(
        n_splits=5 # cantidad de particiones temporales
    )   

    # Predictores a eliminar de los datasets train y live
    predictores_a_excluir = []
    X_train, y_train, _, _ = preprocesar_forex(par=simbolo, variables_a_excluir=predictores_a_excluir)

    print("Configurando GridSearchCV para {simbolo}")
    modelo_base = Pipeline(steps=[
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [400,450],
        'rf__max_depth': [4],
        'rf__min_samples_leaf': [33,36,39],
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
        cv=tscv,
        n_jobs=-1,
        verbose=2
    )

    # Entrenar con datos completos
    print(f"Iniciando búsqueda de hiperparámetros. Esto tomará unos minutos.")
    grid_search.fit(X_train, y_train)

    # Resultados
    print("\n--- Búsqueda Finalizada ---")
    print("Mejores hiperparámetros encontrados:")
    print(grid_search.best_params_)

# Predictores a excluir segun el par forex
excluir_dict = {
    'EURUSD' : [],
    'GBPUSD' : [], 
    'AUDUSD' : ['Oil_vol_directional_lag1_log'], 
    'USDCAD' : ['USDCHF_variance_lag1_log'],  
    'USDCHF' : [], 
    'USDJPY' : []
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

    X_train, y_train, X_live, ultima_fecha = preprocesar_forex(par=simbolo, variables_a_excluir=predictores_a_excluir)

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
    y_pred = pipe.predict(X_live)
    y_pred_proba = pipe.predict_proba(X_live)
    df_probabilidades = pd.DataFrame(y_pred_proba, columns=pipe.classes_)

    print(f"Entrenamiento modelo {simbolo} finalizado.")
    print(f"Dimensiones del dataset de entrenamiento:", X_train.shape)
    print(f"Categoria predicha para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_pred[0]}")
    print("Probabilidades:")
    print(df_probabilidades.head().to_string(index=False))

    # Evaluar modelo
    if evaluar:
        rf_model = pipe.named_steps['model']
        dict_evaluacion = evaluar_modelo_cat(model=rf_model, X=X_train, y=y_train)

        registrar_evaluacion_excel(
            dict_evaluacion=dict_evaluacion,
            simbolo=simbolo,
            ultima_fecha=ultima_fecha,
            y_pred=y_pred
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

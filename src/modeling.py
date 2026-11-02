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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, cross_validate, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
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


def remover_pares_restantes(
        df:pd.DataFrame,
        target:str
)->pd.DataFrame:
    columnas_clean = df.columns.tolist()
   
    simbolos_restantes = [col for col in forex_cols if col != target]

    # Se eliminan datos de los simbolos forex restantes
    cols_restantes = [
        col
        for s in simbolos_restantes
        for col in encontrar_columnas(columnas_clean, s)
    ]
    df = df.drop(columns=cols_restantes, errors='ignore')    
    return df

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
    df = remover_pares_restantes(df,par)

    # Separar datos de entrenamiento y test  
    ultima_fecha = df.index.max()
    df_train, df_live = separar_datos_live(df=df, fecha=ultima_fecha)

    # Se crea la variable categorica a partir del dataset de entrenamiento
    # Variable objetivo: se crea variable categorica por cuantiles
    objetivo = f"{par}_return"
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

# Transformacion de variables: escalar en una ventana movil

class RollingStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, window=20, min_periods=None, eps=1e-8):
        self.window = window
        self.min_periods = min_periods if min_periods else window
        self.eps = eps

    def fit(self, X, y=None):
        # En series temporales online/rolling no hay parámetros fijos que "aprender" de un set estático
        return self

    def transform(self, X):
        is_df = isinstance(X, pd.DataFrame)
        df = X.copy() if is_df else pd.DataFrame(X)

        # Calculamos sobre la ventana retrospectiva cerrada (shift 1)
        r_mean = (
            df.shift(1).rolling(window=self.window, min_periods=self.min_periods).mean()
        )
        r_std = (
            df.shift(1).rolling(window=self.window, min_periods=self.min_periods).std()
        )

        scaled = (df - r_mean) / (r_std + self.eps)

        return scaled if is_df else scaled.to_numpy()

# Funciones de evaluacion

def evaluar_modelo_cat(
    model,
    X,
    y,
    cv_splits=5,
    random_state=42,
    run_permutation_importance=False,
    class_names=None,
):
    """Evalúa exhaustivamente un RandomForest multiclase mediante validación cruzada estratificada,

    desglose de métricas por clase, matrices de confusión y análisis de
    importancia de variables.

    Parámetros:
    -----------
    model : estimator
        Modelo RandomForestClassifier ya instanciado o preentrenado.
    X : pd.DataFrame o np.ndarray
        Matriz de características (features).
    y : pd.Series o np.ndarray
        Etiquetas de clase categóricas o multiclase.
    cv_splits : int, default=5
        Número de folds para la validación cruzada.
        Semilla para reproducibilidad del particionado.
    run_permutation_importance : bool, default=True
        Calcula importancia por permutación sobre el conjunto completo.
    class_names : list, optional
        Nombres o etiquetas de las clases para reportes legibles.

    Retorna:
    --------
    dict
        Diccionario con métricas agregadas de CV, reporte por clase,
        matriz de confusión y DataFrames de importancia de variables.
    """
    feature_names = (
        X.columns.tolist()
        if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    if class_names is None:
        class_names = (
            [str(cls) for cls in np.unique(y)]
            if hasattr(model, "classes_") is False
            else [str(cls) for cls in model.classes_]
        )

    # Validación Cruzada: 5 particiones secuenciales 
    tscv = TimeSeriesSplit(
        n_splits=cv_splits
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
        "log_loss": "neg_log_loss",
    }

    cv_results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=tscv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # Resumen de métricas de CV (media ± desviación estándar)
    cv_summary = {
        "CV Accuracy": f"{cv_results['test_accuracy'].mean():.4f} (± {cv_results['test_accuracy'].std():.4f})",
        "CV Balanced Accuracy": f"{cv_results['test_balanced_accuracy'].mean():.4f} (± {cv_results['test_balanced_accuracy'].std():.4f})",
        "CV F1 (Macro)": f"{cv_results['test_f1_macro'].mean():.4f} (± {cv_results['test_f1_macro'].std():.4f})",
        "CV F1 (Weighted)": f"{cv_results['test_f1_weighted'].mean():.4f} (± {cv_results['test_f1_weighted'].std():.4f})",
        "CV Log Loss": f"{-cv_results['test_log_loss'].mean():.4f} (± {cv_results['test_log_loss'].std():.4f})",
        "Train vs Test Accuracy Gap": f"{(cv_results['train_accuracy'].mean() - cv_results['test_accuracy'].mean()):.4f}",
    }

    # Desempeño detallado y Matriz de Confusión
    # Asegura ajuste sobre los datos pasados si el estimador requiere cómputos directos
    #model.fit(X, y)
    y_pred = model.predict(X)

    # Reporte de clasificación estructurado por clase
    cls_report_dict = classification_report(
        y, y_pred, target_names=class_names, output_dict=True
    )
    df_cls_report = pd.DataFrame(cls_report_dict).transpose()

    # Matriz de confusión normalizada (por filas / valores verdaderos)
    cm_raw = confusion_matrix(y, y_pred)
    cm_norm = confusion_matrix(y, y_pred, normalize="true")
    df_cm = pd.DataFrame(
        cm_norm,
        index=[f"True: {c}" for c in class_names],
        columns=[f"Pred: {c}" for c in class_names],
    )

    # Importancia de Variables (Feature Importance)
    # MDI (Mean Decrease in Impurity) - nativo de RandomForest
    mdi_importances = model.feature_importances_
    df_importance = pd.DataFrame(
        {"feature": feature_names, "importance_mdi": mdi_importances}
    ).sort_values(by="importance_mdi", ascending=False)

    # Permutation Feature Importance (más robusto frente a cardinalidad alta)
    df_perm_importance = None
    if run_permutation_importance:
        perm_result = permutation_importance(
            model,
            X,
            y,
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

    # Salida en consola
    print("=" * 60)
    print("RESUMEN DE VALIDACIÓN CRUZADA")
    print("=" * 60)
    for metric, val in cv_summary.items():
        print(f"{metric:<30}: {val}")

    print("\n" + "=" * 60)
    print("REPORTE DE CLASIFICACIÓN DETALLADO (POR CLASE)")
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
        "cv_summary": cv_summary,
        "cv_raw_results": cv_results,
        "classification_report": df_cls_report,
        "confusion_matrix_raw": cm_raw,
        "confusion_matrix_normalized": df_cm,
        "feature_importance_mdi": df_importance,
        "feature_importance_permutation": df_perm_importance,
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
    cv_summary = dict_evaluacion.get("cv_summary", {})
    fila_resumen = {
        "run_id": run_id,
        "timestamp_registro": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "simbolo": simbolo,
        "intervalo": intervalo_defecto,
        "ultima_fecha": str(ultima_fecha),
        "y_pred": str(y_pred),
        # Métricas escalares agregadas
        "cv_accuracy": cv_summary.get("CV Accuracy", np.nan),
        "cv_balanced_acc": cv_summary.get("CV Balanced Accuracy", np.nan),
        "cv_f1_macro": cv_summary.get("CV F1 (Macro)", np.nan),
        "cv_f1_weighted": cv_summary.get("CV F1 (Weighted)", np.nan),
        "cv_log_loss": cv_summary.get("CV Log Loss", np.nan),
        "train_test_gap": cv_summary.get("Train vs Test Accuracy Gap", np.nan),
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


def optimizar_hiperparametros(
        simbolo:str='EURUSD'
):
    tscv = TimeSeriesSplit(
        n_splits=5 # cantidad de particiones temporales
    )   

    # Predictores a eliminar de los datasets train y live
    predictores_a_excluir = []
    X_train, y_train, X_live, ultima_fecha = preprocesar_forex(par=simbolo, variables_a_excluir=predictores_a_excluir)

    print("Configurando GridSearchCV para {simbolo}")
    modelo_base = Pipeline(steps=[
        ('scaler', RollingStandardScaler(window=20)),
        ('rf', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'rf__n_estimators': [450],
        'rf__max_depth': [4],
        'rf__min_samples_leaf': [50],
        'rf__min_samples_split': [4],
        'rf__max_samples': [0.4],
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


'''
Bloque principal

Ejecucion desde el terminal:
python -m src.modeling
'''
def main(
        simbolo:str
):
    # Predictores a eliminar de los datasets train y live
    predictores_a_excluir = []
    X_train, y_train, X_live, ultima_fecha = preprocesar_forex(par=simbolo, variables_a_excluir=predictores_a_excluir)

    # Inicializar modelo y entrenar
    pipe = Pipeline([
        #('scaler', RollingStandardScaler(window=10)), 
        ('model', RandomForestClassifier(
            random_state=42,
            n_estimators=450,
            max_depth=4,
            min_samples_leaf=50,
            min_samples_split= 4,
            max_samples= 0.4,
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
    print(f"Categoria predicha para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_pred[0]}")
    print("Probabilidades:")
    print(df_probabilidades.head().to_string(index=False))

    # Evaluar modelo
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
    args = parser.parse_args(argv)
    
    # Ejecuta la lógica central con los argumentos parseados
    return main(simbolo=args.simbolo)

if __name__ == '__main__':
    entrypoint_modeling()

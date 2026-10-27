'''
src/cat_modeling.py

Entrenamiento de modelo de Clasificacion
'''
import pandas as pd
from src.config import carpeta_datos, intervalo_defecto
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.transform import categorizar_en_cuantiles
import argparse

def cargar_datos(
        simbolo:str,
        intervalo:str='1d',
        leer_cols_exclude:bool=False
)->tuple[pd.DataFrame, list[str]]:
    fichero_precios = Path(carpeta_datos) / f"{simbolo}/datos_depurados_{simbolo}_{intervalo}.parquet"
    try:
        df = pd.read_parquet(fichero_precios)
        if leer_cols_exclude:
            fichero_cols_exclude = Path(carpeta_datos) / f"{simbolo}/cols_exclude_{simbolo}_{intervalo}.csv"
            try:                
                df_cols_exclude = pd.read_csv(fichero_cols_exclude, header=None)
                cols_exclude = df_cols_exclude[0].to_list()
                return df, cols_exclude
            except Exception as e:
                print(f"Ocurrio un error al leer el fichero de columnas: {e}")

        return df, []
    except FileNotFoundError:
        print(f"Error: El archivo '{fichero_precios}' no existe en la ruta especificada.")
        return pd.DataFrame(), []
    
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame(), []


def split_ultimo(
    df: pd.DataFrame, 
    label: str, 
    cols_excluded: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Divide el DataFrame en conjuntos de entrenamiento y prueba basándose en la fecha más reciente.
    '''
    
    # Identificamos cuál es la fecha máxima en todo el dataset
    ultima_fecha = df.index.max()
    
    # Separamos mediante máscaras booleanas
    df_train = df[df.index < ultima_fecha]
    df_test = df[df.index == ultima_fecha]

    # Se descartan las columnas excluidas
    columnas_a_descartar = set(cols_excluded)
    cols_drop = [c for c in columnas_a_descartar if c in df.columns]

    X_train = df_train.drop(columns=cols_drop)
    y_train = df_train[label]
    
    X_test = df_test.drop(columns=cols_drop)
    y_test = df_test[label]
    
    return X_train, X_test, y_train, y_test

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.cat_modeling
'''
def main(
        simbolo:str="SPY",
        intervalo:str=intervalo_defecto
):
    # Cargar fichero de precios
    df, cols_absolutas = cargar_datos(simbolo=simbolo, intervalo=intervalo, leer_cols_exclude=True)
    df_clean = df.drop(columns=cols_absolutas)
    posibles_labels = ['High_pct_diff_Open', 'Low_pct_diff_Open', 'Close_pct_diff_Open']

    # Modelo
    nombre_modelo = 'High'
    objetivo = f"{nombre_modelo}_pct_diff_Open"
    objetivo_cat = f"{objetivo}_cat"

    # Crear variable categorica
    df_clean, tabla_bins = categorizar_en_cuantiles(
        df_clean, 
        columna=objetivo, 
        num_cuantiles=3, 
        quitar_original=False
        )

    print("\n--- Rangos por Categoría ---")
    print(tabla_bins)

    
    X_train, X_live, y_train, _ = split_ultimo(df=df_clean, label=objetivo_cat, cols_excluded=posibles_labels+[objetivo_cat])

    # Inicializar y entrenar
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=30,
        min_samples_split= 50,
        max_samples= 0.3,
        max_features=0.25,
        class_weight='balanced'        
    )
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_live)
    ultima_fecha = X_live.index.max()
    ultimo_open = df.loc[ultima_fecha, 'Open']

    print(f"Entrenamiento modelo {nombre_modelo} finalizado.")
    print(f"Categoria predicha para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_pred[0]}")
    print(f"Ultimo Open: {ultimo_open}")

    # Validacion cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        estimator=model, X=X_train, y=y_train, cv=tscv, scoring="accuracy"
    )

    # Mostrar resultados
    print(f"Accuracy promedio Time Series CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simbolo",
        type=str,
        default="SPY",
        help="Símbolo a cargar (por defecto: SPY)",
    )

    args = parser.parse_args()
    main(simbolo=args.simbolo)

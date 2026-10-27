'''
src/modeling.py

Modulo para la fase de modelado
'''
import pandas as pd
from src.config import carpeta_datos
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

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
python -m src.reg_modeling
'''
def main():
    # Cargar fichero de precios
    simbolo = 'SPY'
    intervalo = '1d'
    df, cols_absolutas = cargar_datos(simbolo=simbolo, intervalo=intervalo, leer_cols_exclude=True)
    df_clean = df.drop(columns=cols_absolutas)
    posibles_labels = ['High_pct_diff_Open', 'Low_pct_diff_Open', 'Close_pct_diff_Open']

    # Modelo
    nombre_modelo = 'High'
    objetivo = f"{nombre_modelo}_pct_diff_Open"
    X_train, X_live, y_train, _ = split_ultimo(df=df_clean, label=objetivo, cols_excluded=posibles_labels)

    # Inicializar y entrenar
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_live)
    ultima_fecha = X_live.index.max()
    ultimo_open = df.loc[ultima_fecha, 'Open']
    precio_predicho = ultimo_open * (1+ y_pred[0])

    print(f"Entrenamiento modelo {nombre_modelo} finalizado.")
    print(f"Valor predicho para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_pred[0]}")
    print(f"Ultimo Open: {ultimo_open}")
    print(f"Precio predicho: {precio_predicho}")

    # Validacion cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        estimator=model, X=X_train, y=y_train, cv=tscv, scoring="r2"
    )

    # Mostrar resultados por cada pliegue cronológico
    print("Resultados de la Validación Cruzada:")
    print(f"R² promedio: {cv_scores.mean():.4f}")



if __name__ == '__main__':
    main()
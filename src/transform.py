'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from src.config import carpeta_datos
from pathlib import Path
from typing import List

def cargar_precios(
        simbolo:str,
        intervalo:str='1d'
)->pd.DataFrame:
    fichero_precios = Path(carpeta_datos) / f'{simbolo}/precios_{simbolo}_{intervalo}.parquet'
    try:
        df = pd.read_parquet(fichero_precios)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{fichero_precios}' no existe en la ruta especificada.")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame()


# Feature Engineering

def calcular_medias_moviles(
        df:pd.DataFrame,
        columna: str,
        periodos: List[int],
        min_periodos: int | None = None
)->pd.DataFrame:
    '''
    Calcula la media movil simple (SMA) de una columna para multiples ventanas
    '''
    serie_origen = df[columna]

    # Genera todas las series en un diccionario en memoria
    nuevas_columnas = {
        f"{columna}_sma{p}": serie_origen.rolling(
            window=p, min_periods=min_periodos
        ).mean()
        for p in periodos
    }

    # Concatena el bloque completo al DataFrame original de una sola vez
    return pd.concat([df, pd.DataFrame(nuevas_columnas, index=df.index)], axis=1)

def calcular_extremos(
        df:pd.DataFrame,
        columna:str,
        periodo:int=20,
        min_periodos: int | None = None
)-> pd.DataFrame:
    '''
    Calcula los valores maximos y minimos en el periodo especificado.
    '''
    serie_origen = df[columna]
    nuevas_columnas = {
        f"{columna}_max{periodo}": serie_origen.rolling(
            window=periodo, min_periods=min_periodos
        ).max(),
        f"{columna}_min{periodo}": serie_origen.rolling(
                    window=periodo, min_periods=min_periodos
                ).min()
    }
    return pd.concat([df, pd.DataFrame(nuevas_columnas, index=df.index)], axis=1)


def aplicar_lag(
        df:pd.DataFrame,
        columnas:List[str],
        lag: int=1,
        quitar_original:bool=True
) ->pd.DataFrame:
    '''
    Aplica un desfase temporal (.shift) sobre las columnas indicadas.
    Opcionalmente elimina las versiones sin lag.
    '''
    bloque_lag = df[columnas].shift(lag)
    bloque_lag.columns = [f"{col}_lag{lag}" for col in columnas]

    # Si se descartan las columnas originales, se quitan antes de concatenar
    df_base = df.drop(columns=columnas) if quitar_original else df

    return pd.concat([df_base, bloque_lag], axis=1)


def calcular_variaciones_pct(
    df: pd.DataFrame,
    col_base: str,
    cols_target: List[str],
    lag_base: int = 0,
) -> pd.DataFrame:
    '''
    Calcula la diferencia porcentual de múltiples columnas target
    respecto a una única columna base mediante broadcasting vectorial.
    '''
    # Desplazar la serie base
    base = df[col_base].shift(lag_base)

    # Operación matricial: targets - base / base
    sub_df = df[cols_target]

    pct_diffs = (
        sub_df.sub(base, axis=0)
        .div(base, axis=0)
        .replace([np.inf, -np.inf], np.nan) # reemplaza inf/-inf generados por eventuales ceros en la base
    )

    # Renombrar columnas
    pct_diffs.columns = [f"{col}_pct_diff_{col_base}" for col in cols_target]

    # Concatenar nuevas columnas
    return pd.concat([df, pct_diffs], axis=1)

# EDA

def mostrar_asimetrias(df:pd.DataFrame):
    """
    Muestra la asimetría de las columnas numéricas del DataFrame.
    """
    print(df.select_dtypes(include="number").skew().sort_values(ascending=False).to_string())


def guardar_datos(
        df:pd.DataFrame,
        simbolo:str,
        intervalo:str
        ):
    fichero = Path(carpeta_datos) / f'{simbolo}/datos_depurados_{simbolo}_{intervalo}.parquet'

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.transform
'''
def main():
    # Cargar fichero de precios
    simbolo = 'SPY'
    intervalo = '1d'
    df = cargar_precios(simbolo=simbolo, intervalo=intervalo)

    # Capturar columnas originales
    cols_originales = [col for col in df.columns]

    # Feature Engineering

    # Calcular medias moviles
    periodos_ma = [5,10,20]

    # Precios Open: sin lag al aplicar luego de la apertura
    df = df.pipe(
        lambda d: calcular_medias_moviles(
            d,
            columna="Open",
            periodos= periodos_ma
        )
    )

    # Precios Close: lag = 1 para evitar Lookahead Bias
    df = (
        df.pipe(
            lambda d: calcular_medias_moviles(
                d, columna="Close", periodos=periodos_ma
            )
        ).pipe(
            lambda d: aplicar_lag(
                d,
                # Selecciona las columnas recien creadas
                columnas=[f"Close_sma{p}" for p in periodos_ma],
                lag=1,
            )
        )
    )
    # Capturar columnas de medias moviles
    cols_sma = [col for col in df.columns if "_sma" in col]

    # Volumen: tasa de cambio porcentual respecto a su promedio
    periodo = 20
    columna = "Volume"
    columna_sma = f"{columna}_sma{periodo}"
    columna_pct_diff = f"{columna}_pct_diff_{columna_sma}"


    df = (
        df.pipe(
            lambda d: calcular_medias_moviles(
                d, columna=columna, periodos=[periodo]
            )
        ).pipe(
            lambda d: calcular_variaciones_pct(
                d,
                col_base=columna_sma,
                cols_target=[columna],
                lag_base=1,  # Base: SMA calculada hasta t-1
            )
        ).pipe(
            lambda d: aplicar_lag(
                d,
                columnas=[columna_pct_diff],
                lag=1,
                quitar_original=True,  
            )
        )
    )

    # Calcular cierres maximos y minimos con lag de 1 dia
    columna = 'Close'
    periodo = 20
    columnas_extremos = [f"{columna}_max{periodo}", f"{columna}_min{periodo}"]
    df = (
        df.pipe(
            lambda d: calcular_extremos(
                d,
                columna= columna,
                periodo=periodo
            )
        ).pipe(
            lambda d: aplicar_lag(
                d,
                columnas=columnas_extremos,
                lag=1,
                quitar_original=True
            )
        )
    )
    columnas_extremos_lag = [f"{col}_lag1" for col in columnas_extremos]


    # Transformar columnas a variaciones porcentuales con respecto a Open del mismo dia
    posibles_labels = ['High', 'Low', 'Close'] # High, Low y Close del mismo dia
    targets = cols_sma + posibles_labels + columnas_extremos_lag
    df = df.pipe(
        lambda d: calcular_variaciones_pct(
            d,
            col_base='Open',
            cols_target=targets,
            lag_base=0,
        )
    )
    # Lag de 1 para usar valores previos de High, Low y Close como predictores

    df = df.pipe(
            lambda d: aplicar_lag(
                d,
                columnas=[f"{col}_pct_diff_Open" for col in posibles_labels],
                lag=1,
                quitar_original=False,  
            )
        )


    # Transformar Open a variacion porcentual respecto de Close del dia anterior
    df = df.pipe(
            lambda d: calcular_variaciones_pct(
                d,
                col_base='Close',
                cols_target=['Open'],
                lag_base=1,
            )
        )

    # Eliminar columnas no aplicables al modelado
    columnas_a_quitar = cols_originales + cols_sma + columnas_extremos_lag + ['Volume_sma20']
    df = df.drop(columns=columnas_a_quitar)

    # Eliminar valores missings iniciales
    df = df.dropna()

    df.info()
    guardar_datos(df=df, simbolo=simbolo, intervalo=intervalo)



if __name__ == '__main__':
    main()

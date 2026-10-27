'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from src.config import carpeta_datos
from pathlib import Path
from typing import List
from sklearn.preprocessing import PowerTransformer

# Manejo de datos

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
        quitar_original:bool=False
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


def calcular_varianza(
        df:pd.DataFrame,
        col:str,
        periodo:int=20,
        min_periodos: int | None = None
)->pd.DataFrame:
    serie_origen = df[col]
    nueva_columna = {
        f"{col}_var{periodo}": serie_origen.rolling(
            window= periodo, min_periods= min_periodos
        ).var()
    }
    return pd.concat([df, pd.DataFrame(nueva_columna, index=df.index)], axis=1)


# Transformaciones

def transformar_log(df: pd.DataFrame, cols: list, calculation_1p: bool = False) -> pd.DataFrame:
    """
    Aplica la transformación logarítmica a las columnas especificadas del DataFrame.
    Se crea una nueva columna con el sufijo '_log' o '_log1p' y se elimina la columna original.
    """
    # Creamos una copia para no alterar el DataFrame original accidentalmente
    df_out = df.copy()
    
    # Se define la función y el sufijo
    if calculation_1p:
        funcion_log = np.log1p
        sufijo = '_log1p'
    else:
        funcion_log = np.log
        sufijo = '_log'
        
    # Se aplica la función y elimina la columna original
    for col in cols:
        df_out[f'{col}{sufijo}'] = funcion_log(df_out[col])
        df_out.drop(col, axis=1, inplace=True)
        
    return df_out


def transformar_yeo(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Aplica la transformación de Yeo-Johnson a las columnas especificadas del DataFrame.
    Se crea una nueva columna con el sufijo '_yeo' y se elimina la columna original.
    """
    df_out = df.copy()

    # Inicializar el PowerTransformer
    pt = PowerTransformer(method='yeo-johnson', standardize=True)

    # Se aplica la función y elimina la columna original
    for col in cols:
        df_out[f'{col}_yeo'] = pt.fit_transform(df_out[[col]])
        df_out.drop(col, axis=1, inplace=True)
        
    return df_out



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

    # Feature Engineering

    # Columnas lag1 para evitar lookahead bias
    columnas_lag = ['Close', 'Volume', 'High', 'Low']

    df = df.pipe(
        lambda d: aplicar_lag(
            d,
            columnas=columnas_lag,
            lag=1,
            quitar_original=False
        )
    )

    # Calcular medias moviles
    periodos_ma = [20,50]

    # Precios Close: lag = 1 para evitar Lookahead Bias
    df = df.pipe(
            lambda d: calcular_medias_moviles(
                d, columna="Close_lag1", periodos=periodos_ma, min_periodos=20
            ) # min_periodos = 20: empieza a calcular la media larga antes, reduciendo missings iniciales
    )

    # Capturar columnas de medias moviles
    cols_sma = [col for col in df.columns if "_sma" in col]

    # Volumen: tasa de cambio porcentual respecto a su promedio
    periodo = 20
    col_volumen = "Volume_lag1"
    col_volumen_sma = f"{col_volumen}_sma{periodo}"

    df = df.pipe(
            lambda d: calcular_medias_moviles(
                d, columna=col_volumen, periodos=[periodo]
            )
        )

    # Calcular cierres maximos y minimos con lag de 1 dia
    columna = 'Close_lag1'
    periodo = 20
    columnas_extremos = [f"{columna}_max{periodo}", f"{columna}_min{periodo}"]
    df = df.pipe(
            lambda d: calcular_extremos(
                d,
                columna= columna,
                periodo=periodo
            )
        )

    # Capturar columnas absolutas
    cols_absolutas = [col for col in df.columns]

    # Calcular variaciones porcentuales

    # Volumen
    df = df.pipe(
            lambda d: calcular_variaciones_pct(
                d,
                col_base=col_volumen_sma,
                cols_target=[col_volumen],
                lag_base=1,  # Base: SMA calculada hasta t-2
            )
        )


    # Transformar columnas a diferencias porcentuales con respecto a Open del mismo dia
    columnas_diff_open = cols_sma + columnas_extremos + [
        'High', 'Low', 'Close', 'High_lag1', 'Low_lag1', 'Close_lag1'
        ] # High, Low y Close sin lag para posibles labels / con lag para predictores

    df = df.pipe(
        lambda d: calcular_variaciones_pct(
            d,
            col_base='Open',
            cols_target=columnas_diff_open,
            lag_base=0,
        )
    )

    # Obtener rendimiento y varianza diaria con lag: Cierre en t-1 vs Cierre en t-2
    df = df.pipe(
                lambda d: calcular_variaciones_pct(
                    d,
                    col_base='Close',
                    cols_target=['Close_lag1'],
                    lag_base=2,
                )
            )

    # Renombrar columna
    columna_nueva = 'Daily_return_lag1'
    df = df.rename(columns={'Close_lag1_pct_diff_Close' : columna_nueva})

    df = df.pipe(
        lambda d: calcular_varianza(
            d,
            col=columna_nueva,
            periodo=20,
        )
    )

    # Transformar variables asimetricas
    columnas_log = ['Daily_return_lag1_var20']
    columnas_yeo = ['Close_lag1_max20_pct_diff_Open', 'High_lag1_pct_diff_Open']
    df = transformar_log(df, cols=columnas_log)
    df = transformar_yeo(df, cols=columnas_yeo)


    # Eliminar columnas no aplicables al modelado
    df = df.drop(columns=cols_absolutas)

    # Eliminar valores missings iniciales
    df = df.dropna()

    df.info()
    guardar_datos(df=df, simbolo=simbolo, intervalo=intervalo)


if __name__ == '__main__':
    main()

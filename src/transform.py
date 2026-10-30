'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

from sklearn.preprocessing import PowerTransformer

from src.config import carpeta_datos, ACTIVOS, forex_cols



# Manejo de datos

def cargar_datos_raw(
        nombre_fichero:str
)->pd.DataFrame:
    fichero = Path(carpeta_datos) / f"raw_data/{nombre_fichero}.parquet"
    try:
        df = pd.read_parquet(fichero)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{fichero}' no existe en la ruta especificada.")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame()


def guardar_datos_clean(
        df:pd.DataFrame
        ):
    fichero = Path(carpeta_datos) / "clean_data/datos_depurados.parquet"

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')


# Feature Engineering

def calcular_pct_diff(
       serie_target:pd.Series,
       serie_base:pd.Series,
       lag_base:int=0 
)->pd.Series:
    base = serie_base.shift(lag_base)
    pct_diff = (serie_target.sub(base, axis=0)
                .div(base, axis=0)
                .replace([np.inf, -np.inf], np.nan))

    return pct_diff

                                                                    
def calcular_metricas_close(
        df:pd.DataFrame,
        cols:list[str],
        periodo_corto:int=20,
        periodo_largo:int=50,
        min_periodos:int | None = None
)->pd.DataFrame:
    '''
    Calcula metricas sobre los precios de cierre:
    - Retornos
    - Varianza de los retornos en el periodo corto
    - Maximo y minimo en el periodo corto
    - Medias moviles y diferencia entre medias

    Se expresan Maximo, Minimo y medias moviles relativos al precio Close.
    '''
    df_out = df.copy()
    for col in cols:
        # Retornos y varianzas de los retornos
        df_out[f"{col}_return"] = df_out[col].pct_change(fill_method=None)
        df_out[f"{col}_var{periodo_corto}"] = df_out[f"{col}_return"].rolling(window=periodo_corto, min_periods=min_periodos).var()

        # Maximos y minimos
        df_out[f"{col}_max{periodo_corto}"] = (df_out[col]
                                               .rolling(window=periodo_corto, min_periods=min_periodos)
                                               .max()
                                               )
        df_out[f"{col}_min{periodo_corto}"] = (df_out[col]
                                               .rolling(window=periodo_corto, min_periods=min_periodos)
                                               .min()
                                               )

        # Se expresan en diferencias porcentuales relativas al precio Close
        df_out[f"{col}_max{periodo_corto}"] = calcular_pct_diff(df_out[f"{col}_max{periodo_corto}"], df_out[col])
        df_out[f"{col}_min{periodo_corto}"] = calcular_pct_diff(df_out[f"{col}_min{periodo_corto}"], df_out[col])
               
        # Medias moviles
        df_out[f"{col}_sma{periodo_corto}"] = df_out[col].rolling(window=periodo_corto, min_periods=min_periodos).mean()
        df_out[f"{col}_sma{periodo_largo}"] = df_out[col].rolling(window=periodo_largo, min_periods=min_periodos).mean()

        # Diferencia porcentual entre medias
        df_out[f"{col}_sma{periodo_corto}_pct_diff_sma{periodo_largo}"] = calcular_pct_diff(
            df_out[f"{col}_sma{periodo_corto}"],
            df_out[f"{col}_sma{periodo_largo}"])
        
        # Se expresan las medias moviles relativas al precio Close
        df_out[f"{col}_sma{periodo_corto}"] = calcular_pct_diff(df_out[f"{col}_sma{periodo_corto}"],df_out[col])

        df_out[f"{col}_sma{periodo_largo}"] = calcular_pct_diff(df_out[f"{col}_sma{periodo_largo}"],df_out[col])

        # Se renombran las columnas relativas
        mapeo_relativas = {
            f"{col}_max{periodo_corto}" : f"{col}_max{periodo_corto}_pct_diff_Close",
            f"{col}_min{periodo_corto}" : f"{col}_min{periodo_corto}_pct_diff_Close",
            f"{col}_sma{periodo_corto}" : f"{col}_sma{periodo_corto}_pct_diff_Close",
            f"{col}_sma{periodo_largo}" : f"{col}_sma{periodo_largo}_pct_diff_Close"
        }
        df_out.rename(columns=mapeo_relativas, inplace=True)        

    return df_out


def calcular_metricas_volumen(
        df:pd.DataFrame,
        cols:list[str],
        periodo:int=20,
        min_periodos:int | None = None
)->pd.DataFrame:
    df_out = df.copy()
    for col in cols:
        # Calcular media movil del volumen
        df_out[f"{col}_sma{periodo}"] = df_out[col].rolling(window=periodo, min_periods=min_periodos).mean()

        # Se expresa el volumen relativo a su promedio (lag base=1)
        df_out[f"{col}_pct_diff_sma{periodo}"] = calcular_pct_diff(df_out[col], df_out[f"{col}_sma{periodo}"], lag_base=1)

        # Se elimina la columna del volumen promedio absoluto
        df_out.drop(f"{col}_sma{periodo}", axis=1, inplace=True)

    return df_out


def calcular_metricas_extremos(
        df:pd.DataFrame,
        simbolos:list[str]
)->pd.DataFrame:
    df_out = df.copy()
    for simbolo in simbolos:
        # High - Low Spread
        df_out[f"{simbolo}_High_Low_spread"] = calcular_pct_diff(df_out[f"{simbolo}_High"], df_out[f"{simbolo}_Low"])

        # Expresar High y Low relativos al precio Close
        df_out[f"{simbolo}_High_pct_diff_Close"] = calcular_pct_diff(df_out[f"{simbolo}_High"], df_out[f"{simbolo}_Close"])
        df_out[f"{simbolo}_Low_pct_diff_Close"] = calcular_pct_diff(df_out[f"{simbolo}_Low"], df_out[f"{simbolo}_Close"])

    return df_out


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

# Procesamiento

def encontrar_columnas(
        columnas_target:list[str],
        cadena:str
)->list:
    columnas_encontradas = [col for col in columnas_target if cadena in col]
    return columnas_encontradas

def procesar_precios()->pd.DataFrame:
    df_precios = cargar_datos_raw(nombre_fichero="precios")
    columnas_originales = df_precios.columns.tolist()

    # Metricas sobre Precios Close
    columnas_close = encontrar_columnas(
        columnas_target=columnas_originales,
        cadena="Close"
    )
    df_precios = calcular_metricas_close(df=df_precios, cols=columnas_close)

    # Metricas sobre Volumen
    columnas_volumen = encontrar_columnas(
        columnas_target=columnas_originales,
        cadena="Volume"
    )
    df_precios = calcular_metricas_volumen(df=df_precios, cols=columnas_volumen)

    # Metricas sobre precios extremos High - Low
    simbolos = list(ACTIVOS.values())
    df_precios = calcular_metricas_extremos(df=df_precios, simbolos=simbolos)

    # Eliminar columnas originales (absolutas)
    df_precios.drop(columns=columnas_originales, inplace=True)

    # Columnas restantes: relativas
    columnas_relativas = df_precios.columns.tolist()

    # Se aplica lag1 sobre los predictores para evitar Lookahead Bias
    # Lag sobre retornos Forex: sin eliminar el original para usar como labels
    retornos_forex = []
    for major in forex_cols:
        col_retorno_forex = encontrar_columnas(columnas_relativas, f"{major}_Close_return")[0]
        retornos_forex.append(col_retorno_forex)

    df_precios = aplicar_lag(df=df_precios, columnas=retornos_forex, quitar_original=False)

    # Resto de las columnas: se aplica el lag y se eliminan las originales
    excluir = set(retornos_forex)
    variables_restantes = [col for col in columnas_relativas if col not in excluir]

    df_precios = aplicar_lag(df=df_precios, columnas=variables_restantes, quitar_original=True)    


    return df_precios

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

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.transform
'''
def main(
):
    # Cargar fichero de precios    
    df_precios = procesar_precios()

    # Eliminar missings iniciales
    df_precios.dropna(inplace=True)

    print(df_precios.info())

    # Se guardan los datos depurados
    guardar_datos_clean(df_precios)


if __name__ == "__main__":
    main()

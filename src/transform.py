'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Sequence

from sklearn.preprocessing import PowerTransformer

from src.config import carpeta_datos, intervalo_defecto



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


def guardar_datos(
        df:pd.DataFrame,
        simbolo:str,
        intervalo:str,
        cols_exclude:list[str] | None = None
        ):
    fichero = Path(carpeta_datos) / f"{simbolo}/datos_depurados_{simbolo}_{intervalo}.parquet"

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')

    if cols_exclude:
        df_cols_exclude = pd.DataFrame(cols_exclude, columns=['Columna'])
        fichero_exclude = Path(carpeta_datos) / "cols_exclude.csv"
        df_cols_exclude.to_csv(fichero_exclude, index=False, encoding="utf-8", header=False)

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

    # Calcular metricas: Precios Close
    columnas_close = encontrar_columnas(
        columnas_target=columnas_originales,
        cadena="Close"
    )
    df_precios = calcular_metricas_close(df=df_precios, cols=columnas_close)

    return df_precios

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
    - Varianza de los retornos
    - Extremos: minimo y maximo
    - SMA: medias moviles
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

    # Calcular medias moviles [Media Corta, Media Larga]
    periodos_ma = [20,50]
    columna_sma = 'Close_lag1' # Precios Close lag1 para evitar Lookahead Bias
    col_sma_corta = f"{columna_sma}_sma{periodos_ma[0]}"
    col_sma_larga = f"{columna_sma}_sma{periodos_ma[1]}"
    
    df = df.pipe(
            lambda d: calcular_medias_moviles(
                d, columna= columna_sma, periodos=periodos_ma
            )
    )

    # Volumen promedio
    periodo_volumen = 20
    col_volumen = "Volume_lag1"
    col_volumen_sma = f"{col_volumen}_sma{periodo_volumen}"

    df = df.pipe(
            lambda d: calcular_medias_moviles(
                d, columna=col_volumen, periodos=[periodo_volumen]
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

    # Calcular columnas relativas: variaciones porcentuales

    # Diferencia entre medias
    df = df.pipe(
                lambda d: calcular_variaciones_pct(
                    d,
                    col_base=col_sma_larga,
                    cols_target=[col_sma_corta],
                    lag_base=0, 
                )
            )

    # Spread: diferencia entre H y L con lag1
    df = df.pipe(
                lambda d: calcular_variaciones_pct(
                    d,
                    col_base= 'Low_lag1',
                    cols_target=['High_lag1'],
                    lag_base=0, 
                )
            )


    # Volumen: tasa de cambio porcentual respecto a su promedio
    df = df.pipe(
            lambda d: calcular_variaciones_pct(
                d,
                col_base=col_volumen_sma,
                cols_target=[col_volumen],
                lag_base=1,  # Base: SMA calculada hasta t-2
            )
        )


    # Transformar columnas a diferencias porcentuales con respecto a Open del mismo dia
    columnas_diff_open = columnas_extremos + [
        'High', 'Low', 'Close', 'High_lag1', 'Close_lag1', col_sma_larga
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

    df = df.pipe(
        lambda d: calcular_varianza(
            d,
            col='Close_lag1_pct_diff_Close',
            periodo=20,
        )
    )

    # Renombrar columnas: Nombre viejo => Nombre nuevo
    renombrar_columnas = {
        'Close_lag1_pct_diff_Close' : 'Daily_return_lag1',
        'Close_lag1_pct_diff_Close_var20' : 'Daily_return_lag1_var20',
        'High_lag1_pct_diff_Low_lag1' : 'HL_spread_lag1',
        f'{col_sma_corta}_pct_diff_{col_sma_larga}' : f'{col_sma_corta}_pct_diff_sma{periodos_ma[1]}',
        f'Volume_lag1_pct_diff_Volume_lag1_sma{periodo_volumen}' : f'Volume_lag1_pct_diff_sma{periodo_volumen}'
    } 
    df = df.rename(columns=renombrar_columnas)



    # Transformar variables asimetricas
    columnas_log = ['Daily_return_lag1_var20', 'HL_spread_lag1']
    columnas_yeo = ['Close_lag1_max20_pct_diff_Open', 'High_lag1_pct_diff_Open']
    df = transformar_log(df, cols=columnas_log)
    df = transformar_yeo(df, cols=columnas_yeo)

    # Eliminar valores missings iniciales
    df = df.dropna()

    df.info()
    guardar_datos(
        df=df, 
        simbolo=simbolo, 
        intervalo=intervalo, 
        cols_exclude=cols_absolutas
        )


if __name__ == "__main__":
    main()

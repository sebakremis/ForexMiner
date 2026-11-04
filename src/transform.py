'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List

from sklearn.preprocessing import PowerTransformer

from src.config import (carpeta_datos, activos_cols, forex_cols,
                        otros_activos_cols
                        )

EPSILON = 1e-8

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


def calcular_zscore(
    serie: pd.Series,
    periodo: int = 20,
    min_periodos: int | None = None,
    asumir_media_cero: bool = False,
    exponencial: bool = True
) -> pd.Series:
    """Normaliza la vela actual (t) utilizando los parámetros históricos

    calculados hasta la vela anterior (t-1).
    """
    # 1. Parámetros calculados estrictamente hasta t-1
    if exponencial:
        ventana = serie.shift(1).ewm(
            span=periodo, min_periods=min_periodos, adjust=False
        )
    else:
        ventana = serie.shift(1).rolling(
            window=periodo, min_periods=min_periodos
        )

    desvio = ventana.std()

    # 2. La serie en curso (t) se compara contra la distribución previa (t-1)
    if asumir_media_cero:
        zscore = serie / (desvio + EPSILON)
    else:
        promedio = ventana.mean()
        zscore = (serie - promedio) / (desvio + EPSILON)

    zscore.name = f"{serie.name}_zscore" if serie.name else "zscore"
    return zscore


def calcular_volatilidad_direccional(
        retorno:pd.Series,
        span:int=20
)->pd.Series:
    downside_sq = pd.Series(
        np.where(retorno < 0, retorno**2, 0.0), index=retorno.index
    )
    upside_sq = pd.Series(
        np.where(retorno > 0, retorno**2, 0.0), index=retorno.index
    )
    # Media móvil exponencial de los desvíos al cuadrado
    ewma_down_sq = downside_sq.ewm(span=span, adjust=False).mean()
    ewma_up_sq = upside_sq.ewm(span=span, adjust=False).mean()

    vol_down = np.sqrt(ewma_down_sq)
    vol_up = np.sqrt(ewma_up_sq)

    ratio = vol_down / (vol_up + EPSILON)
    return ratio


def calcular_asimetria(
    serie: pd.Series,
    periodo: int = 50,
    min_periodos: int | None = None,
) -> pd.Series:
    """Calcula la asimetría (skewness) móvil contemporánea de una serie temporal.

    Parámetros:
    -----------
    serie : pd.Series
        Serie temporal (ej. retornos logarítmicos, spreads, etc.).
    periodo : int, por defecto 50
        Tamaño de la ventana móvil retrospectiva.
    min_periodos : int | None, opcional
        Número mínimo de observaciones requeridas. Si es None, requiere
        el período completo.

    Retorna:
    --------
    pd.Series con el sesgo móvil de la serie.
    """
    if min_periodos is None:
        min_periodos = periodo

    asimetria = serie.rolling(
        window=periodo, min_periods=min_periodos
    ).skew()

    asimetria.name = f"{serie.name}_skew" if serie.name else "skew"
    return asimetria

                                                                   
def calcular_metricas_precios(
        df:pd.DataFrame,
        simbolos:list[str],
        periodo_corto:int=20,
        periodo_largo:int | None = None,
        medias:list[int] | None = None,        
        periodo_rango:int | None = None,
        min_periodos:int | None = None
)->pd.DataFrame:
    '''
    Calcula metricas sobre los precios de cierre:
    - Retornos
    - Varianza de los retornos
    - Aceleraciones
    - Ratios y diferencias: si se especifica periodo_retorno_largo
    - Range position: posicion en el rango [min,max] para el periodo especificado
    - Diferencia entre medias moviles del periodo corto y largo [periodo_corto, periodo_largo]

    '''
    df_out = df.copy()
    for simbolo in simbolos:
        
        # Retornos, aceleraciones y Varianzas de los retornos
        columna_precio = f"{simbolo}_Close"
        varianza_col = f"{simbolo}_variance"

        retorno = df_out[columna_precio].pct_change(fill_method=None)               
        aceleracion = retorno.diff()
        df_out[varianza_col] = retorno.rolling(window=periodo_corto, min_periods=min_periodos).var()

        # Con los retornos puros se calcula la volatilidad direccional
        df_out[f"{simbolo}_vol_directional"] = calcular_volatilidad_direccional(retorno=retorno)

        # Se escala el retorno por su z-score, asumiendo media cero
        df_out[f"{simbolo}_return_zscore"] = calcular_zscore(retorno, asumir_media_cero=True)
        df_out[f"{simbolo}_acceleration_zscore"] = calcular_zscore(aceleracion)

        # Se calcula la asimetria sobre los retornos
        df_out[f"{simbolo}_return_skew"] = calcular_asimetria(serie=retorno)

        if periodo_largo:

            # Retorno: diferencia entre retorno corto y largo escalado           
            retorno_largo = df_out[columna_precio].pct_change(periodo_largo,fill_method=None)
            diferencia_retornos = retorno - retorno_largo
            df_out[f"{simbolo}_ret_diff_zscore"] = calcular_zscore(diferencia_retornos)

            # Lo mismo con Aceleraciones           
            aceleracion_larga = retorno_largo.diff()
            diferencia_aceleraciones = aceleracion - aceleracion_larga
            df_out[f"{simbolo}_accel_diff_zscore"] = calcular_zscore(diferencia_aceleraciones)


            # Ratio de volatilidad: ratio entre desvio corto y largo 
            desvio_corto = retorno_largo.rolling(window=periodo_corto).std()          
            desvio_largo = retorno_largo.rolling(window=periodo_largo).std()           
            df_out[f"{simbolo}_vol_ratio"] = (desvio_corto + EPSILON) / (desvio_largo + EPSILON)

        # Range position
        if periodo_rango:
            # Se calculan maximos y minimos
            maximo = (df_out[columna_precio]
                                .rolling(window=periodo_rango, min_periods=min_periodos)
                                .max()
                                )
            minimo = (df_out[columna_precio]
                                .rolling(window=periodo_rango, min_periods=min_periodos)
                                .min()
                                )

            # Se calcula la posicion dentro del rango [-1 (Close igual al Minimo), 1 (Close igual al Maximo)]
            df_out[f"{simbolo}_range{periodo_rango}_position"] = ((df_out[columna_precio] - minimo) - (maximo - df_out[columna_precio])) / (maximo-minimo)
              
        # Medias moviles
        if medias:
            periodo_sma_corta = medias[0]
            periodo_sma_larga = medias[1]
            sma_corta = df_out[columna_precio].rolling(window=periodo_sma_corta, min_periods=min_periodos).mean()
            sma_larga = df_out[columna_precio].rolling(window=periodo_sma_larga, min_periods=min_periodos).mean()

            # Diferencia porcentual entre medias
            df_out[f"{simbolo}_sma{periodo_sma_corta}_pct_diff_sma{periodo_sma_larga}"] = calcular_pct_diff(
                sma_corta,
                sma_larga)   

    return df_out


def calcular_spread(
        df:pd.DataFrame,
        simbolos:list[str]
)->pd.DataFrame:
    df_out = df.copy()
    for simbolo in simbolos:
        columna_high = f"{simbolo}_High"
        columna_low = f"{simbolo}_Low"

        # High - Low Spread escalado por z-score
        spread = calcular_pct_diff(df_out[columna_high], df_out[columna_low])
        df_out[f"{simbolo}_spread_zscore"] = calcular_zscore(spread)     

    return df_out


def calcular_metricas_intrasemanales(
        df:pd.DataFrame,
        simbolos:list[str]
)->pd.DataFrame:
    df_out = df.copy()
    for simbolo in simbolos:
        # Gap de apertura: diferencia entre apertura y cierre anterior
        open_gap = calcular_pct_diff(df_out[f"{simbolo}_Open"], df_out[f"{simbolo}_Close"], lag_base=1)
        df_out[f"{simbolo}_open_gap_zscore"] = calcular_zscore(open_gap)

        # Asimetria del open gap
        df_out[f"{simbolo}_gap_skew"] = calcular_asimetria(df_out[f"{simbolo}_open_gap_zscore"])

        # Diferencia entre el cierre y la apertura semanal (para construir target)
        df_out[f"{simbolo}_intra_weekly"] = calcular_pct_diff(df_out[f"{simbolo}_Close"], df_out[f"{simbolo}_Open"], lag_base=0)

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


def procesar_precios(
)->pd.DataFrame:
    df_precios = cargar_datos_raw(nombre_fichero="precios")
    columnas_originales = df_precios.columns.tolist()

    # Metricas sobre precios: forex

    df_precios = calcular_metricas_precios(
        df=df_precios, 
        simbolos=forex_cols,
        periodo_largo=12,
        medias=[15,50], # [media corta, media larga]
        periodo_rango=10
        )

    # Otros activos: sin calcular medias ni extremos
    df_precios = calcular_metricas_precios(
        df=df_precios, 
        simbolos=otros_activos_cols
        )

    # Metricas sobre Volumen: zscore
    columnas_volumen = encontrar_columnas(
        columnas_target=columnas_originales,
        cadena="Volume"
    )
    df_volumen_zscore = df_precios[columnas_volumen].apply(calcular_zscore)
    df_volumen_zscore.columns = [f"{col}_zscore" for col in df_volumen_zscore.columns]

    # Metricas sobre precios extremos High - Low para pares forex
    df_precios = calcular_spread(df=df_precios, simbolos=forex_cols)



    # Metricas intrasemanales: Rendimiento semanal entre precios Close y Open (target continuo)
    df_precios = calcular_metricas_intrasemanales(df=df_precios, simbolos=forex_cols)   

    # Eliminar columnas originales (absolutas)
    df_precios.drop(columns=columnas_originales, inplace=True)

    df_features = pd.concat([df_precios,df_volumen_zscore], axis=1)

    # Columnas restantes: relativas
    columnas_relativas = df_features.columns.tolist()

    # Se aplica lag1 sobre los predictores para evitar Lookahead Bias

    # Se excluyen las variables intrasemanales
    cols_gap = encontrar_columnas(columnas_relativas, "_open_gap")
    cols_intra = encontrar_columnas(columnas_relativas, "_intra_weekly")
    cols_exclude = set(cols_gap+cols_intra)

    # Resto de las columnas: se aplica el lag y se eliminan las originales
    variables_restantes = [col for col in columnas_relativas if col not in cols_exclude]

    df_features = aplicar_lag(df=df_features, columnas=variables_restantes, quitar_original=True)    


    return df_features


def procesar_datos_macro() -> pd.DataFrame:
    # Cargar datos diarios
    df_macro = cargar_datos_raw(nombre_fichero="datos_macro_daily")

    # Asegurar índice datetime ordenado y asegurar frecuencia de días hábiles
    df_macro.index = pd.to_datetime(df_macro.index)
    df_macro = df_macro.sort_index()

    # Cálculo de variaciones (días hábiles de mercado)
    periodo = 50

    # Variables de nivel (variación porcentual)
    df_macro["VIX_pct_chg"] = df_macro["VIX_index"].pct_change(
        periods=periodo, fill_method=None
    )
    df_macro["DXY_pct_chg"] = df_macro["DXY_broad_goods"].pct_change(
        periods=periodo, fill_method=None
    )

    # Variables expresadas en tasas/spreads (variación absoluta delta)
    df_macro["Real_yield_delta"] = df_macro["Real_yield_10y"].diff(
        periods=periodo
    )
    df_macro["STLFSI_delta"] = df_macro["St_louis_financial_stress"].diff(
        periods=periodo
    )

    # Se conserva VIX en nivel como predictor
    df_features_daily = pd.DataFrame(index=df_macro.index)
    df_features_daily["VIX_level"] = df_macro["VIX_index"]
    df_features_daily[f"VIX_pct_chg_{periodo}d"] = df_macro["VIX_pct_chg"]
    df_features_daily[f"DXY_pct_chg_{periodo}d"] = df_macro["DXY_pct_chg"]
    df_features_daily[f"Real_yield_delta_{periodo}d"] = df_macro["Real_yield_delta"]
    df_features_daily[f"STLFSI_delta_{periodo}d"] = df_macro["STLFSI_delta"]

    # Lags por retardo de publicación (en DÍAS HÁBILES de mercado)
    # VIX y Real Yield: Disponibles al cierre del día / apertura t+1 -> 1 día hábil
    cols_lag_1d = ["VIX_level", f"VIX_pct_chg_{periodo}d", f"Real_yield_delta_{periodo}d"]
    df_features_daily[cols_lag_1d] = df_features_daily[cols_lag_1d].shift(1)

    # DXY Broad: 2 días hábiles de rezago
    df_features_daily[f"DXY_pct_chg_{periodo}d"] = df_features_daily[
        f"DXY_pct_chg_{periodo}d"
    ].shift(2)

    # STLFSI4: Se calcula al viernes T, pero la Fed lo publica el JUEVES T+1.
    # Entre el viernes y el jueves hábil transcurren 4 días hábiles (Lun, Mar, Mié, Jue).
    df_features_daily[f"STLFSI_delta_{periodo}d"] = df_features_daily[
        f"STLFSI_delta_{periodo}d"
    ].shift(4)

    # Resampleo semanal estándar sobre los datos históricos observados
    df_weekly = df_features_daily.resample("W-FRI").last()

    # Alinear la etiqueta al lunes previo
    df_weekly.index = df_weekly.index - pd.Timedelta(days=4)
    df_weekly.index.name = "Date"

    # Aplicar shift(1) para modelado
    df_weekly_features = df_weekly.shift(1)

    # Forzar la inclusión de la semana actual si no está presente
    # Determinamos el lunes de la semana en curso
    hoy = pd.Timestamp.now().normalize()
    lunes_actual = hoy - pd.Timedelta(days=hoy.weekday())

    if lunes_actual not in df_weekly_features.index:
        # La información para predecir la semana actual es exactamente
        # el último estado consolidado sin shift (df_weekly.iloc[-1])
        df_weekly_features.loc[lunes_actual] = df_weekly.iloc[-1]

    # Descartar las primeras filas con NaNs debidas a los cálculos iniciales
    #df_weekly_features = df_weekly_features.dropna(how="all")

    # Feature engineering sobre variables macro
    #df_weekly_features["VIX_level_skew"] = calcular_asimetria(df_weekly_features["VIX_level"])


    return df_weekly_features


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
def main(
):
    # Cargar y procesar fichero de precios 
    df_precios = procesar_precios(
    )

    # Cargar y procesar datos macro
    df_macro = procesar_datos_macro()

    df_merged = df_precios.join(df_macro, how='inner')


    # Eliminar missings iniciales
    df_merged.dropna(inplace=True)

    # Transformaciones
    columnas_features = df_merged.columns.tolist()
    
    # Transformaciones log
    columnas_var = encontrar_columnas(columnas_features, f"_variance_")
    columnas_vol_directional = encontrar_columnas(columnas_features, f"_vol_directional_")
    

    columnas_log = columnas_var + columnas_vol_directional + [
        "USDCAD_vol_ratio_lag1"
    ]
    df_transform = transformar_log(df_merged, cols=columnas_log)

    # Transformaciones yeo-johnson
    cols_gap = encontrar_columnas(columnas_features, f"_open_gap")
    cols_spread = encontrar_columnas(columnas_features, f"_spread_")
    cols_volume = encontrar_columnas(columnas_features, f"_Volume_")

    columnas_yeo = cols_gap + cols_spread + cols_volume + [
        'VIX_pct_chg_50d'
        ]
    df_transform = transformar_yeo(df=df_transform, cols=columnas_yeo)

    # Se eliminan posibles missings luego de transformar
    df_transform.dropna(inplace=True)

    print(df_transform.info())

    # Se guardan los datos depurados
    guardar_datos_clean(df_transform)


if __name__ == "__main__":
    main()

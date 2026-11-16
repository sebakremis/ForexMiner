'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, TypedDict, Sequence
import argparse

from scipy.stats import yeojohnson, yeojohnson_normmax

from src.config import (carpeta_datos, forex_cols,
                        otros_activos_cols, yfinance_cols,
                        bond_cols
                        )

EPSILON = 1e-8

ultimos_yields: Dict[str, float] = {
    "USD_yield_avg": 4.94,
    "JPY_yield_avg": 2.675,
    "EUR_yield_avg": 3.253,
    "GBP_yield_avg": 5.154,
    "CHF_yield_avg": 0.436,
    "CAD_yield_avg": 3.645,
    "AUD_yield_avg": 4.997
}


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


def preprocesar_precios()->pd.DataFrame:
    df_precios = cargar_datos_raw(nombre_fichero="precios")
    # Se asignan a NaN valores negativos
    # Solo sobre columnas numéricas para evitar errores de tipo
    cols_num = df_precios.select_dtypes(include="number").columns
    df_precios[cols_num] = df_precios[cols_num].mask(df_precios[cols_num] < 0)
    return df_precios


def calcular_yield_spreads(df:pd.DataFrame):
    '''
    Calcula la diferencia entre las tasas de la moneda base y de cotizacion.
    Devuelve la variable escalada por el z-score.
    '''
    df_out = df.copy()
    columnas_originales = df_out.columns.tolist()

    for pair in forex_cols:
        base_currency = pair[:3]
        quote_currency = pair[3:6]

        serie_yield_spread = df_out[f"{base_currency}_yield_avg"] - df_out[f"{quote_currency}_yield_avg"]
        df_out[f"{pair}_yield_spread_zscore"] = calcular_zscore(serie_yield_spread, usar_mad=False, exponencial=True, asumir_media_cero=False)

    df_out.drop(columns=columnas_originales, inplace=True)
    return df_out


def preprocesar_macro()->tuple[pd.DataFrame, pd.DataFrame]:
    # Cargar datos diarios
    df_macro = cargar_datos_raw(nombre_fichero="datos_macro_daily")

    # Asegurar índice datetime ordenado y asegurar frecuencia de días hábiles
    df_macro.index = pd.to_datetime(df_macro.index)
    df_macro = df_macro.sort_index()
    df_macro = df_macro.ffill()

    # Separar columnas de rendimientos de bonos
    df_bonds = df_macro[bond_cols]
    df_macro.drop(columns=bond_cols, inplace=True)

    # Agrupa por año y mes natural y aplica la recolocación en el último registro disponible
    df_bonds = df_bonds.groupby(
        [df_bonds.index.year, df_bonds.index.month], group_keys=False
    ).apply(trasladar_df_mensual)

    # Resampleo semanal, trasladando el ultimo valor al lunes de cada semana
    df_bonds_weekly = df_bonds.resample("W-MON", label="left", closed="left").last()
    df_bonds_weekly.index.name = "Date"

    # Se imputan manualmente los ultimos yields
    df_bonds_weekly = imputar_manual(df_bonds_weekly)

    # Se imputan los huecos por interpolacion lineal
    df_bonds_weekly = imputar_interpolacion(df_bonds_weekly)

    # Se calculan los spread de tasas para cada par forex
    df_bonds_weekly = calcular_yield_spreads(df_bonds_weekly)

    return df_bonds_weekly, df_macro


def guardar_datos_clean(
        par:str,
        df:pd.DataFrame
        ):
    fichero = Path(carpeta_datos) / f"clean_data/{par}_clean.parquet"

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')


# Feature Engineering

def calcular_pct_diff(
    serie_target: pd.Series,
    serie_base: pd.Series,
    lag_base: int = 0
) -> pd.Series:
    '''
    Calcula diferencia porcentuales mediante logaritmos
    '''
    base = serie_base.shift(lag_base)
    # ln(target / base) = ln(target) - ln(base)
    log_diff = np.log(serie_target) - np.log(base)
    return log_diff.replace([np.inf, -np.inf], np.nan)


def calcular_zscore(
    serie: pd.Series,
    periodo: int = 20,
    min_periodos: int | None = None,
    asumir_media_cero: bool = False,
    exponencial: bool = True,
    usar_mad: bool = False,
) -> pd.Series:
    """Normaliza la vela actual (t) utilizando parámetros históricos hasta (t-1)."""
    if min_periodos is None:
        min_periodos = periodo

    # Desplazamiento estricto para evitar lookahead bias
    serie_previa = serie.shift(1)

    if usar_mad:
        # MAD requiere ventana rolling (pandas no soporta ewm.median() nativamente)
        if asumir_media_cero:
            centro = 0.0
            mad = (
                serie_previa.abs()
                .rolling(window=periodo, min_periods=min_periodos)
                .median()
            )
        else:
            centro = serie_previa.rolling(
                window=periodo, min_periods=min_periodos
            ).median()
            # Nota: calcula la desviación respecto a la mediana de cada punto histórico
            mad = (
                (serie_previa - centro)
                .abs()
                .rolling(window=periodo, min_periods=min_periodos)
                .median()
            )

        escala = 1.4826 * mad
        zscore = (serie - centro) / (escala + EPSILON)

    else:
        # calculo Mean / Std
        if exponencial:
            ventana = serie_previa.ewm(
                span=periodo, min_periods=min_periodos, adjust=False
            )
        else:
            ventana = serie_previa.rolling(
                window=periodo, min_periods=min_periodos
            )

        desvio = ventana.std()

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
    periodo: int = 40, 
    percentil: float | None = None
) -> pd.Series:
    """
    Calcula Kelly's Skewness sobre una ventana móvil en Pandas.
    
    Parámetros:
    -----------
    series : pd.Series
        Serie temporal univariada (ej. retornos, volumen diferencial, spread).
    window : int
        Tamaño de la ventana hacia atrás (W).
    p : float
        Percentil de cola inferior (ej. 0.05 para colas 5% y 95%).
    
    Regla practica para los parametros: W = (1/p) x 3
    """
    if percentil is None:
        percentil = round(3/periodo, 3)

    rolling_obj = serie.rolling(window=periodo, min_periods=periodo, center=False)
    
    q_sup = rolling_obj.quantile(1.0 - percentil)
    q_med = rolling_obj.quantile(0.50)
    q_inf = rolling_obj.quantile(percentil)
    
    numerador = (q_sup - q_med) - (q_med - q_inf)
    denominador = (q_sup - q_inf) + EPSILON
    
    kelly_skew = numerador / denominador
    kelly_skew.name = f"{serie.name}_skew"
    return kelly_skew

                                                                   
def calcular_metricas_precios(
        df:pd.DataFrame,
        simbolos:list[str],
        periodo_largo:int=40,
        periodo_corto:int | None = None,
        medias:list[int] | None = None,        
        periodo_rango:int | None = None,
        min_periodos:int | None = None
)->pd.DataFrame:
    '''
    Calcula metricas sobre los precios de cierre:
    - Retornos
    - Desvio estandar de los retornos
    - Aceleraciones
    - Ratios y diferencias: si se especifica periodo_retorno_largo
    - Range position: posicion en el rango [min,max] para el periodo especificado
    - Diferencia entre medias moviles corta y larga [media_corta, media_larga]

    '''
    df_out = df.copy()
    for simbolo in simbolos:        
        # Retornos logaritmicos para 1 periodo
        columna_precio = f"{simbolo}_Close"
        log_precios = np.log(df_out[columna_precio])
        retorno_1p = log_precios.diff()             
        
        # Desvio estandar sobre el periodo largo
        desvio_col = f"{simbolo}_std"
        df_out[desvio_col] = retorno_1p.rolling(window=periodo_largo, min_periods=min_periodos).std()

        # Velocidad media del precio para el periodo largo (retorno normalizado por longitud del periodo)
        vel_larga = log_precios.diff(periodo_largo) / periodo_largo
        df_out[f"{simbolo}_vel_zscore"] = calcular_zscore(vel_larga, periodo=periodo_largo, usar_mad=True, asumir_media_cero=True)

        # Aceleracion: tasa de cambio de la velocidad
        accel_larga = vel_larga.diff(periodo_largo) / periodo_largo
        df_out[f"{simbolo}_acceleration_zscore"] = calcular_zscore(accel_larga, periodo=periodo_largo, usar_mad=True, asumir_media_cero=True)

        # Con los retornos logaritmicos se calcula la volatilidad direccional
        df_out[f"{simbolo}_vol_directional"] = calcular_volatilidad_direccional(retorno=retorno_1p, span=periodo_largo)

        # Se escala el retorno por su z-score, asumiendo media cero
        df_out[f"{simbolo}_return_zscore"] = calcular_zscore(retorno_1p, periodo=periodo_largo, usar_mad=True, asumir_media_cero=True)
        
        # Se calcula la asimetria sobre los retornos
        df_out[f"{simbolo}_return_skew"] = calcular_asimetria(serie=retorno_1p, periodo=periodo_largo)

        if periodo_corto:
            # Diferencia entre aceleraciones           
            vel_corta = log_precios.diff(periodo_corto) / periodo_corto           
            accel_corta = vel_corta.diff(periodo_corto) / periodo_corto            
            diferencia_aceleraciones = accel_corta - accel_larga            
            df_out[f"{simbolo}_accel_diff_zscore"] = calcular_zscore(diferencia_aceleraciones, periodo=periodo_largo, usar_mad=True, asumir_media_cero=True)

            # Ratio de volatilidad: ratio entre desvio corto y largo
            desvio_corto = retorno_1p.rolling(window=periodo_corto).std()           
            df_out[f"{simbolo}_vol_ratio"] = (desvio_corto + EPSILON) / (df_out[desvio_col] + EPSILON)

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


def calcular_hl_spread(
        df:pd.DataFrame,
        simbolos:list[str]
)->pd.DataFrame:
    nuevas_columnas = {}
    for simbolo in simbolos:
        columna_high = f"{simbolo}_High"
        columna_low = f"{simbolo}_Low"

        # High - Low Spread: diferencia porcentual entre el precio máximo y mínimo
        spread = calcular_pct_diff(df[columna_high], df[columna_low])

        # Proteger contra bad ticks donde High < Low
        nuevas_columnas[f"{simbolo}_HL_spread"] = spread.clip(lower=0.0)
   

    return pd.concat([df, pd.DataFrame(nuevas_columnas, index=df.index)], axis=1)


def calcular_open_gap(
        df:pd.DataFrame,
        simbolos:list[str],
        periodo:int=20
)->pd.DataFrame:
    df_out = df.copy()
    for simbolo in simbolos:
        columna_open = f"{simbolo}_Open"
        columna_close = f"{simbolo}_Close"

        # Open Gap: diferencia porcentual entre el cierre anterior y la apertura actual
        open_gap = calcular_pct_diff(df_out[columna_open], df_out[columna_close], lag_base=1)

        # Se calcula el z-score y se complimen colas extremas
        z_gap = calcular_zscore(
            serie=open_gap,
            periodo=periodo,
            usar_mad=False, # Obligatorio: MAD sera 0 si mas de la mitad de las velas no tienen gap
            exponencial=True, # Adaptacion rapida a cambios
            asumir_media_cero=True, # preserva el signo del gap
        )
        df_out[f"{simbolo}_Open_gap_zscore"] = np.arcsinh(z_gap)

        # Asimetria del open gap
        df_out[f"{simbolo}_Open_gap_skew"] = calcular_asimetria(open_gap, periodo=periodo)

    return df_out
    

def calcular_log_rvol(
    serie_vol: pd.Series, periodo: int = 20, min_periodos: int | None = None
) -> pd.Series:
    """Calcula el logaritmo del Volumen Relativo (RVOL) estrictamente causal."""
    if min_periodos is None:
        min_periodos = periodo

    # Mediana histórica hasta t-1 para evitar distorsión por spikes previos
    vol_base = (
        serie_vol.shift(1)
        .rolling(window=periodo, min_periods=min_periodos)
        .median()
    )

    # RVOL centrado en 1.0 -> log(RVOL) centrado en 0.0
    rvol = (serie_vol + 1e-8) / (vol_base + 1e-8)
    return np.log(rvol)


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


def obtener_features_precios(
        df:pd.DataFrame,
        periodo_largo:tuple[int, int],
        periodo_corto:tuple,
        medias:tuple,
        rango:tuple
        )->pd.DataFrame:

    df_features = df.copy()
    columnas_originales = df_features.columns.tolist()
    # Metricas sobre precios: forex

    df_features = calcular_metricas_precios(
        df=df_features, 
        simbolos=forex_cols,
        periodo_largo=periodo_largo[0],
        periodo_corto=periodo_corto[0],
        medias=medias[0], 
        periodo_rango=rango[0]
        )

    # Otros activos
    df_features = calcular_metricas_precios(
        df=df_features,
        simbolos=otros_activos_cols,
        periodo_largo=periodo_largo[1],
        periodo_corto=periodo_corto[1],
        medias=medias[1], 
        periodo_rango=rango[1]        
        )

    # Metricas sobre Volumen: zscore
    columnas_volumen = encontrar_columnas(
        columnas_target=columnas_originales,
        cadena="Volume"
    )
    nuevas_cols = {}

    for col in columnas_volumen:
        nuevas_cols[f"{col}_relative_log"] = calcular_log_rvol(
            serie_vol=df_features[col], periodo=periodo_largo[1]
        )

    df_features = pd.concat(
        [df_features, pd.DataFrame(nuevas_cols, index=df_features.index)], axis=1
    )

    # Metricas sobre precios extremos High - Low para pares forex y otros activos
    df_features = calcular_hl_spread(df=df_features, simbolos=yfinance_cols)

    # Open gap para otros activos (se evita aplicar sobre forex para evitar fuga de datos)
    df_features = calcular_open_gap(df=df_features, simbolos=otros_activos_cols, periodo=periodo_largo[1])      

    # Eliminar columnas originales (absolutas)
    df_features.drop(columns=columnas_originales, inplace=True)

    # Columnas restantes: relativas
    columnas_relativas = df_features.columns.tolist()

    # Se aplica lag1 sobre los predictores para evitar Lookahead Bias

    # Retornos forex: se mantienen los retornos sin lag para usar como target (quitar_original=False)
    cols_retorno_fx = []
    for col in forex_cols:
        cols_retorno_fx.append(f"{col}_return_zscore")

    df_features = aplicar_lag(df=df_features, columnas=cols_retorno_fx, quitar_original=False)

    # Resto de las columnas: se aplica el lag y se eliminan las originales (quitar_original=True)
    cols_exclude = set(cols_retorno_fx)
    variables_restantes = [col for col in columnas_relativas if col not in cols_exclude]

    df_features = aplicar_lag(df=df_features, columnas=variables_restantes, quitar_original=True)    

    return df_features


def trasladar_df_mensual(grupo):
    # Creamos un bloque vacío con el mismo índice y columnas
    res = pd.DataFrame(np.nan, index=grupo.index, columns=grupo.columns)
    
    # Tomamos el primer valor no nulo de cada columna
    primeros = grupo.bfill().iloc[0]
    
    # Lo colocamos en la última fila del mes
    res.iloc[-1] = primeros
    return res


def imputar_manual(df:pd.DataFrame)->pd.DataFrame:
    '''
    Imputa manualmente los ultimos yields registrados en el diccionario ultimos_yields
    '''
    df_out = df.copy()
    # Filtrar solo las columnas que realmente existen en el DataFrame
    cols_comunes = [col for col in ultimos_yields if col in df_out.columns]
    
    if not cols_comunes or df_out.empty:
        return df_out

    # Asignación limpia en la última fecha/fila sin crear índices espurios
    ultima_fecha = df_out.index[-1]
    df_out.loc[ultima_fecha, cols_comunes] = [ultimos_yields[c] for c in cols_comunes]

    return df_out


def imputar_interpolacion(
    df: pd.DataFrame, 
    rellenar_extremos: bool = False
) -> pd.DataFrame:
    """
    Imputa valores NaN intermedios en un DataFrame mediante interpolación lineal.
    
    Parámetros:
    - df: DataFrame de entrada.
    - metodo: 'linear' para interpolación basada en posición/pasos iguales, 
              o 'time'/'index' para ponderar por la distancia real entre fechas.
    - rellenar_extremos: Si es True, aplica bfill y ffill para cubrir posibles NaNs
                         al inicio o final de las series que la interpolación no cubre.
    """
    df_out = df.copy()
    
    # Interpolación lineal a lo largo de las columnas (filas hacia abajo)
    df_out = df_out.interpolate(method='index', axis=0)
    
    # La interpolación no puede estimar NaNs antes del primer valor 
    # ni después del último; se cubren con forward/backward fill si se solicita
    if rellenar_extremos:
        df_out = df_out.bfill().ffill()
        
    return df_out


def obtener_features_macro(
        df_b:pd.DataFrame,
        df_m:pd.DataFrame,
        periodo_variaciones:int,
        periodo_skew:int
) -> pd.DataFrame:
    df_bonds_weekly = df_b.copy()
    df_macro = df_m.copy()

    # Se calculan las velocidades y asimetrias de los yield spreads
    columnas_bonos = df_bonds_weekly.columns.to_list()
    for col in columnas_bonos:
        df_bonds_weekly[f"{col}_skew"] = calcular_asimetria(df_bonds_weekly[col], periodo=periodo_skew)
        df_bonds_weekly[f"{col}_vel"] = df_bonds_weekly[col].diff(periodo_variaciones) / periodo_variaciones
   
    # Se conserva VIX en nivel como predictor
    df_features_daily = pd.DataFrame(index=df_macro.index)
    vix_level_col = "VIX_level"
    df_features_daily[vix_level_col] = df_macro["VIX_index"]

    # Variables de nivel (variación porcentual) mediante diferencia logaritmica
    vix_pct_col = f"VIX_vel_{periodo_variaciones}d"
    df_features_daily[vix_pct_col] = np.log(df_macro["VIX_index"]).diff(periods=periodo_variaciones) / periodo_variaciones

    dxy_pct_col = f"DXY_vel_{periodo_variaciones}d"
    df_features_daily[dxy_pct_col] = np.log(df_macro["DXY_broad_goods"]).diff(periods=periodo_variaciones) / periodo_variaciones

    # Variables expresadas en tasas/spreads (variación absoluta delta)
    yield_delta_col = f"Real_yield_vel_{periodo_variaciones}d"
    df_features_daily[yield_delta_col] = df_macro["Real_yield_10y"].diff(
        periods=periodo_variaciones
    ) / periodo_variaciones
    stl_delta_col = f"STLFSI_vel_{periodo_variaciones}d"
    df_features_daily[stl_delta_col] = df_macro["St_louis_financial_stress"].diff(
        periods=periodo_variaciones
    ) / periodo_variaciones

    # Lags por retardo de publicación (en DÍAS HÁBILES de mercado)
    # VIX y Real Yield: Disponibles al cierre del día / apertura t+1 -> 1 día hábil
    cols_lag_1d = [vix_level_col, vix_pct_col, yield_delta_col]
    df_features_daily[cols_lag_1d] = df_features_daily[cols_lag_1d].shift(1)

    # DXY Broad: 2 días hábiles de rezago
    df_features_daily[dxy_pct_col] = df_features_daily[dxy_pct_col].shift(2)

    # STLFSI4: Se calcula al viernes T, pero la Fed lo publica el JUEVES T+1.
    # Entre el viernes y el jueves hábil transcurren 4 días hábiles (Lun, Mar, Mié, Jue).
    df_features_daily[stl_delta_col] = df_features_daily[stl_delta_col].shift(4)

    # Resampleo semanal estándar sobre los datos históricos, alineando al lunes previo
    df_weekly = df_features_daily.resample("W-MON", label="left", closed="left").last()

    df_weekly.index.name = "Date"

    # Unir con rendimientos de bonos
    df_merged = pd.concat([df_weekly, df_bonds_weekly], axis=1) 

    # Aplicar shift(1) para modelado
    df_weekly_features = df_merged.shift(1)

    # Forzar la inclusión de la semana actual si no está presente
    # Determinamos el lunes de la semana en curso
    hoy = pd.Timestamp.now().normalize()
    lunes_actual = hoy - pd.Timedelta(days=hoy.weekday())

    if lunes_actual not in df_weekly_features.index:
        # La información para predecir la semana actual es exactamente
        # el último estado consolidado sin shift (df_merged.iloc[-1])
        df_weekly_features.loc[lunes_actual] = df_merged.iloc[-1]

    # Descartar las primeras filas con NaNs debidas a los cálculos iniciales
    #df_weekly_features = df_weekly_features.dropna(how="all")

    # Feature engineering sobre variables macro: asimetrias
    df_weekly_features["VIX_level_skew"] = calcular_asimetria(df_weekly_features[vix_level_col], periodo=periodo_skew)
    df_weekly_features["STLFSI_skew"] = calcular_asimetria(df_weekly_features[stl_delta_col], periodo=periodo_skew)


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


def transformar_yeo(
    df: pd.DataFrame,
    columnas: list[str],
    min_periodos: int = 100,
    step: int = 20,
    reemplazar: bool = True,
) -> pd.DataFrame:
    """Aplica Yeo-Johnson de forma estrictamente causal (ajustando lambda hasta t-1)

    sobre las columnas indicadas y les agrega el sufijo '_yj'.
    """
    nuevas_cols = {}

    for col in columnas:
        if col not in df.columns:
            continue

        vals = df[col].to_numpy(dtype=float)
        out = np.full(len(vals), np.nan)
        lmbda = None

        for i in range(min_periodos, len(vals)):
            # Re-estimar lambda cada `step` velas usando datos estrictamente hasta i-1
            if lmbda is None or (i - min_periodos) % step == 0:
                hist = vals[:i]
                hist = hist[~np.isnan(hist)]
                if len(hist) >= min_periodos:
                    lmbda = yeojohnson_normmax(hist)

            if lmbda is not None and not np.isnan(vals[i]):
                out[i] = yeojohnson(vals[i], lmbda)

        nuevas_cols[f"{col}_yj"] = out

    df_yj = pd.DataFrame(nuevas_cols, index=df.index)

    if reemplazar:
        cols_existentes = [c for c in columnas if c in df.columns]
        df_base = df.drop(columns=cols_existentes)
        return pd.concat([df_base, df_yj], axis=1)

    return pd.concat([df, df_yj], axis=1)

# Funcion de procesamiento con los parametros de cada par

class ParametrosPar(TypedDict):
    periodo_largo: tuple[int, int]
    periodo_corto: tuple[int, int]
    medias: tuple[list[int], list[int]]
    rango: tuple[int, int]
    periodo_variaciones: int
    periodo_asimetria: int

parametros_forex: dict[str, ParametrosPar] = {
    "EURUSD": {
        # Parametros de precios (Forex, Otros activos)
        "periodo_largo": (50, 50),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]), # [media corta, media larga]
        "rango": (15, 15),

        # Parametros macro
        "periodo_variaciones": 50,
        "periodo_asimetria": 60,
        },
    "GBPUSD": {
        # Parametros de precios 
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),
        "rango": (20, 10),

        # Parametros macro
        "periodo_variaciones": 55,
        "periodo_asimetria": 60,
        },
    "AUDUSD": {
        # Parametros de precios 
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),  
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 50,
        "periodo_asimetria": 60,
        },
    "USDJPY": {
        # Parametros de precios 
        "periodo_largo": (40, 40),
        "periodo_corto": (20, 25),
        "medias": ([20, 50], [15, 50]),  
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 35,
        "periodo_asimetria": 50,
    },
    "USDCAD": {
        # Parametros de precios
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 50,
        "periodo_asimetria": 40,
    },
    "USDCHF": {
        # Parametros de precios 
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 45,
        "periodo_asimetria": 50,
        }
}

def procesar_par(
        df_precios:pd.DataFrame,
        df_bonds:pd.DataFrame,
        df_macro:pd.DataFrame,
        parametro: ParametrosPar
)->pd.DataFrame:

    precios_features = obtener_features_precios(
        df=df_precios,
        periodo_largo = parametro["periodo_largo"],
        periodo_corto = parametro["periodo_corto"],
        medias = parametro["medias"],
        rango = parametro["rango"] 
        )

    macro_features = obtener_features_macro(
        df_b=df_bonds, 
        df_m=df_macro,
        periodo_variaciones=parametro["periodo_variaciones"],
        periodo_skew=parametro["periodo_asimetria"]
        )

    df_merged = precios_features.join(macro_features, how='inner')

    return df_merged


# Post proceso: transformaciones comunes y almacenamiento de ficheros de datos clean

def post_procesar_par(
        par:str,
        df:pd.DataFrame
):
    # Eliminar missings iniciales
    df.dropna(inplace=True)

    # Transformaciones
    columnas_features = df.columns.tolist()
    
    # Transformaciones log
    columnas_std = encontrar_columnas(columnas_features, f"_std_")
    columnas_vol_directional = encontrar_columnas(columnas_features, f"_vol_directional_")
    

    columnas_log = columnas_std + columnas_vol_directional
    df_transform = transformar_log(df, cols=columnas_log)

    # Transformaciones yeo-johnson
    cols_hl_spread = encontrar_columnas(columnas_features, f"_HL_spread_")

    columnas_yeo = cols_hl_spread 
    df_transform = transformar_yeo(
        df=df_transform,
        columnas=columnas_yeo,
        min_periodos=100,
        step=20,
        reemplazar=True,
    )

    # Se eliminan posibles missings luego de transformar
    df_transform.dropna(inplace=True)

    print(df_transform.info())

    # Se guardan los datos depurados
    guardar_datos_clean(par,df_transform)    



'''
Bloque principal

Ejecucion desde el terminal:
python -m src.transform
'''
def main(
        simbolo_consola:str | None = None
):    
    # Cargar y pre-procesar fichero de precios 
    precios_raw = preprocesar_precios()
    bonds_raw, macro_raw = preprocesar_macro()

    # Procesar pares
    if simbolo_consola:        
        pares_forex=[simbolo_consola]
    else:
        pares_forex= list(parametros_forex.keys())

    for par in pares_forex:
        parametros = parametros_forex.get(par)
        if parametros:
            df_par = procesar_par(
                df_precios=precios_raw,
                df_bonds=bonds_raw,
                df_macro=macro_raw,
                parametro=parametros
            )
            post_procesar_par(par, df_par)

                  
def entrypoint_transform(argv: Sequence[str] | None = None):
    """Punto de entrada de línea de comandos."""
    parser = argparse.ArgumentParser(description="Módulo de Modelado Categórico")

    parser.add_argument(
        "--simbolo", type=str, help="Símbolo del activo"
    )
    args = parser.parse_args(argv)

    simbolo_entrada=args.simbolo

    if simbolo_entrada:
    # Limpiar y validar que el simbolo tenga parametros configurados
        simbolo_limpio = simbolo_entrada.strip().upper()
        if simbolo_limpio not in parametros_forex:
                print(f"Aviso: El símbolo '{simbolo_limpio}' no tiene configuración en parametros_forex.")
                return
        return main(simbolo_limpio)
    return main()


if __name__ == "__main__":
    entrypoint_transform()

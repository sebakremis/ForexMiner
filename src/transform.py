'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from typing import Dict, TypedDict

from sklearn.preprocessing import PowerTransformer

from src.config import (carpeta_datos, forex_cols,
                        otros_activos_cols, bond_cols
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

        # Aceleracion: tasa de cambio de la velocidad
        accel_larga = vel_larga.diff(periodo_largo) / periodo_largo
        df_out[f"{simbolo}_acceleration_zscore"] = calcular_zscore(accel_larga)

        # Con los retornos logaritmicos se calcula la volatilidad direccional
        df_out[f"{simbolo}_vol_directional"] = calcular_volatilidad_direccional(retorno=retorno_1p)

        # Se escala el retorno por su z-score, asumiendo media cero
        df_out[f"{simbolo}_return_zscore"] = calcular_zscore(retorno_1p, asumir_media_cero=True)
        
        # Se calcula la asimetria sobre los retornos
        df_out[f"{simbolo}_return_skew"] = calcular_asimetria(serie=retorno_1p, periodo=periodo_largo)

        if periodo_corto:
            # Diferencia entre aceleraciones           
            vel_corta = log_precios.diff(periodo_corto) / periodo_corto           
            accel_corta = vel_corta.diff(periodo_corto) / periodo_corto            
            diferencia_aceleraciones = accel_corta - accel_larga            
            df_out[f"{simbolo}_accel_diff_zscore"] = calcular_zscore(diferencia_aceleraciones)

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
    df_out = df.copy()
    for simbolo in simbolos:
        columna_high = f"{simbolo}_High"
        columna_low = f"{simbolo}_Low"

        # High - Low Spread escalado por z-score
        spread = calcular_pct_diff(df_out[columna_high], df_out[columna_low])
        df_out[f"{simbolo}_HL_spread_zscore"] = calcular_zscore(spread)     

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
        intra_semanal = calcular_pct_diff(df_out[f"{simbolo}_Close"], df_out[f"{simbolo}_Open"], lag_base=0)
        df_out[f"{simbolo}_intra_weekly"] = calcular_zscore(intra_semanal)

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


def calcular_features_precios(
        df_precios:pd.DataFrame,
        periodo_largo:tuple[int, int],
        periodo_corto:tuple,
        medias:tuple,
        rango:tuple
        )->pd.DataFrame:
    
    columnas_originales = df_precios.columns.tolist()
    # Metricas sobre precios: forex

    df_precios = calcular_metricas_precios(
        df=df_precios, 
        simbolos=forex_cols,
        periodo_largo=periodo_largo[0],
        periodo_corto=periodo_corto[0],
        medias=medias[0], 
        periodo_rango=rango[0]
        )

    # Otros activos
    df_precios = calcular_metricas_precios(
        df=df_precios,
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
    df_volumen_zscore = df_precios[columnas_volumen].apply(calcular_zscore)
    df_volumen_zscore.columns = [f"{col}_zscore" for col in df_volumen_zscore.columns]

    # Metricas sobre precios extremos High - Low para pares forex
    df_precios = calcular_hl_spread(df=df_precios, simbolos=forex_cols)

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
        df_out[f"{pair}_yield_spread_zscore"] = calcular_zscore(serie_yield_spread)

    df_out.drop(columns=columnas_originales, inplace=True)
    return df_out


def calcular_features_macro(
        df_bonds_weekly:pd.DataFrame,
        df_macro:pd.DataFrame,
        periodo_variaciones:int,
        periodo_skew:int
) -> pd.DataFrame:
    # Feature engineering

    # Se calculan las asimetrias de los yield spreads
    columnas_bonos = df_bonds_weekly.columns.to_list()
    for col in columnas_bonos:
        df_bonds_weekly[f"{col}_skew"] = calcular_asimetria(df_bonds_weekly[col], periodo=periodo_skew)

    # Tasa de cambio (velocidad) del USDCAD
    columna_usdcad = 'USDCAD_yield_spread_zscore'
    df_bonds_weekly[f"{columna_usdcad}_vel"] = df_bonds_weekly[columna_usdcad].diff(periodo_variaciones) / periodo_variaciones
   
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
        # el último estado consolidado sin shift (df_weekly.iloc[-1])
        df_weekly_features.loc[lunes_actual] = df_weekly.iloc[-1]

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


# Funcion de procesamiento con los parametros de cada par

class ParametrosPar(TypedDict):
    periodo_largo: tuple[int, int]
    periodo_corto: tuple[int, int]
    medias: tuple[list[int], list[int]]
    rango: tuple[int, int]
    periodo_variaciones: int
    periodo_asimetria: int

parametros_forex: dict[str, ParametrosPar] = {
    "USDCAD": {
        # Parametros de precios (Forex, Otros activos)
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),  # [media corta, media larga]
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 50,
        "periodo_asimetria": 40,
    },
    "EURUSD": {
        # Parametros de precios 
        "periodo_largo": (50, 50),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),
        "rango": (15, 15),

        # Parametros macro
        "periodo_variaciones": 50,
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
    "GBPUSD": {
        # Parametros de precios 
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 50,
        "periodo_asimetria": 60,
        },
    "generico": {
        # Parametros de precios 
        "periodo_largo": (40, 40),
        "periodo_corto": (25, 25),
        "medias": ([15, 50], [15, 50]),
        "rango": (10, 10),

        # Parametros macro
        "periodo_variaciones": 50,
        "periodo_asimetria": 50,
        }
}

def procesar_par(
        df_precios:pd.DataFrame,
        df_bonds:pd.DataFrame,
        df_macro:pd.DataFrame,
        parametro: ParametrosPar
)->pd.DataFrame:

    precios_features = calcular_features_precios(
        df_precios=df_precios,
        periodo_largo = parametro["periodo_largo"],
        periodo_corto = parametro["periodo_corto"],
        medias = parametro["medias"],
        rango = parametro["rango"] 
        )

    macro_features = calcular_features_macro(
        df_bonds_weekly=df_bonds, 
        df_macro=df_macro,
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
    

    columnas_log = columnas_std + columnas_vol_directional + []
    df_transform = transformar_log(df, cols=columnas_log)

    # Transformaciones yeo-johnson
    cols_gap = encontrar_columnas(columnas_features, f"_open_gap")
    cols_hl_spread = encontrar_columnas(columnas_features, f"_HL_spread_")
    cols_volume = encontrar_columnas(columnas_features, f"_Volume_")
    col_vix_vel = encontrar_columnas(columnas_features, f"VIX_vel_")

    columnas_yeo = cols_gap + cols_hl_spread + cols_volume + col_vix_vel
    df_transform = transformar_yeo(df=df_transform, cols=columnas_yeo)

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
):
    # Cargar y pre-procesar fichero de precios 
    precios_raw = preprocesar_precios()
    bonds_raw, macro_raw = preprocesar_macro()

    # Ejecutar solo los pares que tienen parametros definidos en el diccionario parametros
    for par in forex_cols:
        parametros = parametros_forex.get(par)
        if parametros:
            df_par = procesar_par(
                df_precios=precios_raw,
                df_bonds=bonds_raw,
                df_macro=macro_raw,
                parametro=parametros
            )
            post_procesar_par(par, df_par)

    # Si existe al menos un par no mapeado, ejecutar la vía genérica una única vez
    hay_pares_genericos = any(par not in parametros_forex for par in forex_cols)
    if hay_pares_genericos:
        df_generico = procesar_par(
            df_precios=precios_raw,
            df_bonds=bonds_raw,
            df_macro=macro_raw,
            parametro=parametros_forex["generico"]
        )
        post_procesar_par("generic", df_generico)
                  


if __name__ == "__main__":
    main()

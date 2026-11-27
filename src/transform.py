'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
import numpy as np
from pathlib import Path
from typing import (
    List, Dict, Optional
    )

from src.config import (
    CARPETA_DATOS, FOREX_COLS, DIVISAS,
    OTROS_ACTIVOS_COLS, MAP_DIVISAS
    )

EPSILON = 1e-8

# Manejo de datos

def cargar_datos_raw(
        nombre_fichero:str
)->pd.DataFrame:
    fichero = Path(CARPETA_DATOS) / f"raw_data/{nombre_fichero}.parquet"
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


def preprocesar_macro()->pd.DataFrame:
    # Cargar datos diarios
    df_macro = cargar_datos_raw(nombre_fichero="datos_macro_daily")

    # Asegurar índice datetime ordenado y asegurar frecuencia de días hábiles
    df_macro.index = pd.to_datetime(df_macro.index)
    df_macro = df_macro.sort_index()
    df_macro = df_macro.ffill()

    return df_macro


def guardar_datos_clean(
        df:pd.DataFrame
        ):
    fichero = Path(CARPETA_DATOS) / f"clean_data/clean.parquet"

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')


# Feature Engineering

def aplicar_clipping(
          series:pd.Series,
          clip_z:float=2.5)->pd.Series:
    return series.clip(-clip_z, clip_z)


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
    periodo: int = 26,
    min_periodos: int | None = None,
    asumir_media_cero: bool = False,
    exponencial: bool = True
) -> pd.Series:
    """Normaliza la vela actual (t) utilizando parámetros históricos hasta (t-1)."""
    if min_periodos is None:
        min_periodos = periodo

    # Desplazamiento estricto para evitar lookahead bias
    serie_previa = serie.shift(1)
   
    # calculo Mean / Std
    if exponencial:
        ventana = serie_previa.ewm(
            span=periodo, min_periods=min_periodos, adjust=True
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
        span:int=26
)->pd.Series:
    '''
    Calcula la volatilidad direccional de los retornos logaritmicos'''
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

    # Ratio logaritmico: simetrico en cero, positivo significa volatilidad positiva
    log_ratio = np.log((vol_up + EPSILON) / (vol_down + EPSILON))
    return log_ratio


def calcular_asimetria(
    serie: pd.Series, 
    periodo: int = 52, 
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


def calcular_retornos(
          df:pd.DataFrame,
          simbolos:Optional[list[str]] = None
):
    if simbolos is None:
        simbolos = FOREX_COLS + OTROS_ACTIVOS_COLS
    nuevas_columnas = {}

    for simbolo in simbolos:
        nuevas_columnas[f"{simbolo}_Ret"] = np.log(df[f"{simbolo}_Close"]).diff()

    return pd.concat([df, pd.DataFrame(nuevas_columnas, index=df.index)], axis=1)


def calcular_metricas_retornos(
        df:pd.DataFrame,
        simbolos:list[str],
        periodo_largo:int=52,
        periodo_corto:int | None = None,
        min_periodos:int | None = None
)->pd.DataFrame:
    '''
    Calcula metricas sobre los retornos:
    - Desvio estandar y asimetria de los retornos
    - Volatilidad direccional
    - Ratio de volatilidad si se especifica periodo corto
    '''
    retornos_simbolos = [f"{simbolo}_Ret" for simbolo in simbolos]
    nuevas_columnas = {}

    for col in retornos_simbolos:
        # Metricas sobre los retornos para el periodo largo  
          
        # Desvio estandar: log corrige la asimetria positiva antes de estandarizar
        desvio_largo = df[col].rolling(window=periodo_largo, min_periods=min_periodos).std()
        log_desvio_largo = np.log(desvio_largo + EPSILON)
        nuevas_columnas[f"{col}_std_zscore"] = calcular_zscore(
             log_desvio_largo,
             periodo=periodo_largo,
             asumir_media_cero=False
        )

        # Volatilidad direccional
        vol_dir = calcular_volatilidad_direccional(df[col])
        nuevas_columnas[f"{col}_vol_dir_zscore"] = calcular_zscore(
             vol_dir,
             periodo=periodo_largo,
             asumir_media_cero=False
        )

        if periodo_corto:
            # Velocidad: tasa de cambio de los retornos
            vel = df[col].diff(periodo_corto) / periodo_corto

            # Aceleracion: tasa de cambio de la velocidad
            acel = vel.diff(periodo_corto) / periodo_corto
            nuevas_columnas[f"{col}_accel_zscore"] = calcular_zscore(acel, asumir_media_cero=True)

    return pd.concat([df, pd.DataFrame(nuevas_columnas, index=df.index)], axis=1)          
    

def calcular_exposiciones(
          df:pd.DataFrame,
          cols:list[str],
          lag:int=0,
          ventana:int=26,
          min_periodos:int | None=None
)->pd.DataFrame:
    '''
    Calcula la exposicion de una lista de variables por divisa,
    multiplicando las series por su correlacion con el retorno de cada moneda.
    Aplica z-score a ambas series para normalizarlas.
    '''
    
    nuevas_columnas = {}
    for ccy in DIVISAS:
        ccy_ret = calcular_zscore(df[f"{ccy}_Ret"], asumir_media_cero=True)
        for col in cols:
            # Variables ya expresadas en z-score: se evita duplicar el calculo y se remueve "_zscore" del nombre
            if col.endswith("_zscore"):
                base_col = col.removesuffix("_zscore")
                raw_series = df[col]
            else:
                base_col = col
                # Se calculan z-scores asumiendo media cero para features que sean retornos
                es_retorno = col.endswith(("_Ret", "Open_gap"))
                raw_series = calcular_zscore(df[col], asumir_media_cero=es_retorno)

            # correlacion movil
            rolling_corr = ccy_ret.rolling(window=ventana, min_periods=min_periodos).corr(raw_series).shift(lag)

            # Recortar columnas de activos para controlar valores outliers extremos
            series_clipped = aplicar_clipping(raw_series)

            # Exposicion por divisa: producto de la correlacion y la serie recortada       
            nuevas_columnas[f"{ccy}_{base_col}_zscore_exposure"] = rolling_corr * series_clipped

    return pd.concat([df,pd.DataFrame(nuevas_columnas, index=df.index)], axis=1)
    

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
        periodo_largo:int,
        periodo_corto:int
)->pd.DataFrame:
    df_features = df.copy()
    columnas_originales = df_features.columns.tolist()

    # Obtener Features

    # Metricas sobre retornos de divisas
    df_features = calcular_metricas_retornos(
        df=df_features,
        simbolos = DIVISAS, 
        periodo_largo=periodo_largo,
        periodo_corto=periodo_corto
        )

    # Metricas sobre retornos de otros activos
    df_features = calcular_metricas_retornos(
        df=df_features,
        simbolos = OTROS_ACTIVOS_COLS, 
        periodo_largo=periodo_largo
        )    

    # Eliminar columnas absolutas (originales excepto retornos)
    retornos_cols = [f"{simbolo}_Ret" for simbolo in DIVISAS+OTROS_ACTIVOS_COLS]  
    cols_absolutas = [col for col in columnas_originales if col not in retornos_cols]
    df_features = df_features.drop(columns=cols_absolutas)

    # Exposicion por divisa de las features de otros activos

    otros_activos_feat = []
    for activo in OTROS_ACTIVOS_COLS:
        cols = [c for c in encontrar_columnas(df_features, cadena=f"{activo}")]
        otros_activos_feat.extend(cols)
    df_features = calcular_exposiciones(df=df_features, cols=otros_activos_feat)  

    # Se eliminan las features de otros activos sin exposiciones
    df_features = df_features.drop(columns=otros_activos_feat)

    # Columnas restantes: relativas
    columnas_relativas = df_features.columns.tolist()

    # Se aplica lag1 sobre los predictores para evitar Lookahead Bias

    # Se excluyen retornos de divisas para usar al crear el target_rank
    ret_divisas = [f"{divisa}_Ret" for divisa in DIVISAS]
    variables_restantes = [col for col in columnas_relativas if col not in ret_divisas]

    df_features = aplicar_lag(df=df_features, columnas=variables_restantes, quitar_original=True) 
 
    '''
    # Retornos: se calcula lag para usar como feature, manteniendo los originales
    Se reemplaza por ranking lag1 al mostrar mejor R2

    df_features = aplicar_lag(df=df_features, columnas=cols_divisas, quitar_original=False)
    cols_divisas_lag = [f"{col}_lag1" for col in cols_divisas]
    df_features[cols_divisas_lag] = df_features[cols_divisas_lag].apply(lambda x: calcular_zscore(
         x,
         periodo = periodo_largo, 
         asumir_media_cero=False))  
    '''  

    return df_features


def obtener_features_macro(
        df_m:pd.DataFrame,
        periodo:int=26
        ) -> pd.DataFrame:
            df_macro = df_m.copy()
            # Se aplica forward fill para cubrir huecos en datos (necesario para datos STLFSI semanales)
            df_macro = df_macro.ffill()

            # Lags por retardo de publicación (en DÍAS HÁBILES de mercado)
            df_daily_lagged = pd.DataFrame(index=df_macro.index)

            # VIX y Real Yield: Disponibles al cierre del día / apertura t+1 -> 1 día hábil
            cols_lag_1d = ["VIX_index", "Real_yield_10y", "T10Y2Y"]
            df_daily_lagged[cols_lag_1d] = df_macro[cols_lag_1d].shift(1)

            # STLFSI: Se calcula al viernes T, pero la Fed lo publica el JUEVES T+1.
            # Entre el viernes y el jueves hábil transcurren 4 días hábiles (Lun, Mar, Mié, Jue).
            cols_lag_4d = ["St_louis_financial_stress"]
            df_daily_lagged["St_louis_financial_stress"] = df_macro["St_louis_financial_stress"].shift(4)

            # Resampleo semanal estándar sobre los datos históricos, alineando al lunes previo
            df_weekly = df_daily_lagged.resample("W-MON", label="left", closed="left").last()
            df_weekly.index.name = "Date"

            # Calculo de features: Variaciones, Velocidades y Aceleraciones

            # Vix index: variacion porcentual mediante diferencia logaritmica
            vix_log = np.log(df_weekly["VIX_index"])
            df_weekly["VIX_pct_diff"] = vix_log.diff()
            vix_vel = vix_log.diff(periodo) / periodo
            df_weekly[f"VIX_accel{periodo}w"] = vix_vel.diff(periodo) / periodo

            # Variables originalmente expresadas en tasas (variación absoluta delta)

            # Real Yield
            ry = df_weekly["Real_yield_10y"]
            df_weekly["Real_yield_delta"] = ry.diff()
            ry_vel = ry.diff(periodo) / periodo
            df_weekly[f"Real_yield_accel{periodo}w"] = ry_vel.diff(periodo) / periodo

            # T10Y2Y
            t10y2y = df_weekly["T10Y2Y"]
            df_weekly["T10Y2Y_delta"] = t10y2y.diff()
            t10y2y_vel = t10y2y.diff(periodo) / periodo
            df_weekly[f"T10Y2Y_accel{periodo}w"] = t10y2y_vel.diff(periodo) / periodo            

            # STLFSI
            stlfsi = df_weekly["St_louis_financial_stress"]
            df_weekly["STLFSI_delta"] = stlfsi.diff()
            stlfsi_vel = stlfsi.diff(periodo) / periodo
            df_weekly[f"STLFSI_accel{periodo}w"] = stlfsi_vel.diff(periodo) / periodo

            # Se eliminan las columnas macro originales
            df_weekly = df_weekly.drop(columns=cols_lag_1d+cols_lag_4d)


            df_weekly = df_weekly.copy().sort_index()

            # Asegurar que la ultima semana se encuentre en el dataframe
            tz = getattr(df_weekly.index, "tz", None)
            hoy = pd.Timestamp.now(tz=tz).normalize()
            lunes_actual = hoy - pd.Timedelta(days=hoy.weekday())

            if lunes_actual not in df_weekly.index:
                ultima_fecha = df_weekly.index[-1]
                if (lunes_actual - ultima_fecha).days > 7:
                    raise ValueError(
                        f"Gap detected: last available week is {ultima_fecha}, expected"
                        f" {lunes_actual - pd.Timedelta(days=7)}"
                )

            df_weekly.loc[lunes_actual] = np.nan

            # Aplicar lag a todas las variables
            df_weekly_features = aplicar_lag(
                df=df_weekly, columnas=df_weekly.columns.to_list(), quitar_original=True
            )             

            return df_weekly_features


# Post proceso: obtener target_rank y tratamiento de outliers

def descomponer_retornos(df:pd.DataFrame)->pd.DataFrame:
    '''
    Reemplaza los retornos de N pares XXX/USD por los N+1 retornos 
    latentes individuales (incluyendo USD) bajo la
    restricción de suma cero (canasta equiponderada).
    '''
    cols_pares = [f"{par}_Ret" for par, _ in MAP_DIVISAS.values()]

    # Validacion de columnas faltantes
    faltantes = set(cols_pares) - set(df.columns)
    if faltantes:
         raise KeyError(f"Faltan columnas requeridas en el DataFrame: {faltantes}")

    ret_divisas = {
        f"{divisa}_Ret": signo * df[f"{par}_Ret"]
        for divisa, (par, signo) in MAP_DIVISAS.items()
    }

    df_divisas = pd.DataFrame(ret_divisas, index=df.index)
    df_restante = df.drop(columns=cols_pares)

    # s_USD = -1 / (N + 1) * sum(r_{k/USD})
    n_total = df_divisas.shape[1] + 1
    usd_ret = -df_divisas.sum(axis=1, skipna=False) / n_total

    # s_k = r_{k/USD} + s_USD
    df_divisas_usd = df_divisas.add(usd_ret, axis=0)
    df_divisas_usd["USD_Ret"] = usd_ret

    return pd.concat([df_restante, df_divisas_usd], axis=1, sort=False)


def rankear_divisas(
    df_currencies: pd.DataFrame,
    normalize: bool = True,
    ) -> pd.DataFrame:
        """Calcula el ranking cross-sectional semanal a partir de los retornos

        latentes de las 7 divisas (formato ancho: index=date, cols=currencies).

        Parámetros:
        - df_currencies: DataFrame retornado por decompose_currency_returns().
        - normalize: Si True, escala los rangos a [0.0, 1.0] usando
            (rank - 1) / (N - 1), dejando la mediana en 0.5. Si False, retorna
            rangos enteros de 1 a N.
        """
        # Rankear horizontalmente (axis=1) entre las 7 divisas en cada fecha
        # method="average" maneja empates exactos asignando el rango medio
        ranks = df_currencies.rank(axis=1, method="average", ascending=True)

        if normalize:
            # Contar divisas válidas por fila (por si hubiera algún NaN puntual)
            n_valid = df_currencies.count(axis=1)
            ranks = ranks.sub(1, axis=0).div(n_valid - 1, axis=0)

        return ranks.add_suffix('_rank')


def post_procesar(
        df:pd.DataFrame,
        ):
            # Obtener exposiciones por divisa para variables macro
            columnas_vix = encontrar_columnas(df, cadena="VIX_")
            columnas_ry = encontrar_columnas(df, cadena="Real_yield_")
            columnas_stlfsi = encontrar_columnas(df, cadena="STLFSI_")
            columnas_t10y2y = encontrar_columnas(df, cadena="T10Y2Y_")

            columnas_macro_delta = columnas_vix+columnas_ry+columnas_stlfsi+columnas_t10y2y
            df = calcular_exposiciones(df, cols=columnas_macro_delta)
            df = df.drop(columns=columnas_macro_delta)

            # Construir variable objetivo
            
            # Se calcula un ranking cross sectional normalizado entre 0 y 1
            ret_divisas = [f"{divisa}_Ret" for divisa in DIVISAS]
            df_divisas = df[ret_divisas]
            df_labels = rankear_divisas(df_currencies=df_divisas, normalize=True)

            # Eliminar missings iniciales
            print("\nTop 5 Valores missings por variable:")
            print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False).head())
            df = df.dropna()

            # Separar predictores
            df_predictores = df.drop(columns=ret_divisas)

            # Se aplica clipping sobre valores extremos
            df_predictores = df_predictores.apply(lambda x : aplicar_clipping(x))

            # Se aplica lag sobre labels para usar como predictores
            df_labels = aplicar_lag(df=df_labels, columnas=df_labels.columns.to_list(), quitar_original=False)

            # Se unen los dataframes para el almacenamiento
            df_final = pd.concat([df_predictores, df_labels], axis=1, sort=False)

            # Se eliminan posibles missings luego de transformar
            df_final = df_final.dropna()

            print("\n Datos limpios finales:")
            print(df_final.info())

            # Se guardan los datos depurados
            guardar_datos_clean(df_final)    


'''
Bloque principal

Ejecucion desde el terminal:
python -m src.transform
'''
def main():    
    # Cargar ficheros datos macro y pre-procesar
    macro_raw = preprocesar_macro()

    # Obtener features macro
    macro_features = obtener_features_macro(
            df_m=macro_raw,
            periodo=26
            )
    # Cargar fichero de precios y pre-procesar
    precios_raw = preprocesar_precios()

    # Calcular retornos para todos los activos
    df_precios = calcular_retornos(precios_raw)

    # Obtener retornos por divisa, incorporando el USD implicito
    df_retornos = descomponer_retornos(df=df_precios)

    # Obtener features de precios para todos los pares y activos
    price_features = obtener_features_precios(
        df=df_retornos,
        periodo_largo= 52,
        periodo_corto = 26
    )

    # Unir y post-procesar: transformaciones, crear label y almacenamiento
    df_merged = pd.concat([price_features, macro_features], axis=1, sort=False)
    post_procesar(df=df_merged)

                  
if __name__ == "__main__":
    main()

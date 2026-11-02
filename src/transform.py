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
        simbolos:list[str],
        periodo_var:int=20,
        periodo_retorno_largo:int | None = None,
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
        columna_close = f"{simbolo}_Close"
        ret_col = f"{simbolo}_return"
        acc_col = f"{simbolo}_acceleration"
        var_col = f"{simbolo}_var_{periodo_var}"
        df_out[ret_col] = df_out[columna_close].pct_change(fill_method=None)
        df_out[acc_col] = df_out[ret_col].diff()
        df_out[var_col] = df_out[ret_col].rolling(window=periodo_var, min_periods=min_periodos).var()

        if periodo_retorno_largo:

            # Retorno: diferencia entre retorno corto y largo           
            retorno_largo = df_out[columna_close].pct_change(periodo_retorno_largo,fill_method=None)
            df_out[f"{simbolo}_return_diff"] = df_out[ret_col] - retorno_largo

            # Lo mismo con Aceleraciones            
            aceleracion_larga = retorno_largo.diff()
            df_out[f"{simbolo}_acceleration_diff"] = df_out[acc_col] - aceleracion_larga

            # Varianzas: ratio entre varianza corta y larga           
            varianza_larga = retorno_largo.rolling(window=periodo_var, min_periods=min_periodos).var()
            epsilon = 1e-8
            df_out[f"{simbolo}_var_ratio"] = df_out[var_col] / (varianza_larga+epsilon)

        # Range position
        if periodo_rango:
            # Se calculan maximos y minimos
            maximo = (df_out[columna_close]
                                .rolling(window=periodo_rango, min_periods=min_periodos)
                                .max()
                                )
            minimo = (df_out[columna_close]
                                .rolling(window=periodo_rango, min_periods=min_periodos)
                                .min()
                                )

            # Se calcula la posicion dentro del rango [-1 (Close igual al Minimo), 1 (Close igual al Maximo)]
            df_out[f"{simbolo}_range{periodo_rango}_position"] = ((df_out[columna_close] - minimo) - (maximo - df_out[columna_close])) / (maximo-minimo)
              
        # Medias moviles
        if medias:
            periodo_sma_corta = medias[0]
            periodo_sma_larga = medias[1]
            sma_corta = df_out[columna_close].rolling(window=periodo_sma_corta, min_periods=min_periodos).mean()
            sma_larga = df_out[columna_close].rolling(window=periodo_sma_larga, min_periods=min_periodos).mean()

            # Diferencia porcentual entre medias
            df_out[f"{simbolo}_sma{periodo_sma_corta}_pct_diff_sma{periodo_sma_larga}"] = calcular_pct_diff(
                sma_corta,
                sma_larga)   

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


def calcular_metricas_high_low(
        df:pd.DataFrame,
        simbolos:list[str]
)->pd.DataFrame:
    df_out = df.copy()
    for simbolo in simbolos:
        columna_close = f"{simbolo}_Close"
        columna_high = f"{simbolo}_High"
        columna_low = f"{simbolo}_Low"
        # High - Low Spread
        df_out[f"{simbolo}_High_Low_spread"] = calcular_pct_diff(df_out[columna_high], df_out[columna_low])

        # Posicion del cierre semanal respecto del rango [Low, High]
        df_out[f"{simbolo}_Close_position"] = ((df[columna_close]-df[columna_low])-(df[columna_high]-df[columna_close])) / (df[columna_high]-df[columna_low])

    return df_out


def calcular_metricas_open(
        df:pd.DataFrame,
        simbolos:list[str]
)->pd.DataFrame:
    df_out = df.copy()
    for simbolo in simbolos:
        # Diferencia entre la apertura y el cierre anterior
        df_out[f"{simbolo}_Open_pct_diff_last_Close"] = calcular_pct_diff(df_out[f"{simbolo}_Open"], df_out[f"{simbolo}_Close"], lag_base=1)

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

    # Metricas sobre precios Close: forex

    df_precios = calcular_metricas_close(
        df=df_precios, 
        simbolos=forex_cols,
        periodo_retorno_largo=12,
        medias=[15,50],
        periodo_rango=10
        )

    # Otros activos: sin calcular medias ni extremos
    df_precios = calcular_metricas_close(
        df=df_precios, 
        simbolos=otros_activos_cols
        )

    # Metricas sobre Volumen
    columnas_volumen = encontrar_columnas(
        columnas_target=columnas_originales,
        cadena="Volume"
    )
    df_precios = calcular_metricas_volumen(df=df_precios, cols=columnas_volumen)

    # Metricas sobre precios extremos High - Low para pares forex
    df_precios = calcular_metricas_high_low(df=df_precios, simbolos=forex_cols)

    # Metricas sobre precios open: todos los activos
    df_precios = calcular_metricas_open(df=df_precios, simbolos=activos_cols)   

    # Eliminar columnas originales (absolutas)
    df_precios.drop(columns=columnas_originales, inplace=True)

    # Columnas restantes: relativas
    columnas_relativas = df_precios.columns.tolist()

    # Se aplica lag1 sobre los predictores para evitar Lookahead Bias
    # Lag sobre retornos Forex: sin eliminar el original para usar como labels
    retornos_forex = []
    for major in forex_cols:
        col_retorno_forex = encontrar_columnas(columnas_relativas, f"{major}_return")[0]
        retornos_forex.append(col_retorno_forex)

    df_precios = aplicar_lag(df=df_precios, columnas=retornos_forex, quitar_original=False)

    # Variables lag: se excluyen los retornos forex que ya se calcularon, y las columnas
    # Open_pct_diff_last_Close que no llevan lag
    columnas_diff_open = encontrar_columnas(columnas_relativas, f"Open_pct_diff_last_Close")
    excluir = set(retornos_forex+columnas_diff_open)

    # Resto de las columnas: se aplica el lag y se eliminan las originales
    variables_restantes = [col for col in columnas_relativas if col not in excluir]

    df_precios = aplicar_lag(df=df_precios, columnas=variables_restantes, quitar_original=True)    


    return df_precios


def procesar_datos_macro() -> pd.DataFrame:
    # Cargar datos diarios
    df_macro = cargar_datos_raw(nombre_fichero="datos_macro_daily")

    # Asegurar índice datetime ordenado y asegurar frecuencia de días hábiles
    df_macro.index = pd.to_datetime(df_macro.index)
    df_macro = df_macro.sort_index()

    # Cálculo de variaciones (20 días hábiles de mercado)
    periodo = 20

    # Variables de nivel (variación porcentual)
    df_macro["VIX_pct_chg_20d"] = df_macro["VIX_index"].pct_change(
        periods=periodo, fill_method=None
    )
    df_macro["DXY_pct_chg_20d"] = df_macro["DXY_broad_goods"].pct_change(
        periods=periodo, fill_method=None
    )

    # Variables expresadas en tasas/spreads (variación absoluta delta)
    df_macro["Real_yield_delta_20d"] = df_macro["Real_yield_10y"].diff(
        periods=periodo
    )
    df_macro["STLFSI_delta_20d"] = df_macro["St_louis_financial_stress"].diff(
        periods=periodo
    )

    # Se conserva VIX en nivel como predictor
    df_features_daily = pd.DataFrame(index=df_macro.index)
    df_features_daily["VIX_level"] = df_macro["VIX_index"]
    df_features_daily["VIX_pct_chg_20d"] = df_macro["VIX_pct_chg_20d"]
    df_features_daily["DXY_pct_chg_20d"] = df_macro["DXY_pct_chg_20d"]
    df_features_daily["Real_yield_delta_20d"] = df_macro["Real_yield_delta_20d"]
    df_features_daily["STLFSI_delta_20d"] = df_macro["STLFSI_delta_20d"]

    # Lags por retardo de publicación (en DÍAS HÁBILES de mercado)
    # VIX y Real Yield: Disponibles al cierre del día / apertura t+1 -> 1 día hábil
    cols_lag_1d = ["VIX_level", "VIX_pct_chg_20d", "Real_yield_delta_20d"]
    df_features_daily[cols_lag_1d] = df_features_daily[cols_lag_1d].shift(1)

    # DXY Broad: 2 días hábiles de rezago
    df_features_daily["DXY_pct_chg_20d"] = df_features_daily[
        "DXY_pct_chg_20d"
    ].shift(2)

    # STLFSI4: Se calcula al viernes T, pero la Fed lo publica el JUEVES T+1.
    # Entre el viernes y el jueves hábil transcurren 4 días hábiles (Lun, Mar, Mié, Jue).
    df_features_daily["STLFSI_delta_20d"] = df_features_daily[
        "STLFSI_delta_20d"
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
    return df_weekly_features.dropna(how="all")


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
    columnas_spread = encontrar_columnas(columnas_features, "High_Low_spread")
    columnas_var = encontrar_columnas(columnas_features, f"_var_")
    

    columnas_log = columnas_spread + columnas_var
    df_transform = transformar_log(df_merged, cols=columnas_log)

    # Transformaciones yeo-johnson
    columnas_volume_pct_diff = encontrar_columnas(columnas_features, "Volume_pct_diff_sma")
    columnas_open_pct_diff = encontrar_columnas(columnas_features, f"Open_pct_diff_last_Close")

    columnas_yeo =  columnas_volume_pct_diff + columnas_open_pct_diff +[
        "USDCAD_return_lag1",
        "USDCAD_acceleration_lag1",
        "VIX_pct_chg_20d",
        "STLFSI_delta_20d"
        ]
    df_transform = transformar_yeo(df=df_transform, cols=columnas_yeo)

    # Se eliminan posibles missings luego de transformar
    df_transform.dropna(inplace=True)

    print(df_transform.info())

    # Se guardan los datos depurados
    guardar_datos_clean(df_transform)


if __name__ == "__main__":
    main()

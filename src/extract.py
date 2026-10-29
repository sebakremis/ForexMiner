'''
src/extract.py

Modulo para la fase de extraccion
'''
import yfinance as yf
from fredapi import Fred
import pandas as pd
from pathlib import Path
from typing import Optional, Dict

from src.config import (
    carpeta_datos, fecha_inicio_defecto, intervalo_defecto,
    ACTIVOS, MACRO_SERIES
)
from src.data_sources import fred_api_key

def extraer_yfinance(
        activos:Optional[Dict[str, str]] = None,
        intervalo:str=intervalo_defecto,
        fecha_inicio: str = fecha_inicio_defecto,
        fecha_fin: Optional[str] = None,
)->pd.DataFrame:
    if activos is None:
        activos = ACTIVOS
    simbolos = list(activos.keys())

    # Asegurar que no existan duplicados
    tickers_unicos = list(set([str(t) for t in simbolos]))

    df_batch = yf.download(
        tickers=tickers_unicos,
        start=fecha_inicio,
        end=fecha_fin,
        interval= intervalo,
        auto_adjust= True, # precios ajustados
        ignore_tz=True, # Evita discrepancias horarias entre distintos mercados       
    )
    if df_batch.empty:
        print("No se pudo extraer ningún dato.")
        return pd.DataFrame()

    # Restructurar datos
    # Transformar de formato "ancho" a "largo"
    # yfinance devuelve un MultiIndex en columnas cuando son varios tickers
    if isinstance(df_batch.columns, pd.MultiIndex):
        # Nombramos los niveles para no perdernos al apilar
        df_batch.columns.names = ['Atributos', 'Ticker']
        
        # .stack() pasa los tickers de las columnas a las filas
        # future_stack=True evita advertencias en versiones nuevas de pandas
        df_prices = df_batch.stack(level='Ticker', future_stack=True).reset_index()
    else:
        # En caso de que la lista tenga solo 1 ticker
        df_prices = df_batch.reset_index()
        df_prices['Ticker'] = tickers_unicos[0]

    # Homogeneizar el nombre de la columna de fecha
    col_fecha = 'Date' if 'Date' in df_prices.columns else 'Datetime'
    if col_fecha in df_prices.columns:
        df_prices.rename(columns={col_fecha: 'Date'}, inplace=True)
        df_prices['Date'] = pd.to_datetime(df_prices['Date']).dt.normalize()
    

    columnas_innecesarias = ['Dividends', 'Stock Splits', 'Capital Gains']
    drop_cols = [col for col in columnas_innecesarias if col in df_prices.columns]
    df_prices.drop(drop_cols, axis=1, inplace=True)

    # Eliminar filas donde no hubo cotización real para el ticker
    columnas_precio = [c for c in ['Close', 'Open', 'High', 'Low'] if c in df_prices.columns]
    if columnas_precio:
        df_prices.dropna(subset=columnas_precio, how='all', inplace=True)

    # Si aún quedaran duplicados por solapamiento de fechas, conservar el registro con datos válidos
    df_prices.drop_duplicates(subset=['Date', 'Ticker'], keep='first', inplace=True)

    df_prices = df_prices.sort_values(by=['Date', 'Ticker']).reset_index(drop=True)
    df_prices.columns.name = None  # <--- Quita la etiqueta 'Atributos'  
    return df_prices  


def get_fred_client(api_key: Optional[str] = None) -> Fred:
    """Valida y crea una instancia de conexion con el cliente de FRED."""
    key = api_key or fred_api_key
    if not key:
        raise ValueError(
            "FRED_API_KEY no encontrada. Revisa tu archivo .env o la variable de entorno."
        )
    return Fred(api_key=key)


def extraer_fred(
    series_map: Optional[Dict[str, str]] = None,
    start_date: str = fecha_inicio_defecto,
    end_date: Optional[str] = None,
    client: Optional[Fred] = None,
) -> pd.DataFrame:
    """
    Descarga un conjunto de series temporales de FRED y las consolida en un DataFrame.

    Parameters
    ----------
    series_map : Dict[str, str], optional
        Diccionario {series_id_fred: nombre_columna}. Si es None usa DEFAULT_GOLD_MACRO_SERIES.
    start_date : str
        Fecha inicial de consulta (formato YYYY-MM-DD).
    end_date : str, optional
        Fecha final de consulta. Si es None, trae hasta el dato mas reciente disponible.
    client : Fred, optional
        Instancia de Fred ya configurada.

    Returns
    -------
    pd.DataFrame
        DataFrame con DatetimeIndex y cada serie en una columna, ordenado cronologicamente.
    """
    if series_map is None:
        series_map = MACRO_SERIES

    fred = client or get_fred_client()
    data_frames = []

    print("Extrayendo datos macro de la api Fred...")
    for series_id, col_name in series_map.items():
        try:
            # fred.get_series devuelve una pd.Series indexada por fecha
            s = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            s.name = col_name
            # Forzar tipo numerico (coercing nulos de dias festivos/sin cotizacion)
            s = pd.to_numeric(s, errors="coerce")
            data_frames.append(s)
        except Exception as e:
            print(f"[WARN] Error al descargar la serie {series_id} ({col_name}): {e}")

    if not data_frames:
        raise RuntimeError("No se pudo obtener ninguna serie desde FRED.")

    # Outer join sobre las fechas para alinear los calendarios de negociacion
    df = pd.concat(data_frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.sort_index(inplace=True)

    return df


def guardar_datos_raw(
        df:pd.DataFrame,
        nombre_fichero:str
        ):
    fichero = Path(carpeta_datos) / f'raw_data/{nombre_fichero}.parquet'

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.extract
'''
def main(
):
    # Extraer y guardar en fichero precios de activos

    df_activos = extraer_yfinance()
    # Se reemplazan los simbolos de Tickers segun el diccionario
    df_activos['Ticker'] = df_activos['Ticker'].map(ACTIVOS)   
    guardar_datos_raw(df=df_activos, nombre_fichero="precios")
    print(df_activos.info())

    # Extraer datos macro
    df_macro = extraer_fred()
    guardar_datos_raw(df=df_macro, nombre_fichero="datos_macro")
    print(df_macro.info())


if __name__ == "__main__":
    main()
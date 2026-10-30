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
    ACTIVOS, DAILY_MACRO_SERIES, forex_cols
)
from src.data_sources import fred_api_key

def extraer_yfinance(
    activos: Optional[Dict[str, str]] = None,
    intervalo: str = intervalo_defecto,
    fecha_inicio: str = fecha_inicio_defecto,
    fecha_fin: Optional[str] = None,
) -> pd.DataFrame:
    if activos is None:
        activos = ACTIVOS

    # Conservar el orden original y eliminar duplicados manteniendo el mapeo
    tickers_unicos = list(dict.fromkeys(activos.keys()))

    df_batch = yf.download(
        tickers=tickers_unicos,
        start=fecha_inicio,
        end=fecha_fin,
        interval=intervalo,
        auto_adjust=True,  # precios ajustados
        ignore_tz=True,  # Evita discrepancias horarias entre mercados
    )

    if df_batch.empty:
        print("No se pudo extraer ningún dato.")
        return pd.DataFrame()

    # Normalizar el índice temporal (Date / Datetime)
    df_batch.index = pd.to_datetime(df_batch.index).normalize()
    df_batch.index.name = "Date"

    # yfinance retorna un MultiIndex en columnas cuando son múltiples tickers.
    # Si se solicitó un único ticker, creamos el MultiIndex para estandarizar el procesamiento.
    if not isinstance(df_batch.columns, pd.MultiIndex):
        ticker_unico = tickers_unicos[0]
        df_batch.columns = pd.MultiIndex.from_tuples(
            [(col, ticker_unico) for col in df_batch.columns],
            names=["Atributo", "Ticker"],
        )

    # Filtrar columnas no deseadas a nivel de atributo
    columnas_innecesarias = {"Dividends", "Stock Splits", "Capital Gains", "Open"}
    atributos_validos = [
        col
        for col in df_batch.columns.levels[0]
        if col not in columnas_innecesarias
    ]
    df_batch = df_batch.loc[:, df_batch.columns.get_level_values(0).isin(atributos_validos)]

    # Aplanar columnas al formato: {NombreNuevo}_{Atributo} (ej: EURUSD_Close)
    nuevas_columnas = []
    for atributo, ticker in df_batch.columns:
        nombre_amigable = activos.get(ticker, ticker)
        nuevas_columnas.append(f"{nombre_amigable}_{atributo}")

    df_batch.columns = nuevas_columnas
    df_batch.columns.name = None

    # Eliminar duplicados en el índice de fechas y ordenar cronológicamente
    df_batch = df_batch[~df_batch.index.duplicated(keep="first")]
    df_batch.sort_index(inplace=True)

    return df_batch  


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
        series_map = DAILY_MACRO_SERIES

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
    # Extraer precios de activos
    df_activos = extraer_yfinance()

    # Se eliminan las columnas de volumen forex
    cols_a_eliminar = [
        col
        for col in df_activos.columns
        if any(col.startswith(f"{fx}_") for fx in forex_cols)
        and col.endswith("_Volume")
    ]

    df_activos = df_activos.drop(columns=cols_a_eliminar)

    guardar_datos_raw(df=df_activos, nombre_fichero="precios")
    print(df_activos.info())

    # Extraer datos macro
    print("Datos macro diarios")
    df_macro_daily = extraer_fred()
    guardar_datos_raw(df=df_macro_daily, nombre_fichero="datos_macro_daily")
    print(df_macro_daily.info())
  


if __name__ == "__main__":
    main()
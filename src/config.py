'''
src/config.py

Configuracion global del proyecto
'''
from typing import Dict

# Ruta al fichero de datos
CARPETA_DATOS = "data"

# Fecha inicial para descargar datos
FECHA_INICIO_DEFECTO = "2006-01-03"

# Intervalo por defecto
INTERVALO_DEFECTO = "1wk"

# Mapeo de series de precios
FOREX_MAJORS: Dict[str, str] = {
    "EURUSD=X": "EURUSD",
    "GBPUSD=X": "GBPUSD",
    "USDJPY=X": "USDJPY",
    "USDCAD=X": "USDCAD",
    "USDCHF=X": "USDCHF",
    "AUDUSD=X": "AUDUSD"
}
OTROS_ACTIVOS: Dict[str, str] = {
    "GC=F": "Gold",
    "CL=F": "Oil",
    "SPY": "SPY"
}

# Diccionario de precios consolidado
YFINANCE_SERIES: Dict[str, str] = {**FOREX_MAJORS, **OTROS_ACTIVOS}

# Listas automaticas:
FOREX_COLS = list(FOREX_MAJORS.values())
OTROS_ACTIVOS_COLS = list(OTROS_ACTIVOS.values())
YFINANCE_COLS = list(YFINANCE_SERIES.values())

# Mapeo de series macro: Series ID de FRED -> Nombre descriptivo de columna
# Factores financieros / mercado (diarios o semanales en origen)
MACRO_SERIES: Dict[str, str] = {
    "DFII10": "Real_yield_10y",                 # 10-Year Real Yield (Diaria)                 
    "DTWEXBGS": "DXY_broad_goods",              # Trade Weighted USD (Diaria)
    "VIXCLS": "VIX_index",                      # CBOE Volatility Index (Diaria)
    "STLFSI4": "St_louis_financial_stress"      # Financial Stress Index (Semanal)
}
BOND_YIELDS: Dict[str, str] = {
    "IRLTLT01USM156N": "USD_yield_avg",
    "IRLTLT01JPM156N": "JPY_yield_avg",
    "IRLTLT01DEM156N": "EUR_yield_avg",
    "IRLTLT01GBM156N": "GBP_yield_avg",
    "IRLTLT01CHM156N": "CHF_yield_avg",
    "IRLTLT01CAM156N": "CAD_yield_avg",
    "IRLTLT01AUM156N": "AUD_yield_avg"
}

# Diccionario de precios consolidado
FRED_SERIES: Dict[str, str] = {**MACRO_SERIES, **BOND_YIELDS}

# Listas automaticas:
MACRO_COLS = list(MACRO_SERIES.values())
BOND_COLS = list(BOND_YIELDS.values())
FRED_COLS = list(FRED_SERIES.values())

# Mapeo de cada divisa extranjera a su par contra USD y su signo direccional
# (+1 si el par es CCY/USD, -1 si el par es USD/CCY para que positivo = mayor tasa en CCY)
MAP_DIVISAS = {
    "EUR": ("EURUSD", 1.0),
    "GBP": ("GBPUSD", 1.0),
    "AUD": ("AUDUSD", 1.0),
    "JPY": ("USDJPY", -1.0),
    "CAD": ("USDCAD", -1.0),
    "CHF": ("USDCHF", -1.0),
}
DIVISAS = ["EUR", "GBP", "AUD", "JPY", "CAD", "CHF", "USD"]

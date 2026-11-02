'''
src/config.py

Configuracion global del proyecto
'''
from typing import Dict

# Ruta al fichero de datos
carpeta_datos = "data"

# Fecha inicial para descargar datos
fecha_inicio_defecto = "2006-01-03"

# Intervalo por defecto
intervalo_defecto = "1wk"

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
# Diccionario consolidado
ACTIVOS: Dict[str, str] = {**FOREX_MAJORS, **OTROS_ACTIVOS}

# Mapeo de series macro: Series ID de FRED -> Nombre descriptivo de columna
# Factores financieros / mercado (diarios o semanales en origen)
DAILY_MACRO_SERIES: Dict[str, str] = {
    "DFII10": "Real_yield_10y",                 # 10-Year Real Yield (Diaria)
    "DTWEXBGS": "DXY_broad_goods",              # Trade Weighted USD (Diaria)
    "VIXCLS": "VIX_index",                      # CBOE Volatility Index (Diaria)
    "STLFSI4": "St_louis_financial_stress",     # Financial Stress Index (Semanal -> cierre de mes)
}

# Listas automaticas:
forex_cols = list(FOREX_MAJORS.values())
otros_activos_cols = list(OTROS_ACTIVOS.values())
activos_cols = list(ACTIVOS.values())
daily_macro_cols = list(DAILY_MACRO_SERIES.values())

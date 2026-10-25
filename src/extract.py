'''
src/extract.py

Modulo para la fase de extraccion
'''
import yfinance as yf
import pandas as pd
from src.config import carpeta_datos, fecha_inicio_defecto
from pathlib import Path
from typing import Optional

def extraer_precios(
        simbolo:str='SPY',
        intervalo:str='1d',
        fecha_inicio: str = fecha_inicio_defecto,
        fecha_fin: Optional[str] = None,
)->pd.DataFrame:
    ticker = yf.Ticker(simbolo)
    df = ticker.history(
        start=fecha_inicio,
        end=fecha_fin,
        interval= intervalo,
        auto_adjust= True
    )
    columnas_innecesarias = ['Dividends', 'Stock Splits', 'Capital Gains']
    df.drop(columnas_innecesarias, axis=1, inplace=True)

    return df


def guardar_precios(
        df:pd.DataFrame,
        simbolo:str,
        intervalo:str
        ):
    fichero = Path(carpeta_datos) / f'{simbolo}/precios_{simbolo}_{intervalo}.parquet'

    # Asegurar que el directorio exista
    fichero.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(fichero)
    print(f'Fichero guardado correctamente en {Path(fichero)}')


'''
Bloque principal

Ejecucion desde el terminal:
python -m src.extract
'''
def main():
    simbolo = 'SPY'
    intervalo = '1d'
    df = extraer_precios(simbolo=simbolo, intervalo=intervalo)
    df.info()

    guardar_precios(df, simbolo=simbolo, intervalo=intervalo)

if __name__ == '__main__':
    main()
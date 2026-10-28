'''
src/extract.py

Modulo para la fase de extraccion
'''
import yfinance as yf
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence

from src.config import carpeta_datos, fecha_inicio_defecto, intervalo_defecto
from src.cli import get_base_parser

def extraer_precios(
        simbolo:str='SPY',
        intervalo:str=intervalo_defecto,
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
    drop_cols = [col for col in columnas_innecesarias if col in df.columns]
    df.drop(drop_cols, axis=1, inplace=True)

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

Para extraer un ticker diferente a SPY (Ej: NVDA):
python -m src.extract --simbolo NVDA

Para intervalo distinto a '1d' (Ej: 1wk):
python -m src.extract --intervalo 1wk
'''
def main(
        simbolo:str,
        intervalo:str
):
    df = extraer_precios(simbolo=simbolo, intervalo=intervalo)
    df.info()

    guardar_precios(df, simbolo=simbolo, intervalo=intervalo)


def entrypoint_extract(argv: Sequence[str] | None = None):
    """Punto de entrada de línea de comandos."""
    parser = get_base_parser()

    # parse_known_args retorna (args_conocidos, lista_de_no_reconocidos)
    args, _ = parser.parse_known_args(argv)
    
    # Ejecuta la lógica central con los argumentos parseados
    return main(simbolo=args.simbolo, intervalo=args.intervalo)

if __name__ == "__main__":
    entrypoint_extract()
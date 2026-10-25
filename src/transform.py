'''
src/transform.py

Modulo de la fase de transformacion

'''
import pandas as pd
from src.config import carpeta_datos
from pathlib import Path
from typing import List

def cargar_precios(
        simbolo:str,
        intervalo:str='1d'
)->pd.DataFrame:
    fichero_precios = Path(carpeta_datos) / f'{simbolo}/precios_{simbolo}_{intervalo}.parquet'
    try:
        df = pd.read_parquet(fichero_precios)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{fichero_precios}' no existe en la ruta especificada.")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame()


def calcular_variaciones_pct(
    df: pd.DataFrame,
    col_base: str,
    cols_target: List[str],
    lag_base: int = 0,
) -> pd.DataFrame:
    '''
    Calcula la diferencia porcentual de múltiples columnas target
    respecto a una única columna base mediante broadcasting vectorial.
    '''
    # Desplazar la serie base
    base = df[col_base].shift(lag_base)

    # Operación matricial: targets - base / base
    sub_df = df[cols_target]

    try:
        pct_diffs = sub_df.sub(base, axis=0).div(base, axis=0)

    except ZeroDivisionError:
        print(f"Ocurrió un error: división por zero.")
        return pd.DataFrame()

    # Renombrar columnas
    pct_diffs.columns = [f"{col}_pct_diff_{col_base}" for col in cols_target]

    # Concatenar nuevas columnas
    return pd.concat([df, pct_diffs], axis=1)

    
'''
Bloque principal

Ejecucion desde el terminal:
python -m src.transform
'''
def main():
    # Cargar fichero de precios
    simbolo = 'SPY'
    intervalo = '1d'
    df = cargar_precios(simbolo=simbolo, intervalo=intervalo)

    # Transformar columnas a variaciones porcentuales con respecto a Open del mismo dia
    targets = ['High', 'Low', 'Close']
    df = df.pipe(
        lambda d: calcular_variaciones_pct(
            d,
            col_base='Open',
            cols_target=targets,
            lag_base=0,
        )
    )

    # Transformar a variacion porcentual respecto de Close del dia anterior
    df = df.pipe(
            lambda d: calcular_variaciones_pct(
                d,
                col_base='Close',
                cols_target=['Open'],
                lag_base=1,
            )
        )

    df.info()



if __name__ == '__main__':
    main()

'''
src.filtering.py

Modulo opcional para filtar variables debajo del umbral de ruido.
'''
import pandas as pd
import json
from src.modeling import (
    cargar_datos_clean, wide_to_long_panel
    )
from src.eda_utils import rankear_variables
from src.config import CARPETA_DATOS
from pathlib import Path



def filtrar_cols(
        col_objetivo:str,
)->list:
    '''
    Filtra las columnas que se encuentran por debajo del umbral de ruido
    de acuerdo a la tabla de Cramer. 
    Devuelve una lista con los nombres de las columnas a filtrar.
    '''
    df_clean = cargar_datos_clean()
    # Convertir a formato largo
    df_long = wide_to_long_panel(df_clean)

    df_panel = df_long.set_index(["date", "Currency"])
    
    y = df_panel[col_objetivo].astype('category')
    X = df_panel.drop(columns=col_objetivo)

    # Crear tabla de Cramer
    tabla_cramer = rankear_variables(X=X, y=y, graficar=False)
    cols_filtrar = tabla_cramer.index[
        tabla_cramer["Status"] == "Below Noise"
    ].tolist() 
    return cols_filtrar

def main():
    '''
    Genera un fichero JSON con las columnas a filtrar para cada par de divisas.
    '''
    fichero = Path(CARPETA_DATOS) / "cols_filtrar.json"
    col_objetivo = "Target_rank"
    cols_filtrar = filtrar_cols(col_objetivo)

    with fichero.open("w", encoding="utf-8") as f:
        json.dump({col_objetivo:cols_filtrar}, f, indent=4, ensure_ascii=False)

    print(f"Datos guardados en fichero {fichero}.")


if __name__ == '__main__':
    main()



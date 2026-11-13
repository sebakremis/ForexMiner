'''
src.filtering.py

Modulo opcional para filtar variables debajo del umbral de ruido.
'''
import pandas as pd
import json
from src.modeling import (
    cargar_datos_clean, filtrar_pares_restantes, separar_train_test, 
    categorizar_en_cuantiles
    )
from src.eda_utils import rankear_variables
from src.config import carpeta_datos, forex_cols
from pathlib import Path



def filtrar_cols(
        par:str
)->list:
    '''
    Filtra las columnas de un par de divisas que se encuentran por debajo del umbral de ruido.
    de acuerdo a la tabla de Cramer. Devuelve una lista con los nombres de las columnas a filtrar.
    '''
    # Carga y preparacion de datos
    df_clean = cargar_datos_clean(simbolo=par)   
    df_filtrado = filtrar_pares_restantes(df_clean, par)
    ultima_fecha = df_filtrado.index.max()
    df_train, _ = separar_train_test(df=df_filtrado, fecha=ultima_fecha)

    # Crear variable objetivo categorica
    objetivo_cont = f"{par}_intra_weekly"
    objetivo_cat = f"{objetivo_cont}_cat"
    df_train, _ = categorizar_en_cuantiles(
    df_train, 
    columna=objetivo_cont, 
    num_cuantiles=3, 
    quitar_original=True
    )

    # Separar predictores y var objetivo, que se convierte a tipo 'category'
    X = df_train.drop(objetivo_cat, axis=1)
    y = df_train[objetivo_cat].astype('category')

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
    fichero = Path(carpeta_datos) / "cols_filtrar.json"
    datos = {}
    for pair in forex_cols:
        datos[pair] = filtrar_cols(par=pair)

    with fichero.open("w", encoding="utf-8") as f:
        json.dump(datos, f, indent=4, ensure_ascii=False)

    print(f"Datos guardados en fichero {fichero}.")


if __name__ == '__main__':
    main()



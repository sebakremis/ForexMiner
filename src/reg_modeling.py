'''
src/modeling.py

Modulo para la fase de modelado
'''
import pandas as pd
from src.config import carpeta_datos, intervalo_defecto
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from src.cat_modeling import cargar_datos, split_ultimo
import argparse

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.reg_modeling
'''
def main(
        simbolo:str="SPY",
        intervalo:str=intervalo_defecto        
):
    # Cargar fichero de precios
    df, cols_absolutas = cargar_datos(simbolo=simbolo, intervalo=intervalo, leer_cols_exclude=True)
    df_clean = df.drop(columns=cols_absolutas)
    posibles_labels = ['High_pct_diff_Open', 'Low_pct_diff_Open', 'Close_pct_diff_Open']

    # Modelo
    nombre_modelo = 'High'
    objetivo = f"{nombre_modelo}_pct_diff_Open"
    X_train, X_live, y_train, _ = split_ultimo(df=df_clean, label=objetivo, cols_excluded=posibles_labels)

    # Inicializar y entrenar
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_live)
    ultima_fecha = X_live.index.max()
    ultimo_open = df.loc[ultima_fecha, 'Open']
    precio_predicho = ultimo_open * (1+ y_pred[0])

    print(f"Entrenamiento modelo {nombre_modelo} finalizado.")
    print(f"Valor predicho para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_pred[0]}")
    print(f"Ultimo Open: {ultimo_open}")
    print(f"Precio predicho: {precio_predicho}")

    # Validacion cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        estimator=model, X=X_train, y=y_train, cv=tscv, scoring="r2"
    )

    # Mostrar resultados por cada pliegue cronológico
    print("Resultados de la Validación Cruzada:")
    print(f"R² promedio: {cv_scores.mean():.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simbolo",
        type=str,
        default="SPY",
        help="Símbolo a cargar (por defecto: SPY)",
    )

    args = parser.parse_args()
    main(simbolo=args.simbolo)
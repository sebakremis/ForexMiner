'''
src/cat_modeling.py

Entrenamiento de modelo de Clasificacion
'''
import pandas as pd
from src.config import carpeta_datos
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from src.reg_modeling import cargar_datos, split_ultimo
from src.transform import categorizar_en_cuantiles

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.cat_modeling
'''
def main():
    # Cargar fichero de precios
    simbolo = 'SPY'
    intervalo = '1d'
    df, cols_absolutas = cargar_datos(simbolo=simbolo, intervalo=intervalo, leer_cols_exclude=True)
    df_clean = df.drop(columns=cols_absolutas)
    posibles_labels = ['High_pct_diff_Open', 'Low_pct_diff_Open', 'Close_pct_diff_Open']

    # Modelo
    nombre_modelo = 'High'
    objetivo = f"{nombre_modelo}_pct_diff_Open"
    objetivo_cat = f"{objetivo}_cat"

    # Crear variable categorica
    df_clean, tabla_bins = categorizar_en_cuantiles(
        df_clean, 
        columna=objetivo, 
        num_cuantiles=3, 
        quitar_original=False
        )

    print("\n--- Rangos por Categoría ---")
    print(tabla_bins)

    
    X_train, X_live, y_train, _ = split_ultimo(df=df_clean, label=objetivo_cat, cols_excluded=posibles_labels+[objetivo_cat])

    # Inicializar y entrenar
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=30,
        min_samples_split= 50,
        max_samples= 0.3,
        max_features=0.25,
        class_weight='balanced'        
    )
    model.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = model.predict(X_live)
    ultima_fecha = X_live.index.max()
    ultimo_open = df.loc[ultima_fecha, 'Open']

    print(f"Entrenamiento modelo {nombre_modelo} finalizado.")
    print(f"Valor predicho para el {ultima_fecha.strftime("%Y-%m-%d")}: {y_pred[0]}")
    print(f"Ultimo Open: {ultimo_open}")

    # Validacion cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        estimator=model, X=X_train, y=y_train, cv=tscv, scoring="accuracy"
    )

    # Mostrar resultados
    print(f"Accuracy promedio Time Series CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")



if __name__ == '__main__':
    main()

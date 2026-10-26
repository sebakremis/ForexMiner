'''
src/modeling.py

Modulo para la fase de modelado
'''
import pandas as pd
from src.config import carpeta_datos
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def cargar_datos(
        simbolo:str,
        intervalo:str='1d'
)->pd.DataFrame:
    fichero_precios = Path(carpeta_datos) / f'{simbolo}/datos_depurados_{simbolo}_{intervalo}.parquet'
    try:
        df = pd.read_parquet(fichero_precios)
        return df
    except FileNotFoundError:
        print(f"Error: El archivo '{fichero_precios}' no existe en la ruta especificada.")
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Parquet: {e}")
        return pd.DataFrame()


def split_ultimo(
    df: pd.DataFrame, 
    label: str, 
    cols_excluded: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Divide el DataFrame en conjuntos de entrenamiento y prueba basándose en la fecha más reciente.
    '''
    
    # Identificamos cuál es la fecha máxima en todo el dataset
    ultima_fecha = df.index.max()
    
    # Separamos mediante máscaras booleanas
    df_train = df[df.index < ultima_fecha]
    df_test = df[df.index == ultima_fecha]

    # Se descartan las columnas excluidas y la variable objetivo
    columnas_a_descartar = set(cols_excluded) | {label}
    cols_drop = [c for c in columnas_a_descartar if c in df.columns]

    X_train = df_train.drop(columns=cols_drop)
    y_train = df_train[label]
    
    X_test = df_test.drop(columns=cols_drop)
    y_test = df_test[label]
    
    return X_train, X_test, y_train, y_test

'''
Bloque principal

Ejecucion desde el terminal:
python -m src.modeling
'''
def main():
    # Cargar fichero de precios
    simbolo = 'SPY'
    intervalo = '1d'
    df = cargar_datos(simbolo=simbolo, intervalo=intervalo)

    # Modelo High
    objetivo = 'High_pct_diff_Open'
    columnas_a_excluir = ['Close_pct_diff_Open', 'Low_pct_diff_Open']
    X_train_high, X_live_high, y_train_high, _ = split_ultimo(df=df, label=objetivo, cols_excluded=columnas_a_excluir)

    # Inicializar y entrenar
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_high, y_train_high)

    # Realizar predicciones
    y_pred = model.predict(X_live_high)

    print(f"Entrenamiento finalizado.\nValor predicho para el {X_live_high.index.max().strftime("%Y-%m-%d")}:",y_pred)

    # Validacion cruzada
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(
        estimator=model, X=X_train_high, y=y_train_high, cv=tscv, scoring="r2"
    )

    # Mostrar resultados por cada pliegue cronológico
    print("Resultados de la Validación Cruzada:")
    print(f"R² promedio: {cv_scores.mean():.4f}")



if __name__ == '__main__':
    main()
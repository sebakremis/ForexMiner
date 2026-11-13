'''
src/eda_utils.py

Funciones auxiliares para el notebook de Analisis Exploratorio de Datos
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Optional
import scipy.stats as stats

def mostrar_asimetrias(df:pd.DataFrame):
    """
    Muestra la asimetría de las columnas numéricas del DataFrame.
    """
    print(df.select_dtypes(include="number").skew().sort_values(ascending=False).to_string())


def mostrar_missings(df:pd.DataFrame)->pd.Series:
    """
    Calcula la proporción de valores nulos por columna usando operaciones vectorizadas.
    """
    # Se calcula la proporción de nulos para todas las columnas
    missings = df.isna().mean()
    
    # Se filtran donde el valor es mayor a 0, ordenando luego de mayor a menor
    return missings[missings > 0].sort_values(ascending=False)


def histograma_boxplot(data, xlabel = None, title = None, font_scale=2, figsize=(9,8), bins = None):
    """ Boxplot and histogram combined
    data: 1-d data array
    xlabel: xlabel 
    title: title
    font_scale: the scale of the font (default 2)
    figsize: size of fig (default (9,8))
    bins: number of bins (default None / auto)

    example use: histogram_boxplot(np.random.rand(100), bins = 20, title="Fancy plot")
    """
    # Definir tamaño letra
    sns.set(font_scale=font_scale)
    # Crear ventana para los subgráficos
    f2, (ax_box2, ax_hist2) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=figsize)
    # Crear boxplot
    sns.boxplot(x=data, ax=ax_box2)
    # Crear histograma
    sns.histplot(x=data, ax=ax_hist2, bins=bins) if bins else sns.histplot(x=data, ax=ax_hist2)
    # Pintar una línea con la media
    ax_hist2.axvline(np.mean(data),color='g',linestyle='-')
    # Pintar una línea con la mediana
    ax_hist2.axvline(np.median(data),color='y',linestyle='--')
    # Asignar título y nombre de eje si tal
    if xlabel: ax_hist2.set(xlabel=xlabel)
    if title: ax_box2.set(title=title, xlabel="")
    # Mostrar gráfico
    plt.show()


def cat_chart(col):
     """
     Gráfico de barras para variables categóricas.
     """
     if col.dtypes == 'category':
        fig = px.bar(col.value_counts())
        #fig = sns.countplot(x=col)
        return(fig)


def pintar_columna(col):
     """
     Función general para aplicar al archivo por columnas, detectando el tipo de variable y aplicando el gráfico adecuado.
     """
     if col.dtypes != 'category':
        print('Cont')
        histograma_boxplot(col, xlabel = col.name, title = 'Distibución continua')
     else:
        print('Cat')
        cat_chart(col).show()


def top_correlaciones(df:pd.DataFrame, top_n:int=20):
    corr = df.corr().abs()
    # Eliminar duplicados y diagonal
    upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper_tri.unstack()
        .dropna()
        .reset_index(name="correlation")
        .rename(columns={"level_0": "Variable A", "level_1": "Variable B"})
    )
    return pairs.sort_values(by="correlation", ascending=False).head(top_n)


def top_correlaciones_objetivo(
    df: pd.DataFrame,
    target_col: str,
    top_n: int = 20,
    usar_absoluto: bool = True,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calcula las correlaciones de todas las columnas de un DataFrame respecto a una columna objetivo.

    Parámetros:
    - df: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo.
    - top_n: Cantidad de variables a retornar en el ranking.
    - usar_absoluto: Si es True, ordena por la fuerza de la relación sin importar el signo.
                     Si es False, conserva y ordena por valor real (+1 a -1).
    - method: 'pearson', 'kendall' o 'spearman'.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"La columna '{target_col}' no se encuentra en el DataFrame."
        )

    # 1. Separar la serie objetivo y las variables explicativas numéricas
    target_series = df[target_col]
    features_df = df.drop(columns=[target_col]).select_dtypes(
        include="number"
    )  # Asegura evaluar solo columnas numéricas

    # 2. Calcular correlación directa con la columna objetivo
    correlations = features_df.corrwith(target_series, method=method)

    # 3. Formatear como DataFrame
    res = correlations.reset_index()
    res.columns = ["Variable", "Correlación"]

    # 4. Ordenar según el criterio deseado
    if usar_absoluto:
        # Se ordena por magnitud conservando el signo original en la tabla
        res = (
            res.reindex(res["Correlación"].abs().sort_values(ascending=False).index)
            .dropna()
            .head(top_n)
        )
    else:
        res = res.sort_values(
            by="Correlación", ascending=False
        ).dropna().head(top_n)

    return res.reset_index(drop=True)


# Función para calcular VCramer
def cramers_v(var1: pd.Series, varObj: pd.Series) -> float:
    # Align indexes and drop missing pairs so pd.crosstab never produces an empty table
    df = pd.DataFrame({'v1': var1.to_numpy(), 'v2': varObj.to_numpy()}).dropna()
    if df.empty:
        return 0.0

    v1, v2 = df['v1'], df['v2']

    if pd.api.types.is_numeric_dtype(v1):
        v1 = pd.qcut(v1, q=5, duplicates='drop')
    if pd.api.types.is_numeric_dtype(v2):
        v2 = pd.qcut(v2, q=5, duplicates='drop')

    data = pd.crosstab(v1, v2).values

    # Association requires at least a 2x2 contingency table
    if data.size == 0 or data.shape[0] < 2 or data.shape[1] < 2:
        return 0.0

    return float(stats.contingency.association(data, method='cramer'))

def rankear_variables(
        X:pd.DataFrame,
        y:pd.Series,
        n_simulations:int=100,
        percentil:int=95,
        graficar:bool=True
):
    # Se crean n variables de ruido aleatorias y se computan su V de Cramer
    rng = np.random.default_rng(seed=42)
    random_scores = [
        cramers_v(pd.Series(rng.uniform(0, 1, size=len(y)), index=y.index), y)
        for _ in range(n_simulations)
    ]

    # Se toma el percentil como umbral de ruido estable
    threshold = np.percentile(random_scores, percentil)

    # Tabla de Cramer
    tablaCramer = pd.DataFrame(X.apply(lambda x: cramers_v(x,y)),columns=['VCramer'])

    # Se categorizan las variables basandose en el umbral
    tablaCramer = tablaCramer.sort_values(by='VCramer', ascending=True)
    tablaCramer['Status'] = np.where(
        tablaCramer['VCramer'] > threshold, 'Above Noise', 'Below Noise'
    )
    if graficar:
        fig = px.bar(
            tablaCramer,
            x='VCramer',
            y=tablaCramer.index,
            color='Status',
            color_discrete_map={
                'Above Noise': '#2E86C1',
                'Below Noise': '#AAB7B8'
            },
            title=f"V de Cramer vs. Objetivo (Umbral de Ruido según percentil {percentil} de Monte Carlo)",
            height=max(600, len(tablaCramer) * 22)
        )

        fig.update_yaxes(dtick=1, title="")
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="#E74C3C",
            annotation_text=f"Umbral de Ruido ({threshold:.4f})",
            annotation_position="bottom right"
        )
        fig.show()
    else:
        return tablaCramer



def pintar_violin_plots(
    df: pd.DataFrame, numeric_cols: list[str], target_col: str
):
    """Grafica grilla de violines respecto a la variable target categorica."""
    if not numeric_cols:
        return

    # squeeze=False fuerza a axes a ser siempre un array 2D
    fig, axes = plt.subplots(
        nrows=len(numeric_cols),
        ncols=1,
        figsize=(8, 3.2 * len(numeric_cols)),
        sharex=True,
        squeeze=False,
    )

    # Aplanar axes para iterar como una lista 1D sin importar el tamaño
    axes_flat = axes.ravel()

    # Iterar y graficar cada variable numérica
    for ax, num_var in zip(axes_flat, numeric_cols):
        sns.violinplot(
            data=df,
            x=target_col,
            y=num_var,
            ax=ax,
            palette="muted",
            inner="quartile",
            cut=0,
            legend=False,
            hue=target_col,
        )

        if ax == axes_flat[0]:
            ax.set_title(
                f"Distribución de features por categoria de {target_col}",
                fontsize=15,
                loc="center",
            )

        ax.set_ylabel(num_var, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_xlabel(target_col, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()    
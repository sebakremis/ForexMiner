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


def pintar_matriz_correlacion(
    df: pd.DataFrame,
    method: str = "spearman",
    figsize: tuple = (16, 12),
    cmap: str = "vlag",
    annot: bool = True,
    top_n_pairs: int = 10,
    title: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula y grafica la matriz de correlación triangular para la fase EDA

    de CiclosMacro, extrayendo además los pares más correlacionados.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las 20 features macroeconómicas y de precios procesadas.
    method : str, default 'spearman'
        Método de correlación ('spearman' o 'pearson'). Spearman se recomienda
        por la presencia de shocks de volatilidad no lineales (VIX, retornos).
    figsize : tuple, default (16, 12)
        Dimensiones de la figura de Matplotlib.
    cmap : str, default 'vlag'
        Paleta divergente centrada en cero (e.g. 'vlag', 'coolwarm',
        'RdBu_r').
    annot : bool, default True
        Si se imprimen los coeficientes numéricos en cada celda.
    top_n_pairs : int, default 10
        Cantidad de pares de mayor correlación absoluta a reportar en tabla.
    title : str, optional
        Título del heatmap.

    Retorna
    -------
    corr_matrix : pd.DataFrame
        Matriz cuadrada de correlaciones completa.
    top_pairs_df : pd.DataFrame
        Ranking con los pares de variables más fuertemente correlacionados.
    """
    # Asegurar únicamente columnas numéricas
    numeric_df = df.select_dtypes(include=[np.number])

    # Calcular matriz de correlación
    corr_matrix = numeric_df.corr(method=method)

    # Generar máscara para el triángulo superior y diagonal
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Graficar Heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0.0,
        annot=annot,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": f"Correlación de {method.capitalize()}"},
        annot_kws={"size": 8 if len(corr_matrix.columns) >= 20 else 10},
    )

    plot_title = (
        title
        if title
        else f"CiclosMacro — Matriz de Correlación ({method.capitalize()})"
    )
    plt.title(plot_title, fontsize=14, weight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.show()

    # Extraer ranking de pares con mayor correlación absoluta (sin duplicados ni diagonal)
    pairs = (corr_matrix.where(~mask).stack().reset_index(name="correlation").rename(columns={"level_0": "feature_1", "level_1": "feature_2"})
    )

    pairs["abs_correlation"] = pairs["correlation"].abs()
    top_pairs_df = (
        pairs.sort_values(by="abs_correlation", ascending=False)
        .drop(columns=["abs_correlation"])
        .head(top_n_pairs)
        .reset_index(drop=True)
    )

    return corr_matrix, top_pairs_df

def pintar_violin_plots(
        df:pd.DataFrame,
        numeric_cols:list[str],
        target_col:str
):
    '''
    Grafica grilla de violines respecto a la variable target categorica.
    '''
    # Configurar la figura:
    fig, axes = plt.subplots(
        nrows=len(numeric_cols),
        ncols=1,
        figsize=(8, 3.2 * len(numeric_cols)),
        sharex=True,
    )

    # Iterar y graficar cada variable numérica
    for ax, num_var in zip(axes, numeric_cols):
        sns.violinplot(
            data=df,
            x=target_col,
            y=num_var,
            ax=ax,
            palette="muted",
            inner="quartile",  # Muestra mediana y cuartiles dentro del violín
            cut=0,  # Limita el violín al rango real de los datos
            legend=False,
            hue=target_col
        )
        if ax == axes[0]:
            ax.set_title(f"Distribución de features por categoria de {target_col}", fontsize=15, loc="center")
        ax.set_ylabel(num_var, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        # Etiqueta del eje X
        ax.set_xlabel(target_col, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.show()
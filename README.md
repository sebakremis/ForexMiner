# 📊 ForexMiner

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Pipeline-green)

Este proyecto implementa un pipeline ETL (Extracción, Transformación y Carga) para construir un dataset financiero destinado al modelado de los principales pares de divisas Forex:
* `EURUSD`: el Euro contra el dólar USA.
* `AUDUSD`: dólar australiano contra el dólar USA.
* `GBPUSD`: libra británica contra el dólar.
* `USDJPY`: el dólar USA contra el yen japones.
* `USDCAD`: el dólar USA vs. el dólar canadiense.
* `USDCHF`: dólar USA vs. franco suizo.

Una vez procesados, normalizados y limpios, los datos son utilizados en la fase de modelado predictivo para entrenar algoritmos de Machine Learning, siendo la variable objetivo el rendimiento intrasemanal (Close vs Open de la misma semana) para cada par de divisas. El rendimiento es cuantificado en clases según 3 cuantiles, obteniendo de esta manera una **Clase 1 "Bajista"**, **Clase 2 "Lateral"** y una **Clase 3 "Alcista"**.

El proyecto esta diseñado para realizar predicciones contemporáneas en t (semana en curso), luego de la apertura del lunes de los mercados de Nueva York. El gap de apertura semanal (Open vs. Close de la semana anterior) resulta ser una de las variables predictoras más relevantes, junto con su asimetría.

Cuenta con tres fases:
* *Extracción*: obtención de datos a través de 2 APIs gratuitas: `yfinance` y `fred` (la segunda requiere obtener una clave api, se gestiona en la pagina oficial de la Reserva Federal: https://fred.stlouisfed.org/docs/api/api_key.html). Las intrucciones para ingresar tu clave en el repositorio local se encuentran en el fichero `src/data_sources.examples.py`.
* *Transformación*: tratamiento de valores inválidos, feature engineering, imputación de valores perdidos, escalado de variables con z-score, transformaciones logarítmicas y de yeo-johnson. Adicionalmente se calculan las variables lag, sobre aquellas que requieran ser desplazadas un período a fin de evitar el sesgo de anticipación (*Lookahead Bias*).
* *Modelado*: aplicación de modelo  clasificador *RandomForest* con datos completos, prueba de validación con partición de series temporales, evaluación de resultados, exportación de datos y predicciones y módulo para la optimización de hiper-parámetros.

Ademas el fichero `notebook_EDA.ipynb` ubicado en la raiz del proyecto permite realizar un análisis exploratorio de los datos transformados para cada par forex, incluyendo análisis de asimetrías, valores faltantes, gráficos de distribuciones, correlaciones y gráficos de violines representando las variables seleccionadas frente a las 3 clases de la variable objetivo seleccionada.

El mismo pipeline permite modelar los 6 cruces principales de divisas Forex contra el dólar estadounidense. Cada par de divisas cuenta con sus propios parámetros en el procesamiento de datos durante la fase de transformación.


El modelado con *RandomForestClassifier* arroja métricas de ajuste *moderadas*, actualmente con precisiones promedio del 41.5%. Este resultado es previsible dada la naturaleza ruidosa y no estacionaria de los datos financieros, asi como la disponibilidad limitada de datos gratuitos. Dicho esto, los valores obtenidos se encuentran por encima de la precisión del 33% que se obtendría mediante una clasificación al azar, indicando que el modelado permite encontrar algunos patrones y generalizar.

---
## 💻 Ejecución

Cada fase del proyecto cuenta con un modulo ejecutable desde el terminal:
- Fase de Extracción:

```bash
python -m src.extract
```
permite extraer los datos crudos comunes para todo el proyecto. Se almacenan en 2 ficheros tipo parquet dentro de la carpeta `/data/raw_data`.

- Fase de transformación:

```bash
python -m src.transform
```
Aplica la transformación para los 6 pares principales, generando 6 ficheros de datos limpios en la carpeta `/data/clean_data`.

Alternativamente, se puede transformar un único par, agregando el símbolo correspondiente en el parámetro --simbolo.
Ejemplo para transformar solo el par  dólar vs franco suizo:
```bash
python -m src.transform --simbolo usdchf
```

- Fase de modelado:
```bash
python -m src.modeling --simbolo usdchf --validar
```
Permite entrenar el modelo del símbolo indicado, y la inclusión del parámetro `--validar` habilita la validación y exportación de las métricas encontradas a un fichero excel en la carpeta `evaluacion_modelos`.

- Ejecución del flujo completo:

El fichero `main.py` permite ejecutar de una sola vez el flujo completo, incluyendo el modelado de los 6 pares Forex y la exportación de los resultados a excel:
```bash
python main.py
```


---
## 📊 Dataset y Variables

Se extraen de `yfinance` las series de precios semanales para los 6 pares forex estudiados, asi como de los siguientes activos:
`SPY`: etf representando el índice de acciones en Estados Unidos.
`Oil`: precio futuro del petróleo.
`Gold`: precio futuro del oro.

Por otro lado, se extraen mediante la api fred los siguientes datos macroeconómicos con granularidad diaria:
* `Real_yield_10y`: rendimiento real de los bonos del tesoro a 10 anos.
* `DXY_broad_goods`: índice del dólar USA frente a una canasta de monedas según el comercio de bienes.
* `VIX index`: índice de volatilidad CBOE.
* `St_louis_financial_stress`: índice de estrés financiero de la Reseva Federal de St. Louis (datos publicados semanalmente).

Además, se extraen mediante la api Fred los rendimientos históricos de los bonos a 10 años para cada uno de los países en las divisas principales, utilizando los bonos de Alemania como referencia en el caso del Euro. A partir de estos rendimientos se calculan los spreads con dólar estadounidense para cada divisa. 

La principal limitación de los datos de los rendimientos de bonos es que son publicados en Fred mensualmente, con una demora de hasta 2 meses. Ya que los datos informados representan el promedio mensual a fin de mes, se imputan los valores semanales faltantes mediante una extrapolación lineal, aprovechando el hecho que las medias móviles se mueven en forma de línea. 

**Importante**: Es necesario imputar de forma manual los últimos valores conocidos para los bonos a 10 años, actualizando los valores en el diccionario `ultimos_yields`. 

---
## 🗂️ Estructura del repo

```text
FOREXMINER/
├── data/                            # Almacenamiento local de datos
│   ├── clean_data/                  # Almacena los datos depurados para cada par forex
│   ├── evaluacion_modelos/          # Almacena los resultados obtenidos en el modelado
│   ├── raw_data/                    # Sub-directorio para almacenar los datos crudos
├── src/                             # Sub-directorio con los módulos de funciones 
│   ├── __init__.py                  # Fichero vacío, inicializa la carpeta como paquete
│   ├── config.py                    # Configuración global del proyecto 
│   ├── data_sources.example.py      # Fichero ejemplo para gestionar la clave API de Fred
│   ├── eda_utils.py                 # Funciones para el notebook de análisis exploratorio de datos
│   ├── extract.py                   # Módulo de la fase de extracción
│   ├── modeling.py                  # Módulo de la fase de modelado
│   └── transform.py                 # Módulo de la fase transformación
├── .gitignore                       # Reglas de git ignore
├── main.py                          # Programa principal para ejecutar el flujo completo del proyecto
├── notebook_EDA.ipynb               # Análisis Exploratorio de Datos (EDA)
├── LICENSE                          # Licencia del proyecto
├── README.md                        # Descripción del proyecto
└── requirements.txt                 # Dependencias necesarias
```

---
## 🚀 Requisitos e Instalación

1. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/sebakremis/ForexMiner.git
   cd ForexMiner
   ```

* Antes de instalar las librerías, se recomienda crear un entorno virtual específico.

2. Instala el entorno de dependencias requerido utilizando `pip`:
   ```bash
   pip install -r requirements.txt
   ```

---
## 📜 Licencia
Este repositorio se encuentra bajo una licencia de código abierto **MIT**. 
Puedes usar, modificar y distribuir el código libremente para cualquier propósito.

Para obtener más información, consulta el archivo [LICENCIA MIT](LICENSE).

⚠️ **Importante:** Este proyecto se desarrolló estrictamente con fines **académicos y educativos**.
`ForexMiner` **no** proporciona asesoramiento financiero ni recomendaciones de inversión. Los usuarios son los únicos responsables de las decisiones que pudiesen tomar basándose en la información generada por el mismo.

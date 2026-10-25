'''
src/config.py

Configuracion global del proyecto
'''
from pathlib import Path

# Ruta al fichero de datos
data_folder = "data"
clean_data_file = Path(data_folder) / "clean_data.parquet"

# Fecha inicial para descargar datos
default_start_date = "2006-01-01"
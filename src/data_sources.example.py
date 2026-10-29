"""
src/data_sources.example.py
Ejemplo de archivo para la gestión de claves API DE Fred.

-- NO ESCRIBAS tu clave real en este archivo. La clave se guarda en el archivo .env

-- Asegúrate de mantener tanto .env como /src/data_sources.py 
en tu .gitignore para evitar subir tus claves a repositorios públicos.


Después de clonar:
1. Crea un archivo llamado data_sources.py en la carpeta src, copiando este archivo.
2. Crea un archivo .env en la raíz del proyecto con la siguiente linea, reemplazando tu_clave_real_aqui por tu clave real:

FRED_API_KEY="tu_clave_real_aqui"


"""

"""
src/data_sources.py
Modulo privado para gestionar la carga de claves API.
NO ESCRIBAS tu clave real en este archivo. La clave se guarda en el archivo .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Se obtiene la clave API desde las variables de entorno
fred_api_key = os.getenv("FRED_API_KEY")
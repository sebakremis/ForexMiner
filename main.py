'''
main.py

Aplicacion principal para ejecutar el flujo completo sobre todos
los pares forex.
'''
from src.config import forex_cols
from src.extract import main as extraer
from src.transform import main as transformar
from src.modeling import main as modelar


'''
Bloque principal

Ejecutar desde el terminal:
python main.py
'''
def main():
    #extraer()
    #transformar()
    for pair in forex_cols:
        modelar(simbolo=pair)


if __name__ == '__main__':
    main()
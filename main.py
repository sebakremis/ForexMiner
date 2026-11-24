'''
main.py

Aplicacion principal para ejecutar el flujo completo sobre todos
los pares forex.
'''
from src.extract import main as extraer
from src.transform import main as transformar
from src.filtering import main as filtrar
from src.modeling import main as modelar


'''
Bloque principal

Ejecutar desde el terminal:
python main.py
'''
def main():
    extraer()
    transformar()
    filtrar()
    modelar(modelo="rf", evaluar=True)


if __name__ == '__main__':
    main()
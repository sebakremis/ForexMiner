'''
main.py

Modulo principal para ejecutar el flujo completo
'''
import argparse

from src.extract import main as extraer
from src.transform import main as transformar
from src.cat_modeling import main as modelar, get_parser as parser_modelado

def parse_cli():
    # Se reutiliza el parser de modelado que cubre todos los flags
    parser = argparse.ArgumentParser(
        parents=[parser_modelado()],
        description="Pipeline completo",
    )
    return parser.parse_args()

def main():
    args = parse_cli()

    # Cada fase recibe lo que necesita
    extraer(simbolo=args.simbolo, intervalo=args.intervalo)
    transformar(simbolo=args.simbolo, intervalo=args.intervalo)
    modelar(simbolo=args.simbolo,intervalo=args.intervalo, evaluar=args.evaluar)

if __name__ == "__main__":
    main()
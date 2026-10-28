'''
src/cli.py
Define el parser de argumentos compartidos
'''
from src.config import intervalo_defecto
import argparse

def get_base_parser() -> argparse.ArgumentParser:
    """Argumentos compartidos por casi todo el pipeline."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--simbolo", type=str, default="SPY", help="Símbolo del activo")
    parser.add_argument("--intervalo", type=str, default=intervalo_defecto, help=f"Intervalo temporal (por defecto '{intervalo_defecto}')")
    return parser
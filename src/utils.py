"""
Funciones auxiliares del proyecto.
"""

import os


def create_directories(paths: list) -> None:
    """
    Crea directorios si no existen.

    Args:
        paths: Lista de rutas a crear
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)

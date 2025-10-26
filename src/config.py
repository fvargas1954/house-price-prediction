"""
Módulo de configuración del proyecto.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Diccionario con la configuración
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No se encontró el archivo: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Obtiene los parámetros del modelo.

    Args:
        config: Diccionario de configuración

    Returns:
        Diccionario con hiperparámetros
    """
    return config["model"].get("params", {})

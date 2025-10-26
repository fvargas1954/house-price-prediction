"""
Tests básicos para el modelo.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def test_data_exists():
    """Verifica que existan los datos."""
    import os

    assert os.path.exists("data/train_data.csv"), "Falta train_data.csv"


def test_model_training():
    """Verifica que el modelo pueda entrenarse."""
    # Datos sintéticos
    X = pd.DataFrame(np.random.rand(100, 5))
    y = pd.Series(np.random.randint(0, 2, 100))

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    assert model is not None
    assert hasattr(model, "predict")


def test_config_loads():
    """Verifica que el config se cargue."""
    from src.config import load_config

    config = load_config()
    assert "project" in config
    assert "model" in config
    assert "mlflow" in config

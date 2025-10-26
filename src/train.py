"""
Script de entrenamiento de detección de fraude.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Agregar raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config, get_model_params
from src.utils import create_directories

print("🚀 Iniciando entrenamiento del modelo de detección de fraude...")

# Cargar configuración
config = load_config()
print(f"✅ Configuración cargada: {config['project']['name']}")

# Crear directorios
create_directories(["mlruns"])

# Configurar MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

# Cargar datos
print("\n📥 Cargando datos...")
train_df = pd.read_csv(config["data"]["train_path"])
print(f"✅ Train: {train_df.shape[0]:,} filas")

# Separar X e y
X_train = train_df.drop("Class", axis=1)
y_train = train_df["Class"]

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[["Time", "Amount"]])
X_train[["Time", "Amount"]] = X_train_scaled

# Entrenar
print("\n🤖 Entrenando modelo...")
params = get_model_params(config)
model = LogisticRegression(**params)
model.fit(X_train, y_train)
print("✅ Modelo entrenado")

# Evaluar
y_pred = model.predict(X_train)
y_pred_proba = model.predict_proba(X_train)[:, 1]

accuracy = accuracy_score(y_train, y_pred)
precision = precision_score(y_train, y_pred)
recall = recall_score(y_train, y_pred)
f1 = f1_score(y_train, y_pred)
roc_auc = roc_auc_score(y_train, y_pred_proba)

print(f"\n📊 Métricas:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1: {f1:.4f}")
print(f"   ROC-AUC: {roc_auc:.4f}")

# Registrar en MLflow
print("\n💾 Registrando en MLflow...")
with mlflow.start_run():
    # Parámetros
    mlflow.log_params(params)

    # Métricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Signature
    signature = infer_signature(X_train, y_pred_proba)
    input_example = X_train.iloc[:5]

    # Modelo
    mlflow.sklearn.log_model(
        model, "model", signature=signature, input_example=input_example
    )

    print("✅ Modelo registrado en MLflow")

print("\n🎉 Entrenamiento completado!")

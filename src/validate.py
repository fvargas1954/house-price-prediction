"""
Script de validación del modelo de detección de fraude.
"""

import os
import sys
import pandas as pd
import numpy as np
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

# Agregar raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import load_config

print("🔍 Iniciando validación del modelo...")

# Cargar configuración
config = load_config()

# Configurar MLflow
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

# Obtener último modelo
experiment = mlflow.get_experiment_by_name(config["mlflow"]["experiment_name"])

if experiment is None:
    print("❌ No se encontró el experimento. Ejecuta train.py primero.")
    sys.exit(1)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
)

if runs.empty:
    print("❌ No hay runs disponibles.")
    sys.exit(1)

run_id = runs.iloc[0]["run_id"]
print(f"✅ Modelo encontrado - Run ID: {run_id}")

# Cargar modelo
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
print("✅ Modelo cargado desde MLflow")

# Cargar datos de test
print("\n📥 Cargando datos de test...")
test_df = pd.read_csv(config["data"]["test_path"])
print(f"✅ Test: {test_df.shape[0]:,} filas")

# Separar X e y
X_test = test_df.drop("Class", axis=1)
y_test = test_df["Class"]

# Escalar
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test[["Time", "Amount"]])
X_test[["Time", "Amount"]] = X_test_scaled

# Predecir
print("\n🔮 Realizando predicciones...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluar
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📈 MÉTRICAS DE VALIDACIÓN:")
print(f"   Accuracy: {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")
print(f"   F1: {f1:.4f}")
print(f"   ROC-AUC: {roc_auc:.4f}")

# Validar umbrales
print(f"\n🎯 VALIDACIÓN DE UMBRALES:")

THRESHOLD_F1 = config["model"]["threshold_f1"]
THRESHOLD_RECALL = config["model"]["threshold_recall"]
THRESHOLD_ROC_AUC = config["model"]["threshold_roc_auc"]

checks = [
    ("F1", f1, THRESHOLD_F1),
    ("Recall", recall, THRESHOLD_RECALL),
    ("ROC-AUC", roc_auc, THRESHOLD_ROC_AUC),
]

all_passed = True
for metric_name, value, threshold in checks:
    status = "✅ PASA" if value >= threshold else "❌ FALLA"
    if value < threshold:
        all_passed = False
    print(f"   {metric_name:<10} {value:.4f} >= {threshold:.2f}  {status}")

print("\n" + "=" * 60)
if all_passed:
    print("✅ MODELO VALIDADO EXITOSAMENTE")
    print("✅ Apto para producción")
    sys.exit(0)
else:
    print("❌ MODELO NO CUMPLE CRITERIOS")
    print("❌ NO apto para producción")
    sys.exit(1)

# Proyecto

Pipeline de Machine Learning para Detección de Fraude

# Descripción

Este proyecto implementa un pipeline automatizado de machine learning para la detección de fraude en transacciones con tarjeta de crédito, utilizando MLOps, integración continua (CI/CD) mediante GitHub Actions, tracking de experimentos con MLflow, y automatización de tareas con Makefile.


## Dataset

Fuente: Credit Card Fraud Detection Dataset
Características:

Tamaño del dataset: 284,807
Total de transacciones de la muestra a trabajar: 50,000
Distribución: 40,000 entrenamiento / 10,000 prueba
Variables: 30 características (V1-V28, Time, Amount)
Variable objetivo: Class (0 = legítima, 1 = fraude)

El dataset contiene transacciones realizadas con tarjetas de crédito, donde la clase positiva (fraude) representa una pequeña proporción del total de transacciones, simulando un escenario realista de detección de fraude.

## Estructura del Proyecto

mlflow-deploy/
├── data/
│   ├── creditcard.csv          # Total Muestra
│   ├── train_data.csv          # Datos de entrenamiento
│   └── test_data.csv           # Datos de validación externa
├── mlruns/                     # Tracking de MLflow (generado)
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml       # Pipeline CI/CD automatizado
│       └── ml.yml              # Define el pipeline de CI/CD en GitHub Actions.
├── src/
│   └── __init__.py             # Convierte la carpeta src/ en un paquete de Python
│   └── config.py
│   └── train.py                # Script de entrenamiento
│   └── utils.py                # Contiene funciones auxiliares reutilizables
│   └── validate.py             # Script de validación
├── tets/
│   └── __init__.py             # Convierte la carpeta tests/ en un paquete de Python.
│   └── test_model.py
├── .gitignore       
├── Makefile                    # Comandos simplificados
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación

## Modelo

**Algoritmo:** Logistic Regression
- Solver: liblinear
- Class weight: balanced (para manejar desbalance)
- Max iterations: 1000

**Preprocesamiento:**
- Escalado de features Time y Amount con StandardScaler
- Features V1-V28 ya están normalizadas (PCA)

## Métricas

El modelo se evalúa con múltiples métricas apropiadas para clasificación desbalanceada:

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **Accuracy** | 97.56% | Predicciones correctas totales |
| **Precision** | 6.12% | De los predichos como fraude, % reales |
| **Recall** | 91.84% | % de fraudes detectados |
| **F1-Score** | 11.47% | Balance precision-recall |
| **ROC-AUC** | 97.21% | Capacidad de discriminación |



## Uso Local

### Instalación
```bash
# Clonar repositorio
git clone https://github.com/fvargas1954/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### Entrenamiento
```bash
# Opción 1: Con Makefile
make train

# Opción 2: Directo
python train.py
```

### Validación
```bash
# Opción 1: Con Makefile
make validate

# Opción 2: Directo
python validate.py
```

### MLflow UI
```bash
mlflow ui
# Abrir: http://127.0.0.1:5000
```

## Pipeline CI/CD

El workflow de GitHub Actions se ejecuta automáticamente en cada push a `main`:

### Pasos del Pipeline

1. **Checkout** - Clona el código
2. **Setup Python** - Configura Python 3.9
3. **Install** - Instala dependencias
4. **Train** - Entrena el modelo y registra en MLflow
5. **Validate** - Valida con datos externos y verifica umbrales
6. **Upload Artifacts** - Guarda mlruns/ como artefacto descargable

### Umbrales de Validación

El modelo debe cumplir estos criterios para pasar:

- **F1-Score ≥ 0.10**
- **Recall ≥ 0.80** (crítico: detectar fraudes)
- **ROC-AUC ≥ 0.90**

Si no cumple, el pipeline falla y no se promociona.

## Registro en MLflow

El modelo registra:

### Parámetros
- model_type: LogisticRegression
- solver: liblinear
- class_weight: balanced
- max_iter: 1000
- n_features: 30
- n_samples_train: 227,845

### Métricas
- accuracy
- precision
- recall
- f1_score
- roc_auc

### Artefactos
- Modelo entrenado (con signature e input_example)

## Características Técnicas

### Buenas Prácticas Implementadas

**Dataset Externo:** Datos reales de Kaggle (no sklearn.datasets)  
**Validación Externa:** Evaluación con datos no vistos  
**Signature & Input Example:** Modelo listo para inferencia  
**Código Modular:** Funciones con docstrings  
**Manejo de Desbalance:** Class weighting + métricas apropiadas  
**Tracking Completo:** Parámetros, métricas y artefactos en MLflow  
**Automatización:** CI/CD con GitHub Actions  
**Quality Gates:** Validación con umbrales automáticos

## Tecnologías

- **Python 3.9+**
- **MLflow 2.8.1** - Tracking y registro de modelos
- **Scikit-learn 1.3.0** - Machine Learning
- **Pandas 2.0.3** - Manipulación de datos
- **GitHub Actions** - CI/CD automatizado

"""
model_factory
=============
Paquete modular para el preprocesamiento, evaluación baseline, optimización de
hiperparámetros, entrenamiento final y exportación de los modelos de clasificación
de aprobación de préstamos.
"""

from model_factory.config import (
    CATEGORICAL_COLUMNS,
    CSV_FILENAME,
    CSV_PATH,
    FEATURE_ORDER,
    MODELS_DIR,
    NUMERICAL_COLUMNS,
    RESULTS_FILE,
    TARGET_COLUMN,
)

__all__ = [
    "CATEGORICAL_COLUMNS",
    "CSV_FILENAME",
    "CSV_PATH",
    "FEATURE_ORDER",
    "MODELS_DIR",
    "NUMERICAL_COLUMNS",
    "RESULTS_FILE",
    "TARGET_COLUMN",
]

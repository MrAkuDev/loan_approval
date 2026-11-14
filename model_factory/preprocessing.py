"""Construcción del preprocesamiento y ensamblado de Pipelines de scikit-learn."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_factory.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Construye el ColumnTransformer compartido para variables numéricas y categóricas."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
    )


def build_training_pipeline(estimator) -> Pipeline:
    """Envuelve el preprocesamiento y el estimador en un Pipeline unificado."""
    return Pipeline(
        [
            ("preprocesamiento", build_preprocessor()),
            ("model", estimator),
        ]
    )

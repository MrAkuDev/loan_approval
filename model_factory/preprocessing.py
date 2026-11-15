"""Construcción del preprocesamiento y ensamblado de Pipelines de scikit-learn."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_factory.config import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Construye el ColumnTransformer con imputación y escalamiento/codificación sin fuga de datos."""
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_COLUMNS),
            ("cat", cat_pipeline, CATEGORICAL_COLUMNS),
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

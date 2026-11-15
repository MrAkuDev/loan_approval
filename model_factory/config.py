"""Configuración central, rutas, esquema de variables y especificaciones de modelos."""

import os
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

CSV_FILENAME = "credit_risk_dataset.csv"
CSV_PATH = os.path.join(PROJECT_ROOT, CSV_FILENAME)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_FILE = os.path.join(MODELS_DIR, "evaluation_results.json")

TARGET_COLUMN = "loan_status"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5
SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
TUNING_SCORING = "f1"

MAX_VALID_AGE = 100
MAX_VALID_EMP_LENGTH = 60

# Orden exacto de características tal como aparecen en credit_risk_dataset.csv (Kaggle: laotse/credit-risk-dataset)
CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

NUMERICAL_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

FEATURE_ORDER = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length",
]


def get_model_specs() -> dict[str, dict[str, Any]]:
    """Define estimadores, espacios de búsqueda y artefactos de salida por modelo."""
    return {
        "SVM RBF": {
            "filename": "svm_rbf.joblib",
            "estimator_cls": SVC,
            "base_params": {"kernel": "rbf", "random_state": RANDOM_STATE},
            "final_extra_params": {"probability": True},
            "search_strategy": "random",
            "n_iter": 12,
            "param_space": {
                "C": [0.01, 0.1, 1.0, 5.0],
                "gamma": ["scale", "auto", 0.001, 0.01],
                "class_weight": [None, "balanced"],
            },
            "heatmap_config": {
                "x_param": "C",
                "x_values": [0.01, 0.1, 1.0, 5.0],
                "x_label": "Regularización (C)",
                "y_param": "gamma",
                "y_values": ["scale", "auto", 0.001, 0.01],
                "y_label": "Coeficiente Kernel (gamma)",
            },
        },
        "Random Forest": {
            "filename": "random_forest.joblib",
            "estimator_cls": RandomForestClassifier,
            "base_params": {"random_state": RANDOM_STATE, "n_jobs": -1},
            "final_extra_params": {},
            "search_strategy": "random",
            "n_iter": 12,
            "param_space": {
                "n_estimators": [100, 150, 200, 300],
                "max_depth": [None, 10, 15, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
            },
            "heatmap_config": {
                "x_param": "n_estimators",
                "x_values": [100, 150, 200, 300],
                "x_label": "Número de Estimadores (n_estimators)",
                "y_param": "max_depth",
                "y_values": [10, 15, 20, 30, None],
                "y_label": "Profundidad Máxima (max_depth)",
            },
        },
        "Logistic Regression": {
            "filename": "logistic_regression.joblib",
            "estimator_cls": LogisticRegression,
            "base_params": {"random_state": RANDOM_STATE, "max_iter": 1000},
            "final_extra_params": {},
            "search_strategy": "grid",
            "n_iter": None,
            "param_space": {
                "C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "class_weight": [None, "balanced"],
                "solver": ["lbfgs"],
            },
            "heatmap_config": {
                "x_param": "C",
                "x_values": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
                "x_label": "Inversa de Regularización (C)",
                "y_param": "class_weight",
                "y_values": [None, "balanced"],
                "y_label": "Balanceo de Clases (class_weight)",
            },
        },
    }

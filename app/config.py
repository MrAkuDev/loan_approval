"""Configuración global, rutas de archivos y esquema de variables para la app Streamlit."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

CSV_FILENAME = "credit_risk_dataset.csv"
CSV_PATH = os.path.join(PROJECT_ROOT, CSV_FILENAME)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

MAX_VALID_AGE = 100
MAX_VALID_EMP_LENGTH = 60

MODEL_FILES = {
    "SVM RBF": os.path.join(MODELS_DIR, "svm_rbf.joblib"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
}

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

NUMERICAL_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

VALID_CATEGORIES = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "DEBTCONSOLIDATION",
        "HOMEIMPROVEMENT",
    ],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}

MODEL_PALETTE = ["#1D4ED8", "#059669", "#D97706"]

TOP_BOUNDARY_FEATURES = [
    "loan_percent_income",
    "loan_int_rate",
    "person_income",
    "loan_amnt",
    "person_emp_length",
    "person_age",
    "cb_person_cred_hist_length",
]

FEATURE_LABELS = {
    "loan_percent_income": "Proporción Préstamo / Ingreso (loan_percent_income)",
    "loan_int_rate": "Tasa de Interés % (loan_int_rate)",
    "person_income": "Ingreso Anual USD (person_income)",
    "loan_amnt": "Monto Solicitado USD (loan_amnt)",
    "person_emp_length": "Antigüedad Laboral en años (person_emp_length)",
    "person_age": "Edad del Solicitante (person_age)",
    "cb_person_cred_hist_length": "Antigüedad Historial Crediticio (años)",
}

FEATURE_SHORT_LABELS = {
    "person_age": "Edad",
    "person_income": "Ingreso Anual",
    "person_home_ownership": "Vivienda",
    "person_emp_length": "Antig. Laboral",
    "loan_intent": "Propósito",
    "loan_grade": "Grado Riesgo",
    "loan_amnt": "Monto Préstamo",
    "loan_int_rate": "Tasa Interés",
    "loan_percent_income": "% Ingreso",
    "cb_person_default_on_file": "Mora Previa",
    "cb_person_cred_hist_length": "Antig. Historial",
}

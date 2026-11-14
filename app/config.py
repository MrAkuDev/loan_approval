"""Configuración global, rutas de archivos y esquema de variables para la app Streamlit."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

CSV_FILENAME = "loan_data.csv"
CSV_PATH = os.path.join(PROJECT_ROOT, CSV_FILENAME)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

MODEL_FILES = {
    "SVM RBF": os.path.join(MODELS_DIR, "svm_rbf.joblib"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
}

FEATURE_ORDER = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

NUMERICAL_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

VALID_CATEGORIES = {
    "person_gender": ["female", "male"],
    "person_education": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

MODEL_PALETTE = ["#1D4ED8", "#059669", "#D97706"]

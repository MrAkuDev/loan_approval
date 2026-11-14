"""Carga cacheada de modelos/datos, cálculo de correlaciones e inferencia."""

import json
import os

import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.config import CSV_PATH, FEATURE_ORDER, MODEL_FILES, RESULTS_PATH


@st.cache_resource
def load_models(model_mtimes: tuple):
    """Carga los pipelines serializados `.joblib` desde `models/`."""
    loaded = {}
    missing = []
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            missing.append(path)
    return loaded, missing


@st.cache_data
def load_evaluation_results(file_mtime: float):
    """Carga el JSON con métricas experimentales reales de CV y Test."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_cv5_eda_data(
    csv_mtime: float,
    fold_choice: str = "Todos los 5 folds (cv_5)",
    balanced: bool = True,
):
    """Carga X_train usando StratifiedKFold(n_splits=5, shuffle=True, random_state=42) y opcionalmente balancea 50/50."""
    if not os.path.exists(CSV_PATH):
        return None
    df_all = pd.read_csv(CSV_PATH)
    X_all = df_all[FEATURE_ORDER]
    y_all = df_all["loan_status"]
    X_tr, _, y_tr, _ = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

    cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv_5.split(X_tr, y_tr))

    if fold_choice.startswith("Fold "):
        fold_idx = int(fold_choice.split()[1]) - 1
        _, val_idx = splits[fold_idx]
        df_sub = X_tr.iloc[val_idx].copy()
        df_sub["loan_status"] = y_tr.iloc[val_idx].values
    else:
        parts = []
        for _, val_idx in splits:
            part = X_tr.iloc[val_idx].copy()
            part["loan_status"] = y_tr.iloc[val_idx].values
            parts.append(part)
        df_sub = pd.concat(parts, axis=0)

    if balanced:
        c0 = df_sub[df_sub["loan_status"] == 0]
        c1 = df_sub[df_sub["loan_status"] == 1]
        n_min = min(len(c0), len(c1))
        df_sub = pd.concat(
            [c0.sample(n=n_min, random_state=42), c1.sample(n=n_min, random_state=42)],
            axis=0,
        ).sample(frac=1.0, random_state=42)

    return df_sub


@st.cache_data
def compute_corr_payload_from_df(df_sub: pd.DataFrame) -> dict:
    """Calcula la matriz de correlación One-Hot ordenada por relación con loan_status."""
    df_enc = pd.get_dummies(df_sub[FEATURE_ORDER + ["loan_status"]], drop_first=True, dtype=float)
    corr_df = df_enc.corr()
    target_corr = corr_df["loan_status"].drop("loan_status").sort_values(ascending=False)
    ordered_cols = list(target_corr.index) + ["loan_status"]
    corr_ordered = corr_df.loc[ordered_cols, ordered_cols]
    return {
        "columns": ordered_cols,
        "matrix": [[round(float(val), 4) for val in row] for row in corr_ordered.values],
        "target_correlations": {col: round(float(val), 4) for col, val in target_corr.items()},
    }


def predict_with_model(pipeline, X_df: pd.DataFrame) -> dict:
    """Genera predicción real, probabilidad de clase 1 (y 0) y decision_function si aplica."""
    pred_class = int(pipeline.predict(X_df)[0])

    prob_0, prob_1 = None, None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_df)[0]
        prob_0 = float(probs[0])
        prob_1 = float(probs[1])

    dec_score = None
    if hasattr(pipeline, "decision_function"):
        dec_score = float(pipeline.decision_function(X_df)[0])

    return {
        "pred": pred_class,
        "label": "Aprobado (1)" if pred_class == 1 else "No Aprobado (0)",
        "prob_0": prob_0,
        "prob_1": prob_1,
        "decision_score": dec_score,
    }


def load_artifacts():
    """Carga los 3 modelos entrenados y el archivo JSON de resultados de evaluación."""
    model_mtimes = tuple(
        (name, os.path.getmtime(path) if os.path.exists(path) else 0.0)
        for name, path in MODEL_FILES.items()
    )
    eval_mtime = os.path.getmtime(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else 0.0

    models, missing_files = load_models(model_mtimes)
    eval_results = load_evaluation_results(eval_mtime)
    return models, missing_files, eval_results


def load_sample_template(nrows: int = 5) -> pd.DataFrame | None:
    """Carga las primeras `nrows` filas de `loan_data.csv` como plantilla descargable."""
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH, nrows=nrows)
    return None

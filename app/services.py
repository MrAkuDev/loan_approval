"""Carga cacheada de modelos/datos, cálculo de correlaciones e inferencia."""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from app.config import (
    CSV_PATH,
    FEATURE_ORDER,
    MODEL_FILES,
    NUMERICAL_COLUMNS,
    RESULTS_PATH,
    VALID_CATEGORIES,
)


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


def balance_binary_target(df_sub: pd.DataFrame) -> pd.DataFrame:
    """Iguala 50/50 las clases 0 y 1 de `loan_status` cuando ambas están presentes."""
    c0 = df_sub[df_sub["loan_status"] == 0]
    c1 = df_sub[df_sub["loan_status"] == 1]
    n_min = min(len(c0), len(c1))
    if n_min == 0:
        return df_sub.copy()
    return pd.concat(
        [c0.sample(n=n_min, random_state=42), c1.sample(n=n_min, random_state=42)],
        axis=0,
    ).sample(frac=1.0, random_state=42)


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
        df_sub = balance_binary_target(df_sub)

    return df_sub


@st.cache_data
def load_test_eda_data(csv_mtime: float) -> pd.DataFrame | None:
    """Carga el conjunto oficial Test Holdout (20%, random_state=42, estratificado)."""
    if not os.path.exists(CSV_PATH):
        return None
    df_all = pd.read_csv(CSV_PATH)
    X_all = df_all[FEATURE_ORDER]
    y_all = df_all["loan_status"]
    _, X_te, _, y_te = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)
    df_test = X_te.copy()
    df_test["loan_status"] = y_te.values
    return df_test


@st.cache_data
def compute_corr_payload_from_df(df_sub: pd.DataFrame) -> dict:
    """Calcula la matriz de correlación One-Hot ordenada por relación con loan_status."""
    df_enc = pd.get_dummies(df_sub[FEATURE_ORDER + ["loan_status"]], drop_first=True, dtype=float)
    corr_df = df_enc.corr().fillna(0.0)
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


def _feature_axis_bounds(series: pd.Series, col: str) -> tuple[float, float]:
    """Calcula límites limpios de visualización (evitando outliers extremos en gráficas 2D)."""
    if col == "person_income":
        lo, hi = float(series.quantile(0.01)), float(series.quantile(0.95))
    elif col == "loan_percent_income":
        lo, hi = 0.0, float(min(series.max(), max(0.48, series.quantile(0.995))))
    else:
        lo, hi = float(series.quantile(0.01)), float(series.quantile(0.99))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


@st.cache_data
def compute_decision_boundary_grids(
    _models: dict,
    _df_eda: pd.DataFrame,
    feat_x: str,
    feat_y: str,
    default_val: str,
    home_val: str,
    balanced_tag: bool,
    dataset_tag: str = "cv5_train",
    grid_res: int = 75,
) -> dict:
    """Evalúa la malla 2D de probabilidades de cada modelo sobre las dos características elegidas."""
    baseline = {
        col: (
            float(_df_eda[col].median())
            if col in NUMERICAL_COLUMNS
            else str(_df_eda[col].mode().iloc[0])
        )
        for col in FEATURE_ORDER
    }
    baseline["previous_loan_defaults_on_file"] = default_val
    baseline["person_home_ownership"] = home_val

    x_min, x_max = _feature_axis_bounds(_df_eda[feat_x], feat_x)
    y_min, y_max = _feature_axis_bounds(_df_eda[feat_y], feat_y)

    xs = np.linspace(x_min, x_max, grid_res)
    ys = np.linspace(y_min, y_max, grid_res)
    xx, yy = np.meshgrid(xs, ys)

    grid_df = pd.DataFrame([baseline] * (grid_res * grid_res))[FEATURE_ORDER]
    grid_df[feat_x] = xx.ravel()
    grid_df[feat_y] = yy.ravel()

    prob_grids = {}
    for m_name, pipe in _models.items():
        if hasattr(pipe, "predict_proba"):
            probs = pipe.predict_proba(grid_df)[:, 1]
        else:
            probs = pipe.predict(grid_df).astype(float)
        prob_grids[m_name] = probs.reshape(xx.shape)

    in_bounds = (
        (_df_eda[feat_x] >= x_min)
        & (_df_eda[feat_x] <= x_max)
        & (_df_eda[feat_y] >= y_min)
        & (_df_eda[feat_y] <= y_max)
    )
    both_mask = (
        (_df_eda["previous_loan_defaults_on_file"] == default_val)
        & (_df_eda["person_home_ownership"] == home_val)
        & in_bounds
    )
    matched_both = int(both_mask.sum()) >= 30
    scatter_pool = _df_eda[both_mask] if matched_both else _df_eda[
        (_df_eda["previous_loan_defaults_on_file"] == default_val) & in_bounds
    ]
    scatter_sample = scatter_pool.sample(n=min(450, len(scatter_pool)), random_state=42)

    return {
        "xx": xx,
        "yy": yy,
        "prob_grids": prob_grids,
        "scatter": scatter_sample[[feat_x, feat_y, "loan_status"]].copy(),
        "x_bounds": (x_min, x_max),
        "y_bounds": (y_min, y_max),
        "baseline": baseline,
        "matched_both_cats": matched_both,
    }


def compute_batch_eval_metrics(
    model_names: list[str],
    y_true: pd.Series,
    df_out: pd.DataFrame,
) -> tuple[dict[str, dict], bool]:
    """Calcula métricas reales, curvas ROC y matrices 2x2 sobre un CSV etiquetado con `loan_status`."""
    y_arr = y_true.to_numpy(dtype=int)
    has_both_classes = len(np.unique(y_arr)) == 2
    metrics_by_model: dict[str, dict] = {}

    for m_name in model_names:
        short_key = m_name.lower().replace(" ", "_")
        preds = df_out[f"pred_{short_key}"].to_numpy(dtype=int)
        probs = df_out[f"prob1_{short_key}"].to_numpy(dtype=float)

        if has_both_classes:
            auc_val = round(float(roc_auc_score(y_arr, probs)), 4)
            roc_curve_vals = roc_curve(y_arr, probs)
            fpr, tpr = roc_curve_vals[0], roc_curve_vals[1]
            step = max(1, len(fpr) // 250)
            idx_sub = list(range(0, len(fpr), step))
            if idx_sub[-1] != len(fpr) - 1:
                idx_sub.append(len(fpr) - 1)
            roc_payload = {
                "fpr": [round(float(fpr[i]), 4) for i in idx_sub],
                "tpr": [round(float(tpr[i]), 4) for i in idx_sub],
            }
        else:
            auc_val = None
            roc_payload = None

        metrics_by_model[m_name] = {
            "accuracy": round(float(accuracy_score(y_arr, preds)), 4),
            "precision": round(float(precision_score(y_arr, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_arr, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_arr, preds, zero_division=0)), 4),
            "roc_auc": auc_val,
            "roc_curve": roc_payload,
            "confusion_matrix": confusion_matrix(y_arr, preds, labels=[0, 1]).tolist(),
        }

    return metrics_by_model, has_both_classes

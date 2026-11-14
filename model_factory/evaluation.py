"""Funciones de evaluación en validación cruzada (Stage 1) y conjunto de prueba."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from model_factory.config import SCORING_METRICS
from model_factory.preprocessing import build_training_pipeline


def compute_scores_and_probs(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Obtiene predicciones de clase, probabilidades de clase 1 y scores continuos para ROC."""
    y_pred = pipeline.predict(X)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        dec = pipeline.decision_function(X)
        y_prob = 1.0 / (1.0 + np.exp(-np.clip(dec, -500, 500)))
    else:
        y_prob = y_pred.astype(float)
    return y_pred, y_prob


def evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calcula métricas reales, matriz de confusión y puntos de curva ROC sobre Test."""
    y_pred, y_prob = compute_scores_and_probs(pipeline, X_test)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    if len(fpr) > 120:
        idx = np.linspace(0, len(fpr) - 1, 120, dtype=int)
        fpr_sample = fpr[idx].tolist()
        tpr_sample = tpr[idx].tolist()
    else:
        fpr_sample = fpr.tolist()
        tpr_sample = tpr.tolist()

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(x, 4) for x in fpr_sample],
            "tpr": [round(x, 4) for x in tpr_sample],
        },
    }


def evaluate_baseline_cv(
    specs: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Evalúa los modelos con parámetros por defecto mediante validación cruzada estratificada."""
    baseline_cv_results = {}

    for name, cfg in specs.items():
        estimator = cfg["estimator_cls"](**cfg["base_params"])
        pipe = build_training_pipeline(estimator)

        t0 = time.time()
        print(f"Evaluando baseline (5-fold CV): {name}...")
        cv_out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING_METRICS, n_jobs=-1)
        y_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=-1)
        cm_cv = confusion_matrix(y_train, y_pred_cv)
        elapsed = time.time() - t0

        baseline_cv_results[name] = {
            "fit_time_sec": round(elapsed, 2),
            "accuracy_mean": round(float(np.mean(cv_out["test_accuracy"])), 4),
            "accuracy_std": round(float(np.std(cv_out["test_accuracy"])), 4),
            "precision_mean": round(float(np.mean(cv_out["test_precision"])), 4),
            "recall_mean": round(float(np.mean(cv_out["test_recall"])), 4),
            "f1_mean": round(float(np.mean(cv_out["test_f1"])), 4),
            "f1_std": round(float(np.std(cv_out["test_f1"])), 4),
            "roc_auc_mean": round(float(np.mean(cv_out["test_roc_auc"])), 4),
            "confusion_matrix": cm_cv.tolist(),
        }
        print(
            f"   -> Train CV Acc: {baseline_cv_results[name]['accuracy_mean']:.4f} | "
            f"F1: {baseline_cv_results[name]['f1_mean']:.4f} | "
            f"AUC: {baseline_cv_results[name]['roc_auc_mean']:.4f} | "
            f"Tiempo: {elapsed:.1f}s"
        )

    return baseline_cv_results

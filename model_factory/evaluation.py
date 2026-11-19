"""Funciones de evaluación en validación cruzada (Stage 1) y conjunto de prueba."""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
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
        "f2": round(float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": [round(x, 4) for x in fpr_sample],
            "tpr": [round(x, 4) for x in tpr_sample],
        },
    }


def _evaluate_cv_stage(
    specs: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
    params_by_model: dict[str, dict] | None = None,
    stage_label: str = "baseline",
) -> dict[str, dict]:
    """Evalúa los modelos en validación cruzada estratificada sobre Train."""
    cv_results = {}

    for name, cfg in specs.items():
        extra_params = params_by_model.get(name, {}) if params_by_model else {}
        estimator = cfg["estimator_cls"](**{**cfg["base_params"], **extra_params})
        pipe = build_training_pipeline(estimator)

        t0 = time.time()
        print(f"Evaluando {stage_label} (5-fold CV): {name}...")
        cv_out = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING_METRICS, n_jobs=-1)
        y_pred_cv = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=-1)
        cm_cv = confusion_matrix(y_train, y_pred_cv)
        elapsed = time.time() - t0

        entry = {
            "fit_time_sec": round(elapsed, 2),
            "accuracy_mean": round(float(np.mean(cv_out["test_accuracy"])), 4),
            "accuracy_std": round(float(np.std(cv_out["test_accuracy"])), 4),
            "precision_mean": round(float(np.mean(cv_out["test_precision"])), 4),
            "recall_mean": round(float(np.mean(cv_out["test_recall"])), 4),
            "f1_mean": round(float(np.mean(cv_out["test_f1"])), 4),
            "f1_std": round(float(np.std(cv_out["test_f1"])), 4),
            "f2_mean": round(float(np.mean(cv_out["test_f2"])), 4),
            "f2_std": round(float(np.std(cv_out["test_f2"])), 4),
            "roc_auc_mean": round(float(np.mean(cv_out["test_roc_auc"])), 4),
            "confusion_matrix": cm_cv.tolist(),
        }
        if params_by_model is not None:
            entry["best_params"] = extra_params
        cv_results[name] = entry
        print(
            f"   -> Train CV Acc: {entry['accuracy_mean']:.4f} | "
            f"F1: {entry['f1_mean']:.4f} | F2: {entry['f2_mean']:.4f} | "
            f"AUC: {entry['roc_auc_mean']:.4f} | "
            f"Tiempo: {elapsed:.1f}s"
        )

    return cv_results


def evaluate_baseline_cv(
    specs: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Evalúa los modelos con parámetros por defecto mediante validación cruzada estratificada."""
    return _evaluate_cv_stage(
        specs,
        X_train,
        y_train,
        cv=cv,
        params_by_model=None,
        stage_label="baseline",
    )


def evaluate_optimized_cv(
    specs: dict[str, dict],
    best_params_by_model: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Evalúa los modelos con sus hiperparámetros óptimos mediante 5-fold CV sobre Train."""
    return _evaluate_cv_stage(
        specs,
        X_train,
        y_train,
        cv=cv,
        params_by_model=best_params_by_model,
        stage_label="optimizado",
    )


def compute_threshold_metrics(y_true, y_prob: np.ndarray, threshold: float) -> dict:
    """Calcula métricas de clasificación y matriz de confusión para un umbral dado sobre P(y=1)."""
    y_pred = (np.asarray(y_prob) >= float(threshold)).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
    }


def evaluate_threshold_tuning(
    specs: dict[str, dict],
    best_params_by_model: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Optimiza y prueba el umbral de decisión exclusivamente sobre 5-fold CV en Train (max F2)."""
    thresholds = [round(float(t), 2) for t in np.arange(0.10, 0.91, 0.01)]
    results: dict[str, dict] = {}

    for name, cfg in specs.items():
        kwargs = {**cfg["base_params"], **cfg["final_extra_params"], **best_params_by_model[name]}
        cv_pipe = build_training_pipeline(cfg["estimator_cls"](**kwargs))

        print(f"Optimizando umbral para Recall/F2 (5-fold CV OOF en Train): {name}...")
        oof_probs = cross_val_predict(
            cv_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]

        curve_acc, curve_prec, curve_rec, curve_f1, curve_f2 = [], [], [], [], []
        for thr in thresholds:
            m_cv = compute_threshold_metrics(y_train, oof_probs, thr)
            curve_acc.append(m_cv["accuracy"])
            curve_prec.append(m_cv["precision"])
            curve_rec.append(m_cv["recall"])
            curve_f1.append(m_cv["f1"])
            curve_f2.append(m_cv["f2"])

        best_idx = int(np.argmax(curve_f2))
        optimal_threshold = thresholds[best_idx]

        cv_at_050 = compute_threshold_metrics(y_train, oof_probs, 0.50)
        cv_at_optimal = compute_threshold_metrics(y_train, oof_probs, optimal_threshold)

        results[name] = {
            "criterion": "f2_score_cv_oof",
            "optimal_threshold": optimal_threshold,
            "cv_at_050": cv_at_050,
            "cv_at_optimal": cv_at_optimal,
            "cv_curve": {
                "thresholds": thresholds,
                "accuracy": curve_acc,
                "precision": curve_prec,
                "recall": curve_rec,
                "f1": curve_f1,
                "f2": curve_f2,
            },
        }
        print(
            f"   -> Umbral óptimo F2 (Train CV): {optimal_threshold:.2f} | "
            f"CV Recall: {cv_at_050['recall']:.4f} -> {cv_at_optimal['recall']:.4f} | "
            f"Precision: {cv_at_050['precision']:.4f} -> {cv_at_optimal['precision']:.4f} | "
            f"F2: {cv_at_050['f2']:.4f} -> {cv_at_optimal['f2']:.4f}"
        )

    return results

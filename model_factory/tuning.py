import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from model_factory.config import MODELS_DIR, RANDOM_STATE, RESULTS_FILE, TUNING_SCORING
from model_factory.evaluation import evaluate_on_test
from model_factory.preprocessing import build_preprocessor, build_training_pipeline


def tune_single_model(
    cfg: dict,
    X_train_proc: np.ndarray,
    y_train: pd.Series,
    cv,
) -> tuple[dict, float]:
    """Ejecuta GridSearchCV o RandomizedSearchCV según la especificación del modelo."""
    estimator = cfg["estimator_cls"](**cfg["base_params"])

    if cfg["search_strategy"] == "grid":
        search = GridSearchCV(
            estimator,
            param_grid=cfg["param_space"],
            cv=cv,
            scoring=TUNING_SCORING,
            n_jobs=-1,
        )
    else:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=cfg["param_space"],
            n_iter=cfg["n_iter"],
            cv=cv,
            scoring=TUNING_SCORING,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    search.fit(X_train_proc, y_train)
    return search.best_params_, float(search.best_score_)


def tune_all_models(
    specs: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Preprocesa X_train sin fuga de datos y optimiza los hiperparámetros de cada modelo."""
    search_pre = build_preprocessor()
    X_train_proc = search_pre.fit_transform(X_train)

    best_params_by_model = {}
    total = len(specs)
    for idx, (name, cfg) in enumerate(specs.items(), start=1):
        print(f"\n[{idx}/{total}] Buscando mejores hiperparámetros para {name}...")
        best_params, best_score = tune_single_model(cfg, X_train_proc, y_train, cv)
        best_params_by_model[name] = best_params
        print(f"   Mejores hiperparámetros {name}: {best_params} (CV F1={best_score:.4f})")

    return best_params_by_model


def train_and_export_final_models(
    specs: dict[str, dict],
    best_params_by_model: dict[str, dict],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models_dir: str = MODELS_DIR,
) -> dict[str, dict]:
    """Entrena los pipelines finales sobre el 100% de Train, evalúa en Test y guarda .joblib."""
    os.makedirs(models_dir, exist_ok=True)
    optimized_test_results = {}

    for name, cfg in specs.items():
        best_params = best_params_by_model[name]
        final_kwargs = {
            **cfg["base_params"],
            **cfg["final_extra_params"],
            **best_params,
        }
        pipe = build_training_pipeline(cfg["estimator_cls"](**final_kwargs))

        t0 = time.time()
        print(f"\nEntrenando modelo final '{name}' con {best_params}...")
        pipe.fit(X_train, y_train)
        fit_secs = round(time.time() - t0, 2)

        eval_metrics = evaluate_on_test(pipe, X_test, y_test)
        eval_metrics["fit_time_sec"] = fit_secs
        eval_metrics["best_params"] = best_params
        eval_metrics["model_file"] = cfg["filename"]
        optimized_test_results[name] = eval_metrics

        out_path = os.path.join(models_dir, cfg["filename"])
        joblib.dump(pipe, out_path, compress=3)
        size_mb = os.path.getsize(out_path) / (1024 * 1024)

        print(
            f"   -> Test Acc: {eval_metrics['accuracy']:.4f} | "
            f"Prec: {eval_metrics['precision']:.4f} | "
            f"Rec: {eval_metrics['recall']:.4f} | "
            f"F1: {eval_metrics['f1']:.4f} | "
            f"ROC-AUC: {eval_metrics['roc_auc']:.4f}"
        )
        print(f"   -> Guardado en '{out_path}' ({size_mb:.2f} MB) en {fit_secs}s")

    return optimized_test_results


def save_evaluation_results(payload: dict, results_path: str = RESULTS_FILE) -> None:
    """Persiste el resumen JSON de dataset, correlaciones y métricas de evaluación."""
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

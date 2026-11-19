import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

from model_factory.config import (
    F2_SCORER,
    HYPERPARAMETER_LABELS,
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_FILE,
    TUNING_SCORING,
    TUNING_SCORING_NAME,
)
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
        print(f"   Mejores hiperparámetros {name}: {best_params} (CV {TUNING_SCORING_NAME.upper()}={best_score:.4f})")

    return best_params_by_model


def compute_tuning_heatmaps(
    specs: dict[str, dict],
    best_params_by_model: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Evalúa grillas 2D de hiperparámetros en 5-fold CV (F2, F1 y ROC-AUC) para los heatmaps."""
    search_pre = build_preprocessor()
    X_train_proc = search_pre.fit_transform(X_train)

    heatmaps = {}
    for name, cfg in specs.items():
        hm_cfg = cfg.get("heatmap_config")
        if not hm_cfg:
            continue

        x_param = hm_cfg["x_param"]
        y_param = hm_cfg["y_param"]
        x_values = hm_cfg["x_values"]
        y_values = hm_cfg["y_values"]

        fixed_params = {
            k: v
            for k, v in best_params_by_model.get(name, {}).items()
            if k not in {x_param, y_param}
        }
        estimator = cfg["estimator_cls"](**{**cfg["base_params"], **fixed_params})

        print(f"Generando heatmap CV para {name} ({y_param} vs {x_param})...")
        grid_search = GridSearchCV(
            estimator,
            param_grid={y_param: y_values, x_param: x_values},
            cv=cv,
            scoring={"f2": F2_SCORER, "f1": "f1", "roc_auc": "roc_auc"},
            refit=False,
            n_jobs=-1,
        )
        grid_search.fit(X_train_proc, y_train)

        lookup = {
            (params[y_param], params[x_param]): (
                round(float(f2_val), 4),
                round(float(f1_val), 4),
                round(float(auc_val), 4),
            )
            for params, f2_val, f1_val, auc_val in zip(
                grid_search.cv_results_["params"],
                grid_search.cv_results_["mean_test_f2"],
                grid_search.cv_results_["mean_test_f1"],
                grid_search.cv_results_["mean_test_roc_auc"],
            )
        }

        scores_f2 = [[lookup[(yv, xv)][0] for xv in x_values] for yv in y_values]
        scores_f1 = [[lookup[(yv, xv)][1] for xv in x_values] for yv in y_values]
        scores_roc_auc = [[lookup[(yv, xv)][2] for xv in x_values] for yv in y_values]

        best_model_params = best_params_by_model.get(name, {})
        best_x = best_model_params.get(x_param)
        best_y = best_model_params.get(y_param)

        heatmaps[name] = {
            "x_param": x_param,
            "y_param": y_param,
            "x_label": hm_cfg["x_label"],
            "y_label": hm_cfg["y_label"],
            "x_labels": ["None" if v is None else str(v) for v in x_values],
            "y_labels": ["None" if v is None else str(v) for v in y_values],
            "fixed_params": fixed_params,
            "best_pair": {
                "x": "None" if best_x is None else str(best_x),
                "y": "None" if best_y is None else str(best_y),
            },
            "scores_f2": scores_f2,
            "scores_f1": scores_f1,
            "scores_roc_auc": scores_roc_auc,
        }

    return heatmaps


def _ordered_param_values(values: list) -> list:
    """Ordena valores numéricos de menor a mayor dejando `None` (sin límite) al final."""
    if all(v is None or isinstance(v, (int, float)) for v in values):
        return sorted(values, key=lambda v: (v is None, float(v) if v is not None else 0.0))
    return list(values)


def compute_hyperparameter_effects(
    specs: dict[str, dict],
    best_params_by_model: dict[str, dict],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv,
) -> dict[str, dict]:
    """Evalúa curvas 1D Train vs Validación (5-fold CV) para cada hiperparámetro en `param_space`."""
    scoring = {"f2": F2_SCORER, "f1": "f1", "roc_auc": "roc_auc", "accuracy": "accuracy"}
    effects: dict[str, dict] = {}

    for name, cfg in specs.items():
        base_best = best_params_by_model.get(name, {})
        eval_cache: dict[tuple, dict] = {}
        param_curves: list[dict] = []

        print(f"Calculando efecto por hiperparámetro (Train vs CV) para {name}...")
        for param_name, raw_values in cfg["param_space"].items():
            ordered_vals = _ordered_param_values(raw_values)
            if len(ordered_vals) <= 1:
                continue

            value_labels: list[str] = []
            metrics_series: dict[str, list[float]] = {
                "train_f2_mean": [],
                "train_f2_std": [],
                "val_f2_mean": [],
                "val_f2_std": [],
                "train_f1_mean": [],
                "train_f1_std": [],
                "val_f1_mean": [],
                "val_f1_std": [],
                "train_roc_auc_mean": [],
                "train_roc_auc_std": [],
                "val_roc_auc_mean": [],
                "val_roc_auc_std": [],
                "train_accuracy_mean": [],
                "val_accuracy_mean": [],
                "fit_time_mean": [],
            }

            for val in ordered_vals:
                candidate_params = {**cfg["base_params"], **base_best, param_name: val}
                cache_key = tuple(sorted((k, repr(v)) for k, v in candidate_params.items()))
                if cache_key not in eval_cache:
                    pipe = build_training_pipeline(cfg["estimator_cls"](**candidate_params))
                    cv_out = cross_validate(
                        pipe,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=scoring,
                        return_train_score=True,
                        n_jobs=-1,
                    )
                    eval_cache[cache_key] = {
                        "train_f2_mean": round(float(np.mean(cv_out["train_f2"])), 4),
                        "train_f2_std": round(float(np.std(cv_out["train_f2"])), 4),
                        "val_f2_mean": round(float(np.mean(cv_out["test_f2"])), 4),
                        "val_f2_std": round(float(np.std(cv_out["test_f2"])), 4),
                        "train_f1_mean": round(float(np.mean(cv_out["train_f1"])), 4),
                        "train_f1_std": round(float(np.std(cv_out["train_f1"])), 4),
                        "val_f1_mean": round(float(np.mean(cv_out["test_f1"])), 4),
                        "val_f1_std": round(float(np.std(cv_out["test_f1"])), 4),
                        "train_roc_auc_mean": round(float(np.mean(cv_out["train_roc_auc"])), 4),
                        "train_roc_auc_std": round(float(np.std(cv_out["train_roc_auc"])), 4),
                        "val_roc_auc_mean": round(float(np.mean(cv_out["test_roc_auc"])), 4),
                        "val_roc_auc_std": round(float(np.std(cv_out["test_roc_auc"])), 4),
                        "train_accuracy_mean": round(float(np.mean(cv_out["train_accuracy"])), 4),
                        "val_accuracy_mean": round(float(np.mean(cv_out["test_accuracy"])), 4),
                        "fit_time_mean": round(float(np.mean(cv_out["fit_time"])), 3),
                    }

                entry = eval_cache[cache_key]
                value_labels.append("None" if val is None else str(val))
                for k in metrics_series:
                    metrics_series[k].append(entry[k])

            best_val = base_best.get(param_name)
            param_curves.append(
                {
                    "param": param_name,
                    "label": HYPERPARAMETER_LABELS.get(param_name, param_name),
                    "values": value_labels,
                    "best_value": "None" if best_val is None else str(best_val),
                    **metrics_series,
                }
            )

        effects[name] = {
            "fixed_best_params": {
                k: ("None" if v is None else str(v)) for k, v in base_best.items()
            },
            "params": param_curves,
        }

    return effects


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

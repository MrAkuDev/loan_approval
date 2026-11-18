import os
import sys

# Permite ejecutar el script directamente (`python model_factory/train_model.py`)
if __package__ is None or __package__ == "":
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from model_factory.config import (
    CATEGORICAL_COLUMNS,
    CSV_FILENAME,
    FEATURE_ORDER,
    MODELS_DIR,
    NUMERICAL_COLUMNS,
    RESULTS_FILE,
    get_model_specs,
)
from model_factory.data import (
    build_cv_splitter,
    compute_train_correlations,
    load_and_split_data,
)
from model_factory.evaluation import evaluate_baseline_cv
from model_factory.tuning import (
    compute_hyperparameter_effects,
    compute_tuning_heatmaps,
    save_evaluation_results,
    train_and_export_final_models,
    tune_all_models,
)


def main() -> None:
    """Orquesta el flujo completo de entrenamiento, optimización y exportación."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    specs = get_model_specs()

    print("=" * 72)
    print("1. CARGA DE DATOS, LIMPIEZA Y PARTICIÓN TRAIN / TEST (80/20 ESTRATIFICADA)")
    print("=" * 72)
    df, X_train, X_test, y_train, y_test, cleaning_stats = load_and_split_data()
    y = df["loan_status"]
    print(
        f"Dataset bruto: {cleaning_stats['raw_rows']:,} | "
        f"Duplicados eliminados: {cleaning_stats['duplicates_removed']:,} | "
        f"Atípicos imposibles eliminados: {cleaning_stats['outliers_removed']:,}"
    )
    print(
        f"Dataset limpio: {len(df):,} registros | Train: {len(X_train):,} | Test: {len(X_test):,} | "
        f"Nulos imputados en Pipeline: {cleaning_stats['missing_values_clean']}"
    )

    print("\n" + "=" * 72)
    print("2. ETAPA 1 — EVALUACIÓN BASELINE (PARÁMETROS POR DEFECTO)")
    print("=" * 72)
    cv_5 = build_cv_splitter()
    baseline_cv_results = evaluate_baseline_cv(specs, X_train, y_train, cv=cv_5)
    baseline_test_results = {}

    print("\n" + "=" * 72)
    print("3. ETAPA 2 — BÚSQUEDA DE MEJORES HIPERPARÁMETROS")
    print("=" * 72)
    best_params_by_model = tune_all_models(specs, X_train, y_train, cv=cv_5)
    tuning_heatmaps = compute_tuning_heatmaps(specs, best_params_by_model, X_train, y_train, cv=cv_5)
    hyperparameter_effects = compute_hyperparameter_effects(
        specs, best_params_by_model, X_train, y_train, cv=cv_5
    )

    print("\n" + "=" * 72)
    print("4. ETAPA 3 — ENTRENAMIENTO FINAL (100% X_train) Y EVALUACIÓN EN TEST SET")
    print("=" * 72)
    optimized_test_results = train_and_export_final_models(
        specs=specs,
        best_params_by_model=best_params_by_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        models_dir=MODELS_DIR,
    )

    feature_correlations = compute_train_correlations(X_train, y_train)
    summary_payload = {
        "dataset": {
            "file": CSV_FILENAME,
            "source": "Kaggle — Credit Risk Dataset (laotse/credit-risk-dataset)",
            "raw_rows": cleaning_stats["raw_rows"],
            "duplicates_removed": cleaning_stats["duplicates_removed"],
            "outliers_removed": cleaning_stats["outliers_removed"],
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "missing_values_clean": cleaning_stats["missing_values_clean"],
            "class_distribution": {
                "0_no_default": int((y == 0).sum()),
                "1_default": int((y == 1).sum()),
            },
            "feature_order": FEATURE_ORDER,
            "numerical_columns": NUMERICAL_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
        "feature_correlations": feature_correlations,
        "baseline_cv": baseline_cv_results,
        "baseline_test": baseline_test_results,
        "tuning_heatmaps": tuning_heatmaps,
        "hyperparameter_effects": hyperparameter_effects,
        "optimized_test": optimized_test_results,
    }
    save_evaluation_results(summary_payload, results_path=RESULTS_FILE)

    print("\n" + "=" * 72)
    print(f"✅ Todos los modelos y resultados guardados en '{MODELS_DIR}/'")
    print("=" * 72)


if __name__ == "__main__":
    main()

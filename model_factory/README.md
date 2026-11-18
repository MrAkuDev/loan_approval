# `model_factory` — Training, Tuning & Export Pipeline

`model_factory` is the offline machine learning package responsible for loading `credit_risk_dataset.csv` (`laotse/credit-risk-dataset`), applying deterministic deduplication and physical-bound cleaning, splitting the dataset without data leakage, running baseline 5-fold cross-validation, optimizing hyperparameters for three classifiers (`SVM RBF`, `Random Forest`, and `Logistic Regression`), computing sensitivity heatmaps and feature correlations, and exporting fitted scikit-learn pipelines and evaluation metrics to `models/`.

---

## Package Structure

| File | Responsibility | Key Symbols |
| :--- | :--- | :--- |
| `__init__.py` | Public package exports for schema constants and paths. | `FEATURE_ORDER`, `NUMERICAL_COLUMNS`, `CATEGORICAL_COLUMNS`, `TARGET_COLUMN`, `CSV_PATH`, `MODELS_DIR`, `RESULTS_FILE` |
| `config.py` | Paths, global training constants, feature schema, estimators, parameter search spaces, and heatmap grids. | `TEST_SIZE`, `RANDOM_STATE`, `CV_SPLITS`, `SCORING_METRICS`, `TUNING_SCORING`, `get_model_specs()` |
| `data.py` | CSV ingestion, stratified Train/Test split, CV splitter factory, and train-only correlation calculation. | `load_and_split_data()`, `build_cv_splitter()`, `compute_train_correlations()` |
| `preprocessing.py` | Shared feature transformations and unified scikit-learn `Pipeline` builder. | `build_preprocessor()`, `build_training_pipeline()` |
| `evaluation.py` | Baseline CV evaluation, probability/decision score extraction, and held-out test metrics + ROC sampling. | `evaluate_baseline_cv()`, `evaluate_on_test()`, `compute_scores_and_probs()` |
| `tuning.py` | Hyperparameter search (`GridSearchCV` / `RandomizedSearchCV`), 2D CV heatmap evaluation, final model fitting, `.joblib` compression/export, and JSON persistence. | `tune_single_model()`, `tune_all_models()`, `compute_tuning_heatmaps()`, `train_and_export_final_models()`, `save_evaluation_results()` |
| `train_model.py` | CLI orchestrator executing Stages 1–3 end to end. | `main()` |

---

## How to Run

From the repository root:

```bash
python model_factory/train_model.py
```

Or as a Python module:

```bash
python -m model_factory.train_model
```

(`train_model.py` automatically prepends the project root to `sys.path` when executed directly as a script.)

---

## End-to-End Execution Flow (`train_model.py`)

```
credit_risk_dataset.csv
   │
   ▼  load_and_split_data()  [clean_credit_risk_dataframe() + 80% Train / 20% Test, stratified, random_state=42]
   ├──► Stage 1: evaluate_baseline_cv()
   │       └─ 5-fold StratifiedKFold CV using base_params only -> summary_payload["baseline_cv"]
   │
   ├──► Stage 2: tune_all_models() + compute_tuning_heatmaps()
   │       ├─ Fits ColumnTransformer (SimpleImputer + StandardScaler/OHE) on X_train once (no X_test leakage)
   │       ├─ Optimizes TUNING_SCORING="f1" via RandomizedSearchCV / GridSearchCV
   │       └─ Evaluates 2D CV grids (F1 & ROC-AUC) -> summary_payload["tuning_heatmaps"]
   │
   ├──► Stage 3: train_and_export_final_models()
   │       ├─ Merges base_params + final_extra_params + best_params
   │       ├─ Fits full Pipeline("preprocesamiento" -> "model") on 100% of X_train
   │       ├─ Evaluates once on X_test -> summary_payload["optimized_test"]
   │       └─ Saves models/{svm_rbf,random_forest,logistic_regression}.joblib (compress=3)
   │
   └──► compute_train_correlations() + save_evaluation_results()
           └─ Writes models/evaluation_results.json
```

### Step-by-Step Breakdown

1. **Data Loading, Cleaning & Stratified Split (`data.py`)**:
   - Reads `CSV_PATH` (`credit_risk_dataset.csv` in the project root, `32,581` raw rows × `12` columns).
   - `clean_credit_risk_dataframe()` removes `165` exact duplicate rows (preventing train/test duplicate leakage) and `7` physically impossible outlier rows (`person_age > 100` or `person_emp_length > 60`), yielding `32,409` clean rows while keeping missing values (`person_emp_length`: `887`, `loan_int_rate`: `3,094`) untouched for leak-free in-pipeline imputation.
   - Selects `FEATURE_ORDER` (`11` input columns) and `TARGET_COLUMN` (`"loan_status"`).
   - Splits data with `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
   - Instantiates a shared `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` via `build_cv_splitter()`.

2. **Stage 1 — Baseline Evaluation (`evaluation.py`)**:
   - Wraps each model instantiated with `base_params` in `build_training_pipeline()`.
   - Runs `cross_validate` over `SCORING_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]` and `cross_val_predict` (`n_jobs=-1`) on `(X_train, y_train)` to compute out-of-fold mean/std metrics and the CV confusion matrix.
   - Note: `baseline_test` in the JSON output is initialized to `{}` (baseline models are evaluated via CV on `X_train`, keeping `X_test` untouched until Stage 3).

3. **Stage 2 — Hyperparameter Tuning, 2D Heatmaps & 1D Sensitivity Curves (`tuning.py`)**:
   - `tune_all_models()` fits `build_preprocessor()` on `X_train` once to obtain `X_train_proc` so candidate evaluations do not repeatedly re-encode raw features, while `X_test` remains isolated.
   - `tune_single_model()` runs `GridSearchCV` (when `search_strategy == "grid"`) or `RandomizedSearchCV` (when `search_strategy == "random"`, with `n_iter=12` and `random_state=42`) optimizing `TUNING_SCORING = "f1"`.
   - `compute_tuning_heatmaps()` fixes non-plotted tuned parameters (`fixed_params`) to their winning values from `best_params_by_model` and runs a 2D `GridSearchCV(..., scoring={"f1": "f1", "roc_auc": "roc_auc"}, refit=False)` over `heatmap_config["y_values"]` × `heatmap_config["x_values"]`.
   - `compute_hyperparameter_effects()` varies one hyperparameter at a time across `param_space` around each model's `best_params` using `build_training_pipeline()` + `cross_validate(..., return_train_score=True)` (`f1`, `roc_auc`, `accuracy`) and stores Train vs. 5-Fold CV validation curves in `evaluation_results.json["hyperparameter_effects"]`.

4. **Stage 3 — Final Training, Test Evaluation & Export (`tuning.py`, `evaluation.py`)**:
   - Combines `{**cfg["base_params"], **cfg["final_extra_params"], **best_params}` for each model.
   - Wraps the estimator in `build_training_pipeline()` and fits on 100% of `(X_train, y_train)`.
   - Evaluates the fitted pipeline on `(X_test, y_test)` via `evaluate_on_test()`:
     - `compute_scores_and_probs()` calls `pipeline.predict_proba(X)[:, 1]`, falling back to a clipped sigmoid of `decision_function` if `predict_proba` is unavailable, or `y_pred.astype(float)` otherwise.
     - Computes `accuracy`, `precision`, `recall`, `f1`, `roc_auc` (rounded to 4 decimals), `confusion_matrix`, and downsampled `roc_curve` (`fpr`, `tpr`, capped at 120 evenly spaced indices via `np.linspace`).
   - Serializes each fitted `Pipeline` to `models/<filename>` using `joblib.dump(..., compress=3)`.

5. **Train-Only Correlations & JSON Export (`data.py`, `tuning.py`)**:
   - `compute_train_correlations()` dummy-encodes `X_train` + `y_train` via `pd.get_dummies(..., drop_first=True, dtype=float)`, computes Pearson correlations, and orders features in descending order of correlation with `loan_status`.
   - `save_evaluation_results()` writes the consolidated report to `models/evaluation_results.json`.

---

## Global Configuration Constants (`config.py`)

| Constant | Value | Description |
| :--- | :--- | :--- |
| `CSV_FILENAME` | `"credit_risk_dataset.csv"` | Source dataset filename in `PROJECT_ROOT`. |
| `MODELS_DIR` | `<PROJECT_ROOT>/models` | Output directory for `.joblib` pipelines and JSON results. |
| `RESULTS_FILE` | `<MODELS_DIR>/evaluation_results.json` | Consolidated metrics, correlations, and heatmap output file. |
| `TARGET_COLUMN` | `"loan_status"` | Binary target (`0` = Sin Incumplimiento, `1` = Incumplimiento). |
| `MAX_VALID_AGE` | `100` | Upper age threshold applied during deterministic cleaning. |
| `MAX_VALID_EMP_LENGTH` | `60` | Upper employment length threshold (years) during deterministic cleaning. |
| `TEST_SIZE` | `0.20` | 20% stratified holdout test proportion (80% Train). |
| `RANDOM_STATE` | `42` | Seed shared across `train_test_split`, `StratifiedKFold`, estimators, and `RandomizedSearchCV`. |
| `CV_SPLITS` | `5` | Number of folds in `StratifiedKFold(shuffle=True)`. |
| `SCORING_METRICS` | `["accuracy", "precision", "recall", "f1", "roc_auc"]` | Metrics computed in Stage 1 baseline CV. |
| `TUNING_SCORING` | `"f1"` | Primary optimization objective in Stage 2 search. |

---

## Feature Schema & Preprocessing (`config.py`, `preprocessing.py`)

`FEATURE_ORDER` defines the exact 11 input features expected by the pipelines:

1. `person_age` *(numerical)*
2. `person_income` *(numerical)*
3. `person_home_ownership` *(categorical: `RENT`, `OWN`, `MORTGAGE`, `OTHER`)*
4. `person_emp_length` *(numerical, has missing values)*
5. `loan_intent` *(categorical: `EDUCATION`, `MEDICAL`, `VENTURE`, `PERSONAL`, `DEBTCONSOLIDATION`, `HOMEIMPROVEMENT`)*
6. `loan_grade` *(categorical: `A`–`G`)*
7. `loan_amnt` *(numerical)*
8. `loan_int_rate` *(numerical, has missing values)*
9. `loan_percent_income` *(numerical)*
10. `cb_person_default_on_file` *(categorical: `N`, `Y`)*
11. `cb_person_cred_hist_length` *(numerical)*

`build_training_pipeline(estimator)` creates a scikit-learn `Pipeline` with two named steps:
- `"preprocesamiento"` (`build_preprocessor()`): A `ColumnTransformer(..., remainder="passthrough")` with:
  - `"num"`: `SimpleImputer(strategy="median")` + `StandardScaler()` applied to the 7 `NUMERICAL_COLUMNS`.
  - `"cat"`: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)` applied to the 4 `CATEGORICAL_COLUMNS`.
- `"model"`: The configured scikit-learn classifier instance.

---

## Model Specifications & Hyperparameter Spaces (`get_model_specs()`)

### 1. `SVM RBF` (`sklearn.svm.SVC`)
- **Artifact**: `models/svm_rbf.joblib`
- **Base Parameters (`base_params`)**: `{"kernel": "rbf", "random_state": 42}`
- **Final-Only Parameters (`final_extra_params`)**: `{"probability": True}` *(enabled only during Stage 3 final training so Platt scaling probabilities are available for inference and ROC-AUC without slowing down Stage 1/Stage 2 CV)*
- **Search Strategy**: `RandomizedSearchCV` (`n_iter=12`, `cv=5`, `scoring="f1"`, `random_state=42`)
- **Parameter Space (`param_space`)**:
  - `C`: `[0.01, 0.1, 1.0, 5.0]`
  - `gamma`: `["scale", "auto", 0.001, 0.01]`
  - `class_weight`: `[None, "balanced"]`
- **Heatmap Axes (`heatmap_config`)**: X = `C` (`"Regularización (C)"`), Y = `gamma` (`"Coeficiente Kernel (gamma)"`)

### 2. `Random Forest` (`sklearn.ensemble.RandomForestClassifier`)
- **Artifact**: `models/random_forest.joblib`
- **Base Parameters (`base_params`)**: `{"random_state": 42, "n_jobs": -1}`
- **Final-Only Parameters (`final_extra_params`)**: `{}`
- **Search Strategy**: `RandomizedSearchCV` (`n_iter=12`, `cv=5`, `scoring="f1"`, `random_state=42`)
- **Parameter Space (`param_space`)**:
  - `n_estimators`: `[100, 150, 200, 300]`
  - `max_depth`: `[None, 10, 15, 20, 30]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `max_features`: `["sqrt", "log2"]`
  - `class_weight`: `[None, "balanced"]`
- **Heatmap Axes (`heatmap_config`)**: X = `n_estimators` (`"Número de Estimadores (n_estimators)"`), Y = `max_depth` (`[10, 15, 20, 30, None]`, `"Profundidad Máxima (max_depth)"`)

### 3. `Logistic Regression` (`sklearn.linear_model.LogisticRegression`)
- **Artifact**: `models/logistic_regression.joblib`
- **Base Parameters (`base_params`)**: `{"random_state": 42, "max_iter": 1000}`
- **Final-Only Parameters (`final_extra_params`)**: `{}`
- **Search Strategy**: `GridSearchCV` (`search_strategy="grid"`, `n_iter=None`, exhaustive evaluation over 12 combinations, `cv=5`, `scoring="f1"`)
- **Parameter Space (`param_space`)**:
  - `C`: `[0.01, 0.1, 0.5, 1.0, 5.0, 10.0]`
  - `class_weight`: `[None, "balanced"]`
  - `solver`: `["lbfgs"]`
- **Heatmap Axes (`heatmap_config`)**: X = `C` (`"Inversa de Regularización (C)"`), Y = `class_weight` (`[None, "balanced"]`, `"Balanceo de Clases (class_weight)"`)

---

## Output Artifacts

Running `train_model.py` produces the following files inside `models/`:

- **`svm_rbf.joblib`**, **`random_forest.joblib`**, **`logistic_regression.joblib`**: Full scikit-learn `Pipeline` objects (`preprocesamiento` + `model`) compressed with `compress=3`. Any downstream consumer (such as the Streamlit app) can pass raw DataFrames ordered by `FEATURE_ORDER` directly into `.predict()` / `.predict_proba()`.
- **`evaluation_results.json`**: Top-level keys consumed by the dashboard and evaluation reports:
  - `dataset`: `file`, `source`, `raw_rows`, `duplicates_removed`, `outliers_removed`, `total_rows`, `train_rows`, `test_rows`, `missing_values_clean`, `class_distribution` (`0_no_default`, `1_default`), `feature_order`, `numerical_columns`, `categorical_columns`.
  - `feature_correlations`: `columns`, `matrix`, `target_correlations` (computed strictly on `X_train`).
  - `baseline_cv`: Per-model baseline CV stats (`fit_time_sec`, `accuracy_mean`, `accuracy_std`, `precision_mean`, `recall_mean`, `f1_mean`, `f1_std`, `roc_auc_mean`, `confusion_matrix`).
  - `baseline_test`: Empty dictionary `{}`.
  - `tuning_heatmaps`: Per-model 2D grid evaluation (`x_param`, `y_param`, `x_label`, `y_label`, `x_labels`, `y_labels`, `fixed_params`, `best_pair`, `scores_f1`, `scores_roc_auc`).
  - `optimized_test`: Per-model test set evaluation (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `confusion_matrix`, `roc_curve`, `fit_time_sec`, `best_params`, `model_file`).

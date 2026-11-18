"""Carga de datos, partición estratificada Train/Test y estadísticas de correlación."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from model_factory.config import (
    CSV_PATH,
    CV_SPLITS,
    FEATURE_ORDER,
    MAX_VALID_AGE,
    MAX_VALID_EMP_LENGTH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def _normalize_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte cadenas vacías o solo espacios a pd.NA sin alterar valores numéricos como 0."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or out[col].dtype == object:
            stripped = out[col].where(out[col].isna(), out[col].astype(str).str.strip())
            out[col] = stripped.replace("", pd.NA)
    return out


def clean_credit_risk_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Elimina duplicados exactos, filas con valores vacíos (cualquier columna) y atípicos imposibles."""
    raw_rows = int(len(df_raw))
    df_dedup = _normalize_empty_strings(df_raw.drop_duplicates())
    duplicates_removed = raw_rows - int(len(df_dedup))

    empty_mask = df_dedup.isna().any(axis=1)
    missing_by_column_removed = {
        col: int(cnt) for col, cnt in df_dedup.isna().sum().items() if cnt > 0
    }
    df_nonempty = df_dedup[~empty_mask].copy()
    missing_rows_removed = int(len(df_dedup)) - int(len(df_nonempty))

    valid_age = df_nonempty["person_age"] <= MAX_VALID_AGE
    valid_emp = df_nonempty["person_emp_length"] <= MAX_VALID_EMP_LENGTH
    df_clean = df_nonempty[valid_age & valid_emp].reset_index(drop=True)
    outliers_removed = int(len(df_nonempty)) - int(len(df_clean))

    stats = {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates_removed,
        "missing_rows_removed": missing_rows_removed,
        "missing_by_column_removed": missing_by_column_removed,
        "outliers_removed": outliers_removed,
        "clean_rows": int(len(df_clean)),
        "missing_values_clean": {
            col: int(cnt) for col, cnt in df_clean.isna().sum().items() if cnt > 0
        },
    }
    return df_clean, stats


def load_and_split_data(
    csv_path: str = CSV_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Carga el CSV fuente, aplica limpieza determinista y realiza una partición estratificada Train/Test."""
    df_raw = pd.read_csv(csv_path)
    df, cleaning_stats = clean_credit_risk_dataframe(df_raw)
    X = df[FEATURE_ORDER].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return df, X_train, X_test, y_train, y_test, cleaning_stats


def build_cv_splitter(
    n_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Construye el divisor StratifiedKFold reutilizado en evaluación y tuning."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def compute_train_correlations(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Calcula la matriz de correlación ordenada respecto al objetivo usando solo Train."""
    df_train = X_train.copy()
    df_train[TARGET_COLUMN] = y_train
    df_encoded = pd.get_dummies(
        df_train[FEATURE_ORDER + [TARGET_COLUMN]],
        drop_first=True,
        dtype=float,
    )
    corr_df = df_encoded.corr()
    target_corr = corr_df[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
    ordered_cols = list(target_corr.index) + [TARGET_COLUMN]
    corr_ordered = corr_df.loc[ordered_cols, ordered_cols]

    return {
        "columns": ordered_cols,
        "matrix": [[round(float(val), 4) for val in row] for row in corr_ordered.values],
        "target_correlations": {
            col: round(float(val), 4) for col, val in target_corr.items()
        },
    }

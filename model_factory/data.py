"""Carga de datos, partición estratificada Train/Test y estadísticas de correlación."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from model_factory.config import (
    CSV_PATH,
    CV_SPLITS,
    FEATURE_ORDER,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_and_split_data(
    csv_path: str = CSV_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga el CSV fuente y realiza una partición estratificada Train/Test."""
    df = pd.read_csv(csv_path)
    X = df[FEATURE_ORDER].copy()
    y = df[TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return df, X_train, X_test, y_train, y_test


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

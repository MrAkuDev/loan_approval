"""Validación de instancias individuales y archivos CSV cargados por el usuario."""

import numpy as np
import pandas as pd

from app.config import FEATURE_ORDER, NUMERICAL_COLUMNS, VALID_CATEGORIES


def validate_single_instance(data: dict) -> tuple[list[str], list[str]]:
    """Valida únicamente que todos los campos obligatorios estén presentes y tengan un tipo/categoría válido."""
    errors = []
    warnings = []

    for col in FEATURE_ORDER:
        val = data.get(col)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"El campo `{col}` es obligatorio.")
            continue

        if col in NUMERICAL_COLUMNS:
            try:
                num_val = float(val)
                if not np.isfinite(num_val):
                    errors.append(f"El campo numérico `{col}` es obligatorio y debe ser un número finito.")
            except (TypeError, ValueError):
                errors.append(f"El campo numérico `{col}` debe ser un valor numérico válido.")
        elif col in VALID_CATEGORIES:
            if str(val).strip() not in VALID_CATEGORIES[col]:
                errors.append(f"Valor inválido '{val}' para `{col}`.")

    return errors, warnings


def build_typed_instance(raw_instance: dict) -> dict:
    """Convierte los valores ya validados al tipo exacto esperado por los pipelines."""
    typed = {
        **{col: float(raw_instance[col]) for col in NUMERICAL_COLUMNS},
        **{col: str(raw_instance[col]).strip() for col in VALID_CATEGORIES},
    }
    typed["person_emp_exp"] = int(typed["person_emp_exp"])
    typed["credit_score"] = int(typed["credit_score"])
    return typed


def validate_uploaded_dataframe(df_up: pd.DataFrame) -> tuple[bool, list[str], pd.DataFrame | None]:
    """Valida estructura, nulos, tipos y categorías de una base de datos CSV cargada."""
    errors = []
    missing_cols = [c for c in FEATURE_ORDER if c not in df_up.columns]
    if missing_cols:
        errors.append(f"Faltan columnas obligatorias: {', '.join(missing_cols)}")
        return False, errors, None

    df_clean = df_up[FEATURE_ORDER].copy()

    # Verificar nulos
    null_counts = df_clean.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        details = ", ".join([f"{col} ({cnt})" for col, cnt in cols_with_nulls.items()])
        errors.append(f"Existen valores nulos en columnas obligatorias: {details}")

    # Validar tipos numéricos
    for col in NUMERICAL_COLUMNS:
        converted = pd.to_numeric(df_clean[col], errors="coerce")
        bad_rows = int(converted.isnull().sum() - df_clean[col].isnull().sum())
        if bad_rows > 0:
            errors.append(f"La columna numérica `{col}` contiene {bad_rows} valor(es) no numérico(s).")
        else:
            df_clean[col] = converted

    # Validar categorías
    for col, allowed in VALID_CATEGORIES.items():
        df_clean[col] = df_clean[col].astype(str).str.strip()
        invalid_vals = set(df_clean[col].unique()) - set(allowed)
        if invalid_vals:
            errors.append(
                f"La columna `{col}` contiene categorías no válidas: {sorted(invalid_vals)}. "
                f"Permitidas: {allowed}"
            )

    if errors:
        return False, errors, None

    return True, [], df_clean

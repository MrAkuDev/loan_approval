"""Validación de instancias individuales y archivos CSV cargados por el usuario."""

import numpy as np
import pandas as pd

from app.config import FEATURE_ORDER, NUMERICAL_COLUMNS, VALID_CATEGORIES


def validate_single_instance(data: dict) -> tuple[list[str], list[str]]:
    """Valida que los campos estén presentes, con tipo/categoría válidos y advierte sobre rangos atípicos."""
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

    if not errors:
        if float(data["person_age"]) > 100 or float(data["person_age"]) < 18:
            warnings.append("Edad fuera del rango típico de entrenamiento (18–100 años).")
        if float(data["person_emp_length"]) > 60:
            warnings.append("Antigüedad laboral superior a 60 años; posible valor atípico.")

    return errors, warnings


def build_typed_instance(raw_instance: dict) -> dict:
    """Convierte los valores ya validados al tipo exacto esperado por los pipelines."""
    typed = {
        **{col: float(raw_instance[col]) for col in NUMERICAL_COLUMNS},
        **{col: str(raw_instance[col]).strip() for col in VALID_CATEGORIES},
    }
    typed["person_age"] = int(round(typed["person_age"]))
    typed["person_income"] = int(round(typed["person_income"]))
    typed["loan_amnt"] = int(round(typed["loan_amnt"]))
    typed["cb_person_cred_hist_length"] = int(round(typed["cb_person_cred_hist_length"]))
    return typed


def validate_uploaded_dataframe(df_up: pd.DataFrame) -> tuple[bool, list[str], list[str], pd.DataFrame | None]:
    """Valida estructura, tipos y categorías de un CSV, permitiendo nulos imputables por el Pipeline."""
    errors = []
    warnings = []
    missing_cols = [c for c in FEATURE_ORDER if c not in df_up.columns]
    if missing_cols:
        errors.append(f"Faltan columnas obligatorias: {', '.join(missing_cols)}")
        return False, errors, warnings, None

    df_clean = df_up[FEATURE_ORDER].copy()

    # Informar nulos (el Pipeline incluye SimpleImputer para imputar sin fuga de datos)
    null_counts = df_clean.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        details = ", ".join([f"{col} ({cnt})" for col, cnt in cols_with_nulls.items()])
        warnings.append(
            f"Se detectaron valores faltantes en {details}; el `Pipeline` los imputará automáticamente (mediana/moda de entrenamiento)."
        )

    # Validar tipos numéricos respetando NaNs
    for col in NUMERICAL_COLUMNS:
        converted = pd.to_numeric(df_clean[col], errors="coerce")
        bad_rows = int(converted.isnull().sum() - df_clean[col].isnull().sum())
        if bad_rows > 0:
            errors.append(f"La columna numérica `{col}` contiene {bad_rows} valor(es) no numérico(s).")
        else:
            df_clean[col] = converted.astype(float)

    # Validar categorías no nulas
    for col, allowed in VALID_CATEGORIES.items():
        non_null_mask = df_clean[col].notna()
        cleaned_vals = df_clean.loc[non_null_mask, col].astype(str).str.strip()
        df_clean.loc[non_null_mask, col] = cleaned_vals
        invalid_vals = set(cleaned_vals.unique()) - set(allowed)
        if invalid_vals:
            errors.append(
                f"La columna `{col}` contiene categorías no válidas: {sorted(invalid_vals)}. "
                f"Permitidas: {allowed}"
            )

    if errors:
        return False, errors, warnings, None

    return True, [], warnings, df_clean

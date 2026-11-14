"""
app.py
======
Aplicación web interactiva en Streamlit para el Proyecto Final de Machine Learning:
Predicción de Aprobación de Préstamos (Loan Approval Classification).

Integra los 3 mejores modelos entrenados y optimizados del proyecto:
  1. SVM (Kernel RBF)
  2. Random Forest
  3. Regresión Logística
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Configuración de Página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval — Comparador de 3 Modelos ML",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR = "models"
RESULTS_PATH = os.path.join(MODELS_DIR, "evaluation_results.json")

MODEL_FILES = {
    "SVM RBF": os.path.join(MODELS_DIR, "svm_rbf.joblib"),
    "Random Forest": os.path.join(MODELS_DIR, "random_forest.joblib"),
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
}

FEATURE_ORDER = [
    "person_age",
    "person_gender",
    "person_education",
    "person_income",
    "person_emp_exp",
    "person_home_ownership",
    "loan_amnt",
    "loan_intent",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
    "previous_loan_defaults_on_file",
]

NUMERICAL_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

VALID_CATEGORIES = {
    "person_gender": ["female", "male"],
    "person_education": ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

# ---------------------------------------------------------------------------
# Estilos CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .model-card-approved {
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-top: 5px solid #059669;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .model-card-rejected {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-top: 5px solid #DC2626;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .pred-badge-ok {
        font-size: 1.25rem;
        font-weight: 700;
        color: #065F46;
        margin: 8px 0;
    }
    .pred-badge-no {
        font-size: 1.25rem;
        font-weight: 700;
        color: #991B1B;
        margin: 8px 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Carga de Modelos y Métricas Reales
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models(model_mtimes: tuple):
    loaded = {}
    missing = []
    for name, path in MODEL_FILES.items():
        if os.path.exists(path):
            loaded[name] = joblib.load(path)
        else:
            missing.append(path)
    return loaded, missing


@st.cache_data
def load_evaluation_results(file_mtime: float):
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_data
def load_cv5_eda_data(csv_mtime: float, fold_choice: str = "Todos los 5 folds (cv_5)", balanced: bool = True):
    """Carga X_train usando StratifiedKFold(n_splits=5, shuffle=True, random_state=42) y opcionalmente balancea 50/50."""
    if not os.path.exists("loan_data.csv"):
        return None
    df_all = pd.read_csv("loan_data.csv")
    X_all = df_all[FEATURE_ORDER]
    y_all = df_all["loan_status"]
    X_tr, _, y_tr, _ = train_test_split(X_all, y_all, test_size=0.20, random_state=42, stratify=y_all)

    cv_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(cv_5.split(X_tr, y_tr))

    if fold_choice.startswith("Fold "):
        fold_idx = int(fold_choice.split()[1]) - 1
        _, val_idx = splits[fold_idx]
        df_sub = X_tr.iloc[val_idx].copy()
        df_sub["loan_status"] = y_tr.iloc[val_idx].values
    else:
        # Concatena los 5 folds estratificados de validación de cv_5 (36,000 registros completos de Train)
        parts = []
        for _, val_idx in splits:
            part = X_tr.iloc[val_idx].copy()
            part["loan_status"] = y_tr.iloc[val_idx].values
            parts.append(part)
        df_sub = pd.concat(parts, axis=0)

    if balanced:
        c0 = df_sub[df_sub["loan_status"] == 0]
        c1 = df_sub[df_sub["loan_status"] == 1]
        n_min = min(len(c0), len(c1))
        df_sub = pd.concat(
            [c0.sample(n=n_min, random_state=42), c1.sample(n=n_min, random_state=42)],
            axis=0,
        ).sample(frac=1.0, random_state=42)

    return df_sub


@st.cache_data
def compute_corr_payload_from_df(df_sub: pd.DataFrame) -> dict:
    """Calcula la matriz de correlación One-Hot ordenada por relación con loan_status."""
    df_enc = pd.get_dummies(df_sub[FEATURE_ORDER + ["loan_status"]], drop_first=True, dtype=float)
    corr_df = df_enc.corr()
    target_corr = corr_df["loan_status"].drop("loan_status").sort_values(ascending=False)
    ordered_cols = list(target_corr.index) + ["loan_status"]
    corr_ordered = corr_df.loc[ordered_cols, ordered_cols]
    return {
        "columns": ordered_cols,
        "matrix": [[round(float(val), 4) for val in row] for row in corr_ordered.values],
        "target_correlations": {col: round(float(val), 4) for col, val in target_corr.items()},
    }


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


def predict_with_model(pipeline, X_df: pd.DataFrame) -> dict:
    """Genera predicción real, probabilidad de clase 1 (y 0) y decision_function si aplica."""
    pred_class = int(pipeline.predict(X_df)[0])

    prob_0, prob_1 = None, None
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(X_df)[0]
        prob_0 = float(probs[0])
        prob_1 = float(probs[1])

    dec_score = None
    if hasattr(pipeline, "decision_function"):
        dec_score = float(pipeline.decision_function(X_df)[0])

    return {
        "pred": pred_class,
        "label": "Aprobado (1)" if pred_class == 1 else "No Aprobado (0)",
        "prob_0": prob_0,
        "prob_1": prob_1,
        "decision_score": dec_score,
    }


model_mtimes = tuple(
    (name, os.path.getmtime(path) if os.path.exists(path) else 0.0)
    for name, path in MODEL_FILES.items()
)
eval_mtime = os.path.getmtime(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else 0.0

models, missing_files = load_models(model_mtimes)
eval_results = load_evaluation_results(eval_mtime)

# ---------------------------------------------------------------------------
# Sidebar — Información Académica y Navegación
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=68)
    st.title("Loan Approval ML")
    st.caption("Proyecto Final — Aplicación Interactiva en Streamlit")

    st.divider()
    st.subheader("🎓 Información del Proyecto")
    st.markdown(
        """
    - **Proyecto:** Clasificación de Aprobación de Préstamos
    - **Alumno:** Joel Arturo Álvarez Hernández
    - **Matrícula:** 403233
    - **Institución:** Universidad Autónoma de Chihuahua (UACH)
    - **Facultad:** Facultad de Ingeniería
    - **Programa / Materia:** Maestría — Machine Learning
    - **Asesora:** Dra. Olanda Prieto Ordaz
    """
    )

    st.divider()
    st.subheader("🤖 Modelos Integrados")
    st.markdown(
        """
    1. **SVM (Kernel RBF)**
    2. **Random Forest**
    3. **Regresión Logística**
    """
    )
    st.caption("Preprocesamiento unificado vía `Pipeline` + `ColumnTransformer` guardado con `joblib`.")


# ---------------------------------------------------------------------------
# Encabezado Principal
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🏦 Predicción de Aprobación de Préstamos Bancarios</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Comparación en tiempo real de los tres mejores modelos de Machine Learning del proyecto final: "
    "<b>SVM (RBF)</b>, <b>Random Forest</b> y <b>Regresión Logística</b>."
    "</div>",
    unsafe_allow_html=True,
)

if missing_files or eval_results is None:
    st.error(
        "⚠️ Faltan archivos de modelos entrenados o resultados de evaluación. "
        "Ejecuta `python model_factory/train_model.py` primero para entrenar y guardar los tres modelos en `models/`."
    )
    if missing_files:
        st.code("\n".join(missing_files))
    st.stop()

# ---------------------------------------------------------------------------
# Pestañas Principales
# ---------------------------------------------------------------------------
tab_info, tab_predict, tab_upload, tab_eval, tab_interp = st.tabs(
    [
        "ℹ️ Información del Proyecto",
        "🔮 Predicción Individual",
        "📂 Upload Database",
        "📊 Evaluación Experimental",
        "📘 Interpretación y Limitaciones",
    ]
)

# ===========================================================================
# TAB 1: INFORMACIÓN GENERAL DEL PROYECTO
# ===========================================================================
with tab_info:
    st.header("Información General del Proyecto")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("🎯 Problema y Objetivo")
        st.markdown(
            """
        - **Descripción del problema:** Las instituciones financieras evalúan múltiples variables
          demográficas, financieras e historiales crediticios para decidir si una solicitud de
          préstamo debe aprobarse (`1`) o rechazarse (`0`). Se trata de un problema de
          **clasificación binaria supervisada** con desbalance de clases (77.78% vs 22.22%).
        - **Objetivo de la aplicación:** Integrar en una interfaz web los **tres modelos con
          mejor desempeño** evaluados en `loan_approval.ipynb`, permitiendo ingresar instancias
          individuales o lotes en CSV para generar, comparar e interpretar sus predicciones en vivo.
        """
        )

        st.subheader("📋 Ficha Académica")
        st.markdown(
            """
        | Campo | Detalle |
        | :--- | :--- |
        | **Proyecto** | Predicción de Aprobación de Préstamos (*Loan Approval*) |
        | **Alumno / Matrícula** | Joel Arturo Álvarez Hernández (`403233`) |
        | **Institución** | Universidad Autónoma de Chihuahua (UACH) |
        | **Facultad / Materia** | Facultad de Ingeniería — Machine Learning (Proyecto Final) |
        | **Asesora / Docente** | Dra. Olanda Prieto Ordaz |
        """
        )

    with col_p2:
        ds_info = eval_results["dataset"]
        st.subheader("🗄️ Base de Datos Utilizada (`loan_data.csv`)")
        st.markdown(
            f"""
        - **Origen:** Kaggle — *Loan Approval Classification Dataset* (`taweilo`).
        - **Total de registros:** **{ds_info['total_rows']:,}** filas × **14** columnas (0 nulos, 0 duplicados).
        - **Partición utilizada:** 80% Entrenamiento (**{ds_info['train_rows']:,}**) / 20% Prueba (**{ds_info['test_rows']:,}**) estratificada (`random_state=42`).
        - **Distribución de la variable objetivo (`loan_status`):**
          - Clase `0` (**No Aprobado**): **{ds_info['class_distribution']['0_no_aprobado']:,}** registros (77.78%)
          - Clase `1` (**Aprobado**): **{ds_info['class_distribution']['1_aprobado']:,}** registros (22.22%)
        - **Preprocesamiento idéntico en los 3 pipelines:**
          - `StandardScaler` sobre las **8 variables numéricas**.
          - `OneHotEncoder(drop='first', handle_unknown='ignore')` sobre las **5 variables categóricas**.
        """
        )

    st.divider()
    st.subheader("🧠 Descripción de los Tres Modelos Implementados")
    m1, m2, m3 = st.columns(3)

    with m1:
        opt_rbf = eval_results["optimized_test"]["SVM RBF"]
        st.info(
            f"""
        **1. SVM — Kernel RBF**
        - **Tipo:** Máquina de Vectores de Soporte con función de base radial (gaussiana).
        - **Por qué se seleccionó:** Fue el modelo con mayor desempeño global en validación cruzada (`F1 = {eval_results['baseline_cv']['SVM RBF']['f1_mean']:.4f}`, `ROC-AUC = {eval_results['baseline_cv']['SVM RBF']['roc_auc_mean']:.4f}`).
        - **Hiperparámetros óptimos aplicados:** `{opt_rbf['best_params']}`
        """
        )

    with m2:
        opt_rf = eval_results["optimized_test"]["Random Forest"]
        st.info(
            f"""
        **2. Random Forest**
        - **Tipo:** Ensamble de árboles de decisión (`bagging`) capaz de capturar relaciones no lineales e interacciones complejas entre variables financieras.
        - **Desempeño en CV (parámetros base):** `F1 = {eval_results['baseline_cv']['Random Forest']['f1_mean']:.4f}`, `ROC-AUC = {eval_results['baseline_cv']['Random Forest']['roc_auc_mean']:.4f}`.
        - **Hiperparámetros óptimos aplicados:** `{opt_rf['best_params']}`
        """
        )

    with m3:
        opt_lr = eval_results["optimized_test"]["Logistic Regression"]
        st.info(
            f"""
        **3. Regresión Logística**
        - **Tipo:** Clasificador lineal probabilístico con regularización L2.
        - **Por qué se seleccionó:** Mejor modelo lineal del proyecto, alta interpretabilidad, probabilidades calibradas nativas y `ROC-AUC = {eval_results['baseline_cv']['Logistic Regression']['roc_auc_mean']:.4f}`.
        - **Hiperparámetros óptimos aplicados:** `{opt_lr['best_params']}`
        """
        )

# ===========================================================================
# TAB 2: PREDICCIÓN INDIVIDUAL (COMPARACIÓN DE LOS 3 MODELOS)
# ===========================================================================
with tab_predict:
    st.header("Ingresar Datos del Solicitante para Predicción")
    st.markdown(
        "Complete los 13 atributos de entrada (en el mismo orden y formato de `loan_data.csv`). "
        "Los tres modelos entrenados evaluarán simultáneamente la instancia."
    )

    with st.form("single_prediction_form"):
        st.subheader("1. Datos Personales del Solicitante")
        c1, c2, c3 = st.columns(3)
        with c1:
            person_age = st.number_input("Edad (`person_age`)", value=25.0, step=1.0)
            person_gender = st.selectbox("Género (`person_gender`)", options=VALID_CATEGORIES["person_gender"], index=0)
        with c2:
            person_education = st.selectbox(
                "Escolaridad (`person_education`)",
                options=VALID_CATEGORIES["person_education"],
                index=2,
            )
            person_home_ownership = st.selectbox(
                "Tipo de Vivienda (`person_home_ownership`)",
                options=VALID_CATEGORIES["person_home_ownership"],
                index=0,
            )
        with c3:
            person_income = st.number_input(
                "Ingreso Anual USD (`person_income`)",
                value=65000.0,
                step=500.0,
                format="%.2f",
            )
            person_emp_exp = st.number_input(
                "Experiencia Laboral en años (`person_emp_exp`)",
                value=3,
                step=1,
            )

        st.subheader("2. Características del Préstamo")
        c4, c5, c6 = st.columns(3)
        with c4:
            loan_amnt = st.number_input(
                "Monto Solicitado USD (`loan_amnt`)",
                value=12000.0,
                step=500.0,
                format="%.2f",
            )
        with c5:
            loan_intent = st.selectbox(
                "Propósito del Préstamo (`loan_intent`)",
                options=VALID_CATEGORIES["loan_intent"],
                index=0,
            )
        with c6:
            loan_int_rate = st.number_input(
                "Tasa de Interés % (`loan_int_rate`)",
                value=11.5,
                step=0.1,
                format="%.2f",
            )

        auto_ratio = (
            round(loan_amnt / person_income, 2)
            if (loan_amnt is not None and person_income is not None and person_income != 0)
            else 0.0
        )
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            use_auto_ratio = st.checkbox(
                "Calcular `loan_percent_income` automáticamente (`loan_amnt / person_income`)",
                value=True,
            )
        with col_r2:
            manual_ratio = st.number_input(
                "Proporción Préstamo / Ingreso (`loan_percent_income`)",
                value=float(auto_ratio),
                step=0.01,
                format="%.2f",
                disabled=use_auto_ratio,
            )
        loan_percent_income = auto_ratio if use_auto_ratio else manual_ratio

        st.subheader("3. Historial Crediticio")
        c7, c8, c9 = st.columns(3)
        with c7:
            credit_score = st.number_input("Puntaje Crediticio (`credit_score`)", value=640, step=1)
        with c8:
            cb_person_cred_hist_length = st.number_input(
                "Antigüedad del Historial Crediticio en años (`cb_person_cred_hist_length`)",
                value=4.0,
                step=1.0,
            )
        with c9:
            previous_loan_defaults_on_file = st.selectbox(
                "¿Incumplimientos Previos? (`previous_loan_defaults_on_file`)",
                options=VALID_CATEGORIES["previous_loan_defaults_on_file"],
                index=0,
            )

        submit_btn = st.form_submit_button("🔮 Generar Predicciones con los 3 Modelos", use_container_width=True)

    if submit_btn:
        raw_instance = {
            "person_age": person_age,
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "person_home_ownership": person_home_ownership,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_loan_defaults_on_file,
        }

        val_errors, val_warnings = validate_single_instance(raw_instance)
        if val_errors:
            for err in val_errors:
                st.error(f"❌ Error de validación: {err}")
        else:
            for warn in val_warnings:
                st.warning(f"⚠️ Advertencia de validación: {warn}")

            instance_dict = {
                **{col: float(raw_instance[col]) for col in NUMERICAL_COLUMNS},
                **{col: str(raw_instance[col]).strip() for col in VALID_CATEGORIES},
            }
            instance_dict["person_emp_exp"] = int(instance_dict["person_emp_exp"])
            instance_dict["credit_score"] = int(instance_dict["credit_score"])

            # Construir DataFrame con el orden exacto de características del entrenamiento
            X_input = pd.DataFrame([instance_dict])[FEATURE_ORDER]

            preds = {name: predict_with_model(pipe, X_input) for name, pipe in models.items()}

            st.markdown("---")
            st.subheader("🎯 Resultados de Predicción por Modelo")

            m_cols = st.columns(3)
            for idx, (m_name, res) in enumerate(preds.items()):
                with m_cols[idx]:
                    card_cls = "model-card-approved" if res["pred"] == 1 else "model-card-rejected"
                    badge_cls = "pred-badge-ok" if res["pred"] == 1 else "pred-badge-no"
                    icon = "✅" if res["pred"] == 1 else "❌"
                    prob_1_pct = f"{res['prob_1'] * 100:.2f}%" if res["prob_1"] is not None else "N/D"
                    prob_0_pct = f"{res['prob_0'] * 100:.2f}%" if res["prob_0"] is not None else "N/D"
                    dec_txt = f"{res['decision_score']:+.4f}" if res["decision_score"] is not None else "N/A"

                    st.markdown(
                        f"""
                        <div class="{card_cls}">
                            <div style="font-weight:700; font-size:1.1rem; color:#1F2937;">{m_name}</div>
                            <div class="{badge_cls}">{icon} {res['label']}</div>
                            <div style="font-size:0.95rem; color:#374151;">
                                <b>Prob. Aprobado (1):</b> {prob_1_pct}<br>
                                <b>Prob. No Aprobado (0):</b> {prob_0_pct}<br>
                                <b>Score de Decisión:</b> <code>{dec_txt}</code>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Comparación de diferencias entre modelos
            st.markdown("### 🔍 Comparación y Diferencias entre Modelos")
            classes_set = {res["pred"] for res in preds.values()}
            approved_models = [k for k, v in preds.items() if v["pred"] == 1]
            rejected_models = [k for k, v in preds.items() if v["pred"] == 0]

            if len(classes_set) == 1:
                consensus_lbl = "Aprobado (1)" if list(classes_set)[0] == 1 else "No Aprobado (0)"
                st.success(
                    f"🤝 **Consenso Unánime (3 de 3 modelos):** Todos los modelos predicen **{consensus_lbl}**."
                )
            else:
                st.warning(
                    f"⚖️ **Discrepancia entre modelos:** "
                    f"**Aprobado (1):** {', '.join(approved_models)} | "
                    f"**No Aprobado (0):** {', '.join(rejected_models)}. "
                    "Esto ocurre cuando la instancia se encuentra cerca de la frontera de decisión y "
                    "la frontera lineal (Regresión Logística), el ensamble de árboles (Random Forest) y la frontera gaussiana (SVM RBF) "
                    "ponderan distinto las interacciones entre ingreso, tasa de interés y porcentaje comprometido."
                )

            comp_rows = []
            for m_name, res in preds.items():
                comp_rows.append(
                    {
                        "Modelo": m_name,
                        "Clase Predicha": res["pred"],
                        "Etiqueta": res["label"],
                        "Prob. Aprobado (Clase 1)": round(res["prob_1"], 4) if res["prob_1"] is not None else None,
                        "Prob. No Aprobado (Clase 0)": round(res["prob_0"], 4) if res["prob_0"] is not None else None,
                        "Decision Function": round(res["decision_score"], 4) if res["decision_score"] is not None else None,
                    }
                )
            df_comp = pd.DataFrame(comp_rows)

            c_tbl, c_fig = st.columns([1.2, 1])
            with c_tbl:
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
            with c_fig:
                fig, ax = plt.subplots(figsize=(6, 3.2))
                probs_plot = [r["Prob. Aprobado (Clase 1)"] * 100 for r in comp_rows]
                colors = ["#059669" if r["Clase Predicha"] == 1 else "#DC2626" for r in comp_rows]
                bars = ax.barh([r["Modelo"] for r in comp_rows], probs_plot, color=colors)
                ax.axvline(50, color="#374151", linestyle="--", linewidth=1, label="Umbral 50%")
                ax.set_xlim(0, 105)
                ax.set_xlabel("Probabilidad de Aprobación — Clase 1 (%)")
                ax.set_title("Confianza por Modelo")
                for bar, val in zip(bars, probs_plot):
                    ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with st.expander("📋 Ver vector de características exacto enviado a los pipelines"):
                st.dataframe(X_input, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 3: UPLOAD DATABASE (CARGA DE ARCHIVO CSV)
# ===========================================================================
with tab_upload:
    st.header("📂 Upload Database (Predicción por Lotes en CSV)")
    st.markdown(
        "Sube un archivo `.csv` con las 13 columnas de entrada en formato compatible con `loan_data.csv` "
        "(puede incluir opcionalmente la columna real `loan_status` para comparar aciertos). "
        "Los 3 modelos procesarán el archivo aplicando sus pipelines de preprocesamiento."
    )

    # Plantilla descargable
    sample_template = pd.read_csv("loan_data.csv", nrows=5) if os.path.exists("loan_data.csv") else None
    if sample_template is not None:
        st.download_button(
            label="⬇️ Descargar CSV de ejemplo (5 registros reales de loan_data.csv)",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="ejemplo_upload_database.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Upload Database (.csv)", type=["csv"], key="upload_database_csv")

    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.write(f"Archivo cargado: **{uploaded_file.name}** ({len(df_raw):,} filas × {df_raw.shape[1]} columnas)")

            is_valid, val_errs, df_features = validate_uploaded_dataframe(df_raw)
            if not is_valid:
                for e in val_errs:
                    st.error(f"❌ Validación fallida: {e}")
            else:
                st.success("✅ Validación exitosa: columnas, tipos de datos y categorías verificadas.")

                df_out = df_raw.copy()
                for m_name, pipe in models.items():
                    preds_arr = pipe.predict(df_features)
                    probs_arr = pipe.predict_proba(df_features)[:, 1]
                    short_key = m_name.lower().replace(" ", "_")
                    df_out[f"pred_{short_key}"] = preds_arr
                    df_out[f"prob1_{short_key}"] = np.round(probs_arr, 4)

                pred_cols = [f"pred_{k.lower().replace(' ', '_')}" for k in models.keys()]
                df_out["consenso_3_modelos"] = df_out[pred_cols].nunique(axis=1).map(
                    lambda n: "Unánime (3/3)" if n == 1 else "Discrepancia (2/1)"
                )

                # Resumen estadístico del lote
                st.subheader("📈 Resumen de Predicciones sobre la Base Cargada")
                u_cols = st.columns(4)
                for idx, (m_name, col_name) in enumerate(zip(models.keys(), pred_cols)):
                    n_app = int((df_out[col_name] == 1).sum())
                    pct_app = (n_app / len(df_out)) * 100
                    u_cols[idx].metric(f"{m_name} — Aprobados", f"{n_app:,} / {len(df_out):,}", f"{pct_app:.1f}%")
                unanimous_cnt = int((df_out["consenso_3_modelos"] == "Unánime (3/3)").sum())
                u_cols[3].metric(
                    "Consenso Unánime",
                    f"{unanimous_cnt:,} / {len(df_out):,}",
                    f"{(unanimous_cnt / len(df_out)) * 100:.1f}%",
                )

                # Si la base incluye la columna real loan_status, mostrar comparación real
                if "loan_status" in df_raw.columns and df_raw["loan_status"].isin([0, 1]).all():
                    from sklearn.metrics import accuracy_score, f1_score

                    st.info("ℹ️ La base subida contiene la columna real `loan_status`. Desempeño de los 3 modelos en esta muestra:")
                    batch_metrics = []
                    for m_name, col_name in zip(models.keys(), pred_cols):
                        batch_metrics.append(
                            {
                                "Modelo": m_name,
                                "Accuracy en CSV": round(float(accuracy_score(df_raw["loan_status"], df_out[col_name])), 4),
                                "F1-Score en CSV": round(float(f1_score(df_raw["loan_status"], df_out[col_name], zero_division=0)), 4),
                            }
                        )
                    st.dataframe(pd.DataFrame(batch_metrics), use_container_width=True, hide_index=True)

                st.subheader("📋 Vista Previa de Resultados (primeras 100 filas)")
                st.dataframe(df_out.head(100), use_container_width=True)

                st.download_button(
                    label="📥 Descargar CSV con Predicciones de los 3 Modelos",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name="predicciones_3_modelos.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Error al procesar el archivo CSV: {exc}")

def plot_confusion_matrices_row(results_by_model: dict, model_names: list, cmap: str, subtitle_map: dict):
    """Dibuja una fila de matrices de confusión 2x2 para los modelos indicados."""
    fig_cm, axes_cm = plt.subplots(1, len(model_names), figsize=(15, 4.3))
    labels_cm = ["No Aprobado (0)", "Aprobado (1)"]

    for idx, m_name in enumerate(model_names):
        cm = np.array(results_by_model[m_name]["confusion_matrix"])
        ax = axes_cm[idx]
        ax.imshow(cm, cmap=cmap)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_cm, fontsize=9)
        ax.set_yticklabels(labels_cm, fontsize=9)
        ax.set_xlabel("Predicho")
        if idx == 0:
            ax.set_ylabel("Real")
        ax.set_title(f"{m_name}\n{subtitle_map[m_name]}", fontsize=10.5, fontweight="bold")
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax.text(
                    c, r, f"{cm[r, c]:,}",
                    ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontsize=11, fontweight="bold",
                )
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)


def plot_grouped_status_pct(df_eda: pd.DataFrame, col: str, title: str, xlabel: str):
    """Dibuja gráfica de barras agrupadas (% Denegado 0 vs Aprobado 1) para una variable categórica."""
    ct = pd.crosstab(df_eda[col], df_eda["loan_status"], normalize="index") * 100.0
    for cls in (0, 1):
        if cls not in ct.columns:
            ct[cls] = 0.0
    ct = ct[[0, 1]]

    cats = list(ct.index)
    x = np.arange(len(cats))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars_0 = ax.bar(x - width / 2, ct[0].values, width, label="Denegado (0)", color="#DC2626")
    bars_1 = ax.bar(x + width / 2, ct[1].values, width, label="Aprobado (1)", color="#059669")

    for bars in (bars_0, bars_1):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20 if len(cats) > 4 else 0, ha="right" if len(cats) > 4 else "center", fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Porcentaje de Solicitudes (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="loan_status", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_status_boxplot(df_eda: pd.DataFrame, col: str, title: str, ylabel: str):
    """Dibuja boxplot comparando Denegado (0) vs Aprobado (1) para una variable numérica."""
    vals_0 = df_eda.loc[df_eda["loan_status"] == 0, col].values
    vals_1 = df_eda.loc[df_eda["loan_status"] == 1, col].values

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bp = ax.boxplot(
        [vals_0, vals_1],
        tick_labels=["Denegado (0)", "Aprobado (1)"],
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "#111827", "linewidth": 1.8},
    )
    for patch, color in zip(bp["boxes"], ["#FCA5A5", "#6EE7B7"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xlabel("Estado del Préstamo (loan_status)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_interactive_3d_scatter(df_3d: pd.DataFrame, inc_cap: float):
    """Renderiza una gráfica 3D interactiva (HTML5 Canvas) con líneas verticales de altura y control por cursor."""
    sample = df_3d.sample(n=min(1600, len(df_3d)), random_state=42)
    z_min = float(sample["loan_int_rate"].min())
    z_max = float(sample["loan_int_rate"].max())
    y_min = 0.0
    y_max = float(max(sample["loan_amnt"].max(), 1.0))

    points = [
        {
            "x": round(float(r.person_income), 0),
            "y": round(float(r.loan_amnt), 0),
            "z": round(float(r.loan_int_rate), 2),
            "c": int(r.loan_status),
        }
        for r in sample.itertuples(index=False)
    ]

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
      * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
      body {{ margin: 0; padding: 0; background: transparent; }}
      .wrap {{
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        background: #FFFFFF;
        padding: 12px 14px 10px 14px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
      }}
      .toolbar {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 8px;
      }}
      .btn-group {{ display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }}
      .btn {{
        border: 1px solid #D1D5DB;
        background: #F9FAFB;
        color: #1F2937;
        padding: 5px 10px;
        border-radius: 7px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
      }}
      .btn:hover {{ background: #EFF6FF; border-color: #93C5FD; }}
      .btn.active {{ background: #1E3A8A; color: #FFFFFF; border-color: #1E3A8A; }}
      .legend {{ display: flex; gap: 12px; align-items: center; font-size: 12px; color: #374151; }}
      .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
      .canvas-box {{ position: relative; width: 100%; height: 520px; cursor: grab; }}
      .canvas-box:active {{ cursor: grabbing; }}
      canvas {{ width: 100%; height: 100%; display: block; border-radius: 8px; background: #FFFFFF; }}
      .tooltip {{
        position: absolute;
        pointer-events: none;
        display: none;
        background: rgba(15, 23, 42, 0.92);
        color: #F8FAFC;
        padding: 7px 10px;
        border-radius: 7px;
        font-size: 12px;
        line-height: 1.45;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 10;
      }}
      .hint {{ font-size: 11.5px; color: #6B7280; margin-top: 6px; text-align: center; }}
    </style>
    </head>
    <body>
      <div class="wrap">
        <div class="toolbar">
          <div class="btn-group">
            <button id="mode-cursor" class="btn active">🖱️ Rotar con Cursor</button>
            <button id="mode-drag" class="btn">✋ Arrastrar Libre</button>
            <button id="mode-auto" class="btn">🔄 Auto-Animación</button>
            <button id="btn-stems" class="btn active">📏 Líneas Verticales</button>
            <button id="btn-reset" class="btn">↺ Reset</button>
          </div>
          <div class="btn-group">
            <button id="cnt-400" class="btn active">400 pts</button>
            <button id="cnt-800" class="btn">800 pts</button>
            <button id="cnt-1600" class="btn">1,600 pts</button>
          </div>
          <div class="btn-group">
            <button id="flt-all" class="btn active">Todos</button>
            <button id="flt-1" class="btn">Aprobado (1)</button>
            <button id="flt-0" class="btn">Denegado (0)</button>
          </div>
          <div class="legend">
            <span><span class="dot" style="background:#DC2626;"></span>Denegado (0)</span>
            <span><span class="dot" style="background:#059669;"></span>Aprobado (1)</span>
          </div>
        </div>
        <div class="canvas-box" id="box">
          <canvas id="c3d"></canvas>
          <div class="tooltip" id="tip"></div>
        </div>
        <div class="hint">
          Cada línea vertical conecta el punto con el piso para resaltar la altura de <b>loan_int_rate (%)</b>. Mueve el cursor para rotar, arrastra para girar 360° y usa la rueda para zoom.
        </div>
      </div>

      <script>
        const POINTS = {json.dumps(points)};
        const X_MAX = {round(inc_cap, 2)};
        const Y_MIN = {y_min}, Y_MAX = {y_max};
        const Z_MIN = {z_min}, Z_MAX = {z_max};

        const box = document.getElementById("box");
        const canvas = document.getElementById("c3d");
        const ctx = canvas.getContext("2d");
        const tip = document.getElementById("tip");

        let W = 900, H = 520, dpr = window.devicePixelRatio || 1;
        function resize() {{
          const rect = box.getBoundingClientRect();
          W = Math.max(320, rect.width);
          H = rect.height;
          canvas.width = W * dpr;
          canvas.height = H * dpr;
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        }}
        resize();
        window.addEventListener("resize", resize);

        let mode = "cursor"; // cursor | drag | auto
        let classFilter = "all"; // all | 0 | 1
        let showStems = true;
        let maxPoints = 400;
        let baseYaw = -0.58, basePitch = 0.34;
        let yaw = baseYaw, pitch = basePitch;
        let targetYaw = baseYaw, targetPitch = basePitch;
        let zoom = 0.98, targetZoom = 0.98;
        let dragging = false, lastX = 0, lastY = 0;
        let mouseX = -9999, mouseY = -9999;

        function setMode(m) {{
          mode = m;
          ["cursor", "drag", "auto"].forEach(k => {{
            document.getElementById("mode-" + k).classList.toggle("active", k === m);
          }});
          if (m === "drag") {{ targetYaw = yaw; targetPitch = pitch; }}
        }}
        function setFilter(f) {{
          classFilter = f;
          ["all", "1", "0"].forEach(k => {{
            document.getElementById("flt-" + k).classList.toggle("active", k === f);
          }});
        }}
        function setCount(n) {{
          maxPoints = n;
          [400, 800, 1600].forEach(k => {{
            document.getElementById("cnt-" + k).classList.toggle("active", k === n);
          }});
        }}

        document.getElementById("mode-cursor").onclick = () => setMode("cursor");
        document.getElementById("mode-drag").onclick = () => setMode("drag");
        document.getElementById("mode-auto").onclick = () => setMode("auto");
        document.getElementById("btn-stems").onclick = () => {{
          showStems = !showStems;
          document.getElementById("btn-stems").classList.toggle("active", showStems);
        }};
        document.getElementById("btn-reset").onclick = () => {{
          baseYaw = -0.58; basePitch = 0.34;
          targetYaw = baseYaw; targetPitch = basePitch;
          targetZoom = 0.98;
        }};
        document.getElementById("cnt-400").onclick = () => setCount(400);
        document.getElementById("cnt-800").onclick = () => setCount(800);
        document.getElementById("cnt-1600").onclick = () => setCount(1600);
        document.getElementById("flt-all").onclick = () => setFilter("all");
        document.getElementById("flt-1").onclick = () => setFilter("1");
        document.getElementById("flt-0").onclick = () => setFilter("0");

        canvas.addEventListener("mousedown", (e) => {{
          dragging = true;
          lastX = e.clientX;
          lastY = e.clientY;
        }});
        window.addEventListener("mouseup", () => {{ dragging = false; }});
        window.addEventListener("mousemove", (e) => {{
          const rect = canvas.getBoundingClientRect();
          mouseX = e.clientX - rect.left;
          mouseY = e.clientY - rect.top;
          if (dragging) {{
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            lastX = e.clientX; lastY = e.clientY;
            baseYaw += dx * 0.008;
            basePitch = Math.max(-1.35, Math.min(1.35, basePitch + dy * 0.008));
            targetYaw = baseYaw;
            targetPitch = basePitch;
            return;
          }}
          if (mode === "cursor" && mouseX >= 0 && mouseX <= rect.width && mouseY >= 0 && mouseY <= rect.height) {{
            const nx = (mouseX / rect.width - 0.5) * 2;
            const ny = (mouseY / rect.height - 0.5) * 2;
            targetYaw = baseYaw + nx * 0.95;
            targetPitch = Math.max(-1.35, Math.min(1.35, basePitch + ny * 0.65));
          }}
        }});
        canvas.addEventListener("mouseleave", () => {{
          mouseX = -9999; mouseY = -9999; tip.style.display = "none";
          if (mode === "cursor" && !dragging) {{
            targetYaw = baseYaw; targetPitch = basePitch;
          }}
        }});
        canvas.addEventListener("wheel", (e) => {{
          e.preventDefault();
          targetZoom = Math.max(0.55, Math.min(2.6, targetZoom * (e.deltaY < 0 ? 1.09 : 0.92)));
        }}, {{ passive: false }});

        function project(nx, ny, nz) {{
          const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
          const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
          const x1 = nx * cosY - ny * sinY;
          const y1 = nx * sinY + ny * cosY;
          const z1 = nz;
          const y2 = y1 * cosP - z1 * sinP;
          const z2 = y1 * sinP + z1 * cosP;
          const persp = 3.6 / (3.6 - y2);
          const scale = zoom * (Math.min(W, H) * 0.33) * persp;
          return {{ sx: W * 0.5 + x1 * scale, sy: H * 0.51 - z2 * scale, depth: y2, persp }};
        }}

        const NORM_PTS = POINTS.map(p => ({{
          nx: (p.x / X_MAX) * 2 - 1,
          ny: ((p.y - Y_MIN) / (Y_MAX - Y_MIN)) * 2 - 1,
          nz: ((p.z - Z_MIN) / Math.max(Z_MAX - Z_MIN, 0.01)) * 2 - 1,
          raw: p
        }}));

        const CUBE_VERTS = [
          [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
          [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
        ];
        const CUBE_EDGES = [
          [0,1],[1,2],[2,3],[3,0],
          [4,5],[5,6],[6,7],[7,4],
          [0,4],[1,5],[2,6],[3,7]
        ];

        function drawTickLabel(px, py, txt, dx, dy, align) {{
          ctx.fillStyle = "#334155";
          ctx.font = "500 10.5px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.textAlign = align || "center";
          ctx.fillText(txt, px + dx, py + dy);
        }}

        function draw() {{
          if (mode === "auto" && !dragging) {{
            baseYaw += 0.0055;
            targetYaw = baseYaw;
          }}
          yaw += (targetYaw - yaw) * 0.10;
          pitch += (targetPitch - pitch) * 0.10;
          zoom += (targetZoom - zoom) * 0.14;

          ctx.clearRect(0, 0, W, H);

          // Floor grid (z = -1)
          ctx.strokeStyle = "#CBD5E1";
          ctx.lineWidth = 0.75;
          for (let t = -1; t <= 1.001; t += 0.5) {{
            const a = project(t, -1, -1), b = project(t, 1, -1);
            ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
            const c = project(-1, t, -1), d = project(1, t, -1);
            ctx.beginPath(); ctx.moveTo(c.sx, c.sy); ctx.lineTo(d.sx, d.sy); ctx.stroke();
          }}

          // Wireframe 3D bounding cube (clean dark slate border like reference plot)
          const pv = CUBE_VERTS.map(v => project(v[0], v[1], v[2]));
          ctx.strokeStyle = "#334155";
          ctx.lineWidth = 1.15;
          CUBE_EDGES.forEach(([i, j]) => {{
            ctx.beginPath();
            ctx.moveTo(pv[i].sx, pv[i].sy);
            ctx.lineTo(pv[j].sx, pv[j].sy);
            ctx.stroke();
          }});

          // Axis numeric ticks on X (front bottom), Y (right depth bottom), Z (left vertical)
          for (let i = 0; i <= 4; i++) {{
            const f = i / 4;
            const n = f * 2 - 1;

            const tx = project(n, -1, -1);
            ctx.fillStyle = "#334155";
            ctx.fillRect(tx.sx - 1, tx.sy - 2, 2, 5);
            drawTickLabel(tx.sx, tx.sy, "$" + Math.round((f * X_MAX) / 1000) + "k", 0, 14, "center");

            const ty = project(1, n, -1);
            ctx.fillRect(ty.sx - 2, ty.sy - 1, 5, 2);
            drawTickLabel(ty.sx, ty.sy, "$" + Math.round((Y_MIN + f * (Y_MAX - Y_MIN)) / 1000) + "k", 14, 4, "left");

            const tz = project(-1, -1, n);
            ctx.fillRect(tz.sx - 3, tz.sy - 1, 5, 2);
            drawTickLabel(tz.sx, tz.sy, (Z_MIN + f * (Z_MAX - Z_MIN)).toFixed(1) + "%", -8, 4, "right");
          }}

          // Axis titles placed along each axis midpoint
          ctx.font = "700 12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
          ctx.textAlign = "center";
          const midX = project(0, -1, -1);
          ctx.fillStyle = "#0F172A";
          ctx.fillText("Ingreso Anual (USD)", midX.sx, midX.sy + 32);

          const midY = project(1, 0, -1);
          ctx.textAlign = "left";
          ctx.fillText("Monto Préstamo (USD)", midY.sx + 48, midY.sy + 4);

          const midZ = project(-1, -1, 0);
          ctx.save();
          ctx.translate(midZ.sx - 56, midZ.sy);
          ctx.rotate(-Math.PI / 2);
          ctx.textAlign = "center";
          ctx.fillStyle = "#6D28D9";
          ctx.fillText("Tasa de Interés (% — loan_int_rate)", 0, 0);
          ctx.restore();

          // Project both floor anchor (nz=-1) and data dot (nz) & depth-sort
          const projected = [];
          const limit = Math.min(maxPoints, NORM_PTS.length);
          for (let i = 0; i < limit; i++) {{
            const item = NORM_PTS[i];
            if (classFilter !== "all" && String(item.raw.c) !== classFilter) continue;
            const top = project(item.nx, item.ny, item.nz);
            const base = showStems ? project(item.nx, item.ny, -1) : null;
            projected.push({{ top, base, raw: item.raw }});
          }}
          projected.sort((a, b) => a.top.depth - b.top.depth);

          let bestDist = 11, bestPt = null;
          for (let i = 0; i < projected.length; i++) {{
            const pt = projected[i];
            const top = pt.top;
            const alpha = Math.max(0.32, Math.min(0.95, 0.55 + top.depth * 0.22));
            const r = Math.max(2.6, Math.min(4.8, 3.2 * top.persp));

            // Vertical stem line showing loan_int_rate height from floor
            if (showStems && pt.base) {{
              ctx.strokeStyle = pt.raw.c === 1
                ? "rgba(5, 150, 105, " + (alpha * 0.58).toFixed(2) + ")"
                : "rgba(220, 38, 38, " + (alpha * 0.58).toFixed(2) + ")";
              ctx.lineWidth = 1.1;
              ctx.beginPath();
              ctx.moveTo(pt.base.sx, pt.base.sy);
              ctx.lineTo(top.sx, top.sy);
              ctx.stroke();
            }}

            ctx.fillStyle = pt.raw.c === 1
              ? "rgba(5, 150, 105, " + alpha.toFixed(2) + ")"
              : "rgba(220, 38, 38, " + alpha.toFixed(2) + ")";
            ctx.beginPath();
            ctx.arc(top.sx, top.sy, r, 0, Math.PI * 2);
            ctx.fill();

            const dx = top.sx - mouseX, dy = top.sy - mouseY;
            const d = Math.hypot(dx, dy);
            if (d < bestDist) {{ bestDist = d; bestPt = pt; }}
          }}

          if (bestPt) {{
            const top = bestPt.top;
            if (showStems && bestPt.base) {{
              ctx.strokeStyle = "#0F172A";
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.moveTo(bestPt.base.sx, bestPt.base.sy);
              ctx.lineTo(top.sx, top.sy);
              ctx.stroke();
            }}
            ctx.strokeStyle = "#0F172A";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(top.sx, top.sy, 6.5, 0, Math.PI * 2);
            ctx.stroke();

            tip.style.display = "block";
            tip.style.left = Math.min(W - 190, Math.max(8, top.sx + 12)) + "px";
            tip.style.top = Math.min(H - 95, Math.max(8, top.sy - 14)) + "px";
            const statusTxt = bestPt.raw.c === 1
              ? "<span style='color:#34D399;font-weight:700;'>Aprobado (1)</span>"
              : "<span style='color:#F87171;font-weight:700;'>Denegado (0)</span>";
            tip.innerHTML =
              statusTxt + "<br>" +
              "Ingreso: <b>$" + bestPt.raw.x.toLocaleString() + "</b><br>" +
              "Préstamo: <b>$" + bestPt.raw.y.toLocaleString() + "</b><br>" +
              "Tasa Interés: <b>" + bestPt.raw.z.toFixed(2) + "%</b>";
          }} else {{
            tip.style.display = "none";
          }}

          requestAnimationFrame(draw);
        }}
        requestAnimationFrame(draw);
      </script>
    </body>
    </html>
    """
    components.html(html_code, height=615)


# ===========================================================================
# TAB 4: EVALUACIÓN EXPERIMENTAL
# ===========================================================================
with tab_eval:
    st.header("📊 Evaluación Experimental de los Modelos")
    st.markdown(
        "Todos los valores presentados en esta sección provienen directamente de los experimentos "
        "ejecutados sobre la partición oficial del proyecto (**80% Train = 36,000** / **20% Test = 9,000**, `random_state=42`)."
    )

    model_names = list(eval_results["optimized_test"].keys())

    # Fuente de datos estratificada / balanceada desde StratifiedKFold (cv_5 completo) para gráficas EDA y Heatmap
    st.markdown("### ⚖️ Fuente de Datos para Gráficas: `StratifiedKFold (cv_5)` — 5 Folds de `X_train`")
    fold_choice = "Todos los 5 folds (cv_5)"
    use_balanced_cv5 = st.checkbox(
        "Balancear clases 50% / 50% dentro de `cv_5` (igualar Denegado 0 vs Aprobado 1)",
        value=True,
        help="StratifiedKFold conserva la proporción original (~77.8% vs ~22.2%) en cada fold; activar este balanceo iguala 50/50 ambas clases dentro de cv_5.",
    )

    csv_mtime = os.path.getmtime("loan_data.csv") if os.path.exists("loan_data.csv") else 0.0
    df_eda = load_cv5_eda_data(csv_mtime, fold_choice=fold_choice, balanced=use_balanced_cv5)

    if df_eda is not None:
        n_total_eda = len(df_eda)
        n_c0 = int((df_eda["loan_status"] == 0).sum())
        n_c1 = int((df_eda["loan_status"] == 1).sum())
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Registros activos (`cv_5`)", f"{n_total_eda:,}")
        mc2.metric("Clase 0 — Denegado", f"{n_c0:,} ({(n_c0 / n_total_eda) * 100:.1f}%)")
        mc3.metric("Clase 1 — Aprobado", f"{n_c1:,} ({(n_c1 / n_total_eda) * 100:.1f}%)")
        st.caption(
            "Nota técnica: `StratifiedKFold (cv_5)` garantiza que los 5 folds mantengan exactamente la misma proporción de clases sin sesgo. "
            + (
                "Con el balanceo 50/50 activado dentro de los 5 folds de `cv_5`, ambas clases tienen el mismo peso visual para comparar patrones reales sin dominancia de la clase mayoritaria."
                if use_balanced_cv5
                else "Mostrando la proporción estratificada original (~77.8% Denegado vs ~22.2% Aprobado) de los 5 folds de `cv_5`."
            )
        )

    # 1. Mapa de Calor de Correlaciones (Clase 1 vs Clase 0)
    st.subheader("1. Mapa de Calor de Correlaciones (Características asociadas a Clase 1 vs Clase 0)")
    st.markdown(
        "Correlación de Pearson calculada sobre los 5 folds de `cv_5` "
        "entre las características del solicitante y `loan_status` "
        "(**valores positivos en azul** se asocian con **Clase 1: Aprobado**; **valores negativos en rojo** se asocian con **Clase 0: No Aprobado**)."
    )

    corr_info = compute_corr_payload_from_df(df_eda) if df_eda is not None else eval_results.get("feature_correlations", {})
    if corr_info:
        all_cols = corr_info["columns"]
        corr_df_full = pd.DataFrame(corr_info["matrix"], index=all_cols, columns=all_cols)
        target_corr = pd.Series(corr_info["target_correlations"]).sort_values(ascending=False)

        show_all_corr = st.checkbox(
            "Mostrar todas las 22 variables codificadas (One-Hot completo) en el mapa de calor",
            value=False,
        )
        if show_all_corr:
            selected_cols = all_cols
        else:
            key_feats = [
                col for col in target_corr.index
                if col in NUMERICAL_COLUMNS or abs(target_corr[col]) >= 0.05
            ]
            selected_cols = key_feats + ["loan_status"]

        corr_sub = corr_df_full.loc[selected_cols, selected_cols]

        hcol1, hcol2 = st.columns([1.45, 1])
        with hcol1:
            n_vars = len(selected_cols)
            fig_hm, ax_hm = plt.subplots(
                figsize=(10.5 if not show_all_corr else 13.0, 8.2 if not show_all_corr else 10.5)
            )
            im = ax_hm.imshow(corr_sub.values, cmap="RdBu", vmin=-1, vmax=1)
            ax_hm.set_xticks(range(n_vars))
            ax_hm.set_yticks(range(n_vars))
            ax_hm.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=8 if not show_all_corr else 7)
            ax_hm.set_yticklabels(selected_cols, fontsize=8 if not show_all_corr else 7)
            ax_hm.set_title("Mapa de Calor de Correlaciones (Características vs loan_status)", fontsize=11, fontweight="bold")
            cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
            cbar.set_label("Correlación (r)", fontsize=9)

            font_cell = 6.8 if not show_all_corr else 5.2
            for r in range(n_vars):
                for c in range(n_vars):
                    val = corr_sub.values[r, c]
                    ax_hm.text(
                        c, r, f"{val:.2f}",
                        ha="center", va="center",
                        color="white" if abs(val) > 0.55 else "black",
                        fontsize=font_cell,
                    )
            plt.tight_layout()
            st.pyplot(fig_hm)
            plt.close(fig_hm)

        with hcol2:
            bar_series = target_corr.loc[[c for c in selected_cols if c != "loan_status"]].sort_values(ascending=True)
            fig_tc, ax_tc = plt.subplots(figsize=(6.8, 7.2 if not show_all_corr else 9.0))
            colors_tc = ["#DC2626" if v < 0 else "#1D4ED8" for v in bar_series.values]
            bars_tc = ax_tc.barh(bar_series.index, bar_series.values, color=colors_tc)
            ax_tc.axvline(0, color="black", linewidth=0.9)
            x_min_tc = min(-0.68, float(bar_series.min()) - 0.14)
            x_max_tc = max(0.52, float(bar_series.max()) + 0.12)
            ax_tc.set_xlim(x_min_tc, x_max_tc)
            ax_tc.set_xlabel("Correlación con loan_status (Clase 0 ← 0 → Clase 1)")
            ax_tc.set_title("Relación Directa con loan_status", fontsize=11, fontweight="bold")
            ax_tc.tick_params(axis="y", labelsize=8 if not show_all_corr else 7)
            for bar, val in zip(bars_tc, bar_series.values):
                offset = 0.015 if val >= 0 else -0.015
                ha = "left" if val >= 0 else "right"
                ax_tc.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
            ax_tc.grid(axis="x", linestyle="--", alpha=0.35)
            plt.tight_layout()
            st.pyplot(fig_tc)
            plt.close(fig_tc)

    # 2. Análisis Exploratorio de Datos (EDA) usando la fuente StratifiedKFold (cv_5)
    eda_mode_tag = "Balanceado 50/50" if use_balanced_cv5 else "Estratificado"
    st.subheader(f"2. Análisis Exploratorio de Datos (EDA — {fold_choice}, {eda_mode_tag})")
    st.markdown(
        f"Comportamiento de las principales características respecto al resultado del préstamo "
        f"(**1 = Aprobado**, **0 = Denegado**) sobre la partición de `StratifiedKFold (cv_5)` seleccionada "
        f"(**{len(df_eda):,} registros**)." if df_eda is not None else
        "Comportamiento de las principales características respecto al resultado del préstamo (**1 = Aprobado**, **0 = Denegado**)."
    )
    if df_eda is not None:
        eda_r1_c1, eda_r1_c2 = st.columns(2)
        with eda_r1_c1:
            plot_grouped_status_pct(
                df_eda,
                "loan_intent",
                "Propósito del Préstamo (loan_intent) vs loan_status",
                "Propósito del Préstamo (loan_intent)",
            )
        with eda_r1_c2:
            plot_grouped_status_pct(
                df_eda,
                "person_home_ownership",
                "Tipo de Vivienda (person_home_ownership) vs loan_status",
                "Tipo de Vivienda",
            )

        eda_r2_c1, eda_r2_c2 = st.columns(2)
        with eda_r2_c1:
            plot_grouped_status_pct(
                df_eda,
                "previous_loan_defaults_on_file",
                "Incumplimientos Previos vs loan_status",
                "Incumplimiento Previo Registrado",
            )
        with eda_r2_c2:
            plot_status_boxplot(
                df_eda,
                "loan_percent_income",
                "% de Ingreso Comprometido (loan_percent_income) vs loan_status",
                "Proporción del Ingreso (loan_percent_income)",
            )

        eda_r3_c1, eda_r3_c2 = st.columns(2)
        with eda_r3_c1:
            plot_status_boxplot(
                df_eda,
                "loan_int_rate",
                "Tasa de Interés (%) (loan_int_rate) vs loan_status",
                "Tasa de Interés (%)",
            )
        with eda_r3_c2:
            df_scat = df_eda.sample(n=min(4000, len(df_eda)), random_state=42)
            inc_cap = float(df_eda["person_income"].quantile(0.95))
            fig_sc, ax_sc = plt.subplots(figsize=(6.8, 4.4))
            for cls, label_txt, color_hex in [
                (0, "Denegado (0)", "#DC2626"),
                (1, "Aprobado (1)", "#059669"),
            ]:
                sub_s = df_scat[df_scat["loan_status"] == cls]
                ax_sc.scatter(
                    sub_s["person_income"],
                    sub_s["loan_amnt"],
                    s=14,
                    alpha=0.45,
                    color=color_hex,
                    label=label_txt,
                )
            ax_sc.set_xlim(0, inc_cap)
            ax_sc.set_xlabel("Ingreso Anual (USD, hasta percentil 95)")
            ax_sc.set_ylabel("Monto del Préstamo (USD)")
            ax_sc.set_title("Ingreso Anual vs Monto Solicitado por loan_status", fontsize=11, fontweight="bold")
            ax_sc.legend(title="loan_status", fontsize=8.5)
            ax_sc.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        st.markdown("**Vista 3D Interactiva: Ingreso Anual × Monto Solicitado × Tasa de Interés (`loan_int_rate`) por `loan_status`**")
        df_scat_3d = df_scat[df_scat["person_income"] <= inc_cap]
        render_interactive_3d_scatter(df_scat_3d, inc_cap)

        st.markdown("**¿Por qué los préstamos aprobados (verde) se concentran en menores ingresos y montos moderados?**")
        exp_c1, exp_c2 = st.columns(2)

        with exp_c1:
            df_lpi = df_eda[["loan_percent_income", "previous_loan_defaults_on_file", "loan_status"]].copy()
            lpi_bins = [0.0, 0.10, 0.20, 0.30, 0.40, 1.0]
            lpi_labels = ["0–10%", "10–20%", "20–30%", "30–40%", "> 40%"]
            df_lpi["lpi_bin"] = pd.cut(df_lpi["loan_percent_income"], bins=lpi_bins, labels=lpi_labels, include_lowest=True)

            rate_all = df_lpi.groupby("lpi_bin", observed=False)["loan_status"].mean().fillna(0.0) * 100.0
            df_no_def = df_lpi[df_lpi["previous_loan_defaults_on_file"] == "No"]
            rate_nodef = df_no_def.groupby("lpi_bin", observed=False)["loan_status"].mean().fillna(0.0) * 100.0

            x_b = np.arange(len(lpi_labels))
            w_b = 0.36
            fig_lpi, ax_lpi = plt.subplots(figsize=(6.8, 4.5))
            b_all = ax_lpi.bar(
                x_b - w_b / 2,
                rate_all.values,
                w_b,
                label="Todas las solicitudes (con y sin mora)",
                color="#1D4ED8",
            )
            b_nd = ax_lpi.bar(
                x_b + w_b / 2,
                rate_nodef.values,
                w_b,
                label="Solo sin mora previa (defaults = 'No')",
                color="#059669",
            )
            for bars in (b_all, b_nd):
                for b in bars:
                    h = b.get_height()
                    ax_lpi.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)
            ax_lpi.set_xticks(x_b)
            ax_lpi.set_xticklabels(lpi_labels)
            ax_lpi.set_ylim(0, 115)
            ax_lpi.set_xlabel("% del Ingreso Comprometido (loan_percent_income = Monto / Ingreso)")
            ax_lpi.set_ylabel("% de Préstamos Aprobados (Clase 1)")
            ax_lpi.set_title("Aprobación según % de Ingreso Solicitado (General vs Sin Mora)", fontsize=10.5, fontweight="bold")
            ax_lpi.legend(loc="upper left", fontsize=8.2)
            ax_lpi.grid(axis="y", linestyle="--", alpha=0.35)
            plt.tight_layout()
            st.pyplot(fig_lpi)
            plt.close(fig_lpi)

        with exp_c2:
            if "Random Forest" in models:
                rf_pipe = models["Random Forest"]
                raw_names = rf_pipe.named_steps["preprocesamiento"].get_feature_names_out()
                clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
                imp_vals = rf_pipe.named_steps["model"].feature_importances_ * 100.0
                top_imp = pd.Series(imp_vals, index=clean_names).sort_values(ascending=True).tail(10)

                fig_imp, ax_imp = plt.subplots(figsize=(6.8, 4.5))
                bar_colors = ["#059669" if v >= 10.0 else "#64748B" for v in top_imp.values]
                bars_imp = ax_imp.barh(top_imp.index, top_imp.values, color=bar_colors)
                for b, v in zip(bars_imp, top_imp.values):
                    ax_imp.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontsize=8)
                ax_imp.set_xlim(0, max(top_imp.values) * 1.18)
                ax_imp.set_xlabel("Importancia en el Modelo Random Forest (%)")
                ax_imp.set_title("Variables que Más Afectan la Predicción (Top 10)", fontsize=10.8, fontweight="bold")
                ax_imp.tick_params(axis="y", labelsize=8)
                ax_imp.grid(axis="x", linestyle="--", alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)
            else:
                st.info("Carga `models/random_forest.joblib` para visualizar la importancia de variables.")

        low_mask = df_eda["person_income"] <= 50000
        def_low = float((df_eda.loc[low_mask, "previous_loan_defaults_on_file"] == "Yes").mean() * 100.0)
        def_high = float((df_eda.loc[~low_mask, "previous_loan_defaults_on_file"] == "Yes").mean() * 100.0)
        rent_low = float((df_eda.loc[low_mask, "person_home_ownership"] == "RENT").mean() * 100.0)
        rent_high = float((df_eda.loc[~low_mask, "person_home_ownership"] == "RENT").mean() * 100.0)
        rate_ok = float(df_eda.loc[df_eda["loan_status"] == 1, "loan_int_rate"].mean())
        rate_no = float(df_eda.loc[df_eda["loan_status"] == 0, "loan_int_rate"].mean())

        st.info(
            "**¿Por qué aparece esa frontera diagonal y qué otras variables influyen?**\n"
            f"- **1. Razón `loan_percent_income = loan_amnt / person_income` (la diagonal):** "
            f"La frontera superior izquierda corresponde a solicitudes donde el monto representa **≥ 25–30%** del ingreso "
            f"(con tope frecuente en **$25,000**). En el conjunto de entrenamiento, cuando `loan_percent_income` supera `30%`, "
            f"la aprobación sube a **{rate_all['30–40%']:.1f}%–{rate_all['> 40%']:.1f}%**, frente a solo **{rate_all['0–10%']:.1f}%** cuando es `0–10%`.\n"
            f"- **2. Menor tasa de incumplimiento previo (`previous_loan_defaults_on_file`):** "
            f"Es la variable más influyente del modelo. En ingresos `≤ $50k`, el **{def_low:.1f}%** registra mora previa (`Yes`, donde se deniega el 100%), "
            f"mientras que en ingresos `> $50k` sube a **{def_high:.1f}%**.\n"
            f"- **3. Vivienda en renta (`RENT`) y tasa de interés (`loan_int_rate`):** "
            f"El **{rent_low:.1f}%** del grupo `≤ $50k` reporta vivienda `RENT` (vs **{rent_high:.1f}%** en `> $50k`), y los préstamos aprobados "
            f"concentran tasas de interés más altas (**{rate_ok:.2f}%** vs **{rate_no:.2f}%** en denegados), reflejando el patrón del dataset sintético (`SMOTENC`)."
        )
    else:
        st.warning("No se encontró `loan_data.csv` para generar las gráficas EDA.")

    # 3. Etapa Inicial (Baseline CV) — después de las gráficas estratificadas
    st.subheader("3. Etapa Inicial (Parámetros por Defecto — 5-Fold Cross-Validation en Train)")
    cv_rows = []
    for m_name, vals in eval_results["baseline_cv"].items():
        cv_rows.append(
            {
                "Modelo (Default)": m_name,
                "Accuracy (CV)": f"{vals['accuracy_mean']:.4f} (±{vals['accuracy_std']:.4f})",
                "Precision (CV)": f"{vals['precision_mean']:.4f}",
                "Recall (CV)": f"{vals['recall_mean']:.4f}",
                "F1-Score (CV)": f"{vals['f1_mean']:.4f} (±{vals['f1_std']:.4f})",
                "ROC-AUC (CV)": f"{vals['roc_auc_mean']:.4f}",
            }
        )
    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    # 4. Etapa Optimizada (Después de ajustar hiperparámetros)
    st.subheader("4. Etapa Optimizada — Después de Ajustar Hiperparámetros (Test Set 20% Holdout)")
    baseline_test_map = eval_results.get("baseline_test", {})
    test_rows = []
    for m_name, vals in eval_results["optimized_test"].items():
        if m_name in baseline_test_map:
            base_f1 = baseline_test_map[m_name]["f1"]
        else:
            base_f1 = eval_results["baseline_cv"][m_name]["f1_mean"]
        delta_f1 = vals["f1"] - base_f1
        test_rows.append(
            {
                "Modelo": m_name,
                "Mejores Hiperparámetros": str(vals["best_params"]),
                "Accuracy": vals["accuracy"],
                "Precision": vals["precision"],
                "Recall": vals["recall"],
                "F1-Score": vals["f1"],
                "Δ F1 vs Default": f"{delta_f1:+.4f}",
                "ROC-AUC": vals["roc_auc"],
            }
        )
    df_test_tbl = pd.DataFrame(test_rows)
    st.dataframe(df_test_tbl, use_container_width=True, hide_index=True)

    # 5. Gráficas comparativas: Barras de métricas y Curvas ROC reales
    st.subheader("5. Visualización Comparativa en Test Set (Modelos Optimizados)")
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = np.arange(len(metric_labels))
        width = 0.25
        palette = ["#1D4ED8", "#059669", "#D97706"]

        for i, m_name in enumerate(model_names):
            vals = [eval_results["optimized_test"][m_name][k] for k in metric_keys]
            bars = ax_bar.bar(x + (i - 1) * width, vals, width, label=m_name, color=palette[i])
            for b in bars:
                h = b.get_height()
                ax_bar.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_labels)
        ax_bar.set_ylim(0, 1.09)
        ax_bar.set_ylabel("Puntaje")
        ax_bar.set_title("Comparación de Métricas en Test Set")
        ax_bar.legend(loc="lower right", fontsize=8.5)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with gcol2:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 4.5))
        for i, m_name in enumerate(model_names):
            roc_data = eval_results["optimized_test"][m_name]["roc_curve"]
            auc_val = eval_results["optimized_test"][m_name]["roc_auc"]
            ax_roc.plot(roc_data["fpr"], roc_data["tpr"], lw=2, color=palette[i], label=f"{m_name} (AUC = {auc_val:.4f})")

        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2, label="Azar (AUC = 0.50)")
        ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax_roc.set_title("Curvas ROC en Test Set")
        ax_roc.legend(loc="lower right", fontsize=8.5)
        ax_roc.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

    # 6. Matrices de Confusión DESPUÉS de ajustar hiperparámetros + Comparativa Antes vs Después
    st.subheader("6. Matrices de Confusión Finales — Test Set Después del Ajuste de Hiperparámetros")
    opt_subtitles = {
        m: (
            f"Test Optimizado | Acc: {eval_results['optimized_test'][m]['accuracy']:.4f} | "
            f"F1: {eval_results['optimized_test'][m]['f1']:.4f} "
            f"({eval_results['optimized_test'][m]['f1'] - (baseline_test_map[m]['f1'] if m in baseline_test_map else eval_results['baseline_cv'][m]['f1_mean']):+.4f})"
        )
        for m in model_names
    }
    plot_confusion_matrices_row(eval_results["optimized_test"], model_names, cmap="Blues", subtitle_map=opt_subtitles)

    bva_rows = []
    if all(m in baseline_test_map for m in model_names):
        for m_name in model_names:
            cm_before = baseline_test_map[m_name]["confusion_matrix"]
            cm_after = eval_results["optimized_test"][m_name]["confusion_matrix"]
            tn0, fp0, fn0, tp0 = cm_before[0][0], cm_before[0][1], cm_before[1][0], cm_before[1][1]
            tn1, fp1, fn1, tp1 = cm_after[0][0], cm_after[0][1], cm_after[1][0], cm_after[1][1]
            bva_rows.append(
                {
                    "Modelo": m_name,
                    "TN (Antes → Después)": f"{tn0:,} → {tn1:,} ({tn1 - tn0:+d})",
                    "FP (Antes → Después)": f"{fp0:,} → {fp1:,} ({fp1 - fp0:+d})",
                    "FN (Antes → Después)": f"{fn0:,} → {fn1:,} ({fn1 - fn0:+d})",
                    "TP (Antes → Después)": f"{tp0:,} → {tp1:,} ({tp1 - tp0:+d})",
                    "F1 (Antes → Después)": (
                        f"{baseline_test_map[m_name]['f1']:.4f} → "
                        f"{eval_results['optimized_test'][m_name]['f1']:.4f} "
                        f"({eval_results['optimized_test'][m_name]['f1'] - baseline_test_map[m_name]['f1']:+.4f})"
                    ),
                }
            )
        st.markdown("**Comparativa Antes vs Después del ajuste de hiperparámetros en el Test Set (9,000 registros):**")
    else:
        for m_name in model_names:
            cv_b = eval_results["baseline_cv"][m_name]
            opt_a = eval_results["optimized_test"][m_name]
            bva_rows.append(
                {
                    "Modelo": m_name,
                    "Accuracy (CV Base → Test Opt)": f"{cv_b['accuracy_mean']:.4f} → {opt_a['accuracy']:.4f} ({opt_a['accuracy'] - cv_b['accuracy_mean']:+.4f})",
                    "Precision (CV Base → Test Opt)": f"{cv_b['precision_mean']:.4f} → {opt_a['precision']:.4f} ({opt_a['precision'] - cv_b['precision_mean']:+.4f})",
                    "Recall (CV Base → Test Opt)": f"{cv_b['recall_mean']:.4f} → {opt_a['recall']:.4f} ({opt_a['recall'] - cv_b['recall_mean']:+.4f})",
                    "F1-Score (CV Base → Test Opt)": f"{cv_b['f1_mean']:.4f} → {opt_a['f1']:.4f} ({opt_a['f1'] - cv_b['f1_mean']:+.4f})",
                    "ROC-AUC (CV Base → Test Opt)": f"{cv_b['roc_auc_mean']:.4f} → {opt_a['roc_auc']:.4f} ({opt_a['roc_auc'] - cv_b['roc_auc_mean']:+.4f})",
                }
            )
        st.markdown("**Comparativa Antes (5-Fold CV Train Default) vs Después (Test Set Optimizado):**")
    st.dataframe(pd.DataFrame(bva_rows), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 5: INTERPRETACIÓN Y LIMITACIONES
# ===========================================================================
with tab_interp:
    st.header("📘 Guía de Interpretación y Limitaciones")

    st.subheader("1. ¿Qué representa la clase predicha?")
    st.markdown(
        """
    - **Clase `1` — Aprobado (`Loan Approved`):** Indica que, según los patrones aprendidos en los datos históricos
      de `loan_data.csv`, la solicitud presenta características compatibles con solicitudes aprobadas.
    - **Clase `0` — No Aprobado (`Loan Not Approved`):** Indica que la solicitud se asemeja a perfiles rechazados
      en la base de entrenamiento.
    """
    )

    st.subheader("2. ¿Cómo se interpretan las probabilidades y puntuaciones de confianza?")
    st.markdown(
        """
    - **Probabilidad estimada (`predict_proba`):**
      - En **Regresión Logística**, la probabilidad proviene directamente de la función sigmoide aplicada a la combinación lineal de características.
      - En **SVM RBF**, la probabilidad se calcula mediante calibración de Platt (`probability=True`) sobre la distancia al hiperplano separador.
      - En **Random Forest**, la probabilidad corresponde a la proporción promedio de votos entre los árboles de decisión del ensamble.
      - Un valor cercano a **100%** en Clase `1` significa alta confianza estadística de aprobación; un valor cercano a **50%** indica incertidumbre cerca de la frontera de decisión, donde los 3 modelos pueden discrepar.
    - **Score de decisión (`decision_function`):** En SVM y modelos lineales representa la distancia firmada respecto al hiperplano de separación (`>= 0` corresponde a Clase `1` y `< 0` a Clase `0`).
    """
    )

    st.subheader("3. Limitaciones de las predicciones generadas")
    st.warning(
        """
    1. **Naturaleza sintética de la base de datos (`SMOTENC`):** `loan_data.csv` es un conjunto aumentado sintéticamente a partir de datos de riesgo crediticio. Como se observó en el análisis exploratorio del notebook (`loan_approval.ipynb`), entre registros sin mora previa, variables como `loan_percent_income` y `loan_int_rate` muestran correlación positiva con la clase `1`, lo que refleja la estructura del dataset sintético y no necesariamente una regla bancaria real de bajo riesgo.
    2. **Desbalance de clases (77.78% `0` vs 22.22% `1`):** El modelo prioriza el desempeño global (`F1` y `ROC-AUC`), por lo que la clase minoritaria (`1`) presenta menor *recall* que la clase mayoritaria (`0`).
    3. **Regla estricta aprendida sobre incumplimientos previos:** En los 45,000 registros de `loan_data.csv`, el 100% de los solicitantes con `previous_loan_defaults_on_file = "Yes"` (22,858 casos) pertenecen a la clase `0`. Por ello, cualquier entrada con incumplimiento previo será clasificada como `0` con altísima probabilidad.
    4. **Riesgo de extrapolación fuera de distribución:** Valores extremos no presentes en el entrenamiento (por ejemplo, `loan_percent_income > 0.66`, `loan_amnt > person_income` o edades atípicas) pueden generar predicciones inestables.
    5. **Uso como herramienta de apoyo:** Los resultados constituyen una estimación estadística basada en el proyecto académico y no sustituyen el análisis financiero, legal ni normativo de una entidad bancaria real.
    """
    )

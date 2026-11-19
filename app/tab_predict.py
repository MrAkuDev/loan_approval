"""Pestaña 2: Formulario de predicción individual y comparación entre los 3 modelos."""

import os

import pandas as pd
import streamlit as st

from app.charts import (
    plot_instance_feature_contributions,
    plot_single_prediction_confidence,
)
from app.config import CSV_PATH, FEATURE_ORDER, FEATURE_SHORT_LABELS, VALID_CATEGORIES
from app.frontend import (
    render_hyperparameters_list,
    render_prediction_cards,
    render_threshold_selector,
    show_inline_loading,
)
from app.services import (
    apply_threshold_to_single_preds,
    compute_local_feature_contributions,
    load_explainability_background,
    predict_with_model,
)
from app.validation import build_typed_instance, validate_single_instance


def _render_consensus_summary(preds: dict[str, dict]) -> None:
    """Muestra el resumen de consenso o discrepancia entre los 3 modelos."""
    st.markdown("### 🔍 Comparación y Diferencias entre Modelos")
    classes_set = {res["pred"] for res in preds.values()}
    default_models = [k for k, v in preds.items() if v["pred"] == 1]
    safe_models = [k for k, v in preds.items() if v["pred"] == 0]

    if len(classes_set) == 1:
        pred_val = list(classes_set)[0]
        if pred_val == 0:
            st.success(
                "🤝 **Consenso Unánime (3 de 3 modelos):** Todos los modelos predicen **Sin Incumplimiento (0)** (bajo riesgo)."
            )
        else:
            st.error(
                "🤝 **Consenso Unánime (3 de 3 modelos):** Todos los modelos predicen **Incumplimiento (1)** (alto riesgo)."
            )
    else:
        st.warning(
            f"⚖️ **Discrepancia entre modelos:** "
            f"**Sin Incumplimiento (0):** {', '.join(safe_models)} | "
            f"**Incumplimiento (1):** {', '.join(default_models)}. "
            "Esto ocurre cuando la instancia se encuentra cerca de la frontera de decisión y "
            "la frontera lineal (Regresión Logística), el ensamble de árboles (Random Forest) y la frontera gaussiana (SVM RBF) "
            "ponderan distinto las interacciones entre grado crediticio, ingreso, tasa de interés y porcentaje comprometido."
        )


def _render_local_explanations(models: dict, X_input: pd.DataFrame) -> None:
    """Calcula y grafica cuánto empujó cada característica la decisión de cada modelo."""
    st.subheader("🧠 Contribución de Cada Característica en Esta Predicción (por Modelo)")
    st.caption(
        "Atribución local por permutaciones (Shapley) respecto a casos reales de entrenamiento: "
        "las barras **rojas (+ pp)** incrementan la probabilidad de **Incumplimiento (1)** y las **verdes (− pp)** reducen el riesgo hacia **Sin Incumplimiento (0)**."
    )

    csv_mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0
    sh_loader = st.empty()
    show_inline_loading(
        sh_loader,
        "Calculando explicación por característica",
        "Midiendo el impacto Shapley en SVM RBF, Random Forest y Regresión Logística...",
    )
    bg_df = load_explainability_background(csv_mtime, n_samples=24)
    attributions = compute_local_feature_contributions(
        models=models,
        X_input=X_input,
        bg_df=bg_df,
        bg_sig=f"{csv_mtime}:{len(bg_df)}",
    )
    sh_loader.empty()

    plot_instance_feature_contributions(attributions, X_input)

    driver_rows = []
    for m_name, info in attributions.items():
        contribs = info["contributions"]
        top_risk = max(contribs.items(), key=lambda kv: kv[1])
        top_safe = min(contribs.items(), key=lambda kv: kv[1])
        driver_rows.append(
            {
                "Modelo": m_name,
                "Principal Factor de Riesgo (Incumplimiento +)": f"{FEATURE_SHORT_LABELS.get(top_risk[0], top_risk[0])} ({top_risk[1] * 100:+.1f} pp)",
                "Principal Factor Protector (Sin Incumplimiento −)": f"{FEATURE_SHORT_LABELS.get(top_safe[0], top_safe[0])} ({top_safe[1] * 100:+.1f} pp)",
                "Base Entrenamiento → P(Incumplimiento)": f"{info['base_value'] * 100:.1f}% → {info['pred_value'] * 100:.1f}%",
            }
        )
    st.dataframe(pd.DataFrame(driver_rows), use_container_width=True, hide_index=True)


def render_tab_predict(models: dict, eval_results: dict | None = None) -> None:
    """Renderiza la tarjeta 'Predecir' en Inicio y los resultados comparativos de predicción."""
    with st.container(border=True):
        st.markdown('<div class="m3-section-title">Predecir</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="m3-section-subtitle">'
            "Complete los 11 atributos del solicitante (en el formato de <code>credit_risk_dataset.csv</code>). "
            "Los tres modelos entrenados evaluarán simultáneamente la probabilidad de incumplimiento (<code>loan_status</code>)."
            "</div>",
            unsafe_allow_html=True,
        )

        if eval_results is not None:
            with st.expander("📊 Ver detalle de hiperparametros"):
                render_hyperparameters_list(eval_results)

        with st.form("single_prediction_form"):
            st.subheader("1. Datos Personales y Situación Laboral")
            c1, c2 = st.columns(2)
            with c1:
                person_age = st.number_input("Edad (`person_age`)", min_value=18, max_value=100, value=26, step=1)
                person_home_ownership = st.selectbox(
                    "Tipo de Vivienda (`person_home_ownership`)",
                    options=VALID_CATEGORIES["person_home_ownership"],
                    index=0,
                )
            with c2:
                person_income = st.number_input(
                    "Ingreso Anual USD (`person_income`)",
                    min_value=1000,
                    value=65000,
                    step=500,
                )
                person_emp_length = st.number_input(
                    "Antigüedad Laboral en años (`person_emp_length`)",
                    min_value=0.0,
                    max_value=60.0,
                    value=4.0,
                    step=1.0,
                    format="%.1f",
                )

            st.subheader("2. Características del Préstamo")
            c4, c5, c6, c7_g = st.columns(4)
            with c4:
                loan_amnt = st.number_input(
                    "Monto Solicitado USD (`loan_amnt`)",
                    min_value=100,
                    value=10000,
                    step=500,
                )
            with c5:
                loan_intent = st.selectbox(
                    "Propósito (`loan_intent`)",
                    options=VALID_CATEGORIES["loan_intent"],
                    index=0,
                )
            with c6:
                loan_grade = st.selectbox(
                    "Grado del Préstamo (`loan_grade`)",
                    options=VALID_CATEGORIES["loan_grade"],
                    index=1,
                )
            with c7_g:
                loan_int_rate = st.number_input(
                    "Tasa de Interés % (`loan_int_rate`)",
                    min_value=1.0,
                    max_value=35.0,
                    value=10.99,
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
                    min_value=0.0,
                    max_value=1.0,
                    value=float(min(auto_ratio, 1.0)),
                    step=0.01,
                    format="%.2f",
                    disabled=use_auto_ratio,
                )
            loan_percent_income = auto_ratio if use_auto_ratio else manual_ratio

            st.subheader("3. Historial Crediticio (Buró)")
            c8, c9 = st.columns(2)
            with c8:
                cb_person_cred_hist_length = st.number_input(
                    "Antigüedad del Historial Crediticio en años (`cb_person_cred_hist_length`)",
                    min_value=0,
                    max_value=60,
                    value=4,
                    step=1,
                )
            with c9:
                cb_person_default_on_file = st.selectbox(
                    "¿Incumplimiento Previo Registrado? (`cb_person_default_on_file`)",
                    options=VALID_CATEGORIES["cb_person_default_on_file"],
                    index=0,
                )

            _, col_submit = st.columns([3, 1])
            with col_submit:
                submit_btn = st.form_submit_button("➜  Predict", use_container_width=True)

    if submit_btn:
        raw_instance = {
            "person_age": person_age,
            "person_income": person_income,
            "person_home_ownership": person_home_ownership,
            "person_emp_length": person_emp_length,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_default_on_file": cb_person_default_on_file,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
        }

        val_errors, val_warnings = validate_single_instance(raw_instance)
        if val_errors:
            st.session_state.pop("home_last_prediction", None)
            for err in val_errors:
                st.error(f"❌ Error de validación: {err}")
            return

        pred_loader = st.empty()
        show_inline_loading(
            pred_loader,
            "Evaluando solicitud",
            "Ejecutando SVM RBF, Random Forest y Regresión Logística...",
        )
        instance_dict = build_typed_instance(raw_instance)
        X_input = pd.DataFrame([instance_dict])[FEATURE_ORDER]
        preds = {name: predict_with_model(pipe, X_input) for name, pipe in models.items()}
        pred_loader.empty()

        st.session_state["home_last_prediction"] = {
            "val_warnings": val_warnings,
            "X_input": X_input,
            "preds": preds,
        }

    cached_pred = st.session_state.get("home_last_prediction")
    if cached_pred:
        with st.container(border=True):
            for warn in cached_pred["val_warnings"]:
                st.warning(f"⚠️ Advertencia de validación: {warn}")

            st.subheader("🎯 Resultados de Predicción por Modelo")
            active_thresholds = render_threshold_selector(
                eval_results,
                list(cached_pred["preds"].keys()),
                key_prefix="predict",
            )
            effective_preds = apply_threshold_to_single_preds(
                cached_pred["preds"],
                active_thresholds,
            )
            comp_rows = [
                {
                    "Modelo": m_name,
                    "Umbral": round(res.get("threshold", 0.5), 2),
                    "Clase Predicha": res["pred"],
                    "Etiqueta": res["label"],
                    "Prob. Sin Incumplimiento (Clase 0)": round(res["prob_0"], 4) if res["prob_0"] is not None else None,
                    "Prob. Incumplimiento (Clase 1)": round(res["prob_1"], 4) if res["prob_1"] is not None else None,
                    "Decision Function": round(res["decision_score"], 4) if res["decision_score"] is not None else None,
                }
                for m_name, res in effective_preds.items()
            ]
            render_prediction_cards(effective_preds)
            _render_consensus_summary(effective_preds)

            df_comp = pd.DataFrame(comp_rows)
            c_tbl, c_fig = st.columns([1.2, 1])
            with c_tbl:
                st.dataframe(df_comp, use_container_width=True, hide_index=True)
            with c_fig:
                plot_single_prediction_confidence(comp_rows)

            _render_local_explanations(models, cached_pred["X_input"])

            with st.expander("📋 Ver vector de características exacto enviado a los pipelines"):
                st.dataframe(cached_pred["X_input"], use_container_width=True, hide_index=True)

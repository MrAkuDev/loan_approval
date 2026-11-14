"""Pestaña 2: Formulario de predicción individual y comparación entre los 3 modelos."""

import pandas as pd
import streamlit as st

from app.charts import plot_single_prediction_confidence
from app.config import FEATURE_ORDER, VALID_CATEGORIES
from app.frontend import render_prediction_cards
from app.services import predict_with_model
from app.validation import build_typed_instance, validate_single_instance


def _render_consensus_summary(preds: dict[str, dict]) -> None:
    """Muestra el resumen de consenso o discrepancia entre los 3 modelos."""
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


def render_tab_predict(models: dict) -> None:
    """Renderiza el formulario individual y los resultados comparativos de predicción."""
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
            return

        for warn in val_warnings:
            st.warning(f"⚠️ Advertencia de validación: {warn}")

        instance_dict = build_typed_instance(raw_instance)
        X_input = pd.DataFrame([instance_dict])[FEATURE_ORDER]
        preds = {name: predict_with_model(pipe, X_input) for name, pipe in models.items()}

        st.markdown("---")
        st.subheader("🎯 Resultados de Predicción por Modelo")
        render_prediction_cards(preds)
        _render_consensus_summary(preds)

        comp_rows = [
            {
                "Modelo": m_name,
                "Clase Predicha": res["pred"],
                "Etiqueta": res["label"],
                "Prob. Aprobado (Clase 1)": round(res["prob_1"], 4) if res["prob_1"] is not None else None,
                "Prob. No Aprobado (Clase 0)": round(res["prob_0"], 4) if res["prob_0"] is not None else None,
                "Decision Function": round(res["decision_score"], 4) if res["decision_score"] is not None else None,
            }
            for m_name, res in preds.items()
        ]
        df_comp = pd.DataFrame(comp_rows)

        c_tbl, c_fig = st.columns([1.2, 1])
        with c_tbl:
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        with c_fig:
            plot_single_prediction_confidence(comp_rows)

        with st.expander("📋 Ver vector de características exacto enviado a los pipelines"):
            st.dataframe(X_input, use_container_width=True, hide_index=True)

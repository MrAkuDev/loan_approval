"""Diseño de interfaz (frontend): configuración de página, CSS, barra lateral y tarjetas."""

import streamlit as st

CUSTOM_CSS = """
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
"""


def configure_page() -> None:
    """Configura los metadatos de página e inyecta los estilos CSS personalizados."""
    st.set_page_config(
        page_title="Loan Approval — Comparador de 3 Modelos ML",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    """Renderiza la barra lateral con información académica y los modelos integrados."""
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


def render_header() -> None:
    """Renderiza el encabezado principal de la aplicación."""
    st.markdown(
        '<div class="main-header">🏦 Predicción de Aprobación de Préstamos Bancarios</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">'
        "Comparación en tiempo real de los tres mejores modelos de Machine Learning del proyecto final: "
        "<b>SVM (RBF)</b>, <b>Random Forest</b> y <b>Regresión Logística</b>."
        "</div>",
        unsafe_allow_html=True,
    )


def render_missing_artifacts_error(missing_files: list[str]) -> None:
    """Muestra un error claro cuando faltan los archivos entrenados en `models/`."""
    st.error(
        "⚠️ Faltan archivos de modelos entrenados o resultados de evaluación. "
        "Ejecuta `python model_factory/train_model.py` primero para entrenar y guardar los tres modelos en `models/`."
    )
    if missing_files:
        st.code("\n".join(missing_files))
    st.stop()


def render_prediction_cards(preds: dict[str, dict]) -> None:
    """Renderiza las tarjetas comparativas de predicción para los 3 modelos."""
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

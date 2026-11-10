import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from features import agregar_features, evaluar_politicas_duras

# --- Page Configuration ---
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .status-approved {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-rejected {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-policy {
        background: linear-gradient(135deg, #991B1B 0%, #7F1D1D 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 700;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }
    .risk-card {
        background-color: #FFF5F5;
        border: 1px solid #FED7D7;
        border-left: 5px solid #E53E3E;
        border-radius: 8px;
        padding: 18px;
        margin-top: 8px;
        height: 100%;
    }
    .advice-card {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #10B981;
        border-radius: 8px;
        padding: 18px;
        margin-top: 8px;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loader & Diagnostic Logic ---
@st.cache_resource
def load_model():
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()


def analizar_causas_declinacion(row):
    """
    Analiza las variables del solicitante para identificar los factores que llevaron
    al rechazo y genera recomendaciones prácticas y accionables para mejorar.
    """
    reasons = []
    recommendations = []

    # 1. Historial de Incumplimiento Previo
    if row["previous_loan_defaults_on_file"] == "Yes":
        reasons.append(
            "**Incumplimiento Previo en Historial:** Existe un registro histórico de default/mora. "
            "En el modelo, este es el factor de mayor peso negativo para el perfil de riesgo."
        )
        recommendations.append(
            "**Sanear y regularizar antecedentes:** Asegure cero atrasos en créditos y servicios durante los próximos "
            "12 a 24 meses y solicite una carta de no adeudo o finiquito de cuentas cerradas."
        )

    # 2. Ratio Préstamo / Ingreso (Debt-to-Income)
    ratio = row["loan_percent_income"]
    if ratio >= 0.30:
        pct = ratio * 100
        reasons.append(
            f"**Carga Financiera Excesiva ({pct:.1f}% de su ingreso):** El préstamo solicitado (${row['loan_amnt']:,.0f}) "
            "supera el umbral prudencial del 30% del ingreso anual del solicitante."
        )
        monto_sugerido = row["person_income"] * 0.22
        recommendations.append(
            f"**Reducir el monto solicitado:** Pruebe solicitar un monto menor (ej. hasta **${monto_sugerido:,.0f}**) "
            "o aumente el plazo de amortización para reducir la mensualidad."
        )

    # 3. Puntaje de Crédito (Credit Score)
    score = row["credit_score"]
    if score < 640:
        reasons.append(
            f"**Puntaje Crediticio por Debajo del Promedio ({score} pts):** Se encuentra en rango 'Fair/Poor', "
            "por debajo del umbral óptimo bancario (650–700+ puntos)."
        )
        recommendations.append(
            "**Incrementar el Score Crediticio:** Mantenga el uso de sus tarjetas por debajo del 30% del límite asignado, "
            "evite aperturas múltiples de crédito y conserve cuentas antiguas activas."
        )

    # 4. Tasa de Interés Asignada
    tasa = row["loan_int_rate"]
    if tasa >= 14.0:
        reasons.append(
            f"**Tasa de Interés Elevada ({tasa:.1f}%):** Implica un costo financiero mensual alto, lo que incrementa "
            "la probabilidad estadística de impago."
        )
        recommendations.append(
            "**Negociar mejores términos o garantía:** Solicite cotizaciones con tasa fija más baja o considere "
            "aportar una garantía prendaria/hipotecaria para acceder a una tasa preferencial."
        )

    # 5. Experiencia / Antigüedad Laboral
    exp = row["person_emp_exp"]
    if exp < 2:
        reasons.append(
            f"**Antigüedad Laboral Corta ({exp} año(s)):** El modelo penaliza la falta de historial de estabilidad "
            "laboral continua (menos de 2 años)."
        )
        recommendations.append(
            "**Acreditar estabilidad en el empleo:** Considere re-aplicar tras cumplir al menos 1 a 2 años de trabajo "
            "continuo con el mismo empleador o adjuntar comprobantes de ingresos adicionales."
        )

    # 6. Vivienda en Arrendamiento + Carga de Crédito
    if row["person_home_ownership"] == "RENT" and ratio > 0.20:
        reasons.append(
            "**Gasto Fijo en Alquiler:** Al pagar renta de vivienda y sumarle una deuda superior al 20% del ingreso, "
            "el flujo neto disponible para imprevistos se reduce notablemente."
        )
        recommendations.append(
            "**Incorporar un Co-deudor / Aval:** Presentar una solicitud mancomunada con un familiar que sea propietario "
            "o tenga ingresos complementarios eleva drásticamente la probabilidad de aprobación."
        )

    # Caso en que ningún factor sea extremo individualmente pero la combinación descalifique
    if not reasons:
        reasons.append(
            "**Perfil de Riesgo Multivariable Combinado:** Aunque ningún parámetro individual rebasa los límites críticos, "
            "la interacción combinada de monto, nivel de ingreso e historial ubica la solicitud fuera de los parámetros de aceptación."
        )
        recommendations.append(
            "**Ajuste Marginal:** Reducir el monto solicitado entre un 10% y 15% o realizar un abono de enganche inicial "
            "suele ser suficiente para cruzar al cuadrante de aprobación."
        )

    return reasons, recommendations

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.title("Loan Approval AI")
    st.markdown("Machine Learning model to assess credit risk and predict loan approval eligibility.")
    
    st.divider()
    st.subheader("🎓 Información Académica")
    st.markdown("""
    - **Alumno:** Joel Arturo Álvarez Hernández
    - **Matrícula:** 403233
    - **Universidad:** Univ. Autónoma de Chihuahua (UACH)
    - **Facultad:** Facultad de Ingeniería
    - **Materia:** Machine Learning — Proyecto Final
    - **Docente:** Dra. Olanda Prieto Ordaz
    """)

    st.divider()
    st.subheader("📊 Arquitectura del Modelo")
    st.markdown("""
    - **Tipo:** Ensamble Stacking (`StackingClassifier`)
    - **Modelos Base:**
      1. Gradient Boosting (Optimizado)
      2. SVM RBF (Kernel Gaussiano)
      3. Regresión Logística (L2)
    - **Meta-Clasificador:** Regresión Logística
    - **Validación Cruzada:** 5 Folds
    - **Accuracy:** ~93.0%
    - **F1-Score:** ~83.5%
    - **ROC-AUC:** ~97.5%
    """)
    st.divider()
    st.caption("Powered by Streamlit & Scikit-Learn")

# --- Main Header ---
st.markdown('<div class="main-header">🏦 Loan Approval Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enter applicant details below to predict loan status in real-time.</div>', unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `model.joblib` was not found. Please run `python train_model.py` first to generate the trained model.")
    st.stop()

# --- Input Form ---
with st.form("loan_application_form"):
    st.subheader("1. Applicant Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        person_age = st.number_input("Age", min_value=18, max_value=100, value=28, step=1)
        person_gender = st.selectbox("Gender", options=["female", "male"])
        
    with col2:
        person_education = st.selectbox(
            "Education Level", 
            options=["High School", "Associate", "Bachelor", "Master", "Doctorate"],
            index=2
        )
        person_home_ownership = st.selectbox(
            "Home Ownership", 
            options=["RENT", "OWN", "MORTGAGE", "OTHER"],
            index=0
        )
        
    with col3:
        person_income = st.number_input("Annual Income ($)", min_value=1000.0, max_value=10000000.0, value=65000.0, step=1000.0, format="%.2f")
        person_emp_exp = st.number_input("Employment Experience (years)", min_value=0, max_value=60, value=4, step=1)

    st.subheader("2. Loan Details")
    col4, col5, col6 = st.columns(3)
    
    with col4:
        loan_amnt = st.number_input("Requested Loan Amount ($)", min_value=500.0, max_value=100000.0, value=12000.0, step=500.0, format="%.2f")
        
    with col5:
        loan_intent = st.selectbox(
            "Loan Purpose", 
            options=["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            index=0
        )
        
    with col6:
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=11.5, step=0.1, format="%.2f")

    # Dynamic calculation of loan percent of income
    calculated_percent_income = round(loan_amnt / person_income, 2) if person_income > 0 else 0.0
    st.info(f"💡 Calculated Debt-to-Income Ratio (`loan_percent_income`): **{calculated_percent_income:.2f}** ({calculated_percent_income*100:.1f}% of annual income)")

    st.subheader("3. Credit History")
    col7, col8, col9 = st.columns(3)
    
    with col7:
        credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=650, step=1)
        
    with col8:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=40, value=5, step=1)
        
    with col9:
        previous_loan_defaults_on_file = st.selectbox(
            "Previous Defaults on File?", 
            options=["No", "Yes"],
            index=0
        )

    submit_button = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True)

# --- Prediction Logic ---
if submit_button:
    # Prepare input dictionary and DataFrame
    raw_dict = {
        "person_age": float(person_age),
        "person_gender": person_gender,
        "person_education": person_education,
        "person_income": float(person_income),
        "person_emp_exp": int(person_emp_exp),
        "person_home_ownership": person_home_ownership,
        "loan_amnt": float(loan_amnt),
        "loan_intent": loan_intent,
        "loan_int_rate": float(loan_int_rate),
        "loan_percent_income": float(calculated_percent_income),
        "cb_person_cred_hist_length": float(cb_person_cred_hist_length),
        "credit_score": int(credit_score),
        "previous_loan_defaults_on_file": previous_loan_defaults_on_file
    }
    input_data = pd.DataFrame([raw_dict])

    try:
        st.markdown("---")
        st.subheader("🎯 Resultado de la Evaluación Crediticia")

        # -------------------------------------------------------------
        # 1. EVALUAR POLÍTICAS DURAS DE SUSCRIPCIÓN (KNOCKOUT RULES)
        # -------------------------------------------------------------
        es_rechazo_politica, motivos_politica, recs_politica = evaluar_politicas_duras(raw_dict)

        if es_rechazo_politica:
            st.markdown(
                '<div class="status-policy">❌ LOAN DECLINED BY CREDIT POLICY (RECHAZO POR POLÍTICA)</div>', 
                unsafe_allow_html=True
            )
            st.error(
                "**Rechazo Automático de Suscripción:** La solicitud infringe las reglas duras mínimas "
                "de la política de crédito institucional (Knockout Criteria). No califica para evaluación del modelo estadístico."
            )

            # Resumen de Métricas
            st.markdown("#### 🔍 Applicant Evaluation Summary")
            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                st.metric("Loan / Income Ratio", f"{calculated_percent_income*100:.1f}%", "Exceso Crítico (≥ 45%)" if calculated_percent_income > 0.45 else "Evaluado", delta_color="inverse" if calculated_percent_income > 0.45 else "normal")
            with rcol2:
                st.metric("Credit Score", credit_score, "Zona de Exclusión (< 550)" if credit_score < 550 else "Evaluado", delta_color="inverse" if credit_score < 550 else "normal")
            with rcol3:
                st.metric("Historical Defaults", previous_loan_defaults_on_file, "Criterio Excluyente" if previous_loan_defaults_on_file == "Yes" else "Sin Mora", delta_color="inverse" if previous_loan_defaults_on_file == "Yes" else "normal")

            # Diagnóstico de Política
            st.markdown("---")
            st.markdown("### 🛑 Diagnóstico de Incumplimiento de Política y Plan de Acción")
            col_risk, col_advice = st.columns(2)

            with col_risk:
                st.markdown('<div class="risk-card">', unsafe_allow_html=True)
                st.markdown("#### 🚩 Reglas Duras Infringidas")
                st.caption("Criterios de política crediticia que detuvieron la solicitud:")
                for m in motivos_politica:
                    st.markdown(f"- {m}")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_advice:
                st.markdown('<div class="advice-card">', unsafe_allow_html=True)
                st.markdown("#### 💡 ¿Cómo superar las políticas institucionales?")
                st.caption("Ajustes obligatorios requeridos para que una solicitud sea admisible:")
                for r in recs_politica:
                    st.markdown(f"- {r}")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            # -------------------------------------------------------------
            # 2. EVALUACIÓN CON EL MODELO STACKING (ML INFERENCE)
            # -------------------------------------------------------------
            # Generar variables de ingeniería financiera
            input_data_fe = agregar_features(input_data)

            # Obtener probabilidades del ensamble
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_data_fe)[0]
                # En el dataset de riesgo crediticio: Clase 0 = Solvente/Cumplido, Clase 1 = Riesgo de Mora
                prob_solvency = float(probs[0])
                prob_risk = float(probs[1])
            else:
                raw_pred = model.predict(input_data_fe)[0]
                prob_solvency = 1.0 if raw_pred == 0 else 0.0
                prob_risk = 1.0 - prob_solvency

            # Umbral prudencial de aprobación (requiere al menos 65% de probabilidad de solvencia)
            APPROVAL_THRESHOLD = 0.65
            is_approved = (prob_solvency >= APPROVAL_THRESHOLD)

            if is_approved:
                st.balloons()
                st.markdown(
                    f'<div class="status-approved">✅ LOAN APPROVED (APROBADO — Solvencia: {prob_solvency*100:.1f}%)</div>', 
                    unsafe_allow_html=True
                )
                st.success(
                    f"La solicitud cumple satisfactoriamente las políticas bancarias y superó el umbral "
                    f"del modelo Stacking con un **{prob_solvency*100:.1f}%** de probabilidad de solvencia proyectada."
                )
            else:
                st.markdown(
                    f'<div class="status-rejected">❌ LOAN REJECTED (NO APROBADO — Riesgo de Impago: {prob_risk*100:.1f}%)</div>', 
                    unsafe_allow_html=True
                )
                st.error(
                    f"Aunque la solicitud pasó los filtros duros iniciales, el ensamble de Machine Learning "
                    f"detectó un riesgo de mora del **{prob_risk*100:.1f}%** (Solvencia proyectada: **{prob_solvency*100:.1f}%**, requerido mínimo: **{APPROVAL_THRESHOLD*100:.0f}%**)."
                )

            # Resumen de Métricas
            st.markdown("#### 🔍 Applicant Evaluation Summary")
            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                ratio_status = "Healthy (< 30%)" if calculated_percent_income < 0.3 else "High Burden (≥ 30%)"
                st.metric("Loan / Income Ratio", f"{calculated_percent_income*100:.1f}%", ratio_status)
            with rcol2:
                score_status = "Excellent" if credit_score >= 720 else ("Good" if credit_score >= 650 else "Fair/Poor")
                st.metric("Credit Score", credit_score, score_status)
            with rcol3:
                st.metric("Historical Defaults", previous_loan_defaults_on_file, "Clean Record", delta_color="normal")

            # Diagnóstico en caso de rechazo estadístico
            if not is_approved:
                st.markdown("---")
                st.markdown("### ⚠️ Diagnóstico Estadístico y Plan de Mejora")
                reasons, recommendations = analizar_causas_declinacion(input_data.iloc[0])

                col_risk, col_advice = st.columns(2)
                with col_risk:
                    st.markdown('<div class="risk-card">', unsafe_allow_html=True)
                    st.markdown("#### 🚩 Factores de Riesgo Estadístico")
                    st.caption("Puntos desfavorables identificados por el modelo Stacking:")
                    for r in reasons:
                        st.markdown(f"- {r}")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_advice:
                    st.markdown('<div class="advice-card">', unsafe_allow_html=True)
                    st.markdown("#### 💡 Recomendaciones para Mejorar")
                    st.caption("Pasos sugeridos para incrementar el score en una próxima evaluación:")
                    for rec in recommendations:
                        st.markdown(f"- {rec}")
                    st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("📋 View Sent Input Payload"):
            st.dataframe(input_data)

    except Exception as e:
        st.error(f"Error making prediction: {e}")

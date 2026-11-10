import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

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
</style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def load_model():
    model_path = "model.joblib"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
    st.title("Loan Approval AI")
    st.markdown("Machine Learning model to assess credit risk and predict loan approval eligibility.")
    
    st.divider()
    st.subheader("📊 Model Information")
    st.markdown("""
    - **Model:** Support Vector Classifier (SVM)
    - **Kernel:** RBF (Radial Basis Function)
    - **Regularization (C):** 5.0
    - **Class Weight:** Balanced
    - **Accuracy:** ~87.3%
    - **Recall:** ~92.7%
    - **ROC-AUC:** ~88.8%
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
    # Prepare input DataFrame with exact training column names
    input_data = pd.DataFrame([{
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
    }])

    try:
        prediction = int(model.predict(input_data)[0])
        decision_score = float(model.decision_function(input_data)[0]) if hasattr(model, "decision_function") else 0.0

        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        if prediction == 1:
            st.balloons()
            st.markdown(
                '<div class="status-approved">✅ LOAN APPROVED (APROBADO)</div>', 
                unsafe_allow_html=True
            )
            st.success("The model predicts this loan application meets standard approval criteria and is eligible for disbursement.")
        else:
            st.markdown(
                '<div class="status-rejected">❌ LOAN REJECTED (NO APROBADO)</div>', 
                unsafe_allow_html=True
            )
            st.error("The model flags this application as elevated credit risk. Approval is not recommended under current terms.")
            
        # Key Indicators & Risk Assessment
        st.markdown("#### 🔍 Applicant Evaluation Summary")
        rcol1, rcol2, rcol3 = st.columns(3)
        
        with rcol1:
            ratio_status = "Healthy (< 30%)" if calculated_percent_income < 0.3 else "High Burden (≥ 30%)"
            st.metric("Loan / Income Ratio", f"{calculated_percent_income*100:.1f}%", ratio_status)
            
        with rcol2:
            score_status = "Excellent" if credit_score >= 720 else ("Good" if credit_score >= 650 else "Fair/Poor")
            st.metric("Credit Score", credit_score, score_status)
            
        with rcol3:
            default_status = "Clean Record" if previous_loan_defaults_on_file == "No" else "Default History Detected"
            st.metric("Historical Defaults", previous_loan_defaults_on_file, default_status, delta_color="inverse" if previous_loan_defaults_on_file == "Yes" else "normal")
            
        with st.expander("📋 View Sent Input Payload"):
            st.dataframe(input_data)
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

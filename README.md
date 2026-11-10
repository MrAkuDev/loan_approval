# 🏦 Loan Approval Prediction — Machine Learning Project

This repository contains an end-to-end Machine Learning practice to predict loan approval status (`loan_status`) based on applicant credit risk, financial indicators, and historical loan data.

The project includes an interactive web application built with **Streamlit** ready for one-click deployment on **Streamlit Community Cloud**.

---

## 🚀 Live Demo on Streamlit Cloud

When deployed to Streamlit Community Cloud:
- **Repository:** `MrAkuDev/loan_approval`
- **Branch:** `main`
- **Main file path:** `app.py`

---

## 🧠 Machine Learning Model & Performance

After exploratory data analysis (EDA), data cleaning, feature scaling, and one-hot encoding, multiple classification algorithms were benchmarked:
- Dummy Classifier (baseline)
- SGD Classifier
- Logistic Regression
- Support Vector Machine (Linear, Polynomial, and RBF Kernels)

The best-performing model was **SVM with an RBF (Radial Basis Function) Kernel**, optimized using `RandomizedSearchCV` with balanced class weights to maximize recall on minority approved/rejected cases.

### Final Test Metrics
- **Accuracy:** 87.33%
- **Precision:** 65.10%
- **Recall:** 92.70%
- **F1 Score:** 0.7649
- **ROC-AUC:** 0.9640

---

## 📁 Repository Structure

```text
├── app.py              # Streamlit web application
├── model.joblib        # Pre-trained Scikit-Learn Pipeline (Scaler + OneHotEncoder + SVM)
├── train_model.py      # Standalone training & model export script
├── loan_approval.ipynb # Jupyter notebook with full EDA, benchmarks, and visualizations
├── loan_data.csv       # Dataset (45,000 credit records)
├── requirements.txt    # Python dependencies for Streamlit Cloud
└── README.md           # Documentation
```

---

## 💻 Local Setup & Execution

### 1. Clone the repository
```bash
git clone git@github.com:MrAkuDev/loan_approval.git
cd loan_approval
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the model
```bash
python train_model.py
```

### 4. Run the Streamlit application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

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

## 🧠 Machine Learning Model: Stacking Ensemble Architecture

To achieve superior predictive power, this project implements a **Stacking Classifier (`StackingClassifier`)** combining three diverse algorithms as base estimators with an independent meta-learner:

1. **Gradient Boosting Classifier (`GradientBoostingClassifier`):** Captures non-linear feature interactions and high-order patterns.
2. **Support Vector Machine (`SVC` with RBF Kernel):** Maximizes margin separation in high-dimensional space.
3. **Logistic Regression (`LogisticRegression`):** Provides a robust linear baseline with calibrated probabilities.
4. **Final Meta-Estimador:** Logistic Regression combining out-of-fold predictions via 5-fold cross-validation.

### ⚙️ Automated Hyperparameter Tuning
Before stacking, each base model undergoes hyperparameter optimization (`GridSearchCV` and `RandomizedSearchCV`) with stratified cross-validation on the preprocessed feature space.

### Final Performance on Holdout Test Set (20%)
- **Accuracy:** 93.31%
- **Precision:** 88.79%
- **Recall:** 80.00%
- **F1 Score:** 84.17%
- **ROC-AUC:** 0.9763

### 🛡️ Two-Tier Credit Decision Architecture
To prevent high-risk loans from being approved through statistical extrapolation:
1. **Tier 1: Underwriting Knockout Policies:** Automatic decline if DTI > 45%, Annual Income < $12,000, Credit Score < 550, or Prior Defaults exist.
2. **Tier 2: Stacking ML Inference:** If hard policies pass, the Stacking Ensemble evaluates the applicant across 18 features with a conservative approval threshold ($\ge 65\%$ projected solvency).

---

## 📁 Repository Structure

```text
├── app.py              # Streamlit web application with two-tier decisioning
├── features.py         # Financial feature engineering & underwriting policy rules
├── model.joblib        # Pre-trained Stacking Ensemble pipeline (GB + SVM + LR)
├── train_model.py      # Standalone training script with cross-validation tuning
├── loan_approval.ipynb # Jupyter notebook with EDA, benchmarks, and research references
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
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

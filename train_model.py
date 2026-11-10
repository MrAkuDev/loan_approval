import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train_and_save():
    print("Loading loan_data.csv...")
    df = pd.read_csv("loan_data.csv")

    y = df["loan_status"]
    X = df.drop(columns=["loan_status"])

    # Split train/test (80/20) with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    categorical_features = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    ]

    numerical_features = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="passthrough"
    )

    # Best model tuned in loan_approval.ipynb: SVM RBF with C=5.0, gamma='auto', class_weight='balanced'
    model = Pipeline([
        ("preprocesamiento", preprocessor),
        ("svm", SVC(kernel="rbf", C=5.0, gamma="auto", class_weight="balanced", probability=True, random_state=42))
    ])

    print("Training SVM RBF model...")
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }

    print("Model Evaluation on Test Set:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Saving pipeline to model.joblib...")
    joblib.dump(model, "model.joblib")
    print("Model successfully saved to model.joblib!")

if __name__ == "__main__":
    train_and_save()

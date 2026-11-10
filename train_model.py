"""
train_model.py
==============
Entrenamiento de Ensamble Stacking con Ingeniería de Características Financieras.

Modelos Base:
  1. Gradient Boosting Classifier (Optimizado)
  2. Support Vector Machine con Kernel RBF
  3. Regresión Logística (L2)

Meta-Estimador:
  - Regresión Logística

Mejoras incorporadas:
  - 5 variables financieras de riesgo (features.py)
  - Acotamiento de apalancamiento para evitar extrapolación fuera de distribución
  - Búsqueda de hiperparámetros con Stratified Cross-Validation
"""

import os
import time
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from features import agregar_features

CSV_FILE = "loan_data.csv"
MODEL_FILE = "model.joblib"


def cargar_y_preparar_datos(ruta_csv=CSV_FILE):
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_csv}")
    
    print("=" * 70)
    print("1. CARGANDO Y PREPARANDO DATASET CON INGENIERÍA DE VARIABLES")
    print("=" * 70)
    df = pd.read_csv(ruta_csv)
    print(f"Dimensiones iniciales: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    # Aplicar ingeniería de características
    df_fe = agregar_features(df)
    print(f"Dimensiones tras feature engineering: {df_fe.shape[0]:,} filas x {df_fe.shape[1]} columnas")
    return df_fe


def partir_datos(df):
    print("\n" + "=" * 70)
    print("2. PARTICIÓN TRAIN / TEST (80% / 20%, Stratified)")
    print("=" * 70)
    y = df["loan_status"]
    X = df.drop(columns=["loan_status"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Muestras de entrenamiento (Train): {X_train.shape[0]:,}")
    print(f"Muestras de evaluación (Test):       {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


def crear_preprocesador():
    columnas_categoricas = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    ]

    # Incluye las columnas originales y las 5 nuevas variables de riesgo financiero
    columnas_numericas = [
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "credit_score",
        # Nuevas variables calculadas
        "loan_percent_income_clipped",
        "debt_burden_index",
        "residual_income",
        "credit_leverage_ratio",
        "extreme_dti_flag",
    ]

    print("\n" + "=" * 70)
    print("3. CONSTRUCCIÓN DEL PREPROCESADOR")
    print("=" * 70)
    print(f"Numéricas ({len(columnas_numericas)}): {columnas_numericas}")
    print(f"Categóricas ({len(columnas_categoricas)}): {columnas_categoricas}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_numericas),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), columnas_categoricas),
        ],
        remainder="passthrough"
    )
    return preprocessor


def optimizar_modelos_base(X_train_trans, y_train):
    print("\n" + "=" * 70)
    print("4. OPTIMIZACIÓN DE HIPERPARÁMETROS PARA MODELOS BASE")
    print("=" * 70)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # A) Logistic Regression
    print("\n[A] Regresión Logística...")
    param_grid_lr = {
        "C": [0.1, 1.0, 5.0],
        "class_weight": [None],
        "solver": ["lbfgs"],
        "max_iter": [1000],
    }
    grid_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid=param_grid_lr, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid_lr.fit(X_train_trans, y_train)
    best_lr = grid_lr.best_estimator_
    print(f"    Parámetros LR: {grid_lr.best_params_} | ROC-AUC: {grid_lr.best_score_:.4f}")

    # B) SVM RBF
    print("\n[B] SVM con Kernel RBF...")
    sample_size = min(15000, len(y_train))
    sample_indices = np.random.RandomState(42).choice(len(y_train), size=sample_size, replace=False)
    
    param_dist_svm = {
        "C": [1.0, 5.0],
        "gamma": ["scale", "auto"],
    }
    search_svm = RandomizedSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_distributions=param_dist_svm,
        n_iter=3,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )
    search_svm.fit(X_train_trans[sample_indices], y_train.iloc[sample_indices])
    best_svm = SVC(kernel="rbf", random_state=42, **search_svm.best_params_)
    print(f"    Parámetros SVM: {search_svm.best_params_} | ROC-AUC: {search_svm.best_score_:.4f}")

    # C) Gradient Boosting
    print("\n[C] Gradient Boosting...")
    param_dist_gb = {
        "n_estimators": [100, 150],
        "learning_rate": [0.1, 0.15],
        "max_depth": [4, 5],
        "subsample": [0.8, 1.0],
    }
    search_gb = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_distributions=param_dist_gb,
        n_iter=4,
        cv=cv,
        scoring="roc_auc",
        random_state=42,
        n_jobs=-1
    )
    search_gb.fit(X_train_trans, y_train)
    best_gb = search_gb.best_estimator_
    print(f"    Parámetros GB: {search_gb.best_params_} | ROC-AUC: {search_gb.best_score_:.4f}")

    return {
        "LogisticRegression": best_lr,
        "SVM_RBF": best_svm,
        "GradientBoosting": best_gb,
    }


def construir_y_entrenar_stacking(preprocessor, modelos_base, X_train, y_train):
    print("\n" + "=" * 70)
    print("5. ENSAMBLE STACKING (GB + SVM RBF + Logistic Regression)")
    print("=" * 70)

    estimadores = [
        ("gb", modelos_base["GradientBoosting"]),
        ("svm", modelos_base["SVM_RBF"]),
        ("lr", modelos_base["LogisticRegression"]),
    ]

    meta_clasificador = LogisticRegression(random_state=42, max_iter=1000)

    stacking_clf = StackingClassifier(
        estimators=estimadores,
        final_estimator=meta_clasificador,
        cv=5,
        n_jobs=-1
    )

    pipeline_completo = Pipeline([
        ("preprocesamiento", preprocessor),
        ("stacking", stacking_clf)
    ])

    print("Entrenando StackingClassifier con validación cruzada (5 folds)...")
    t0 = time.time()
    pipeline_completo.fit(X_train, y_train)
    print(f"✅ Ajuste completado en {time.time() - t0:.2f} segundos.")
    return pipeline_completo


def evaluar_rendimiento(pipeline_stacking, X_test, y_test):
    print("\n" + "=" * 70)
    print("6. EVALUACIÓN FINAL EN TEST SET (20% Holdout)")
    print("=" * 70)
    y_pred = pipeline_stacking.predict(X_test)
    y_proba = pipeline_stacking.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precisión: {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusión FINAL:")
    print(f"{'':>25} {'Predicho No (0)':>16} {'Predicho Si (1)':>16}")
    print(f"{'Real No Aprobado (0)':>25} {cm[0, 0]:>16,d} {cm[0, 1]:>16,d}")
    print(f"{'Real Aprobado (1)':>25} {cm[1, 0]:>16,d} {cm[1, 1]:>16,d}")

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No Aprobado (0)", "Aprobado (1)"]))


def guardar_modelo(pipeline, ruta_salida=MODEL_FILE):
    print("\n" + "=" * 70)
    print(f"7. GUARDANDO MODELO SERIALIZADO EN '{ruta_salida}'")
    print("=" * 70)
    joblib.dump(pipeline, ruta_salida)
    tamano_mb = os.path.getsize(ruta_salida) / (1024 * 1024)
    print(f"✅ Pipeline Stacking guardado exitosamente ({tamano_mb:.2f} MB).")


def main():
    df = cargar_y_preparar_datos(CSV_FILE)
    X_train, X_test, y_train, y_test = partir_datos(df)

    preprocessor = crear_preprocesador()
    print("\nAjustando preprocesador sobre X_train...")
    X_train_trans = preprocessor.fit_transform(X_train)

    modelos_optimizados = optimizar_modelos_base(X_train_trans, y_train)
    pipeline_stacking = construir_y_entrenar_stacking(preprocessor, modelos_optimizados, X_train, y_train)
    evaluar_rendimiento(pipeline_stacking, X_test, y_test)
    guardar_modelo(pipeline_stacking, MODEL_FILE)


if __name__ == "__main__":
    main()

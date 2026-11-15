"""Pestaña 1: Información general del proyecto, base de datos y modelos integrados."""

import streamlit as st


def render_tab_info(eval_results: dict) -> None:
    """Renderiza el contenido de la pestaña de información general del proyecto."""
    st.header("Información General del Proyecto")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("🎯 Problema y Objetivo")
        st.markdown(
            """
        - **Descripción del problema:** Las instituciones financieras evalúan múltiples variables
          demográficas, financieras e historiales crediticios para predecir si una solicitud presenta
          riesgo de incumplimiento (`1 = Incumplimiento`) o solvencia (`0 = Sin Incumplimiento`). Se trata de un problema de
          **clasificación binaria supervisada** con desbalance de clases (~78.1% vs ~21.9%).
        - **Objetivo de la aplicación:** Integrar en una interfaz web los **tres modelos con
          mejor desempeño**, permitiendo ingresar instancias individuales o lotes en CSV para generar,
          comparar e interpretar sus predicciones en vivo.
        """
        )

        st.subheader("📋 Ficha Académica")
        st.markdown(
            """
        | Campo | Detalle |
        | :--- | :--- |
        | **Proyecto** | Clasificación de Riesgo Crediticio (*Credit Risk Dataset*) |
        | **Alumno / Matrícula** | Joel Arturo Álvarez Hernández (`403233`) |
        | **Institución** | Universidad Autónoma de Chihuahua (UACH) |
        | **Facultad / Materia** | Facultad de Ingeniería — Machine Learning (Proyecto Final) |
        | **Asesora / Docente** | Dra. Olanda Prieto Ordaz |
        """
        )

    with col_p2:
        ds_info = eval_results["dataset"]
        cd = ds_info.get("class_distribution", {})
        c0_n = cd.get("0_no_default", cd.get("0_no_aprobado", 0))
        c1_n = cd.get("1_default", cd.get("1_aprobado", 0))
        tot_n = max(ds_info.get("total_rows", c0_n + c1_n), 1)
        st.subheader("🗄️ Base de Datos Utilizada (`credit_risk_dataset.csv`)")
        st.markdown(
            f"""
        - **Origen:** Kaggle — *Credit Risk Dataset* (`laotse/credit-risk-dataset`).
        - **Registros limpios:** **{ds_info['total_rows']:,}** filas × **12** columnas (desde **{ds_info.get('raw_rows', ds_info['total_rows']):,}** brutos tras eliminar {ds_info.get('duplicates_removed', 0)} duplicados y {ds_info.get('outliers_removed', 0)} atípicos imposibles).
        - **Partición utilizada:** 80% Entrenamiento (**{ds_info['train_rows']:,}**) / 20% Prueba (**{ds_info['test_rows']:,}**) estratificada (`random_state=42`).
        - **Distribución de la variable objetivo (`loan_status`):**
          - Clase `0` (**Sin Incumplimiento**): **{c0_n:,}** registros ({c0_n / tot_n * 100:.2f}%)
          - Clase `1` (**Incumplimiento**): **{c1_n:,}** registros ({c1_n / tot_n * 100:.2f}%)
        - **Preprocesamiento idéntico en los 3 pipelines:**
          - `SimpleImputer(strategy='median')` + `StandardScaler()` sobre las **7 variables numéricas**.
          - `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(drop='first', handle_unknown='ignore')` sobre las **4 variables categóricas**.
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

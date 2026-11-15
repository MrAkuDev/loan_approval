"""Pestaña 5: Guía de interpretación de resultados y limitaciones del modelo."""

import streamlit as st


def render_tab_interp() -> None:
    """Renderiza la tarjeta 'Acerca de' con la guía de interpretación y limitaciones."""
    with st.container(border=True):
        st.markdown('<div class="m3-section-title">Acerca de</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="m3-section-subtitle">Guía de Interpretación de Resultados y Limitaciones del Modelo</div>',
            unsafe_allow_html=True,
        )

        st.subheader("1. ¿Qué representa la clase predicha?")
        st.markdown(
            """
        - **Clase `0` — Sin Incumplimiento (`0`):** Indica que, según los patrones aprendidos en los datos históricos
          de `credit_risk_dataset.csv`, la solicitud presenta bajo riesgo de incumplimiento y un perfil solvente.
        - **Clase `1` — Incumplimiento (`1`):** Indica que la solicitud se asemeja a perfiles con incumplimiento
          crediticio en la base de entrenamiento.
        """
        )

        st.subheader("2. ¿Cómo se interpretan las probabilidades y puntuaciones de confianza?")
        st.markdown(
            """
        - **Probabilidad estimada (`predict_proba`):**
          - En **Regresión Logística**, la probabilidad proviene directamente de la función sigmoide aplicada a la combinación lineal de características.
          - En **SVM RBF**, la probabilidad se calcula mediante calibración de Platt (`probability=True`) sobre la distancia al hiperplano separador.
          - En **Random Forest**, la probabilidad corresponde a la proporción promedio de votos entre los árboles de decisión del ensamble.
          - Un valor cercano a **100%** en Clase `1` significa alta probabilidad de **Incumplimiento (1)**; un valor cercano a **0%** en Clase `1` (es decir, ~100% en Clase `0`) indica alta confianza de solvencia; valores cercanos a **50%** indican incertidumbre en la frontera de decisión, donde los 3 modelos pueden discrepar.
        - **Score de decisión (`decision_function`):** En SVM y modelos lineales representa la distancia firmada respecto al hiperplano de separación (`>= 0` corresponde a Clase `1: Incumplimiento` y `< 0` a Clase `0: Sin Incumplimiento`).
        """
        )

        st.subheader("3. Limitaciones de las predicciones generadas")
        st.warning(
            """
        1. **Preprocesamiento e imputación de valores faltantes:** El archivo original `credit_risk_dataset.csv` (`32,581` registros) presenta `165` filas duplicadas, `7` valores físicamente imposibles (`person_age > 100` o `person_emp_length > 60`), y valores nulos en `person_emp_length` (~2.7%) y `loan_int_rate` (~9.5%). Tras la limpieza determinista (`32,409` registros), los nulos se imputan mediante mediana/moda dentro del `Pipeline` usando exclusivamente `X_train`.
        2. **Desbalance de clases (~78.1% `0` vs ~21.9% `1`):** El modelo prioriza el desempeño global (`F1` y `ROC-AUC`), por lo que la clase minoritaria (`1: Incumplimiento`) puede presentar menor *recall* que la clase mayoritaria (`0: Sin Incumplimiento`).
        3. **Dependencia del grado crediticio (`loan_grade`) y carga financiera (`loan_percent_income`):** En esta base de riesgo real, los grados `D–G`, las tasas de interés elevadas y una razón préstamo-ingreso alta incrementan fuertemente el riesgo de incumplimiento.
        4. **Riesgo de extrapolación fuera de distribución:** Valores extremos no presentes en el entrenamiento (por ejemplo, `loan_percent_income > 0.70`, ingresos atípicos o edades extremas) pueden generar predicciones inestables.
        5. **Uso como herramienta de apoyo:** Los resultados constituyen una estimación estadística basada en el proyecto académico y no sustituyen el análisis financiero, legal ni normativo de una entidad bancaria real.
        """
        )

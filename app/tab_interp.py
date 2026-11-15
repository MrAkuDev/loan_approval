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
        - **Clase `1` — Aprobado (`Loan Approved`):** Indica que, según los patrones aprendidos en los datos históricos
          de `loan_data.csv`, la solicitud presenta características compatibles con solicitudes aprobadas.
        - **Clase `0` — No Aprobado (`Loan Not Approved`):** Indica que la solicitud se asemeja a perfiles rechazados
          en la base de entrenamiento.
        """
        )

        st.subheader("2. ¿Cómo se interpretan las probabilidades y puntuaciones de confianza?")
        st.markdown(
            """
        - **Probabilidad estimada (`predict_proba`):**
          - En **Regresión Logística**, la probabilidad proviene directamente de la función sigmoide aplicada a la combinación lineal de características.
          - En **SVM RBF**, la probabilidad se calcula mediante calibración de Platt (`probability=True`) sobre la distancia al hiperplano separador.
          - En **Random Forest**, la probabilidad corresponde a la proporción promedio de votos entre los árboles de decisión del ensamble.
          - Un valor cercano a **100%** en Clase `1` significa alta confianza estadística de aprobación; un valor cercano a **50%** indica incertidumbre cerca de la frontera de decisión, donde los 3 modelos pueden discrepar.
        - **Score de decisión (`decision_function`):** En SVM y modelos lineales representa la distancia firmada respecto al hiperplano de separación (`>= 0` corresponde a Clase `1` y `< 0` a Clase `0`).
        """
        )

        st.subheader("3. Limitaciones de las predicciones generadas")
        st.warning(
            """
        1. **Naturaleza sintética de la base de datos (`SMOTENC`):** `loan_data.csv` es un conjunto aumentado sintéticamente a partir de datos de riesgo crediticio. Como se observó en el análisis exploratorio del notebook (`loan_approval.ipynb`), entre registros sin mora previa, variables como `loan_percent_income` y `loan_int_rate` muestran correlación positiva con la clase `1`, lo que refleja la estructura del dataset sintético y no necesariamente una regla bancaria real de bajo riesgo.
        2. **Desbalance de clases (77.78% `0` vs 22.22% `1`):** El modelo prioriza el desempeño global (`F1` y `ROC-AUC`), por lo que la clase minoritaria (`1`) presenta menor *recall* que la clase mayoritaria (`0`).
        3. **Regla estricta aprendida sobre incumplimientos previos:** En los 45,000 registros de `loan_data.csv`, el 100% de los solicitantes con `previous_loan_defaults_on_file = "Yes"` (22,858 casos) pertenecen a la clase `0`. Por ello, cualquier entrada con incumplimiento previo será clasificada como `0` con altísima probabilidad.
        4. **Riesgo de extrapolación fuera de distribución:** Valores extremos no presentes en el entrenamiento (por ejemplo, `loan_percent_income > 0.66`, `loan_amnt > person_income` o edades atípicas) pueden generar predicciones inestables.
        5. **Uso como herramienta de apoyo:** Los resultados constituyen una estimación estadística basada en el proyecto académico y no sustituyen el análisis financiero, legal ni normativo de una entidad bancaria real.
        """
        )

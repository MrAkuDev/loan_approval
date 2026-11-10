"""
features.py
===========
Módulo centralizado de:
 1. Ingeniería de Características Financieras (Feature Engineering)
 2. Reglas Duras de Política Crediticia (Underwriting Knockout Rules)
"""

import numpy as np
import pandas as pd


def agregar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables financieras avanzadas y restringe el apalancamiento
    extremo para evitar extrapolaciones fuera de distribución en producción.
    """
    d = df.copy()

    # 1. Acotar el ratio deuda/ingreso al rango visto en entrenamiento [0.0, 0.65]
    #    Evita que valores extremos (ej. 200%) extrapolen artificialmente hacia aprobación.
    d["loan_percent_income_clipped"] = np.clip(d["loan_percent_income"].astype(float), 0.0, 0.65)

    # 2. Índice de carga financiera combinada: (Tasa de Interés x DTI acotado)
    d["debt_burden_index"] = d["loan_percent_income_clipped"] * d["loan_int_rate"].astype(float)

    # 3. Ingreso residual anual estimado tras el servicio de la deuda del crédito
    d["residual_income"] = d["person_income"].astype(float) - (
        d["loan_amnt"].astype(float) * (1.0 + d["loan_int_rate"].astype(float) / 100.0)
    )

    # 4. Ratio de solvencia crediticia sobre apalancamiento
    d["credit_leverage_ratio"] = d["credit_score"].astype(float) / (
        d["loan_percent_income_clipped"] * 1000.0 + 1.0
    )

    # 5. Indicador binario de sobreendeudamiento severo (> 40% del ingreso)
    d["extreme_dti_flag"] = (d["loan_percent_income"].astype(float) > 0.40).astype(float)

    return d


def evaluar_politicas_duras(row) -> tuple[bool, list[str], list[str]]:
    """
    Aplica las Reglas Duras de Suscripción (Underwriting Knockout Rules) estándar en la banca.
    Si se viola alguna regla, la solicitud es rechazada de inmediato sin depender
    únicamente de la probabilidad estadística del modelo ML.

    Retorna:
      (es_rechazo_automatico, motivos_rechazo, recomendaciones)
    """
    motivos = []
    recomendaciones = []

    income = float(row.get("person_income", 0))
    loan_amnt = float(row.get("loan_amnt", 0))
    dti = float(row.get("loan_percent_income", 0))
    credit_score = int(row.get("credit_score", 0))
    prev_defaults = str(row.get("previous_loan_defaults_on_file", "No")).strip()

    # Regla 1: Historial de mora previo (En el dataset, 100% de morosos previos fueron rechazados)
    if prev_defaults.lower() in ["yes", "y", "si", "sí"]:
        motivos.append(
            "🛑 **Regla Dura de Política:** Registro de incumplimiento o mora previa en el historial. "
            "La política bancaria exige solvencia histórica limpia."
        )
        recomendaciones.append(
            "📌 **Rehabilitar historial:** Liquidar deudas pendientes y mantener un récord impecable durante 12-24 meses."
        )

    # Regla 2: Carga de deuda excesiva (> 45% del ingreso) o préstamo superior al ingreso anual
    if dti > 0.45 or loan_amnt > income:
        pct = dti * 100
        monto_max = income * 0.30
        motivos.append(
            f"🛑 **Regla Dura de Apalancamiento:** El préstamo representa el **{pct:.1f}%** de su ingreso anual "
            f"(máximo permitido por política: **45.0%**)."
        )
        recomendaciones.append(
            f"📌 **Ajustar monto solicitado:** Reduzca el monto del préstamo a un máximo de **${monto_max:,.0f} USD** "
            "o sume los ingresos de un co-titular."
        )

    # Regla 3: Ingreso mínimo de subsistencia (< $12,000 USD anuales)
    if income < 12000.0:
        motivos.append(
            f"🛑 **Regla Dura de Ingreso Mínimo:** Su ingreso anual (${income:,.0f} USD) está por debajo del umbral "
            "mínimo de subsistencia requerido para créditos individuales ($12,000 USD)."
        )
        recomendaciones.append(
            "📌 **Comprobación de ingresos:** Presente comprobantes de ingresos adicionales o aplique con un aval solvente."
        )

    # Regla 4: Score crediticio en zona de exclusión (< 550 pts - Deep Subprime)
    if credit_score < 550:
        motivos.append(
            f"🛑 **Regla Dura de Score Crediticio:** Su puntaje ({credit_score} pts) se ubica en el rango de exclusión "
            "automática (< 550 puntos)."
        )
        recomendaciones.append(
            "📌 **Elevar Score:** Pague a tiempo sus compromisos corrientes y baje la utilización de tarjetas por debajo del 30%."
        )

    es_rechazo = len(motivos) > 0
    return es_rechazo, motivos, recomendaciones

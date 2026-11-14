"""Pestaña 3: Upload Database — predicción por lotes sobre archivos CSV."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score

from app.services import load_sample_template
from app.validation import validate_uploaded_dataframe


def render_tab_upload(models: dict) -> None:
    """Renderiza la pestaña de carga y predicción masiva por CSV."""
    st.header("📂 Upload Database (Predicción por Lotes en CSV)")
    st.markdown(
        "Sube un archivo `.csv` con las 13 columnas de entrada en formato compatible con `loan_data.csv` "
        "(puede incluir opcionalmente la columna real `loan_status` para comparar aciertos). "
        "Los 3 modelos procesarán el archivo aplicando sus pipelines de preprocesamiento."
    )

    sample_template = load_sample_template(nrows=5)
    if sample_template is not None:
        st.download_button(
            label="⬇️ Descargar CSV de ejemplo (5 registros reales de loan_data.csv)",
            data=sample_template.to_csv(index=False).encode("utf-8"),
            file_name="ejemplo_upload_database.csv",
            mime="text/csv",
        )

    uploaded_file = st.file_uploader("Upload Database (.csv)", type=["csv"], key="upload_database_csv")
    if uploaded_file is None:
        return

    try:
        df_raw = pd.read_csv(uploaded_file)
        st.write(f"Archivo cargado: **{uploaded_file.name}** ({len(df_raw):,} filas × {df_raw.shape[1]} columnas)")

        is_valid, val_errs, df_features = validate_uploaded_dataframe(df_raw)
        if not is_valid:
            for e in val_errs:
                st.error(f"❌ Validación fallida: {e}")
            return

        st.success("✅ Validación exitosa: columnas, tipos de datos y categorías verificadas.")

        df_out = df_raw.copy()
        for m_name, pipe in models.items():
            preds_arr = pipe.predict(df_features)
            probs_arr = pipe.predict_proba(df_features)[:, 1]
            short_key = m_name.lower().replace(" ", "_")
            df_out[f"pred_{short_key}"] = preds_arr
            df_out[f"prob1_{short_key}"] = np.round(probs_arr, 4)

        pred_cols = [f"pred_{k.lower().replace(' ', '_')}" for k in models.keys()]
        df_out["consenso_3_modelos"] = df_out[pred_cols].nunique(axis=1).map(
            lambda n: "Unánime (3/3)" if n == 1 else "Discrepancia (2/1)"
        )

        st.subheader("📈 Resumen de Predicciones sobre la Base Cargada")
        u_cols = st.columns(4)
        for idx, (m_name, col_name) in enumerate(zip(models.keys(), pred_cols)):
            n_app = int((df_out[col_name] == 1).sum())
            pct_app = (n_app / len(df_out)) * 100
            u_cols[idx].metric(f"{m_name} — Aprobados", f"{n_app:,} / {len(df_out):,}", f"{pct_app:.1f}%")
        unanimous_cnt = int((df_out["consenso_3_modelos"] == "Unánime (3/3)").sum())
        u_cols[3].metric(
            "Consenso Unánime",
            f"{unanimous_cnt:,} / {len(df_out):,}",
            f"{(unanimous_cnt / len(df_out)) * 100:.1f}%",
        )

        if "loan_status" in df_raw.columns and df_raw["loan_status"].isin([0, 1]).all():
            st.info("ℹ️ La base subida contiene la columna real `loan_status`. Desempeño de los 3 modelos en esta muestra:")
            batch_metrics = [
                {
                    "Modelo": m_name,
                    "Accuracy en CSV": round(float(accuracy_score(df_raw["loan_status"], df_out[col_name])), 4),
                    "F1-Score en CSV": round(float(f1_score(df_raw["loan_status"], df_out[col_name], zero_division=0)), 4),
                }
                for m_name, col_name in zip(models.keys(), pred_cols)
            ]
            st.dataframe(pd.DataFrame(batch_metrics), use_container_width=True, hide_index=True)

        st.subheader("📋 Vista Previa de Resultados (primeras 100 filas)")
        st.dataframe(df_out.head(100), use_container_width=True)

        st.download_button(
            label="📥 Descargar CSV con Predicciones de los 3 Modelos",
            data=df_out.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_3_modelos.csv",
            mime="text/csv",
        )
    except Exception as exc:
        st.error(f"Error al procesar el archivo CSV: {exc}")

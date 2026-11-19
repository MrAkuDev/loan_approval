"""Pestaña 3: Upload Database — predicción por lotes y análisis completo estilo Evaluación Experimental."""

import hashlib

import numpy as np
import pandas as pd
import streamlit as st

from app.charts import (
    plot_confusion_matrices_row,
    plot_test_metrics_and_roc,
)
from app.config import FEATURE_ORDER
from app.frontend import render_threshold_selector, show_inline_loading
from app.services import (
    balance_binary_target,
    compute_batch_eval_metrics,
    load_sample_template,
)
from app.tab_eval import (
    render_correlation_section,
    render_decision_boundary_section,
    render_eda_section,
)
from app.validation import validate_uploaded_dataframe


def _run_batch_inference(models: dict, df_raw: pd.DataFrame, df_features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ejecuta la inferencia por lotes sobre los 3 modelos mostrando animación y progreso."""
    df_out = df_raw.copy()
    total_models = len(models)

    with st.status(f"Cargando y procesando {len(df_features):,} registros con {total_models} modelos...", expanded=True) as status:
        progress_bar = st.progress(0, text="Iniciando predicción por lotes...")
        for idx, (m_name, pipe) in enumerate(models.items(), start=1):
            progress_bar.progress((idx - 1) / total_models, text=f"[{idx}/{total_models}] Ejecutando {m_name}...")
            st.write(f"⏳ Procesando **{m_name}** ({idx}/{total_models})...")
            preds_arr = pipe.predict(df_features)
            probs_arr = pipe.predict_proba(df_features)[:, 1]
            short_key = m_name.lower().replace(" ", "_")
            df_out[f"pred_{short_key}"] = preds_arr
            df_out[f"prob1_{short_key}"] = np.round(probs_arr, 4)
            st.write(f"✅ **{m_name}** completado.")
            progress_bar.progress(idx / total_models, text=f"[{idx}/{total_models}] {m_name} completado")

        status.update(
            label=f"✅ Predicción completada para {len(df_out):,} registros.",
            state="complete",
            expanded=False,
        )

    pred_cols = [f"pred_{k.lower().replace(' ', '_')}" for k in models.keys()]
    df_out["consenso_3_modelos"] = df_out[pred_cols].nunique(axis=1).map(
        lambda n: "Unánime (3/3)" if n == 1 else "Discrepancia (2/1)"
    )
    return df_out, pred_cols


def _build_upload_eda_source(
    df_raw: pd.DataFrame,
    df_out: pd.DataFrame,
    model_names: list[str],
    pred_cols: list[str],
    has_ground_truth: bool,
    file_sig: str,
) -> tuple[pd.DataFrame, str]:
    """Construye el DataFrame con columna `loan_status` (real o predicha) para EDA y correlaciones."""
    df_base = df_out[FEATURE_ORDER].copy()
    if has_ground_truth:
        df_base["loan_status"] = df_raw["loan_status"].astype(int).values
        return df_base, "columna real `loan_status` del CSV"

    target_choice = st.radio(
        "El CSV no incluye `loan_status`. Etiqueta de referencia para las gráficas EDA y Correlación:",
        options=["Consenso (Voto Mayoritario)"] + model_names,
        index=0,
        horizontal=True,
        key=f"upload_target_src_{file_sig}",
    )
    if target_choice == "Consenso (Voto Mayoritario)":
        df_base["loan_status"] = (df_out[pred_cols].mean(axis=1) >= 0.5).astype(int).values
    else:
        short_key = target_choice.lower().replace(" ", "_")
        df_base["loan_status"] = df_out[f"pred_{short_key}"].astype(int).values
    return df_base, f"predicción de `{target_choice}`"


def _render_model_comparison_section(
    models: dict,
    df_out: pd.DataFrame,
    df_eda: pd.DataFrame,
    model_names: list[str],
    pred_cols: list[str],
    has_ground_truth: bool,
    batch_eval_metrics: dict[str, dict] | None,
    use_balanced: bool,
    file_sig: str,
    eval_results: dict | None,
) -> None:
    """Renderiza Sección 3: tabla comparativa y fronteras de decisión sobre el CSV."""
    st.subheader("3. Desempeño Comparativo y Fronteras de Decisión sobre el CSV Cargado")
    majority_preds = (df_out[pred_cols].mean(axis=1) >= 0.5).astype(int).to_numpy()
    summary_rows = []
    for m_name, pred_col in zip(model_names, pred_cols):
        short_key = m_name.lower().replace(" ", "_")
        n_def = int((df_out[pred_col] == 1).sum())
        n_ok = int((df_out[pred_col] == 0).sum())
        mean_prob = float(df_out[f"prob1_{short_key}"].mean())

        if has_ground_truth and batch_eval_metrics is not None:
            m_vals = batch_eval_metrics[m_name]
            summary_rows.append(
                {
                    "Modelo": m_name,
                    "Sin Incumplimiento (0)": f"{n_ok:,} ({(n_ok / len(df_out)) * 100:.1f}%)",
                    "Incumplimiento (1)": f"{n_def:,} ({(n_def / len(df_out)) * 100:.1f}%)",
                    "Accuracy": m_vals["accuracy"],
                    "Precision": m_vals["precision"],
                    "Recall": m_vals["recall"],
                    "F1-Score": m_vals["f1"],
                    "ROC-AUC": m_vals["roc_auc"] if m_vals["roc_auc"] is not None else "N/A",
                }
            )
        else:
            agree_pct = float((df_out[pred_col].to_numpy(dtype=int) == majority_preds).mean() * 100.0)
            summary_rows.append(
                {
                    "Modelo": m_name,
                    "Sin Incumplimiento (0)": f"{n_ok:,} ({(n_ok / len(df_out)) * 100:.1f}%)",
                    "Incumplimiento (1)": f"{n_def:,} ({(n_def / len(df_out)) * 100:.1f}%)",
                    "Prob. Media P(Incumplimiento=1)": round(mean_prob, 4),
                    "Acuerdo con Voto Mayoritario": f"{agree_pct:.1f}%",
                }
            )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    render_decision_boundary_section(
        df_source=df_eda,
        models=models,
        eval_results=eval_results,
        dataset_tag=f"upload_{file_sig}",
        source_label=f"CSV Cargado ({len(df_eda):,} registros)",
        balanced_tag=use_balanced,
    )


def _render_batch_results(
    models: dict,
    df_raw: pd.DataFrame,
    df_out: pd.DataFrame,
    pred_cols: list[str],
    file_name: str,
    file_sig: str,
    eval_results: dict | None,
) -> None:
    """Renderiza el reporte experimental completo usando el CSV cargado y sus predicciones."""
    model_names = list(models.keys())
    has_ground_truth = "loan_status" in df_raw.columns and df_raw["loan_status"].isin([0, 1]).all()

    st.markdown(f"### 📊 Evaluación sobre Base Cargada: `{file_name}`")
    df_base, target_desc = _build_upload_eda_source(
        df_raw, df_out, model_names, pred_cols, has_ground_truth, file_sig
    )

    both_classes_in_eda = df_base["loan_status"].nunique() == 2
    use_balanced = False
    if both_classes_in_eda:
        use_balanced = st.checkbox(
            "Balancear clases 50% / 50% dentro del CSV cargado (igualar Sin Incumplimiento 0 vs Incumplimiento 1 para EDA)",
            value=False,
            key=f"upload_bal_{file_sig}",
        )
    df_eda = balance_binary_target(df_base) if use_balanced else df_base

    n_total = len(df_eda)
    n_c0 = int((df_eda["loan_status"] == 0).sum())
    n_c1 = int((df_eda["loan_status"] == 1).sum())
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Registros activos (CSV)", f"{n_total:,}")
    mc2.metric("Clase 0 — Sin Incumplimiento", f"{n_c0:,} ({(n_c0 / max(n_total, 1)) * 100:.1f}%)")
    mc3.metric("Clase 1 — Incumplimiento", f"{n_c1:,} ({(n_c1 / max(n_total, 1)) * 100:.1f}%)")
    st.caption(f"Fuente de etiquetas en las gráficas EDA y Correlación: {target_desc}.")

    # 1. Mapa de Calor de Correlaciones
    corr_sub = (
        f"Correlación de Pearson calculada sobre `{file_name}` (**{n_total:,} registros**) usando {target_desc} "
        "(**valores positivos en rojo** se asocian con **Clase 1: Incumplimiento**; **valores negativos en verde** con **Clase 0: Sin Incumplimiento**)."
    )
    render_correlation_section(
        df_source=df_eda,
        eval_results=eval_results,
        key_prefix=f"upload_{file_sig}",
        subtitle=corr_sub,
    )

    # 2. Análisis Exploratorio de Datos (EDA) + Vista 3D + Diagnósticos
    eda_mode_tag = "Balanceado 50/50" if use_balanced else "Datos Cargados"
    render_eda_section(
        df_eda,
        models,
        section_title=f"2. Análisis Exploratorio de Datos (EDA — `{file_name}`, {eda_mode_tag})",
        description=(
            f"Comportamiento de las principales características respecto a `loan_status` (**0 = Sin Incumplimiento**, **1 = Incumplimiento**) "
            f"en los **{n_total:,} registros** activos de `{file_name}`."
        ),
    )

    # 3. Métricas / Comparativa de Modelos + Fronteras de Decisión 2D
    batch_eval_metrics = None
    has_both_gt = False
    if has_ground_truth:
        batch_eval_metrics, has_both_gt = compute_batch_eval_metrics(model_names, df_raw["loan_status"], df_out)

    _render_model_comparison_section(
        models=models,
        df_out=df_out,
        df_eda=df_eda,
        model_names=model_names,
        pred_cols=pred_cols,
        has_ground_truth=has_ground_truth,
        batch_eval_metrics=batch_eval_metrics,
        use_balanced=use_balanced,
        file_sig=file_sig,
        eval_results=eval_results,
    )

    # 4 y 5. Gráficas Comparativas / ROC y Matrices de Confusión (cuando existe loan_status real)
    if has_ground_truth and batch_eval_metrics is not None:
        if has_both_gt:
            st.subheader("4. Visualización Comparativa en la Base Cargada (Métricas y Curvas ROC)")
            plot_test_metrics_and_roc(batch_eval_metrics, model_names)
        else:
            st.info("ℹ️ La columna `loan_status` del CSV contiene una sola clase; se requieren ambas clases (`0` y `1`) para trazar curvas ROC.")

        st.subheader("5. Matrices de Confusión — Desempeño Real sobre el CSV Cargado")
        gt_subtitles = {
            m: f"CSV Real | Acc: {batch_eval_metrics[m]['accuracy']:.4f} | F1: {batch_eval_metrics[m]['f1']:.4f}"
            for m in model_names
        }
        plot_confusion_matrices_row(batch_eval_metrics, model_names, cmap="Blues", subtitle_map=gt_subtitles)
    else:
        st.info(
            "ℹ️ El CSV cargado no incluye la columna real `loan_status`; "
            "las secciones **4 (Métricas y Curvas ROC)** y **5 (Matrices de Confusión reales)** se muestran automáticamente cuando el archivo incluye `loan_status`."
        )

    # 6. Vista Previa y Descarga
    st.subheader("6. Vista Previa de Resultados (primeras 100 filas) y Descarga")
    st.dataframe(df_out.head(100), use_container_width=True)
    st.download_button(
        label="📥 Descargar CSV con Predicciones de los 3 Modelos",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="predicciones_3_modelos.csv",
        mime="text/csv",
        key=f"upload_dl_{file_sig}",
    )


def render_tab_upload(models: dict, eval_results: dict | None = None) -> None:
    """Renderiza la pestaña de carga, validación, ejecución bajo demanda y evaluación estilo Tab 4."""
    with st.container(border=True):
        st.markdown('<div class="m3-section-title">Try yours — Upload Database (CSV)</div>', unsafe_allow_html=True)
        st.markdown(
            "Sube un archivo `.csv` con las 11 columnas de entrada en formato compatible con `credit_risk_dataset.csv` "
            "(puede incluir opcionalmente la columna real `loan_status` para evaluar métricas, curvas ROC y matrices de confusión). "
            "Pulsa **Predict** para procesar el lote con los 3 modelos y visualizar el análisis completo."
        )

        sample_template = load_sample_template(nrows=5)
        if sample_template is not None:
            st.download_button(
                label="⬇️ Descargar CSV de ejemplo (5 registros reales de credit_risk_dataset.csv)",
                data=sample_template.to_csv(index=False).encode("utf-8"),
                file_name="ejemplo_upload_database.csv",
                mime="text/csv",
            )

        uploaded_file = st.file_uploader("Upload Database (.csv)", type=["csv"], key="upload_database_csv")

        df_raw = None
        df_features = None
        file_name = None
        file_sig = None
        val_warnings = []

        try:
            if uploaded_file is not None:
                raw_bytes = uploaded_file.getvalue()
                file_hash = hashlib.md5(raw_bytes).hexdigest()[:10]
                file_sig = f"{uploaded_file.name}:{uploaded_file.size}:{file_hash}"

                persisted = st.session_state.get("upload_persisted_file")
                if persisted and persisted.get("file_sig") == file_sig:
                    df_raw = persisted["df_raw"]
                    df_features = persisted["df_features"]
                    file_name = persisted["file_name"]
                    val_warnings = persisted.get("val_warnings", [])
                else:
                    uploaded_file.seek(0)
                    df_raw = pd.read_csv(uploaded_file)
                    file_name = uploaded_file.name

                    is_valid, val_errs, val_warnings, df_features = validate_uploaded_dataframe(df_raw)
                    if not is_valid:
                        st.session_state.pop("upload_persisted_file", None)
                        st.session_state.pop("upload_batch_results", None)
                        for e in val_errs:
                            st.error(f"❌ Validación fallida: {e}")
                        return

                    st.session_state["upload_persisted_file"] = {
                        "file_name": file_name,
                        "file_sig": file_sig,
                        "df_raw": df_raw,
                        "df_features": df_features,
                        "val_warnings": val_warnings,
                    }
                    cached_batch = st.session_state.get("upload_batch_results")
                    if cached_batch and cached_batch.get("file_sig") != file_sig:
                        st.session_state.pop("upload_batch_results", None)
            else:
                persisted = st.session_state.get("upload_persisted_file")
                if not persisted:
                    return
                df_raw = persisted["df_raw"]
                df_features = persisted["df_features"]
                file_name = persisted["file_name"]
                file_sig = persisted["file_sig"]
                val_warnings = persisted.get("val_warnings", [])

            st.write(f"Archivo cargado: **{file_name}** ({len(df_raw):,} filas × {df_raw.shape[1]} columnas)")
            st.success("✅ Validación exitosa: columnas, tipos de datos y categorías verificadas.")
            for w in val_warnings:
                st.info(f"ℹ️ {w}")

            col_act1, col_act2, _ = st.columns([1, 1, 2])
            with col_act1:
                predict_clicked = st.button("➜  Predict", type="primary", key="upload_predict_btn", use_container_width=True)
            with col_act2:
                if st.button("🧹 Limpiar CSV", key="upload_clear_btn", use_container_width=True):
                    st.session_state.pop("upload_persisted_file", None)
                    st.session_state.pop("upload_batch_results", None)
                    st.rerun()

        except Exception as exc:
            st.error(f"Error al procesar el archivo CSV: {exc}")
            return

    cached_batch = st.session_state.get("upload_batch_results")
    if predict_clicked:
        df_out, pred_cols = _run_batch_inference(models, df_raw, df_features)
        st.session_state["upload_batch_results"] = {
            "file_sig": file_sig,
            "df_out": df_out,
            "pred_cols": pred_cols,
        }
        cached_batch = st.session_state["upload_batch_results"]

    if cached_batch:
        model_names = list(models.keys())
        active_thresholds = render_threshold_selector(
            eval_results,
            model_names,
            key_prefix=f"upload_{file_sig}",
        )
        df_thresholded = cached_batch["df_out"].copy()
        pred_cols = cached_batch["pred_cols"]
        for m_name in model_names:
            short_key = m_name.lower().replace(" ", "_")
            df_thresholded[f"pred_{short_key}"] = (
                df_thresholded[f"prob1_{short_key}"] >= active_thresholds[m_name]
            ).astype(int)
        df_thresholded["consenso_3_modelos"] = df_thresholded[pred_cols].nunique(axis=1).map(
            lambda n: "Unánime (3/3)" if n == 1 else "Discrepancia (2/1)"
        )

        batch_loader = st.empty()
        show_inline_loading(
            batch_loader,
            "Preparando análisis visual del lote",
            f"Generando métricas y gráficas para {file_name}...",
        )
        _render_batch_results(
            models=models,
            df_raw=df_raw,
            df_out=df_thresholded,
            pred_cols=pred_cols,
            file_name=file_name,
            file_sig=file_sig,
            eval_results=eval_results,
        )
        batch_loader.empty()

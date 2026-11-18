"""Pestaña 4: Evaluación experimental, correlaciones, EDA en StratifiedKFold y métricas Test."""

import os

import pandas as pd
import streamlit as st

from app.charts import (
    plot_confusion_matrices_row,
    plot_correlation_heatmap_and_target_bars,
    plot_grouped_status_pct,
    plot_hyperparameter_effects,
    plot_hyperparameter_heatmaps,
    plot_income_ratio_default_bars,
    plot_income_vs_loan_scatter,
    plot_model_decision_boundaries,
    plot_random_forest_top_importance,
    plot_status_boxplot,
    plot_test_metrics_and_roc,
)
from app.charts_3d import render_interactive_3d_scatter
from app.config import CSV_PATH, MAX_VALID_AGE, MAX_VALID_EMP_LENGTH, TOP_BOUNDARY_FEATURES, VALID_CATEGORIES
from app.frontend import show_inline_loading
from app.services import (
    compute_corr_payload_from_df,
    compute_decision_boundary_grids,
    load_cleaning_audit,
    load_cv5_eda_data,
    load_test_eda_data,
)


def render_eda_diagnostics(df_eda: pd.DataFrame, models: dict) -> None:
    """Renderiza el análisis de riesgo por carga de ingreso y la importancia de variables."""
    st.markdown("**¿Cómo se relacionan el porcentaje de ingreso comprometido y los atributos del solicitante con el riesgo de Incumplimiento?**")
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        plot_income_ratio_default_bars(df_eda)
    with exp_c2:
        if "Random Forest" in models:
            plot_random_forest_top_importance(models["Random Forest"])
        else:
            st.info("Carga `models/random_forest.joblib` para visualizar la importancia de variables.")


def render_correlation_section(
    df_source: pd.DataFrame | None,
    eval_results: dict | None,
    key_prefix: str = "eval",
    subtitle: str | None = None,
) -> None:
    """Renderiza la Sección 1 (Mapa de Calor de Correlaciones y relación directa con loan_status)."""
    st.subheader("1. Mapa de Calor de Correlaciones (Características asociadas a Clase 1 (Incumplimiento) vs Clase 0 (Sin Incumplimiento))")
    st.markdown(
        subtitle
        or (
            "Correlación de Pearson calculada sobre los 5 folds de `cv_5` "
            "entre las características del solicitante y `loan_status` "
            "(**valores positivos en rojo** se asocian con **Clase 1: Incumplimiento**; **valores negativos en verde** se asocian con **Clase 0: Sin Incumplimiento**)."
        )
    )

    corr_info = compute_corr_payload_from_df(df_source) if df_source is not None else (eval_results or {}).get("feature_correlations", {})
    if corr_info:
        n_encoded = max(len(corr_info.get("columns", [])) - 1, 1)
        show_all_corr = st.checkbox(
            f"Mostrar todas las {n_encoded} variables codificadas (One-Hot completo) en el mapa de calor",
            value=False,
            key=f"{key_prefix}_show_all_corr",
        )
        plot_correlation_heatmap_and_target_bars(corr_info, show_all_corr)


def render_eda_section(
    df_eda: pd.DataFrame,
    models: dict,
    section_title: str,
    description: str,
) -> None:
    """Renderiza la Sección 2 (EDA 2D por categoría/numérica, vista 3D interactiva y diagnósticos)."""
    st.subheader(section_title)
    st.markdown(description)

    eda_r1_c1, eda_r1_c2 = st.columns(2)
    with eda_r1_c1:
        plot_grouped_status_pct(df_eda, "loan_grade", "Grado del Préstamo (loan_grade) vs loan_status", "Grado Crediticio (loan_grade)")
    with eda_r1_c2:
        plot_grouped_status_pct(df_eda, "person_home_ownership", "Tipo de Vivienda (person_home_ownership) vs loan_status", "Tipo de Vivienda")

    eda_r2_c1, eda_r2_c2 = st.columns(2)
    with eda_r2_c1:
        plot_grouped_status_pct(df_eda, "cb_person_default_on_file", "Mora Previa en Buró (cb_person_default_on_file) vs loan_status", "Incumplimiento Previo en Buró")
    with eda_r2_c2:
        plot_grouped_status_pct(df_eda, "loan_intent", "Propósito del Préstamo (loan_intent) vs loan_status", "Propósito del Préstamo")

    eda_r3_c1, eda_r3_c2 = st.columns(2)
    with eda_r3_c1:
        plot_status_boxplot(df_eda, "loan_percent_income", "% de Ingreso Comprometido (loan_percent_income) vs loan_status", "Proporción del Ingreso (loan_percent_income)")
    with eda_r3_c2:
        plot_status_boxplot(df_eda, "loan_int_rate", "Tasa de Interés (%) (loan_int_rate) vs loan_status", "Tasa de Interés (%)")

    eda_r4_c1, eda_r4_c2 = st.columns(2)
    with eda_r4_c1:
        df_scat = df_eda.sample(n=min(4000, len(df_eda)), random_state=42)
        inc_cap = max(float(df_eda["person_income"].quantile(0.95)), 1.0)
        plot_income_vs_loan_scatter(df_scat, inc_cap)
    with eda_r4_c2:
        plot_status_boxplot(df_eda, "person_emp_length", "Antigüedad Laboral (person_emp_length) vs loan_status", "Antigüedad Laboral (años)")

    st.markdown("**Vista 3D Interactiva: Ingreso Anual × Monto Solicitado × Tasa de Interés (`loan_int_rate`) por `loan_status`**")
    df_scat_3d = df_eda[df_eda["person_income"] <= inc_cap]
    if df_scat_3d.empty:
        df_scat_3d = df_eda
    render_interactive_3d_scatter(df_scat_3d, inc_cap)
    render_eda_diagnostics(df_eda, models)


def render_decision_boundary_section(
    df_source: pd.DataFrame,
    models: dict,
    eval_results: dict | None,
    dataset_tag: str,
    source_label: str,
    balanced_tag: bool,
) -> None:
    """Renderiza las fronteras de decisión 2D de cada modelo sobre las variables principales."""
    st.markdown(f"**Fronteras de Decisión por Modelo sobre las Características Más Influyentes ({source_label})**")
    st.caption(
        f"Visualiza cómo cada modelo entrenado separa **Sin Incumplimiento (0)** vs **Incumplimiento (1)** sobre dos variables principales "
        f"usando **{source_label}**, manteniendo las demás características fijas en la mediana/moda del conjunto. "
        "La línea amarilla marca la frontera donde la probabilidad estimada de Incumplimiento cruza `P = 0.50`."
    )

    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        feat_x = st.selectbox(
            "Eje X (Característica 1)",
            options=TOP_BOUNDARY_FEATURES,
            index=0,
            key=f"db_{dataset_tag}_feat_x",
        )
    with bc2:
        feat_y = st.selectbox(
            "Eje Y (Característica 2)",
            options=TOP_BOUNDARY_FEATURES,
            index=1,
            key=f"db_{dataset_tag}_feat_y",
        )
    with bc3:
        grade_val = st.selectbox(
            "Grado crediticio (`loan_grade`)",
            options=VALID_CATEGORIES["loan_grade"],
            index=1,
            key=f"db_{dataset_tag}_grade",
        )
    with bc4:
        home_val = st.selectbox(
            "Tipo de vivienda (`home_ownership`)",
            options=VALID_CATEGORIES["person_home_ownership"],
            index=0,
            key=f"db_{dataset_tag}_home",
        )

    if feat_x == feat_y:
        st.warning("Selecciona dos características diferentes para los ejes X e Y.")
        return

    db_loader = st.empty()
    show_inline_loading(
        db_loader,
        "Calculando fronteras de decisión 2D",
        f"Evaluando {feat_x} × {feat_y} sobre los 3 modelos...",
    )
    boundary_data = compute_decision_boundary_grids(
        _models=models,
        _df_eda=df_source,
        feat_x=feat_x,
        feat_y=feat_y,
        grade_val=grade_val,
        home_val=home_val,
        balanced_tag=balanced_tag,
        dataset_tag=dataset_tag,
    )
    db_loader.empty()
    plot_model_decision_boundaries(boundary_data, feat_x, feat_y, eval_results or {})

    if not boundary_data["matched_both_cats"]:
        st.caption(
            f"Nota: La combinación `loan_grade='{grade_val}'` + `home_ownership='{home_val}'` tiene pocos registros en {source_label}; "
            "los puntos mostrados corresponden al subconjunto disponible."
        )

    st.markdown(
        """
    - **SVM (Kernel RBF):** Traza fronteras curvas suaves (islas/contornos gaussianos según `C` y `gamma`).
    - **Random Forest:** Divide el espacio mediante regiones rectangulares en escalera producto de cortes por umbrales de árboles.
    - **Regresión Logística:** Separa ambas clases mediante una frontera lineal recta en el espacio estandarizado.
    """
    )


def _render_before_vs_after_table(eval_results: dict, model_names: list[str], baseline_test_map: dict) -> None:
    """Renderiza la tabla comparativa Antes vs Después del ajuste de hiperparámetros."""
    bva_rows = []
    if all(m in baseline_test_map for m in model_names):
        for m_name in model_names:
            cm_before = baseline_test_map[m_name]["confusion_matrix"]
            cm_after = eval_results["optimized_test"][m_name]["confusion_matrix"]
            tn0, fp0, fn0, tp0 = cm_before[0][0], cm_before[0][1], cm_before[1][0], cm_before[1][1]
            tn1, fp1, fn1, tp1 = cm_after[0][0], cm_after[0][1], cm_after[1][0], cm_after[1][1]
            bva_rows.append(
                {
                    "Modelo": m_name,
                    "TN (Antes → Después)": f"{tn0:,} → {tn1:,} ({tn1 - tn0:+d})",
                    "FP (Antes → Después)": f"{fp0:,} → {fp1:,} ({fp1 - fp0:+d})",
                    "FN (Antes → Después)": f"{fn0:,} → {fn1:,} ({fn1 - fn0:+d})",
                    "TP (Antes → Después)": f"{tp0:,} → {tp1:,} ({tp1 - tp0:+d})",
                    "F1 (Antes → Después)": (
                        f"{baseline_test_map[m_name]['f1']:.4f} → "
                        f"{eval_results['optimized_test'][m_name]['f1']:.4f} "
                        f"({eval_results['optimized_test'][m_name]['f1'] - baseline_test_map[m_name]['f1']:+.4f})"
                    ),
                }
            )
        st.markdown("**Comparativa Antes vs Después del ajuste de hiperparámetros en el Test Set:**")
    else:
        for m_name in model_names:
            cv_b = eval_results["baseline_cv"][m_name]
            opt_a = eval_results["optimized_test"][m_name]
            bva_rows.append(
                {
                    "Modelo": m_name,
                    "Accuracy (CV Base → Test Opt)": f"{cv_b['accuracy_mean']:.4f} → {opt_a['accuracy']:.4f} ({opt_a['accuracy'] - cv_b['accuracy_mean']:+.4f})",
                    "Precision (CV Base → Test Opt)": f"{cv_b['precision_mean']:.4f} → {opt_a['precision']:.4f} ({opt_a['precision'] - cv_b['precision_mean']:+.4f})",
                    "Recall (CV Base → Test Opt)": f"{cv_b['recall_mean']:.4f} → {opt_a['recall']:.4f} ({opt_a['recall'] - cv_b['recall_mean']:+.4f})",
                    "F1-Score (CV Base → Test Opt)": f"{cv_b['f1_mean']:.4f} → {opt_a['f1']:.4f} ({opt_a['f1'] - cv_b['f1_mean']:+.4f})",
                    "ROC-AUC (CV Base → Test Opt)": f"{cv_b['roc_auc_mean']:.4f} → {opt_a['roc_auc']:.4f} ({opt_a['roc_auc'] - cv_b['roc_auc_mean']:+.4f})",
                }
            )
        st.markdown("**Comparativa Antes (5-Fold CV Train Default) vs Después (Test Set Optimizado):**")
    st.dataframe(pd.DataFrame(bva_rows), use_container_width=True, hide_index=True)


def _render_cleaning_summary(ds_info: dict, csv_mtime: float) -> None:
    """Muestra las cifras de limpieza determinista del dataset y el detalle de registros eliminados."""
    raw_n = ds_info.get("raw_rows", ds_info.get("total_rows", 0))
    dup_n = ds_info.get("duplicates_removed", 0)
    out_n = ds_info.get("outliers_removed", 0)
    clean_n = ds_info.get("total_rows", raw_n - dup_n - out_n)
    cd = ds_info.get("class_distribution", {})
    c0_n = cd.get("0_no_default", cd.get("0_no_aprobado", 0))
    c1_n = cd.get("1_default", cd.get("1_aprobado", 0))
    nulls_map = ds_info.get("missing_values_clean", {})
    null_txt = (
        ", ".join(f"`{col}` ({cnt:,} nulos)" for col, cnt in nulls_map.items())
        if nulls_map
        else "0 nulos"
    )

    st.markdown("**🧹 Resumen de Limpieza y Datos Depurados (`credit_risk_dataset.csv`)**")
    cl1, cl2, cl3, cl4 = st.columns(4)
    cl1.metric("Registros brutos", f"{raw_n:,}")
    cl2.metric("Duplicados eliminados", f"-{dup_n:,}", f"{(dup_n / max(raw_n, 1)) * 100:.2f}% del bruto", delta_color="off")
    cl3.metric(
        "Atípicos imposibles eliminados",
        f"-{out_n:,}",
        f"edad > {MAX_VALID_AGE} o emp > {MAX_VALID_EMP_LENGTH}",
        delta_color="off",
    )
    cl4.metric(
        "Dataset limpio total",
        f"{clean_n:,}",
        f"Train {ds_info['train_rows']:,} · Test {ds_info['test_rows']:,}",
        delta_color="off",
    )
    st.caption(
        f"Sobre los **{clean_n:,}** registros limpios: **Sin Incumplimiento (0): {c0_n:,} ({(c0_n / max(clean_n, 1)) * 100:.2f}%)** · "
        f"**Incumplimiento (1): {c1_n:,} ({(c1_n / max(clean_n, 1)) * 100:.2f}%)**. "
        f"Valores faltantes conservados para imputación por mediana dentro del `Pipeline` (fit en `X_train`): {null_txt}."
    )

    audit = load_cleaning_audit(csv_mtime)
    if audit is not None:
        df_out_rows = audit["outliers_df"]
        df_dup_rows = audit["duplicates_df"]
        with st.expander(
            f"🔍 Ver valores eliminados durante la limpieza ({len(df_out_rows)} atípicos y {len(df_dup_rows)} duplicados)",
            expanded=False,
        ):
            st.markdown(f"**Atípicos imposibles eliminados (`{len(df_out_rows)}` filas):** `person_age > {MAX_VALID_AGE}` o `person_emp_length > {MAX_VALID_EMP_LENGTH}`")
            if not df_out_rows.empty:
                st.dataframe(df_out_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron valores atípicos fuera de rango.")

            st.markdown(f"**Filas duplicadas exactas eliminadas (`{len(df_dup_rows)}` filas conservando la primera ocurrencia):**")
            if not df_dup_rows.empty:
                st.dataframe(df_dup_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No se detectaron duplicados exactos.")


def render_tab_eval(models: dict, eval_results: dict) -> None:
    """Renderiza toda la pestaña de evaluación experimental y EDA."""
    ds_info = eval_results["dataset"]
    st.header("📊 Evaluación Experimental de los Modelos")
    st.markdown(
        f"Todos los valores presentados en esta sección provienen directamente de los experimentos "
        f"ejecutados sobre la partición oficial del proyecto (**80% Train = {ds_info['train_rows']:,}** / **20% Test = {ds_info['test_rows']:,}**, `random_state=42`) "
        f"de `credit_risk_dataset.csv`."
    )

    model_names = list(eval_results["optimized_test"].keys())
    csv_mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0

    _render_cleaning_summary(ds_info, csv_mtime)

    st.markdown("### ⚖️ Fuente de Datos para Gráficas: `StratifiedKFold (cv_5)` — 5 Folds de `X_train`")
    fold_choice = "Todos los 5 folds (cv_5)"
    use_balanced_cv5 = st.checkbox(
        "Balancear clases 50% / 50% dentro de `cv_5` (igualar Sin Incumplimiento 0 vs Incumplimiento 1)",
        value=True,
        help="StratifiedKFold conserva la proporción original (~78.1% vs ~21.9%) en cada fold; activar este balanceo iguala 50/50 ambas clases dentro de cv_5.",
    )

    df_eda = load_cv5_eda_data(csv_mtime, fold_choice=fold_choice, balanced=use_balanced_cv5)

    if df_eda is not None:
        n_total_eda = len(df_eda)
        n_c0 = int((df_eda["loan_status"] == 0).sum())
        n_c1 = int((df_eda["loan_status"] == 1).sum())
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Registros activos (`cv_5`)", f"{n_total_eda:,}")
        mc2.metric("Clase 0 — Sin Incumplimiento", f"{n_c0:,} ({(n_c0 / n_total_eda) * 100:.1f}%)")
        mc3.metric("Clase 1 — Incumplimiento", f"{n_c1:,} ({(n_c1 / n_total_eda) * 100:.1f}%)")
        st.caption(
            "Nota técnica: `StratifiedKFold (cv_5)` garantiza que los 5 folds mantengan exactamente la misma proporción de clases sin sesgo. "
            + (
                "Con el balanceo 50/50 activado dentro de los 5 folds de `cv_5`, ambas clases tienen el mismo peso visual para comparar patrones reales sin dominancia de la clase mayoritaria."
                if use_balanced_cv5
                else "Mostrando la proporción estratificada original (~78.1% Sin Incumplimiento vs ~21.9% Incumplimiento) de los 5 folds de `cv_5`."
            )
        )

    # 1. Mapa de Calor de Correlaciones
    render_correlation_section(df_source=df_eda, eval_results=eval_results, key_prefix="eval")

    # 2. EDA sobre cv_5
    eda_mode_tag = "Balanceado 50/50" if use_balanced_cv5 else "Estratificado"
    eda_title = f"2. Análisis Exploratorio de Datos (EDA — {fold_choice}, {eda_mode_tag})"
    if df_eda is not None:
        eda_desc = (
            f"Comportamiento de las principales características respecto al resultado del préstamo "
            f"(**0 = Sin Incumplimiento**, **1 = Incumplimiento**) sobre la partición de `StratifiedKFold (cv_5)` seleccionada "
            f"(**{len(df_eda):,} registros**)."
        )
        render_eda_section(df_eda, models, section_title=eda_title, description=eda_desc)
    else:
        st.subheader(eda_title)
        st.warning("No se encontró `credit_risk_dataset.csv` para generar las gráficas EDA.")

    # 3. Etapa Inicial (Baseline CV)
    st.subheader("3. Etapa Inicial (Parámetros por Defecto — 5-Fold Cross-Validation en Train)")
    cv_rows = [
        {
            "Modelo (Default)": m_name,
            "Accuracy (CV)": f"{vals['accuracy_mean']:.4f} (±{vals['accuracy_std']:.4f})",
            "Precision (CV)": f"{vals['precision_mean']:.4f}",
            "Recall (CV)": f"{vals['recall_mean']:.4f}",
            "F1-Score (CV)": f"{vals['f1_mean']:.4f} (±{vals['f1_std']:.4f})",
            "ROC-AUC (CV)": f"{vals['roc_auc_mean']:.4f}",
        }
        for m_name, vals in eval_results["baseline_cv"].items()
    ]
    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

    if df_eda is not None:
        render_decision_boundary_section(
            df_source=df_eda,
            models=models,
            eval_results=eval_results,
            dataset_tag="train_cv5",
            source_label=f"Train cv_5 ({len(df_eda):,} registros)",
            balanced_tag=use_balanced_cv5,
        )

    # 3.1 Heatmaps de Hiperparámetros (5-Fold CV en Train)
    st.subheader("3.1 Exploración de Hiperparámetros — Heatmaps Comparativos por Modelo (5-Fold CV en Train)")
    tuning_heatmaps = eval_results.get("tuning_heatmaps", {})
    if tuning_heatmaps:
        st.caption(
            "Comparación del desempeño promedio (**F1-Score**) en validación cruzada (5-Fold CV sobre `X_train`) al variar dos hiperparámetros clave "
            "por modelo. El recuadro rojo marca la combinación seleccionada como óptima (`TUNING_SCORING = f1`)."
        )
        plot_hyperparameter_heatmaps(
            tuning_heatmaps=tuning_heatmaps,
            model_names=model_names,
            metric_key="scores_f1",
            metric_label="F1-Score",
        )
        fixed_notes = []
        for m_name in model_names:
            hm_m = tuning_heatmaps.get(m_name)
            if hm_m and hm_m.get("fixed_params"):
                fixed_str = ", ".join(f"{k}={v}" for k, v in hm_m["fixed_params"].items())
                fixed_notes.append(f"**{m_name}** (fijos: `{fixed_str}`)")
        if fixed_notes:
            st.caption("Parámetros fijos por grilla: " + " · ".join(fixed_notes))
    else:
        st.info(
            "No se encontraron grillas de hiperparámetros en `evaluation_results.json`. "
            "Ejecuta `python model_factory/train_model.py` para generarlas."
        )

    # 3.2 Curvas de Validación 1D por Hiperparámetro (Train vs 5-Fold CV)
    st.subheader("3.2 Efecto de Cada Hiperparámetro en el Aprendizaje del Modelo (Train vs. Validación 5-Fold CV)")
    hp_effects = eval_results.get("hyperparameter_effects", {})
    available_fx_models = [m for m in model_names if m in hp_effects]
    if available_fx_models:
        st.caption(
            "Variación controlada de un hiperparámetro a la vez (manteniendo los demás fijos en su valor óptimo) "
            "mostrando **F1-Score en entrenamiento (naranja discontinuo)** vs **validación 5-Fold CV (azul sólido ±1σ)** "
            "para detectar subajuste, sobreajuste y punto óptimo de generalización (estrella roja)."
        )
        fx_model = st.radio(
            "Modelo a inspeccionar:",
            options=available_fx_models,
            index=available_fx_models.index("Random Forest") if "Random Forest" in available_fx_models else 0,
            horizontal=True,
            key="etapa3_2_fx_model",
        )
        plot_hyperparameter_effects(
            model_effects=hp_effects[fx_model],
            model_name=fx_model,
            metric_key="f1",
            metric_label="F1-Score",
        )
        fixed_best = hp_effects[fx_model].get("fixed_best_params", {})
        if fixed_best:
            fixed_txt = ", ".join(f"{k}={v}" for k, v in fixed_best.items())
            st.caption(f"Configuración base de referencia para **{fx_model}**: `{fixed_txt}`.")
    else:
        st.info(
            "No se encontraron curvas por hiperparámetro en `evaluation_results.json`. "
            "Ejecuta `python model_factory/train_model.py` para generarlas."
        )

    # 4. Etapa Optimizada (Test Set Holdout)
    st.subheader("4. Etapa Optimizada — Después de Ajustar Hiperparámetros (Test Set 20% Holdout)")
    baseline_test_map = eval_results.get("baseline_test", {})
    test_rows = []
    for m_name, vals in eval_results["optimized_test"].items():
        base_f1 = baseline_test_map[m_name]["f1"] if m_name in baseline_test_map else eval_results["baseline_cv"][m_name]["f1_mean"]
        delta_f1 = vals["f1"] - base_f1
        test_rows.append(
            {
                "Modelo": m_name,
                "Mejores Hiperparámetros": str(vals["best_params"]),
                "Accuracy": vals["accuracy"],
                "Precision": vals["precision"],
                "Recall": vals["recall"],
                "F1-Score": vals["f1"],
                "Δ F1 vs Default": f"{delta_f1:+.4f}",
                "ROC-AUC": vals["roc_auc"],
            }
        )
    st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)

    df_test_eval = load_test_eda_data(csv_mtime)
    if df_test_eval is not None:
        render_decision_boundary_section(
            df_source=df_test_eval,
            models=models,
            eval_results=eval_results,
            dataset_tag="test_holdout",
            source_label=f"Test Set 20% Holdout ({len(df_test_eval):,} registros)",
            balanced_tag=False,
        )

    # 5. Gráficas comparativas
    st.subheader("5. Visualización Comparativa en Test Set (Modelos Optimizados)")
    plot_test_metrics_and_roc(eval_results["optimized_test"], model_names)

    # 6. Matrices de Confusión Finales
    st.subheader("6. Matrices de Confusión Finales — Test Set Después del Ajuste de Hiperparámetros")
    opt_subtitles = {
        m: (
            f"Test Optimizado | Acc: {eval_results['optimized_test'][m]['accuracy']:.4f} | "
            f"F1: {eval_results['optimized_test'][m]['f1']:.4f} "
            f"({eval_results['optimized_test'][m]['f1'] - (baseline_test_map[m]['f1'] if m in baseline_test_map else eval_results['baseline_cv'][m]['f1_mean']):+.4f})"
        )
        for m in model_names
    }
    plot_confusion_matrices_row(eval_results["optimized_test"], model_names, cmap="Blues", subtitle_map=opt_subtitles)
    _render_before_vs_after_table(eval_results, model_names, baseline_test_map)

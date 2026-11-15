"""Pestaña 4: Evaluación experimental, correlaciones, EDA en StratifiedKFold y métricas Test."""

import os

import pandas as pd
import streamlit as st

from app.charts import (
    plot_confusion_matrices_row,
    plot_correlation_heatmap_and_target_bars,
    plot_grouped_status_pct,
    plot_hyperparameter_heatmaps,
    plot_income_ratio_approval_bars,
    plot_income_vs_loan_scatter,
    plot_model_decision_boundaries,
    plot_random_forest_top_importance,
    plot_status_boxplot,
    plot_test_metrics_and_roc,
)
from app.charts_3d import render_interactive_3d_scatter
from app.config import CSV_PATH, TOP_BOUNDARY_FEATURES, VALID_CATEGORIES
from app.frontend import show_inline_loading
from app.services import (
    compute_corr_payload_from_df,
    compute_decision_boundary_grids,
    load_cv5_eda_data,
    load_test_eda_data,
)


def render_eda_diagnostics(df_eda: pd.DataFrame, models: dict) -> None:
    """Renderiza el análisis de la frontera diagonal y la importancia de variables."""
    st.markdown("**¿Por qué los préstamos aprobados (verde) se concentran en menores ingresos y montos moderados?**")
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        plot_income_ratio_approval_bars(df_eda)
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
    st.subheader("1. Mapa de Calor de Correlaciones (Características asociadas a Clase 1 (Aprobado) vs Clase 0 (No Aprobado))")
    st.markdown(
        subtitle
        or (
            "Correlación de Pearson calculada sobre los 5 folds de `cv_5` "
            "entre las características del solicitante y `loan_status` "
            "(**valores positivos en azul** se asocian con **Clase 1: Aprobado**; **valores negativos en rojo** se asocian con **Clase 0: No Aprobado**)."
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
        plot_grouped_status_pct(df_eda, "loan_intent", "Propósito del Préstamo (loan_intent) vs loan_status", "Propósito del Préstamo (loan_intent)")
    with eda_r1_c2:
        plot_grouped_status_pct(df_eda, "person_home_ownership", "Tipo de Vivienda (person_home_ownership) vs loan_status", "Tipo de Vivienda")

    eda_r2_c1, eda_r2_c2 = st.columns(2)
    with eda_r2_c1:
        plot_grouped_status_pct(df_eda, "previous_loan_defaults_on_file", "Incumplimientos Previos vs loan_status", "Incumplimiento Previo Registrado")
    with eda_r2_c2:
        plot_status_boxplot(df_eda, "loan_percent_income", "% de Ingreso Comprometido (loan_percent_income) vs loan_status", "Proporción del Ingreso (loan_percent_income)")

    eda_r3_c1, eda_r3_c2 = st.columns(2)
    with eda_r3_c1:
        plot_status_boxplot(df_eda, "loan_int_rate", "Tasa de Interés (%) (loan_int_rate) vs loan_status", "Tasa de Interés (%)")
    with eda_r3_c2:
        df_scat = df_eda.sample(n=min(4000, len(df_eda)), random_state=42)
        inc_cap = max(float(df_eda["person_income"].quantile(0.95)), 1.0)
        plot_income_vs_loan_scatter(df_scat, inc_cap)

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
        f"Visualiza cómo cada modelo entrenado separa **Denegado (0)** vs **Aprobado (1)** sobre dos variables principales "
        f"usando **{source_label}**, manteniendo las demás características fijas en la mediana/moda del conjunto. "
        "La línea amarilla marca la frontera donde la probabilidad estimada cruza `P = 0.50`."
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
        default_val = st.selectbox(
            "Incumplimientos previos (`defaults`)",
            options=VALID_CATEGORIES["previous_loan_defaults_on_file"],
            index=0,
            key=f"db_{dataset_tag}_defaults",
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
        default_val=default_val,
        home_val=home_val,
        balanced_tag=balanced_tag,
        dataset_tag=dataset_tag,
    )
    db_loader.empty()
    plot_model_decision_boundaries(boundary_data, feat_x, feat_y, eval_results or {})

    if not boundary_data["matched_both_cats"]:
        st.caption(
            f"Nota: La combinación `defaults='{default_val}'` + `home_ownership='{home_val}'` tiene pocos registros en {source_label}; "
            "los puntos mostrados corresponden a todas las viviendas con ese historial de mora."
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
        st.markdown("**Comparativa Antes vs Después del ajuste de hiperparámetros en el Test Set (9,000 registros):**")
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


def render_tab_eval(models: dict, eval_results: dict) -> None:
    """Renderiza toda la pestaña de evaluación experimental y EDA."""
    st.header("📊 Evaluación Experimental de los Modelos")
    st.markdown(
        "Todos los valores presentados en esta sección provienen directamente de los experimentos "
        "ejecutados sobre la partición oficial del proyecto (**80% Train = 36,000** / **20% Test = 9,000**, `random_state=42`)."
    )

    model_names = list(eval_results["optimized_test"].keys())

    st.markdown("### ⚖️ Fuente de Datos para Gráficas: `StratifiedKFold (cv_5)` — 5 Folds de `X_train`")
    fold_choice = "Todos los 5 folds (cv_5)"
    use_balanced_cv5 = st.checkbox(
        "Balancear clases 50% / 50% dentro de `cv_5` (igualar Denegado 0 vs Aprobado 1)",
        value=True,
        help="StratifiedKFold conserva la proporción original (~77.8% vs ~22.2%) en cada fold; activar este balanceo iguala 50/50 ambas clases dentro de cv_5.",
    )

    csv_mtime = os.path.getmtime(CSV_PATH) if os.path.exists(CSV_PATH) else 0.0
    df_eda = load_cv5_eda_data(csv_mtime, fold_choice=fold_choice, balanced=use_balanced_cv5)

    if df_eda is not None:
        n_total_eda = len(df_eda)
        n_c0 = int((df_eda["loan_status"] == 0).sum())
        n_c1 = int((df_eda["loan_status"] == 1).sum())
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Registros activos (`cv_5`)", f"{n_total_eda:,}")
        mc2.metric("Clase 0 — Denegado", f"{n_c0:,} ({(n_c0 / n_total_eda) * 100:.1f}%)")
        mc3.metric("Clase 1 — Aprobado", f"{n_c1:,} ({(n_c1 / n_total_eda) * 100:.1f}%)")
        st.caption(
            "Nota técnica: `StratifiedKFold (cv_5)` garantiza que los 5 folds mantengan exactamente la misma proporción de clases sin sesgo. "
            + (
                "Con el balanceo 50/50 activado dentro de los 5 folds de `cv_5`, ambas clases tienen el mismo peso visual para comparar patrones reales sin dominancia de la clase mayoritaria."
                if use_balanced_cv5
                else "Mostrando la proporción estratificada original (~77.8% Denegado vs ~22.2% Aprobado) de los 5 folds de `cv_5`."
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
            f"(**1 = Aprobado**, **0 = Denegado**) sobre la partición de `StratifiedKFold (cv_5)` seleccionada "
            f"(**{len(df_eda):,} registros**)."
        )
        render_eda_section(df_eda, models, section_title=eda_title, description=eda_desc)
    else:
        st.subheader(eda_title)
        st.warning("No se encontró `loan_data.csv` para generar las gráficas EDA.")

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
            "Comparación del desempeño promedio en validación cruzada (5-Fold CV sobre `X_train`) al variar dos hiperparámetros clave "
            "por modelo. El recuadro rojo marca la combinación seleccionada como óptima (`TUNING_SCORING = f1`)."
        )
        hm_metric_label = st.radio(
            "Métrica visualizada en los heatmaps:",
            options=["F1-Score", "ROC-AUC"],
            index=0,
            horizontal=True,
            key="etapa3_1_hm_metric",
        )
        metric_key = "scores_f1" if hm_metric_label == "F1-Score" else "scores_roc_auc"
        plot_hyperparameter_heatmaps(
            tuning_heatmaps=tuning_heatmaps,
            model_names=model_names,
            metric_key=metric_key,
            metric_label=hm_metric_label,
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

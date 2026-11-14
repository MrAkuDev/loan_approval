"""Pestaña 4: Evaluación experimental, correlaciones, EDA en StratifiedKFold y métricas Test."""

import os

import pandas as pd
import streamlit as st

from app.charts import (
    plot_confusion_matrices_row,
    plot_correlation_heatmap_and_target_bars,
    plot_grouped_status_pct,
    plot_income_ratio_approval_bars,
    plot_income_vs_loan_scatter,
    plot_random_forest_top_importance,
    plot_status_boxplot,
    plot_test_metrics_and_roc,
)
from app.charts_3d import render_interactive_3d_scatter
from app.config import CSV_PATH
from app.services import compute_corr_payload_from_df, load_cv5_eda_data


def _render_eda_diagnostics(df_eda: pd.DataFrame, models: dict) -> None:
    """Renderiza el análisis de la frontera diagonal y la importancia de variables."""
    st.markdown("**¿Por qué los préstamos aprobados (verde) se concentran en menores ingresos y montos moderados?**")
    exp_c1, exp_c2 = st.columns(2)

    with exp_c1:
        rate_all = plot_income_ratio_approval_bars(df_eda)
    with exp_c2:
        if "Random Forest" in models:
            plot_random_forest_top_importance(models["Random Forest"])
        else:
            st.info("Carga `models/random_forest.joblib` para visualizar la importancia de variables.")

    low_mask = df_eda["person_income"] <= 50000
    def_low = float((df_eda.loc[low_mask, "previous_loan_defaults_on_file"] == "Yes").mean() * 100.0)
    def_high = float((df_eda.loc[~low_mask, "previous_loan_defaults_on_file"] == "Yes").mean() * 100.0)
    rent_low = float((df_eda.loc[low_mask, "person_home_ownership"] == "RENT").mean() * 100.0)
    rent_high = float((df_eda.loc[~low_mask, "person_home_ownership"] == "RENT").mean() * 100.0)
    rate_ok = float(df_eda.loc[df_eda["loan_status"] == 1, "loan_int_rate"].mean())
    rate_no = float(df_eda.loc[df_eda["loan_status"] == 0, "loan_int_rate"].mean())

    st.info(
        "**¿Por qué aparece esa frontera diagonal y qué otras variables influyen?**\n"
        f"- **1. Razón `loan_percent_income = loan_amnt / person_income` (la diagonal):** "
        f"La frontera superior izquierda corresponde a solicitudes donde el monto representa **≥ 25–30%** del ingreso "
        f"(con tope frecuente en **$25,000**). En el conjunto de entrenamiento, cuando `loan_percent_income` supera `30%`, "
        f"la aprobación sube a **{rate_all['30–40%']:.1f}%–{rate_all['> 40%']:.1f}%**, frente a solo **{rate_all['0–10%']:.1f}%** cuando es `0–10%`.\n"
        f"- **2. Menor tasa de incumplimiento previo (`previous_loan_defaults_on_file`):** "
        f"Es la variable más influyente del modelo. En ingresos `≤ $50k`, el **{def_low:.1f}%** registra mora previa (`Yes`, donde se deniega el 100%), "
        f"mientras que en ingresos `> $50k` sube a **{def_high:.1f}%**.\n"
        f"- **3. Vivienda en renta (`RENT`) y tasa de interés (`loan_int_rate`):** "
        f"El **{rent_low:.1f}%** del grupo `≤ $50k` reporta vivienda `RENT` (vs **{rent_high:.1f}%** en `> $50k`), y los préstamos aprobados "
        f"concentran tasas de interés más altas (**{rate_ok:.2f}%** vs **{rate_no:.2f}%** en denegados), reflejando el patrón del dataset sintético (`SMOTENC`)."
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
    st.subheader("1. Mapa de Calor de Correlaciones (Características asociadas a Clase 1 vs Clase 0)")
    st.markdown(
        "Correlación de Pearson calculada sobre los 5 folds de `cv_5` "
        "entre las características del solicitante y `loan_status` "
        "(**valores positivos en azul** se asocian con **Clase 1: Aprobado**; **valores negativos en rojo** se asocian con **Clase 0: No Aprobado**)."
    )

    corr_info = compute_corr_payload_from_df(df_eda) if df_eda is not None else eval_results.get("feature_correlations", {})
    if corr_info:
        show_all_corr = st.checkbox(
            "Mostrar todas las 22 variables codificadas (One-Hot completo) en el mapa de calor",
            value=False,
        )
        plot_correlation_heatmap_and_target_bars(corr_info, show_all_corr)

    # 2. EDA sobre cv_5
    eda_mode_tag = "Balanceado 50/50" if use_balanced_cv5 else "Estratificado"
    st.subheader(f"2. Análisis Exploratorio de Datos (EDA — {fold_choice}, {eda_mode_tag})")
    st.markdown(
        f"Comportamiento de las principales características respecto al resultado del préstamo "
        f"(**1 = Aprobado**, **0 = Denegado**) sobre la partición de `StratifiedKFold (cv_5)` seleccionada "
        f"(**{len(df_eda):,} registros**)." if df_eda is not None else
        "Comportamiento de las principales características respecto al resultado del préstamo (**1 = Aprobado**, **0 = Denegado**)."
    )
    if df_eda is not None:
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
            inc_cap = float(df_eda["person_income"].quantile(0.95))
            plot_income_vs_loan_scatter(df_scat, inc_cap)

        st.markdown("**Vista 3D Interactiva: Ingreso Anual × Monto Solicitado × Tasa de Interés (`loan_int_rate`) por `loan_status`**")
        df_scat_3d = df_scat[df_scat["person_income"] <= inc_cap]
        render_interactive_3d_scatter(df_scat_3d, inc_cap)
        _render_eda_diagnostics(df_eda, models)
    else:
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

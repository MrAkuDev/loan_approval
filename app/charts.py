"""Renderizado de gráficas 2D con Matplotlib para predicción, EDA y evaluación."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.config import MODEL_PALETTE, NUMERICAL_COLUMNS


def plot_single_prediction_confidence(comp_rows: list[dict]) -> None:
    """Dibuja la gráfica horizontal de confianza por modelo para una instancia individual."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    probs_plot = [r["Prob. Aprobado (Clase 1)"] * 100 for r in comp_rows]
    colors = ["#059669" if r["Clase Predicha"] == 1 else "#DC2626" for r in comp_rows]
    bars = ax.barh([r["Modelo"] for r in comp_rows], probs_plot, color=colors)
    ax.axvline(50, color="#374151", linestyle="--", linewidth=1, label="Umbral 50%")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Probabilidad de Aprobación — Clase 1 (%)")
    ax.set_title("Confianza por Modelo")
    for bar, val in zip(bars, probs_plot):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9)
    ax.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_confusion_matrices_row(
    results_by_model: dict,
    model_names: list[str],
    cmap: str,
    subtitle_map: dict[str, str],
) -> None:
    """Dibuja una fila de matrices de confusión 2x2 para los modelos indicados."""
    fig_cm, axes_cm = plt.subplots(1, len(model_names), figsize=(15, 4.3))
    labels_cm = ["No Aprobado (0)", "Aprobado (1)"]

    for idx, m_name in enumerate(model_names):
        cm = np.array(results_by_model[m_name]["confusion_matrix"])
        ax = axes_cm[idx]
        ax.imshow(cm, cmap=cmap)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels_cm, fontsize=9)
        ax.set_yticklabels(labels_cm, fontsize=9)
        ax.set_xlabel("Predicho")
        if idx == 0:
            ax.set_ylabel("Real")
        ax.set_title(f"{m_name}\n{subtitle_map[m_name]}", fontsize=10.5, fontweight="bold")
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    f"{cm[r, c]:,}",
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > thresh else "black",
                    fontsize=11,
                    fontweight="bold",
                )
    plt.tight_layout()
    st.pyplot(fig_cm)
    plt.close(fig_cm)


def plot_correlation_heatmap_and_target_bars(corr_info: dict, show_all_corr: bool) -> None:
    """Dibuja el mapa de calor de Pearson y las barras de relación directa con loan_status."""
    all_cols = corr_info["columns"]
    corr_df_full = pd.DataFrame(corr_info["matrix"], index=all_cols, columns=all_cols)
    target_corr = pd.Series(corr_info["target_correlations"]).sort_values(ascending=False)

    if show_all_corr:
        selected_cols = all_cols
    else:
        key_feats = [
            col for col in target_corr.index
            if col in NUMERICAL_COLUMNS or abs(target_corr[col]) >= 0.05
        ]
        selected_cols = key_feats + ["loan_status"]

    corr_sub = corr_df_full.loc[selected_cols, selected_cols]

    hcol1, hcol2 = st.columns([1.45, 1])
    with hcol1:
        n_vars = len(selected_cols)
        fig_hm, ax_hm = plt.subplots(
            figsize=(10.5 if not show_all_corr else 13.0, 8.2 if not show_all_corr else 10.5)
        )
        im = ax_hm.imshow(corr_sub.values, cmap="RdBu", vmin=-1, vmax=1)
        ax_hm.set_xticks(range(n_vars))
        ax_hm.set_yticks(range(n_vars))
        ax_hm.set_xticklabels(selected_cols, rotation=45, ha="right", fontsize=8 if not show_all_corr else 7)
        ax_hm.set_yticklabels(selected_cols, fontsize=8 if not show_all_corr else 7)
        ax_hm.set_title("Mapa de Calor de Correlaciones (Características vs loan_status)", fontsize=11, fontweight="bold")
        cbar = fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("Correlación (r)", fontsize=9)

        font_cell = 6.8 if not show_all_corr else 5.2
        for r in range(n_vars):
            for c in range(n_vars):
                val = corr_sub.values[r, c]
                ax_hm.text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color="white" if abs(val) > 0.55 else "black",
                    fontsize=font_cell,
                )
        plt.tight_layout()
        st.pyplot(fig_hm)
        plt.close(fig_hm)

    with hcol2:
        bar_series = target_corr.loc[[c for c in selected_cols if c != "loan_status"]].sort_values(ascending=True)
        fig_tc, ax_tc = plt.subplots(figsize=(6.8, 7.2 if not show_all_corr else 9.0))
        colors_tc = ["#DC2626" if v < 0 else "#1D4ED8" for v in bar_series.values]
        bars_tc = ax_tc.barh(bar_series.index, bar_series.values, color=colors_tc)
        ax_tc.axvline(0, color="black", linewidth=0.9)
        x_min_tc = min(-0.68, float(bar_series.min()) - 0.14)
        x_max_tc = max(0.52, float(bar_series.max()) + 0.12)
        ax_tc.set_xlim(x_min_tc, x_max_tc)
        ax_tc.set_xlabel("Correlación con loan_status (Clase 0 ← 0 → Clase 1)")
        ax_tc.set_title("Relación Directa con loan_status", fontsize=11, fontweight="bold")
        ax_tc.tick_params(axis="y", labelsize=8 if not show_all_corr else 7)
        for bar, val in zip(bars_tc, bar_series.values):
            offset = 0.015 if val >= 0 else -0.015
            ha = "left" if val >= 0 else "right"
            ax_tc.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
        ax_tc.grid(axis="x", linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig_tc)
        plt.close(fig_tc)


def plot_grouped_status_pct(df_eda: pd.DataFrame, col: str, title: str, xlabel: str) -> None:
    """Dibuja gráfica de barras agrupadas (% Denegado 0 vs Aprobado 1) para una variable categórica."""
    ct = pd.crosstab(df_eda[col], df_eda["loan_status"], normalize="index") * 100.0
    for cls in (0, 1):
        if cls not in ct.columns:
            ct[cls] = 0.0
    ct = ct[[0, 1]]

    cats = list(ct.index)
    x = np.arange(len(cats))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bars_0 = ax.bar(x - width / 2, ct[0].values, width, label="Denegado (0)", color="#DC2626")
    bars_1 = ax.bar(x + width / 2, ct[1].values, width, label="Aprobado (1)", color="#059669")

    for bars in (bars_0, bars_1):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=20 if len(cats) > 4 else 0, ha="right" if len(cats) > 4 else "center", fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Porcentaje de Solicitudes (%)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(title="loan_status", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_status_boxplot(df_eda: pd.DataFrame, col: str, title: str, ylabel: str) -> None:
    """Dibuja boxplot comparando Denegado (0) vs Aprobado (1) para una variable numérica."""
    vals_0 = df_eda.loc[df_eda["loan_status"] == 0, col].values
    vals_1 = df_eda.loc[df_eda["loan_status"] == 1, col].values

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bp = ax.boxplot(
        [vals_0, vals_1],
        tick_labels=["Denegado (0)", "Aprobado (1)"],
        patch_artist=True,
        widths=0.45,
        medianprops={"color": "#111827", "linewidth": 1.8},
    )
    for patch, color in zip(bp["boxes"], ["#FCA5A5", "#6EE7B7"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xlabel("Estado del Préstamo (loan_status)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_income_vs_loan_scatter(df_scat: pd.DataFrame, inc_cap: float) -> None:
    """Dibuja el scatter 2D de Ingreso Anual vs Monto Solicitado por loan_status."""
    fig_sc, ax_sc = plt.subplots(figsize=(6.8, 4.4))
    for cls, label_txt, color_hex in [
        (0, "Denegado (0)", "#DC2626"),
        (1, "Aprobado (1)", "#059669"),
    ]:
        sub_s = df_scat[df_scat["loan_status"] == cls]
        ax_sc.scatter(
            sub_s["person_income"],
            sub_s["loan_amnt"],
            s=14,
            alpha=0.45,
            color=color_hex,
            label=label_txt,
        )
    ax_sc.set_xlim(0, inc_cap)
    ax_sc.set_xlabel("Ingreso Anual (USD, hasta percentil 95)")
    ax_sc.set_ylabel("Monto del Préstamo (USD)")
    ax_sc.set_title("Ingreso Anual vs Monto Solicitado por loan_status", fontsize=11, fontweight="bold")
    ax_sc.legend(title="loan_status", fontsize=8.5)
    ax_sc.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close(fig_sc)


def plot_income_ratio_approval_bars(df_eda: pd.DataFrame) -> pd.Series:
    """Dibuja la tasa de aprobación por tramo de `loan_percent_income` (General vs Sin Mora) y retorna `rate_all`."""
    df_lpi = df_eda[["loan_percent_income", "previous_loan_defaults_on_file", "loan_status"]].copy()
    lpi_bins = [0.0, 0.10, 0.20, 0.30, 0.40, 1.0]
    lpi_labels = ["0–10%", "10–20%", "20–30%", "30–40%", "> 40%"]
    df_lpi["lpi_bin"] = pd.cut(df_lpi["loan_percent_income"], bins=lpi_bins, labels=lpi_labels, include_lowest=True)

    rate_all = df_lpi.groupby("lpi_bin", observed=False)["loan_status"].mean().fillna(0.0) * 100.0
    df_no_def = df_lpi[df_lpi["previous_loan_defaults_on_file"] == "No"]
    rate_nodef = df_no_def.groupby("lpi_bin", observed=False)["loan_status"].mean().fillna(0.0) * 100.0

    x_b = np.arange(len(lpi_labels))
    w_b = 0.36
    fig_lpi, ax_lpi = plt.subplots(figsize=(6.8, 4.5))
    b_all = ax_lpi.bar(
        x_b - w_b / 2,
        rate_all.values,
        w_b,
        label="Todas las solicitudes (con y sin mora)",
        color="#1D4ED8",
    )
    b_nd = ax_lpi.bar(
        x_b + w_b / 2,
        rate_nodef.values,
        w_b,
        label="Solo sin mora previa (defaults = 'No')",
        color="#059669",
    )
    for bars in (b_all, b_nd):
        for b in bars:
            h = b.get_height()
            ax_lpi.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)
    ax_lpi.set_xticks(x_b)
    ax_lpi.set_xticklabels(lpi_labels)
    ax_lpi.set_ylim(0, 115)
    ax_lpi.set_xlabel("% del Ingreso Comprometido (loan_percent_income = Monto / Ingreso)")
    ax_lpi.set_ylabel("% de Préstamos Aprobados (Clase 1)")
    ax_lpi.set_title("Aprobación según % de Ingreso Solicitado (General vs Sin Mora)", fontsize=10.5, fontweight="bold")
    ax_lpi.legend(loc="upper left", fontsize=8.2)
    ax_lpi.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig_lpi)
    plt.close(fig_lpi)
    return rate_all


def plot_random_forest_top_importance(rf_pipe) -> None:
    """Dibuja las 10 variables con mayor importancia en Random Forest."""
    raw_names = rf_pipe.named_steps["preprocesamiento"].get_feature_names_out()
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
    imp_vals = rf_pipe.named_steps["model"].feature_importances_ * 100.0
    top_imp = pd.Series(imp_vals, index=clean_names).sort_values(ascending=True).tail(10)

    fig_imp, ax_imp = plt.subplots(figsize=(6.8, 4.5))
    bar_colors = ["#059669" if v >= 10.0 else "#64748B" for v in top_imp.values]
    bars_imp = ax_imp.barh(top_imp.index, top_imp.values, color=bar_colors)
    for b, v in zip(bars_imp, top_imp.values):
        ax_imp.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center", ha="left", fontsize=8)
    ax_imp.set_xlim(0, max(top_imp.values) * 1.18)
    ax_imp.set_xlabel("Importancia en el Modelo Random Forest (%)")
    ax_imp.set_title("Variables que Más Afectan la Predicción (Top 10)", fontsize=10.8, fontweight="bold")
    ax_imp.tick_params(axis="y", labelsize=8)
    ax_imp.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig_imp)
    plt.close(fig_imp)


def plot_test_metrics_and_roc(optimized_test: dict, model_names: list[str]) -> None:
    """Dibuja las gráficas comparativas de métricas en Test Set y las curvas ROC lado a lado."""
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x = np.arange(len(metric_labels))
        width = 0.25

        for i, m_name in enumerate(model_names):
            vals = [optimized_test[m_name][k] for k in metric_keys]
            bars = ax_bar.bar(x + (i - 1) * width, vals, width, label=m_name, color=MODEL_PALETTE[i])
            for b in bars:
                h = b.get_height()
                ax_bar.text(b.get_x() + b.get_width() / 2, h + 0.01, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(metric_labels)
        ax_bar.set_ylim(0, 1.09)
        ax_bar.set_ylabel("Puntaje")
        ax_bar.set_title("Comparación de Métricas en Test Set")
        ax_bar.legend(loc="lower right", fontsize=8.5)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with gcol2:
        fig_roc, ax_roc = plt.subplots(figsize=(7, 4.5))
        for i, m_name in enumerate(model_names):
            roc_data = optimized_test[m_name]["roc_curve"]
            auc_val = optimized_test[m_name]["roc_auc"]
            ax_roc.plot(roc_data["fpr"], roc_data["tpr"], lw=2, color=MODEL_PALETTE[i], label=f"{m_name} (AUC = {auc_val:.4f})")

        ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2, label="Azar (AUC = 0.50)")
        ax_roc.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax_roc.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        ax_roc.set_title("Curvas ROC en Test Set")
        ax_roc.legend(loc="lower right", fontsize=8.5)
        ax_roc.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close(fig_roc)

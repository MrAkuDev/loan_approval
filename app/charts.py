"""Renderizado de gráficas 2D con Matplotlib para predicción, EDA y evaluación."""

import hashlib
import io
import json

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from app.config import (
    FEATURE_LABELS,
    FEATURE_SHORT_LABELS,
    MODEL_PALETTE,
    NUMERICAL_COLUMNS,
)


def _hash_payload(payload) -> str:
    """Genera una firma corta y determinista para cachear PNGs sin costo en re-renders."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:16]


def _df_sig(df: pd.DataFrame) -> tuple:
    """Firma O(1)/O(N suma simple) ultra-rápida para distinguir DataFrames en caché de figuras."""
    inc_sum = round(float(df["person_income"].sum()), 2) if "person_income" in df else 0.0
    loan_sum = round(float(df["loan_amnt"].sum()), 2) if "loan_amnt" in df else 0.0
    status_sum = int(df["loan_status"].sum()) if "loan_status" in df else -1
    return (len(df), df.shape[1], inc_sum, loan_sum, status_sum)


@st.cache_data(show_spinner=False, max_entries=96)
def _build_png_cached(cache_key: tuple, _builder_fn) -> bytes:
    """Renderiza una figura Matplotlib a bytes PNG una sola vez por clave."""
    fig = _builder_fn()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _display_cached_fig(cache_key: tuple, builder_fn) -> None:
    """Muestra inmediatamente la imagen PNG cacheada de una gráfica."""
    png_bytes = _build_png_cached(cache_key, builder_fn)
    st.image(png_bytes, use_container_width=True)


def plot_single_prediction_confidence(comp_rows: list[dict]) -> None:
    """Dibuja la gráfica horizontal de probabilidad de riesgo (Incumplimiento, Clase 1) por modelo."""
    fig, ax = plt.subplots(figsize=(6, 3.2))
    probs_plot = [r["Prob. Incumplimiento (Clase 1)"] * 100 for r in comp_rows]
    colors = ["#DC2626" if r["Clase Predicha"] == 1 else "#059669" for r in comp_rows]
    bars = ax.barh([r["Modelo"] for r in comp_rows], probs_plot, color=colors)
    ax.axvline(50, color="#374151", linestyle="--", linewidth=1, label="Umbral 50%")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Probabilidad de Incumplimiento (Clase 1) (%)")
    ax.set_title("Riesgo de Incumplimiento Estimado por Modelo")
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
    cache_key = (
        "cm_row",
        _hash_payload(
            {
                "cms": {m: results_by_model[m]["confusion_matrix"] for m in model_names},
                "cmap": cmap,
                "subs": subtitle_map,
            }
        ),
    )

    def _build():
        fig_cm, axes_cm = plt.subplots(1, len(model_names), figsize=(15, 4.3))
        labels_cm = ["Sin Incumplimiento (0)", "Incumplimiento (1)"]

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
        return fig_cm

    _display_cached_fig(cache_key, _build)


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
    corr_sig = _hash_payload({"cols": selected_cols, "tc": corr_info["target_correlations"]})

    hcol1, hcol2 = st.columns([1.45, 1])
    with hcol1:
        def _build_hm():
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
            return fig_hm

        _display_cached_fig(("corr_hm", corr_sig, show_all_corr), _build_hm)

    with hcol2:
        def _build_tc():
            bar_series = target_corr.loc[[c for c in selected_cols if c != "loan_status"]].sort_values(ascending=True)
            fig_tc, ax_tc = plt.subplots(figsize=(6.8, 7.2 if not show_all_corr else 9.0))
            colors_tc = ["#059669" if v < 0 else "#DC2626" for v in bar_series.values]
            bars_tc = ax_tc.barh(bar_series.index, bar_series.values, color=colors_tc)
            ax_tc.axvline(0, color="black", linewidth=0.9)
            x_min_tc = min(-0.55, float(bar_series.min()) - 0.14)
            x_max_tc = max(0.55, float(bar_series.max()) + 0.14)
            ax_tc.set_xlim(x_min_tc, x_max_tc)
            ax_tc.set_xlabel("Correlación con loan_status (Sin Incumplimiento 0 ← 0 → Incumplimiento 1)")
            ax_tc.set_title("Relación Directa con loan_status", fontsize=11, fontweight="bold")
            ax_tc.tick_params(axis="y", labelsize=8 if not show_all_corr else 7)
            for bar, val in zip(bars_tc, bar_series.values):
                offset = 0.015 if val >= 0 else -0.015
                ha = "left" if val >= 0 else "right"
                ax_tc.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}", va="center", ha=ha, fontsize=7.5)
            ax_tc.grid(axis="x", linestyle="--", alpha=0.35)
            plt.tight_layout()
            return fig_tc

        _display_cached_fig(("corr_tc", corr_sig, show_all_corr), _build_tc)


def plot_grouped_status_pct(df_eda: pd.DataFrame, col: str, title: str, xlabel: str) -> None:
    """Dibuja gráfica de barras agrupadas (% Sin Incumplimiento 0 vs Incumplimiento 1) para una variable categórica."""
    cache_key = ("grouped_pct", _df_sig(df_eda), col, title, xlabel)

    def _build():
        ct = pd.crosstab(df_eda[col], df_eda["loan_status"], normalize="index") * 100.0
        for cls in (0, 1):
            if cls not in ct.columns:
                ct[cls] = 0.0
        ct = ct[[0, 1]]

        cats = list(ct.index)
        x = np.arange(len(cats))
        width = 0.36

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        bars_0 = ax.bar(x - width / 2, ct[0].values, width, label="Sin Incumplimiento (0)", color="#059669")
        bars_1 = ax.bar(x + width / 2, ct[1].values, width, label="Incumplimiento (1)", color="#DC2626")

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
        return fig

    _display_cached_fig(cache_key, _build)


def plot_status_boxplot(df_eda: pd.DataFrame, col: str, title: str, ylabel: str) -> None:
    """Dibuja boxplot comparando Sin Incumplimiento (0) vs Incumplimiento (1) para una variable numérica."""
    cache_key = ("status_box", _df_sig(df_eda), col, title, ylabel)

    def _build():
        vals_0 = df_eda.loc[df_eda["loan_status"] == 0, col].dropna().values
        vals_1 = df_eda.loc[df_eda["loan_status"] == 1, col].dropna().values

        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        bp = ax.boxplot(
            [vals_0, vals_1],
            tick_labels=["Sin Incumplimiento (0)", "Incumplimiento (1)"],
            patch_artist=True,
            widths=0.45,
            medianprops={"color": "#111827", "linewidth": 1.8},
        )
        for patch, color in zip(bp["boxes"], ["#6EE7B7", "#FCA5A5"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_xlabel("Estado del Préstamo (loan_status)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        return fig

    _display_cached_fig(cache_key, _build)


def plot_income_vs_loan_scatter(df_scat: pd.DataFrame, inc_cap: float) -> None:
    """Dibuja el scatter 2D de Ingreso Anual vs Monto Solicitado por loan_status."""
    cache_key = ("income_loan_scat", _df_sig(df_scat), round(float(inc_cap), 4))

    def _build():
        fig_sc, ax_sc = plt.subplots(figsize=(6.8, 4.4))
        for cls, label_txt, color_hex in [
            (0, "Sin Incumplimiento (0)", "#059669"),
            (1, "Incumplimiento (1)", "#DC2626"),
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
        ax_sc.set_xlim(0, max(inc_cap, 1.0))
        ax_sc.set_xlabel("Ingreso Anual (USD, hasta percentil 95)")
        ax_sc.set_ylabel("Monto del Préstamo (USD)")
        ax_sc.set_title("Ingreso Anual vs Monto Solicitado por loan_status", fontsize=11, fontweight="bold")
        ax_sc.legend(title="loan_status", fontsize=8.5)
        ax_sc.grid(alpha=0.3)
        plt.tight_layout()
        return fig_sc

    _display_cached_fig(cache_key, _build)


def plot_income_ratio_default_bars(df_eda: pd.DataFrame) -> pd.Series:
    """Dibuja la tasa de incumplimiento (Incumplimiento %) por tramo de `loan_percent_income` (General vs Sin Mora) y retorna `rate_all`."""
    df_lpi = df_eda[["loan_percent_income", "cb_person_default_on_file", "loan_status"]].copy()
    lpi_bins = [0.0, 0.10, 0.20, 0.30, 0.40, 1.0]
    lpi_labels = ["0–10%", "10–20%", "20–30%", "30–40%", "> 40%"]
    df_lpi["lpi_bin"] = pd.cut(df_lpi["loan_percent_income"], bins=lpi_bins, labels=lpi_labels, include_lowest=True)

    rate_all = df_lpi.groupby("lpi_bin", observed=False)["loan_status"].mean().fillna(0.0) * 100.0
    cache_key = ("income_ratio_bars", _df_sig(df_eda))

    def _build():
        df_no_def = df_lpi[df_lpi["cb_person_default_on_file"] == "N"]
        rate_nodef = df_no_def.groupby("lpi_bin", observed=False)["loan_status"].mean().fillna(0.0) * 100.0

        x_b = np.arange(len(lpi_labels))
        w_b = 0.36
        fig_lpi, ax_lpi = plt.subplots(figsize=(6.8, 4.5))
        b_all = ax_lpi.bar(
            x_b - w_b / 2,
            rate_all.values,
            w_b,
            label="Todas las solicitudes (con y sin mora)",
            color="#DC2626",
        )
        b_nd = ax_lpi.bar(
            x_b + w_b / 2,
            rate_nodef.values,
            w_b,
            label="Solo sin mora previa (default_on_file = 'N')",
            color="#D97706",
        )
        for bars in (b_all, b_nd):
            for b in bars:
                h = b.get_height()
                ax_lpi.text(b.get_x() + b.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)
        ax_lpi.set_xticks(x_b)
        ax_lpi.set_xticklabels(lpi_labels)
        ax_lpi.set_ylim(0, 115)
        ax_lpi.set_xlabel("% del Ingreso Comprometido (loan_percent_income = Monto / Ingreso)")
        ax_lpi.set_ylabel("% de Incumplimiento (Clase 1)")
        ax_lpi.set_title("Tasa de Incumplimiento según % de Ingreso Solicitado (General vs Sin Mora)", fontsize=10.5, fontweight="bold")
        ax_lpi.legend(loc="upper left", fontsize=8.2)
        ax_lpi.grid(axis="y", linestyle="--", alpha=0.35)
        plt.tight_layout()
        return fig_lpi

    _display_cached_fig(cache_key, _build)
    return rate_all


def plot_random_forest_top_importance(rf_pipe) -> None:
    """Dibuja las 10 variables con mayor importancia en Random Forest."""
    imp_arr = rf_pipe.named_steps["model"].feature_importances_
    cache_key = ("rf_imp", len(rf_pipe.named_steps["model"].estimators_), round(float(imp_arr.sum()), 6), round(float(imp_arr.max()), 6))

    def _build():
        raw_names = rf_pipe.named_steps["preprocesamiento"].get_feature_names_out()
        clean_names = [n.replace("num__", "").replace("cat__", "") for n in raw_names]
        imp_vals = imp_arr * 100.0
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
        return fig_imp

    _display_cached_fig(cache_key, _build)


def plot_test_metrics_and_roc(optimized_test: dict, model_names: list[str]) -> None:
    """Dibuja las gráficas comparativas de métricas en Test Set y las curvas ROC lado a lado."""
    sig = _hash_payload(
        {
            m: {
                "accuracy": optimized_test[m]["accuracy"],
                "precision": optimized_test[m]["precision"],
                "recall": optimized_test[m]["recall"],
                "f1": optimized_test[m]["f1"],
                "roc_auc": optimized_test[m]["roc_auc"],
            }
            for m in model_names
        }
    )
    gcol1, gcol2 = st.columns(2)

    with gcol1:
        def _build_bar():
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
            return fig_bar

        _display_cached_fig(("test_bar", sig, tuple(model_names)), _build_bar)

    with gcol2:
        def _build_roc():
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
            return fig_roc

        _display_cached_fig(("test_roc", sig, tuple(model_names)), _build_roc)


def _format_params_subtitle(params: dict) -> str:
    """Formatea los hiperparámetros clave para el subtítulo de cada frontera."""
    if not params:
        return ""
    items = [f"{k}={v}" for k, v in params.items()]
    mid = (len(items) + 1) // 2
    if len(items) > 3:
        return ", ".join(items[:mid]) + "\n" + ", ".join(items[mid:])
    return ", ".join(items)


def plot_model_decision_boundaries(
    boundary_data: dict,
    feat_x: str,
    feat_y: str,
    eval_results: dict,
) -> None:
    """Dibuja las fronteras de decisión 2D reales de los 3 modelos (regiones + puntos + umbral P=0.5)."""
    prob_grids = boundary_data["prob_grids"]
    model_names = list(prob_grids.keys())
    if not model_names:
        return

    xx = boundary_data["xx"]
    yy = boundary_data["yy"]
    df_pts = boundary_data["scatter"]
    x_bounds = boundary_data["x_bounds"]
    y_bounds = boundary_data["y_bounds"]
    grid_sig = tuple((m, float(np.sum(prob_grids[m]))) for m in model_names)
    cache_key = ("decision_bounds", feat_x, feat_y, _df_sig(df_pts), x_bounds, y_bounds, grid_sig)

    def _build():
        region_cmap = LinearSegmentedColormap.from_list(
            "loan_decision_regions",
            ["#34D399", "#D1FAE5", "#F8FAFC", "#FEE2E2", "#F87171"],
        )

        fig, axes = plt.subplots(1, len(model_names), figsize=(16.5, 5.0), sharey=True)
        if len(model_names) == 1:
            axes = [axes]

        last_cf = None
        for idx, m_name in enumerate(model_names):
            ax = axes[idx]
            z_prob = prob_grids[m_name]

            last_cf = ax.contourf(
                xx,
                yy,
                z_prob,
                levels=np.linspace(0.0, 1.0, 21),
                cmap=region_cmap,
                alpha=0.82,
                vmin=0.0,
                vmax=1.0,
            )

            # Línea de frontera P = 0.50 con halo oscuro para resaltar sobre ambos fondos
            if float(z_prob.min()) < 0.5 < float(z_prob.max()):
                ax.contour(xx, yy, z_prob, levels=[0.5], colors=["#1E293B"], linewidths=3.0)
                ax.contour(xx, yy, z_prob, levels=[0.5], colors=["#FACC15"], linewidths=1.8)

            for cls, lbl, color_hex in [
                (0, "Sin Incumplimiento (0)", "#065F46"),
                (1, "Incumplimiento (1)", "#991B1B"),
            ]:
                sub = df_pts[df_pts["loan_status"] == cls]
                if not sub.empty:
                    ax.scatter(
                        sub[feat_x],
                        sub[feat_y],
                        s=15,
                        color=color_hex,
                        edgecolors="white",
                        linewidths=0.35,
                        alpha=0.88,
                        label=lbl if idx == 0 else None,
                        zorder=3,
                    )

            if idx == 0:
                ax.plot([], [], color="#FACC15", linewidth=2.2, label="Frontera P = 0.50")
                ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
                ax.set_ylabel(FEATURE_LABELS.get(feat_y, feat_y), fontsize=9.5)

            opt_params = eval_results.get("optimized_test", {}).get(m_name, {}).get("best_params", {})
            param_txt = _format_params_subtitle(opt_params)
            title = f"{m_name}\n{param_txt}" if param_txt else m_name
            ax.set_title(title, fontsize=9.8, fontweight="bold")
            ax.set_xlabel(FEATURE_LABELS.get(feat_x, feat_x), fontsize=9)
            ax.set_xlim(x_bounds)
            ax.set_ylim(y_bounds)
            ax.grid(alpha=0.2, linestyle="--")

        fig.subplots_adjust(right=0.92, wspace=0.14)
        if last_cf is not None:
            cbar_ax = fig.add_axes([0.935, 0.16, 0.014, 0.72])
            cbar = fig.colorbar(last_cf, cax=cbar_ax)
            cbar.set_label("Probabilidad de Incumplimiento P(loan_status = 1)", fontsize=9)
        return fig

    _display_cached_fig(cache_key, _build)


def plot_hyperparameter_heatmaps(
    tuning_heatmaps: dict,
    model_names: list[str],
    metric_key: str,
    metric_label: str,
) -> None:
    """Dibuja 3 heatmaps comparativos (uno por modelo) sobre grillas 2D de hiperparámetros."""
    available = [m for m in model_names if m in tuning_heatmaps]
    if not available:
        st.info("No hay grillas de hiperparámetros disponibles para graficar.")
        return

    cache_key = (
        "hp_heatmaps",
        metric_key,
        metric_label,
        tuple(available),
        _hash_payload({m: tuning_heatmaps[m][metric_key] for m in available}),
    )

    def _build():
        fig, axes = plt.subplots(1, len(available), figsize=(18.5, 4.9))
        if len(available) == 1:
            axes = [axes]

        for ax, m_name in zip(axes, available):
            hm = tuning_heatmaps[m_name]
            grid = np.array(hm[metric_key], dtype=float)
            x_labels = hm["x_labels"]
            y_labels = hm["y_labels"]

            im = ax.imshow(grid, cmap="viridis", aspect="auto")
            cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04)
            cbar.ax.tick_params(labelsize=8.5)

            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, fontsize=9)
            ax.set_yticks(range(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=9)
            ax.set_xlabel(hm["x_label"], fontsize=9.5)
            ax.set_ylabel(hm["y_label"], fontsize=9.5)
            ax.set_title(
                f"Heatmap {metric_label}: {m_name}\n({hm['y_param']} vs {hm['x_param']})",
                fontsize=10.5,
                fontweight="bold",
                pad=10,
            )

            for spine in ax.spines.values():
                spine.set_visible(False)

            vmin = float(np.nanmin(grid))
            vmax = float(np.nanmax(grid))
            span = vmax - vmin
            best_pair = hm.get("best_pair", {})

            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    val = float(grid[r, c])
                    norm_val = (val - vmin) / span if span > 1e-12 else 0.5
                    txt_color = "#111827" if norm_val >= 0.58 else "white"
                    ax.text(
                        c,
                        r,
                        f"{val:.4f}",
                        ha="center",
                        va="center",
                        color=txt_color,
                        fontsize=9.5,
                        fontweight="bold",
                    )
                    if y_labels[r] == best_pair.get("y") and x_labels[c] == best_pair.get("x"):
                        rect = plt.Rectangle(
                            (c - 0.48, r - 0.48),
                            0.96,
                            0.96,
                            fill=False,
                            edgecolor="#DC2626",
                            linewidth=2.5,
                        )
                        ax.add_patch(rect)

        plt.tight_layout(w_pad=2.4)
        return fig

    _display_cached_fig(cache_key, _build)


def plot_hyperparameter_effects(
    model_effects: dict,
    model_name: str,
    metric_key: str,
    metric_label: str,
) -> None:
    """Dibuja curvas 1D Train vs Validación (5-Fold CV) para cada hiperparámetro de un modelo."""
    param_list = model_effects.get("params", [])
    if not param_list:
        st.info(f"No hay curvas de efecto por hiperparámetro disponibles para {model_name}.")
        return

    cache_key = (
        "hp_effects",
        model_name,
        metric_key,
        metric_label,
        _hash_payload(param_list),
    )

    def _build():
        n_params = len(param_list)
        ncols = min(3, n_params)
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.15 * nrows), squeeze=False)
        axes_flat = axes.ravel()

        val_key = f"val_{metric_key}_mean"
        train_key = f"train_{metric_key}_mean"
        val_std_key = f"val_{metric_key}_std"

        for idx, (ax, p_info) in enumerate(zip(axes_flat, param_list)):
            labels = p_info["values"]
            x = np.arange(len(labels))
            y_val = np.array(p_info[val_key], dtype=float)
            y_train = np.array(p_info[train_key], dtype=float)

            ax.plot(
                x,
                y_train,
                color="#D97706",
                linestyle="--",
                marker="s",
                markersize=5.5,
                linewidth=1.9,
                label="Train (en fold)",
            )
            ax.plot(
                x,
                y_val,
                color="#2563EB",
                linestyle="-",
                marker="o",
                markersize=6.5,
                linewidth=2.3,
                label="Validación (5-Fold CV)",
            )

            if val_std_key in p_info:
                y_std = np.array(p_info[val_std_key], dtype=float)
                ax.fill_between(x, y_val - y_std, y_val + y_std, color="#2563EB", alpha=0.14)

            best_val = p_info.get("best_value")
            if best_val in labels:
                bx = labels.index(best_val)
                by = float(y_val[bx])
                ax.axvline(bx, color="#DC2626", linestyle=":", linewidth=1.7, alpha=0.85)
                ax.scatter(
                    [bx],
                    [by],
                    s=115,
                    color="#DC2626",
                    zorder=5,
                    marker="*",
                    label="Óptimo seleccionado",
                )

            all_y = np.concatenate([y_val, y_train])
            y_lo, y_hi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
            span = max(y_hi - y_lo, 0.02)
            pad_bot = max(0.04, span * 0.32)
            pad_top = max(0.045, span * 0.30)
            ax.set_ylim(max(-0.02, y_lo - pad_bot), min(1.05, y_hi + pad_top))

            val_offset = max(0.008, span * 0.075)
            for xi, (yv, yt) in enumerate(zip(y_val, y_train)):
                close_above = yt >= yv and (yt - yv) < span * 0.28
                place_below = yv > 0.92 or close_above
                ax.text(
                    xi,
                    yv - val_offset if place_below else yv + val_offset,
                    f"{yv:.3f}",
                    ha="center",
                    va="top" if place_below else "bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#1E3A8A",
                )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5, rotation=15 if len(labels) > 4 else 0)
            ax.set_xlim(-0.35, len(labels) - 0.65)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_title(p_info["label"], fontsize=10, fontweight="bold", pad=8)
            ax.grid(alpha=0.3, linestyle="--")

        for ax_empty in axes_flat[n_params:]:
            ax_empty.set_visible(False)

        handles, leg_labels = axes_flat[0].get_legend_handles_labels()
        legend_y = 0.915 if nrows == 1 else 0.955
        top_rect = 0.83 if nrows == 1 else 0.91
        fig.suptitle(
            f"{model_name} — Sensibilidad de {metric_label} por Hiperparámetro (Train vs. 5-Fold CV)",
            fontsize=11.5,
            fontweight="bold",
            y=0.985,
        )
        if handles:
            fig.legend(
                handles,
                leg_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, legend_y),
                ncol=len(handles),
                fontsize=8.5,
                framealpha=0.95,
            )
        plt.tight_layout(rect=[0, 0, 1, top_rect], h_pad=2.2, w_pad=2.0)
        return fig

    _display_cached_fig(cache_key, _build)


def _format_feature_value_label(col: str, val) -> str:
    """Formatea cada característica junto al valor ingresado para los ejes de explicabilidad."""
    label = FEATURE_SHORT_LABELS.get(col, col)
    if pd.isna(val):
        return f"{label} (N/D)"
    if col in {"person_income", "loan_amnt"}:
        return f"{label} (${float(val):,.0f})"
    if col == "loan_int_rate":
        return f"{label} ({float(val):.2f}%)"
    if col == "loan_percent_income":
        return f"{label} ({float(val):.2f})"
    if col == "person_emp_length":
        return f"{label} ({float(val):.1f})"
    if col in {"person_age", "cb_person_cred_hist_length"}:
        return f"{label} ({float(val):.0f})"
    return f"{label} ({val})"


def plot_instance_feature_contributions(
    attributions: dict[str, dict],
    X_input: pd.DataFrame,
) -> None:
    """Dibuja 3 gráficas horizontales (una por modelo) mostrando cuánto empuja cada variable hacia Incumplimiento/Sin Incumplimiento."""
    model_names = list(attributions.keys())
    if not model_names:
        return

    row_vals = X_input.iloc[0].to_dict()
    fig, axes = plt.subplots(1, len(model_names), figsize=(19.5, 6.2))
    if len(model_names) == 1:
        axes = [axes]

    for ax, m_name in zip(axes, model_names):
        info = attributions[m_name]
        contribs = info["contributions"]
        ordered_cols = sorted(contribs.keys(), key=lambda c: abs(contribs[c]))

        labels = [_format_feature_value_label(c, row_vals[c]) for c in ordered_cols]
        vals_pp = np.array([contribs[c] * 100.0 for c in ordered_cols], dtype=float)
        colors = ["#DC2626" if v >= 0 else "#059669" for v in vals_pp]

        bars = ax.barh(labels, vals_pp, color=colors, alpha=0.9)
        ax.axvline(0.0, color="#0F172A", linewidth=1.1)

        abs_max = max(float(np.max(np.abs(vals_pp))), 2.0)
        pad = abs_max * 0.04
        ax.set_xlim(-abs_max * 1.42, abs_max * 1.42)

        for b, v in zip(bars, vals_pp):
            offset = pad if v >= 0 else -pad
            ha = "left" if v >= 0 else "right"
            ax.text(
                v + offset,
                b.get_y() + b.get_height() / 2,
                f"{v:+.1f} pp",
                va="center",
                ha=ha,
                fontsize=8,
                fontweight="bold",
                color="#991B1B" if v >= 0 else "#065F46",
            )

        pred_lbl = "Incumplimiento (1)" if info["pred_class"] == 1 else "Sin Incumplimiento (0)"
        title_color = "#DC2626" if info["pred_class"] == 1 else "#059669"
        ax.set_title(
            f"{m_name}\n{pred_lbl} | Base: {info['base_value'] * 100:.1f}% → Pred: {info['pred_value'] * 100:.1f}%",
            fontsize=10.5,
            fontweight="bold",
            color=title_color,
            pad=8,
        )
        ax.set_xlabel("Impacto en P(Incumplimiento = 1) (pp)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout(w_pad=2.6)
    st.pyplot(fig)
    plt.close(fig)

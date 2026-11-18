"""Diseño de interfaz (frontend): tema Material 3 oscuro, navegación superior y tarjetas."""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* --- Ocultar panel lateral izquierdo y header nativo de Streamlit --- */
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* --- Material 3 Dark Surface Base --- */
    [data-testid="stAppViewContainer"],
    .stApp {
        background-color: #141218 !important;
        color: #E6E0E9 !important;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px !important;
        padding-top: 0.5rem !important;
        padding-bottom: 3rem !important;
        overflow-x: clip !important;
        overflow-y: visible !important;
    }
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
    [data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] {
        overflow: visible !important;
    }

    /* --- Sticky Top Navigation Bar --- */
    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > :has(.m3-brand-row),
    [data-testid="stMainBlockContainer"] > div > [data-testid="stVerticalBlock"] > :has(.m3-brand-row) {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
        background-color: rgba(20, 18, 24, 0.94) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        padding-top: 0.75rem !important;
        padding-bottom: 0.75rem !important;
        margin-bottom: 0.75rem !important;
        border-bottom: 1px solid #36343B !important;
    }

    /* --- Material 3 Card Containers (st.container(border=True)) --- */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #211F26 !important;
        border: 1px solid #49454F !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.4rem !important;
        margin-bottom: 1rem !important;
    }

    /* --- Ocultar elementos antiguos tenues mientras Streamlit transiciona --- */
    [data-stale="true"] {
        opacity: 0 !important;
        pointer-events: none !important;
        transition: opacity 0.05s linear !important;
    }

    /* --- Material 3 Loading Screen & Inline Loader --- */
    @keyframes m3-spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes m3-indeterminate {
        0% { transform: translateX(-100%); width: 42%; }
        50% { transform: translateX(65%); width: 55%; }
        100% { transform: translateX(230%); width: 42%; }
    }
    .m3-loader-overlay {
        position: fixed;
        inset: 0;
        z-index: 950;
        background: rgba(20, 18, 24, 0.90);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem;
    }
    .m3-loader-inline {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1.5rem 0;
    }
    .m3-loader-card {
        width: 100%;
        max-width: 440px;
        background-color: #211F26;
        border: 1px solid #49454F;
        border-radius: 20px;
        padding: 2rem 2.1rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.48);
        text-align: center;
    }
    .m3-spinner-ring {
        width: 44px;
        height: 44px;
        margin: 0 auto 1rem auto;
        border-radius: 50%;
        border: 4px solid #36343B;
        border-top-color: #D0BCFF;
        border-right-color: #6750A4;
        animation: m3-spin 0.85s linear infinite;
    }
    .m3-loader-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F4EFF4;
        margin-bottom: 0.35rem;
    }
    .m3-loader-subtitle {
        font-size: 0.9rem;
        color: #CAC4D0;
        margin-bottom: 1.1rem;
    }
    .m3-progress-track {
        width: 100%;
        height: 5px;
        border-radius: 999px;
        background-color: #36343B;
        overflow: hidden;
    }
    .m3-progress-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6750A4, #D0BCFF);
        animation: m3-indeterminate 1.25s ease-in-out infinite;
    }

    /* --- Top Navigation Brand & Title Pills --- */
    .m3-brand-row {
        display: flex;
        align-items: center;
        gap: 14px;
        min-height: 44px;
    }
    .m3-logo-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background-color: transparent;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .m3-title-pill {
        border: 1.5px solid #938F99;
        border-radius: 10px;
        padding: 7px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        color: #F4EFF4;
        background-color: #1D1B20;
        letter-spacing: 0.01em;
    }

    /* --- Tipografía y encabezados M3 --- */
    .m3-section-title {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F4EFF4;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }
    .m3-section-subtitle {
        font-size: 0.96rem;
        color: #CAC4D0;
        margin-bottom: 1rem;
    }

    /* --- Botones Material 3 redondeados --- */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        min-height: 40px !important;
        padding: 0.45rem 1.2rem !important;
        transition: all 0.16s ease !important;
    }

    /* --- Pills de navegación Top Nav, Menú Móvil y Toggle Lab (activo = blanco, inactivo = morado M3) --- */
    [class*="st-key-nav_btn_"] button[kind="primary"],
    [class*="st-key-nav_m_btn_"] button[kind="primary"],
    [class*="st-key-lab_btn_"] button[kind="primary"] {
        background-color: #FFFFFF !important;
        color: #6750A4 !important;
        border: 1px solid #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.28) !important;
    }
    [class*="st-key-nav_btn_"] button[kind="primary"]:hover,
    [class*="st-key-nav_m_btn_"] button[kind="primary"]:hover,
    [class*="st-key-lab_btn_"] button[kind="primary"]:hover {
        background-color: #F3EDF7 !important;
        color: #4F378B !important;
    }
    [class*="st-key-nav_btn_"] button[kind="secondary"],
    [class*="st-key-nav_m_btn_"] button[kind="secondary"],
    [class*="st-key-lab_btn_"] button[kind="secondary"] {
        background-color: #6750A4 !important;
        color: #FFFFFF !important;
        border: 1px solid #6750A4 !important;
    }
    [class*="st-key-nav_btn_"] button[kind="secondary"]:hover,
    [class*="st-key-nav_m_btn_"] button[kind="secondary"]:hover,
    [class*="st-key-lab_btn_"] button[kind="secondary"]:hover {
        background-color: #7F67BE !important;
        border-color: #7F67BE !important;
    }

    /* --- Menú Hamburguesa Responsivo (Tablets y Móviles <= 1024px) --- */
    @media (min-width: 1025px) {
        [class*="st-key-nav_burger_btn"],
        [class*="st-key-nav_m_btn_"] {
            display: none !important;
        }
    }
    @media (max-width: 1024px) {
        [data-testid="stHorizontalBlock"]:has(.m3-brand-row) {
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.65rem !important;
        }
        [data-testid="stHorizontalBlock"]:has(.m3-brand-row) > [data-testid="stColumn"]:first-child,
        [data-testid="stHorizontalBlock"]:has(.m3-brand-row) > [data-testid="column"]:first-child {
            flex: 1 1 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        [data-testid="stHorizontalBlock"]:has(.m3-brand-row) > [data-testid="stColumn"]:last-child,
        [data-testid="stHorizontalBlock"]:has(.m3-brand-row) > [data-testid="column"]:last-child {
            flex: 0 0 auto !important;
            width: auto !important;
            min-width: 0 !important;
        }
        [data-testid="stHorizontalBlock"]:has([class*="st-key-nav_btn_"]):not(:has(.m3-brand-row)) {
            display: none !important;
        }
        [class*="st-key-nav_burger_btn"] {
            display: flex !important;
            justify-content: flex-end !important;
        }
        [class*="st-key-nav_burger_btn"] button {
            width: 44px !important;
            min-width: 44px !important;
            min-height: 42px !important;
            padding: 0 !important;
            border-radius: 12px !important;
            font-size: 1.25rem !important;
            line-height: 1 !important;
            background-color: #2B2930 !important;
            color: #F4EFF4 !important;
            border: 1.5px solid #6750A4 !important;
        }
        [class*="st-key-nav_burger_btn"] button:hover {
            background-color: #6750A4 !important;
            color: #FFFFFF !important;
        }
        .m3-title-pill {
            padding: 6px 12px !important;
            font-size: 0.92rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
    }

    /* --- Botones de acción principal (Predict / CTA morado M3) --- */
    [data-testid="stFormSubmitButton"] > button,
    button[kind="primary"] {
        background-color: #6750A4 !important;
        color: #FFFFFF !important;
        border: 1px solid #7F67BE !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover,
    button[kind="primary"]:hover {
        background-color: #7F67BE !important;
    }

    /* --- Formularios e inputs dentro de superficie M3 --- */
    [data-testid="stForm"] {
        background-color: #2B2930 !important;
        border: 1px solid #49454F !important;
        border-radius: 14px !important;
        padding: 1.2rem !important;
    }
    [data-testid="stExpander"] {
        background-color: #1D1B20 !important;
        border: 1px solid #49454F !important;
        border-radius: 12px !important;
    }

    /* --- Tarjetas de predicción por modelo --- */
    .model-card-approved {
        background-color: #122B22;
        border: 1px solid #34D399;
        border-top: 5px solid #10B981;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: #ECFDF5;
    }
    .model-card-rejected {
        background-color: #2E1518;
        border: 1px solid #F87171;
        border-top: 5px solid #EF4444;
        border-radius: 14px;
        padding: 16px;
        text-align: center;
        color: #FEF2F2;
    }
    .model-card-title {
        font-weight: 700;
        font-size: 1.08rem;
        color: #F9FAFB;
    }
    .model-card-meta {
        font-size: 0.93rem;
        color: #E5E7EB;
    }
    .pred-badge-ok {
        font-size: 1.22rem;
        font-weight: 700;
        color: #6EE7B7;
        margin: 8px 0;
    }
    .pred-badge-no {
        font-size: 1.22rem;
        font-weight: 700;
        color: #FCA5A5;
        margin: 8px 0;
    }
</style>
"""

_ENGINE_SVG = """
<svg width="34" height="34" viewBox="0 0 24 24" fill="#6750A4"
     xmlns="http://www.w3.org/2000/svg">
  <rect x="8.8" y="2.4" width="7.2" height="2.6" rx="1.1"/>
  <rect x="11.4" y="4.9" width="2.0" height="2.1"/>
  <rect x="1.4" y="9.4" width="2.4" height="8.2" rx="1.0"/>
  <rect x="3.6" y="12.5" width="1.7" height="2.0"/>
  <rect x="20.2" y="9.4" width="2.4" height="8.2" rx="1.0"/>
  <rect x="18.4" y="12.5" width="2.0" height="2.0"/>
  <path fill-rule="evenodd" d="
    M 9.6 6.8 H 17.6 Q 18.7 6.8 18.7 7.9 V 17.6 L 15.2 21.0 H 9.2 L 6.3 18.1 H 5.1 V 9.7 H 6.6 Z
    M 12.4 9.2 L 14.1 9.2 L 12.0 13.1 L 14.4 13.1 L 10.3 18.9 L 11.7 14.4 L 9.5 14.4 Z
  "/>
</svg>
"""


def configure_page() -> None:
    """Configura los metadatos de página e inyecta el tema Material 3 oscuro."""
    st.set_page_config(
        page_title="Loan Approval Simulator — ML",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def _set_active_nav(page: str) -> None:
    """Callback previo al render para cambiar de pantalla y cerrar el menú móvil."""
    st.session_state["active_nav"] = page
    st.session_state["mobile_nav_open"] = False


def _toggle_mobile_menu() -> None:
    """Alterna la apertura del menú hamburguesa en tablet/móvil."""
    st.session_state["mobile_nav_open"] = not st.session_state.get("mobile_nav_open", False)


def _set_lab_section(section: str) -> None:
    """Callback previo al render para cambiar de subsección en Lab sin resetear datos."""
    st.session_state["lab_section"] = section


def render_top_navbar() -> str:
    """Renderiza la barra superior Material 3 (horizontal en desktop, hamburguesa en tablet/móvil)."""
    if "active_nav" not in st.session_state:
        st.session_state["active_nav"] = "home"
    if "mobile_nav_open" not in st.session_state:
        st.session_state["mobile_nav_open"] = False

    nav_items = [
        ("home", "Inicio", "home"),
        ("Lab", "Lab", "lab"),
        ("Glossary", "Acerca de", "glossary"),
    ]
    current = st.session_state["active_nav"]

    with st.container():
        nav_left, nav_right = st.columns([1.35, 1.0], vertical_alignment="center")

        with nav_left:
            st.markdown(
                f"""
                <div class="m3-brand-row">
                    <div class="m3-logo-box" title="Loan Approval Simulator">{_ENGINE_SVG}</div>
                    <div class="m3-title-pill">Loan Approval Simulator</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with nav_right:
            menu_open = st.session_state["mobile_nav_open"]
            st.button(
                "✕" if menu_open else "☰",
                key="nav_burger_btn",
                type="secondary",
                on_click=_toggle_mobile_menu,
                help="Abrir o cerrar menú de navegación",
            )

            desktop_cols = st.columns(len(nav_items), gap="small")
            for col, (page_id, label, slug) in zip(desktop_cols, nav_items):
                with col:
                    st.button(
                        label,
                        key=f"nav_btn_{slug}",
                        type="primary" if current == page_id else "secondary",
                        use_container_width=True,
                        on_click=_set_active_nav,
                        args=(page_id,),
                    )

        if st.session_state["mobile_nav_open"]:
            for page_id, label, slug in nav_items:
                st.button(
                    label,
                    key=f"nav_m_btn_{slug}",
                    type="primary" if current == page_id else "secondary",
                    use_container_width=True,
                    on_click=_set_active_nav,
                    args=(page_id,),
                )

    return st.session_state["active_nav"]


def render_hyperparameters_list(eval_results: dict) -> None:
    """Renderiza la lista de hiperparámetros óptimos de los tres modelos."""
    opt_svm = eval_results["optimized_test"]["SVM RBF"]["best_params"]
    opt_rf = eval_results["optimized_test"]["Random Forest"]["best_params"]
    opt_lr = eval_results["optimized_test"]["Logistic Regression"]["best_params"]
    st.markdown(
        f"""
        - **SVM RBF:** `{opt_svm}`
        - **Random Forest:** `{opt_rf}`
        - **Logistic Regression:** `{opt_lr}`
        """
    )


def render_about_card(eval_results: dict | None = None) -> None:
    """Renderiza la tarjeta superior en la pantalla Inicio con ficha académica y modelos."""
    with st.container(border=True):
        st.markdown('<div class="m3-section-title">información</div>', unsafe_allow_html=True)

        col_info, col_models = st.columns([1.35, 1.0], gap="large")
        with col_info:
            st.markdown(
                """
                - **Proyecto:** Clasificación de Riesgo Crediticio (Credit Risk Dataset)
                - **Alumno:** Joel Arturo Álvarez Hernández (`403233`)
                - **Institución:** Universidad Autónoma de Chihuahua (UACH)
                - **Facultad:** Facultad de Ingeniería
                - **Programa / Materia:** Maestría — Machine Learning
                - **Asesora:** Dra. Olanda Prieto Ordaz
                """
            )

        with col_models:
            st.markdown("##### Modelos Integrados")
            st.markdown(
                """
                1. **SVM (Kernel RBF)**
                2. **Random Forest**
                3. **Regresión Logística**
                """
            )
            st.caption("Preprocesamiento unificado vía `Pipeline` + `ColumnTransformer` (`joblib`).")


def render_lab_header(eval_results: dict | None = None) -> str:
    """Renderiza el encabezado 'Lab' con toggle ('Local' / 'Try yours') y detalle de la base de datos."""
    if "lab_section" not in st.session_state:
        st.session_state["lab_section"] = "Local"

    with st.container(border=True):
        c_title, c_toggle, _ = st.columns([0.55, 1.3, 1.15], vertical_alignment="center")
        with c_title:
            st.markdown('<div class="m3-section-title" style="margin-bottom:0;">Lab</div>', unsafe_allow_html=True)

        with c_toggle:
            t_local, t_try = st.columns(2, gap="small")
            current_lab = st.session_state["lab_section"]

            with t_local:
                lbl_local = "✓ Local" if current_lab == "Local" else "Local"
                st.button(
                    lbl_local,
                    key="lab_btn_local",
                    type="primary" if current_lab == "Local" else "secondary",
                    use_container_width=True,
                    on_click=_set_lab_section,
                    args=("Local",),
                )
            with t_try:
                lbl_try = "✓ Try yours" if current_lab == "Try yours" else "Try yours"
                st.button(
                    lbl_try,
                    key="lab_btn_try_yours",
                    type="primary" if current_lab == "Try yours" else "secondary",
                    use_container_width=True,
                    on_click=_set_lab_section,
                    args=("Try yours",),
                )

        if eval_results is not None:
            ds_info = eval_results["dataset"]
            cd = ds_info.get("class_distribution", {})
            c0_n = cd.get("0_no_default", cd.get("0_no_aprobado", 0))
            c1_n = cd.get("1_default", cd.get("1_aprobado", 0))
            tot_n = max(ds_info.get("total_rows", c0_n + c1_n), 1)
            raw_n = ds_info.get("raw_rows", tot_n)
            dup_n = ds_info.get("duplicates_removed", 0)
            miss_n = ds_info.get("missing_rows_removed", 0)
            out_n = ds_info.get("outliers_removed", 0)
            with st.expander("📊 Ver detalle de la base de datos e hiperparámetros óptimos"):
                c_ds, c_hp = st.columns(2)
                with c_ds:
                    st.markdown(
                        f"""
                        - **Origen:** Kaggle — *Credit Risk Dataset* (`credit_risk_dataset.csv`)
                        - **Registros:** **{raw_n:,}** brutos → **{tot_n:,}** limpios × **12** columnas (−{dup_n} duplicados, −{miss_n:,} filas vacías, −{out_n} atípicos)
                        - **Calidad:** **0 valores vacíos** (el valor numérico `0` se conserva como válido)
                        - **Partición:** 80% Train (**{ds_info['train_rows']:,}**) / 20% Test (**{ds_info['test_rows']:,}**) estratificada
                        - **Clases:** `0` Sin Incumplimiento (**{c0_n:,}**, {c0_n / tot_n * 100:.2f}%) vs `1` Incumplimiento (**{c1_n:,}**, {c1_n / tot_n * 100:.2f}%)
                        """
                    )
                with c_hp:
                    render_hyperparameters_list(eval_results)

    return st.session_state["lab_section"]


def render_missing_artifacts_error(missing_files: list[str]) -> None:
    """Muestra un error claro cuando faltan los archivos entrenados en `models/`."""
    st.error(
        "⚠️ Faltan archivos de modelos entrenados o resultados de evaluación. "
        "Ejecuta `python model_factory/train_model.py` primero para entrenar y guardar los tres modelos en `models/`."
    )
    if missing_files:
        st.code("\n".join(missing_files))
    st.stop()


def render_prediction_cards(preds: dict[str, dict]) -> None:
    """Renderiza las tarjetas comparativas de predicción para los 3 modelos."""
    m_cols = st.columns(3)
    for idx, (m_name, res) in enumerate(preds.items()):
        with m_cols[idx]:
            is_default = res["pred"] == 1
            card_cls = "model-card-rejected" if is_default else "model-card-approved"
            badge_cls = "pred-badge-no" if is_default else "pred-badge-ok"
            icon = "⚠️" if is_default else "✅"
            prob_1_pct = f"{res['prob_1'] * 100:.2f}%" if res["prob_1"] is not None else "N/D"
            prob_0_pct = f"{res['prob_0'] * 100:.2f}%" if res["prob_0"] is not None else "N/D"
            dec_txt = f"{res['decision_score']:+.4f}" if res["decision_score"] is not None else "N/A"

            st.markdown(
                f"""
                <div class="{card_cls}">
                    <div class="model-card-title">{m_name}</div>
                    <div class="{badge_cls}">{icon} {res['label']}</div>
                    <div class="model-card-meta">
                        <b>Prob. Sin Incumplimiento (0):</b> {prob_0_pct}<br>
                        <b>Prob. Incumplimiento (1):</b> {prob_1_pct}<br>
                        <b>Score de Decisión:</b> <code>{dec_txt}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _loader_card_html(title: str, subtitle: str, overlay: bool) -> str:
    """Construye el HTML de la tarjeta de carga Material 3 (pantalla completa o inline)."""
    wrap_cls = "m3-loader-overlay" if overlay else "m3-loader-inline"
    return (
        f'<div class="{wrap_cls}">'
        '<div class="m3-loader-card">'
        '<div class="m3-spinner-ring"></div>'
        f'<div class="m3-loader-title">{title}</div>'
        f'<div class="m3-loader-subtitle">{subtitle}</div>'
        '<div class="m3-progress-track"><div class="m3-progress-fill"></div></div>'
        "</div></div>"
    )


def show_loading_screen(slot, title: str, subtitle: str = "Cargando información y gráficas...") -> None:
    """Muestra la pantalla de carga Material 3 cubriendo el área debajo del navbar."""
    slot.markdown(_loader_card_html(title, subtitle, overlay=True), unsafe_allow_html=True)


def show_inline_loading(slot, title: str, subtitle: str = "Procesando con los 3 modelos...") -> None:
    """Muestra una tarjeta de carga Material 3 inline para cálculos locales en curso."""
    slot.markdown(_loader_card_html(title, subtitle, overlay=False), unsafe_allow_html=True)

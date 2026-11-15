"""Orquestador principal de la aplicación Streamlit dentro del paquete `app`."""

import os
import sys

import streamlit as st

if __package__ is None or __package__ == "":
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

from app.frontend import (
    configure_page,
    render_about_card,
    render_lab_header,
    render_missing_artifacts_error,
    render_top_navbar,
    show_loading_screen,
)
from app.services import load_artifacts
from app.tab_eval import render_tab_eval
from app.tab_interp import render_tab_interp
from app.tab_predict import render_tab_predict
from app.tab_upload import render_tab_upload

_SCREEN_LOAD_TITLES = {
    "home:": ("Cargando Inicio", "Preparando ficha del proyecto y simulador individual..."),
    "Lab:Local": ("Cargando Lab — Local", "Generando evaluación experimental, fronteras 2D y gráficas..."),
    "Lab:Try yours": ("Cargando Lab — Try yours", "Preparando cargador de base CSV y análisis por lotes..."),
    "Glossary:": ("Cargando Acerca de", "Cargando guía de interpretación y limitaciones..."),
}


def main() -> None:
    """Configura la página, carga modelos/métricas y renderiza la navegación Material 3."""
    configure_page()
    active_page = render_top_navbar()
    loader_slot = st.empty()
    screen_slot = st.empty()

    if not st.session_state.get("_artifacts_loaded"):
        show_loading_screen(
            loader_slot,
            "Iniciando Loan Approval Simulator",
            "Cargando modelos SVM RBF, Random Forest y Regresión Logística...",
        )

    models, missing_files, eval_results = load_artifacts()
    st.session_state["_artifacts_loaded"] = True

    if missing_files or eval_results is None:
        loader_slot.empty()
        render_missing_artifacts_error(missing_files)

    view_key = f"{active_page}:{st.session_state.get('lab_section', 'Local') if active_page == 'Lab' else ''}"
    is_switching = st.session_state.get("_last_rendered_view") != view_key
    if is_switching:
        screen_slot.empty()
        title_msg, sub_msg = _SCREEN_LOAD_TITLES.get(view_key, ("Cargando vista", "Preparando componentes..."))
        show_loading_screen(loader_slot, title_msg, sub_msg)
        st.session_state["_last_rendered_view"] = view_key

    with screen_slot.container():
        if active_page == "home":
            render_about_card(eval_results)
            render_tab_predict(models, eval_results)
        elif active_page == "Lab":
            lab_section = render_lab_header(eval_results)
            if lab_section == "Local":
                with st.container(border=True):
                    render_tab_eval(models, eval_results)
            else:
                render_tab_upload(models, eval_results)
        elif active_page == "Glossary":
            render_tab_interp()
        else:
            st.session_state["active_nav"] = "home"
            render_about_card(eval_results)
            render_tab_predict(models, eval_results)

    loader_slot.empty()


if __name__ == "__main__":
    main()

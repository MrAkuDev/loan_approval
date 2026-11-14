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
    render_header,
    render_missing_artifacts_error,
    render_sidebar,
)
from app.services import load_artifacts
from app.tab_eval import render_tab_eval
from app.tab_info import render_tab_info
from app.tab_interp import render_tab_interp
from app.tab_predict import render_tab_predict
from app.tab_upload import render_tab_upload


def main() -> None:
    """Configura la página, carga modelos/métricas y renderiza las 5 pestañas."""
    configure_page()
    models, missing_files, eval_results = load_artifacts()
    render_sidebar()
    render_header()

    if missing_files or eval_results is None:
        render_missing_artifacts_error(missing_files)

    tab_info, tab_predict, tab_upload, tab_eval, tab_interp = st.tabs(
        [
            "ℹ️ Información del Proyecto",
            "🔮 Predicción Individual",
            "📂 Upload Database",
            "📊 Evaluación Experimental",
            "📘 Interpretación y Limitaciones",
        ]
    )

    with tab_info:
        render_tab_info(eval_results)
    with tab_predict:
        render_tab_predict(models)
    with tab_upload:
        render_tab_upload(models, eval_results)
    with tab_eval:
        render_tab_eval(models, eval_results)
    with tab_interp:
        render_tab_interp()


if __name__ == "__main__":
    main()

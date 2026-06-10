"""DissolvA page module: Analytical Settings. Extracted from app.py (Phase 3b modularization)."""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
try:
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False
from dissolva.theme import OXFORD, AMBER, PALETTE, style_ax
from dissolva.models import (MODEL_DEFS, CATEGORIES, fit_model, compute_mdt,
    compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (current_tier, require_tier, _safe_profile_names,
    _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape


def render():
    cfg = st.session_state.method_cfg
    cfg = st.session_state.method_cfg
    st.markdown(
        "<h2 style='color:#FFFFFF;margin:0 0 4px;'>Analytical Method Settings</h2>"
        "<p style='color:#9fb0d0;font-size:0.88rem;margin:0 0 20px;'>"
        "UV-Vis or chromatographic (HPLC/UPLC) method parameters. "
        "Included in the Excel report automatically.</p>",
        unsafe_allow_html=True
    )

    anal_opts = ["UV-Vis Spectrophotometry", "HPLC", "UPLC"]
    cur_anal = cfg.get("analytical", "UV-Vis Spectrophotometry")
    if cur_anal not in anal_opts:
        cur_anal = "UV-Vis Spectrophotometry"
    cfg["analytical"] = st.radio(
        "Analytical Method", anal_opts,
        horizontal=True,
        index=anal_opts.index(cur_anal))

    st.markdown("---")

    if cfg["analytical"] == "UV-Vis Spectrophotometry":
        st.markdown("### UV-Vis Parameters")
        c1, c2, c3 = st.columns(3)
        with c1:
            cfg["lambda_max"] = st.number_input(
                "λmax (nm)",
                value=float(cfg.get("lambda_max", 272.0)),
                min_value=190.0, max_value=900.0)
        with c2:
            cfg["slit_nm"] = st.number_input(
                "Slit Width (nm)",
                value=float(cfg.get("slit_nm", 2.0)),
                min_value=0.1, max_value=10.0)
        with c3:
            cfg["ref_wavelength"] = st.text_input(
                "Reference Wavelength (nm)",
                value=cfg.get("ref_wavelength", ""),
                placeholder="e.g. 700 (optional)")
        st.markdown(
            f'<div class="info-banner">UV detection at <strong>{cfg["lambda_max"]:.1f} nm</strong>, slit {cfg["slit_nm"]:.1f} nm</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(f"### {cfg['analytical']} Parameters")
        c1, c2 = st.columns(2)
        with c1:
            cfg["hplc_column"] = st.text_input(
                "Column", value=cfg.get("hplc_column", ""),
                placeholder="e.g. C18 150x4.6mm 5um")
        with c2:
            cfg["hplc_col_temp"] = st.number_input(
                "Column Temperature (°C)",
                value=float(cfg.get("hplc_col_temp", 30.0)),
                min_value=20.0, max_value=80.0)

        c3, c4, c5 = st.columns(3)
        with c3:
            cfg["hplc_mp_a"] = st.text_input(
                "Mobile Phase A", value=cfg.get("hplc_mp_a", ""),
                placeholder="e.g. 0.1% Formic acid/water")
        with c4:
            cfg["hplc_mp_b"] = st.text_input(
                "Mobile Phase B", value=cfg.get("hplc_mp_b", ""),
                placeholder="e.g. Acetonitrile")
        with c5:
            cfg["hplc_gradient"] = st.text_area(
                "Gradient Program",
                value=cfg.get("hplc_gradient", ""), height=68,
                placeholder="e.g. 0 min 10%B, 5 min 90%B, 8 min 10%B")

        c6, c7, c8 = st.columns(3)
        with c6:
            cfg["hplc_flow"] = st.number_input(
                "Flow Rate (mL/min)",
                value=float(cfg.get("hplc_flow", 1.0)),
                min_value=0.1, max_value=5.0, step=0.1)
        with c7:
            cfg["hplc_detection"] = st.number_input(
                "Detection Wavelength (nm)",
                value=float(cfg.get("hplc_detection", 254.0)),
                min_value=190.0, max_value=900.0)
        with c8:
            cfg["hplc_inj_vol"] = st.number_input(
                "Injection Volume (µL)",
                value=float(cfg.get("hplc_inj_vol", 20.0)),
                min_value=1.0, max_value=100.0)

        cfg["hplc_run_time"] = st.number_input(
            "Run Time (min)",
            value=float(cfg.get("hplc_run_time", 10.0)),
            min_value=1.0, max_value=120.0)

    st.session_state.method_cfg = cfg
    st.success("Settings saved automatically.")

# ===========================================================================
# PAGE: DATA INPUT
# ===========================================================================
# ===========================================================================
# GLOBAL: PubChem + FDA Active Substance Functions
# ===========================================================================

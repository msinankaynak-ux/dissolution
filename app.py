import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import curve_fit, root
from scipy.stats import norm as sp_norm
from scipy.integrate import trapezoid
import io

try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False

from dissolva.theme import OXFORD, inject_theme, style_ax
from dissolva.models import (MODEL_DEFS, CATEGORIES, fit_model,
    compute_mdt, compute_de, r2s, r2adj, aic_fn, msc_fn, _nz)
from dissolva.state import (init_session_state, TIER_RANK, current_tier, require_tier,
    _upgrade_cta, _safe_profile_names, _get_index, _rename_profile, _clear_all)
from dissolva.content import show_literature, show_all_references, analyze_profile_shape
from dissolva.pages import (method_settings, analytical_settings, all_references,
    data_input, kinetic_model_fitting, statistical_analysis, f1_f2_similarity,
    bootstrap_f2, ivivc, excel_report, api_information, academy)
from dissolva import auth

st.set_page_config(
    page_title="DissolvA - Predictive Dissolution Suite",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)


inject_theme()

# Feedback button at the bottom of the sidebar
with st.sidebar:
    pass  # top sidebar placeholder

init_session_state()


# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 16px 14px;margin-bottom:8px;">
      <div style="display:inline-flex;align-items:center;gap:12px;
                  background:rgba(255,191,0,0.05);border:1px solid rgba(255,191,0,0.22);
                  border-radius:12px;padding:10px 14px;">
        <div style="position:relative;width:44px;height:44px;background:#003171;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
          <div style="position:absolute;top:0;right:0;width:12px;height:3px;
                      background:#FFBF00;border-radius:0 10px 0 2px;"></div>
          <div style="position:absolute;top:0;right:0;width:3px;height:12px;
                      background:#FFBF00;border-radius:0 10px 0 2px;"></div>
          <span style="font-size:24px;font-weight:500;color:#FFBF00;line-height:1;
                       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">A</span>
        </div>
        <div style="display:flex;flex-direction:column;justify-content:center;gap:4px;">
          <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                      font-size:17px;font-weight:400;color:white;letter-spacing:-0.3px;line-height:1;">
            Dissolv<span style="font-weight:600;color:#FFBF00;">A</span><sup style="font-size:8px;color:white;font-weight:400;">™</sup>
          </div>
          <div style="font-size:8.5px;letter-spacing:1.4px;text-transform:uppercase;
                      color:rgba(255,191,0,0.55);font-weight:600;line-height:1;">
            Predictive Dissolution Suite
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Authentication (Streamlit yerleşik Google OIDC). Yapılandırma yoksa açık mod.
    auth.render_sidebar_auth()
    auth.sync_session()

    if "method_cfg" not in st.session_state:
        st.session_state.method_cfg = {
            "time_unit": "minutes", "conc_unit": "mg/mL", "dose_mg": 100.0,
            "q_time": 45.0, "q_limit": 80.0,
            "internal_spec_enabled": False,
            "internal_spec_time": 45.0,
            "internal_spec_limit": 85.0,
            "internal_spec_name": "Internal Spec",
            "apparatus": "USP II (Paddle)", "medium": "0.1N HCl (pH 1.2)",
            "rpm": 50, "volume_ml": 900, "temp_c": 37.0,
            "analytical": "UV-Vis",
            "lambda_max": 272.0, "slit_nm": 2.0, "ref_wavelength": "",
            "hplc_column": "", "hplc_flow": 1.0, "hplc_mp_a": "",
            "hplc_mp_b": "", "hplc_detection": 254.0,
            "hplc_inj_vol": 20.0, "hplc_col_temp": 30.0,
            "hplc_run_time": 10.0, "notes": "",
        }

    cfg = st.session_state.method_cfg

    # Expose variables globally for rest of app
    time_unit = cfg["time_unit"]
    conc_unit = cfg["conc_unit"]
    dose_mg   = cfg["dose_mg"]
    q_time    = cfg["q_time"]
    q_limit   = cfg["q_limit"]

    # Logical flow: Setup -> Analysis -> Report -> Reference.
    # Returned value is the canonical key; the displayed label is decorated via _nav_label (routing intact).
    _NAV_LABELS = {
        "IVIVC Analysis": "IVIVC Analysis  🚧",   # geçici devre dışı (yeniden yapımda)
    }
    def _nav_label(key: str) -> str:
        return _NAV_LABELS.get(key, key)

    nav = st.radio("Navigation", [
        # Setup
        "Method Settings", "Analytical Settings", "Data Input",
        # Analysis
        "Kinetic Model Fitting", "Statistical Analysis",
        "f1 and f2 Similarity", "Bootstrap f2 Analysis", "IVIVC Analysis",
        # Report
        "Excel Report",
        # Reference
        "💊 API Information", "📚 All References",
    ], format_func=_nav_label, label_visibility="collapsed",
       on_change=lambda: st.session_state.update(academy_open=False))

    # Active substance sidebar badge
    _as_sb = st.session_state.get("active_substance", {})
    if _as_sb.get("fetch_done") and _as_sb.get("name"):
        _pc_sb = _as_sb.get("pubchem") or {}
        _bcs_sb = _as_sb.get("bcs_class")
        st.markdown(
            f'<div style="background:rgba(0,33,71,0.6);border:1px solid rgba(255,191,0,0.25);'
            f'border-radius:8px;padding:8px 12px;margin-top:8px;">' 
            f'<div style="font-size:9px;font-weight:700;color:#FFBF00;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:4px;">💊 API Loaded</div>'
            f'<div style="font-size:12px;font-weight:600;color:white;">{_as_sb["name"]}</div>'
            f'<div style="font-size:10px;color:rgba(255,255,255,0.55);margin-top:2px;">'
            f'{_pc_sb.get("formula","")}' 
            f' · MW {_pc_sb.get("mw","")} g/mol</div>'
            f'<div style="margin-top:5px;display:flex;gap:4px;flex-wrap:wrap;">'
            f'<span style="background:rgba(255,191,0,0.15);color:#FFBF00;font-size:9px;'
            f'padding:1px 6px;border-radius:10px;">FDA: {len(_as_sb.get("fda_methods",[]))} methods</span>'
            + (f'<span style="background:rgba(39,174,96,0.2);color:#27ae60;font-size:9px;'
               f'padding:1px 6px;border-radius:10px;">Lit. validated</span>' 
               if _as_sb.get("fda_methods") else '') +
            f'</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<hr style="border:1px solid rgba(255,191,0,0.15);margin:10px 0 6px 0;">', unsafe_allow_html=True)

    st.markdown("""<style>
    div[data-testid="stSidebarContent"] div.stButton > button {
        background: transparent !important;
        border: 1px solid rgba(255,191,0,0.35) !important;
        color: rgba(255,191,0,0.75) !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        border-radius: 8px !important;
    }
    div[data-testid="stSidebarContent"] div.stButton > button:hover {
        border-color: #FFBF00 !important;
        color: #FFBF00 !important;
        background: rgba(255,191,0,0.06) !important;
    }
    </style>""", unsafe_allow_html=True)
    # New Session button
    if st.button("✦ New Session", use_container_width=True,
                 help="Clear all profiles, results and start fresh."):
        st.session_state["_confirm_new"] = True
        st.rerun()

    if st.session_state.get("_confirm_new"):
        st.warning("All data will be lost!", icon="⚠️")
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("✅ Confirm", use_container_width=True):
                _clear_all()
                st.session_state["_confirm_new"] = False
                st.rerun()
        with cc2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state["_confirm_new"] = False
                st.rerun()

    # DissolvA Academy — free / public, open to everyone (outside the paywall)
    if st.button("🎓 DissolvA Academy", use_container_width=True,
                 help="Free kinetic-model school — open to everyone, no account needed."):
        st.session_state.academy_open = True
        st.rerun()

    # Data privacy — same button style as above; opens a dialog with the statement
    @st.dialog("🔒 Data privacy")
    def _privacy_dialog():
        st.markdown(
            "**DissolvA does not save, log, or store your data.**\n\n"
            "- The dissolution data, profiles, and results you enter live only in session "
            "memory and are **erased** when you close or refresh the page. There is no database.\n"
            "- The app runs on Streamlit Community Cloud infrastructure, where data is processed "
            "in memory only and is **not persisted by us**.\n"
            "- The **only** thing sent off your device is the **drug name** you type on the "
            "**API Information** page — used to query public databases (PubChem · FDA · PubMed) "
            "and scite.ai. **Your dissolution data is never sent anywhere.**"
        )

    if st.button("🔒 Data privacy", use_container_width=True,
                 help="How DissolvA handles your data."):
        _privacy_dialog()

    # Feedback link
    st.markdown(
        '''<div style="padding:6px 12px;margin-top:4px;">
        <a href="https://tally.so/r/44oM55" target="_blank"
           style="display:flex;align-items:center;gap:8px;text-decoration:none;
                  color:rgba(232,224,208,0.4);font-size:0.75rem;"
           onmouseover="this.style.color='#FFBF00'" onmouseout="this.style.color='rgba(232,224,208,0.4)'">
          <span>✉</span><span>Share Feedback</span>
        </a></div>''',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center;padding:8px 0 4px 0;">'
        '<div style="font-size:0.6rem;color:#4a5a70;">DissolvA™ v3.0 &nbsp;|&nbsp; 2026 &nbsp;|&nbsp; Powered by AI</div>'
        '</div>',
        unsafe_allow_html=True
    )


# --- Constants ---


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown(
    '<h1 style="margin:0;font-size:2.4rem;color:#002147;">' +
    'DissolvA<sup style="font-size:1rem;">(TM)</sup> ' +
    '<span style="font-size:1rem;color:#888;font-style:italic;font-weight:400;">' +
    '- Predictive Dissolution Suite</span></h1>' +
    '<div style="color:#5a6480;font-size:0.9rem;margin-top:4px;">' +
    'FDA-Compliant - 62 Kinetic Models - Statistical Profiling - IVIVC</div>',
    unsafe_allow_html=True
)
st.markdown('<hr style="border:1px solid #FFBF00;margin:10px 0 4px 0;">', unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.76rem;color:#8aadcc;padding:3px 0 14px 0;'>"
    "<strong style='color:#5a8ab0;'>DissolvA Team</strong>"
    ""
    " &nbsp;&bull;&nbsp; "
    "<a href='mailto:dissolva.app@gmail.com' style='color:#7a9dbf;text-decoration:none;'>"
    "dissolva.app@gmail.com</a>"
    "</div>",
    unsafe_allow_html=True
)

# ===========================================================================
# MAIN DISPATCH
# ===========================================================================
# DissolvA Academy — free, open to everyone (no tier gate); opened via the sidebar button.
if st.session_state.get("academy_open"):
    if st.button("← Back to app"):
        st.session_state.academy_open = False
        st.rerun()
    academy.render()
    st.stop()

if nav == "Method Settings":
    method_settings.render()
elif nav == "Analytical Settings":
    analytical_settings.render()
if nav == "Data Input":
    data_input.render()
elif nav == "Kinetic Model Fitting":
    kinetic_model_fitting.render()
elif nav == "Statistical Analysis":
    statistical_analysis.render()
elif nav == "f1 and f2 Similarity":
    f1_f2_similarity.render()
elif nav == "Bootstrap f2 Analysis":
    bootstrap_f2.render()
elif nav == "IVIVC Analysis":
    ivivc.render()
elif nav == "Excel Report":
    excel_report.render()
elif nav == "💊 API Information":
    api_information.render()
elif nav == "📚 All References":
    all_references.render()

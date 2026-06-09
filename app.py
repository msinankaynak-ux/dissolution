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
    bootstrap_f2, ivivc, excel_report, api_information, academy, admin)
from dissolva import auth, engine_client, extras


def _admin_emails():
    try:
        e = st.secrets.get("admin", {}).get("emails")
        if e:
            return {str(x).strip().lower() for x in e}
    except Exception:
        pass
    return {"msinankaynak@gmail.com"}  # default owner


def _is_admin():
    em = (auth.current_user() or {}).get("email") or ""
    return em.strip().lower() in _admin_emails()

st.set_page_config(
    page_title="DissolvA - Predictive Dissolution Suite",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded"
)


inject_theme()
extras.init_sentry()  # crash reporting (no-op without a DSN; never sends PII)

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
        "IVIVC Analysis": "IVIVC Analysis — Coming Soon",   # disabled / under construction
    }
    def _nav_label(key: str) -> str:
        return _NAV_LABELS.get(key, key)

    _nav_options = [
        # Setup
        "Method Settings", "Analytical Settings", "Data Input",
        # Analysis
        "Kinetic Model Fitting", "Statistical Analysis",
        "f1 and f2 Similarity", "Bootstrap f2 Analysis", "IVIVC Analysis",
        # Report
        "Excel Report",
        # Reference
        "API Information", "All References",
    ]
    _NAV_ICONS = {
        "Method Settings": "gear", "Analytical Settings": "sliders",
        "Data Input": "table", "Kinetic Model Fitting": "graph-up",
        "Statistical Analysis": "bar-chart-line", "f1 and f2 Similarity": "shuffle",
        "Bootstrap f2 Analysis": "dice-5", "IVIVC Analysis": "hourglass-split",
        "Excel Report": "file-earmark-spreadsheet", "API Information": "capsule",
        "All References": "book",
    }
    # Modern grouped sidebar nav via streamlit-option-menu, with a safe fallback to
    # st.radio if the optional component is unavailable (the app must never break).
    nav = None
    try:
        from streamlit_option_menu import option_menu
        _disp = [_nav_label(k) for k in _nav_options]
        _disp_to_key = {_nav_label(k): k for k in _nav_options}
        _cur_key = st.session_state.get("_nav_key", _nav_options[0])
        _idx = _nav_options.index(_cur_key) if _cur_key in _nav_options else 0
        _sel = option_menu(
            menu_title=None, options=_disp,
            icons=[_NAV_ICONS.get(k, "circle") for k in _nav_options],
            default_index=_idx, key="main_nav_menu",
            styles={
                "container": {"padding": "0", "background-color": "transparent"},
                "icon": {"color": "rgba(255,191,0,0.7)", "font-size": "14px"},
                "nav-link": {"font-size": "13px", "color": "rgba(255,255,255,0.78)",
                             "padding": "7px 12px", "margin": "2px 4px",
                             "border-radius": "6px",
                             "--hover-color": "rgba(255,191,0,0.10)"},
                "nav-link-selected": {"background-color": "#BA7517",
                                      "color": "#FAEEDA", "font-weight": "500"},
            },
        )
        nav = _disp_to_key.get(_sel, _nav_options[0])
    except Exception:
        nav = st.radio("Navigation", _nav_options,
            format_func=_nav_label, label_visibility="collapsed")

    # Leave any full-screen overlay (Academy/Admin) when navigating to a page.
    if nav != st.session_state.get("_nav_key"):
        st.session_state["_nav_key"] = nav
        st.session_state.academy_open = False
        st.session_state.admin_open = False

    # Privacy-safe usage analytics: log a page-view once per page change (best-effort).
    if nav != st.session_state.get("_last_logged_page"):
        st.session_state["_last_logged_page"] = nav
        engine_client.log_event(nav, (auth.current_user() or {}).get("email") or "")

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

    # Load demo data — instant onboarding (Reference + Test profiles, ready to analyse)
    if st.button("⚗ Load demo data", use_container_width=True,
                 help="Load example Reference + Test profiles so you can try fitting and f2 instantly."):
        extras.load_demo_data()
        st.session_state["_nav_key"] = "Data Input"
        st.toast("Demo profiles loaded — open Data Input or Kinetic Model Fitting.", icon="⚗")
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
            "**Your dissolution data is never stored.**\n\n"
            "- The dissolution data, profiles, and results you enter live only in your session "
            "memory and are **erased** when you close or refresh the page. We have no database of "
            "your scientific data and never sell or share it.\n\n"
            "**During the free beta we do record a minimal amount to run the service and improve it:**\n"
            "- Your **account email and name** from Google sign-in (to create your free account).\n"
            "- Your **country**, derived once at sign-in (we store only the country, not your IP).\n"
            "- **Anonymous usage events** — which features you open/run (no scientific data, ever).\n\n"
            "**External lookups:** only the **drug name** you type on the *API Information* page is "
            "sent out, to query PubChem · FDA · PubMed (public databases) and **scite.ai** "
            "(a commercial citation service by Digital Science). Your dissolution data is never sent."
        )

    if st.button("🔒 Data privacy", use_container_width=True,
                 help="How DissolvA handles your data."):
        _privacy_dialog()

    # Cite this tool — APA + BibTeX (Zenodo-ready) for academic users
    if st.button("❝ Cite this tool", use_container_width=True,
                 help="Get an APA citation and BibTeX entry for DissolvA."):
        extras.citation_dialog()

    # Admin Console — shown ONLY to admin emails (st.secrets[admin][emails], default owner)
    if _is_admin():
        if st.button("🛡️ Admin Console", use_container_width=True,
                     help="Members & usage analytics (admins only)."):
            st.session_state.admin_open = True
            st.rerun()

    # Feedback — styled like the New Session / Academy / Data privacy buttons above
    st.markdown(
        '''<a href="https://tally.so/r/44oM55" target="_blank"
           style="display:block;text-align:center;background:transparent;
                  border:1px solid rgba(255,191,0,0.35);color:rgba(255,191,0,0.75);
                  font-size:0.78rem;font-weight:600;letter-spacing:0.5px;border-radius:8px;
                  padding:0.5rem 0.75rem;margin-top:4px;text-decoration:none;"
           onmouseover="this.style.borderColor='#FFBF00';this.style.color='#FFBF00';this.style.background='rgba(255,191,0,0.06)'"
           onmouseout="this.style.borderColor='rgba(255,191,0,0.35)';this.style.color='rgba(255,191,0,0.75)';this.style.background='transparent'">
          ✉ Share Feedback</a>''',
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
    'FDA/EMA guidance-aligned · 62 Kinetic Models · f1/f2 · Bootstrap f2 · Statistical Profiling' +
    '<span style="background:#eef2f7;color:#5a6480;font-size:0.7rem;font-weight:600;' +
    'padding:1px 8px;border-radius:10px;margin-left:8px;">BETA · research use only</span></div>',
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

# GDPR cookie/usage consent — dismissible, shown until accepted (session-level).
extras.consent_banner(_privacy_dialog)

# Thin workflow stepper (Setup -> Analysis -> Report -> Reference) reflecting the
# current page — makes the linear analysis flow visible. Hidden in overlays.
_PHASES = [
    ("Setup",     {"Method Settings", "Analytical Settings", "Data Input"}),
    ("Analysis",  {"Kinetic Model Fitting", "Statistical Analysis",
                   "f1 and f2 Similarity", "Bootstrap f2 Analysis", "IVIVC Analysis"}),
    ("Report",    {"Excel Report"}),
    ("Reference", {"API Information", "All References"}),
]
if not (st.session_state.get("academy_open") or st.session_state.get("admin_open")):
    _active = next((i for i, (_, ks) in enumerate(_PHASES) if nav in ks), 0)
    _chips = []
    for i, (label, _) in enumerate(_PHASES):
        if i == _active:
            bg, col, bd = "#BA7517", "#FAEEDA", "#BA7517"
        elif i < _active:
            bg, col, bd = "rgba(186,117,23,0.10)", "#854F0B", "rgba(186,117,23,0.5)"
        else:
            bg, col, bd = "transparent", "#9aa6b5", "rgba(150,166,181,0.4)"
        _chips.append(
            f'<span style="background:{bg};color:{col};border:1px solid {bd};'
            f'border-radius:14px;padding:3px 12px;font-size:0.72rem;font-weight:600;'
            f'white-space:nowrap;">{i+1}. {label}</span>')
    _sep = '<span style="color:#c3ccd6;">→</span>'
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;'
        'padding:2px 0 12px 0;">' + _sep.join(_chips) + '</div>',
        unsafe_allow_html=True)

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

# Admin Console — admin-only, opened via the sidebar button (not a nav peer).
if st.session_state.get("admin_open") and _is_admin():
    if st.button("← Back to app"):
        st.session_state.admin_open = False
        st.rerun()
    admin.render()
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
elif nav == "API Information":
    api_information.render()
elif nav == "All References":
    all_references.render()

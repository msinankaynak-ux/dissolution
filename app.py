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
                  background:rgba(255,204,0,0.05);border:1px solid rgba(255,204,0,0.22);
                  border-radius:12px;padding:10px 14px;">
        <div style="position:relative;width:44px;height:44px;background:#003171;border-radius:10px;
                    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
          <div style="position:absolute;top:0;right:0;width:12px;height:3px;
                      background:#FFCC00;border-radius:0 10px 0 2px;"></div>
          <div style="position:absolute;top:0;right:0;width:3px;height:12px;
                      background:#FFCC00;border-radius:0 10px 0 2px;"></div>
          <span style="font-size:24px;font-weight:500;color:#FFCC00 !important;line-height:1;
                       font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">A</span>
        </div>
        <div style="display:flex;flex-direction:column;justify-content:center;gap:4px;">
          <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                      font-size:17px;font-weight:400;color:#FFFFFF !important;letter-spacing:-0.3px;line-height:1;">
            Dissolv<span style="font-weight:600;color:#FFCC00 !important;">A</span><sup style="font-size:8px;color:#9fb0d0;font-weight:400;">™</sup>
          </div>
          <div style="font-size:8.5px;letter-spacing:1.4px;text-transform:uppercase;
                      color:rgba(255,191,0,0.55);font-weight:600;line-height:1;">
            Predictive Dissolution Suite
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # (Sign-in moved to the top header, right side.)

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
    # ── Categorized icon navigation (button-based; routing VALUES unchanged) ──
    # Each item = (routing_value, display_label, material_icon). Display labels may
    # differ from the canonical routing value; the page dispatch still uses the value.
    _NAV_CATEGORIES = [
        ("Configuration & Setup", [
            ("Method Settings",       "Method Settings",       ":material/settings:"),
            ("Analytical Settings",   "Analytical Settings",   ":material/science:"),
            ("Data Input",            "Data Input",            ":material/table_chart:"),
        ]),
        ("Predictive Analysis", [
            ("Kinetic Model Fitting", "Kinetic Model Fitting", ":material/show_chart:"),
            ("Statistical Analysis",  "Statistical Analysis",  ":material/functions:"),
            ("f1 and f2 Similarity",  "f1 & f2 Similarity",    ":material/compare_arrows:"),
            ("Bootstrap f2 Analysis", "Bootstrap f2",          ":material/sync:"),
            ("IVIVC Analysis",        "IVIVC Correlation",     ":material/link:"),
        ]),
        ("Results & Documentation", [
            ("Excel Report",          "Excel Reporting",       ":material/description:"),
            ("All References",        "References",            ":material/menu_book:"),
            ("API Information",       "API Information",        ":material/medication:"),
        ]),
    ]
    _nav_options = [v for _, _items in _NAV_CATEGORIES for v, _, _ in _items]

    # Programmatic navigation (e.g. the demo-data button) sets the target first.
    _pending = st.session_state.pop("_pending_nav", None)
    if _pending in _nav_options:
        st.session_state["main_nav_radio"] = _pending
    nav = st.session_state.get("main_nav_radio") or _nav_options[0]
    if nav not in _nav_options:
        nav = _nav_options[0]
    st.session_state["main_nav_radio"] = nav

    # Nav-row styling scoped to the navmenu container (extra .st-key-navmenu class →
    # higher specificity, so it overrides the generic sidebar-button styles).
    st.markdown("""<style>
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button {
        background: transparent !important; border: none !important; box-shadow: none !important;
        justify-content: flex-start !important; text-align: left !important;
        gap: 11px !important; padding: 9px 12px !important; border-radius: 7px !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button [data-testid="stIconMaterial"] {
        margin-right: 0 !important;
    }
    /* Force left-alignment robustly across Streamlit versions (button inner wrappers) */
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button > div,
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button > div > div,
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button [data-testid="stMarkdownContainer"] {
        justify-content: flex-start !important; text-align: left !important; width: 100% !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button p,
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button span {
        text-align: left !important; margin: 0 !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button,
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button * {
        color: #9fb0d0 !important; font-weight: 400 !important; letter-spacing: 0 !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button [data-testid="stIconMaterial"] {
        font-size: 21px !important; width: 21px !important; height: 21px !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button:hover { background: rgba(255,204,0,0.12) !important; }
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button:hover,
    [data-testid="stSidebar"] .st-key-navmenu .stButton > button:hover * { color: #FFCC00 !important; }
    [data-testid="stSidebar"] .st-key-navmenu [data-testid="stBaseButton-primary"] {
        background: rgba(255,204,0,0.10) !important;
        box-shadow: 0 0 0 1px rgba(255,204,0,0.28), 0 0 14px rgba(255,204,0,0.10) !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] .st-key-navmenu [data-testid="stBaseButton-primary"] * {
        color: #FFCC00 !important; font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .nav-cat {
        font-size: 0.68rem !important; letter-spacing: 1.2px !important; text-transform: uppercase !important;
        color: rgba(203,213,225,0.5) !important; padding: 13px 12px 4px !important; font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .st-key-navmenu .stCaption, [data-testid="stSidebar"] .st-key-navmenu .stCaption * {
        color: rgba(255,204,0,0.65) !important; font-size: 0.66rem !important; padding-left: 12px !important;
    }
    </style>""", unsafe_allow_html=True)

    with st.container(key="navmenu"):
        for _cat, _items in _NAV_CATEGORIES:
            st.markdown(f"<div class='nav-cat'>{_cat}</div>", unsafe_allow_html=True)
            for _val, _label, _icon in _items:
                if st.button(_label, icon=_icon, key=f"nav_{_val}", use_container_width=True,
                             type=("primary" if nav == _val else "secondary")):
                    st.session_state["main_nav_radio"] = _val
                    st.rerun()
            if _cat == "Predictive Analysis":
                st.caption("IVIVC Correlation · v3.5 — coming soon")
        # Learn — DissolvA Academy (free overlay; not a routed page)
        st.markdown("<div class='nav-cat'>Learn</div>", unsafe_allow_html=True)
        if st.button("DissolvA Academy", icon=":material/school:", key="nav_academy",
                     use_container_width=True, type="secondary"):
            st.session_state.academy_open = True
            st.rerun()

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
    # (New Session + Load demo moved to the top header, right side.)

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
# HEADER  (title left · actions right: New Session · Load demo · auth · Try Free)
# ===========================================================================
@st.dialog("Start a new session?")
def _new_session_dialog():
    st.write("This clears all loaded profiles and results. This cannot be undone.")
    _dc1, _dc2 = st.columns(2)
    if _dc1.button("Yes, clear everything", type="primary", use_container_width=True):
        _clear_all()
        st.rerun()
    if _dc2.button("Cancel", use_container_width=True):
        st.rerun()

_hl, _hr = st.columns([0.56, 0.44])
with _hl:
    st.markdown(
        '<h1 style="margin:0;font-size:2.0rem;color:#FFFFFF;">'
        'Dissolv<span style="color:#FFCC00;">A</span><sup style="font-size:0.85rem;color:#8593AD;">(TM)</sup> '
        '<span style="font-size:0.92rem;color:#9fb0d0;font-style:italic;font-weight:400;">'
        '- Predictive Dissolution Suite</span></h1>'
        '<div style="color:#7E8DAB;font-size:0.84rem;margin-top:4px;">'
        'FDA/EMA guidance-aligned · 62 Kinetic Models · f1/f2 · Bootstrap f2 · Statistical Profiling'
        '<span style="background:rgba(255,204,0,0.10);color:#FFCC00;border:1px solid rgba(255,204,0,0.35);'
        'font-size:0.7rem;font-weight:600;padding:1px 8px;border-radius:10px;margin-left:8px;">'
        'BETA · research use only</span></div>',
        unsafe_allow_html=True
    )
with _hr:
    with st.container(horizontal=True, horizontal_alignment="right", key="hdractions"):
        if st.button("New Session", icon=":material/add:", key="hdr_new",
                     help="Clear all profiles and results and start fresh."):
            _new_session_dialog()
        if st.button("Load demo", icon=":material/science:", key="hdr_demo",
                     help="Load example Reference + Test profiles."):
            extras.load_demo_data()
            st.session_state["_pending_nav"] = "Data Input"
            st.toast("Demo profiles loaded — open Data Input or Kinetic Model Fitting.", icon="⚗")
            st.rerun()
        auth.render_sidebar_auth()
        st.markdown(
            '<a href="#" style="color:#FFD86B;font-weight:600;font-size:0.86rem;'
            'text-decoration:none;white-space:nowrap;padding:6px 4px;">Try Free</a>',
            unsafe_allow_html=True)
st.markdown('<hr style="border:1px solid #FFCC00;margin:8px 0 4px 0;">', unsafe_allow_html=True)

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
            bg, col, bd, sh = ("rgba(255,204,0,0.12)", "#FFCC00", "rgba(255,204,0,0.40)",
                               "box-shadow:0 0 0 1px rgba(255,204,0,0.28),0 0 12px rgba(255,204,0,0.10);")
        elif i < _active:
            bg, col, bd, sh = "rgba(255,204,0,0.06)", "#C9A94A", "rgba(255,204,0,0.22)", ""
        else:
            bg, col, bd, sh = "transparent", "#7E8DAB", "rgba(126,141,171,0.35)", ""
        _chips.append(
            f'<span style="background:{bg};color:{col};border:1px solid {bd};{sh}'
            f'border-radius:14px;padding:3px 12px;font-size:0.72rem;font-weight:600;'
            f'white-space:nowrap;">{i+1}. {label}</span>')
    _sep = '<span style="color:#4a5a7e;">→</span>'
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

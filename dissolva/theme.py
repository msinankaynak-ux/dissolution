"""DissolvA theme: brand colors, global CSS injection and matplotlib axis styling.
Premium dark-mode workspace (sidebar #0B132B, workspace #1C2541, gold #FFCC00)."""
import streamlit as st

# Chart constants (UNCHANGED — charts render as light cards on the dark workspace)
OXFORD = "#002147"
AMBER  = "#FFBF00"
FIGSIZE = (10, 5)
DPI = 150
PALETTE = [
    "#e6194B","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4",
    "#f032e6","#bfef45","#469990","#dcbeff","#9A6324","#800000",
    "#aaffc3","#000075","#a9a9a9","#ffe119","#008080","#ffd8b1",
]

# Premium dark-mode UI palette (chrome / CSS only)
NAVY_SIDEBAR = "#0B132B"   # sidebar background
GRAPHITE     = "#1C2541"   # workspace background
SURFACE      = "#16203F"   # cards / inputs
GOLD         = "#FFCC00"   # premium accent / highlights
TXT          = "#FFFFFF"   # primary text (headers)
TXT2         = "#CBD5E1"   # secondary text (descriptions)

VERSION      = "3.0"       # single source of truth for the visible app version


def brand_html(extra_css: str = "", white: str = "#FFFFFF") -> str:
    """Canonical DissolvA wordmark — white 'Dissolv' + gold 'A' + small grey ™.
    Use this EVERYWHERE the brand is shown as a lockup; never re-spell it inline
    (keeps the trademark style consistent: always ™, never (TM))."""
    return (f'<span style="{extra_css}color:{white};font-weight:600;">Dissolv'
            f'<span style="color:{GOLD};">A</span>'
            f'<sup style="font-size:0.5em;color:#8593AD;font-weight:400;'
            f'vertical-align:super;">™</sup></span>')

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'EB Garamond', Georgia, serif !important;
}

/* ── Sidebar — premium navy ─────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #0B132B !important;
  border-right: 1px solid rgba(255,204,0,0.18) !important;
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

section[data-testid="stSidebar"] label {
  color: rgba(203,213,225,0.65) !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.02em !important;
  background: transparent !important;
  padding: 0 !important;
  margin-bottom: 2px !important;
}

section[data-testid="stSidebar"] .stRadio { margin-top: 0 !important; }
section[data-testid="stSidebar"] .stRadio > div {
  gap: 4px !important; display: flex !important; flex-direction: column !important;
}
section[data-testid="stSidebar"] .stRadio label {
  display: flex !important; flex-direction: row !important; align-items: center !important;
  gap: 10px !important; background: transparent !important; border: none !important;
  border-radius: 7px !important; padding: 8px 12px !important;
  font-size: 0.91rem !important; color: #9fb0d0 !important; cursor: pointer !important;
  transition: background 0.15s, color 0.15s !important; width: 100% !important;
  letter-spacing: 0.01em !important; white-space: nowrap !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255,204,0,0.12) !important; color: #FFCC00 !important;
}
section[data-testid="stSidebar"] .stRadio label:hover > div:first-child { border-color: #FFCC00 !important; }
section[data-testid="stSidebar"] .stRadio label > div:first-child {
  width: 16px !important; height: 16px !important; min-width: 16px !important;
  border-radius: 50% !important; border: 1.5px solid rgba(255,255,255,0.25) !important;
  background: transparent !important; display: flex !important; align-items: center !important;
  justify-content: center !important; flex-shrink: 0 !important;
}
section[data-testid="stSidebar"] .stRadio label > div:first-child > div {
  width: 7px !important; height: 7px !important; border-radius: 50% !important; background: transparent !important;
}
section[data-testid="stSidebar"] .stRadio label > div[data-testid="stMarkdownContainer"] {
  display: flex !important; align-items: center !important; flex: 1 !important;
}
section[data-testid="stSidebar"] .stRadio label > div[data-testid="stMarkdownContainer"] p {
  color: inherit !important; font-size: 0.91rem !important; margin: 0 !important;
  line-height: 1.2 !important; font-weight: inherit !important; white-space: nowrap !important;
}
section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label,
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background: rgba(255,204,0,0.10) !important; color: #FFCC00 !important; font-weight: 600 !important;
  box-shadow: 0 0 0 1px rgba(255,204,0,0.28), 0 0 14px rgba(255,204,0,0.10) !important;
}
section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label > div:first-child,
section[data-testid="stSidebar"] .stRadio label:has(input:checked) > div:first-child {
  border-color: #FFCC00 !important; background: #FFCC00 !important;
}
section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label > div:first-child > div,
section[data-testid="stSidebar"] .stRadio label:has(input:checked) > div:first-child > div {
  background: #0B132B !important;
}

/* Sidebar inputs — dark */
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput > div,
section[data-testid="stSidebar"] .stTextInput input {
  background: #16203F !important; border: 1px solid rgba(255,204,0,0.30) !important;
  border-radius: 6px !important; color: #E6ECF8 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div > div { background: #16203F !important; border: none !important; color: #E6ECF8 !important; }
section[data-testid="stSidebar"] .stNumberInput input { background: #16203F !important; border: none !important; color: #E6ECF8 !important; }
section[data-testid="stSidebar"] .stSelectbox svg,
section[data-testid="stSidebar"] .stNumberInput svg { fill: #FFD966 !important; }
section[data-testid="stSidebar"] .stNumberInput button { background: #16203F !important; border: none !important; color: #FFD966 !important; }

/* ── Workspace metric cards — dark ──────────────────────────────────────── */
[data-testid="metric-container"], [data-testid="stMetric"] {
  background: #16203F; border: 1px solid rgba(255,255,255,0.07);
  border-left: 4px solid #FFCC00; border-radius: 8px; padding: 12px;
}

/* ── Workspace primary buttons ──────────────────────────────────────────── */
.stButton > button {
  background: #16203F !important; color: #FFCC00 !important;
  border: 1px solid #FFCC00 !important;
  font-family: 'EB Garamond', serif !important;
  font-size: 1rem !important; font-weight: 600 !important;
  border-radius: 7px !important; padding: 8px 20px !important;
}
.stButton > button:hover { background: #FFCC00 !important; color: #0B132B !important; }

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important; color: rgba(203,213,225,0.7) !important;
  border: 1px solid rgba(255,204,0,0.25) !important;
  font-size: 0.85rem !important; font-weight: 400 !important;
  padding: 5px 14px !important; border-radius: 7px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,204,0,0.1) !important; color: #FFCC00 !important; border-color: rgba(255,204,0,0.6) !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background: rgba(255,204,0,0.10) !important; color: #FFCC00 !important;
  border: 1px solid rgba(255,204,0,0.5) !important; font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * { color: #FFCC00 !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover { background: #FFCC00 !important; border-color: #FFCC00 !important; }
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover * { color: #0B132B !important; }

/* streamlit-oauth "Sign in with Google" button (iframe) — sidebar variant (scaled) */
[data-testid="stSidebar"] iframe[title="streamlit_oauth.authorize_button"] {
  border-radius: 12px !important; overflow: hidden !important;
  transform: scale(0.68) !important; transform-origin: top left !important;
  width: 147% !important; margin-bottom: -23px !important;
}
/* OAuth button in the TOP HEADER (right actions) — rounded, keep one line */
.st-key-hdractions iframe[title="streamlit_oauth.authorize_button"] {
  border-radius: 8px !important; overflow: hidden !important;
}
.st-key-hdractions { align-items: center !important; gap: 10px !important; }
.stDownloadButton > button {
  background: #FFCC00 !important; color: #0B132B !important;
  border: 1px solid #FFCC00 !important;
  font-family: 'EB Garamond', serif !important; font-weight: 700 !important;
}
button[data-baseweb="tab"] { font-family: 'EB Garamond', serif !important; font-size: 1.05rem !important; }
button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #FFCC00 !important; font-weight: 700 !important; }

/* ── Workspace data-entry hierarchy — muted label vs bright editable field ─ */
/* Field labels: muted, small, uppercase → clearly distinct from the white value */
[data-testid="stMain"] .stTextInput label p,
[data-testid="stMain"] .stNumberInput label p,
[data-testid="stMain"] .stSelectbox label p,
[data-testid="stMain"] .stTextArea label p,
[data-testid="stMain"] .stDateInput label p,
[data-testid="stMain"] .stTimeInput label p,
[data-testid="stMain"] .stMultiSelect label p,
[data-testid="stMain"] .stSlider label p,
[data-testid="stMain"] .stFileUploader label p {
  color: #8EA0BC !important; font-size: 0.74rem !important; font-weight: 600 !important;
  letter-spacing: 0.05em !important; text-transform: uppercase !important;
}
/* Editable fields: defined border + gold focus so they read as "type here" */
[data-testid="stMain"] .stTextInput input,
[data-testid="stMain"] .stNumberInput input,
[data-testid="stMain"] .stTextArea textarea,
[data-testid="stMain"] .stDateInput input,
[data-testid="stMain"] .stSelectbox div[data-baseweb="select"] > div {
  background: #16203F !important; color: #FFFFFF !important;
  border: 1px solid rgba(255,204,0,0.28) !important; border-radius: 7px !important;
}
[data-testid="stMain"] .stTextInput input:focus,
[data-testid="stMain"] .stNumberInput input:focus,
[data-testid="stMain"] .stTextArea textarea:focus {
  border-color: #FFCC00 !important; box-shadow: 0 0 0 2px rgba(255,204,0,0.22) !important;
}
[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder { color: #6f7d97 !important; }
[data-testid="stMain"] .stCaption, [data-testid="stMain"] .stCaption * {
  color: #6f7d97 !important;
}

/* ── Reusable dark-safe status boxes (replace ad-hoc light hex on pages) ──── */
.dv-note { background: rgba(255,255,255,0.04); border-left: 4px solid #5b86c4;
  color: #DDE6F2; border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; }
.dv-warn { background: rgba(255,204,0,0.08); border-left: 4px solid #FFCC00;
  color: #F0E2B0; border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 8px 0; font-size: 0.88rem; }
.dv-pass { background: rgba(38,174,96,0.14); border: 1px solid rgba(38,174,96,0.45);
  color: #7EDD9A; border-radius: 6px; padding: 10px 14px; text-align: center; font-weight: 600; }
.dv-fail { background: rgba(231,76,60,0.14); border: 1px solid rgba(231,76,60,0.45);
  color: #F2998E; border-radius: 6px; padding: 10px 14px; text-align: center; font-weight: 600; }

/* ── Custom content boxes — dark surfaces + light text ──────────────────── */
.eq-box {
  font-family: 'JetBrains Mono', monospace; color: #E6ECF8;
  background: #16203F; border-left: 4px solid #FFCC00;
  padding: 8px 14px; font-size: 0.82rem; border-radius: 0 6px 6px 0; margin: 4px 0 8px 0;
}
.info-banner {
  background: rgba(255,255,255,0.04); border: 1px solid rgba(120,160,220,0.30); color: #DDE6F2;
  border-radius: 6px; padding: 10px 14px; font-size: 0.93rem; margin: 8px 0;
}
.step-box {
  background: rgba(255,204,0,0.07); border: 1px solid rgba(255,204,0,0.30); color: #E7D9A8;
  border-radius: 8px; padding: 12px 16px; margin: 6px 0; font-size: 0.93rem;
}
.nav-active {
  border-left: 3px solid #FFCC00 !important; background: rgba(255,204,0,0.10) !important;
  color: #FFCC00 !important; font-weight: 600 !important;
}
.nav-section-label {
  font-size: 0.58rem !important; letter-spacing: 2px !important; text-transform: uppercase !important;
  color: rgba(255,204,0,0.55) !important; padding: 10px 12px 2px 12px !important; display: block !important;
}
.nav-divider {
  border: none !important; border-top: 1px solid rgba(255,204,0,0.15) !important; margin: 6px 0 !important;
}
</style>
"""

def inject_theme():
    """Injects global CSS. Must be called AFTER st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def style_ax(fig, ax):
    # Charts stay light (render as clean light cards on the dark workspace) — zero
    # risk to existing chart text/markers that use OXFORD.
    try: fig.set_dpi(DPI)
    except Exception: pass
    fig.patch.set_facecolor("#FDFAF5")
    ax.set_facecolor("#F8F4EC")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(OXFORD); ax.spines["bottom"].set_color(OXFORD)
    ax.tick_params(colors=OXFORD, labelsize=9)
    ax.xaxis.label.set_color(OXFORD); ax.yaxis.label.set_color(OXFORD)
    ax.title.set_color(OXFORD)

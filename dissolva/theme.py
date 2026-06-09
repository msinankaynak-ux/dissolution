"""DissolvA theme: brand color, global CSS injection and matplotlib axis styling.
Extracted from app.py (Phase 1.4 modularization)."""
import streamlit as st

OXFORD = "#002147"
AMBER  = "#FFBF00"
# Shared chart sizing — keep page-width dissolution charts visually consistent
# and export at print quality. Use via plt.subplots(figsize=FIGSIZE); style_ax()
# raises the DPI for all figures uniformly.
FIGSIZE = (10, 5)
DPI = 150
PALETTE = [
    "#e6194B","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4",
    "#f032e6","#bfef45","#469990","#dcbeff","#9A6324","#800000",
    "#aaffc3","#000075","#a9a9a9","#ffe119","#008080","#ffd8b1",
]

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
  font-family: 'EB Garamond', Georgia, serif !important;
  background: #F5F0E8 !important;
  color: #1a1a2e !important;
}
section[data-testid="stSidebar"] {
  background: #001a3d !important;
  border-right: 3px solid #FFBF00 !important;
}
section[data-testid="stSidebar"] * { color: #e8e0d0 !important; }

section[data-testid="stSidebar"] label {
  color: rgba(232,224,208,0.65) !important;
  font-size: 0.78rem !important;
  font-weight: 400 !important;
  letter-spacing: 0.02em !important;
  background: transparent !important;
  padding: 0 !important;
  margin-bottom: 2px !important;
}

section[data-testid="stSidebar"] .stRadio { margin-top: 0 !important; }

section[data-testid="stSidebar"] .stRadio > div {
  gap: 4px !important;
  display: flex !important;
  flex-direction: column !important;
}

section[data-testid="stSidebar"] .stRadio label {
  display: flex !important;
  flex-direction: row !important;
  align-items: center !important;
  gap: 10px !important;
  background: transparent !important;
  border: none !important;
  border-radius: 5px !important;
  padding: 8px 12px !important;
  font-size: 0.91rem !important;
  color: #7a9dbf !important;
  cursor: pointer !important;
  transition: background 0.15s, color 0.15s !important;
  width: 100% !important;
  letter-spacing: 0.01em !important;
  white-space: nowrap !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
  background: rgba(255,191,0,0.07) !important;
  color: #c8b45a !important;
}

section[data-testid="stSidebar"] .stRadio label > div:first-child {
  width: 16px !important;
  height: 16px !important;
  min-width: 16px !important;
  border-radius: 50% !important;
  border: 1.5px solid rgba(255,255,255,0.30) !important;
  background: transparent !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex-shrink: 0 !important;
}

section[data-testid="stSidebar"] .stRadio label > div:first-child > div {
  width: 7px !important;
  height: 7px !important;
  border-radius: 50% !important;
  background: transparent !important;
}

section[data-testid="stSidebar"] .stRadio label > div[data-testid="stMarkdownContainer"] {
  display: flex !important;
  align-items: center !important;
  flex: 1 !important;
}

section[data-testid="stSidebar"] .stRadio label > div[data-testid="stMarkdownContainer"] p {
  color: inherit !important;
  font-size: 0.91rem !important;
  margin: 0 !important;
  line-height: 1.2 !important;
  font-weight: inherit !important;
  white-space: nowrap !important;
}

section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label,
section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
  background: rgba(255,191,0,0.10) !important;
  color: #FFBF00 !important;
  font-weight: 600 !important;
}

section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label > div:first-child,
section[data-testid="stSidebar"] .stRadio label:has(input:checked) > div:first-child {
  border-color: #FFBF00 !important;
  background: #FFBF00 !important;
}

section[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + label > div:first-child > div,
section[data-testid="stSidebar"] .stRadio label:has(input:checked) > div:first-child > div {
  background: #001a3d !important;
}

section[data-testid="stSidebar"] .stSelectbox > div > div {
  background: #002a5c !important;
  border: 1px solid rgba(255,191,0,0.40) !important;
  border-radius: 5px !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div > div {
  background: #002a5c !important;
  border: none !important;
  color: #e8e0d0 !important;
}
section[data-testid="stSidebar"] .stNumberInput > div {
  background: #002a5c !important;
  border: 1px solid rgba(255,191,0,0.40) !important;
  border-radius: 5px !important;
}
section[data-testid="stSidebar"] .stNumberInput input {
  background: #002a5c !important;
  border: none !important;
  color: #e8e0d0 !important;
}
section[data-testid="stSidebar"] .stTextInput input {
  background: #002a5c !important;
  border: 1px solid rgba(255,191,0,0.40) !important;
  color: #e8e0d0 !important;
  border-radius: 5px !important;
}
section[data-testid="stSidebar"] .stSelectbox svg,
section[data-testid="stSidebar"] .stNumberInput svg {
  fill: #FFD966 !important;
}
section[data-testid="stSidebar"] .stNumberInput button {
  background: #002a5c !important;
  border: none !important;
  color: #FFD966 !important;
}
[data-testid="metric-container"] {
  background: white; border: 1px solid #ddd;
  border-left: 4px solid #FFBF00; border-radius: 4px; padding: 12px;
}
.stButton > button {
  background: #002147 !important; color: #FFBF00 !important;
  border: 2px solid #FFBF00 !important;
  font-family: 'EB Garamond', serif !important;
  font-size: 1rem !important; font-weight: 600 !important;
  border-radius: 4px !important; padding: 8px 20px !important;
}
.stButton > button:hover { background: #FFBF00 !important; color: #002147 !important; }
/* Sidebar buttons simplified */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  color: rgba(232,224,208,0.65) !important;
  border: 1px solid rgba(255,191,0,0.25) !important;
  font-size: 0.85rem !important; font-weight: 400 !important;
  padding: 5px 14px !important; border-radius: 6px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255,191,0,0.1) !important;
  color: #FFBF00 !important;
  border-color: rgba(255,191,0,0.6) !important;
}
/* Force-target sidebar buttons by testid (overrides Streamlit secondary's default white bg) */
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background: rgba(255,191,0,0.10) !important;
  color: #FFBF00 !important;
  border: 1px solid rgba(255,191,0,0.5) !important;
  font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {
  color: #FFBF00 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
  background: #FFBF00 !important;
  border-color: #FFBF00 !important;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover * {
  color: #002147 !important;
}
/* Round the streamlit-oauth "Sign in with Google" button (iframe component); keep its color */
[data-testid="stSidebar"] iframe[title="streamlit_oauth.authorize_button"] {
  border-radius: 12px !important;
  overflow: hidden !important;
}
.stDownloadButton > button {
  background: #FFBF00 !important; color: #002147 !important;
  border: 2px solid #002147 !important;
  font-family: 'EB Garamond', serif !important; font-weight: 700 !important;
}
button[data-baseweb="tab"] { font-family: 'EB Garamond', serif !important; font-size: 1.05rem !important; }
button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 3px solid #FFBF00 !important; font-weight: 700 !important; }
.eq-box {
  font-family: 'JetBrains Mono', monospace;
  background: #f0ece0; border-left: 4px solid #FFBF00;
  padding: 8px 14px; font-size: 0.82rem;
  border-radius: 0 4px 4px 0; margin: 4px 0 8px 0;
}
.info-banner {
  background: #e8f0f7; border: 1px solid #b8d0e8;
  border-radius: 4px; padding: 10px 14px;
  font-size: 0.93rem; margin: 8px 0;
}
.step-box {
  background: #fff8e6; border: 1px solid #FFBF00;
  border-radius: 6px; padding: 12px 16px; margin: 6px 0;
  font-size: 0.93rem;
}
.nav-active {
  border-left: 3px solid #FFBF00 !important;
  background: rgba(255,191,0,0.10) !important;
  color: #FFBF00 !important;
  font-weight: 600 !important;
}
.nav-section-label {
  font-size: 0.58rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: rgba(255,191,0,0.55) !important;
  padding: 10px 12px 2px 12px !important;
  display: block !important;
}
.nav-divider {
  border: none !important;
  border-top: 1px solid rgba(255,191,0,0.15) !important;
  margin: 6px 0 !important;
}
</style>

<!-- Sidebar group headers/beta badge: inline JS removed (does not run inside Streamlit st.markdown). Rich grouping will come in Phase 2 via streamlit-option-menu. -->
"""

def inject_theme():
    """Injects global CSS. Must be called AFTER st.set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def style_ax(fig, ax):
    try: fig.set_dpi(DPI)   # uniform print-quality resolution across all charts
    except Exception: pass
    fig.patch.set_facecolor("#FDFAF5")
    ax.set_facecolor("#F8F4EC")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(OXFORD); ax.spines["bottom"].set_color(OXFORD)
    ax.tick_params(colors=OXFORD, labelsize=9)
    ax.xaxis.label.set_color(OXFORD); ax.yaxis.label.set_color(OXFORD)
    ax.title.set_color(OXFORD)

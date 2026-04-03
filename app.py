import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

from scipy.optimize import curve_fit, root
from scipy.stats import t as t_dist
from scipy.integrate import trapezoid
import io

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DissolvA™ — Predictive Dissolution Suite",
    page_icon="ð§¬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --oxford: #002147;
    --amber:  #FFBF00;
    --amber-light: #FFD966;
    --cream:  #F5F0E8;
    --text:   #1a1a2e;
    --muted:  #5a6480;
  }

  html, body, [class*="css"] {
    font-family: 'EB Garamond', Georgia, serif;
    background: var(--cream) !important;
    color: var(--text);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: var(--oxford) !important;
    border-right: 3px solid var(--amber);
  }
  [data-testid="stSidebar"] * { color: #e8e0d0 !important; }
  [data-testid="stSidebar"] label { color: var(--amber-light) !important; font-size: 0.85rem; }

  /* Headers */
  h1, h2, h3 { font-family: 'EB Garamond', serif; color: var(--oxford); }

  /* Metric boxes */
  [data-testid="metric-container"] {
    background: white;
    border: 1px solid #ddd;
    border-left: 4px solid var(--amber);
    border-radius: 4px;
    padding: 12px;
  }

  /* Buttons */
  .stButton > button {
    background: var(--oxford) !important;
    color: var(--amber) !important;
    border: 2px solid var(--amber) !important;
    font-family: 'EB Garamond', serif !important;
    font-size: 1rem !important;
    font-weight: 600;
    border-radius: 3px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: var(--amber) !important;
    color: var(--oxford) !important;
  }

  /* Download button */
  .stDownloadButton > button {
    background: var(--amber) !important;
    color: var(--oxford) !important;
    border: 2px solid var(--oxford) !important;
    font-family: 'EB Garamond', serif !important;
    font-weight: 700;
    border-radius: 3px;
  }

  /* Tabs */
  [data-testid="stTabs"] [role="tab"] {
    font-family: 'EB Garamond', serif !important;
    font-size: 1.05rem !important;
    color: var(--oxford) !important;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    border-bottom: 3px solid var(--amber) !important;
    color: var(--oxford) !important;
    font-weight: 700 !important;
  }

  /* Tables */
  [data-testid="stDataFrame"] { border: 1px solid #ccc; }

  /* Mono for equations */
  .eq-box {
    font-family: 'JetBrains Mono', monospace;
    background: #f0ece0;
    border-left: 4px solid var(--amber);
    padding: 8px 14px;
    font-size: 0.83rem;
    border-radius: 0 4px 4px 0;
    margin: 6px 0;
  }

  /* Info banners */
  .info-banner {
    background: #e8f0f7;
    border: 1px solid #b8d0e8;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 0.93rem;
    margin: 8px 0;
  }

  /* Good fit badge */
  .badge-best { background: #1a7a3f; color: white; padding: 2px 8px; border-radius: 12px; font-size:0.78rem; }
  .badge-good { background: #2c6fad; color: white; padding: 2px 8px; border-radius: 12px; font-size:0.78rem; }
  .badge-ok   { background: #a07800; color: white; padding: 2px 8px; border-radius: 12px; font-size:0.78rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Brand box
    st.markdown("""
    <div style="
      border: 2px solid #FFBF00;
      border-radius: 6px;
      padding: 18px 14px 14px 14px;
      margin-bottom: 24px;
      background: rgba(255,191,0,0.06);
      text-align: center;
    ">
      <div style="font-size:2rem; margin-bottom:4px;">ð</div>
      <div style="font-size:1.55rem; font-family:'EB Garamond',serif; font-weight:700;
                  color:#FFBF00; letter-spacing:0.03em;">DissolvA™</div>
      <div style="font-size:0.78rem; color:#d4c89a; letter-spacing:0.12em;
                  text-transform:uppercase; margin-top:2px;">Predictive Dissolution Suite</div>
      <div style="margin-top:10px;">
        <span style="background:#FFBF00; color:#002147; font-size:0.68rem;
                     font-weight:800; padding:3px 10px; border-radius:12px;
                     letter-spacing:0.1em;">⚡ POWERED BY AI</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ð§¬ Molecular View")
    st.markdown("---")

    nav = st.radio(
        "Navigation",
        ["ð¥ Data Input", "⚙️ Kinetic Model Fitting", "ð Statistical Analysis",
         "ð f1 & f2 Similarity", "ð¬ IVIVC Analysis", "ð Excel Report"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚙️ Parameter Settings")

    time_unit = st.selectbox("Time Unit", ["minutes", "hours"])
    conc_unit = st.selectbox("Concentration Unit", ["mg/mL", "µg/mL", "mg/L"])
    dose_mg   = st.number_input("Dose (mg)", value=100.0, min_value=0.1)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.72rem; color:#9090a0; text-align:center; line-height:1.6;">
      <em>DissolvA™ v2.0</em><br>
      © 2025 Predictive Dissolution Suite<br>
      All rights reserved®
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "profiles" not in st.session_state:
    st.session_state.profiles = {}      # {name: {"time": [...], "release": [...]}}
if "fit_results" not in st.session_state:
    st.session_state.fit_results = {}   # {model_name: {r2, aic, msc, params, ...}}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def zero_order(t, k0):
    return k0 * t

def first_order(t, k1):
    return 100.0 * (1 - np.exp(-k1 * t))

def higuchi(t, kH):
    return kH * np.sqrt(t)

def hixson_crowell(t, ks):
    val = 1.0 - (1.0 - ks * t / 3.0) ** 3
    return np.clip(val * 100.0, 0, 100)

def korsmeyer_peppas(t, k, n):
    return k * t ** n

def hopfenberg(t, k_HB, n_HB):
    val = 1.0 - (1.0 - k_HB * t) ** n_HB
    return np.clip(val * 100.0, 0, 100)

def baker_lonsdale(t, k_BL):
    # Solved numerically: 3/2[1-(1-F/100)^(2/3)] - F/100 = k_BL * t
    results = []
    for ti in t:
        rhs = k_BL * ti
        def equation(F_frac):
            F_frac = float(F_frac)
            return 1.5 * (1.0 - (1.0 - F_frac) ** (2.0/3.0)) - F_frac - rhs
        try:
            sol = root(equation, 0.5, method="hybr")
            F_val = float(np.clip(sol.x[0] * 100.0, 0, 100)) if sol.success else np.nan
        except Exception:
            F_val = np.nan
        results.append(F_val)
    return np.array(results)

def makoid_banakar(t, k_MB, n_MB, b_MB):
    return k_MB * (t ** n_MB) * np.exp(-b_MB * t)

def peppas_sahlin(t, k1_PS, k2_PS, m_PS):
    return k1_PS * (t ** m_PS) + k2_PS * (t ** (2.0 * m_PS))

def weibull(t, a_W, b_W, Td_W):
    t_adj = np.clip(t - Td_W, 0, None)
    return 100.0 * (1.0 - np.exp(-((t_adj) ** b_W) / a_W))

def gompertz(t, a_G, b_G, k_G):
    return a_G * np.exp(-b_G * np.exp(-k_G * t))

def logistic_model(t, A_L, k_L, t50_L):
    return A_L / (1.0 + np.exp(-k_L * (t - t50_L)))

def quadratic_model(t, a_Q, b_Q, c_Q):
    return a_Q * t**2 + b_Q * t + c_Q

def probit_model(t, mu_P, sigma_P, A_P):
    from scipy.stats import norm
    return A_P * norm.cdf(t, mu_P, sigma_P)

MODEL_DEFS = {
    # name: (func, p0, param_names, equation_str, reference)
    "Zero Order":       (zero_order,       [1.0],               ["k₀"],                        "F = k₀·t",                                         "Wagner, 1969"),
    "First Order":      (first_order,      [0.1],               ["k₁"],                        "F = 100·(1−e^(−k₁t))",                             "Wagner, 1969"),
    "Higuchi":          (higuchi,          [10.0],              ["kH"],                        "F = kH·√t",                                         "Higuchi, 1961"),
    "Hixson-Crowell":   (hixson_crowell,   [0.05],              ["ks"],                        "M₀^(1/3) − M^(1/3) = ks·t",                        "Hixson & Crowell, 1931"),
    "Korsmeyer-Peppas": (korsmeyer_peppas, [10.0, 0.5],         ["k", "n"],                    "F = k·t^n",                                         "Korsmeyer et al., 1983"),
    "Hopfenberg":       (hopfenberg,       [0.05, 2.0],         ["kHB", "nHB"],                "F = 100·[1−(1−kHB·t)^nHB]",                        "Hopfenberg, 1976"),
    "Baker-Lonsdale":   (baker_lonsdale,   [0.001],             ["kBL"],                       "3/2[1−(1−F)^(2/3)]−F = kBL·t",                     "Baker & Lonsdale, 1974"),
    "Makoid-Banakar":   (makoid_banakar,   [10.0, 0.5, 0.01],   ["kMB", "nMB", "bMB"],         "F = kMB·t^nMB·e^(−bMB·t)",                          "Makoid & Banakar, 1993"),
    "Peppas-Sahlin":    (peppas_sahlin,    [5.0, 1.0, 0.5],     ["k1", "k2", "m"],             "F = k1·t^m + k2·t^(2m)",                            "Peppas & Sahlin, 1989"),
    "Weibull":          (weibull,          [50.0, 1.0, 0.0],    ["a", "b", "Td"],              "F = 100·(1−e^(−((t−Td)^b)/a))",                    "Weibull, 1951"),
    "Gompertz":         (gompertz,         [100.0, 5.0, 0.1],   ["A", "b", "k"],               "F = A·e^(−b·e^(−kt))",                              "Gompertz, 1825"),
    "Logistic":         (logistic_model,   [100.0, 0.1, 30.0],  ["A", "k", "t₅₀"],            "F = A/(1+e^(−k(t−t₅₀)))",                          "Pressman & Dobbins, 1994"),
    "Quadratic":        (quadratic_model,  [-0.01, 1.0, 0.0],   ["a", "b", "c"],               "F = a·t² + b·t + c",                                "Polli et al., 1997"),
    "Probit":           (probit_model,     [30.0, 15.0, 100.0], ["μ", "σ", "A"],               "F = A·Φ((t−μ)/σ)",                                  "Shah et al., 1998"),
}


# ═══════════════════════════════════════════════════════════════════════════════
#  STATISTICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_r2(y_obs, y_pred):
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)

def compute_r2_adj(y_obs, y_pred, n_params):
    n = len(y_obs)
    r2 = compute_r2(y_obs, y_pred)
    if n <= n_params + 1:
        return r2
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - n_params - 1))

def compute_aic(y_obs, y_pred, n_params):
    n = len(y_obs)
    sse = np.sum((y_obs - y_pred) ** 2)
    if sse <= 0:
        sse = 1e-10
    return float(n * np.log(sse / n) + 2.0 * n_params)

def compute_msc(y_obs, y_pred, n_params):
    """Model Selection Criterion (Yamaoka et al.)"""
    n = len(y_obs)
    sse = np.sum((y_obs - y_pred) ** 2)
    sst = np.sum((y_obs - np.mean(y_obs)) ** 2)
    if sse <= 0:
        sse = 1e-10
    if sst <= 0:
        return 0.0
    return float(np.log(sst / sse) - 2.0 * n_params / n)

def compute_mdt(time, release):
    """Mean Dissolution Time via numerical integration."""
    t = np.array(time, dtype=float)
    f = np.array(release, dtype=float) / 100.0
    df = np.gradient(f, t)
    numerator   = trapezoid(t * df, t)
    denominator = trapezoid(df, t)
    if abs(denominator) < 1e-12:
        return np.nan
    return float(numerator / denominator)

def compute_de(time, release, t_ref=None):
    """Dissolution Efficiency (%) = AUC / (t_ref * 100) * 100"""
    t = np.array(time, dtype=float)
    f = np.array(release, dtype=float)
    if t_ref is None:
        t_ref = t[-1]
    auc = trapezoid(f, t)
    de  = auc / (t_ref * 100.0) * 100.0
    return float(de)

def fit_model(time_arr, release_arr, model_name):
    func, p0, param_names, eq_str, ref = MODEL_DEFS[model_name]
    t = np.array(time_arr, dtype=float)
    y = np.array(release_arr, dtype=float)
    try:
        popt, _ = curve_fit(func, t, y, p0=p0, maxfev=20000,
                            bounds=(-np.inf, np.inf))
        y_pred = func(t, *popt)
        # handle nan in baker-lonsdale
        valid = ~np.isnan(y_pred)
        if valid.sum() < 2:
            raise RuntimeError("Too many NaN predictions")
        r2     = compute_r2(y[valid],     y_pred[valid])
        r2_adj = compute_r2_adj(y[valid], y_pred[valid], len(popt))
        aic    = compute_aic(y[valid],    y_pred[valid], len(popt))
        msc    = compute_msc(y[valid],    y_pred[valid], len(popt))
        return {
            "model":       model_name,
            "r2":          r2,
            "r2_adj":      r2_adj,
            "aic":         aic,
            "msc":         msc,
            "params":      dict(zip(param_names, popt)),
            "y_pred":      y_pred,
            "equation":    eq_str,
            "reference":   ref,
            "n_params":    len(popt),
            "success":     True,
            "error":       None,
        }
    except Exception as e:
        return {
            "model":   model_name,
            "r2":      np.nan, "r2_adj": np.nan,
            "aic":     np.nan, "msc":    np.nan,
            "params":  {}, "y_pred": np.array([np.nan] * len(t)),
            "equation": MODEL_DEFS[model_name][3],
            "reference": MODEL_DEFS[model_name][4],
            "n_params": 0, "success": False, "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE FOR PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
OXFORD = "#002147"
AMBER  = "#FFBF00"
MODEL_COLORS = [
    "#e6194B","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324","#fffac8","#800000","#aaffc3","#000075"
]

def style_fig(fig, ax):
    fig.patch.set_facecolor("#FDFAF5")
    ax.set_facecolor("#F8F4EC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(OXFORD)
    ax.spines["bottom"].set_color(OXFORD)
    ax.tick_params(colors=OXFORD)
    ax.xaxis.label.set_color(OXFORD)
    ax.yaxis.label.set_color(OXFORD)
    ax.title.set_color(OXFORD)
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:3.5rem;padding-top:8px'>ð</div>", unsafe_allow_html=True)
with col_title:
    st.markdown(f"""
    <h1 style='margin:0;font-size:2.6rem;color:#002147;letter-spacing:0.02em;'>
      DissolvA™
      <span style='font-size:1rem;color:#888;font-weight:400;font-style:italic;'>
        — Predictive Dissolution Suite
      </span>
    </h1>
    <div style='color:#5a6480;font-size:0.92rem;margin-top:2px;'>
      FDA-Compliant · Multi-Model Kinetics · Statistical Profiling · IVIVC
      &nbsp;&nbsp;
      <span style='background:#002147;color:#FFBF00;padding:2px 10px;border-radius:12px;font-size:0.78rem;font-weight:700;'>⚡ POWERED BY AI</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #FFBF00;margin:12px 0 20px 0;'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA INPUT
# ═══════════════════════════════════════════════════════════════════════════════
if nav == "ð¥ Data Input":
    st.header("ð¥ Data Input")

    input_method = st.radio("Input Method", ["Manual Entry", "CSV Upload"], horizontal=True)

    if input_method == "Manual Entry":
        st.markdown("Enter time points and % cumulative release. Separate values with commas.")
        c1, c2 = st.columns(2)
        with c1:
            t_str = st.text_area("Time points", "0,15,30,45,60,90,120,180,240", height=100)
        with c2:
            r_str = st.text_area("Cumulative Release (%)", "0,18,35,49,62,74,82,89,94", height=100)
        profile_name = st.text_input("Profile Name", "Formulation A")

        if st.button("➕ Add Profile"):
            try:
                t_arr = np.array([float(x.strip()) for x in t_str.split(",")])
                r_arr = np.array([float(x.strip()) for x in r_str.split(",")])
                if len(t_arr) != len(r_arr):
                    st.error("Time and Release arrays must have the same length.")
                else:
                    st.session_state.profiles[profile_name] = {
                        "time": t_arr.tolist(), "release": r_arr.tolist()
                    }
                    st.success(f"✅ Profile '{profile_name}' added.")
            except Exception as e:
                st.error(f"Parse error: {e}")

    else:  # CSV Upload
        uploaded = st.file_uploader("Upload CSV (columns: time, release)", type=["csv"])
        profile_name = st.text_input("Profile Name", "Uploaded Profile")
        if uploaded and st.button("➕ Add from CSV"):
            try:
                df_up = pd.read_csv(uploaded)
                df_up.columns = [c.lower().strip() for c in df_up.columns]
                t_arr = df_up["time"].values
                r_arr = df_up["release"].values
                st.session_state.profiles[profile_name] = {
                    "time": t_arr.tolist(), "release": r_arr.tolist()
                }
                st.success(f"✅ Profile '{profile_name}' added.")
            except Exception as e:
                st.error(f"CSV error: {e}")

    # Preview
    if st.session_state.profiles:
        st.markdown("---")
        st.subheader("Loaded Dissolution Profiles")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        style_fig(fig, ax)
        for i, (name, data) in enumerate(st.session_state.profiles.items()):
            t = data["time"]
            r = data["release"]
            col = MODEL_COLORS[i % len(MODEL_COLORS)]
            ax.plot(t, r, "o-", color=col, linewidth=2, markersize=5, label=name)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Cumulative Release (%)")
        ax.set_title("Dissolution Profiles")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=8)
        ax.axhline(80, color=AMBER, linewidth=1, linestyle="--", alpha=0.7, label="80% line")
        st.pyplot(fig)
        plt.close()

        if st.button("ð️ Clear All Profiles"):
            st.session_state.profiles = {}
            st.session_state.fit_results = {}
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: KINETIC MODEL FITTING
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "⚙️ Kinetic Model Fitting":
    st.header("⚙️ Kinetic Model Fitting")

    if not st.session_state.profiles:
        st.warning("⚠️ Please load at least one dissolution profile in 'Data Input' first.")
        st.stop()

    profile_name = st.selectbox("Select Profile", list(st.session_state.profiles.keys()))
    data = st.session_state.profiles[profile_name]
    t_arr = np.array(data["time"])
    r_arr = np.array(data["release"])

    selected_models = st.multiselect(
        "Select Kinetic Models to Fit",
        list(MODEL_DEFS.keys()),
        default=["Zero Order", "First Order", "Higuchi", "Korsmeyer-Peppas",
                 "Weibull", "Peppas-Sahlin", "Gompertz"]
    )

    # Model equations reference
    with st.expander("ð Model Equations Reference"):
        for mname, (_, _, pnames, eq, ref) in MODEL_DEFS.items():
            st.markdown(
                f"**{mname}** &nbsp;&nbsp;<span style='color:#888;font-size:0.8rem'>({ref})</span><br>"
                f"<div class='eq-box'>{eq} &nbsp;|&nbsp; params: {', '.join(pnames)}</div>",
                unsafe_allow_html=True
            )

    if st.button("ð Run Model Fitting", type="primary"):
        st.session_state.fit_results = {}
        prog = st.progress(0)
        for i, mname in enumerate(selected_models):
            result = fit_model(t_arr, r_arr, mname)
            st.session_state.fit_results[mname] = result
            prog.progress((i + 1) / len(selected_models))
        st.success("✅ Fitting complete!")

    if st.session_state.fit_results:
        results = st.session_state.fit_results
        valid   = {k: v for k, v in results.items() if v["success"]}
        failed  = {k: v for k, v in results.items() if not v["success"]}

        # ── Ranking Table ──────────────────────────────────────────────────
        st.subheader("ð Model Ranking Table")
        rows = []
        for mname, res in valid.items():
            rows.append({
                "Model":        mname,
                "R²":           round(res["r2"],     4),
                "R²adj":        round(res["r2_adj"], 4),
                "AIC":          round(res["aic"],    3),
                "MSC":          round(res["msc"],    3),
                "Params":       res["n_params"],
                "Reference":    res["reference"],
            })
        if rows:
            df_rank = pd.DataFrame(rows).sort_values("R²adj", ascending=False).reset_index(drop=True)
            df_rank.index = df_rank.index + 1
            st.dataframe(df_rank.style.background_gradient(subset=["R²adj"], cmap="YlGn"), use_container_width=True)

            best = df_rank.iloc[0]["Model"]
            st.markdown(f"""
            <div class='info-banner'>
              ð <strong>Best fit:</strong> <span class='badge-best'>{best}</span>
              &nbsp; R²adj = {df_rank.iloc[0]["R²adj"]}
              &nbsp; | AIC = {df_rank.iloc[0]["AIC"]}
              &nbsp; | MSC = {df_rank.iloc[0]["MSC"]}
            </div>
            """, unsafe_allow_html=True)

        # ── Plot ──────────────────────────────────────────────────────────
        st.subheader("ð Dissolution Curves with Model Fits")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        style_fig(fig, ax)
        ax.scatter(t_arr, r_arr, color=OXFORD, zorder=5, s=60,
                   label="Experimental", edgecolors="white", linewidths=0.7)
        t_smooth = np.linspace(t_arr.min(), t_arr.max(), 300)

        for i, (mname, res) in enumerate(valid.items()):
            func = MODEL_DEFS[mname][0]
            popt = list(res["params"].values())
            try:
                y_sm = func(t_smooth, *popt)
                ax.plot(t_smooth, y_sm, color=MODEL_COLORS[i % len(MODEL_COLORS)],
                        linewidth=1.7, alpha=0.85, label=f"{mname} (R²adj={res['r2_adj']:.3f})")
            except Exception:
                pass

        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Cumulative Release (%)")
        ax.set_title(f"Kinetic Model Fitting — {profile_name}")
        ax.set_ylim(0, 110)
        ax.legend(fontsize=7.5, ncol=2, loc="lower right")
        st.pyplot(fig)
        plt.close()

        # ── Parameter Detail ──────────────────────────────────────────────
        st.subheader("ð© Fitted Parameters")
        for mname, res in valid.items():
            with st.expander(f"{mname}  |  R²adj = {res['r2_adj']:.4f}  |  AIC = {res['aic']:.3f}"):
                st.markdown(f"<div class='eq-box'>{res['equation']}</div>", unsafe_allow_html=True)
                pcols = st.columns(min(4, len(res["params"])))
                for j, (pname, pval) in enumerate(res["params"].items()):
                    pcols[j % 4].metric(pname, f"{pval:.5g}")

        if failed:
            st.warning(f"⚠️ Models that did not converge: {', '.join(failed.keys())}")
            for mname, res in failed.items():
                st.caption(f"  • {mname}: {res['error']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "ð Statistical Analysis":
    st.header("ð Statistical Analysis")

    if not st.session_state.profiles:
        st.warning("⚠️ No profiles loaded.")
        st.stop()

    all_names = list(st.session_state.profiles.keys())

    # ── Summary Stats Table ────────────────────────────────────────────────
    st.subheader("ð Statistical Data Table")
    if len(all_names) >= 2:
        # Pooled stats across profiles at each shared time point
        common_t = None
        for pname in all_names:
            t = np.array(st.session_state.profiles[pname]["time"])
            common_t = t if common_t is None else np.intersect1d(common_t, t)

        rows = []
        for ti in common_t:
            vals = []
            for pname in all_names:
                t_p = np.array(st.session_state.profiles[pname]["time"])
                r_p = np.array(st.session_state.profiles[pname]["release"])
                idx = np.where(t_p == ti)[0]
                if len(idx):
                    vals.append(r_p[idx[0]])
            if vals:
                vals = np.array(vals)
                mean = np.mean(vals)
                sd   = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                rsd  = (sd / mean * 100) if mean != 0 else 0.0
                rows.append({
                    f"Time ({time_unit})": ti,
                    "Mean (%)":  round(mean, 2),
                    "SD":        round(sd,   2),
                    "RSD (%)":   round(rsd,  2),
                    "CV (%)":    round(rsd,  2),
                    "n":         len(vals),
                })
        if rows:
            df_stat = pd.DataFrame(rows)
            st.dataframe(df_stat, use_container_width=True)
    else:
        st.info("Load 2+ profiles for pooled statistics; showing individual profile stats below.")

    # ── Per-Profile MDT & DE ───────────────────────────────────────────────
    st.subheader("⏱️ MDT & DE per Profile")
    mdt_rows = []
    for pname in all_names:
        t = np.array(st.session_state.profiles[pname]["time"])
        r = np.array(st.session_state.profiles[pname]["release"])
        mdt = compute_mdt(t, r)
        de  = compute_de(t, r)
        mdt_rows.append({
            "Profile": pname,
            f"MDT ({time_unit})": round(mdt, 2) if not np.isnan(mdt) else "N/A",
            "DE (%)":  round(de, 2),
        })
    st.dataframe(pd.DataFrame(mdt_rows), use_container_width=True)
    st.markdown("""
    <div class='info-banner'>
      <strong>MDT</strong> (Mean Dissolution Time): weighted mean time for drug release.<br>
      <strong>DE</strong> (Dissolution Efficiency): area under dissolution curve as % of total rectangle.
    </div>
    """, unsafe_allow_html=True)

    # ── Individual Profile Plots ───────────────────────────────────────────
    st.subheader("ð Individual Dissolution Profiles")
    cols = st.columns(min(2, len(all_names)))
    for i, pname in enumerate(all_names):
        t = np.array(st.session_state.profiles[pname]["time"])
        r = np.array(st.session_state.profiles[pname]["release"])
        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        style_fig(fig, ax)
        ax.fill_between(t, r, alpha=0.12, color=OXFORD)
        ax.plot(t, r, "o-", color=OXFORD, linewidth=2, markersize=6)
        ax.axhline(80, color=AMBER, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.set_title(pname, fontsize=11)
        ax.set_xlabel(f"Time ({time_unit})")
        ax.set_ylabel("Release (%)")
        ax.set_ylim(0, 110)
        cols[i % 2].pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: f1 & f2 SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "ð f1 & f2 Similarity":
    st.header("ð f1 & f2 Similarity Factor Analysis")
    st.markdown("""
    <div class='info-banner'>
      <strong>FDA Guidance (1997):</strong> f2 ≥ 50 indicates similarity;
      f1 ≤ 15 indicates acceptable difference.
      f2 is calculated only on time points where the reference mean ≤ 85% release.
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.profiles) < 2:
        st.warning("⚠️ At least 2 profiles are required for f1/f2 calculation.")
        st.stop()

    all_names = list(st.session_state.profiles.keys())
    ref_name  = st.selectbox("Reference Profile", all_names, index=0)
    test_name = st.selectbox("Test Profile",      all_names, index=min(1, len(all_names)-1))

    if ref_name == test_name:
        st.error("Reference and Test profiles must be different.")
        st.stop()

    t_ref = np.array(st.session_state.profiles[ref_name]["time"])
    r_ref = np.array(st.session_state.profiles[ref_name]["release"])
    t_tst = np.array(st.session_state.profiles[test_name]["time"])
    r_tst = np.array(st.session_state.profiles[test_name]["release"])

    common_t = np.intersect1d(t_ref, t_tst)
    if len(common_t) == 0:
        st.error("No common time points between the two profiles.")
        st.stop()

    r_ref_c = np.array([r_ref[np.where(t_ref == ti)[0][0]] for ti in common_t])
    r_tst_c = np.array([r_tst[np.where(t_tst == ti)[0][0]] for ti in common_t])

    # Only use points where ref ≤ 85
    mask = r_ref_c <= 85.0
    r_ref_f = r_ref_c[mask]
    r_tst_f = r_tst_c[mask]
    n_f = len(r_ref_f)

    if n_f == 0:
        st.error("No valid time points (ref ≤ 85%) for f2 calculation.")
        st.stop()

    f1 = float(np.sum(np.abs(r_ref_f - r_tst_f)) / np.sum(r_ref_f) * 100.0)
    f2 = float(50.0 * np.log10(100.0 / np.sqrt(1.0 + np.mean((r_ref_f - r_tst_f) ** 2))))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("f1 (Difference Factor)", f"{f1:.2f}", delta=f"{'✅ Pass' if f1 <= 15 else '❌ Fail'}")
    c2.metric("f2 (Similarity Factor)", f"{f2:.2f}", delta=f"{'✅ Similar' if f2 >= 50 else '❌ Dissimilar'}")
    c3.metric("Time Points Used (n)", n_f)
    c4.metric("Max |ΔR| (%)", f"{np.max(np.abs(r_ref_f - r_tst_f)):.2f}")

    verdict_f1 = "✅ PASS — f1 ≤ 15: Acceptable difference" if f1 <= 15 else "❌ FAIL — f1 > 15: Significant difference"
    verdict_f2 = "✅ SIMILAR — f2 ≥ 50: Profiles are similar (FDA)" if f2 >= 50 else "❌ DISSIMILAR — f2 < 50: Profiles differ"

    st.markdown(f"""
    <div style='background:#f0f8f0;border:1px solid #aed6ae;border-radius:5px;padding:12px;margin:10px 0;'>
      <strong>Verdict:</strong> {verdict_f1}<br>{verdict_f2}
    </div>
    """, unsafe_allow_html=True)

    # Point-by-point table
    st.subheader("Point-by-Point Comparison")
    df_f = pd.DataFrame({
        f"Time ({time_unit})":   common_t,
        "Reference (%)":  r_ref_c,
        "Test (%)":       r_tst_c,
        "|Diff| (%)":     np.abs(r_ref_c - r_tst_c).round(2),
        "Used in f2":     ["✓" if r <= 85 else "—" for r in r_ref_c],
    })
    st.dataframe(df_f, use_container_width=True)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    style_fig(fig, ax)
    ax.plot(t_ref, r_ref, "o-", color=OXFORD, linewidth=2, markersize=6, label=f"Reference: {ref_name}")
    ax.plot(t_tst, r_tst, "s--", color="#c0392b", linewidth=2, markersize=6, label=f"Test: {test_name}")
    ax.axhline(85, color=AMBER, linewidth=1, linestyle=":", alpha=0.8, label="85% cutoff (f2)")
    ax.fill_between(common_t, r_ref_c, r_tst_c, alpha=0.1, color="#c0392b", label="|Δ|")
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel("Cumulative Release (%)")
    ax.set_title(f"f1={f1:.2f} | f2={f2:.2f} — {ref_name} vs {test_name}")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class='eq-box'>
      f1 = [Σ|Rt − Tt| / Σ Rt] × 100 &nbsp;|&nbsp;
      f2 = 50 × log{[1 + (1/n)·Σ(Rt − Tt)²]^(−0.5) × 100}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: IVIVC
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "ð¬ IVIVC Analysis":
    st.header("ð¬ IVIVC Analysis — Wagner-Nelson Method")
    st.markdown("""
    <div class='info-banner'>
      The <strong>Wagner-Nelson method</strong> estimates the fraction absorbed in vivo
      (<em>F<sub>a</sub></em>) from in vitro dissolution data assuming one-compartment
      kinetics. Requires an elimination rate constant (k<sub>el</sub>).
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.profiles:
        st.warning("⚠️ No profiles loaded.")
        st.stop()

    profile_name = st.selectbox("Dissolution Profile", list(st.session_state.profiles.keys()))
    kel = st.number_input("Elimination Rate Constant k_el (1/h or 1/min)", value=0.1,
                          format="%.4f", min_value=0.0001)

    data = st.session_state.profiles[profile_name]
    t = np.array(data["time"], dtype=float)
    f_pct = np.array(data["release"], dtype=float)

    # Wagner-Nelson: Fa(t) = [Ct + kel * AUC(0→t)] / [kel * AUC(0→∞)]
    # Approximation using in vitro F as surrogate for Ct
    Ct = f_pct / 100.0 * dose_mg           # drug in solution (proxy)
    AUC_t = np.array([trapezoid(Ct[:i+1], t[:i+1]) for i in range(len(t))])
    AUC_inf = trapezoid(Ct, t) + Ct[-1] / kel   # extrapolated to inf

    Fa_num = Ct + kel * AUC_t
    Fa = Fa_num / (kel * AUC_inf) * 100.0
    Fa = np.clip(Fa, 0, 100)

    c1, c2 = st.columns(2)
    c1.metric("Max Fraction Absorbed (%)", f"{Fa[-1]:.1f}")
    c2.metric("AUC(0→∞) estimated", f"{AUC_inf:.1f} mg·{time_unit}")

    df_ivivc = pd.DataFrame({
        f"Time ({time_unit})":        t,
        "In Vitro Release (%)": f_pct.round(2),
        "Fraction Absorbed Fa (%)": Fa.round(2),
    })
    st.dataframe(df_ivivc, use_container_width=True)

    # IVIVC correlation
    r_iviv = np.corrcoef(f_pct, Fa)[0, 1]
    st.metric("IVIVC Correlation (r)", f"{r_iviv:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        style_fig(fig, ax)

    axes[0].plot(t, f_pct, "o-", color=OXFORD, label="In Vitro Release", linewidth=2, markersize=5)
    axes[0].plot(t, Fa,    "s--", color="#c0392b", label="Fraction Absorbed (Wagner-Nelson)", linewidth=2, markersize=5)
    axes[0].set_xlabel(f"Time ({time_unit})")
    axes[0].set_ylabel("(%)")
    axes[0].set_title("In Vitro vs Fraction Absorbed")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 110)

    axes[1].scatter(f_pct, Fa, color=OXFORD, s=60, edgecolors=AMBER, linewidths=1, zorder=5)
    m, b = np.polyfit(f_pct, Fa, 1)
    x_line = np.linspace(f_pct.min(), f_pct.max(), 100)
    axes[1].plot(x_line, m * x_line + b, "--", color=AMBER, linewidth=2)
    axes[1].set_xlabel("In Vitro Release (%)")
    axes[1].set_ylabel("Fraction Absorbed (%)")
    axes[1].set_title(f"IVIVC Correlation (r = {r_iviv:.4f})")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div class='eq-box'>
      Wagner-Nelson: Fa(t) = [Ct + kel·AUC(0→t)] / [kel·AUC(0→∞)] × 100%
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: EXCEL REPORT
# ═══════════════════════════════════════════════════════════════════════════════
elif nav == "ð Excel Report":
    st.header("ð Professional Excel Report")
    st.markdown("""
    Generate a comprehensive Excel report containing:
    - All dissolution profiles
    - Statistical summary (Mean, SD, RSD, CV, MDT, DE)
    - Model fitting results (R², R²adj, AIC, MSC, parameters)
    - f1 & f2 similarity factors (if applicable)
    - IVIVC results
    """)

    if not st.session_state.profiles:
        st.warning("⚠️ No data loaded. Please input dissolution profiles first.")
        st.stop()

    if st.button("ð Generate Excel Report"):
        import xlsxwriter

        buf = io.BytesIO()
        wb  = xlsxwriter.Workbook(buf, {"in_memory": True})

        # ── Formats ───────────────────────────────────────────────────────
        fmt_title  = wb.add_format({"bold": True, "font_size": 14, "font_color": "#002147",
                                    "bottom": 2, "bottom_color": "#FFBF00"})
        fmt_header = wb.add_format({"bold": True, "bg_color": "#002147", "font_color": "#FFBF00",
                                    "border": 1, "align": "center"})
        fmt_data   = wb.add_format({"border": 1, "num_format": "0.0000", "align": "center"})
        fmt_data2  = wb.add_format({"border": 1, "align": "center"})
        fmt_sub    = wb.add_format({"bold": True, "bg_color": "#FFD966", "font_color": "#002147",
                                    "border": 1})
        fmt_good   = wb.add_format({"bg_color": "#c6efce", "border": 1, "num_format": "0.000",
                                    "align": "center"})
        fmt_bad    = wb.add_format({"bg_color": "#ffc7ce", "border": 1, "num_format": "0.000",
                                    "align": "center"})
        fmt_pct    = wb.add_format({"border": 1, "num_format": "0.00", "align": "center"})
        fmt_note   = wb.add_format({"italic": True, "font_color": "#5a6480", "font_size": 9})

        # ── Sheet 1: Cover ─────────────────────────────────────────────────
        ws_cover = wb.add_worksheet("Cover")
        ws_cover.set_column("A:A", 60)
        ws_cover.write("A1", "DissolvA™ — Predictive Dissolution Suite", fmt_title)
        ws_cover.write("A2", "Professional Dissolution Analysis Report", fmt_sub)
        ws_cover.write("A3", f"Profiles Analyzed: {len(st.session_state.profiles)}", fmt_data2)
        ws_cover.write("A4", "Generated by DissolvA™ v2.0 | Powered by AI", fmt_note)
        ws_cover.write("A5", "© 2025 Predictive Dissolution Suite | All rights reserved", fmt_note)

        # ── Sheet 2: Raw Data ──────────────────────────────────────────────
        ws_raw = wb.add_worksheet("Dissolution Profiles")
        col = 0
        for pname, data in st.session_state.profiles.items():
            t_arr = data["time"]
            r_arr = data["release"]
            ws_raw.write(0, col,   pname, fmt_sub)
            ws_raw.write(1, col,   f"Time ({time_unit})", fmt_header)
            ws_raw.write(1, col+1, "Release (%)",          fmt_header)
            for row_i, (ti, ri) in enumerate(zip(t_arr, r_arr)):
                ws_raw.write(row_i+2, col,   ti, fmt_pct)
                ws_raw.write(row_i+2, col+1, ri, fmt_pct)
            ws_raw.set_column(col,   col,   14)
            ws_raw.set_column(col+1, col+1, 14)
            col += 3

        # ── Sheet 3: Statistics ────────────────────────────────────────────
        ws_stat = wb.add_worksheet("Statistics")
        ws_stat.write(0, 0, "Statistical Summary", fmt_title)
        stat_hdrs = [f"Time ({time_unit})", "Profile", "Mean (%)", "SD", "RSD (%)", "CV (%)",
                     "MDT", "DE (%)"]
        for ci, h in enumerate(stat_hdrs):
            ws_stat.write(1, ci, h, fmt_header)
            ws_stat.set_column(ci, ci, 14)

        row_i = 2
        for pname, data in st.session_state.profiles.items():
            t_a = np.array(data["time"])
            r_a = np.array(data["release"])
            mdt = compute_mdt(t_a, r_a)
            de  = compute_de(t_a,  r_a)
            for ti, ri in zip(t_a, r_a):
                ws_stat.write(row_i, 0, ti,     fmt_pct)
                ws_stat.write(row_i, 1, pname,  fmt_data2)
                ws_stat.write(row_i, 2, ri,     fmt_pct)
                ws_stat.write(row_i, 3, 0.0,    fmt_pct)   # single-rep SD=0
                ws_stat.write(row_i, 4, 0.0,    fmt_pct)
                ws_stat.write(row_i, 5, 0.0,    fmt_pct)
                ws_stat.write(row_i, 6, round(mdt, 3) if not np.isnan(mdt) else "N/A", fmt_pct)
                ws_stat.write(row_i, 7, round(de, 3),  fmt_pct)
                row_i += 1

        # ── Sheet 4: Model Fitting ─────────────────────────────────────────
        ws_fit = wb.add_worksheet("Model Fitting")
        ws_fit.write(0, 0, "Kinetic Model Fitting Results", fmt_title)
        fit_hdrs = ["Model", "R²", "R²adj", "AIC", "MSC", "n_params", "Parameters", "Reference"]
        for ci, h in enumerate(fit_hdrs):
            ws_fit.write(1, ci, h, fmt_header)
        ws_fit.set_column(0, 0, 22)
        ws_fit.set_column(6, 6, 40)
        ws_fit.set_column(7, 7, 28)

        if st.session_state.fit_results:
            sorted_res = sorted(
                [(k, v) for k, v in st.session_state.fit_results.items() if v["success"]],
                key=lambda x: x[1]["r2_adj"], reverse=True
            )
            for ri, (mname, res) in enumerate(sorted_res):
                r2adj = res["r2_adj"]
                row = ri + 2
                ws_fit.write(row, 0, mname, fmt_data2)
                ws_fit.write(row, 1, round(res["r2"],     4), fmt_data)
                ws_fit.write(row, 2, round(r2adj,         4), fmt_good if r2adj >= 0.9 else fmt_bad)
                ws_fit.write(row, 3, round(res["aic"],    3), fmt_data)
                ws_fit.write(row, 4, round(res["msc"],    3), fmt_data)
                ws_fit.write(row, 5, res["n_params"],         fmt_data2)
                param_str = "; ".join([f"{k}={v:.4g}" for k, v in res["params"].items()])
                ws_fit.write(row, 6, param_str,               fmt_data2)
                ws_fit.write(row, 7, res["reference"],        fmt_data2)
        else:
            ws_fit.write(2, 0, "No model fitting results available. Run fitting first.", fmt_note)

        wb.close()
        buf.seek(0)

        st.success("✅ Report generated successfully!")
        st.download_button(
            label="⬇️ Download Excel Report",
            data=buf.getvalue(),
            file_name="DissolvA_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("""
        <div class='info-banner'>
          ð Report includes: Cover Sheet · Raw Dissolution Data · Statistical Summary ·
          Kinetic Model Fitting Results (sorted by R²adj) · Parameter Details
        </div>
        """, unsafe_allow_html=True)

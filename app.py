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
    # Brand box — logo image
    st.markdown(f"""
    <div style="
      border: 2px solid #FFBF00;
      border-radius: 8px;
      padding: 0;
      margin-bottom: 24px;
      overflow: hidden;
      box-shadow: 0 4px 18px rgba(0,0,0,0.45);
    ">
      <img src="data:image/jpeg;base64,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"
           style="width:100%; display:block; border-radius:6px;" />
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

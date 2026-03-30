import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io
import base64

# --- 1. CONFIGURATION & CUSTOM CSS ---
st.set_page_config(page_title="DissolvA v16.0", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7bcf,#2e7bcf); color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SIDEBAR BRANDING (ORIGINAL HTML) ---
sidebar_header_html = """
<div style="background-color: #002147; padding: 25px 20px; border-radius: 12px; border-left: 5px solid #FFBF00; margin-bottom: 25px; text-align: center; box-shadow: 0px 4px 15px rgba(0,0,0,0.4);">
    <h1 style="color: #FFBF00; margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -1px; font-family: 'Montserrat', sans-serif;">DissolvA™</h1>
    <p style="color: #DCDCDC; margin: 10px 0 0 0; font-size: 1.0rem; font-style: italic; font-weight: 400; opacity: 0.9;">Predictive Dissolution Suite</p>
    <hr style="border: 0.5px solid #FFBF00; margin: 15px 0 20px 0; opacity: 0.4;">
    <div style="border: 1px solid rgba(255,191,0,0.3); padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.05);">
         <p style="color: white; margin: 0; font-size: 0.75rem; font-weight: bold; letter-spacing: 2.5px; text-transform: uppercase;">POWERED BY AI</p>
    </div>
</div>
<div style="margin-left: 15px; margin-right: 15px; margin-bottom: 25px;">
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <span style="font-size: 1.2rem; margin-right: 12px;">🧬</span>
        <span style="color: #FFBF00; font-size: 1.15rem; font-weight: 600;">Molecular View</span>
    </div>
    <div style="display: flex; align-items: center;">
        <span style="font-size: 1.2rem; margin-right: 12px;">⚙️</span>
        <span style="color: #FFBF00; font-size: 1.15rem; font-weight: 600;">Parameter Settings</span>
    </div>
</div>
<hr style="border: 0.5px solid #DCDCDC; margin: 10px 0 20px 0; opacity: 0.2;">
"""
st.sidebar.markdown(sidebar_header_html, unsafe_allow_html=True)

# --- 3. ANALYTICAL SUITE SELECTOR ---
st.sidebar.markdown('<p style="color: #333; font-weight: bold;">Analytical Suite:</p>', unsafe_allow_html=True)
menu = st.sidebar.radio("Select Module", 
    ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"], 
    label_visibility="collapsed")
st.sidebar.divider()

# --- 4. LANGUAGE & MODEL KNOWLEDGE BASE ---
LANG_DICT = {
    "Türkçe": {
        "time": "Zaman (Dakika)", "release": "Kümülatif İlaç Salımı (%)", "calc": "✅ Hesaplandı", "unsuitable": "❌ Uyumsuz",
        "best": "🏆 En Uygun Model", "stats": "📊 Veri İstatistiği & Profil", "graph": "🛠️ Model Uyumu Grafiği",
        "report": "📝 Akademik Değerlendirme", "model_title": "16 Kinetik Model Analizi", "unit": "dk"
    },
    "English": {
        "time": "Time (Minutes)", "release": "Cumulative Drug Release (%)", "calc": "✅ Calculated", "unsuitable": "❌ Unsuitable",
        "best": "🏆 Best Fit Model", "stats": "📊 Statistics & Profile", "graph": "🛠️ Model Fit Graph",
        "report": "📝 Academic Evaluation", "model_title": "16 Kinetic Model Analysis", "unit": "min"
    }
}

MODEL_KNOWLEDGE = {
    "Sıfır Derece": "Zamandan bağımsız, sabit hızda salımı açıklar. Genellikle kontrollü salım sistemlerinde görülür.",
    "Birinci Derece": "Salım hızı, kalan ilaç konsantrasyonu ile orantılıdır.",
    "Higuchi": "Suda çözünmeyen matris sistemlerinden difüzyon temelli salımı tanımlar.",
    "Korsmeyer-Peppas": "Polimerik sistemlerden salım mekanizmasını (Fickian/non-Fickian) 'n' değeriyle açıklar.",
    "Hixson-Crowell": "İlaç parçacıklarının yüzey alanı ve çapının zamanla azaldığı erozyonu tanımlar.",
    "Hopfenberg": "Silindirik veya küresel polimerlerin yüzey erozyonunu matematiksel olarak ifade eder.",
    "Makoid-Banakar": "Difüzyon, birinci derece ve patlama (burst) etkisini içeren hibrit bir modeldir.",
    "Peppas-Sahlin": "Difüzyonel ve gevşeme (relaxation) kaynaklı salımı birbirinden ayırır.",
    "Weibull (w/ Td)": "Salım sürecinin gecikme süresini (Td) ve dağılım şeklini analiz eder.",
    "Baker-Lonsdale": "Küresel matrislerden kontrollü ilaç salımını açıklayan Higuchi türevidir.",
    "Gompertz": "İn vitro salımın başlangıçta yavaş, sonra hızlanan ve doyuma ulaşan sigmoid yapısını açıklar.",
    "Kopcha": "Difüzyon ve erozyon oranlarını ayrıştırarak baskın mekanizmayı belirler.",
    "Quadratic": "Kısa süreli verilerde doğrusal olmayan eğilimleri modellemek için kullanılır.",
    "Logistic": "Simetrik bir S-eğrisi (sigmoid) gösteren salım profilleri için uygundur.",
    "Peppas-Rincon": "Fraktal boyutlu ve gözenekli matris sistemleri için optimize edilmiştir.",
    "Square Root of Mass": "Kütle değişimine dayalı erozyon kinetiğini hesaplar."
}

# --- 5. MATHEMATICAL ENGINE (ALL 16 MODELS) ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull_complex(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))
def hopfenberg(t, k, n): return 100 * (1 - (1 - np.maximum(k * t, 0))**n)
def makoid_banakar(t, k, n, c): return k * (t**n) * np.exp(-c * t)
def sq_root_mass(t, k): return 100 * (1 - np.sqrt(np.maximum(1 - k * t, 0)))
def quadratic(t, a, b): return a*t + b*(t**2)
def logistic(t, a, b, c): return a / (1 + np.exp(-b * (t - c)))
def peppas_rincon(t, k, n): return k * (t**n)
def baker_lonsdale(t, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t])

# --- 6. CORE CALCULATION FUNCTIONS ---
def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p_count

def calculate_model_independent(t, q):
    dt = np.diff(t, prepend=0)
    auc = np.cumsum(q * dt)
    de = (auc[-1] / (t[-1] * 100)) * 100
    dq = np.diff(q, prepend=0)
    t_mid = t - (dt / 2)
    mdt = np.sum(t_mid * dq) / q[-1] if q[-1] > 0 else 0
    return de, mdt

def calculate_f1_f2(ref_mean, test_mean):
    R, T = np.array(ref_mean), np.array(test_mean)
    n = len(R)
    f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
    f2 = 50 * np.log10((1 + (1/n) * np.sum((R - T)**2))**-0.5 * 100)
    return f1, f2

# --- 7. DATA LOADING ENGINE ---
test_file = st.sidebar.file_uploader("Test Formulation (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Reference Formulation (XLSX/CSV)", type=['xlsx', 'csv'])
selected_lang = st.sidebar.selectbox("Dil / Language", ["Türkçe", "English"])
L = LANG_DICT[selected_lang]

def load_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": v.shape[1], "raw": v[mask]}
    except: return None

test_data = load_data(test_file)
ref_data = load_data(ref_file)

# --- 8. MAIN UI LOGIC ---
if test_data:
    t_raw, q_raw = test_data["t"], test_data["mean"]
    de, mdt = calculate_model_independent(t_raw, q_raw)
    
    if menu == "📈 Salım Profilleri":
        st.subheader(L['stats'])
        c1, c2, c3 = st.columns(3)
        c1.metric("MDT", f"{mdt:.2f} {L['unit']}")
        c2.metric("DE", f"% {de:.2f}")
        c3.metric("Samples (n)", test_data["n"])
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_raw, q_raw, yerr=test_data["std"], fmt='-o', color='#002147', capsize=5, label="Test")
        if ref_data:
            ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--s', color='#FFBF00', capsize=5, label="Ref")
        ax.set_xlabel(L['time']); ax.set_ylabel(L['release']); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    elif menu == "🧮 Kinetik Model Fitting":
        st.subheader(L['model_title'])
        tf, qf = t_raw[(t_raw>0)&(q_raw>0)], q_raw[(t_raw>0)&(q_raw>0)]
        
        model_list = [
            ("Sıfır Derece", zero_order, [0.1], [0], [100]),
            ("Birinci Derece", first_order, [0.01], [0], [10]),
            ("Higuchi", higuchi, [1.0], [0], [500]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [500, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [1]),
            ("Hopfenberg", hopfenberg, [0.01, 1.0], [0, 1.0], [1, 3.0]),
            ("Makoid-Banakar", makoid_banakar, [1.0, 0.5, 0.01], [0, 0, 0], [500, 2, 1]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [100, 100, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 5, 500]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [10000, 10.0, 100]),
            ("Baker-Lonsdale", baker_lonsdale, [0.01], [0], [1]),
            ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [500, 100]),
            ("Quadratic", quadratic, [0.1, 0.01], [0, -1], [100, 1]),
            ("Logistic", logistic, [100, 0.1, 10], [50, 0, 0], [110, 2, 500]),
            ("Peppas-Rincon", peppas_rincon, [1.0, 0.5], [0, 0.1], [500, 2.0]),
            ("Square Root of Mass", sq_root_mass, [0.01], [0], [1])
        ]
        
        fitting_results = []
        best_aic = float('inf')
        best_model_obj = None
        
        for name, func, p0, low, up in model_list:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=10000)
                y_p = func(tf, *popt)
                r2 = r2_score(qf, y_p)
                aic = calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0))
                fitting_results.append({"Model": name, "R²": r2, "AIC": aic, "Status": L['calc']})
                if aic < best_aic:
                    best_aic = aic
                    best_model_obj = (name, func, popt)
            except:
                fitting_results.append({"Model": name, "R²": 0, "AIC": 9999, "Status": L['unsuitable']})
        
        res_df = pd.DataFrame(fitting_results).sort_values("AIC")
        st.table(res_df.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}))
        
        if best_model_obj:
            st.success(f"{L['best']}: {best_model_obj[0]}")
            st.info(MODEL_KNOWLEDGE.get(best_model_obj[0], ""))
            fig2, ax2 = plt.subplots()
            ax2.scatter(tf, qf, color='black', label='Experimental')
            t_plot = np.linspace(tf.min(), tf.max(), 100)
            ax2.plot(t_plot, best_model_obj[1](t_plot, *best_model_obj[2]), 'r-', label='Best Fit Line')
            ax2.set_xlabel(L['time']); ax2.set_ylabel(L['release']); ax2.legend()
            st.pyplot(fig2)

    elif menu == "🧬 IVIVC Analizi":
        st.subheader("Wagner-Nelson Absorption Analysis")
        ke = st.number_input("Elimination Constant (ke) [1/h]", value=0.1500, step=0.0001, format="%.4f")
        dt = np.diff(t_raw, prepend=0) / 60
        cum_auc = np.cumsum(q_raw * dt)
        f_abs = (q_raw + ke * cum_auc) / (ke * (cum_auc[-1] + q_raw[-1]/ke))
        st.line_chart(pd.DataFrame({"Fraction Absorbed": f_abs}, index=t_raw))

    elif menu == "📊 f1 & f2 Benzerlik Analizi":
        if ref_data:
            common_n = min(len(t_raw), len(ref_data["t"]))
            f1, f2 = calculate_f1_f2(ref_data["mean"][:common_n], q_raw[:common_n])
            st.metric("f1 (Difference Factor)", f"{f1:.2f}")
            st.metric("f2 (Similarity Factor)", f"{f2:.2f}")
            if f2 >= 50: st.balloons(); st.success("Similarity confirmed (f2 >= 50)")
            else: st.warning("Similarity not confirmed (f2 < 50)")
        else:
            st.error("Please upload a Reference dataset to compare.")

# --- 9. EXCEL REPORTING ENGINE ---
def generate_excel(df_res, test_meta):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False, sheet_name='ModelFitting')
        pd.DataFrame([test_meta]).to_excel(writer, index=False, sheet_name='Summary')
    return output.getvalue()

if test_data and 'res_df' in locals():
    excel_data = generate_excel(res_df, {"MDT": mdt, "DE": de, "Samples": test_data["n"]})
    st.sidebar.download_button("📥 Akademik Raporu İndir (.xlsx)", excel_data, "DissolvA_Full_Report.xlsx")

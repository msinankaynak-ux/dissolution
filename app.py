import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. CONFIG ---
st.set_page_config(page_title="DissolvA v16.0", layout="wide")

# --- 2. SIDEBAR LOGO VE TASARIM ---
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
        <span style="color: #DCDCDC; font-size: 0.85rem; font-weight: 400; margin-left: auto; opacity: 0.8;">(Prediction)</span>
    </div>
    <div style="display: flex; align-items: center;">
        <span style="font-size: 1.2rem; margin-right: 12px;">⚙️</span>
        <span style="color: #FFBF00; font-size: 1.15rem; font-weight: 600;">Parameter Settings</span>
        <span style="color: #DCDCDC; font-size: 0.85rem; font-weight: 400; margin-left: auto; opacity: 0.8;">(Optimization)</span>
    </div>
</div>
<hr style="border: 0.5px solid #DCDCDC; margin: 10px 0 20px 0; opacity: 0.2;">
"""

st.sidebar.markdown(sidebar_header_html, unsafe_allow_html=True)

# --- 3. ANALYTICAL SUITE SELECTION ---
st.sidebar.markdown('<div style="color: #333333; font-size: 1.1rem; font-weight: bold; margin-bottom: 10px;">Analytical Suite:</div>', unsafe_allow_html=True)
menu = st.sidebar.radio(
    label="Menu Selection",
    options=["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"],
    label_visibility="collapsed"
)
st.sidebar.divider()

# --- 4. MODEL BİLGİ BANKASI ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Zamandan bağımsız sabit hızda salımı açıklar.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Matris sistemlerinden difüzyon temelli salımı açıklar.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Mekanizma 'n' üsteli ile tanımlanır.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Yüzey alanı ve çapın zamanla küçüldüğü erozyonu açıklar.",
        "Hopfenberg": "Hopfenberg modeline uymaktadır. Yüzeyden aşınan polimerlerin geometrik erozyonunu açıklar.",
        "Makoid-Banakar": "Makoid-Banakar modeline uymaktadır. Hem difüzyon hem de birinci dereceyi kapsar.",
        "Square Root of Mass": "Kütle karekök modeline uymaktadır. Erozyonu kütle değişimi üzerinden hesaplar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Difüzyonel ve erozyon katkısını birbirinden ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. Gecikmeli başlayan sigmoid profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Ölçek, şekil ve gecikme süresini karakterize eder.",
        "Baker-Lonsdale": "Baker-Lonsdale modeline uymaktadır. Küresel matrislerden salımı açıklar.",
        "Kopcha": "Kopcha modeline uymaktadır. Difüzyon ve erozyon oranlarını ayrıştırır.",
        "Quadratic": "Quadratic modeline uymaktadır. Kısa süreli ve doğrusal olmayan salımları açıklar.",
        "Peppas-Rincon": "Peppas-Rincon modeline uymaktadır. Karmaşık geometriler için geliştirilmiştir.",
        "Logistic": "Lojistik modele uymaktadır. Simetrik sigmoid (S-tipi) salımları açıklar."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics.",
        "Birinci Derece": "fits First-Order kinetics.",
        "Higuchi": "fits the Higuchi model.",
        "Korsmeyer-Peppas": "fits the Korsmeyer-Peppas model.",
        "Hixson-Crowell": "fits Hixson-Crowell kinetics.",
        "Hopfenberg": "fits the Hopfenberg model.",
        "Makoid-Banakar": "fits the Makoid-Banakar model.",
        "Square Root of Mass": "fits the Square Root of Mass model.",
        "Peppas-Sahlin": "fits the Peppas-Sahlin model.",
        "Gompertz": "fits the Gompertz model.",
        "Weibull (w/ Td)": "fits the Weibull model.",
        "Baker-Lonsdale": "fits the Baker-Lonsdale model.",
        "Kopcha": "fits the Kopcha model.",
        "Quadratic": "fits the Quadratic model.",
        "Peppas-Rincon": "fits the Peppas-Rincon model.",
        "Logistic": "fits the Logistic model."
    }
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman (Dakika)", "release": "Kümülatif İlaç Salımı", "calc": "✅ Hesaplandı", "unsuitable": "❌ Uyumsuz", 
        "best": "🏆 En Uygun Model", "stats": "📊 Veri İstatistiği & Profil", "graph": "🛠️ Model Uyumu Grafiği", 
        "report": "📝 Akademik Değerlendirme", "model_title": "16 Kinetik Model Analizi", "unit": "dk"
    },
    "English": {
        "time": "Time (Minutes)", "release": "Cumulative Drug Release", "calc": "✅ Calculated", "unsuitable": "❌ Unsuitable", 
        "best": "🏆 Best Fit Model", "stats": "📊 Statistics & Profile", "graph": "🛠️ Model Fit Graph", 
        "report": "📝 Academic Evaluation", "model_title": "16 Kinetic Model Analysis", "unit": "min"
    }
}

# --- 5. MATEMATİKSEL FONKSİYONLAR ---
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

def baker_lonsdale_for_fit(t_data, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t_data])

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p_count

def calculate_f1_f2(ref_mean, test_mean):
    R, T = np.array(ref_mean), np.array(test_mean)
    n = len(R)
    f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
    sum_sq_diff = np.sum((R - T)**2)
    f2 = 50 * np.log10((1 + (1/n) * sum_sq_diff)**-0.5 * 100)
    return f1, f2

def calculate_model_independent(t, q):
    dt = np.diff(t, prepend=0)
    auc = np.cumsum(q * dt)
    de = (auc[-1] / (t[-1] * 100)) * 100
    dq = np.diff(q, prepend=0)
    t_mid = t - (dt / 2)
    mdt = np.sum(t_mid * dq) / q[-1] if q[-1] > 0 else 0
    return de, mdt

def generate_excel_report(test_data, model_results, best_model, mdt_de, f1f2=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_data = {
            "Parametre": ["En Uygun Model", "MDT", "DE (%)", "n"],
            "Değer": [best_model, f"{mdt_de[1]:.2f}", f"{mdt_de[0]:.2f}", test_data['n']]
        }
        if f1f2:
            summary_data["Parametre"].extend(["f1", "f2"])
            summary_data["Değer"].extend([f"{f1f2[0]:.2f}", f"{f1f2[1]:.2f}"])
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Ozet', index=False)
        if model_results is not None:
            pd.DataFrame(model_results).to_excel(writer, sheet_name='Modeller', index=False)
    return output.getvalue()

# --- 6. VERİ GİRİŞİ ---
test_file = st.sidebar.file_uploader("Test Verisi (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi (Opsiyonel)", type=['xlsx', 'csv'])
selected_lang = st.sidebar.selectbox("Dil / Language:", ["Türkçe", "English"])
L = LANG_DICT[selected_lang]

def process_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": v.shape[1]}
    except: return None

test_data = process_data(test_file)
ref_data = process_data(ref_file)
results, best_name, de, mdt, f1, f2 = None, "Analiz Edilmedi", 0, 0, None, None

if test_data:
    t_raw, q_raw = test_data["t"], test_data["mean"]
    de, mdt = calculate_model_independent(t_raw, q_raw)
    
    if menu == "📈 Salım Profilleri":
        st.subheader(L['stats'])
        stats_df = pd.DataFrame({L['time']: t_raw, "Mean": q_raw, "SD": test_data["std"]})
        st.table(stats_df.style.format("{:.2f}").hide(axis="index"))
        fig, ax = plt.subplots(); ax.errorbar(t_raw, q_raw, yerr=test_data["std"], fmt='-ok', label="Test")
        if ref_data: ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Ref")
        ax.legend(); st.pyplot(fig)

    elif menu == "🧮 Kinetik Model Fitting":
        st.subheader(L['model_title'])
        tf, qf = t_raw[(t_raw>0)&(q_raw>0)], q_raw[(t_raw>0)&(q_raw>0)]
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1], [0], [100]), ("Birinci Derece", first_order, [0.01], [0], [10]),
            ("Higuchi", higuchi, [1.0], [0], [500]), ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [500, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [1]), ("Hopfenberg", hopfenberg, [0.01, 1.0], [0, 1.0], [1, 3.0]),
            ("Makoid-Banakar", makoid_banakar, [1.0, 0.5, 0.01], [0, 0, 0], [500, 2, 1]),
            ("Square Root of Mass", sq_root_mass, [0.01], [0], [1]), ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [500, 100]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [100, 100, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 5, 500]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [10000, 10.0, 100]),
            ("Quadratic", quadratic, [0.1, 0.01], [0, -1], [100, 1]), ("Logistic", logistic, [100, 0.1, 10], [50, 0, 0], [110, 2, 500]),
            ("Peppas-Rincon", peppas_rincon, [1.0, 0.5], [0, 0.1], [500, 2.0])
        ]
        res_list = []
        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=10000)
                y_p = func(tf, *popt)
                res_list.append({"Model": name, "R²": r2_score(qf, y_p), "AIC": calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0)), "Durum": L['calc']})
            except: res_list.append({"Model": name, "R²": 0, "AIC": 9999, "Durum": L['unsuitable']})
        results = pd.DataFrame(res_list)
        best_idx = results[results["Durum"] == L['calc']]["AIC"].idxmin()
        best_name = results.loc[best_idx, "Model"]
        st.table(results.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))
        st.info(f"🏆 {best_name}: {MODEL_KNOWLEDGE[selected_lang].get(best_name, '')}")

    elif menu == "🧬 IVIVC Analizi":
        ke = st.number_input("ke [1/h]:", value=0.1500, format="%.4f")
        dt = np.diff(t_raw, prepend=0)
        cum_auc = np.cumsum(q_raw * dt)
        f_abs = (q_raw + ke * cum_auc) / (ke * (cum_auc[-1] + q_raw[-1]/ke))
        st.line_chart(f_abs)

    elif menu == "📊 f1 & f2 Benzerlik Analizi":
        if ref_data:
            c_len = min(len(t_raw), len(ref_data["t"]))
            f1, f2 = calculate_f1_f2(ref_data["mean"][:c_len], q_raw[:c_len])
            st.metric("f1", f"{f1:.2f}"); st.metric("f2", f"{f2:.2f}")
        else: st.info("Ref yükleyin.")

if test_data:
    excel_data = generate_excel_report(test_data, results, best_name, (de, mdt), (f1, f2) if f1 else None)
    st.sidebar.download_button("📥 Raporu İndir", excel_data, "DissolvA_Report.xlsx")

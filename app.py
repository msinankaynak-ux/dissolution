import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. GENİŞLETİLMİŞ LİTERATÜR BİLGİ BANKASI ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Zamandan bağımsız sabit hızda salımı açıklar.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Matris sistemlerinden difüzyon temelli salımı açıklar.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Mekanizma 'n' üsteli ile tanımlanır.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. İlaç parçacıklarının yüzey alanı ve çapının zamanla küçüldüğü (erozyon) durumları açıklar.",
        "Hopfenberg": "Hopfenberg modeline uymaktadır. Yüzeyden aşınan (surface-eroding) polimerlerin geometrik (levha, silindir, küre) erozyonunu açıklar.",
        "Makoid-Banakar": "Makoid-Banakar modeline uymaktadır. Hem difüzyon hem de birinci dereceyi kapsar; başlangıçtaki 'burst release' (ani salım) etkisini ölçer.",
        "Square Root of Mass": "Kütle karekök modeline uymaktadır. Hixson-Crowell'e benzer ancak erozyonu kütle değişimi üzerinden hesaplar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Difüzyonel ve polimer relaksasyonu (erozyon) katkısını birbirinden ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. Gecikmeli başlayan sigmoid (S-tipi) profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Profilin ölçek, şekil ve gecikme süresini karakterize eder."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics, describing a constant release rate independent of time.",
        "Birinci Derece": "fits First-Order kinetics. The release rate is concentration-dependent.",
        "Higuchi": "fits the Higuchi model, describing diffusion-based release from matrix systems.",
        "Korsmeyer-Peppas": "fits the Korsmeyer-Peppas model, where the mechanism is defined by the 'n' exponent.",
        "Hixson-Crowell": "fits Hixson-Crowell kinetics, explaining cases where surface area and particle diameter decrease (erosion) over time.",
        "Hopfenberg": "fits the Hopfenberg model, explaining surface-eroding polymers for specific geometries (slab, cylinder, sphere).",
        "Makoid-Banakar": "fits the Makoid-Banakar model, covering both diffusion and first-order release while accounting for initial 'burst release'.",
        "Square Root of Mass": "fits the Square Root of Mass model, similar to Hixson-Crowell but based on mass change erosion.",
        "Peppas-Sahlin": "fits the Peppas-Sahlin model, separating diffusion and relaxation contributions.",
        "Gompertz": "fits the Gompertz model, explaining sigmoid (S-type) lag-time profiles.",
        "Weibull (w/ Td)": "fits the Weibull model, characterizing profile scale, shape, and lag time."
    }
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman", "release": "Salım", "test": "Test", "ref": "Referans", "model_fit": "Model Uygunluğu", 
        "comment": "Yorum", "calc": "✅ Hesaplanabilir", "unsuitable": "❌ Uyumsuz", "best": "🏆 En Uygun", 
        "stats": "📊 Veri İstatistiği", "graph": "🛠️ Model Uyumu Grafiği", "report_title": "📝 Akademik Değerlendirme"
    },
    "English": {
        "time": "Time", "release": "Release", "test": "Test", "ref": "Reference", "model_fit": "Model Suitability", 
        "comment": "Comment", "calc": "✅ Calculable", "unsuitable": "❌ Unsuitable", "best": "🏆 Best Fit", 
        "stats": "📊 Data Statistics", "graph": "🛠️ Model Fit Graph", "report_title": "📝 Academic Evaluation"
    }
}

# --- 2. MATEMATİKSEL MODELLER ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull_complex(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))
def hopfenberg(t, k, n): return 100 * (1 - (1 - k * t)**n)
def makoid_banakar(t, k, n, c): return k * (t**n) * np.exp(-c * t)
def sq_root_mass(t, k): return 100 * (1 - np.sqrt(1 - k * t))

def baker_lonsdale_for_fit(t_data, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t_data])

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p_count

# --- 3. STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="PharmTech Lab v15.7", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

menu = st.sidebar.radio("Analiz Adımları / Steps:", ["📈 1. Salım Profilleri", "🧮 2. Tüm Modelleri Test Et"])
st.sidebar.divider()
test_file = st.sidebar.file_uploader("Test Verisi (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans (Opsiyonel)", type=['xlsx', 'csv'])
st.sidebar.divider()
selected_lang = st.sidebar.selectbox("Dil / Language:", ["Türkçe", "English"])
L = LANG_DICT[selected_lang]

def load_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None

test = load_data(test_file)

if test:
    t_raw, q_raw = test["t"], test["mean"]
    fit_mask = (t_raw > 0) & (q_raw > 0)
    tf, qf = t_raw[fit_mask], q_raw[fit_mask]

    if "1." in menu:
        st.subheader(f"📍 {L['release']} Profili")
        rsd = (test["std"] / np.where(q_raw==0, 1, q_raw)) * 100
        stats_df = pd.DataFrame({L['time']: t_raw, "Mean (%)": q_raw, "SD": test["std"], "RSD (%)": rsd})
        st.write(f"### {L['stats']}")
        st.table(stats_df.style.format("{:.2f}").hide(axis="index"))
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.errorbar(t_raw, q_raw, yerr=test["std"], fmt='-ok', capsize=5)
        ax.set_xlabel(L['time']); ax.set_ylabel(L['release'] + " (%)"); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif "2." in menu:
        st.subheader(f"🔍 {L['model_fit']}")
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1], [0], [100]),
            ("Birinci Derece", first_order, [0.01], [0], [10]),
            ("Higuchi", higuchi, [1.0], [0], [500]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [500, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [1]),
            ("Hopfenberg", hopfenberg, [0.01, 1.0], [0, 1.0], [1, 3.0]),
            ("Makoid-Banakar", makoid_banakar, [1.0, 0.5, 0.01], [0, 0, 0], [500, 2, 1]),
            ("Square Root of Mass", sq_root_mass, [0.01], [0], [1]),
            ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [500, 100]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [100, 100, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 5, 500]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [10000, 10.0, 100])
        ]
        
        results = []; fit_plots = {}
        try:
            popt_bl, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[0.001])
            y_bl = baker_lonsdale_for_fit(t_raw, *popt_bl)
            results.append({"Model": "Baker-Lonsdale", "R²": r2_score(q_raw, y_bl), "AIC": calculate_aic(len(t_raw), np.sum((q_raw-y_bl)**2), 1), L['model_fit']: L['calc'], L['comment']: f"k: {popt_bl[0]:.5f}"})
            fit_plots["Baker-Lonsdale"] = (baker_lonsdale_for_fit, popt_bl)
        except: pass

        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=15000)
                y_p = func(tf, *popt)
                res = {"Model": name, "R²": r2_score(qf, y_p), "AIC": calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0)), L['model_fit']: L['calc']}
                res[L['comment']] = f"k: {popt[0]:.4f}" + (f", n: {popt[1]:.3f}" if len(popt)>1 else "")
                results.append(res); fit_plots[name] = (func, popt)
            except:
                results.append({"Model": name, "R²": 0, "AIC": 9999, L['model_fit']: L['unsuitable'], L['comment']: "-"})

        df_res = pd.DataFrame(results)
        best_idx = df_res[df_res[L['model_fit']] == L['calc']]["AIC"].idxmin()
        best_name = df_res.loc[best_idx, "Model"]
        df_res.at[best_idx, L['comment']] += f" {L['best']}"
        st.table(df_res.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))

        st.divider(); st.subheader(L['report_title'])
        know = MODEL_KNOWLEDGE[selected_lang]
        st.info(f"**{best_name}** {know.get(best_name, '')}\n\n(R²: {df_res.loc[best_idx, 'R²']:.4f}, AIC: {df_res.loc[best_idx, 'AIC']:.2f})")

        st.write(f"### {L['graph']}")
        sel = st.multiselect("Grafikte Göster:", list(fit_plots.keys()), default=[best_name])
        if sel:
            fig_m, ax_m = plt.subplots(figsize=(10,6)); ax_m.scatter(t_raw, q_raw, c='k', label="Data")
            t_p = np.linspace(t_raw.min(), t_raw.max(), 100)
            for m in sel:
                f, p = fit_plots[m]; ax_m.plot(t_p, f(t_p, *p), label=m)
            ax_m.legend(); ax_m.set_xlabel(L['time']); ax_m.set_ylabel(L['release']+" (%)"); st.pyplot(fig_m)
else:
    st.info("Lütfen bir veri dosyası yükleyerek başlayın.")

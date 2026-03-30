import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. GENİŞLETİLMİŞ MODEL VE HATA TANIMLARI ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Zamandan bağımsız sabit hızda salımı açıklar.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Matris sistemlerinden difüzyon temelli salımı açıklar.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Mekanizma 'n' üsteli ile tanımlanır.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Yüzey alanı ve çapın zamanla küçüldüğü erozyonu açıklar.",
        "Hopfenberg": "Hopfenberg modeline uymaktadır. Yüzeyden aşınan polimerlerin geometrik erozyonunu açıklar.",
        "Makoid-Banakar": "Makoid-Banakar modeline uymaktadır. Başlangıçtaki 'burst release' etkisini ölçer.",
        "Square Root of Mass": "Kütle karekök modeline uymaktadır. Kütle değişimi üzerinden erozyonu hesaplar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Difüzyon ve relaksasyon katkısını ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. Gecikmeli başlayan sigmoid (S-tipi) profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Profilin ölçek ve gecikme süresini karakterize eder.",
        "Baker-Lonsdale": "Baker-Lonsdale modeline uymaktadır. Küresel matrislerden salımı açıklar.",
        "Kopcha": "Kopcha modeline uymaktadır. Difüzyon ve erozyon oranlarını ayrıştırır.",
        "Quadratic": "Quadratic modeline uymaktadır. Çok kısa süreli ve doğrusal olmayan salımları açıklar.",
        "Peppas-Rincon": "Peppas-Rincon modeline uymaktadır. Çok katmanlı karmaşık sistemleri tanımlar.",
        "Logistic": "Lojistik modele uymaktadır. Simetrik sigmoid (S-tipi) salımları açıklar."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics. Describes a constant release rate independent of time.",
        "Birinci Derece": "fits First-Order kinetics. Release rate is concentration-dependent.",
        "Higuchi": "fits the Higuchi model. Describes diffusion-based release from matrices.",
        "Korsmeyer-Peppas": "fits the Korsmeyer-Peppas model. Mechanism defined by 'n' exponent.",
        "Hixson-Crowell": "fits Hixson-Crowell kinetics. Explains surface area and diameter reduction erosion.",
        "Hopfenberg": "fits the Hopfenberg model. Explains geometric erosion of surface-eroding polymers.",
        "Makoid-Banakar": "fits the Makoid-Banakar model. Measures initial 'burst release' effect.",
        "Square Root of Mass": "fits the Square Root of Mass model. Calculates erosion via mass change.",
        "Peppas-Sahlin": "fits the Peppas-Sahlin model. Separates diffusion and relaxation.",
        "Gompertz": "fits the Gompertz model. Explains sigmoid (S-type) lag-time profiles.",
        "Weibull (w/ Td)": "fits the Weibull model. Characterizes profile scale and lag time.",
        "Baker-Lonsdale": "fits the Baker-Lonsdale model. Explains release from spherical matrices.",
        "Kopcha": "fits the Kopcha model. Decouples diffusion and erosion rates.",
        "Quadratic": "fits the Quadratic model. Explains short-term non-linear release.",
        "Peppas-Rincon": "fits the Peppas-Rincon model. Describes complex multilayer systems.",
        "Logistic": "fits the Logistic model. Explains symmetric sigmoid (S-type) profiles."
    }
}

UNSUITABLE_DESC = {
    "Türkçe": "⚠️ Veri yapısı bu modelin varsayımlarına (örn: sigmoid yapı, erozyon hızı veya lag-time) istatistiksel olarak uymuyor.",
    "English": "⚠️ Data structure does not statistically fit the model's assumptions (e.g., sigmoid shape, erosion rate, or lag-time)."
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman", "release": "Salım", "model_fit": "Model Uygunluğu", "comment": "Akademik Yorum", 
        "calc": "✅ Hesaplandı", "unsuitable": "❌ Uyumsuz", "best": "🏆 En Uygun Model",
        "stats": "📊 Veri İstatistiği", "graph": "🛠️ Model Uyumu Grafiği", "report": "📝 Akademik Değerlendirme"
    },
    "English": {
        "time": "Time", "release": "Release", "model_fit": "Model Suitability", "comment": "Academic Comment", 
        "calc": "✅ Calculated", "unsuitable": "❌ Unsuitable", "best": "🏆 Best Fit",
        "stats": "📊 Data Statistics", "graph": "🛠️ Model Fit Graph", "report": "📝 Academic Evaluation"
    }
}

# --- 2. MATEMATİKSEL FONKSİYONLAR ---
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
def quadratic(t, a, b): return a*t + b*(t**2)
def logistic(t, a, b, c): return a / (1 + np.exp(-b * (t - c)))
def peppas_rincon(t, k, n): return k * (t**n) # Basit form, kompleks versiyonu matris gerektirir

def baker_lonsdale_for_fit(t_data, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t_data])

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p_count

# --- 3. STREAMLIT ARAYÜZÜ ---
st.set_page_config(page_title="PharmTech Lab v15.8", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

# Ana Menü
menu = st.sidebar.radio("Ana İşlemler:", ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC & Farmakokinetik"])
st.sidebar.divider()
test_file = st.sidebar.file_uploader("Veri Yükle (XLSX/CSV)", type=['xlsx', 'csv'])
st.sidebar.divider()
selected_lang = st.sidebar.selectbox("Dil / Language:", ["Türkçe", "English"])
L = LANG_DICT[selected_lang]

def load_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    mask = ~np.isnan(t)
    return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}

data = load_data(test_file)

# --- 4. ÇALIŞMA MODULLERİ ---
if data:
    t_raw, q_raw = data["t"], data["mean"]
    tf, qf = t_raw[(t_raw>0)&(q_raw>0)], q_raw[(t_raw>0)&(q_raw>0)]

    if menu == "📈 Salım Profilleri":
        st.subheader(L['stats'])
        stats_df = pd.DataFrame({L['time']: t_raw, "Mean (%)": q_raw, "SD": data["std"]})
        st.table(stats_df.style.format("{:.2f}").hide(axis="index"))
        fig, ax = plt.subplots(); ax.errorbar(t_raw, q_raw, yerr=data["std"], fmt='-ok'); st.pyplot(fig)

    elif menu == "🧮 Kinetik Model Fitting":
        st.subheader("16 Kinetik Model Karşılaştırması")
        
        # 16 Model Tanımı (Baker-Lonsdale hariç 15)
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
        
        results = []; fit_plots = {}
        # Baker-Lonsdale (Özel Root Solver)
        try:
            popt_bl, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[0.001])
            y_bl = baker_lonsdale_for_fit(t_raw, *popt_bl)
            results.append({"Model": "Baker-Lonsdale", "R²": r2_score(q_raw, y_bl), "AIC": calculate_aic(len(t_raw), np.sum((q_raw-y_bl)**2), 1), L['model_fit']: L['calc']})
            fit_plots["Baker-Lonsdale"] = (baker_lonsdale_for_fit, popt_bl)
        except:
            results.append({"Model": "Baker-Lonsdale", "R²": 0, "AIC": 9999, L['model_fit']: L['unsuitable']})

        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=10000)
                y_p = func(tf, *popt)
                results.append({"Model": name, "R²": r2_score(qf, y_p), "AIC": calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0)), L['model_fit']: L['calc']})
                fit_plots[name] = (func, popt)
            except:
                results.append({"Model": name, "R²": 0, "AIC": 9999, L['model_fit']: L['unsuitable']})

        df_res = pd.DataFrame(results)
        best_idx = df_res[df_res[L['model_fit']] == L['calc']]["AIC"].idxmin()
        best_name = df_res.loc[best_idx, "Model"]
        
        st.table(df_res.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))
        
        # Akademik Rapor ve Hata Uyarıları
        st.divider(); st.subheader(L['report'])
        know = MODEL_KNOWLEDGE[selected_lang]
        
        # Seçilen modelin yorumu
        st.info(f"🏆 **{best_name}**: {know.get(best_name, '')}")
        
        # Uygun olmayan modellerin mekanistik uyarısı
        with st.expander("Uyumsuz Modeller Hakkında Notlar / Notes on Unsuitable Models"):
            st.write(UNSUITABLE_DESC[selected_lang])

    elif menu == "🧬 IVIVC & Farmakokinetik":
        st.subheader("IVIVC Analizi (Wagner-Nelson & Loo-Riegelman)")
        st.info("Bu bölüm, in-vitro salım verilerinden in-vivo absorbsiyon tahmini yapar.")
        ke = st.number_input("Eliminasyon Sabiti (ke) [1/h]:", value=0.15, format="%.4f")
        
        # Wagner-Nelson Hesaplama
        # f_abs = (Ct + ke * AUC_0_t) / (ke * AUC_0_inf)
        auc = np.trapz(q_raw, t_raw)
        cum_auc = [np.trapz(q_raw[:i+1], t_raw[:i+1]) for i in range(len(t_raw))]
        f_abs = (q_raw + ke * np.array(cum_auc)) / (ke * auc if auc>0 else 1)
        
        ivivc_df = pd.DataFrame({L['time']: t_raw, "In-vitro Release (%)": q_raw, "Fraction Absorbed (W-N)": f_abs})
        st.write("### Wagner-Nelson Tahminleme Tablosu")
        st.table(ivivc_df.style.format("{:.4f}").hide(axis="index"))
        
        fig_iv, ax_iv = plt.subplots(); ax_iv.plot(t_raw, f_abs, 'r-o', label="Absorbsiyon (Tahmin)"); ax_iv.legend(); st.pyplot(fig_iv)

else:
    st.warning("Lütfen veri yükleyiniz.")

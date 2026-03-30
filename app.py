import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score

# --- 1. MODEL BİLGİ BANKASI ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Zamandan bağımsız sabit hızda salımı açıklar.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı konsantrasyona bağlıdır.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Matris sistemlerinden difüzyon temelli salımı açıklar.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. n üsteli ile mekanizma tanımlanır.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Erozyonla küçülen yüzey alanını açıklar.",
        "Hopfenberg": "Hopfenberg modeline uymaktadır. Yüzeyden aşınan polimer geometrisini açıklar.",
        "Makoid-Banakar": "Makoid-Banakar modeline uymaktadır. Burst release etkisini ölçer.",
        "Square Root of Mass": "Kütle karekök modeline uymaktadır. Kütle değişimi bazlı erozyonu açıklar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Difüzyon ve erozyon katkısını ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. Sigmoid (S-tipi) gecikmeli profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Profil şekli ve gecikme süresini tanımlar.",
        "Baker-Lonsdale": "Baker-Lonsdale modeline uymaktadır. Küresel matris salımını açıklar.",
        "Kopcha": "Kopcha modeline uymaktadır. Difüzyon/erozyon oranlarını verir.",
        "Quadratic": "Quadratic modeline uymaktadır. Kısa süreli non-lineer salımları açıklar.",
        "Peppas-Rincon": "Peppas-Rincon modeline uymaktadır. Çok katmanlı sistemleri açıklar.",
        "Logistic": "Lojistik modele uymaktadır. Simetrik S-tipi profilleri açıklar."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics. Describes constant rate release.",
        "Birinci Derece": "fits First-Order kinetics. Concentration-dependent release.",
        "Higuchi": "fits the Higuchi model. Diffusion-based matrix release.",
        "Korsmeyer-Peppas": "fits Korsmeyer-Peppas. Mechanism defined by n exponent.",
        "Hixson-Crowell": "fits Hixson-Crowell. Erosion-based surface area reduction.",
        "Hopfenberg": "fits Hopfenberg. Geometric erosion of surface-eroding polymers.",
        "Makoid-Banakar": "fits Makoid-Banakar. Measures burst release effect.",
        "Square Root of Mass": "fits Square Root of Mass. Erosion via mass change.",
        "Peppas-Sahlin": "fits Peppas-Sahlin. Separates diffusion and relaxation.",
        "Gompertz": "fits Gompertz. Sigmoid lag-time profiles.",
        "Weibull (w/ Td)": "fits Weibull. Defines profile shape and lag time.",
        "Baker-Lonsdale": "fits Baker-Lonsdale. Spherical matrix release.",
        "Kopcha": "fits Kopcha. Decouples diffusion and erosion.",
        "Quadratic": "fits Quadratic. Short-term non-linear release.",
        "Peppas-Rincon": "fits Peppas-Rincon. Multi-layer systems.",
        "Logistic": "fits Logistic. Symmetric S-type profiles."
    }
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman", "release": "Salım", "calc": "✅ Hesaplandı", "unsuitable": "❌ Uyumsuz", 
        "best": "🏆 En Uygun", "stats": "📊 Veri İstatistiği & Profil", "graph": "🛠️ Model Uyumu Grafiği", 
        "report": "📝 Akademik Değerlendirme", "ref_comp": "🔄 Referans Karşılaştırma"
    },
    "English": {
        "time": "Time", "release": "Release", "calc": "✅ Calculated", "unsuitable": "❌ Unsuitable", 
        "best": "🏆 Best Fit", "stats": "📊 Statistics & Profile", "graph": "🛠️ Model Fit Graph", 
        "report": "📝 Academic Evaluation", "ref_comp": "🔄 Reference Comparison"
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

# --- 3. ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab v15.9", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

menu = st.sidebar.radio("Ana İşlemler:", ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi"])
st.sidebar.divider()
test_file = st.sidebar.file_uploader("Test Verisi (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi (Opsiyonel)", type=['xlsx', 'csv'])
st.sidebar.divider()
selected_lang = st.sidebar.selectbox("Dil / Language:", ["Türkçe", "English"])
L = LANG_DICT[selected_lang]

def process_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    mask = ~np.isnan(t)
    n_count = v.shape[1]
    return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": n_count}

test_data = process_data(test_file)
ref_data = process_data(ref_file)

if test_data:
    t_raw, q_raw = test_data["t"], test_data["mean"]
    
    if menu == "📈 Salım Profilleri":
        st.subheader(L['stats'])
        # Resim 1 Talebi: Mean yanına n, RSD ve VK ekleme
        rsd = (test_data["std"] / np.where(q_raw==0, 1, q_raw)) * 100
        stats_df = pd.DataFrame({
            L['time']: t_raw, 
            f"Mean (n={test_data['n']})": q_raw, 
            "SD": test_data["std"], 
            "RSD (%)": rsd,
            "VK (%)": rsd # VK ve RSD farmasötik tabloda genelde aynıdır
        })
        st.table(stats_df.style.format("{:.2f}").hide(axis="index"))
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.errorbar(t_raw, q_raw, yerr=test_data["std"], fmt='-ok', label="Test", capsize=5)
        if ref_data:
            ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", capsize=5)
        ax.set_xlabel(L['time']); ax.set_ylabel(L['release'] + " (%)"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

    elif menu == "🧮 Kinetik Model Fitting":
        st.subheader("16 Kinetik Model Analizi")
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
        
        results = []; fit_plots = {}
        # Baker-Lonsdale Root
        try:
            popt_bl, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[0.001])
            y_bl = baker_lonsdale_for_fit(t_raw, *popt_bl)
            results.append({"Model": "Baker-Lonsdale", "R²": r2_score(q_raw, y_bl), "AIC": calculate_aic(len(t_raw), np.sum((q_raw-y_bl)**2), 1), "Durum": L['calc']})
            fit_plots["Baker-Lonsdale"] = (baker_lonsdale_for_fit, popt_bl)
        except: results.append({"Model": "Baker-Lonsdale", "R²": 0, "AIC": 9999, "Durum": L['unsuitable']})

        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=10000)
                y_p = func(tf, *popt)
                results.append({"Model": name, "R²": r2_score(qf, y_p), "AIC": calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0)), "Durum": L['calc']})
                fit_plots[name] = (func, popt)
            except: results.append({"Model": name, "R²": 0, "AIC": 9999, "Durum": L['unsuitable']})

        df_res = pd.DataFrame(results)
        best_idx = df_res[df_res["Durum"] == L['calc']]["AIC"].idxmin()
        best_name = df_res.loc[best_idx, "Model"]
        st.table(df_res.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))

        # Resim 2 Talebi: Grafikleri Göster
        st.divider(); st.subheader(L['graph'])
        sel = st.multiselect("Grafik Modelleri:", list(fit_plots.keys()), default=[best_name])
        if sel:
            fig_m, ax_m = plt.subplots(figsize=(10,6)); ax_m.scatter(t_raw, q_raw, c='k', label="Deneysel")
            t_plot = np.linspace(0, t_raw.max(), 100)
            for m in sel:
                f, p = fit_plots[m]; ax_m.plot(t_plot, f(t_plot, *p), label=m)
            ax_m.legend(); ax_m.set_xlabel(L['time']); ax_m.set_ylabel(L['release']+" (%)"); st.pyplot(fig_m)

    elif menu == "🧬 IVIVC Analizi":
        st.subheader("Wagner-Nelson Absorbsiyon Tahmini")
        ke = st.number_input("Eliminasyon Sabiti (ke) [1/h]:", value=0.1500, format="%.4f")
        # Resim 3 Hatası Giderildi: trapz -> trapezoid (veya manuel integral)
        dt = np.diff(t_raw, prepend=0)
        cum_auc = np.cumsum(q_raw * dt)
        total_auc = cum_auc[-1] + (q_raw[-1] / ke if ke > 0 else 0)
        f_abs = (q_raw + ke * cum_auc) / (ke * total_auc if total_auc > 0 else 1)
        
        ivivc_df = pd.DataFrame({L['time']: t_raw, "Release (%)": q_raw, "Fraction Absorbed": f_abs})
        st.table(ivivc_df.style.format("{:.4f}").hide(axis="index"))
        fig_iv, ax_iv = plt.subplots(); ax_iv.plot(t_raw, f_abs, 'r-o'); st.pyplot(fig_iv)
else:
    st.info("Lütfen bir test verisi yükleyerek başlayın.")

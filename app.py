import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. AKADEMİK BİLGİ BANKASI VE TERMİNOLOJİ ---
# Hixson-Crowell ve diğer modellerin mekanistik tanımları
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Bu model, zamandan bağımsız sabit hızda salımı açıklar; genellikle kontrollü salım sistemleri için idealdir.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır; konvansiyonel dozaj formları için karakteristiktir.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Bu model, matris sistemlerinden difüzyon temelli salımı açıklar; zamanın karekökü ile orantılı bir salım gözlenir.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Salım mekanizması 'n' üsteli ile tanımlanır; hem difüzyon hem de polimer şişmesini bir arada açıklar.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Bu model, 'Erozyon ve Şişme Temelli' bir yaklaşım olup, ilaç parçacıklarının yüzey alanı ve çapının zamanla küçüldüğü (erozyon) durumları açıklar.",
        "Kopcha": "Kopcha modeline uymaktadır. Salım sürecinde difüzyon ve polimer erozyonunun nispi katkılarını ayrıştırarak hesaplar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Bu model, salımdaki difüzyonel katkı ile polimer zincir relaksasyonu (erozyon) katkısını matematiksel olarak birbirinden ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. İlacın çözünürlüğünün düşük olduğu ve salımın başlangıçta yavaş, sonra hızlı olduğu sigmoid profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Bu ampirik model, profilin ölçek (alfa), şekil (beta) ve gecikme süresi (Td) parametrelerini karakterize eder."
    },
    "English": {
        "Sıfır Derece": "fits Zero-Order kinetics. This model describes a constant release rate independent of time, ideal for controlled release systems.",
        "Birinci Derece": "fits First-Order kinetics. The release rate is concentration-dependent, typical for conventional dosage forms.",
        "Higuchi": "fits the Higuchi model, describing diffusion-based release from matrix systems proportional to the square root of time.",
        "Korsmeyer-Peppas": "fits the Korsmeyer-Peppas model. The mechanism is defined by the 'n' exponent, explaining both diffusion and polymer swelling.",
        "Hixson-Crowell": "fits Hixson-Crowell kinetics. This is an erosion/swelling-based model explaining cases where surface area and particle diameter decrease over time.",
        "Kopcha": "fits the Kopcha model, which separates the relative contributions of diffusion and polymer erosion in the release process.",
        "Peppas-Sahlin": "fits the Peppas-Sahlin model, mathematically separating the contributions of diffusion and polymer relaxation (erosion).",
        "Gompertz": "fits the Gompertz model, used for analyzing sigmoid profiles where dissolution is initially slow and then accelerates.",
        "Weibull (w/ Td)": "fits the Weibull model, characterizing the profile scale (alpha), shape (beta), and dissolution lag time (Td)."
    }
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman", "release": "Salım", "test": "Test", "ref": "Referans",
        "model_fit": "Model Uygunluğu", "comment": "Yorum", "calc": "✅ Hesaplanabilir",
        "unsuitable": "❌ Uyumsuz", "best": "🏆 En Uygun Model", "sd": "S. Sapma", "rsd": "RSD (%)",
        "stats": "📊 Veri İstatistiği", "graph": "🛠️ Model Uyumu Grafiği", "download": "🖼️ Grafiği İndir",
        "report_title": "📝 Akademik Değerlendirme"
    },
    "English": {
        "time": "Time", "release": "Release", "test": "Test", "ref": "Reference",
        "model_fit": "Model Suitability", "comment": "Comment", "calc": "✅ Calculable",
        "unsuitable": "❌ Unsuitable", "best": "🏆 Best Fit", "sd": "Std. Dev.", "rsd": "RSD (%)",
        "stats": "📊 Data Statistics", "graph": "🛠️ Model Fit Graph", "download": "🖼️ Download Graph",
        "report_title": "📝 Academic Evaluation"
    }
}

# --- 2. MATEMATİKSEL FONKSİYONLAR ---
def interpret_peppas_n(n, lang):
    if n <= 0.45: return "(Fickian)" if lang == "English" else "(Fickian Difüzyon)"
    elif 0.45 < n < 0.89: return "(Anomalous)" if lang == "English" else "(Anomalous Transport)"
    return "(Super Case II)"

# Model denklemleri
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull_complex(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))

def baker_lonsdale_for_fit(t_data, k):
    def bl_root(q_guess, t_single, k_fit):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_fit * t_single
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t_data])

def calculate_aic(n, rss, p_count):
    if n <= p_count or rss <= 0: return 9999
    return n * np.log(rss/n) + 2 * p_count

# --- 3. ARAYÜZ (SIDEBAR SIRALAMASI DÜZELTİLDİ) ---
st.set_page_config(page_title="PharmTech Lab v15.6", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

# A) ANALİZ MENÜSÜ (EN ÜSTTE)
menu_options = ["📈 1. Salım Profilleri", "🧮 2. Tüm Modelleri Test Et"]
menu = st.sidebar.radio("Analiz Adımları / Steps:", menu_options)

# B) VERİ GİRİŞİ (ORTADA)
st.sidebar.divider()
st.sidebar.subheader("📂 Veri Girişi / Data Entry")
test_file = st.sidebar.file_uploader("Test Verisi (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans (Opsiyonel)", type=['xlsx', 'csv'])

# C) DİL SEÇİMİ (EN ALTTA)
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

test = load_data(test_file)

# --- 4. ANA ÇALIŞMA ALANI ---
if test:
    t_raw, q_raw = test["t"], test["mean"]

    if "1." in menu:
        st.subheader(f"📍 {L['release']} Profili & İstatistik")
        
        # RSD ve İstatistik Tablosu
        rsd = (test["std"] / np.where(q_raw==0, 1, q_raw)) * 100
        stats_df = pd.DataFrame({L['time']: t_raw, f"Mean (%)": q_raw, L['sd']: test["std"], L['rsd']: rsd})
        st.write(f"### {L['stats']}")
        st.table(stats_df.style.format("{:.2f}").hide(axis="index"))

        # Grafik
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_raw, q_raw, yerr=test["std"], fmt='-ok', label=L['test'], capsize=5)
        ax.set_xlabel(f"{L['time']} (min)"); ax.set_ylabel(f"{L['release']} (%)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    elif "2." in menu:
        st.subheader(f"🔍 {L['model_fit']}")
        fit_mask = (t_raw > 0) & (q_raw > 0)
        tf, qf = t_raw[fit_mask], q_raw[fit_mask]
        
        model_defs = [
            ("Sıfır Derece", zero_order, [0.1], [0], [100]),
            ("Birinci Derece", first_order, [0.01], [0], [10]),
            ("Higuchi", higuchi, [1.0], [0], [500]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.1], [500, 2.0]),
            ("Hixson-Crowell", hixson, [0.001], [0], [1]),
            ("Kopcha", kopcha, [1.0, 0.1], [0, -10], [500, 100]),
            ("Peppas-Sahlin", peppas_sahlin, [0.1, 0.1, 0.5], [0, 0, 0.1], [100, 100, 1.5]),
            ("Gompertz", gompertz, [100, 0.1, 10], [50, 0, 0], [110, 5, 500]),
            ("Weibull (w/ Td)", weibull_complex, [50, 1.0, 1.0], [1, 0.1, 0], [10000, 10.0, 100])
        ]
        
        results = []; fit_plots = {}
        
        # Özel Fit: Baker-Lonsdale
        try:
            popt_bl, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[0.001])
            y_bl = baker_lonsdale_for_fit(t_raw, *popt_bl)
            results.append({"Model": "Baker-Lonsdale", "R²": r2_score(q_raw, y_bl), "AIC": calculate_aic(len(t_raw), np.sum((q_raw-y_bl)**2), 1), L['model_fit']: L['calc'], L['comment']: f"k: {popt_bl[0]:.5f}"})
            fit_plots["Baker-Lonsdale"] = (baker_lonsdale_for_fit, popt_bl)
        except: pass

        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=10000)
                y_p = func(tf, *popt)
                res = {"Model": name, "R²": r2_score(qf, y_p), "AIC": calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0)), L['model_fit']: L['calc']}
                
                # Yorum ve Mekanizma belirleme
                if name == "Korsmeyer-Peppas":
                    res[L['comment']] = f"n: {popt[1]:.3f} {interpret_peppas_n(popt[1], selected_lang)}"
                elif name == "Hixson-Crowell":
                    res[L['comment']] = f"k: {popt[0]:.5f}"
                else:
                    res[L['comment']] = "-"
                results.append(res); fit_plots[name] = (func, popt)
            except:
                results.append({"Model": name, "R²": 0, "AIC": 9999, L['model_fit']: L['unsuitable'], L['comment']: "-"})

        df_res = pd.DataFrame(results)
        # En uygun modeli seç
        valid_results = df_res[df_res[L['model_fit']] == L['calc']]
        best_idx = valid_results["AIC"].idxmin()
        best_model_name = df_res.loc[best_idx, "Model"]
        df_res.at[best_idx, L['comment']] += f" {L['best']}"
        
        st.table(df_res.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))

        # --- AKADEMİK YORUM PARAGRAFI ---
        st.divider()
        st.subheader(L['report_title'])
        knowledge = MODEL_KNOWLEDGE[selected_lang]
        
        if selected_lang == "Türkçe":
            desc = f"**📊 Analiz Özeti:** Test preparatımız **{best_model_name}** {knowledge.get(best_model_name, '')}"
            stats_text = f"Model uyumu **R²: {df_res.loc[best_idx, 'R²']:.4f}** ve **AIC: {df_res.loc[best_idx, 'AIC']:.2f}** değerleri ile istatistiksel olarak doğrulanmıştır."
        else:
            desc = f"**📊 Analysis Summary:** The test preparation **{knowledge.get(best_model_name, '')}**"
            stats_text = f"Model suitability is statistically verified with **R²: {df_res.loc[best_idx, 'R²']:.4f}** and **AIC: {df_res.loc[best_idx, 'AIC']:.2f}**."
        
        st.info(f"{desc}\n\n{stats_text}")

        # Grafikler
        st.write(f"### {L['graph']}")
        selected = st.multiselect("Grafikte Göster:", list(fit_plots.keys()), default=[best_model_name])
        if selected:
            fig_m, ax_m = plt.subplots(figsize=(10, 5))
            ax_m.scatter(t_raw, q_raw, c='k', label="Deneysel Veri")
            t_p = np.linspace(t_raw.min(), t_raw.max(), 100)
            for m in selected:
                f, p = fit_plots[m]; ax_m.plot(t_p, f(t_p, *p), label=m)
            ax_m.legend(); ax_m.set_xlabel("Zaman (dk)"); ax_m.set_ylabel("Salım (%)")
            st.pyplot(fig_m)

else:
    st.warning("Lütfen sidebar üzerinden bir veri dosyası yükleyiniz. / Please upload a data file.")

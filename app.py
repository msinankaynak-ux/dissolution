import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score

# --- 1. MODEL BİLGİ BANKASI & AKADEMİK YORUMLAR ---
MODEL_KNOWLEDGE = {
    "Türkçe": {
        "Sıfır Derece": "Sıfır derece kinetiğine uymaktadır. Zamandan bağımsız sabit hızda salımı açıklar.",
        "Birinci Derece": "Birinci derece kinetiğine uymaktadır. Salım hızı, kalan ilaç konsantrasyonuna bağlıdır.",
        "Higuchi": "Higuchi kinetiğine uymaktadır. Matris sistemlerinden difüzyon temelli salımı açıklar.",
        "Korsmeyer-Peppas": "Korsmeyer-Peppas modeline uymaktadır. Mekanizma 'n' üsteli ile tanımlanır.",
        "Hixson-Crowell": "Hixson-Crowell kinetiğine uymaktadır. Yüzey alanı ve çapın zamanla küçüldüğü (erozyon) durumları açıklar.",
        "Hopfenberg": "Hopfenberg modeline uymaktadır. Yüzeyden aşınan (surface-eroding) polimerlerin geometrik (levha, silindir, küre) erozyonunu açıklar.",
        "Makoid-Banakar": "Makoid-Banakar modeline uymaktadır. Hem difüzyon hem de birinci dereceyi kapsar; başlangıçtaki 'burst release' (ani salım) etkisini ölçer.",
        "Square Root of Mass": "Kütle karekök modeline uymaktadır. Hixson-Crowell'e benzer ancak erozyonu kütle değişimi üzerinden hesaplar.",
        "Peppas-Sahlin": "Peppas-Sahlin modeline uymaktadır. Difüzyonel ve polimer relaksasyonu (erozyon) katkısını birbirinden ayırır.",
        "Gompertz": "Gompertz modeline uymaktadır. Gecikmeli başlayan sigmoid (S-tipi) profilleri açıklar.",
        "Weibull (w/ Td)": "Weibull modeline uymaktadır. Profilin ölçek, şekil ve gecikme süresini karakterize eder.",
        "Baker-Lonsdale": "Baker-Lonsdale modeline uymaktadır. Küresel matrislerden salımı açıklar.",
        "Kopcha": "Kopcha modeline uymaktadır. Difüzyon ve erozyon oranlarını ayrıştırır.",
        "Quadratic": "Quadratic modeline uymaktadır. Çok kısa süreli ve doğrusal olmayan salımları açıklar.",
        "Peppas-Rincon": "Peppas-Rincon modeline uymaktadır. Çok katmanlı veya karmaşık geometriler için geliştirilmiş versiyondur.",
        "Logistic": "Lojistik modele uymaktadır. Simetrik sigmoid (S-tipi) salımları açıklar."
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
        "Weibull (w/ Td)": "fits the Weibull model, characterizing profile scale, shape, and lag time.",
        "Baker-Lonsdale": "fits the Baker-Lonsdale model, explaining release from spherical matrices.",
        "Kopcha": "fits the Kopcha model, decoupling diffusion and erosion rates.",
        "Quadratic": "fits the Quadratic model, explaining short-term non-linear release.",
        "Peppas-Rincon": "fits the Peppas-Rincon model, developed for multi-layer or complex geometries.",
        "Logistic": "fits the Logistic model, explaining symmetric sigmoid (S-type) profiles."
    }
}

UNSUITABLE_DESC = {
    "Türkçe": "⚠️ Veri yapısı bu modelin matematiksel varsayımlarına (örneğin sigmoid yapı, erozyon hızı veya gecikme süresi) istatistiksel olarak uymuyor.",
    "English": "⚠️ Data structure does not statistically fit the model's mathematical assumptions (e.g., sigmoid shape, erosion rate, or lag-time)."
}

LANG_DICT = {
    "Türkçe": {
        "time": "Zaman", "release": "Salım", "calc": "✅ Hesaplandı", "unsuitable": "❌ Uyumsuz", 
        "best": "🏆 En Uygun Model", "stats": "📊 Veri İstatistiği & Profil", "graph": "🛠️ Model Uyumu Grafiği", 
        "report": "📝 Akademik Değerlendirme", "model_title": "16 Kinetik Model Analizi"
    },
    "English": {
        "time": "Time", "release": "Release", "calc": "✅ Calculated", "unsuitable": "❌ Unsuitable", 
        "best": "🏆 Best Fit Model", "stats": "📊 Statistics & Profile", "graph": "🛠️ Model Fit Graph", 
        "report": "📝 Academic Evaluation", "model_title": "16 Kinetic Model Analysis"
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
    def calculate_f1_f2(ref_mean, test_mean):
    R = np.array(ref_mean)
    T = np.array(test_mean)
    n = len(R)
    # f1: Farklılık Faktörü
    f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
    # f2: Benzerlik Faktörü
    sum_sq_diff = np.sum((R - T)**2)
    f2 = 50 * np.log10((1 + (1/n) * sum_sq_diff)**-0.5 * 100)
    return f1, f2

# --- 3. ARAYÜZ VE VERİ İŞLEME ---
st.set_page_config(page_title="PharmTech Lab v16.0", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

menu = st.sidebar.radio("Ana İşlemler:", ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"])
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
    return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask], "n": v.shape[1]}

test_data = process_data(test_file)
ref_data = process_data(ref_file)

if test_data:
    t_raw, q_raw = test_data["t"], test_data["mean"]
    
    if menu == "📈 Salım Profilleri":
        st.subheader(L['stats'])
        rsd = (test_data["std"] / np.where(q_raw==0, 1, q_raw)) * 100
        stats_df = pd.DataFrame({
            L['time']: t_raw, 
            f"Mean (n={test_data['n']})": q_raw, 
            "SD": test_data["std"], 
            "RSD (%)": rsd,
            "VK (%)": rsd
        })
        st.table(stats_df.style.format("{:.2f}").hide(axis="index"))
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.errorbar(t_raw, q_raw, yerr=test_data["std"], fmt='-ok', label="Test", capsize=5)
        if ref_data:
            ax.errorbar(ref_data["t"], ref_data["mean"], yerr=ref_data["std"], fmt='--sr', label="Referans", capsize=5)
        ax.set_xlabel(L['time']); ax.set_ylabel(L['release'] + " (%)"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)

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
        
        results = []; fit_plots = {}
        # Baker-Lonsdale
        try:
            popt_bl, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[0.001])
            y_bl = baker_lonsdale_for_fit(t_raw, *popt_bl)
            results.append({"Model": "Baker-Lonsdale", "R²": r2_score(q_raw, y_bl), "AIC": calculate_aic(len(t_raw), np.sum((q_raw-y_bl)**2), 1), "Durum": L['calc']})
            fit_plots["Baker-Lonsdale"] = (baker_lonsdale_for_fit, popt_bl)
        except: results.append({"Model": "Baker-Lonsdale", "R²": 0, "AIC": 9999, "Durum": L['unsuitable']})

        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=15000)
                y_p = func(tf, *popt)
                results.append({"Model": name, "R²": r2_score(qf, y_p), "AIC": calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0)), "Durum": L['calc']})
                fit_plots[name] = (func, popt)
            except: results.append({"Model": name, "R²": 0, "AIC": 9999, "Durum": L['unsuitable']})

        df_res = pd.DataFrame(results)
        best_idx = df_res[df_res["Durum"] == L['calc']]["AIC"].idxmin()
        best_name = df_res.loc[best_idx, "Model"]
        st.table(df_res.style.format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))

        # Akademik Değerlendirme & Uygunsuzluk Notları
        st.divider(); st.subheader(L['report'])
        st.info(f"🏆 **{best_name}**: {MODEL_KNOWLEDGE[selected_lang].get(best_name, '')}")
        
        with st.expander("Uyumsuz Modeller Hakkında Notlar / Notes on Unsuitable Models"):
            st.write(UNSUITABLE_DESC[selected_lang])

        # Model Grafiği
        st.subheader(L['graph'])
        sel = st.multiselect("Grafik Modelleri:", list(fit_plots.keys()), default=[best_name])
        if sel:
            fig_m, ax_m = plt.subplots(figsize=(10,6)); ax_m.scatter(t_raw, q_raw, c='k', label="Data")
            t_plot = np.linspace(0, t_raw.max(), 100)
            for m in sel:
                f, p = fit_plots[m]; ax_m.plot(t_plot, f(t_plot, *p), label=m)
            ax_m.legend(); ax_m.set_xlabel(L['time']); ax_m.set_ylabel(L['release']+" (%)"); st.pyplot(fig_m)

    elif menu == "🧬 IVIVC Analizi":
        st.subheader("Wagner-Nelson Absorbsiyon Tahmini")
        ke = st.number_input("Eliminasyon Sabiti (ke) [1/h]:", value=0.1500, format="%.4f")
        dt = np.diff(t_raw, prepend=0)
        cum_auc = np.cumsum(q_raw * dt)
        total_auc = cum_auc[-1] + (q_raw[-1] / ke if ke > 0 else 0)
        f_abs = (q_raw + ke * cum_auc) / (ke * total_auc if total_auc > 0 else 1)
        ivivc_df = pd.DataFrame({L['time']: t_raw, "Release (%)": q_raw, "Fraction Absorbed": f_abs})
        st.table(ivivc_df.style.format("{:.4f}").hide(axis="index"))
        fig_iv, ax_iv = plt.subplots(); ax_iv.plot(t_raw, f_abs, 'r-o'); st.pyplot(fig_iv)
    elif menu == "📊 f1 & f2 Benzerlik Analizi":
        st.subheader("f1 & f2 Faktörleri (Similarity & Difference Factors)")
        
        if ref_data is not None:
            # Zaman noktalarının eşleştiğinden emin olalım
            if len(test_data["t"]) != len(ref_data["t"]):
                st.error("⚠️ Hata: Test ve Referans verilerinin satır sayısı aynı olmalıdır!")
            else:
                f1, f2 = calculate_f1_f2(ref_data["mean"], test_data["mean"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="f1 (Difference Factor)", value=f"{f1:.2f}")
                    if f1 <= 15:
                        st.success("✅ f1 Uygun (0-15)")
                    else:
                        st.warning("❌ f1 Uygun Değil (>15)")
                
                with col2:
                    st.metric(label="f2 (Similarity Factor)", value=f"{f2:.2f}")
                    if f2 >= 50:
                        st.success("✅ f2 Benzer (50-100)")
                    else:
                        st.error("❌ f2 Benzer Değil (<50)")
                
                st.divider()
                st.write("### 📝 Akademik Analiz Notu")
                if f2 >= 50:
                    st.info(f"Hesaplanan f2 değeri ({f2:.2f}), iki profilin istatistiksel olarak benzer olduğunu göstermektedir. Bu durum formülasyonun referans ürünle eşdeğer salım karakteristiğine sahip olduğu şeklinde yorumlanabilir.")
                else:
                    st.error("Düşük f2 değeri, test ve referans profillerinin anlamlı derecede farklı olduğunu gösterir.")

                # Görsel Karşılaştırma
                fig_f12, ax_f12 = plt.subplots(figsize=(10,5))
                ax_f12.plot(ref_data["t"], ref_data["mean"], 's--b', label="Referans")
                ax_f12.plot(test_data["t"], test_data["mean"], 'o-r', label="Test")
                ax_f12.set_title("Test vs Referans Salım Kıyaslaması")
                ax_f12.set_xlabel("Zaman"); ax_f12.set_ylabel("Salım (%)")
                ax_f12.legend(); ax_f12.grid(True, alpha=0.2)
                st.pyplot(fig_f12)
        else:
            st.warning("Bu analizi yapabilmek için sol menüden 'Referans Verisi' yüklemelisiniz.")
else:
    st.info("Lütfen bir test verisi yükleyerek başlayın.")

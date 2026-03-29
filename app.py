import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score

# --- 1. MATEMATİKSEL MODELLER VE YORUMLAMA ---

def interpret_peppas_n(n):
    if n <= 0.45: return "(Fickian Difüzyon)"
    elif 0.45 < n < 0.89: return "(Anomalous Transport)"
    return "(Super Case II)"

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

# --- 2. ARAYÜZ YAPILANDIRMASI (SIDEBAR SABİT) ---

st.set_page_config(page_title="PharmTech Lab v15.3", layout="wide")
st.sidebar.title("🔬 Pro Lab v15.3")

st.sidebar.subheader("📂 Veri Girişi")
test_file = st.sidebar.file_uploader("Test Verisi (Zorunlu)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi (Opsiyonel)", type=['xlsx', 'csv'])

st.sidebar.divider()
menu = st.sidebar.radio("Analiz Adımları:", ["📈 1. Salım Profilleri", "🧮 2. Tüm Modelleri Test Et"])

def load_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except: return None

test = load_data(test_file)
ref = load_data(ref_file)

# --- 3. ANA ANALİZ DÖNGÜSÜ ---

if test:
    t_raw, q_raw = test["t"], test["mean"]

    if menu == "📈 1. Salım Profilleri":
        st.subheader("📍 Kümülatif Salım Profili")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(t_raw, q_raw, yerr=test["std"], fmt='-ok', label="Test", capsize=5, linewidth=2)
        if ref:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.7)
        ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif menu == "🧮 2. Tüm Modelleri Test Et":
        st.subheader("🔍 Kinetik Model Karşılaştırma Tablosu")
        
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
        
        results = []
        fit_data_for_plot = {}

        # Baker-Lonsdale Özel Fit
        try:
            popt_bl, _ = curve_fit(baker_lonsdale_for_fit, t_raw, q_raw, p0=[0.001], maxfev=2000)
            y_bl = baker_lonsdale_for_fit(t_raw, *popt_bl)
            r2_bl = r2_score(q_raw, y_bl)
            aic_bl = calculate_aic(len(t_raw), np.sum((q_raw-y_bl)**2), 1)
            results.append({"Model": "Baker-Lonsdale", "R²": r2_bl, "AIC": aic_bl, "Model Uygunluğu": "✅ Hesaplanabilir", "Yorum": f"k: {popt_bl[0]:.5f}"})
            fit_data_for_plot["Baker-Lonsdale"] = (baker_lonsdale_for_fit, popt_bl)
        except: pass

        # Diğer Tüm Modeller
        for name, func, p0, low, up in model_defs:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=15000)
                y_p = func(tf, *popt)
                r2 = r2_score(qf, y_p)
                aic = calculate_aic(len(tf), np.sum((qf-y_p)**2), len(p0))
                
                comment = "-"
                if name == "Korsmeyer-Peppas": comment = f"n: {popt[1]:.3f} {interpret_peppas_n(popt[1])}"
                
                results.append({"Model": name, "R²": r2, "AIC": aic, "Model Uygunluğu": "✅ Hesaplanabilir", "Yorum": comment})
                fit_data_for_plot[name] = (func, popt)
            except:
                results.append({"Model": name, "R²": 0, "AIC": 9999, "Model Uygunluğu": "❌ Uyumsuz", "Yorum": "-"})

        df_res = pd.DataFrame(results)
        valid_models = df_res[df_res["Model Uygunluğu"] == "✅ Hesaplanabilir"]
        best_idx = valid_models["AIC"].idxmin() if not valid_models.empty else None
        
        if best_idx is not None:
            df_res.at[best_idx, "Yorum"] += " 🏆 En Uygun Model"

        # Tablo Stil Uygulama (İndeks Gizli & Bold)
        st.table(df_res.style.apply(lambda x: ['font-weight: bold' if x.name == best_idx else '' for i in x], axis=1)
                 .format({"R²": "{:.4f}", "AIC": "{:.2f}"}).hide(axis="index"))

        # İnteraktif Grafik
        st.divider()
        st.write("### 🛠️ Model Uyumu Grafiği")
        all_m = list(fit_data_for_plot.keys())
        selected = st.multiselect("Grafikte gösterilecek modeller:", all_m, default=all_m)
        
        if selected:
            fig_f, ax_f = plt.subplots(figsize=(10, 5))
            ax_f.scatter(t_raw, q_raw, color='black', label="Deneysel", zorder=5)
            t_plot = np.linspace(t_raw.min(), t_raw.max(), 100)
            for m in selected:
                f, p = fit_data_for_plot[m]
                ax_f.plot(t_plot, f(t_plot, *p), label=m, alpha=0.8)
            ax_f.set_xlabel("Zaman (dk)"); ax_f.set_ylabel("Salım (%)"); ax_fit = ax_f.legend(); ax_f.grid(alpha=0.1)
            st.pyplot(fig_f)
else:
    st.info("Hocam hoş geldiniz. Lütfen sol panelden 'Test Verisi' yükleyerek analize başlayın.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- HESAPLAMA MOTORLARI ---
def safe_calc_f2(ref_mean, test_mean):
    if ref_mean is None or test_mean is None: return None
    n = len(ref_mean)
    sum_sq = np.sum((ref_mean - test_mean)**2)
    return 50 * np.log10(100 / np.sqrt(1 + sum_sq/n))

def interpret_n(n):
    if n <= 0.45: return "Fickian Diffusion"
    elif 0.45 < n < 0.89: return "Anomalous Transport"
    return "Case II / Super Case II"

# --- MODELLER (Daha Güçlü Sınırlarla) ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-np.clip(k * t, 0, 10)))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.01, 2.0))

# --- ARAYÜZ ---
st.set_page_config(page_title="PharmTech Lab Pro v12.1", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

menu = st.sidebar.radio("Analiz Menüsü:", ["📈 Dissolüsyon Profilleri", "🧮 Model-Bağımlı Analiz", "📊 Model-Bağımsız Analiz"])

st.sidebar.divider()
test_file = st.sidebar.file_uploader("Test Verisi", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_full_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        mask = ~np.isnan(t)
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except: return None

test = load_full_data(test_file)
ref = load_full_data(ref_file)

if test:
    t_raw, q_raw = test["t"], test["mean"]

    if menu == "📈 Dissolüsyon Profilleri":
        st.subheader("📍 Kümülatif Salım Profili")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_raw, q_raw, yerr=test["std"], fmt='-ok', label="Test", capsize=4)
        if ref is not None:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.6)
        ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); ax.grid(True, alpha=0.1)
        st.pyplot(fig)

    elif menu == "🧮 Model-Bağımlı Analiz":
        st.subheader("🔍 Kinetik Modelleme (Yüksek Hassasiyet)")
        fig_m, ax_m = plt.subplots(figsize=(10, 5))
        ax_m.scatter(t_raw, q_raw, color='black', label="Deneysel", zorder=5)
        
        # Fit için 0'ı ve 100'ü geçen yerleri filtreleyelim ama sadece fit için
        fit_mask = (t_raw > 0) & (q_raw > 0)
        tf, qf = t_raw[fit_mask], q_raw[fit_mask]
        
        results = []
        # Model tanımları: (İsim, Fonksiyon, [p0], [Alt Sınır], [Üst Sınır])
        models = [
            ("Sıfır Derece", zero_order, [0.1], [0], [10]),
            ("Birinci Derece", first_order, [0.01], [0], [1]),
            ("Higuchi", higuchi, [1.0], [0], [100]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5], [0, 0.01], [100, 2.0])
        ]

        for name, func, p0, low, up in models:
            try:
                # maxfev artırıldı ve bounds eklendi
                popt, _ = curve_fit(func, tf, qf, p0=p0, bounds=(low, up), maxfev=20000)
                y_fit = func(t_raw, *popt)
                r2 = r2_score(q_raw, y_fit)
                note = interpret_n(popt[1]) if name == "Korsmeyer-Peppas" else "-"
                results.append({"Model": name, "R²": f"{r2:.4f}", "K": f"{popt[0]:.4f}", "n": f"{popt[1]:.3f}" if len(popt)>1 else "-", "Yorum": note})
                ax_m.plot(t_raw, y_fit, label=f"{name}")
            except:
                results.append({"Model": name, "R²": "Uyumsuz", "K": "-", "n": "-", "Yorum": "Yakınsama Hatası"})

        ax_m.legend(); st.pyplot(fig_m)
        st.table(pd.DataFrame(results))

    elif menu == "📊 Model-Bağımsız Analiz":
        st.subheader("📏 Parametrik Özet")
        de = (np.trapz(q_raw, t_raw) / (np.max(t_raw) * 100)) * 100
        st.metric("Dissolution Efficiency (DE)", f"%{de:.2f}")
        
        if ref is not None and len(ref["t"]) == len(t_raw):
            f2 = safe_calc_f2(ref["mean"], q_raw)
            st.metric("f2 Benzerlik Faktörü", f"{f2:.2f}")
        elif ref is None:
            st.info("ℹ️ f2 için Referans verisi yükleyin.")
else:
    st.info("👋 Başlamak için veri yükleyin.")

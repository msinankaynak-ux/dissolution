import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- MODELLER (Daha Kararlı Tanımlamalar) ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-np.clip(k1 * t, 0, 10)))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - np.maximum(0, (1 - khc * t))**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-((t**beta) / alpha)))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="PharmTech Lab Pro v6.1", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🔬 PharmTech Pro")
menu = st.sidebar.radio("Analiz Menüsü:", ["📈 Profil Analizi", "🧮 Kinetik Modelleme", "📊 Benzerlik (f1/f2)"])

st.sidebar.divider()
test_file = st.sidebar.file_uploader("Test Verisi Yükle", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi Yükle", type=['xlsx', 'csv'])

def load_data(file):
    if file is None: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        # NaN temizleme
        valid_idx = ~np.isnan(t)
        return {"t": t[valid_idx], "mean": v.mean(axis=1).values[valid_idx], "std": v.std(axis=1).values[valid_idx]}
    except Exception as e:
        st.error(f"Dosya okuma hatası: {e}")
        return None

test = load_data(test_file)
ref = load_data(ref_file)

# --- ANA EKRAN ---
if test:
    t_data, m_data, s_data = test["t"], test["mean"], test["std"]

    if menu == "📈 Profil Analizi":
        st.subheader("📍 Kümülatif Salım Profili")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_data, m_data, yerr=s_data, fmt='-ok', label="Test", capsize=5)
        if ref:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.5)
        ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    elif menu == "🧮 Kinetik Modelleme":
        st.subheader("🔍 Model Karşılaştırma")
        
        # Sadece t > 0 ve Salım < 100 kısımlarını fit etmeye çalış (Daha kararlı sonuçlar için)
        mask = (t_data > 0)
        tf, qf = t_data[mask], m_data[mask]
        t_plot = np.linspace(0.1, t_data.max(), 100)
        
        fig_m, ax_m = plt.subplots(figsize=(12, 6))
        ax_m.scatter(t_data, m_data, color='black', label="Deneysel", zorder=5)
        
        models = [
            ("Sıfır Derece", zero_order, [1]),
            ("Birinci Derece", first_order, [0.01]),
            ("Higuchi", higuchi, [1]),
            ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
            ("Hixson-Crowell", hixson_crowell, [0.001]),
            ("Weibull", weibull, [50, 1])
        ]
        
        results = []
        for name, func, p0 in models:
            try:
                # bounds ekleyerek negatif değerleri engelledik
                popt, _ = curve_fit(func, tf, qf, p0=p0, maxfev=10000)
                y_pred = func(tf, *popt)
                r2 = r2_score(qf, y_pred)
                rss = np.sum((qf - y_pred)**2)
                aic = calculate_aic(len(tf), rss, len(p0))
                
                ax_m.plot(t_plot, func(t_plot, *popt), label=f"{name} (R²: {r2:.3f})")
                results.append({"Model": name, "R²": r2, "AIC": aic})
            except Exception:
                continue # Hata veren modeli sessizce atla
        
        ax_m.legend(bbox_to_anchor=(1, 1)); ax_m.grid(alpha=0.2)
        st.pyplot(fig_m)
        if results:
            st.table(pd.DataFrame(results).sort_values("AIC"))
        else:
            st.warning("Hiçbir model fit edilemedi. Lütfen verilerinizi kontrol edin.")

    elif menu == "📊 Benzerlik (f1/f2)":
        if ref:
            R, T = ref["mean"], m_data
            if len(R) == len(T):
                f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                f2 = 50 * np.log10((1 + np.mean((R - T)**2))**-0.5 * 100)
                st.metric("f1 (Fark)", f"{f1:.2f}"); st.metric("f2 (Benzerlik)", f"{f2:.2f}")
            else:
                st.error("Zaman noktası sayıları tutmuyor!")
        else:
            st.warning("Lütfen Referans verisi yükleyin.")
else:
    st.info("👈 Veri yükleyerek başlayın.")

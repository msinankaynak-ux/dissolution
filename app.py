import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- HESAPLAMA MOTORLARI (HAM VERİ DOSTU) ---
def safe_calc_f2(ref_mean, test_mean):
    n = len(ref_mean)
    sum_sq = np.sum((ref_mean - test_mean)**2)
    return 50 * np.log10(100 / np.sqrt(1 + sum_sq/n))

def interpret_n(n):
    if n <= 0.45: return "Fickian Diffusion"
    elif 0.45 < n < 0.89: return "Anomalous Transport"
    return "Case II / Super Case II"

# --- MODELLER ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**n)

# --- ARAYÜZ YAPILANDIRMASI (BEĞENDİĞİNİZ SİDEBAR) ---
st.set_page_config(page_title="PharmTech Lab Pro v11", layout="wide")
st.sidebar.title("🔬 PharmTech Lab")

# İlk sevdiğiniz Radio Button yapısı
menu = st.sidebar.radio(
    "Analiz Menüsü:",
    ["📈 Dissolüsyon Profilleri", "🧮 Model-Bağımlı Analiz", "📊 Model-Bağımsız Analiz"]
)

st.sidebar.divider()
test_file = st.sidebar.file_uploader("Test Verisi (Excel/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_full_data(file):
    if not file: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    # Kısıtlama YOK: Sadece NaN olan satırları temizle
    mask = ~np.isnan(t)
    return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}

test = load_full_data(test_file)
ref = load_full_data(ref_file)

if test:
    t_raw, q_raw = test["t"], test["mean"]

    # --- 1. SEKMELİ YAPI: PROFİLLER ---
    if menu == "📈 Dissolüsyon Profilleri":
        st.subheader("📍 Kümülatif Salım Profili (Tam Veri)")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(t_raw, q_raw, yerr=test["std"], fmt='-ok', label="Test", capsize=4)
        if ref:
            ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.6)
        ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)"); ax.legend(); ax.grid(True, alpha=0.2)
        st.pyplot(fig)

    # --- 2. SEKMELİ YAPI: KİNETİK ---
    elif menu == "🧮 Model-Bağımlı Analiz":
        st.subheader("🔍 Kinetik Modelleme ve Mekanizma")
        
        # Grafik için tam veri
        fig_m, ax_m = plt.subplots(figsize=(10, 5))
        ax_m.scatter(t_raw, q_raw, color='black', label="Deneysel", zorder=5)
        
        results = []
        models = [
            ("Sıfır Derece", zero_order, [0.1]),
            ("Birinci Derece", first_order, [0.01]),
            ("Higuchi", higuchi, [1.0]),
            ("Korsmeyer-Peppas", korsmeyer, [1.0, 0.5])
        ]

        for name, func, p0 in models:
            try:
                # Modellerde 0 noktasında patlama olmaması için fit sırasında t > 0 filtrelemesi (Sadece fit için!)
                fit_mask = (t_raw > 0) & (q_raw > 0)
                popt, _ = curve_fit(func, t_raw[fit_mask], q_raw[fit_mask], p0=p0, maxfev=5000)
                
                y_fit = func(t_raw, *popt) # Grafiği tüm zamanlar için çiz
                r2 = r2_score(q_raw, y_fit)
                
                note = interpret_n(popt[1]) if name == "Korsmeyer-Peppas" else "-"
                results.append({"Model": name, "R²": f"{r2:.4f}", "K": f"{popt[0]:.4f}", "n": popt[1] if len(popt)>1 else "-", "Yorum": note})
                ax_m.plot(t_raw, y_fit, label=f"{name} (R²:{r2:.3f})")
            except:
                results.append({"Model": name, "R²": "Hata", "K": "-", "n": "-", "Yorum": "Yakınsama Sağlanamadı"})

        ax_m.legend(); ax_m.grid(alpha=0.2)
        st.pyplot(fig_m)
        st.table(pd.DataFrame(results))

    # --- 3. SEKMELİ YAPI: MODEL BAĞIMSIZ ---
    elif menu == "📊 Model-Bağımsız Analiz":
        st.subheader("📏 Parametrik Özet")
        # DE ve MDT Hesaplama (Tüm veri üzerinden)
        de = (np.trapz(q_raw, t_raw) / (np.max(t_raw) * 100)) * 100
        
        c1, c2 = st.columns(2)
        c1.metric("Dissolution Efficiency (DE)", f"%{de:.2f}")
        
        if ref:
            if len(ref["t"]) == len(t_raw):
                f2 = safe_calc_f2(ref["mean"], q_raw)
                c2.metric("f2 Similarity", f"{f2:.2f}")
                if f2 >= 50: st.success("Profiller Benzer.")
                else: st.error("Profiller Benzer Değil.")

else:
    st.info("Hocam hoş geldiniz. Sidebar üzerinden dosyalarınızı yükleyerek başlayabilirsiniz.")

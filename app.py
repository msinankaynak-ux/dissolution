import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import simps
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score
import io

# --- 1. MODEL-BAĞIMLI MOTOR (MATEMATİKSEL MODELLER) ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))
def hopfenberg(t, khp, n_geom): return 100 * (1 - (1 - khp * t)**n_geom)

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

# --- 2. MODEL-BAĞIMSIZ MOTOR (İSTATİSTİKSEL PARAMETRELER) ---
def calculate_mdt(time, means):
    # Mean Dissolution Time (MDT) Hesabı
    delta_t = np.diff(time, prepend=0)
    mid_q = np.diff(means, prepend=0)
    mid_t = time - (delta_t / 2)
    return np.sum(mid_t * mid_q) / np.sum(mid_q) if np.sum(mid_q) > 0 else 0

def calculate_de(time, means):
    # Dissolution Efficiency (DE) - Trapezoidal Rule
    auc = np.trapz(means, time)
    total_area = time.max() * 100
    return (auc / total_area) * 100

st.set_page_config(page_title="PharmTech Lab Pro v2", layout="wide")
st.title("🔬 Gelişmiş Dissolüsyon Analitik Platformu")

st.sidebar.header("📁 Analiz Kategorisi")
category = st.sidebar.selectbox("Yöntem Seçiniz", ["Model-Bağımlı Analiz (Kinetik)", "Model-Bağımsız Analiz (f1/f2/MDT)"])

def process_data(file):
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    mean_v = values.mean(axis=1).values
    std_v = values.std(axis=1).values
    return time, mean_v, std_v

# --- VERİ YÜKLEME ---
up_file = st.file_uploader("Veri Setini Yükleyin (Zaman ve % Salım Sütunları)", type=['xlsx', 'csv'])

if up_file:
    time, mean_q, std_q = process_data(up_file)
    
    if category == "Model-Bağımlı Analiz (Kinetik)":
        st.subheader("📊 Kinetik Modelleme ve Mekanizma Tayini")
        col1, col2 = st.columns([3, 2])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(time, mean_q, yerr=std_q, fmt='o', label="Deneysel Veri", color='black', alpha=0.6)
            
            t_fit = time[time > 0]
            q_fit = mean_q[time > 0]
            t_plot = np.linspace(0, time.max(), 100)
            kin_results = []
            
            # Literatürdeki Modellerin Uygulanması
            models = [
                ("Zero-Order", zero_order, [1]),
                ("First-Order", first_order, [0.1]),
                ("Higuchi", higuchi, [1]),
                ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
                ("Hixson-Crowell", hixson_crowell, [0.01]),
                ("Weibull", weibull, [100, 1])
            ]
            
            for name, func, p0 in models:
                try:
                    popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
                    y_pred = func(t_fit, *popt)
                    r2 = r2_score(q_fit, y_pred)
                    aic = calculate_aic(len(t_fit), np.sum((q_fit - y_pred)**2), len(p0))
                    
                    ax.plot(t_plot, func(t_plot, *popt), label=f"{name} (R²: {r2:.3f})")
                    
                    # Peppas "n" yorumu eklentisi
                    note = ""
                    if name == "Korsmeyer-Peppas":
                        n = popt[1]
                        if n <= 0.45: note = "Fickian Diffusion"
                        elif 0.45 < n < 0.89: note = "Anomalous"
                        else: note = "Case-II Transport"
                    
                    kin_results.append({"Model": name, "R²": r2, "AIC": aic, "Parametreler": popt, "Mekanizma": note})
                except: continue
            
            ax.set_xlabel("Zaman (dakika)")
            ax.set_ylabel("Kümülatif Salım (%)")
            ax.legend(loc='lower right')
            st.pyplot(fig)

        with col2:
            st.write("**Model Uyumluluk Tablosu**")
            res_df = pd.DataFrame(kin_results).sort_values("AIC")
            st.dataframe(res_df[["Model", "R²", "AIC", "Mekanizma"]], use_container_width=True)
            
            st.info("💡 AIC (Akaike Information Criterion) değeri ne kadar düşükse, model veriyi o kadar iyi temsil ediyor demektir.")

    else:
        st.subheader("📈 Model-Bağımsız Karşılaştırma ve İstatistik")
        # MDT ve DE Hesaplamaları
        mdt_val = calculate_mdt(time, mean_q)
        de_val = calculate_de(time, mean_q)
        
        m1, m2 = st.columns(2)
        m1.metric("MDT (Ort. Dissolüsyon Süresi)", f"{mdt_val:.2f} dk")
        m2.metric("Dissolüsyon Verimi (DE)", f"% {de_val:.2f}")
        
        st.write("---")
        st.write("**Referans vs Test Karşılaştırması (f1, f2)**")
        ref_file = st.file_uploader("Karşılaştırma için Referans Dosyası Yükleyin", type=['xlsx', 'csv'], key="ref")
        
        if ref_file:
            t_ref, m_ref, s_ref = process_data(ref_file)
            # Zaman noktalarının eşleştiğini varsayıyoruz (Literatür kuralı)
            if len(m_ref) == len(mean_q):
                R, T = m_ref, mean_q
                f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                mse = np.mean((R - T)**2)
                f2 = 50 * np.log10((1 + mse)**-0.5 * 100)
                
                c1, c2 = st.columns(2)
                c1.metric("f1 (Farklılık Faktörü)", f"{f1:.2f}")
                c2.metric("f2 (Benzerlik Faktörü)", f"{f2:.2f}")
                
                if f2 >= 50: st.success("Profiller Benzerdir (f2 ≥ 50)")
                else: st.warning("Profiller Benzer Değildir (f2 < 50)")
            else:
                st.error("Hata: Referans ve Test verilerinin zaman noktaları (satır sayısı) aynı olmalıdır.")

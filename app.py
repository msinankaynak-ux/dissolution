import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- KİNETİK MODELLER (LİTERATÜR SETİ) ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))
def hopfenberg(t, k, n): return 100 * (1 - (1 - k * t)**n)

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

def calculate_mdt(time, means):
    delta_t = np.diff(time, prepend=0)
    mid_q = np.diff(means, prepend=0)
    mid_t = time - (delta_t / 2)
    return np.sum(mid_t * mid_q) / np.sum(mid_q) if np.sum(mid_q) > 0 else 0

st.set_page_config(page_title="PharmTech Lab Pro v2.1", layout="wide")
st.title("🔬 Gelişmiş Dissolüsyon Analitik Platformu")

# Veri İşleme Fonksiyonu
def process_data(file):
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
        values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        return time, values.mean(axis=1).values, values.std(axis=1).values
    except:
        return None, None, None

# Yan Menü Sınıflandırma
st.sidebar.header("⚙️ Analiz Seçenekleri")
mode = st.sidebar.radio("Sınıflandırma", ["Model-Bağımlı (Kinetik)", "Model-Bağımsız (f1/f2/MDT)"])

# --- ANA VERİ YÜKLEME (ID Hatasını Çözmek İçin Tek Noktadan) ---
main_file = st.file_uploader("Dosya Yükleyin (Zaman ve % Salım)", type=['xlsx', 'csv'], key="main_loader")

if main_file:
    time, mean_q, std_q = process_data(main_file)
    
    if time is not None:
        if mode == "Model-Bağımlı (Kinetik)":
            st.subheader("📊 Kinetik Modelleme ve Mekanizma Analizi")
            col1, col2 = st.columns([3, 2])
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.errorbar(time, mean_q, yerr=std_q, fmt='o', label="Deneysel", color='black', alpha=0.5)
                
                t_fit, q_fit = time[time > 0], mean_q[time > 0]
                t_plot = np.linspace(0, time.max(), 100)
                kin_res = []
                
                # Literatürdeki 6 Ana Model
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
                        
                        ax.plot(t_plot, func(t_plot, *popt), label=f"{name}")
                        
                        mechanism = ""
                        if name == "Korsmeyer-Peppas":
                            n = popt[1]
                            mechanism = "Fickian" if n <= 0.45 else "Anomalous" if n < 0.89 else "Case-II"
                            
                        kin_res.append({"Model": name, "R²": round(r2, 4), "AIC": round(aic, 2), "Mekanizma": mechanism})
                    except: continue
                
                ax.set_xlabel("Zaman (dk)")
                ax.set_ylabel("Salım (%)")
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.write("**Model Uyumluluk Özeti**")
                st.table(pd.DataFrame(kin_res).sort_values("AIC"))

        else:
            st.subheader("📈 Model-Bağımsız İstatistikler")
            m1, m2 = st.columns(2)
            mdt_val = calculate_mdt(time, mean_q)
            m1.metric("MDT (Mean Dissolution Time)", f"{mdt_val:.2f} dk")
            
            st.divider()
            st.write("**Referans Karşılaştırma (f1/f2)**")
            # Farklı bir key ile referans yükleyici
            ref_file = st.file_uploader("Karşılaştırma için Referans (R) Dosyası", type=['xlsx', 'csv'], key="ref_loader")
            
            if ref_file:
                _, m_ref, _ = process_data(ref_file)
                if m_ref is not None and len(m_ref) == len(mean_q):
                    R, T = m_ref, mean_q
                    f1 = (np.sum(np.abs(R - T)) / np.sum(R)) * 100
                    mse = np.mean((R - T)**2)
                    f2 = 50 * np.log10((1 + mse)**-0.5 * 100)
                    
                    st.success(f"f1: {f1:.2f} | f2: {f2:.2f}")
                else:
                    st.error("Veri boyutları uyuşmuyor!")

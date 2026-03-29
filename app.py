import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import r2_score

# --- KİNETİK MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

st.set_page_config(page_title="PharmTech Lab Pro", layout="wide")
st.title("🔬 Akademik Dissolüsyon Analiz Laboratuvarı")

st.sidebar.header("📁 Veri Yönetimi")
mode = st.sidebar.radio("Analiz Modu", ["Tek Seri Analizi", "Referans vs Test Kıyaslama (f1/f2)"])

def process_data(file):
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return time, values.mean(axis=1).values, values.std(axis=1).values

# --- VERİ YÜKLEME ---
data_loaded = False
if mode == "Tek Seri Analizi":
    up_file = st.file_uploader("Excel/CSV Dosyası Yükle", type=['xlsx', 'csv'])
    if up_file:
        time, mean_q, std_q = process_data(up_file)
        labels, means, stds = ["Numune"], [mean_q], [std_q]
        data_loaded = True
else:
    c_up1, c_up2 = st.columns(2)
    ref_f = c_up1.file_uploader("Referans (R) Dosyası", type=['xlsx', 'csv'])
    test_f = c_up2.file_uploader("Test (T) Dosyası", type=['xlsx', 'csv'])
    if ref_f and test_f:
        time, r_m, r_s = process_data(ref_f)
        _, t_m, t_s = process_data(test_f)
        labels, means, stds = ["Referans", "Test"], [r_m, t_m], [r_s, t_s]
        data_loaded = True

if data_loaded:
    col_main1, col_main2 = st.columns([3, 2])
    
    with col_main1:
        st.subheader("📊 Dissolüsyon Profili ve Model Eğrileri")
        fig, ax = plt.subplots(figsize=(10, 6))
        max_t = float(time.max()) if len(time) > 0 else 60
        ax.set_xlim(left=0, right=max_t * 1.05)
        ax.set_ylim(bottom=0, top=110)
        
        # Ana Veri Noktaları
        for i in range(len(labels)):
            ax.errorbar(time, means[i], yerr=stds[i], fmt='o', label=labels[i], capsize=5, markersize=8)

        # Kinetik Eğrilerin Çizimi (Ortalama veri üzerinden)
        t_fit = time[time > 0]
        q_fit = means[0][time > 0]
        kin_res = []
        models = [("Sıfırıncı D.", zero_order, [1]), ("Birinci D.", first_order, [0.1]), 
                  ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
                  ("Weibull", weibull, [100, 1])]
        
        t_plot = np.linspace(0, max_t, 100)
        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t_fit, q_fit, p0=p0, maxfev=10000)
                ax.plot(t_plot, func(t_plot, *popt), '--', alpha=0.7, label=f"{name} Eğrisi")
                
                # İstatistik Hesaplama
                r2 = r2_score(q_fit, func(t_fit, *popt))
                aic = calculate_aic(len(t_fit), np.sum((q_fit-func(t_fit, *popt))**2), len(p0))
                kin_res.append({"Model": name, "R²": round(r2, 4), "AIC": round(aic, 2)})
            except: continue
        
        ax.set_xlabel("Zaman (dakika)")
        ax.set_ylabel("Çözünen Madde (%)")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        st.pyplot(fig)

    with col_main2:
        if mode == "Referans vs Test Kıyaslama (f1/f2)":
            st.subheader("⚖️ Benzerlik Faktörleri")
            mse = np.mean((means[0] - means[1])**2)
            f2 = 50 * np.log10((1 + mse)**-0.5 * 100)
            f1 = (np.sum(np.abs(means[0] - means[1])) / np.sum(means[0])) * 100
            st.metric("f2 Benzerlik", f"{f2:.2f}")
            st.metric("f1 Farklılık", f"{f1:.2f}")
        
        st.subheader("📈 Model Uyumluluk Tablosu")
        st.table(pd.DataFrame(kin_res).sort_values("AIC"))

    # --- ŞARTLI RAPORLAMA ---
    st.divider()
    st.write("### 📝 Raporlama

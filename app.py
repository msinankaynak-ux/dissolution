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

# --- f1/f2 HESAPLAMA ---
def calculate_f1_f2(ref, test):
    n = len(ref)
    f1 = (np.sum(np.abs(ref - test)) / np.sum(ref)) * 100
    mse = np.mean((ref - test)**2)
    f2 = 50 * np.log10((1 + mse)**-0.5 * 100)
    return f1, f2

st.set_page_config(page_title="PharmTech Lab Pro", layout="wide")
st.title("🔬 Akademik Dissolüsyon Analiz ve Kıyaslama Platformu")

st.sidebar.header("📁 Veri Yönetimi")
mode = st.sidebar.radio("Analiz Modu", ["Tek Seri Analizi", "Referans vs Test Kıyaslama (f1/f2)"])

def process_data(file):
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    # Zaman sütununu sayıya zorla ve hataları temizle
    time = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    values = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return time, values.mean(axis=1).values, values.std(axis=1).values

# --- VERİ YÜKLEME ---
if mode == "Tek Seri Analizi":
    up_file = st.file_uploader("Excel/CSV Dosyası Yükle", type=['xlsx', 'csv'])
    if up_file:
        time, mean_q, std_q = process_data(up_file)
        labels, means, stds = ["Numune"], [mean_q], [std_q]
else:
    c1, c2 = st.columns(2)
    ref_f = c1.file_uploader("Referans (R) Dosyası", type=['xlsx', 'csv'])
    test_f = c2.file_uploader("Test (T) Dosyası", type=['xlsx', 'csv'])
    if ref_f and test_f:
        time, r_m, r_s = process_data(ref_f)
        _, t_m, t_s = process_data(test_f)
        labels, means, stds = ["Referans", "Test"], [r_m, t_m], [r_s, t_s]
    else: st.stop()

if 'time' in locals():
    # --- GRAFİK ---
    st.subheader("📊 Dissolüsyon Profili")
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Eksenleri Dinamik Olarak Belirleme
    max_t = float(time.max()) if len(time) > 0 else 60
    ax.set_xlim(left=0, right=max_t * 1.05) 
    ax.set_ylim(bottom=0, top=110)
    
    for i in range(len(labels)):
        ax.errorbar(time, means[i], yerr=stds[i], fmt='-o', label=labels[i], capsize=5, linewidth=2)
    
    ax.set_xlabel("Zaman (dakika)")
    ax.set_ylabel("Çözünen Madde (%)")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # --- SONUÇLAR ---
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.write("**Model Uyumluluk Tablosu (En İyi Uyum)**")
        t_f, q_f = time[time > 0], means[0][time > 0]
        kin_res = []
        models = [("Sıfırıncı D.", zero_order, [1]), ("Birinci D.", first_order, [0.1]), 
                  ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
                  ("Weibull", weibull, [100, 1])]
        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t_f, q_f, p0=p0, maxfev=10000)
                r2 = r2_score(q_f, func(t_f, *popt))
                aic = calculate_aic(len(t_f), np.sum((q_f-func(t_f, *popt))**2), len(p0))
                kin_res.append({"Model": name, "R²": round(r2, 4), "AIC": round(aic, 2)})
            except: continue
        st.table(pd.DataFrame(kin_res).sort_values("AIC"))

    with col_res2:
        if mode == "Referans vs Test Kıyaslama (f1/f2)":
            f1, f2 = calculate_f1_f2(means[0], means[1])
            st.metric("f2 Benzerlik Faktörü", f"{f2:.2f}")
            st.metric("f1 Farklılık Faktörü", f"{f1:.2f}")
            if f2 >= 50: st.success("Profiller Benzer (f2 ≥ 50)")
            else: st.error("Profiller Benzer Değil (f2 < 50)")
        else:
            st.write("**İstatistiksel Veri Özeti**")
            st.dataframe(pd.DataFrame({"Zaman": time, "Ortalama": means[0], "SD": stds[0]}).style.format(precision=2))

    # --- RAPORLAMA ---
    with st.expander("📝 Deney ve Metod Rapor Detayları"):
        now = datetime.now()
        r1, r2, r3 = st.columns(3)
        api = r1.text_input("API Adı", "Atorvastatin")
        dose = r1.text_input("Dozaj", "10 mg")
        exp_date = r3.date_input("Deney Tarihi", now.date())
        exp_time = r3.time_input("Deney Saati", now.time())
        st.text_input("Analist Notu", "")
        if st.button("Raporu Sisteme Kaydet"):
            st.toast(f"{api} analizi başarıyla kaydedildi!", icon="✅")

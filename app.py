import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- GÜVENLİ HESAPLAMA MOTORLARI ---
def safe_div(n, d): return n / d if d != 0 else 0

def calculate_de(t, q):
    if len(t) < 2: return 0
    auc = np.trapz(q, t)
    total_area = np.max(t) * 100
    return safe_div(auc * 100, total_area)

def calculate_mdt(t, q):
    if len(t) < 2 or np.max(q) <= 0: return 0
    dq = np.diff(q, prepend=0)
    # Negatif farkları (veri hataları) sıfırla
    dq = np.maximum(dq, 0)
    mid_t = [t[0]/2] + [(t[i] + t[i-1])/2 for i in range(1, len(t))]
    return safe_div(np.sum(dq * np.array(mid_t)), np.max(q))

def calculate_f2(r, t_val):
    n = len(r)
    if n == 0: return 0
    sum_sq = np.sum((r - t_val)**2)
    # f2 formülü: 50 * log10([1 + (1/n)*sum(sq)]^-0.5 * 100)
    val = 1 + (sum_sq / n)
    return 50 * np.log10(100 / np.sqrt(val))

# --- KİNETİK MODELLER (Hata Korumalı) ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-np.clip(k * t, 0, 10)))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 2.0))

# --- ANA UYGULAMA ---
st.set_page_config(page_title="PharmTech Lab Pro v8.0", layout="wide")
st.sidebar.title("🔬 Pro Engineer v8.0")

test_file = st.sidebar.file_uploader("Test (Excel/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans (Excel/CSV)", type=['xlsx', 'csv'])

def process_data(file):
    if not file: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        time = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        # NaN ve negatif zamanları temizle
        valid = ~np.isnan(time) & (time >= 0)
        return {"t": time[valid], "mean": vals.mean(axis=1).values[valid], "std": vals.std(axis=1).values[valid]}
    except: return None

test = process_data(test_file)
ref = process_data(ref_file)

menu = st.sidebar.selectbox("İşlem Seçin:", ["Genel Bakış", "Model-Bağımsız Analiz", "Kinetik Modelleme"])

if test:
    t, q = test["t"], test["mean"]
    
    if menu == "Model-Bağımsız Analiz":
        st.header("📊 Modelden Bağımsız Karşılaştırma")
        
        # Temel Metrikler
        de = calculate_de(t, q)
        mdt = calculate_mdt(t, q)
        mdr = safe_div(np.max(q), mdt)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Dissolution Efficiency (DE)", f"%{de:.2f}")
        col2.metric("Mean Dissolution Time (MDT)", f"{mdt:.2f} dk")
        col3.metric("Mean Dissolution Rate (MDR)", f"{mdr:.2f} %/dk")
        
        if ref:
            st.divider()
            # FDA kuralı: f2 için %85 salımdan sonraki tek noktayı al
            # Mühendislik notu: Veri seti boyutları aynı olmalı
            if len(ref["t"]) == len(t):
                f1 = safe_div(np.sum(np.abs(ref["mean"] - q)) * 100, np.sum(ref["mean"]))
                f2 = calculate_f2(ref["mean"], q)
                
                c1, c2 = st.columns(2)
                c1.write(f"**f1 (Fark Faktörü):** {f1:.2f}")
                c2.write(f"**f2 (Benzerlik Faktörü):** {f2:.2f}")
                if f2 >= 50: st.success("✅ Formülasyonlar Benzer (f2 >= 50)")
                else: st.error("❌ Benzerlik Şartı Sağlanmadı")
            else:
                st.warning("⚠️ Satır sayıları farklı olduğu için f1/f2 hesaplanamadı.")

    elif menu == "Kinetik Modelleme":
        st.header("🧮 Kinetik Fit Sonuçları")
        mask = (t > 0) & (q > 0) & (q < 100) # Fit bölgesini optimize et
        tf, qf = t[mask], q[mask]
        
        results = []
        models = [("Sıfır Derece", zero_order, [1]), ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer, [1, 0.5])]
        
        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, tf, qf, p0=p0, maxfev=2000)
                r2 = r2_score

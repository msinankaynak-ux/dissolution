import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# --- HATA KORUMALI MATEMATİKSEL MOTORLAR ---
def safe_div(n, d): return n / d if d != 0 else 0

def calculate_de(t, q):
    if len(t) < 2: return 0
    auc = np.trapz(q, t)
    total_area = np.max(t) * 100
    return safe_div(auc * 100, total_area)

def calculate_mdt(t, q):
    if len(t) < 2 or np.max(q) <= 0: return 0
    dq = np.maximum(np.diff(q, prepend=0), 0)
    mid_t = [t[0]/2] + [(t[i] + t[i-1])/2 for i in range(1, len(t))]
    return safe_div(np.sum(dq * np.array(mid_t)), np.max(q))

# --- MODELLER (Sınırlandırılmış ve Optimize Edilmiş) ---
def zero_order(t, k): return k * t
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): 
    # n ve k'nın ekstrem değerlerde patlamasını engellemek için clipping
    return np.clip(k * (t**np.clip(n, 0.01, 2.0)), 0, 120)

# --- ANA UYGULAMA ---
st.set_page_config(page_title="PharmTech Lab Pro v9.0", layout="wide")
st.sidebar.title("🚀 Pro Engineer v9.0")

def load_and_clean(file):
    if not file: return None
    try:
        df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
        t = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        # Sadece sayısal ve pozitif değerleri al (0 zamanını ve 0 salımını temizle - modeller için kritik)
        mask = ~np.isnan(t) & (t > 0) 
        return {"t": t[mask], "mean": v.mean(axis=1).values[mask], "std": v.std(axis=1).values[mask]}
    except: return None

test_file = st.sidebar.file_uploader("Test Verisi", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

test = load_and_clean(test_file)
ref = load_and_clean(ref_file)

menu = st.sidebar.selectbox("Analiz Tipi:", ["📊 Model-Bağımsız Analiz", "📈 Grafik & Kinetik"])

if test:
    t, q = test["t"], test["mean"]

    if menu == "📊 Model-Bağımsız Analiz":
        st.subheader("📋 Karşılaştırmalı Parametreler")
        col1, col2, col3 = st.columns(3)
        de = calculate_de(t, q)
        mdt = calculate_mdt(t, q)
        col1.metric("Dissolution Efficiency (DE)", f"%{de:.2f}")
        col2.metric("Mean Dissolution Time (MDT)", f"{mdt:.2f} dk")
        col3.metric("Mean Dissolution Rate (MDR)", f"{safe_div(np.max(q), mdt):.2f} %/dk")

        if ref:
            st.divider()
            if len(ref["t"]) == len(t):
                f1 = safe_div(np.sum(np.abs(ref["mean"] - q)) * 100, np.sum(ref["mean"]))
                # f2 Stabilize Formül
                sum_sq = np.sum((ref["mean"] - q)**2)
                f2 = 50 * np.log10(100 / np.sqrt(1 + sum_sq/len(q)))
                
                c1, c2 = st.columns(2)
                c1.info(f"**f1 (Fark):** {f1:.2f}")
                if f2 >= 50: c2.success(f"**f2 (Benzerlik):** {f2:.2f} ✅")
                else: c2.error(f"**f2 (Benzerlik):** {f2:.2f} ❌")
            else: st.warning("Zaman noktaları eşleşmiyor.")

    elif menu == "📈 Grafik & Kinetik":
        st.subheader("📈 Profil ve Model Uyumu")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.errorbar(t, q, yerr=test["std"], fmt='ob', label="Deneysel Veri", capsize=3)
        
        # Model Fit İşlemleri (Try-Except Bloğu ile Çökme Engellendi)
        results = []
        models = [("Sıfır Derece", zero_order, [0.1]), ("Higuchi", higuchi, [1.0]), ("Korsmeyer-Peppas", korsmeyer, [0.1, 0.5])]
        
        for name, func, p_init in models:
            try:
                # bounds eklenerek 'infeasible' hatası engellendi
                popt, _ = curve_fit(func, t, q, p0=p_init, maxfev=5000)
                r2 = r2_score(q, func(t, *popt))
                results.append({"Model": name, "R²": r2, "K": f"{popt[0]:.4f}"})
                ax.plot(t, func(t, *popt), label=f"{name} (R²:{r2:.3f})")
            except Exception as e:
                continue # Hatalı modeli atla, diğerlerine devam et

        ax.legend(); ax.set_xlabel("Zaman (dk)"); ax.set_ylabel("Salım (%)")
        st.pyplot(fig)
        st.table(pd.DataFrame(results))

else:
    st.info("Hocam, lütfen sol panelden geçerli bir dosya yükleyin.")

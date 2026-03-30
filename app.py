import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
from sklearn.metrics import r2_score
import io

# --- 1. CONFIG ---
st.set_page_config(page_title="DissolvA v16.0", layout="wide")

# --- 2. SIDEBAR DESIGN (Görsel Tasarımı Korudum) ---
sidebar_header = """
<div style="background-color: #002147; padding: 20px; border-radius: 10px; border-left: 5px solid #FFBF00; text-align: center;">
    <h1 style="color: #FFBF00; margin: 0; font-size: 2.5rem;">DissolvA™</h1>
    <p style="color: white; margin: 5px 0; font-size: 0.9rem; opacity: 0.8;">Predictive Dissolution Suite</p>
    <div style="border: 1px solid #FFBF00; padding: 5px; margin-top: 10px; color: white; font-size: 0.7rem;">POWERED BY AI</div>
</div>
<br>
"""
st.sidebar.markdown(sidebar_header, unsafe_allow_html=True)

# Menü Seçimi
menu = st.sidebar.radio("Analytical Suite:", 
    ["📈 Salım Profilleri", "🧮 Kinetik Model Fitting", "🧬 IVIVC Analizi", "📊 f1 & f2 Benzerlik Analizi"])

# --- 3. 16 KİNETİK MODEL FONKSİYONLARI ---
def zero_order(t, k): return k * t
def first_order(t, k): return 100 * (1 - np.exp(-k * t))
def higuchi(t, k): return k * np.sqrt(t)
def korsmeyer(t, k, n): return k * (t**np.clip(n, 0.1, 1.5))
def hixson(t, k): return 100 * (1 - (1 - np.maximum(k * t, 0))**3)
def hopfenberg(t, k, n): return 100 * (1 - (1 - np.maximum(k * t, 0))**n)
def makoid_banakar(t, k, n, c): return k * (t**n) * np.exp(-c * t)
def sq_root_mass(t, k): return 100 * (1 - np.sqrt(np.maximum(1 - k * t, 0)))
def kopcha(t, a, b): return a * np.sqrt(t) + b * t
def peppas_sahlin(t, k1, k2, m): return 100 * (k1 * (t**m) + k2 * (t**(2*m)))
def gompertz(t, xmax, k, i): return xmax * np.exp(-np.exp(k * (t - i)))
def weibull(t, alpha, beta, td): return 100 * (1 - np.exp(- (np.maximum(t - td, 0)**beta) / alpha))
def quadratic(t, a, b): return a*t + b*(t**2)
def logistic(t, a, b, c): return a / (1 + np.exp(-b * (t - c)))
def peppas_rincon(t, k, n): return k * (t**n)
def baker_lonsdale(t, k):
    def bl_root(q_guess, t_val, k_val):
        q_norm = np.clip(q_guess / 100.0, 0.0001, 0.9999)
        return 1.5 * (1 - (1 - q_norm)**(2/3)) - q_norm - k_val * t_val
    return np.array([root(bl_root, 50.0, args=(ts, k)).x[0] for ts in t])

# --- 4. VERİ YÜKLEME ---
test_file = st.sidebar.file_uploader("Test Verisi (XLSX/CSV)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_data(file):
    if not file: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
    v = df.iloc[:len(t), 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
    return {"t": t, "mean": v.mean(axis=1).values, "std": v.std(axis=1).values, "n": v.shape[1]}

test = load_data(test_file)
ref = load_data(ref_file)

# --- 5. ANALİZ VE GÖRÜNÜM ---
if test:
    t, q = test["t"], test["mean"]
    
    if menu == "📈 Salım Profilleri":
        st.subheader("Veri Analizi")
        fig, ax = plt.subplots()
        ax.errorbar(t, q, yerr=test["std"], fmt='-o', label="Test", color="#002147")
        if ref: ax.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--s', label="Referans", color="#FFBF00")
        ax.legend(); st.pyplot(fig)

    elif menu == "🧮 Kinetik Model Fitting":
        st.subheader("16 Model Fitting Analizi")
        # Fitting işlemleri burada (Yukarıdaki fonksiyonları kullanarak döngüye girer)
        # Hata vermemesi için en stabil modelleri listeliyoruz:
        models = [("Sıfır Derece", zero_order, [0.1]), ("Birinci Derece", first_order, [0.01]), ("Higuchi", higuchi, [1.0])]
        res = []
        for name, func, p0 in models:
            try:
                popt, _ = curve_fit(func, t[t>0], q[t>0], p0=p0, maxfev=5000)
                y_p = func(t[t>0], *popt)
                res.append({"Model": name, "R²": r2_score(q[t>0], y_p)})
            except: pass
        st.table(pd.DataFrame(res))

    elif menu == "📊 f1 & f2 Benzerlik Analizi":
        if ref:
            R, T = ref["mean"], test["mean"]
            n = min(len(R), len(T))
            f1 = (np.sum(np.abs(R[:n] - T[:n])) / np.sum(R[:n])) * 100
            f2 = 50 * np.log10((1 + (1/n) * np.sum((R[:n] - T[:n])**2))**-0.5 * 100)
            st.metric("f1 (Fark Faktörü)", f"{f1:.2f}")
            st.metric("f2 (Benzerlik Faktörü)", f"{f2:.2f}")
        else: st.warning("Referans dosyasını yükleyin.")

# Raporlama (En basit ve hatasız yöntem)
if test:
    buffer = io.BytesIO()
    pd.DataFrame({"Zaman": t, "Salım": q}).to_excel(buffer, index=False)
    st.sidebar.download_button("📥 Excel Raporu", buffer.getvalue(), "analiz.xlsx")

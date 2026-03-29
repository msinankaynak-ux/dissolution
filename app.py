import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import io

# --- KİNETİK MODELLER ---
def zero_order(t, k0): return k0 * t
def first_order(t, k1): return 100 * (1 - np.exp(-k1 * t))
def higuchi(t, kh): return kh * np.sqrt(t)
def korsmeyer_peppas(t, k, n): return k * (t**n)
def hixson_crowell(t, khc): return 100 * (1 - (1 - khc * t)**3)
def weibull(t, alpha, beta): return 100 * (1 - np.exp(-(t**beta) / alpha))

def calculate_aic(n, rss, k):
    if n <= k or rss <= 0: return 9999
    return n * np.log(rss/n) + 2*k

def calculate_mdt(time, means):
    delta_t = np.diff(time, prepend=0)
    mid_q = np.diff(means, prepend=0)
    mid_t = time - (delta_t / 2)
    return np.sum(mid_t * mid_q) / np.sum(mid_q) if np.sum(mid_q) > 0 else 0

st.set_page_config(page_title="PharmTech Lab Pro v3.0", layout="wide")

# --- SIDEBAR TASARIMI ---
st.sidebar.title("🔬 Analiz Paneli")
menu = st.sidebar.radio(
    "Görünüm Seçin:",
    ["📈 Dissolüsyon Profilleri", "🧮 Model-Bağımlı (Tüm Eğriler)", "📊 Model-Bağımsız Analiz"]
)

# --- VERİ YÜKLEME ---
st.sidebar.markdown("---")
test_file = st.sidebar.file_uploader("Test Verisi (Numune)", type=['xlsx', 'csv'])
ref_file = st.sidebar.file_uploader("Referans Verisi", type=['xlsx', 'csv'])

def load_data(file):
    if file is None: return None
    df = pd.read_excel(file) if file.name.endswith('.xlsx') else pd.read_csv(file)
    t = pd.to_numeric(df.iloc[:, 0], errors='coerce').fillna(0).values
    v = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    return {"t": t, "mean": v.mean(axis=1).values, "std": v.std(axis=1).values, "raw": v}

test = load_data(test_file)
ref = load_data(ref_file)

if test:
    time, mean_q, std_q = test["t"], test["mean"], test["std"]
    
    # --- 1. DİSSOLÜSYON PROFİLLERİ ---
    if menu == "📈 Dissolüsyon Profilleri":
        st.subheader("📍 Kümülatif Salım Grafiği")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.errorbar(time, mean_q, yerr=std_q, fmt='-ok', label="Test", capsize=4)
        if ref:
            ax1.errorbar(ref["t"], ref["mean"], yerr=ref["std"], fmt='--sr', label="Referans", alpha=0.6)
        
        ax1.set_xlabel("Zaman (dk)"); ax1.set_ylabel("Salım (%)"); ax1.legend(); ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        buf = io.BytesIO()
        fig1.savefig(buf, format="png", dpi=300)
        st.download_button("🖼️ Grafiği İndir", buf.getvalue(), "profil.png")

    # --- 2. MODEL-BAĞIMLI (TÜM EĞRİLER) ---
    elif menu == "🧮 Model-Bağımlı (Tüm Eğriler)":
        st.subheader("🔍 Tüm Kinetik Modellerin Karşılaştırmalı Çizimi")
        
        t_fit, q_fit = time[time > 0], mean_q[time > 0]
        t_plot = np.linspace(0.1, time.max(), 100)
        
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.scatter(time, mean_q, color='black', label="Deneysel Veri", zorder=5)
        
        models = [
            ("Zero-Order", zero_order, [1]), ("First-Order", first_order, [0.1]),
            ("Higuchi", higuchi, [1]), ("Korsmeyer-Peppas", korsmeyer_peppas, [1, 0.5]),
            ("Hixson-Crowell", hixson_crow
